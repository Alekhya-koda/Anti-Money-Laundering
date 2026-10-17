import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import average_precision_score
from mlflow.tracking import MlflowClient

from utils.model_utils import load_training_data, precision_recall_at_k

# tables and charts for the report / slides, built from the pipeline outputs after the DAG has run
# run inside the airflow container (needs MLflow): cd /opt/airflow/aml && python3 scripts/report_figures.py

TEST_DATE = "2022-09-09"          # first snapshot scored in production, labels arrive on 09-11
TRAIN_DATE = "2022-09-09"         # the models were trained on the morning of 09-09
# python3 scripts/report_figures.py --slides  -> same charts with bigger text for the deck
SCALE = 1.45 if "--slides" in sys.argv else 1.0
figure_directory = "reports/figures/slides/" if "--slides" in sys.argv else "reports/figures/"
table_directory = "reports/tables/"
for directory in [figure_directory, table_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# chart style: blue / orange / aqua for the 3 models (checked for colour blindness), grey for rule baselines
COLORS = {"LightGBM": "#2a78d6", "LightGBM + graph": "#eb6834", "Logistic regression": "#1baf7a"}
GREY = "#8c8b85"
INK = "#2b2b2b"
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 9 * SCALE, "axes.edgecolor": "#c9c7c0", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ecebe6", "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def model_label(model_type, feature_set):
    if model_type == "logistic_regression":
        return "Logistic regression"
    return "LightGBM + graph" if feature_set == "base_graph" else "LightGBM"


def top_k_set(acct, scores, k, seed=88):
    # same tie breaking as precision_recall_at_k
    pdf = pd.DataFrame({"acct": acct, "score": scores}).sample(frac=1, random_state=seed)
    return set(pdf.sort_values("score", ascending=False, kind="mergesort").head(k)["acct"])


# ------------------------------------------------------------------ 1. test results with 95% intervals
training_pdf = pd.read_csv("datamart/gold/model_training/training_results_" + TRAIN_DATE.replace('-','_') + ".csv", dtype={"version": str})
version_label = {r["version"]: model_label(r["model_type"], r["feature_set"]) for _, r in training_pdf.iterrows()}

eval_pdf = load_training_data(TEST_DATE)
y = eval_pdf["label"].values
scores = {"Rule: out_cnt": eval_pdf["out_cnt"].values, "Rule: out_big_cnt": eval_pdf["out_big_cnt"].values}
roles = {}
for alias in ["champion", "challenger", "baseline"]:
    filepath = "datamart/gold/model_predictions/" + alias + "/" + alias + "_predictions_" + TEST_DATE.replace('-','_') + ".parquet"
    predictions_pdf = pd.read_parquet(filepath)
    name = version_label[str(predictions_pdf["model_version"].iloc[0])]
    merged = eval_pdf[["acct"]].merge(predictions_pdf[["acct", "score"]], on="acct", how="left")
    scores[name] = merged["score"].fillna(0).values
    roles[name] = alias + " (v" + str(predictions_pdf["model_version"].iloc[0]) + ")"

rng = np.random.default_rng(88)
boot_idx = [rng.integers(0, len(y), len(y)) for _ in range(200)]
rows = []
for name, s in scores.items():
    row = {"model": name, "role": roles.get(name, "rule baseline"), "pr_auc": average_precision_score(y, s)}
    for k in (500, 2000, 5000):
        p, r = precision_recall_at_k(y, s, k)
        row["p_at_" + str(k)] = p
        row["r_at_" + str(k)] = r
    pr_auc_boot = [average_precision_score(y[i], s[i]) for i in boot_idx]
    p500_boot = [precision_recall_at_k(y[i], s[i], 500)[0] for i in boot_idx]
    row["pr_auc_low"], row["pr_auc_high"] = np.percentile(pr_auc_boot, [2.5, 97.5])
    row["p_at_500_low"], row["p_at_500_high"] = np.percentile(p500_boot, [2.5, 97.5])
    rows.append(row)
test_pdf = pd.DataFrame(rows)
test_pdf.to_csv(table_directory + "test_results.csv", index=False)
print(test_pdf.round(4).to_string(index=False))

# paired bootstrap: is LightGBM + graph really better than LightGBM on the test snapshot?
a, b = scores["LightGBM + graph"], scores["LightGBM"]
diff_pr = [average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i]) for i in boot_idx]
diff_p500 = [precision_recall_at_k(y[i], a[i], 500)[0] - precision_recall_at_k(y[i], b[i], 500)[0] for i in boot_idx]
paired_pdf = pd.DataFrame([
    {"metric": "pr_auc", "diff": np.mean(diff_pr), "low": np.percentile(diff_pr, 2.5), "high": np.percentile(diff_pr, 97.5), "share_positive": np.mean(np.array(diff_pr) > 0)},
    {"metric": "p_at_500", "diff": np.mean(diff_p500), "low": np.percentile(diff_p500, 2.5), "high": np.percentile(diff_p500, 97.5), "share_positive": np.mean(np.array(diff_p500) > 0)},
])
paired_pdf.to_csv(table_directory + "graph_vs_base_paired.csv", index=False)
print(paired_pdf.round(4).to_string(index=False))

# ------------------------------------------------------------------ 2. precision@K curve on the test snapshot
ks = np.arange(100, 5001, 100)
fig, ax = plt.subplots(figsize=(6.2, 3.0))
for name in ["LightGBM + graph", "LightGBM", "Logistic regression"]:
    curve = [precision_recall_at_k(y, scores[name], k)[0] * 100 for k in ks]
    ax.plot(ks, curve, color=COLORS[name], linewidth=2, label=name)
curve = [precision_recall_at_k(y, scores["Rule: out_cnt"], k)[0] * 100 for k in ks]
ax.plot(ks, curve, color=GREY, linewidth=1.5, linestyle="--", label="Rule: rank by out_cnt")
# K = 500 is the daily review capacity we assume, show the values there
ax.axvline(500, color=GREY, linewidth=1, linestyle=":")
# small vertical nudges so the LR and rule labels (5.0% vs 4.2%) do not overlap
for name, color, nudge in [("LightGBM + graph", COLORS["LightGBM + graph"], 0), ("LightGBM", COLORS["LightGBM"], 0),
                           ("Logistic regression", COLORS["Logistic regression"], 0.7 * SCALE ** 2), ("Rule: out_cnt", GREY, -0.9 * SCALE ** 2)]:
    value = precision_recall_at_k(y, scores[name], 500)[0] * 100
    ax.plot([500], [value], marker="o", markersize=5, color=color, markeredgecolor="white", markeredgewidth=1)
    ax.text(560, value + nudge, "{:.1f}%".format(value), color=INK, fontsize=8 * SCALE, va="center")
ax.text(520, ax.get_ylim()[1] * 0.97, "K = 500 (daily capacity)", color=INK, fontsize=8 * SCALE, va="top")
ax.axhline(100 * y.mean(), color=GREY, linewidth=1, linestyle=":", label="Random list (0.21%)")
ax.set_xlabel("K = accounts reviewed")
ax.set_ylabel("Precision@K (%)")
ax.set_xlim(0, 5100)
ax.set_ylim(0, None)
ax.legend(frameon=False, fontsize=8 * SCALE, loc="upper right")
if SCALE == 1.0: ax.set_title("Test snapshot 2022-09-09: precision of the top-K list", loc="left", fontsize=10 * SCALE, color=INK)
fig.tight_layout()
fig.savefig(figure_directory + "precision_at_k.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ------------------------------------------------------------------ 3. recall by laundering pattern
pattern_rows, current = [], None
for line in open("data/raw/HI-Small_Patterns.txt"):
    line = line.strip()
    if line.startswith("BEGIN LAUNDERING ATTEMPT"):
        current = line.split(" - ")[1].split(":")[0].strip()
    elif line.startswith("END LAUNDERING ATTEMPT"):
        current = None
    elif line and current:
        f = line.split(",")
        pattern_rows.append((f[0], f[2], current))
pattern_pdf = pd.DataFrame(pattern_rows, columns=["timestamp", "src_acct", "pattern"])
pattern_pdf["ts"] = pd.to_datetime(pattern_pdf["timestamp"], format="%Y/%m/%d %H:%M")
in_window = (pattern_pdf["ts"] >= TEST_DATE) & (pattern_pdf["ts"] < pd.Timestamp(TEST_DATE) + pd.Timedelta(days=2))
pattern_pdf = pattern_pdf[in_window][["src_acct", "pattern"]].drop_duplicates()

positives = set(eval_pdf.loc[eval_pdf["label"] == 1, "acct"])
pattern_pdf = pattern_pdf[pattern_pdf["src_acct"].isin(positives)]
print('positives in test:', len(positives), '| with a known pattern:', pattern_pdf["src_acct"].nunique())

k = 2000
recall_rows = []
for name in ["LightGBM", "LightGBM + graph", "Logistic regression", "Rule: out_cnt"]:
    top = top_k_set(eval_pdf["acct"].values, scores[name], k)
    for pattern, group in pattern_pdf.groupby("pattern"):
        accts = set(group["src_acct"])
        recall_rows.append({"model": name, "pattern": pattern, "n_accounts": len(accts), "recall_at_2000": len(accts & top) / len(accts)})
recall_pdf = pd.DataFrame(recall_rows)
recall_pdf.to_csv(table_directory + "recall_by_pattern.csv", index=False)
print(recall_pdf.pivot(index="pattern", columns="model", values="recall_at_2000").round(3).to_string())

order = recall_pdf[recall_pdf["model"] == "LightGBM"].sort_values("recall_at_2000")["pattern"].tolist()
n_accounts = recall_pdf.drop_duplicates("pattern").set_index("pattern")["n_accounts"]
fig, ax = plt.subplots(figsize=(6.2, 3.2))
height = 0.26
for i, name in enumerate(["LightGBM", "LightGBM + graph", "Logistic regression"]):
    values = recall_pdf[recall_pdf["model"] == name].set_index("pattern").loc[order, "recall_at_2000"] * 100
    ax.barh(np.arange(len(order)) + (1 - i) * height, values, height=height - 0.03, color=COLORS[name], label=name)
ax.set_yticks(np.arange(len(order)))
ax.set_yticklabels([p.lower() + " (n=" + str(n_accounts[p]) + ")" for p in order])
ax.set_xlabel("Recall@2000 (%) of the laundering accounts using this pattern")
if SCALE == 1.0: ax.set_title("Which laundering patterns are caught (test snapshot)", loc="left", fontsize=10 * SCALE, color=INK)
if SCALE > 1:
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, fontsize=8 * SCALE)
    ax.set_xlabel("Recall@2000 (%)")
else:
    ax.legend(frameon=False, loc="lower right", fontsize=8 * SCALE)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(figure_directory + "recall_by_pattern.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ------------------------------------------------------------------ 4. class weights: all tuning tries from MLflow
client = MlflowClient()
experiment = client.get_experiment_by_name("aml-account-risk")
runs = client.search_runs([experiment.experiment_id])
trial_pdfs = []
for run in runs:
    # only the first training (train 09-05, valid 09-07), not the drift retrains
    if run.data.params.get("valid_snapshot") != "2022-09-07":
        continue
    local_path = client.download_artifacts(run.info.run_id, "tuning_trials.csv", "/tmp")
    trials = pd.read_csv(local_path)
    trials["model"] = model_label(run.data.params["model_type"], run.data.params["feature_set"])
    trial_pdfs.append(trials)
trials_pdf = pd.concat(trial_pdfs, ignore_index=True)
trials_pdf.to_csv(table_directory + "tuning_trials.csv", index=False)

lgbm_trials = trials_pdf[trials_pdf["model"] != "Logistic regression"]
summary = lgbm_trials.groupby("scale_pos_weight")["valid_pr_auc"].agg(["count", "median", "max"])
print(summary.round(4).to_string())
lr_trials = trials_pdf[trials_pdf["model"] == "Logistic regression"]
print(lr_trials[["C", "class_weight", "valid_pr_auc"]].round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(3.6, 2.8))
weights = sorted(lgbm_trials["scale_pos_weight"].unique())
for i, w in enumerate(weights):
    values = lgbm_trials.loc[lgbm_trials["scale_pos_weight"] == w, "valid_pr_auc"]
    jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(values))
    ax.scatter(np.full(len(values), i) + jitter, values, s=22, color=COLORS["LightGBM"], alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.plot([i - 0.25, i + 0.25], [values.median()] * 2, color=INK, linewidth=1.5)
ax.set_xticks(range(len(weights)))
ax.set_xticklabels(["none (1)" if w == 1 else str(int(w)) for w in weights])
ax.set_xlabel("scale_pos_weight")
ax.set_ylabel("Valid PR-AUC")
if SCALE == 1.0: ax.set_title("LightGBM tuning tries (line = median)", loc="left", fontsize=10 * SCALE, color=INK)
fig.tight_layout()
fig.savefig(figure_directory + "class_weights.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ------------------------------------------------------------------ 5. drift and attribution of the champion
drift_files = sorted(glob.glob("datamart/gold/model_monitoring/drift_report_*.csv"))
drift_pdf = pd.concat([pd.read_csv(f) for f in drift_files], ignore_index=True)
drift_pdf.to_csv(table_directory + "drift_reports.csv", index=False)
first = drift_pdf[drift_pdf["snapshot_date"] == TEST_DATE]
top_features = first.dropna(subset=["ref_attribution_share"]).sort_values("ref_attribution_share", ascending=False).head(10)

# the slide version is wider so the two panels keep their labels apart
fig, axes = plt.subplots(1, 2, figsize=(6.6 * (1.45 if SCALE > 1 else 1), 3.0 * (1.2 if SCALE > 1 else 1)), sharey=True)
names = top_features["feature"].tolist()[::-1]
axes[0].barh(names, top_features["ref_attribution_share"].values[::-1] * 100, color=COLORS["LightGBM"], height=0.6)
axes[0].set_xlabel("Share of |SHAP|, train snapshot (%)")
axes[0].set_title("What the champion uses", loc="left", fontsize=10 * SCALE, color=INK)
psi_values = top_features["psi"].values[::-1]
axes[1].barh(names, psi_values, color=[COLORS["LightGBM + graph"] if v > 0.25 else "#b9b7af" for v in psi_values], height=0.6)
for threshold in (0.1, 0.25):
    axes[1].axvline(threshold, color=GREY, linestyle="--", linewidth=1)
axes[1].text(0.27, len(names) - 1, "0.25 drift line", color=INK, fontsize=7 * SCALE, va="center")
for i, v in enumerate(psi_values):
    if v > 0.25:
        axes[1].text(v + 0.03, i, "{:.2f} drift".format(v), color=INK, fontsize=7 * SCALE, va="center")
axes[1].set_xlabel("PSI, Sep 9 vs train snapshot")
axes[1].set_title("How much it moved", loc="left", fontsize=10 * SCALE, color=INK)
for ax in axes:
    ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(figure_directory + "attribution_and_drift.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(first.sort_values("psi", ascending=False).head(8)[["feature", "psi", "ref_attribution_share", "cur_attribution_share", "status"]].round(4).to_string(index=False))

# ------------------------------------------------------------------ 6. data volume per day (data quality reports)
dq_pdf = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("datamart/silver/data_quality/dq_report_*.csv"))], ignore_index=True)
dq_pdf.to_csv(table_directory + "data_quality.csv", index=False)
fig, ax = plt.subplots(figsize=(6.2, 2.4))
days = pd.to_datetime(dq_pdf["snapshot_date"])
ax.bar(days.dt.strftime("%a %d"), dq_pdf["silver_rows"] / 1000, color=COLORS["LightGBM"], width=0.6)
ax.set_ylabel("Transactions (thousands)")
if SCALE == 1.0: ax.set_title("Daily volume ingested (Sep 1-10, 2022)", loc="left", fontsize=10 * SCALE, color=INK)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(figure_directory + "daily_volume.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ------------------------------------------------------------------ 7. history of the monitor + model registry state
perf_pdf = pd.read_csv("datamart/gold/model_monitoring/model_performance.csv")
perf_pdf.to_csv(table_directory + "model_performance_history.csv", index=False)
registry_rows = []
for mv in client.search_model_versions("name='aml-account-risk'"):
    registry_rows.append({"version": mv.version, "model_type": mv.tags.get("model_type"), "feature_set": mv.tags.get("feature_set"),
                          "model_train_date": mv.tags.get("model_train_date"),
                          "aliases": ",".join(client.get_model_version("aml-account-risk", mv.version).aliases)})
registry_pdf = pd.DataFrame(registry_rows).sort_values("version")
registry_pdf.to_csv(table_directory + "model_registry.csv", index=False)
print(registry_pdf.to_string(index=False))
print('saved figures to', figure_directory, 'and tables to', table_directory)
