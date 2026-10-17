import argparse
import pprint
import pandas as pd

from mlflow.tracking import MlflowClient

from utils.model_utils import MODEL_NAME, CANDIDATE_MODELS, train_valid_snapshots, load_training_data, \
    load_registered_model, evaluate_scores

# last task of aml_training_pipeline: compares the 3 models just trained on the valid snapshot,
# moves the aliases in the MLflow model registry (champion / challenger / baseline)
# and draws champion and challenger against the LR baseline (model_comparison_<date>.png)
# to call this script (from the project folder): python scripts/model_select.py --snapshotdate "2022-09-09"


def choose_models(candidates_pdf, good_enough_share=0.9):
    # pick the simplest model within 10% of the best valid PR-AUC: with ~600 positives small gaps are
    # noise, and the graph features cost one more daily job, so they have to clearly pay off
    # returns (chosen version, best of the other versions, logistic regression version)
    candidates_pdf = candidates_pdf.sort_values("valid_pr_auc", ascending=False)
    good_enough = candidates_pdf[candidates_pdf["valid_pr_auc"] >= good_enough_share * candidates_pdf["valid_pr_auc"].max()]
    best_version = good_enough.sort_values("complexity").iloc[0]["version"]
    second_version = candidates_pdf[candidates_pdf["version"] != best_version].iloc[0]["version"]
    lr_version = candidates_pdf[candidates_pdf["model_type"] == "logistic_regression"].iloc[0]["version"]
    return best_version, second_version, lr_version


def model_label(model_type, feature_set):
    if model_type == "logistic_regression":
        return "Logistic regression"
    return "LightGBM + graph" if feature_set == "base_graph" else "LightGBM"


def comparison_table(candidates_pdf, champion_version, challenger_version, lr_version, champion_row=None):
    # one row per model with its role, champion first, then challenger, baseline, models not used
    # champion_row: the older champion scored on this valid snapshot (only after a retrain)
    compare_pdf = candidates_pdf[["version", "model_type", "feature_set", "valid_pr_auc", "valid_p_at_500"]].copy()
    if champion_row is not None:
        compare_pdf = pd.concat([compare_pdf, pd.DataFrame([champion_row])], ignore_index=True)

    roles = {}
    for role, version in [("champion", champion_version), ("challenger", challenger_version), ("baseline", lr_version)]:
        roles.setdefault(version, []).append(role)
    compare_pdf["role"] = [" + ".join(roles.get(v, ["not used"])) for v in compare_pdf["version"]]
    compare_pdf["order"] = [["champion", "challenger", "baseline", "not used"].index(r.split(" + ")[0]) for r in compare_pdf["role"]]
    return compare_pdf.sort_values("order").reset_index(drop=True)


def plot_comparison(compare_pdf, title, filepath):
    # imported here so the unit tests do not need matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = {"champion": "#2a78d6", "challenger": "#eb6834", "baseline": "#9aa0a6"}
    bar_colors = [colors.get(role.split(" + ")[0], "#d0d4d9") for role in compare_pdf["role"]]
    labels = ["v" + v + " " + model_label(m, f) + "\n" + role for v, m, f, role in
              zip(compare_pdf["version"], compare_pdf["model_type"], compare_pdf["feature_set"], compare_pdf["role"])]

    fig, axes = plt.subplots(1, 2, figsize=(9, 0.6 * len(compare_pdf) + 1.4), sharey=True)
    for ax, column, name, fmt in [(axes[0], "valid_pr_auc", "PR-AUC", "{:.3f}"), (axes[1], "valid_p_at_500", "Precision@500", "{:.1%}")]:
        ax.barh(labels, compare_pdf[column], color=bar_colors)
        for i, value in enumerate(compare_pdf[column]):
            ax.text(value, i, " " + fmt.format(value), va="center", fontsize=9)
        ax.set_title(name, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    axes[1].xaxis.set_major_formatter(PercentFormatter(1.0))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(filepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # --- set up config ---
    config = {}
    config["model_train_date_str"] = snapshotdate
    config["model_name"] = MODEL_NAME
    config["training_results_directory"] = "datamart/gold/model_training/"
    pprint.pprint(config)

    # --- load the results of the 3 training tasks ---
    date_part = snapshotdate.replace('-','_')
    candidate_pdfs = []
    for model_key in CANDIDATE_MODELS:
        filepath = config["training_results_directory"] + "candidate_" + model_key + "_" + date_part + ".csv"
        candidate_pdf = pd.read_csv(filepath, dtype={"version": str})
        candidate_pdf["complexity"] = CANDIDATE_MODELS[model_key]["complexity"]
        candidate_pdfs.append(candidate_pdf)
    candidates_pdf = pd.concat(candidate_pdfs, ignore_index=True).sort_values("valid_pr_auc", ascending=False)
    print(candidates_pdf[["version", "model_type", "feature_set", "valid_pr_auc", "valid_p_at_500", "valid_p_at_2000"]].to_string(index=False))

    best_version, second_version, lr_version = choose_models(candidates_pdf)

    # --- update model registry aliases ---
    client = MlflowClient()
    try:
        champion = client.get_model_version_by_alias(config["model_name"], "champion")
    except Exception:
        champion = None

    if champion is None:
        # first deployment: chosen model becomes champion, the next one runs in shadow
        client.set_registered_model_alias(config["model_name"], "champion", best_version)
        client.set_registered_model_alias(config["model_name"], "challenger", second_version)
        print('champion -> version', best_version, '| challenger -> version', second_version)
    else:
        # retrain: the new model only goes to shadow, a person promotes it after checking the shadow results
        # (MLflow UI -> model -> version -> add alias "champion")
        client.set_registered_model_alias(config["model_name"], "challenger", best_version)
        print('challenger -> version', best_version, '(champion stays version', champion.version + ')')
    client.set_registered_model_alias(config["model_name"], "baseline", lr_version)
    print('baseline -> version', lr_version)

    # --- save training results ---
    filepath = config["training_results_directory"] + "training_results_" + date_part + ".csv"
    candidates_pdf.to_csv(filepath, index=False)
    print('saved to:', filepath)

    # --- compare champion and challenger with the LR baseline ---
    # the baseline is not scored every day, this is where we check that the models are worth it.
    # All bars use this training's valid snapshot; after a retrain the older champion is scored on it too
    champion_version = best_version if champion is None else champion.version
    challenger_version = second_version if champion is None else best_version
    valid_snapshot_date_str = train_valid_snapshots(snapshotdate)[1]
    champion_row = None
    if champion is not None:
        champion_model, champion_mv = load_registered_model("champion")
        valid_pdf = load_training_data(valid_snapshot_date_str)
        results = evaluate_scores(valid_pdf["label"], champion_model.predict_proba(valid_pdf[list(champion_model.feature_names_in_)])[:, 1])
        champion_row = {"version": champion_mv.version, "model_type": champion_mv.tags["model_type"],
                        "feature_set": champion_mv.tags["feature_set"],
                        "valid_pr_auc": results["pr_auc"], "valid_p_at_500": results["p_at_500"]}
    compare_pdf = comparison_table(candidates_pdf, champion_version, challenger_version, lr_version, champion_row)

    baseline_pr_auc = compare_pdf.loc[compare_pdf["version"] == lr_version, "valid_pr_auc"].iloc[0]
    for role, version in [("champion", champion_version), ("challenger", challenger_version)]:
        pr_auc = compare_pdf.loc[compare_pdf["version"] == version, "valid_pr_auc"].iloc[0]
        print(role, 'version', version, 'valid PR-AUC', round(pr_auc, 4), 'vs baseline', round(baseline_pr_auc, 4), '->',
              'better than baseline' if pr_auc > baseline_pr_auc else 'WARNING: not better than the baseline')

    filepath = config["training_results_directory"] + "model_comparison_" + date_part + ".png"
    plot_comparison(compare_pdf, "Training " + snapshotdate + ": models on the valid snapshot " + valid_snapshot_date_str, filepath)
    print('saved to:', filepath)

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)
