import argparse
import pprint

import pandas as pd
from mlflow.tracking import MlflowClient

from utils.model_utils import (
    CANDIDATE_MODELS,
    MODEL_NAME,
    evaluate_scores,
    load_registered_model,
    load_training_data,
    model_label,
    train_valid_snapshots,
)

# last task of aml_training_pipeline: compares the 3 new models on the valid snapshot, sets the
# MLflow aliases and draws the models against the LR baseline (model_comparison_<date>.png)
# run from the project folder:
#   python scripts/model_select.py --snapshotdate "2022-09-09"

ROLE_ORDER = ["champion", "challenger", "shadow_1", "shadow_2", "baseline", "not used"]


def choose_models(candidates_pdf, good_enough_share=0.9):
    # the simplest model within 10% of the best valid PR-AUC wins: with ~600 positives smaller
    # gaps are noise, and the graph features cost an extra daily job
    # returns (chosen version, best of the other versions, logistic regression version)
    candidates_pdf = candidates_pdf.sort_values("valid_pr_auc", ascending=False)
    best_pr_auc = candidates_pdf["valid_pr_auc"].max()
    good_enough = candidates_pdf[candidates_pdf["valid_pr_auc"] >= good_enough_share * best_pr_auc]
    best_version = good_enough.sort_values("complexity").iloc[0]["version"]
    second_version = candidates_pdf[candidates_pdf["version"] != best_version].iloc[0]["version"]
    is_lr = candidates_pdf["model_type"] == "logistic_regression"
    lr_version = candidates_pdf[is_lr].iloc[0]["version"]
    return best_version, second_version, lr_version


def comparison_table(candidates_pdf, roles, champion_row=None):
    # roles: [(role, version)], champion_row: the old champion scored on this valid snapshot
    # (only after a retrain)
    columns = ["version", "model_type", "feature_set", "valid_pr_auc", "valid_p_at_500"]
    compare_pdf = candidates_pdf[columns].copy()
    if champion_row is not None:
        compare_pdf = pd.concat([compare_pdf, pd.DataFrame([champion_row])], ignore_index=True)

    version_roles = {}
    for role, version in roles:
        version_roles.setdefault(version, []).append(role)
    compare_pdf["role"] = [
        " + ".join(version_roles.get(v, ["not used"])) for v in compare_pdf["version"]
    ]
    compare_pdf["order"] = [ROLE_ORDER.index(r.split(" + ")[0]) for r in compare_pdf["role"]]
    return compare_pdf.sort_values("order").reset_index(drop=True)


def plot_comparison(compare_pdf, title, filepath):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    # blue = production, orange = shadow, grey = baseline
    colors = {
        "champion": "#2a78d6",
        "challenger": "#eb6834",
        "shadow_1": "#eb6834",
        "shadow_2": "#eb6834",
        "baseline": "#9aa0a6",
    }
    bar_colors = [colors.get(role.split(" + ")[0], "#d0d4d9") for role in compare_pdf["role"]]
    labels = [
        "v" + v + " " + model_label(m, f) + "\n" + role
        for v, m, f, role in zip(
            compare_pdf["version"],
            compare_pdf["model_type"],
            compare_pdf["feature_set"],
            compare_pdf["role"],
        )
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 0.6 * len(compare_pdf) + 1.4), sharey=True)
    panels = [
        (axes[0], "valid_pr_auc", "PR-AUC", "{:.3f}"),
        (axes[1], "valid_p_at_500", "Precision@500", "{:.1%}"),
    ]
    for ax, column, name, fmt in panels:
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
    print("\n\n---starting job---\n\n")

    config = {}
    config["model_train_date_str"] = snapshotdate
    config["model_name"] = MODEL_NAME
    config["training_results_directory"] = "datamart/gold/model_training/"
    pprint.pprint(config)

    # --- results of the 3 training tasks ---
    date_part = snapshotdate.replace("-", "_")
    candidate_pdfs = []
    for model_key in CANDIDATE_MODELS:
        filename = "candidate_" + model_key + "_" + date_part + ".csv"
        candidate_pdf = pd.read_csv(
            config["training_results_directory"] + filename, dtype={"version": str}
        )
        candidate_pdf["complexity"] = CANDIDATE_MODELS[model_key]["complexity"]
        candidate_pdfs.append(candidate_pdf)
    candidates_pdf = pd.concat(candidate_pdfs, ignore_index=True)
    candidates_pdf = candidates_pdf.sort_values("valid_pr_auc", ascending=False)
    summary_cols = [
        "version",
        "model_type",
        "feature_set",
        "valid_pr_auc",
        "valid_p_at_500",
        "valid_p_at_2000",
    ]
    print(candidates_pdf[summary_cols].to_string(index=False))

    best_version, second_version, lr_version = choose_models(candidates_pdf)

    # --- update model registry aliases ---
    client = MlflowClient()
    try:
        champion = client.get_model_version_by_alias(config["model_name"], "champion")
    except Exception:
        champion = None

    if champion is None:
        # first deployment: the chosen model becomes champion, the next one runs as challenger
        roles = [("champion", best_version), ("challenger", second_version)]
    else:
        # retrain: both new models run in shadow and the champion keeps serving. The old challenger
        # stops, it was trained on the same old data as the champion (promotion: README step 7)
        roles = [("shadow_1", best_version), ("shadow_2", second_version)]
        print("champion stays version", champion.version)
        aliases = client.get_registered_model(config["model_name"]).aliases
        if "challenger" in aliases:
            client.delete_registered_model_alias(config["model_name"], "challenger")
            print("challenger alias removed from version", aliases["challenger"])
    for alias, version in roles + [("baseline", lr_version)]:
        client.set_registered_model_alias(config["model_name"], alias, version)
        print(alias, "-> version", version)

    filepath = config["training_results_directory"] + "training_results_" + date_part + ".csv"
    candidates_pdf.to_csv(filepath, index=False)
    print("saved to:", filepath)

    # --- compare the models with the LR baseline ---
    # the baseline is not scored every day, this is where we check that the models are worth it.
    # After a retrain the old champion is also scored on this training's valid snapshot
    valid_date = train_valid_snapshots(snapshotdate)[1]
    champion_row = None
    if champion is not None:
        champion_model, champion_mv = load_registered_model("champion")
        valid_pdf = load_training_data(valid_date)
        X_valid = valid_pdf[list(champion_model.feature_names_in_)]
        results = evaluate_scores(valid_pdf["label"], champion_model.predict_proba(X_valid)[:, 1])
        champion_row = {
            "version": champion_mv.version,
            "model_type": champion_mv.tags["model_type"],
            "feature_set": champion_mv.tags["feature_set"],
            "valid_pr_auc": results["pr_auc"],
            "valid_p_at_500": results["p_at_500"],
        }
        roles = [("champion", champion_mv.version)] + roles
    compare_pdf = comparison_table(
        candidates_pdf, roles + [("baseline", lr_version)], champion_row
    )

    pr_auc = compare_pdf["valid_pr_auc"]
    baseline_pr_auc = pr_auc[compare_pdf["version"] == lr_version].iloc[0]
    for role, version in roles:
        model_pr_auc = pr_auc[compare_pdf["version"] == version].iloc[0]
        if model_pr_auc > baseline_pr_auc:
            verdict = "better than baseline"
        else:
            verdict = "WARNING: not better than the baseline"
        print(
            f"{role} version {version} valid PR-AUC {round(model_pr_auc, 4)}",
            f"vs baseline {round(baseline_pr_auc, 4)} -> {verdict}",
        )

    filepath = config["training_results_directory"] + "model_comparison_" + date_part + ".png"
    title = "Training " + snapshotdate + ": models on the valid snapshot " + valid_date
    plot_comparison(compare_pdf, title, filepath)
    print("saved to:", filepath)

    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
