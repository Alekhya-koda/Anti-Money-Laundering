import argparse
import os
import pprint
import pandas as pd

from mlflow.tracking import MlflowClient

from utils.model_utils import MODEL_NAME, CANDIDATE_MODELS

# runs after the 3 training tasks: compares today's candidate models on the valid snapshot
# and moves the aliases in the MLflow model registry (champion / challenger / baseline)
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


def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # --- set up config ---
    config = {}
    config["model_train_date_str"] = snapshotdate
    config["model_name"] = MODEL_NAME
    config["retrain_flag_filepath"] = "datamart/gold/model_monitoring/retrain_flag.txt"
    config["training_results_directory"] = "datamart/gold/model_training/"
    pprint.pprint(config)

    # --- load the results of today's training tasks ---
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

    if os.path.exists(config["retrain_flag_filepath"]):
        os.remove(config["retrain_flag_filepath"])

    # --- save training results ---
    filepath = config["training_results_directory"] + "training_results_" + date_part + ".csv"
    candidates_pdf.to_csv(filepath, index=False)
    print('saved to:', filepath)

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)
