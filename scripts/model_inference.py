import argparse
import os
import pprint
import numpy as np
import pandas as pd

from utils.model_utils import MODEL_NAME, MODEL_ALIASES, load_features, load_registered_model, reason_codes

# batch inference: score every account of a snapshot with the champion (production) and the shadow
# models (challenger, or shadow_1 and shadow_2 after a retrain), then write the morning alert list for the investigators
# to call this script (from the project folder): python scripts/model_inference.py --snapshotdate "2022-09-09"

def main(snapshotdate, topk):
    print('\n\n---starting job---\n\n')

    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["model_name"] = MODEL_NAME
    config["alert_list_size"] = topk
    config["predictions_directory"] = "datamart/gold/model_predictions/"
    config["alert_list_directory"] = "datamart/gold/alert_list/"
    pprint.pprint(config)

    # --- load models from the model registry ---
    # a model only scores snapshots from its train date on: older snapshots were used to train or tune it
    # (happens when the inference DAG catches up on old days after the first models were trained)
    models = {}
    for alias in MODEL_ALIASES:
        model, mv = load_registered_model(alias)
        if model is None:
            continue
        if snapshotdate < mv.tags["model_train_date"]:
            print(alias, 'version', mv.version, 'was trained on', mv.tags["model_train_date"], '-> does not score this snapshot')
            continue
        models[alias] = (model, mv)
        print('loaded', alias, '-> version', mv.version, mv.tags.get("model_type"), mv.tags.get("feature_set"))

    if "champion" not in models:
        print('no champion model for this snapshot yet, skipped')
        return

    # --- load feature store ---
    features_pdf = load_features(snapshotdate)
    if features_pdf is None:
        print('no features for this snapshot, skipped')
        return
    print('accounts to score:', len(features_pdf))

    # --- model prediction inference ---
    date_part = snapshotdate.replace('-','_')
    for alias, (model, mv) in models.items():
        X_inference = features_pdf[list(model.feature_names_in_)]

        predictions_pdf = features_pdf[["acct", "snapshot_date"]].copy()
        predictions_pdf["model_alias"] = alias
        predictions_pdf["model_version"] = mv.version
        predictions_pdf["score"] = model.predict_proba(X_inference)[:, 1]

        # save to the inference store, the monitor compares them when the labels arrive
        predictions_directory = config["predictions_directory"] + alias + "/"
        if not os.path.exists(predictions_directory):
            os.makedirs(predictions_directory)
        filepath = predictions_directory + alias + "_predictions_" + date_part + ".parquet"
        predictions_pdf.to_parquet(filepath, index=False)
        print('saved to:', filepath)

    # --- alert list for the investigators, champion only ---
    champion_model, champion_mv = models["champion"]
    champion_cols = list(champion_model.feature_names_in_)
    alert_pdf = features_pdf.copy()
    alert_pdf["score"] = champion_model.predict_proba(alert_pdf[champion_cols])[:, 1]

    # shuffle before sorting so ties are broken randomly
    alert_pdf = alert_pdf.sample(frac=1, random_state=88).sort_values("score", ascending=False, kind="mergesort")
    alert_pdf = alert_pdf.head(config["alert_list_size"]).reset_index(drop=True)
    alert_pdf["rank"] = alert_pdf.index + 1

    # reason codes: top 3 features that push the score up
    reasons = reason_codes(champion_model, alert_pdf[champion_cols], top_n=3)
    for i in range(3):
        alert_pdf["reason_" + str(i + 1)] = [r[i] if len(r) > i else "" for r in reasons]
    alert_pdf["model_version"] = champion_mv.version

    alert_pdf = alert_pdf[["rank", "acct", "score", "reason_1", "reason_2", "reason_3", "model_version", "snapshot_date"]]

    if not os.path.exists(config["alert_list_directory"]):
        os.makedirs(config["alert_list_directory"])
    filepath = config["alert_list_directory"] + "alert_list_" + date_part + ".csv"
    alert_pdf.to_csv(filepath, index=False)
    print('saved to:', filepath)
    print(alert_pdf.head(10).to_string(index=False))

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--topk", type=int, default=500, help="size of the daily alert list")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.topk)
