import argparse
import os
import pprint
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from utils.model_utils import MODEL_ALIASES, load_features, load_training_data, load_registered_model, \
    evaluate_scores, psi, explain

# daily monitoring, runs after the batch inference
#   1. model quality  - labels come 2 days late, so check the predictions made for snapshot_date - 2
#   2. data drift     - PSI of each feature, today's snapshot vs the champion's training snapshot
#   3. score drift    - PSI of the champion scores, same comparison
#   4. attribution drift - share of the champion's |feature contribution|, same comparison
#   5. data quality   - rows ingested yesterday from the silver dq report
# big drift on an important feature writes retrain_flag.txt -> the 3 model_train_*.py scripts train a new challenger
# to call this script (from the project folder): python scripts/model_monitor.py --snapshotdate "2022-09-09"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(snapshotdate, "%Y-%m-%d")
    config["label_snapshot_date_str"] = (config["snapshot_date"] - timedelta(days=2)).strftime("%Y-%m-%d")
    config["predictions_directory"] = "datamart/gold/model_predictions/"
    config["monitoring_directory"] = "datamart/gold/model_monitoring/"
    config["dq_directory"] = "datamart/silver/data_quality/"
    config["psi_threshold"] = 0.25
    config["top_features_to_watch"] = 10
    pprint.pprint(config)

    if not os.path.exists(config["monitoring_directory"]):
        os.makedirs(config["monitoring_directory"])

    # --- 1. model quality on the snapshot whose labels just arrived ---
    label_date = config["label_snapshot_date_str"]
    eval_pdf = load_training_data(label_date)
    if eval_pdf is None:
        print('labels of', label_date, 'not complete yet, skip model quality')
    else:
        rows = []
        # rule baselines: rank accounts by one feature, what the bank could do without a model
        for rule in ["out_cnt", "out_big_cnt"]:
            rows.append(dict(model_alias="rule_" + rule, model_version="-", **evaluate_scores(eval_pdf["label"], eval_pdf[rule])))

        for alias in MODEL_ALIASES:
            filepath = config["predictions_directory"] + alias + "/" + alias + "_predictions_" + label_date.replace('-','_') + ".parquet"
            if not os.path.exists(filepath):
                continue
            predictions_pdf = pd.read_parquet(filepath)
            merged_pdf = eval_pdf[["acct", "label"]].merge(predictions_pdf[["acct", "score"]], on="acct", how="left")
            rows.append(dict(model_alias=alias, model_version=predictions_pdf["model_version"].iloc[0],
                             **evaluate_scores(merged_pdf["label"], merged_pdf["score"].fillna(0))))

        perf_pdf = pd.DataFrame(rows)
        perf_pdf.insert(0, "snapshot_date", label_date)
        perf_pdf.insert(1, "n_accounts", len(eval_pdf))
        perf_pdf.insert(2, "n_positive", int(eval_pdf["label"].sum()))
        print(perf_pdf.round(4).to_string(index=False))

        # append to the history file, replace old rows of this snapshot if the job is re-run
        filepath = config["monitoring_directory"] + "model_performance.csv"
        if os.path.exists(filepath):
            history_pdf = pd.read_csv(filepath, dtype={"model_version": str})
            history_pdf = history_pdf[history_pdf["snapshot_date"] != label_date]
            perf_pdf = pd.concat([history_pdf, perf_pdf], ignore_index=True)
        perf_pdf.to_csv(filepath, index=False)
        print('saved to:', filepath)

    # --- 2, 3, 4. drift of today's snapshot vs the champion's training snapshot ---
    champion_model, champion_mv = load_registered_model("champion")
    current_pdf = load_features(snapshotdate)
    if champion_model is None or current_pdf is None:
        print('no champion or no features yet, skip drift')
        print('\n\n---completed job---\n\n')
        return

    reference_date = champion_mv.tags["train_snapshot"]
    reference_pdf = load_features(reference_date)
    feature_cols = list(champion_model.feature_names_in_)
    print('drift: snapshot', snapshotdate, 'vs champion training snapshot', reference_date)

    drift_pdf = pd.DataFrame({"feature": feature_cols})
    drift_pdf["psi"] = [psi(reference_pdf[c], current_pdf[c]) for c in feature_cols]

    # attribution drift on a random sample of accounts (contributions of 250k rows are slow and not needed)
    reference_contrib = explain(champion_model, reference_pdf[feature_cols].sample(20000, random_state=88)).abs().mean()
    current_contrib = explain(champion_model, current_pdf[feature_cols].sample(20000, random_state=88)).abs().mean()
    drift_pdf["ref_attribution_share"] = (reference_contrib / reference_contrib.sum()).values
    drift_pdf["cur_attribution_share"] = (current_contrib / current_contrib.sum()).values

    # score drift
    reference_scores = champion_model.predict_proba(reference_pdf[feature_cols])[:, 1]
    current_scores = champion_model.predict_proba(current_pdf[feature_cols])[:, 1]
    score_row = pd.DataFrame([{"feature": "score", "psi": psi(reference_scores, current_scores)}])
    drift_pdf = pd.concat([drift_pdf, score_row], ignore_index=True)

    drift_pdf["status"] = np.where(drift_pdf["psi"] > config["psi_threshold"], "drift",
                                   np.where(drift_pdf["psi"] > 0.1, "watch", "ok"))
    drift_pdf.insert(0, "snapshot_date", snapshotdate)
    drift_pdf.insert(1, "reference_snapshot", reference_date)
    drift_pdf.insert(2, "champion_version", champion_mv.version)

    filepath = config["monitoring_directory"] + "drift_report_" + snapshotdate.replace('-','_') + ".csv"
    drift_pdf.to_csv(filepath, index=False)
    print(drift_pdf.sort_values("psi", ascending=False).head(12).round(4).to_string(index=False))
    print('saved to:', filepath)

    # --- 5. data quality of the last ingested day ---
    yesterday = (config["snapshot_date"] - timedelta(days=1)).strftime("%Y_%m_%d")
    dq_filepath = config["dq_directory"] + "dq_report_" + yesterday + ".csv"
    if os.path.exists(dq_filepath):
        dq_pdf = pd.read_csv(dq_filepath)
        print(dq_pdf.to_string(index=False))
        if dq_pdf["silver_rows"].iloc[0] == 0:
            print('WARNING: no transactions ingested yesterday')
        if dq_pdf["invalid_removed"].iloc[0] > 0.01 * dq_pdf["bronze_rows"].iloc[0]:
            print('WARNING: more than 1% of rows removed as invalid')

    # --- retrain flag ---
    # only the features the champion relies on matter, so watch the top ones by attribution
    watch = drift_pdf.dropna(subset=["ref_attribution_share"]).sort_values("ref_attribution_share", ascending=False)
    watch = watch.head(config["top_features_to_watch"])
    drifted = list(watch.loc[watch["psi"] > config["psi_threshold"], "feature"])
    if drift_pdf.loc[drift_pdf["feature"] == "score", "psi"].iloc[0] > config["psi_threshold"]:
        drifted.append("score")

    # if a newer challenger is already waiting for approval, do not ask for another retrain
    challenger_model, challenger_mv = load_registered_model("challenger")
    waiting = challenger_mv is not None and challenger_mv.tags["model_train_date"] > champion_mv.tags["model_train_date"]

    flag_filepath = config["monitoring_directory"] + "retrain_flag.txt"
    if drifted and not waiting:
        with open(flag_filepath, "w") as f:
            f.write(snapshotdate + " drift on: " + ", ".join(drifted) + "\n")
        print('DRIFT on', drifted, '-> retrain flag written:', flag_filepath)
    elif drifted:
        print('drift on', drifted, 'but challenger version', challenger_mv.version, 'is already waiting for approval')
    else:
        print('no drift on the important features')

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)
