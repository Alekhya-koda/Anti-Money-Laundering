import argparse
import logging
import os
import pprint
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from utils.model_utils import (
    MODEL_ALIASES,
    MODEL_NAME,
    evaluate_scores,
    explain,
    load_features,
    load_registered_model,
    load_training_data,
    psi,
)

# daily monitoring after the batch inference: model quality (labels come 2 days late), data
# quality, and feature / score / attribution drift of the champion vs its training snapshot (PSI).
# Big drift on an important feature -> ERROR log "DRIFT ALERT" + a row in drift_alerts.csv,
# nothing is retrained automatically: a person decides whether to run aml_training_pipeline
# run from the project folder:
#   python scripts/model_monitor.py --snapshotdate "2022-09-09"


def check_model_quality(config):
    # the labels of snapshot_date - 2 just arrived, compare them with the scores made back then
    label_date = config["label_snapshot_date_str"]
    eval_pdf = load_training_data(label_date)
    if eval_pdf is None:
        print("labels of", label_date, "not complete yet, skip model quality")
        return

    rows = []
    # rule baselines: rank accounts by one feature, what the bank could do without a model
    for rule in ["out_cnt", "out_big_cnt"]:
        metrics = evaluate_scores(eval_pdf["label"], eval_pdf[rule])
        rows.append(dict(model_alias="rule_" + rule, model_version="-", **metrics))

    for alias in MODEL_ALIASES:
        filename = alias + "_predictions_" + label_date.replace("-", "_") + ".parquet"
        filepath = config["predictions_directory"] + alias + "/" + filename
        if not os.path.exists(filepath):
            continue
        predictions_pdf = pd.read_parquet(filepath)
        merged_pdf = eval_pdf[["acct", "label"]].merge(
            predictions_pdf[["acct", "score"]], on="acct", how="left"
        )
        metrics = evaluate_scores(merged_pdf["label"], merged_pdf["score"].fillna(0))
        version = predictions_pdf["model_version"].iloc[0]
        rows.append(dict(model_alias=alias, model_version=version, **metrics))

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
    print("saved to:", filepath)


def check_data_quality(config):
    # dq report of the last ingested day, also checked before there is a model
    yesterday = (config["snapshot_date"] - timedelta(days=1)).strftime("%Y_%m_%d")
    dq_filepath = config["dq_directory"] + "dq_report_" + yesterday + ".csv"
    if not os.path.exists(dq_filepath):
        return
    dq = pd.read_csv(dq_filepath).iloc[0]
    if dq["silver_rows"] == 0:
        print("WARNING: no transactions ingested yesterday")
    if dq["invalid_removed"] > 0.01 * dq["bronze_rows"]:
        print("WARNING: more than 1% of rows removed as invalid")


def check_drift(config):
    # drift of today's snapshot vs the snapshot the champion was trained on
    snapshotdate = config["snapshot_date_str"]
    champion_model, champion_mv = load_registered_model("champion")
    current_pdf = load_features(snapshotdate)
    if champion_model is None or current_pdf is None:
        print("no champion or no features yet, skip drift")
        return
    train_date = champion_mv.tags["model_train_date"]
    if snapshotdate < train_date:
        # same rule as model_inference.py: no scores for snapshots older than the train date
        print(f"champion version {champion_mv.version} was trained on {train_date} -> skip drift")
        return

    reference_date = champion_mv.tags["train_snapshot"]
    reference_pdf = load_features(reference_date)
    feature_cols = list(champion_model.feature_names_in_)
    print("drift: snapshot", snapshotdate, "vs champion training snapshot", reference_date)

    drift_pdf = pd.DataFrame({"feature": feature_cols})
    drift_pdf["psi"] = [psi(reference_pdf[c], current_pdf[c]) for c in feature_cols]

    # attribution drift on a sample, contributions of all 250k accounts are slow and not needed
    ref_sample = reference_pdf[feature_cols].sample(20000, random_state=88)
    cur_sample = current_pdf[feature_cols].sample(20000, random_state=88)
    ref_contrib = explain(champion_model, ref_sample).abs().mean()
    cur_contrib = explain(champion_model, cur_sample).abs().mean()
    drift_pdf["ref_attribution_share"] = (ref_contrib / ref_contrib.sum()).values
    drift_pdf["cur_attribution_share"] = (cur_contrib / cur_contrib.sum()).values

    ref_scores = champion_model.predict_proba(reference_pdf[feature_cols])[:, 1]
    cur_scores = champion_model.predict_proba(current_pdf[feature_cols])[:, 1]
    score_row = pd.DataFrame([{"feature": "score", "psi": psi(ref_scores, cur_scores)}])
    drift_pdf = pd.concat([drift_pdf, score_row], ignore_index=True)

    threshold = config["psi_threshold"]
    drift_pdf["status"] = np.where(
        drift_pdf["psi"] > threshold, "drift", np.where(drift_pdf["psi"] > 0.1, "watch", "ok")
    )
    drift_pdf.insert(0, "snapshot_date", snapshotdate)
    drift_pdf.insert(1, "reference_snapshot", reference_date)
    drift_pdf.insert(2, "champion_version", champion_mv.version)

    filepath = (
        config["monitoring_directory"] + "drift_report_" + snapshotdate.replace("-", "_") + ".csv"
    )
    drift_pdf.to_csv(filepath, index=False)
    print(drift_pdf.sort_values("psi", ascending=False).head(12).round(4).to_string(index=False))
    print("saved to:", filepath)

    alert_on_drift(config, drift_pdf, champion_mv.version, filepath)


def alert_on_drift(config, drift_pdf, champion_version, report_path):
    snapshotdate = config["snapshot_date_str"]
    threshold = config["psi_threshold"]

    # only the features the champion relies on matter, so watch the top ones by attribution
    watch = drift_pdf.dropna(subset=["ref_attribution_share"])
    watch = watch.sort_values("ref_attribution_share", ascending=False)
    watch = watch.head(config["top_features_to_watch"])
    drifted = list(watch.loc[watch["psi"] > threshold, "feature"])
    if drift_pdf.loc[drift_pdf["feature"] == "score", "psi"].iloc[0] > threshold:
        drifted.append("score")

    if not drifted:
        print("no drift on the important features")
        return

    # no automatic retraining: a person checks the drift report and the shadow results
    aliases = MlflowClient().get_registered_model(MODEL_NAME).aliases
    shadow_models = (
        ", ".join(alias + " v" + aliases[alias] for alias in MODEL_ALIASES[1:] if alias in aliases)
        or "-"
    )
    logging.error(
        "DRIFT ALERT snapshot %s: champion version %s drifted on %s (PSI > %s). "
        "Check %s and the shadow results of %s, "
        "then decide whether to run aml_training_pipeline.",
        snapshotdate,
        champion_version,
        ", ".join(drifted),
        threshold,
        report_path,
        shadow_models,
    )
    # in production the alert would also be pushed to the person on duty here,
    # e.g. a Telegram bot message (token and chat id from environment variables)

    # keep every alert in one file, replace the row of this snapshot if the job is re-run
    alert_pdf = pd.DataFrame(
        [
            {
                "snapshot_date": snapshotdate,
                "champion_version": champion_version,
                "shadow_models": shadow_models,
                "drifted": ", ".join(drifted),
            }
        ]
    )
    alert_filepath = config["monitoring_directory"] + "drift_alerts.csv"
    if os.path.exists(alert_filepath):
        history_pdf = pd.read_csv(alert_filepath, dtype=str)
        history_pdf = history_pdf[history_pdf["snapshot_date"] != snapshotdate]
        alert_pdf = pd.concat([history_pdf, alert_pdf], ignore_index=True)
    alert_pdf.to_csv(alert_filepath, index=False)
    print("alert saved to:", alert_filepath)


def main(snapshotdate):
    print("\n\n---starting job---\n\n")

    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(snapshotdate, "%Y-%m-%d")
    config["label_snapshot_date_str"] = (config["snapshot_date"] - timedelta(days=2)).strftime(
        "%Y-%m-%d"
    )
    config["predictions_directory"] = "datamart/gold/model_predictions/"
    config["monitoring_directory"] = "datamart/gold/model_monitoring/"
    config["dq_directory"] = "datamart/silver/data_quality/"
    config["psi_threshold"] = 0.25
    config["top_features_to_watch"] = 10
    pprint.pprint(config)

    os.makedirs(config["monitoring_directory"], exist_ok=True)

    check_model_quality(config)
    check_data_quality(config)
    check_drift(config)

    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
