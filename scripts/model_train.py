import argparse
import os
import pprint
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.metrics import average_precision_score

from utils.model_utils import MODEL_NAME, load_training_data, feature_columns, evaluate_scores

# trains logistic regression + lightgbm (with and without graph features) and registers them in MLflow
# runs every day in the DAG but only trains when there is no champion yet, the champion is a week old,
# or the monitor raised a drift flag
# to call this script (from the project folder): python scripts/model_train.py --snapshotdate "2022-09-09"


def log_model(model, model_type, feature_set, feature_cols, params, trials_pdf, train_pdf, valid_pdf, config, client):
    # evaluate on train and valid, log everything to MLflow and register a new model version
    train_results = evaluate_scores(train_pdf["label"], model.predict_proba(train_pdf[feature_cols])[:, 1])
    valid_results = evaluate_scores(valid_pdf["label"], model.predict_proba(valid_pdf[feature_cols])[:, 1])
    print(model_type, feature_set, 'valid:', {k: round(v, 4) for k, v in valid_results.items()})

    with mlflow.start_run(run_name=model_type + "_" + feature_set + "_" + config["model_train_date_str"]) as run:
        mlflow.log_params(params)
        mlflow.log_params({
            "model_type": model_type,
            "feature_set": feature_set,
            "n_features": len(feature_cols),
            "train_snapshot": config["train_snapshot_date_str"],
            "valid_snapshot": config["valid_snapshot_date_str"],
        })
        mlflow.log_metrics({"train_" + k: v for k, v in train_results.items()})
        mlflow.log_metrics({"valid_" + k: v for k, v in valid_results.items()})
        # all tuning tries, to show what we tested (e.g. class weights)
        mlflow.log_text(trials_pdf.to_csv(index=False), "tuning_trials.csv")
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=config["model_name"],
                                 input_example=valid_pdf[feature_cols].head(3),
                                 pip_requirements=["scikit-learn==1.6.1", "lightgbm==4.6.0", "numpy==1.26.4", "pandas==2.2.3"])

    version = client.search_model_versions("run_id='" + run.info.run_id + "'")[0].version
    client.set_model_version_tag(config["model_name"], version, "model_type", model_type)
    client.set_model_version_tag(config["model_name"], version, "feature_set", feature_set)
    client.set_model_version_tag(config["model_name"], version, "model_train_date", config["model_train_date_str"])
    client.set_model_version_tag(config["model_name"], version, "train_snapshot", config["train_snapshot_date_str"])

    result = {"version": version, "model_type": model_type, "feature_set": feature_set}
    result.update({"train_" + k: v for k, v in train_results.items()})
    result.update({"valid_" + k: v for k, v in valid_results.items()})
    return result


def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # --- set up config ---
    '''
    labels are 2 days late, so on the train date the newest labelled snapshot is 2 days ago (valid)
    the train snapshot is 2 days before that, so the label windows of train and valid do not overlap

      snapshot     features from     labels from
      09-05 train  09-01 .. 09-04    09-05 .. 09-06
      09-07 valid  09-03 .. 09-06    09-07 .. 09-08
      09-09 test   09-05 .. 09-08    09-09 .. 09-10   <- model trained on 09-09 scores this one
    '''
    config = {}
    config["model_train_date_str"] = snapshotdate
    config["model_train_date"] = datetime.strptime(snapshotdate, "%Y-%m-%d")
    config["valid_snapshot_date_str"] = (config["model_train_date"] - timedelta(days=2)).strftime("%Y-%m-%d")
    config["train_snapshot_date_str"] = (config["model_train_date"] - timedelta(days=4)).strftime("%Y-%m-%d")
    config["model_name"] = MODEL_NAME
    config["retrain_every_days"] = 7
    config["retrain_flag_filepath"] = "datamart/gold/model_monitoring/retrain_flag.txt"
    config["training_results_directory"] = "datamart/gold/model_training/"
    pprint.pprint(config)

    # --- check if we need to train today ---
    client = MlflowClient()
    try:
        champion = client.get_model_version_by_alias(config["model_name"], "champion")
    except Exception:
        champion = None

    if champion is None:
        print('no champion model yet, train the first models')
    elif os.path.exists(config["retrain_flag_filepath"]):
        print('monitor found drift, retrain')
    else:
        champion_train_date = datetime.strptime(champion.tags["model_train_date"], "%Y-%m-%d")
        days_old = (config["model_train_date"] - champion_train_date).days
        if days_old < config["retrain_every_days"]:
            print('champion version', champion.version, 'is', days_old, 'days old, no retrain today, skipped')
            return
        print('champion is', days_old, 'days old, weekly retrain')

    # --- get data ---
    train_pdf = load_training_data(config["train_snapshot_date_str"])
    valid_pdf = load_training_data(config["valid_snapshot_date_str"])
    if train_pdf is None or valid_pdf is None:
        print('train / valid snapshot not ready yet, skipped')
        return

    print('train', train_pdf.shape[0], 'positives', train_pdf["label"].sum(), round(100 * train_pdf["label"].mean(), 3), '%')
    print('valid', valid_pdf.shape[0], 'positives', valid_pdf["label"].sum(), round(100 * valid_pdf["label"].mean(), 3), '%')

    y_train = train_pdf["label"]
    y_valid = valid_pdf["label"]
    mlflow.set_experiment("aml-account-risk")
    candidates = []

    # --- model 1: logistic regression (baseline) ---
    # arcsinh works like log for the big counts and also accepts negative values (fan_diff)
    base_cols = feature_columns(train_pdf, with_graph=False)
    best, trials = None, []
    for params in ParameterGrid({"C": [0.01, 0.1, 1.0], "class_weight": [None, "balanced"]}):
        model = Pipeline([
            ("arcsinh", FunctionTransformer(np.arcsinh)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, **params)),
        ])
        model.fit(train_pdf[base_cols], y_train)
        valid_pr_auc = average_precision_score(y_valid, model.predict_proba(valid_pdf[base_cols])[:, 1])
        trials.append(dict(params, valid_pr_auc=valid_pr_auc))
        if best is None or valid_pr_auc > best[0]:
            best = (valid_pr_auc, model, params)
    print('logistic regression best params:', best[2])
    candidates.append(log_model(best[1], "logistic_regression", "base", base_cols, best[2], pd.DataFrame(trials),
                                train_pdf, valid_pdf, config, client))

    # --- model 2 and 3: lightgbm without / with the graph features ---
    # random search, every try uses early stopping on the valid snapshot
    # scale_pos_weight is in the search to check if class weights help with the 0.2% positive rate
    param_dist = {
        "num_leaves": [7, 15, 31],
        "min_child_samples": [50, 100, 300, 500],
        "learning_rate": [0.02, 0.05],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0, 1, 5],
        "scale_pos_weight": [1, 10, 100],
    }
    for feature_set, with_graph in [("base", False), ("base_graph", True)]:
        cols = feature_columns(train_pdf, with_graph=with_graph)
        best, trials = None, []
        for params in ParameterSampler(param_dist, n_iter=20, random_state=88):
            model = lgb.LGBMClassifier(n_estimators=2000, subsample=0.8, subsample_freq=1, random_state=88, verbose=-1, **params)
            model.fit(train_pdf[cols], y_train,
                      eval_set=[(valid_pdf[cols], y_valid)], eval_metric="average_precision",
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            valid_pr_auc = average_precision_score(y_valid, model.predict_proba(valid_pdf[cols])[:, 1])
            trials.append(dict(params, best_iteration=model.best_iteration_, valid_pr_auc=valid_pr_auc))
            if best is None or valid_pr_auc > best[0]:
                best = (valid_pr_auc, model, dict(params, n_estimators=model.best_iteration_))
        print('lightgbm', feature_set, 'best params:', best[2])
        candidates.append(log_model(best[1], "lightgbm", feature_set, cols, best[2], pd.DataFrame(trials),
                                    train_pdf, valid_pdf, config, client))

    # --- update model registry aliases ---
    candidates_pdf = pd.DataFrame(candidates).sort_values("valid_pr_auc", ascending=False)
    print(candidates_pdf[["version", "model_type", "feature_set", "valid_pr_auc", "valid_p_at_500", "valid_p_at_2000"]].to_string(index=False))

    # pick the simplest model within 10% of the best valid PR-AUC: with ~600 positives small gaps are
    # noise, and the graph features cost one more daily job, so they have to clearly pay off
    complexity = {"logistic_regression_base": 0, "lightgbm_base": 1, "lightgbm_base_graph": 2}
    candidates_pdf["complexity"] = (candidates_pdf["model_type"] + "_" + candidates_pdf["feature_set"]).map(complexity)
    good_enough = candidates_pdf[candidates_pdf["valid_pr_auc"] >= 0.9 * candidates_pdf["valid_pr_auc"].max()]
    best_version = good_enough.sort_values("complexity").iloc[0]["version"]
    # the challenger is the best of the other models
    second_version = candidates_pdf[candidates_pdf["version"] != best_version].iloc[0]["version"]
    lr_version = candidates_pdf[candidates_pdf["model_type"] == "logistic_regression"].iloc[0]["version"]

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

    if os.path.exists(config["retrain_flag_filepath"]):
        os.remove(config["retrain_flag_filepath"])

    # --- save training results ---
    if not os.path.exists(config["training_results_directory"]):
        os.makedirs(config["training_results_directory"])
    filepath = config["training_results_directory"] + "training_results_" + snapshotdate.replace('-','_') + ".csv"
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
