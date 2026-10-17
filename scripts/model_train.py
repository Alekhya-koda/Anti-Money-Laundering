import argparse
import os
import pprint
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.metrics import average_precision_score

from utils.model_utils import MODEL_NAME, CANDIDATE_MODELS, train_valid_snapshots, load_training_data, \
    feature_columns, evaluate_scores

# trains ONE candidate model, tunes it on the valid snapshot and registers it in MLflow
# the DAG runs it once per model (logistic_regression, lightgbm, lightgbm_graph), only on the days
# model_train_check.py says we train; model_select.py then picks the champion
# to call this script (from the project folder): python scripts/model_train.py --snapshotdate "2022-09-09" --model lightgbm


def train_logistic_regression(train_pdf, valid_pdf, cols):
    # grid search, arcsinh works like log for the big counts and also accepts negative values (fan_diff)
    best, trials = None, []
    for params in ParameterGrid({"C": [0.01, 0.1, 1.0], "class_weight": [None, "balanced"]}):
        model = Pipeline([
            ("arcsinh", FunctionTransformer(np.arcsinh)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, **params)),
        ])
        model.fit(train_pdf[cols], train_pdf["label"])
        valid_pr_auc = average_precision_score(valid_pdf["label"], model.predict_proba(valid_pdf[cols])[:, 1])
        trials.append(dict(params, valid_pr_auc=valid_pr_auc))
        if best is None or valid_pr_auc > best[0]:
            best = (valid_pr_auc, model, params)
    return best[1], best[2], pd.DataFrame(trials)


def train_lightgbm(train_pdf, valid_pdf, cols):
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
    best, trials = None, []
    for params in ParameterSampler(param_dist, n_iter=20, random_state=88):
        model = lgb.LGBMClassifier(n_estimators=2000, subsample=0.8, subsample_freq=1, random_state=88, verbose=-1, **params)
        model.fit(train_pdf[cols], train_pdf["label"],
                  eval_set=[(valid_pdf[cols], valid_pdf["label"])], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        valid_pr_auc = average_precision_score(valid_pdf["label"], model.predict_proba(valid_pdf[cols])[:, 1])
        trials.append(dict(params, best_iteration=model.best_iteration_, valid_pr_auc=valid_pr_auc))
        if best is None or valid_pr_auc > best[0]:
            best = (valid_pr_auc, model, dict(params, n_estimators=model.best_iteration_))
    return best[1], best[2], pd.DataFrame(trials)


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


def main(snapshotdate, model_key):
    print('\n\n---starting job---\n\n')

    # --- set up config ---
    config = {}
    config["model_train_date_str"] = snapshotdate
    config["train_snapshot_date_str"], config["valid_snapshot_date_str"] = train_valid_snapshots(snapshotdate)
    config["model_key"] = model_key
    config["model_type"] = CANDIDATE_MODELS[model_key]["model_type"]
    config["feature_set"] = CANDIDATE_MODELS[model_key]["feature_set"]
    config["model_name"] = MODEL_NAME
    config["training_results_directory"] = "datamart/gold/model_training/"
    pprint.pprint(config)

    # --- get data ---
    train_pdf = load_training_data(config["train_snapshot_date_str"])
    valid_pdf = load_training_data(config["valid_snapshot_date_str"])
    if train_pdf is None or valid_pdf is None:
        # in the DAG model_train_check.py already made sure both snapshots are there
        raise ValueError("train / valid snapshot not ready, run model_train_check.py first")

    print('train', train_pdf.shape[0], 'positives', train_pdf["label"].sum(), round(100 * train_pdf["label"].mean(), 3), '%')
    print('valid', valid_pdf.shape[0], 'positives', valid_pdf["label"].sum(), round(100 * valid_pdf["label"].mean(), 3), '%')

    # --- train the model, tuned on the valid snapshot ---
    cols = feature_columns(train_pdf, with_graph=(config["feature_set"] == "base_graph"))
    if config["model_type"] == "logistic_regression":
        model, params, trials_pdf = train_logistic_regression(train_pdf, valid_pdf, cols)
    else:
        model, params, trials_pdf = train_lightgbm(train_pdf, valid_pdf, cols)
    print(model_key, 'best params:', params)

    # --- log to MLflow and register a new model version ---
    mlflow.set_experiment("aml-account-risk")
    client = MlflowClient()
    result = log_model(model, config["model_type"], config["feature_set"], cols, params, trials_pdf,
                       train_pdf, valid_pdf, config, client)
    print('registered', config["model_name"], 'version', result["version"])

    # --- save the result, model_select.py compares the candidates of the day ---
    if not os.path.exists(config["training_results_directory"]):
        os.makedirs(config["training_results_directory"])
    filepath = config["training_results_directory"] + "candidate_" + model_key + "_" + snapshotdate.replace('-','_') + ".csv"
    pd.DataFrame([result]).to_csv(filepath, index=False)
    print('saved to:', filepath)

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--model", type=str, required=True, choices=list(CANDIDATE_MODELS), help="which model to train")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.model)
