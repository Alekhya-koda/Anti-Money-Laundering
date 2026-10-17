import os
import pandas as pd
import mlflow
import mlflow.sklearn

from utils.model_utils import load_training_data, evaluate_scores

# steps shared by the 3 training scripts (model_train_logistic_regression.py, model_train_lightgbm.py,
# model_train_lightgbm_graph.py): load the snapshots, log + register the model in MLflow,
# save the result for model_select.py


def load_train_valid(config):
    train_pdf = load_training_data(config["train_snapshot_date_str"])
    valid_pdf = load_training_data(config["valid_snapshot_date_str"])
    if train_pdf is None or valid_pdf is None:
        # in the DAG model_train_check.py already made sure both snapshots are there
        raise ValueError("train / valid snapshot not ready, run model_train_check.py first")

    print('train', train_pdf.shape[0], 'positives', train_pdf["label"].sum(), round(100 * train_pdf["label"].mean(), 3), '%')
    print('valid', valid_pdf.shape[0], 'positives', valid_pdf["label"].sum(), round(100 * valid_pdf["label"].mean(), 3), '%')
    return train_pdf, valid_pdf


def log_model(model, feature_cols, params, trials_pdf, train_pdf, valid_pdf, config, client):
    # evaluate on train and valid, log everything to MLflow and register a new model version
    train_results = evaluate_scores(train_pdf["label"], model.predict_proba(train_pdf[feature_cols])[:, 1])
    valid_results = evaluate_scores(valid_pdf["label"], model.predict_proba(valid_pdf[feature_cols])[:, 1])
    print(config["model_key"], 'valid:', {k: round(v, 4) for k, v in valid_results.items()})

    mlflow.set_experiment("aml-account-risk")
    with mlflow.start_run(run_name=config["model_type"] + "_" + config["feature_set"] + "_" + config["model_train_date_str"]) as run:
        mlflow.log_params(params)
        mlflow.log_params({
            "model_type": config["model_type"],
            "feature_set": config["feature_set"],
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
    client.set_model_version_tag(config["model_name"], version, "model_type", config["model_type"])
    client.set_model_version_tag(config["model_name"], version, "feature_set", config["feature_set"])
    client.set_model_version_tag(config["model_name"], version, "model_train_date", config["model_train_date_str"])
    client.set_model_version_tag(config["model_name"], version, "train_snapshot", config["train_snapshot_date_str"])
    print('registered', config["model_name"], 'version', version)

    result = {"version": version, "model_type": config["model_type"], "feature_set": config["feature_set"]}
    result.update({"train_" + k: v for k, v in train_results.items()})
    result.update({"valid_" + k: v for k, v in valid_results.items()})
    return result


def save_candidate(result, config):
    # one small csv per model, model_select.py compares the candidates of the day
    if not os.path.exists(config["training_results_directory"]):
        os.makedirs(config["training_results_directory"])
    filepath = config["training_results_directory"] + "candidate_" + config["model_key"] + "_" + \
        config["model_train_date_str"].replace('-','_') + ".csv"
    pd.DataFrame([result]).to_csv(filepath, index=False)
    print('saved to:', filepath)
