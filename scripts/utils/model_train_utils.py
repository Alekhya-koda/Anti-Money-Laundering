import os

import mlflow
import mlflow.sklearn
import pandas as pd

from utils.model_utils import evaluate_scores, load_training_data

# steps shared by the 3 training scripts: load the snapshots, log + register the model in MLflow,
# save the result for model_select.py


def load_train_valid(config):
    train_pdf = load_training_data(config["train_snapshot_date_str"])
    valid_pdf = load_training_data(config["valid_snapshot_date_str"])
    if train_pdf is None or valid_pdf is None:
        # in the DAG model_train_check.py has already checked this
        raise ValueError("train / valid snapshot not ready, run model_train_check.py first")

    for name, pdf in [("train", train_pdf), ("valid", valid_pdf)]:
        positive_pct = round(100 * pdf["label"].mean(), 3)
        print(name, pdf.shape[0], "positives", pdf["label"].sum(), positive_pct, "%")
    return train_pdf, valid_pdf


def log_model(
    model, feature_cols, params, trials_pdf, train_pdf, valid_pdf, config, client, extra_texts=None
):
    # extra_texts: optional {artifact file name: text} saved with the run (e.g. LR coefficients)
    train_scores = model.predict_proba(train_pdf[feature_cols])[:, 1]
    valid_scores = model.predict_proba(valid_pdf[feature_cols])[:, 1]
    train_results = evaluate_scores(train_pdf["label"], train_scores)
    valid_results = evaluate_scores(valid_pdf["label"], valid_scores)
    print(config["model_key"], "valid:", {k: round(v, 4) for k, v in valid_results.items()})

    mlflow.set_experiment("aml-account-risk")
    run_name = (
        config["model_type"] + "_" + config["feature_set"] + "_" + config["model_train_date_str"]
    )
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_params(
            {
                "model_type": config["model_type"],
                "feature_set": config["feature_set"],
                "n_features": len(feature_cols),
                "train_snapshot": config["train_snapshot_date_str"],
                "valid_snapshot": config["valid_snapshot_date_str"],
            }
        )
        mlflow.log_metrics({"train_" + k: v for k, v in train_results.items()})
        mlflow.log_metrics({"valid_" + k: v for k, v in valid_results.items()})
        # all tuning tries, to show what we tested (e.g. class weights)
        mlflow.log_text(trials_pdf.to_csv(index=False), "tuning_trials.csv")
        for name, text in (extra_texts or {}).items():
            mlflow.log_text(text, name)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=config["model_name"],
            input_example=valid_pdf[feature_cols].head(3),
            pip_requirements=[
                "scikit-learn==1.6.1",
                "lightgbm==4.6.0",
                "numpy==1.26.4",
                "pandas==2.2.3",
            ],
        )

    version = client.search_model_versions("run_id='" + run.info.run_id + "'")[0].version
    tags = {
        "model_type": config["model_type"],
        "feature_set": config["feature_set"],
        "model_train_date": config["model_train_date_str"],
        "train_snapshot": config["train_snapshot_date_str"],
    }
    for key, value in tags.items():
        client.set_model_version_tag(config["model_name"], version, key, value)
    print("registered", config["model_name"], "version", version)

    result = {
        "version": version,
        "model_type": config["model_type"],
        "feature_set": config["feature_set"],
    }
    result.update({"train_" + k: v for k, v in train_results.items()})
    result.update({"valid_" + k: v for k, v in valid_results.items()})
    return result


def save_candidate(result, config):
    # one small csv per model, model_select.py compares the candidates of the day
    os.makedirs(config["training_results_directory"], exist_ok=True)
    date_part = config["model_train_date_str"].replace("-", "_")
    filename = "candidate_" + config["model_key"] + "_" + date_part + ".csv"
    filepath = config["training_results_directory"] + filename
    pd.DataFrame([result]).to_csv(filepath, index=False)
    print("saved to:", filepath)
