import argparse
import pprint

import lightgbm as lgb
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler

from utils.model_train_utils import load_train_valid, log_model, save_candidate
from utils.model_utils import CANDIDATE_MODELS, MODEL_NAME, feature_columns, train_valid_snapshots

# LightGBM on the 24 base features, tuned on the valid snapshot and registered in MLflow
# run from the project folder:
#   python scripts/model_train_lightgbm.py --snapshotdate "2022-09-09"

MODEL_KEY = "lightgbm"


def main(snapshotdate):
    print("\n\n---starting job---\n\n")

    config = {}
    config["model_key"] = MODEL_KEY
    config["model_type"] = CANDIDATE_MODELS[MODEL_KEY]["model_type"]
    config["feature_set"] = CANDIDATE_MODELS[MODEL_KEY]["feature_set"]
    config["model_train_date_str"] = snapshotdate
    config["train_snapshot_date_str"], config["valid_snapshot_date_str"] = train_valid_snapshots(
        snapshotdate
    )
    config["model_name"] = MODEL_NAME
    config["training_results_directory"] = "datamart/gold/model_training/"
    pprint.pprint(config)

    train_pdf, valid_pdf = load_train_valid(config)
    cols = feature_columns(train_pdf, with_graph=False)
    X_train, y_train = train_pdf[cols], train_pdf["label"]
    X_valid, y_valid = valid_pdf[cols], valid_pdf["label"]

    # random search, every try uses early stopping on the valid snapshot
    # scale_pos_weight is in the search to check if class weights help with 0.2% positives
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
        model = lgb.LGBMClassifier(
            n_estimators=2000,
            subsample=0.8,
            subsample_freq=1,
            random_state=88,
            verbose=-1,
            **params,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        valid_pr_auc = average_precision_score(y_valid, model.predict_proba(X_valid)[:, 1])
        trials.append(
            dict(params, best_iteration=model.best_iteration_, valid_pr_auc=valid_pr_auc)
        )
        if best is None or valid_pr_auc > best[0]:
            best = (valid_pr_auc, model, dict(params, n_estimators=model.best_iteration_))
    print("best params:", best[2])

    trials_pdf = pd.DataFrame(trials)
    result = log_model(
        best[1], cols, best[2], trials_pdf, train_pdf, valid_pdf, config, MlflowClient()
    )
    save_candidate(result, config)

    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
