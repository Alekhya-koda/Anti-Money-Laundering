import argparse
import pprint
import pandas as pd

import lightgbm as lgb
from mlflow.tracking import MlflowClient
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import average_precision_score

from utils.model_utils import MODEL_NAME, CANDIDATE_MODELS, train_valid_snapshots, feature_columns
from utils.model_train_utils import load_train_valid, log_model, save_candidate

# trains LightGBM on the 24 base features + the 6 graph features, tunes it on the valid snapshot
# and registers it in MLflow. In the DAG it only runs on the days model_train_check.py says we train,
# then model_select.py compares it with the other two models
# to call this script (from the project folder): python scripts/model_train_lightgbm_graph.py --snapshotdate "2022-09-09"

MODEL_KEY = "lightgbm_graph"


def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # --- set up config ---
    config = {}
    config["model_key"] = MODEL_KEY
    config["model_type"] = CANDIDATE_MODELS[MODEL_KEY]["model_type"]
    config["feature_set"] = CANDIDATE_MODELS[MODEL_KEY]["feature_set"]
    config["model_train_date_str"] = snapshotdate
    config["train_snapshot_date_str"], config["valid_snapshot_date_str"] = train_valid_snapshots(snapshotdate)
    config["model_name"] = MODEL_NAME
    config["training_results_directory"] = "datamart/gold/model_training/"
    pprint.pprint(config)

    # --- get data ---
    train_pdf, valid_pdf = load_train_valid(config)
    cols = feature_columns(train_pdf, with_graph=True)

    # --- train: random search, every try uses early stopping on the valid snapshot ---
    # same search as model_train_lightgbm.py, so the only difference between the two is the graph features
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
    print('best params:', best[2])

    # --- log to MLflow, register a new model version, save the result for model_select.py ---
    result = log_model(best[1], cols, best[2], pd.DataFrame(trials), train_pdf, valid_pdf, config, MlflowClient())
    save_candidate(result, config)

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)
