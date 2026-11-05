import argparse
import os
import pprint

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from utils.model_train_utils import load_train_valid, log_model, save_candidate
from utils.model_utils import CANDIDATE_MODELS, MODEL_NAME, feature_columns, train_valid_snapshots

# logistic regression baseline on the 24 base features, tuned on the valid snapshot and
# registered in MLflow
# run from the project folder:
#   python scripts/model_train_logistic_regression.py --snapshotdate "2022-09-09"

MODEL_KEY = "logistic_regression"
C_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 10.0]


def coefficient_table(model, cols):
    # coef is per 1 std of the arcsinh feature, so they can be compared with each other
    # odds_ratio = exp(coef), coef = 0 means the L1 penalty dropped the feature
    coef = model[-1].coef_[0]
    pdf = pd.DataFrame({"feature": cols, "coef": coef, "odds_ratio": np.exp(coef)})
    pdf["direction"] = np.where(
        pdf["coef"] > 0, "raises risk", np.where(pdf["coef"] < 0, "lowers risk", "dropped (0)")
    )
    order = pdf["coef"].abs().sort_values(ascending=False).index
    return pdf.reindex(order).reset_index(drop=True).round(4)


def save_coefficients(coef_pdf, config):
    os.makedirs(config["training_results_directory"], exist_ok=True)
    date_part = config["model_train_date_str"].replace("-", "_")
    filename = "coefficients_" + config["model_key"] + "_" + date_part + ".csv"
    filepath = config["training_results_directory"] + filename
    coef_pdf.to_csv(filepath, index=False)
    print("saved to:", filepath)


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

    # arcsinh works like log for the big counts and also accepts negative values (fan_diff)
    # class_weight is only in the L2 grid, enough to show that weighting the positives hurts;
    # the slower L1 solver only runs with natural weights
    param_grid = [
        {
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "C": C_VALUES,
            "class_weight": [None, "balanced"],
        },
        {"penalty": ["l1"], "solver": ["liblinear"], "C": C_VALUES, "class_weight": [None]},
    ]
    best, trials = None, []
    for params in ParameterGrid(param_grid):
        model = Pipeline(
            [
                ("arcsinh", FunctionTransformer(np.arcsinh)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=88, **params)),
            ]
        )
        model.fit(X_train, y_train)
        valid_pr_auc = average_precision_score(y_valid, model.predict_proba(X_valid)[:, 1])
        trials.append(dict(params, valid_pr_auc=valid_pr_auc))
        if best is None or valid_pr_auc > best[0]:
            best = (valid_pr_auc, model, params)
    print("best params:", best[2])

    # the baseline is the transparent model, so keep what it learned
    coef_pdf = coefficient_table(best[1], cols)
    save_coefficients(coef_pdf, config)

    result = log_model(
        best[1],
        cols,
        best[2],
        pd.DataFrame(trials),
        train_pdf,
        valid_pdf,
        config,
        MlflowClient(),
        extra_texts={"coefficients.csv": coef_pdf.to_csv(index=False)},
    )
    save_candidate(result, config)

    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
