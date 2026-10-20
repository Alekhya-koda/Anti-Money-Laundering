import argparse
import os
import pprint
import numpy as np
import pandas as pd

from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import average_precision_score

from utils.model_utils import MODEL_NAME, CANDIDATE_MODELS, train_valid_snapshots, feature_columns
from utils.model_train_utils import load_train_valid, log_model, save_candidate

# trains the logistic regression baseline on the 24 base features, tunes it on the valid snapshot
# and registers it in MLflow. In the DAG it only runs on the days model_train_check.py says we train,
# then model_select.py compares it with the two LightGBM models
# to call this script (from the project folder): python scripts/model_train_logistic_regression.py --snapshotdate "2022-09-09"

MODEL_KEY = "logistic_regression"
C_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 10.0]


def coefficient_table(model, cols):
    # coefficients are per 1 standard deviation of the arcsinh-transformed feature, so they can be
    # compared with each other. odds_ratio = exp(coef): how much the odds of laundering are multiplied
    # when the feature goes up by one standard deviation (all other features equal).
    # A coefficient of exactly 0 means the L1 penalty dropped the feature.
    coef = model[-1].coef_[0]
    pdf = pd.DataFrame({"feature": cols, "coef": coef, "odds_ratio": np.exp(coef)})
    pdf["direction"] = np.where(pdf["coef"] > 0, "raises risk", np.where(pdf["coef"] < 0, "lowers risk", "dropped (0)"))
    return pdf.reindex(pdf["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True).round(4)


def save_coefficients(coef_pdf, config):
    os.makedirs(config["training_results_directory"], exist_ok=True)
    filepath = config["training_results_directory"] + "coefficients_" + config["model_key"] + "_" + \
        config["model_train_date_str"].replace('-', '_') + ".csv"
    coef_pdf.to_csv(filepath, index=False)
    print('saved to:', filepath)


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
    cols = feature_columns(train_pdf, with_graph=False)

    # --- train: grid search, pick the best on the valid snapshot ---
    # arcsinh works like log for the big counts and also accepts negative values (fan_diff)
    # L2 (ridge) and L1 (lasso, drops weak features) are both tried over a wider range of C.
    # class_weight stays in the L2 grid only: it is the documented evidence that weighting the
    # positives hurts (see the report); the slower L1 solver is only run with natural weights.
    param_grid = [
        {"penalty": ["l2"], "solver": ["lbfgs"], "C": C_VALUES, "class_weight": [None, "balanced"]},
        {"penalty": ["l1"], "solver": ["liblinear"], "C": C_VALUES, "class_weight": [None]},
    ]
    best, trials = None, []
    for params in ParameterGrid(param_grid):
        model = Pipeline([
            ("arcsinh", FunctionTransformer(np.arcsinh)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=88, **params)),
        ])
        model.fit(train_pdf[cols], train_pdf["label"])
        valid_pr_auc = average_precision_score(valid_pdf["label"], model.predict_proba(valid_pdf[cols])[:, 1])
        trials.append(dict(params, valid_pr_auc=valid_pr_auc))
        if best is None or valid_pr_auc > best[0]:
            best = (valid_pr_auc, model, params)
    print('best params:', best[2])

    # --- coefficient table: the baseline is the transparent model, so show what it learned ---
    coef_pdf = coefficient_table(best[1], cols)
    print(coef_pdf.to_string(index=False))
    save_coefficients(coef_pdf, config)

    # --- log to MLflow, register a new model version, save the result for model_select.py ---
    result = log_model(best[1], cols, best[2], pd.DataFrame(trials), train_pdf, valid_pdf, config, MlflowClient(),
                       extra_texts={"coefficients.csv": coef_pdf.to_csv(index=False)})
    save_candidate(result, config)

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)
