import os
import re
from datetime import datetime, timedelta

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

# one registered model in MLflow, the aliases say which version does what:
#   champion                       production, its scores go to the investigators
#   challenger, shadow_1, shadow_2 shadow mode, scores are saved for the monitor but not used
#   baseline                       logistic regression, only compared at training time
MODEL_NAME = "aml-account-risk"
MODEL_ALIASES = ["champion", "challenger", "shadow_1", "shadow_2"]  # scored by batch inference

ID_COLUMNS = ["acct", "snapshot_date"]
GRAPH_FEATURES = [
    "pagerank",
    "scc_size",
    "nbr_out_deg_sum",
    "nbr_out_deg_avg",
    "nbr_in_deg_sum",
    "nbr_in_deg_avg",
]

# one training script per key (model_train_<key>.py), model_select.py prefers the lower complexity
CANDIDATE_MODELS = {
    "logistic_regression": {
        "model_type": "logistic_regression",
        "feature_set": "base",
        "complexity": 0,
    },
    "lightgbm": {"model_type": "lightgbm", "feature_set": "base", "complexity": 1},
    "lightgbm_graph": {"model_type": "lightgbm", "feature_set": "base_graph", "complexity": 2},
}

# reason text shown to the investigators, worded neutrally: a low value can also raise the score
REASON_TEXT = {
    "out_cnt": "sent {:.0f} payments in the last 4 days",
    "out_amt_pct_avg": "average sent amount is at percentile {:.2f} of its currency",
    "out_amt_pct_max": "largest sent amount is at percentile {:.2f} of its currency",
    "out_big_cnt": "sent {:.0f} payments in the top 1% of their currency",
    "out_uniq_cp": "sent money to {:.0f} different accounts",
    "out_uniq_banks": "sent money to {:.0f} different banks",
    "out_cross_bank_share": "{:.0%} of sent payments go to other banks",
    "out_ach_share": "{:.0%} of sent payments are ACH",
    "out_cash_share": "{:.0%} of sent payments are cash",
    "out_btc_share": "{:.0%} of sent payments are bitcoin",
    "out_cnt_24h": "sent {:.0f} payments in the last 24 hours",
    "out_active_days": "sent payments on {:.0f} different days",
    "in_cnt": "received {:.0f} payments in the last 4 days",
    "in_amt_pct_avg": "average received amount is at percentile {:.2f} of its currency",
    "in_amt_pct_max": "largest received amount is at percentile {:.2f} of its currency",
    "in_uniq_cp": "received money from {:.0f} different accounts",
    "in_uniq_banks": "received money from {:.0f} different banks",
    "in_cnt_24h": "received {:.0f} payments in the last 24 hours",
    "out_top_cp_share": "{:.0%} of sent payments go to one account",
    "n_reciprocal": "money goes both ways with {:.0f} accounts",
    "fan_diff": "fan out minus fan in is {:.0f}",
    "out_in_cnt_ratio": "sends {:.1f} payments per payment received",
    "active_span_hours": "active over {:.0f} hours",
    "hours_since_last_txn": "last payment {:.0f} hours ago",
    "pagerank": "pagerank in the payment network is {:.1e}",
    "scc_size": "inside a payment loop of {:.0f} accounts",
    "nbr_out_deg_sum": "its receivers pay {:.0f} other accounts",
    "nbr_out_deg_avg": "its receivers pay {:.1f} accounts on average",
    "nbr_in_deg_sum": "its senders received from {:.0f} accounts",
    "nbr_in_deg_avg": "its senders received from {:.1f} accounts on average",
}


def model_label(model_type, feature_set):
    if model_type == "logistic_regression":
        return "Logistic regression"
    return "LightGBM + graph" if feature_set == "base_graph" else "LightGBM"


# ----------------------------------------------------------------- feature / label store


def load_features(snapshot_date_str, datamart_directory="datamart/"):
    # feature store + graph features of one snapshot, joined on acct
    date_part = snapshot_date_str.replace("-", "_")
    gold_directory = datamart_directory + "gold/"
    feature_path = gold_directory + "feature_store/gold_feature_store_" + date_part + ".parquet"
    graph_path = (
        gold_directory + "graph_feature_store/gold_graph_features_" + date_part + ".parquet"
    )
    if not os.path.exists(feature_path) or not os.path.exists(graph_path):
        print("features not found:", feature_path)
        return None

    features_pdf = pd.read_parquet(feature_path)
    graph_pdf = pd.read_parquet(graph_path).drop(columns=["snapshot_date"])
    features_pdf = features_pdf.merge(graph_pdf, on="acct", how="left").fillna(0)
    features_pdf["snapshot_date"] = snapshot_date_str
    return features_pdf


def load_labels(snapshot_date_str, datamart_directory="datamart/"):
    date_part = snapshot_date_str.replace("-", "_")
    label_path = datamart_directory + "gold/label_store/gold_label_store_" + date_part + ".parquet"
    if not os.path.exists(label_path):
        print("labels not found:", label_path)
        return None
    return pd.read_parquet(label_path, columns=["acct", "label"])


def load_training_data(snapshot_date_str, datamart_directory="datamart/"):
    # features LEFT JOIN labels: accounts that sent nothing in the label window get label 0
    features_pdf = load_features(snapshot_date_str, datamart_directory)
    labels_pdf = load_labels(snapshot_date_str, datamart_directory)
    if features_pdf is None or labels_pdf is None:
        return None
    data_pdf = features_pdf.merge(labels_pdf, on="acct", how="left")
    data_pdf["label"] = data_pdf["label"].fillna(0).astype(int)
    return data_pdf


def train_valid_snapshots(model_train_date_str):
    """
    labels are 2 days late, so on the train date the newest labelled snapshot (valid) is 2 days old
    the train snapshot is 2 days before that, so the train and valid label windows do not overlap

      snapshot     features from     labels from
      09-05 train  09-01 .. 09-04    09-05 .. 09-06
      09-07 valid  09-03 .. 09-06    09-07 .. 09-08
      09-09 test   09-05 .. 09-08    09-09 .. 09-10   <- model trained on 09-09 scores this one
    """
    model_train_date = datetime.strptime(model_train_date_str, "%Y-%m-%d")
    train_snapshot_date_str = (model_train_date - timedelta(days=4)).strftime("%Y-%m-%d")
    valid_snapshot_date_str = (model_train_date - timedelta(days=2)).strftime("%Y-%m-%d")
    return train_snapshot_date_str, valid_snapshot_date_str


def feature_columns(pdf, with_graph=True):
    cols = [c for c in pdf.columns if c not in ID_COLUMNS + ["label"]]
    if not with_graph:
        cols = [c for c in cols if c not in GRAPH_FEATURES]
    return cols


# ----------------------------------------------------------------- metrics


def precision_recall_at_k(y_true, scores, k, seed=88):
    # shuffle first so ties are ranked randomly, sorting alone would rank them by row order
    pdf = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(scores)})
    pdf = pdf.sample(frac=1, random_state=seed)
    top = pdf.sort_values("score", ascending=False, kind="mergesort").head(k)
    hits = top["y"].sum()
    return hits / k, hits / max(pdf["y"].sum(), 1)


def evaluate_scores(y_true, scores, ks=(500, 2000, 5000)):
    results = {}
    results["pr_auc"] = average_precision_score(y_true, scores)
    results["roc_auc"] = roc_auc_score(y_true, scores)
    for k in ks:
        p, r = precision_recall_at_k(y_true, scores, k)
        results["p_at_" + str(k)] = p
        results["r_at_" + str(k)] = r
    return results


def psi(expected, actual, bins=10):
    # population stability index, bins come from the reference data
    # < 0.1 stable, 0.1 - 0.25 some change, > 0.25 big change
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


# ----------------------------------------------------------------- explanations


def explain(model, X):
    # contribution of each feature to the log odds: tree SHAP values for LightGBM,
    # coefficient x scaled feature value for the logistic regression
    if isinstance(model, Pipeline):
        z = model[:-1].transform(X)
        contrib = z * model[-1].coef_[0]
    else:
        contrib = model.predict(X, pred_contrib=True)[:, :-1]  # last column is the bias
    return pd.DataFrame(contrib, columns=X.columns, index=X.index)


def reason_codes(model, X, top_n=3):
    # top features that push the score up, as text
    contrib = explain(model, X)
    reasons = []
    for idx, row in contrib.iterrows():
        top = row[row > 0].sort_values(ascending=False).head(top_n)
        texts = [REASON_TEXT.get(f, f + " = {:.2f}").format(X.at[idx, f]) for f in top.index]
        # "1 hours" -> "1 hour"
        reasons.append(
            [re.sub(r"\b1 (hour|payment|account|bank|day)s\b", r"1 \1", t) for t in texts]
        )
    return reasons


# ----------------------------------------------------------------- model registry


def load_registered_model(alias, model_name=MODEL_NAME):
    # returns (model, model_version), or (None, None) if the alias is not set yet
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
    except Exception:
        return None, None
    model = mlflow.sklearn.load_model("models:/" + model_name + "/" + mv.version)
    return model, mv
