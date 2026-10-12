import os
import sys
import glob
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# in the container utils/ sits next to api/, when running from the repo it is in scripts/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from utils.model_utils import load_registered_model, load_features, reason_codes

# real-time inference: same champion model (MLflow alias) and same feature store as the batch job,
# so a score from the API matches the score in the morning alert list

DATAMART_DIR = os.getenv("DATAMART_DIR", "datamart/")

app = FastAPI(title="AML account risk API",
              description="Scores bank accounts with the champion model from the MLflow model registry")

# champion model and the snapshots already scored are kept in memory
state = {"model": None, "version": None, "snapshots": {}}


def load_champion():
    model, mv = load_registered_model("champion")
    if model is None:
        raise HTTPException(status_code=503, detail="no champion model in the registry yet, run the pipeline first")
    state["model"] = model
    state["version"] = mv.version
    state["snapshots"] = {}
    print('loaded champion version', mv.version)


def get_model():
    if state["model"] is None:
        load_champion()
    return state["model"]


def latest_snapshot():
    files = sorted(glob.glob(DATAMART_DIR + "gold/feature_store/gold_feature_store_*.parquet"))
    if not files:
        raise HTTPException(status_code=404, detail="feature store is empty, run the pipeline first")
    return files[-1].split("gold_feature_store_")[1].replace(".parquet", "").replace("_", "-")


def get_snapshot(snapshot_date):
    # features + champion scores of one snapshot, scored once then cached
    if snapshot_date not in state["snapshots"]:
        model = get_model()
        features_pdf = load_features(snapshot_date, DATAMART_DIR)
        if features_pdf is None:
            raise HTTPException(status_code=404, detail="no features for snapshot " + snapshot_date)
        features_pdf = features_pdf.set_index("acct")
        features_pdf["score"] = model.predict_proba(features_pdf[list(model.feature_names_in_)])[:, 1]
        state["snapshots"][snapshot_date] = features_pdf
    return state["snapshots"][snapshot_date]


@app.on_event("startup")
def startup():
    try:
        load_champion()
    except Exception as e:
        print('champion not loaded yet:', e)


@app.get("/health")
def health():
    return {"status": "ok", "champion_version": state["version"]}


@app.post("/reload")
def reload():
    # call this after a new version gets the "champion" alias in MLflow, no restart needed
    load_champion()
    return {"status": "reloaded", "champion_version": state["version"]}


@app.get("/accounts/{acct}")
def score_account(acct: str, snapshot_date: str = None):
    # one-off check of an account: look it up in the feature store and score it
    snapshot_date = snapshot_date or latest_snapshot()
    snapshot_pdf = get_snapshot(snapshot_date)
    if acct not in snapshot_pdf.index:
        raise HTTPException(status_code=404, detail="account " + acct + " has no payments in the 4 days before " + snapshot_date)

    model = get_model()
    row = snapshot_pdf.loc[[acct]]
    score = float(row["score"].iloc[0])
    reasons = reason_codes(model, row[list(model.feature_names_in_)], top_n=3)[0]

    # is the account already in the alert list of the batch job?
    alert_rank = None
    alert_path = DATAMART_DIR + "gold/alert_list/alert_list_" + snapshot_date.replace("-", "_") + ".csv"
    if os.path.exists(alert_path):
        alert_pdf = pd.read_csv(alert_path, dtype={"acct": str})
        match = alert_pdf[alert_pdf["acct"] == acct]
        if len(match) > 0:
            alert_rank = int(match["rank"].iloc[0])

    return {
        "acct": acct,
        "snapshot_date": snapshot_date,
        "model_version": state["version"],
        "score": round(score, 6),
        "risk_percentile": round(float((snapshot_pdf["score"] < score).mean() * 100), 2),
        "alert_rank": alert_rank,
        "reasons": reasons,
    }


class PredictRequest(BaseModel):
    features: dict


@app.post("/predict")
def predict(request: PredictRequest):
    # score features sent by another system, missing features count as 0
    model = get_model()
    feature_cols = list(model.feature_names_in_)
    unknown = [f for f in request.features if f not in feature_cols]
    if unknown:
        raise HTTPException(status_code=422, detail="unknown features: " + ", ".join(unknown))

    X = pd.DataFrame([{c: float(request.features.get(c, 0)) for c in feature_cols}])
    score = float(model.predict_proba(X)[:, 1][0])
    return {"model_version": state["version"], "score": round(score, 6), "reasons": reason_codes(model, X, top_n=3)[0]}


@app.get("/alerts")
def alerts(snapshot_date: str = None, top: int = 20):
    # the morning alert list from the batch job, for the case management tool / dashboard
    snapshot_date = snapshot_date or latest_snapshot()
    alert_path = DATAMART_DIR + "gold/alert_list/alert_list_" + snapshot_date.replace("-", "_") + ".csv"
    if not os.path.exists(alert_path):
        raise HTTPException(status_code=404, detail="no alert list for snapshot " + snapshot_date)
    alert_pdf = pd.read_csv(alert_path, dtype={"acct": str}).head(top).fillna("")
    return {"snapshot_date": snapshot_date, "alerts": alert_pdf.to_dict(orient="records")}
