import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi.testclient import TestClient

import api.app as app_module

# no MLflow in the tests: put a small model straight into the api state


def small_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "out_cnt": rng.poisson(3, 1000).astype(float),
        "out_uniq_cp": rng.poisson(2, 1000).astype(float),
        "out_ach_share": rng.random(1000),
    })
    y = (X["out_ach_share"] > 0.8).astype(int)
    return lgb.LGBMClassifier(n_estimators=20, verbose=-1).fit(X, y)


def setup_module():
    app_module.state["model"] = small_model()
    app_module.state["version"] = "test"
    app_module.state["snapshots"] = {}


def test_health():
    client = TestClient(app_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["champion_version"] == "test"


def test_predict():
    client = TestClient(app_module.app)
    response = client.post("/predict", json={"features": {"out_cnt": 12, "out_uniq_cp": 9, "out_ach_share": 0.95}})
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["score"] <= 1
    assert len(body["reasons"]) <= 3


def test_predict_unknown_feature():
    client = TestClient(app_module.app)
    response = client.post("/predict", json={"features": {"not_a_feature": 1}})
    assert response.status_code == 422


def test_account_lookup(tmp_path):
    # tiny feature store with 2 accounts
    datamart = str(tmp_path) + "/"
    (tmp_path / "gold" / "feature_store").mkdir(parents=True)
    (tmp_path / "gold" / "graph_feature_store").mkdir(parents=True)
    pd.DataFrame({"acct": ["A1", "B2"], "out_cnt": [30.0, 1.0], "out_uniq_cp": [20.0, 1.0],
                  "out_ach_share": [1.0, 0.0], "snapshot_date": ["2022-09-09", "2022-09-09"]}) \
        .to_parquet(datamart + "gold/feature_store/gold_feature_store_2022_09_09.parquet", index=False)
    pd.DataFrame({"acct": ["A1", "B2"], "pagerank": [0.1, 0.2], "snapshot_date": ["2022-09-09", "2022-09-09"]}) \
        .to_parquet(datamart + "gold/graph_feature_store/gold_graph_features_2022_09_09.parquet", index=False)
    app_module.DATAMART_DIR = datamart
    app_module.state["snapshots"] = {}

    client = TestClient(app_module.app)
    response = client.get("/accounts/A1?snapshot_date=2022-09-09")
    assert response.status_code == 200
    assert response.json()["acct"] == "A1"

    response = client.get("/accounts/ZZZ?snapshot_date=2022-09-09")
    assert response.status_code == 404
