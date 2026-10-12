import numpy as np
import pandas as pd
import lightgbm as lgb

from utils.model_utils import precision_recall_at_k, evaluate_scores, psi, reason_codes


def small_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "out_cnt": rng.poisson(3, 1000),
        "out_uniq_cp": rng.poisson(2, 1000),
        "out_ach_share": rng.random(1000),
    })
    y = (X["out_ach_share"] > 0.8).astype(int)
    return lgb.LGBMClassifier(n_estimators=20, verbose=-1).fit(X, y), X


def test_precision_at_k_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    scores = np.arange(100)[::-1]
    p, r = precision_recall_at_k(y, scores, 10)
    assert p == 1.0
    assert r == 1.0


def test_precision_at_k_ties_not_ranked_by_row_order():
    # every score is the same and the positives are the first rows:
    # if ties kept the row order we would get 100% precision (the bug we had in the EDA baselines)
    y = np.array([1] * 50 + [0] * 950)
    scores = np.zeros(1000)
    p, _ = precision_recall_at_k(y, scores, 50)
    assert p < 0.3


def test_evaluate_scores_has_all_metrics():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 6000)
    results = evaluate_scores(y, rng.random(6000))
    for k in ["pr_auc", "roc_auc", "p_at_500", "r_at_500", "p_at_2000", "p_at_5000", "r_at_5000"]:
        assert k in results


def test_psi_same_and_shifted():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 10000)
    assert psi(reference, rng.normal(0, 1, 10000)) < 0.1
    assert psi(reference, rng.normal(1, 1, 10000)) > 0.25


def test_psi_constant_feature():
    assert psi(np.zeros(100), np.zeros(100)) == 0.0


def test_reason_codes():
    model, X = small_model()
    reasons = reason_codes(model, X.head(5), top_n=3)
    assert len(reasons) == 5
    assert all(len(r) <= 3 for r in reasons)
    assert all(isinstance(text, str) for r in reasons for text in r)
