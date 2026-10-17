import pandas as pd

from model_select import choose_models


def candidates(lr_pr_auc, lightgbm_pr_auc, graph_pr_auc):
    return pd.DataFrame({
        "version": ["1", "2", "3"],
        "model_type": ["logistic_regression", "lightgbm", "lightgbm"],
        "feature_set": ["base", "base", "base_graph"],
        "valid_pr_auc": [lr_pr_auc, lightgbm_pr_auc, graph_pr_auc],
        "complexity": [0, 1, 2],
    })


def test_simpler_model_wins_when_close():
    # graph model only 3% better on valid (our first training): plain LightGBM is champion,
    # the graph model is challenger, LR is baseline
    assert choose_models(candidates(0.042, 0.065, 0.067)) == ("2", "3", "1")


def test_graph_model_wins_when_clearly_better():
    assert choose_models(candidates(0.042, 0.060, 0.080)) == ("3", "2", "1")


def test_logistic_regression_wins_when_good_enough():
    best, second, baseline = choose_models(candidates(0.066, 0.065, 0.067))
    assert best == "1"
    assert second == "3"
    assert baseline == "1"
