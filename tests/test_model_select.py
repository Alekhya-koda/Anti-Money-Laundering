import pandas as pd

from model_select import choose_models, comparison_table


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


def test_comparison_table_roles_after_retrain():
    # retrain: new models 4 (LR), 5 (LightGBM), 6 (graph); the old champion 2 is added as an extra row
    new_models = pd.DataFrame({
        "version": ["4", "5", "6"],
        "model_type": ["logistic_regression", "lightgbm", "lightgbm"],
        "feature_set": ["base", "base", "base_graph"],
        "valid_pr_auc": [0.043, 0.073, 0.070],
        "valid_p_at_500": [0.07, 0.11, 0.13],
    })
    champion_row = {"version": "2", "model_type": "lightgbm", "feature_set": "base", "valid_pr_auc": 0.068, "valid_p_at_500": 0.11}
    table = comparison_table(new_models, "2", "5", "4", champion_row)
    assert list(table["version"]) == ["2", "5", "4", "6"]
    assert list(table["role"]) == ["champion", "challenger", "baseline", "not used"]
