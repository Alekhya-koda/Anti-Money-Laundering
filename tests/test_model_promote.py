import pytest

from model_promote import promotion_plan


def test_promote_shadow_after_retrain():
    # after the retrain v5 and v6 run in shadow: promoting v6 makes v5 challenger and removes the shadow aliases
    aliases = {"champion": "2", "shadow_1": "5", "shadow_2": "6", "baseline": "4"}
    assert promotion_plan(aliases, "6", set()) == ("promotion", {"champion": "6", "challenger": "5"}, ["shadow_1", "shadow_2"])


def test_promote_only_challenger():
    # before any retrain only the challenger runs in shadow, after promoting it there is no challenger left
    aliases = {"champion": "2", "challenger": "3", "baseline": "1"}
    assert promotion_plan(aliases, "3", set()) == ("promotion", {"champion": "3"}, ["challenger"])


def test_rollback_to_old_champion():
    aliases = {"champion": "6", "challenger": "5", "baseline": "4"}
    assert promotion_plan(aliases, "2", {"2"}) == ("rollback", {"champion": "2"}, [])


def test_refuse_other_versions():
    aliases = {"champion": "6", "challenger": "5", "baseline": "4"}
    with pytest.raises(ValueError):
        promotion_plan(aliases, "4", {"2"})   # baseline, never champion
    with pytest.raises(ValueError):
        promotion_plan(aliases, "6", {"2"})   # already champion
