from datetime import UTC, datetime, timedelta

import pytest

from services.paper_training_settlements import (
    _settled_model_versions,
    build_training_profiles,
    training_feedback,
)

NOW = datetime(2026, 11, 20, 18, tzinfo=UTC)


def _row(i, **changes):
    return {
        "id": i, "symbol": f"SYM{i}/USDT", "side": "long", "version": "model1",
        "horizon": 240, "closed": NOW - timedelta(minutes=30), "pnl": -2.0,
        **changes,
    }


def _feedback(rows, **kwargs):
    params = {"side": "long", "version": "model1", "horizon": 240}
    return training_feedback(build_training_profiles(rows, now=NOW), **(params | kwargs))


def test_correlated_training_losses_follow_the_model_across_symbols_not_other_cohorts():
    rows = [_row(i) for i in range(4)]
    for changes in ({"side": "short"}, {"version": "model2"}, {"horizon": 30}):
        assert not _feedback(rows, **changes)["blocked"]
    assert _feedback(rows)["blocked"]
    repeated = [_row(i, symbol="BTC/USDT") for i in range(8)]
    assert not _feedback(repeated)["blocked"]
    assert not _feedback(rows[:3] + [rows[0]])["blocked"]


def test_training_guard_recovers_and_does_not_use_future_or_expired_settlements():
    rows = [_row(i) for i in range(3)]
    for closed in (NOW + timedelta(seconds=1), NOW - timedelta(days=1, seconds=1)):
        assert not _feedback([*rows, _row(4, closed=closed)])["blocked"]
    assert not _feedback([*rows, _row(4), _row(5, pnl=9.0)])["blocked"]
    assert not _feedback([*rows, _row(4), *[_row(i, pnl=0.1) for i in range(5, 9)]])["blocked"]


def test_server_settlements_do_not_contaminate_the_local_model_cohort():
    rows = [_row(i, family="local_ai_tools") for i in range(4)]
    assert _feedback(rows, family="local_ai_tools")["blocked"]
    assert not _feedback(rows)["blocked"]
    assert not _feedback(rows, family="local_ai_tools", version="model2")["blocked"]


def test_settlement_uses_frozen_family_version_before_legacy_route_metadata():
    families = ["local_ai_tools"]
    assert _settled_model_versions(families, {"local_ai_tools": "v2"}, "ml", "v1", "v1") == {
        "local_ai_tools": "v2",
    }
    assert _settled_model_versions(families, None, "ml", "v1", "v1") == {"local_ai_tools": "v1"}
    assert _settled_model_versions(families, None, "ml", "v1", "v2") == {}
    assert _settled_model_versions(families, {"local_ai_tools": None}, "ml", "v1", "v1") == {}


@pytest.mark.parametrize("profiles", [None, {"status": "incomplete"}])
def test_missing_settlements_are_not_a_successful_empty_query(profiles):
    assert training_feedback(profiles, version="model1", side="long", horizon=240)["blocked"]
    assert not _feedback([])["blocked"]
