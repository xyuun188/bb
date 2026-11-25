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


def test_training_guard_uses_net_pnl_across_symbols_instead_of_majority_vote():
    losses = [-6.0, -5.0, -4.5, -3.0, -2.67485666]
    wins = [1.8, 1.8, 1.8, 1.8, 1.8]
    rows = [_row(i, pnl=pnl) for i, pnl in enumerate(losses + wins)]
    result = _feedback(rows)
    assert result["net_usdt"] == pytest.approx(-12.17485666)
    assert result["losing_symbols"] == 5
    assert result["symbol_count"] == 10
    assert result["blocked"] is True
    assert result["reason"] == "cross_symbol_training_loss"
    assert _feedback([*rows, _row(10, pnl=1.0)])["blocked"] is True
    assert _feedback([*rows, _row(10, pnl=15.0)])["blocked"] is False
    assert _feedback([_row(0, pnl=-50), *[_row(i, pnl=1) for i in range(1, 4)]])["blocked"] is False


def test_server_settlements_do_not_contaminate_the_local_model_cohort():
    rows = [_row(i, family="local_ai_tools") for i in range(4)]
    assert _feedback(rows, family="local_ai_tools")["blocked"]
    assert not _feedback(rows)["blocked"]
    assert not _feedback(rows, family="local_ai_tools", version="model2")["blocked"]


def test_new_registry_versions_cannot_clear_recent_losses_in_the_same_training_lane():
    earlier = "20261120T060000000000Z-11111111"
    intermediate = "20261120T080000000000Z-22222222"
    current = "20261120T100000000000Z-33333333"
    future = "20261120T120000000000Z-44444444"
    rows = [
        *[_row(i, version=earlier, family="local_ai_tools") for i in range(4)],
        _row(5, version=intermediate, family="local_ai_tools", pnl=4.0),
        *[_row(i, version=future, family="local_ai_tools", pnl=4.0) for i in range(6, 10)],
    ]
    result = _feedback(rows, family="local_ai_tools", version=current)
    assert result["blocked"]
    assert result["reason"] == "previous_version_cross_symbol_training_loss"
    assert result["previous_losing_cohort"]["cohort"] == f"local_ai_tools|{earlier}|long|240"
    assert "sample_count" not in result
    for changes in ({"family": "local_ml"}, {"side": "short"}, {"horizon": 30}):
        params = {"family": "local_ai_tools", "version": current} | changes
        assert not _feedback(rows, **params)["blocked"]
    assert not training_feedback(
        build_training_profiles(rows, now=NOW + timedelta(days=1)),
        family="local_ai_tools", version=current, side="long", horizon=240,
    )["blocked"]

    current_wins = [_row(i, version=current, family="local_ai_tools", pnl=2.0) for i in range(10, 14)]
    assert _feedback(rows + current_wins[:3], family="local_ai_tools", version=current)["blocked"]
    assert not _feedback(rows + current_wins, family="local_ai_tools", version=current)["blocked"]
    intermediate_wins = [
        _row(i, version=intermediate, family="local_ai_tools", pnl=2.0) for i in range(14, 17)
    ]
    assert not _feedback(
        rows + intermediate_wins, family="local_ai_tools", version=current,
    )["blocked"]


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
