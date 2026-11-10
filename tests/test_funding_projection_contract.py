from services.execution_cost_model import funding_cost_estimate


def _snapshot() -> dict:
    return {
        "timestamp": "2026-08-17T00:00:00+00:00",
        "funding_data_available": True,
        "funding_rate": 0.001,
        "funding_interval_minutes": 480,
        "next_funding_time": "2026-08-17T01:00:00+00:00",
        "funding_rate_observed_at": "2026-08-17T00:00:00+00:00",
    }


def test_projection_counts_only_settlements_inside_horizon() -> None:
    before = funding_cost_estimate(_snapshot(), side="long", horizon_minutes=30)
    after = funding_cost_estimate(_snapshot(), side="long", horizon_minutes=600)

    assert before.production_eligible is True
    assert before.estimated_settlement_count == 0
    assert before.signed_cashflow_pct == 0.0
    assert after.estimated_settlement_count == 2
    assert after.signed_cashflow_pct == -0.2
    assert after.adverse_cost_pct == 0.2


def test_projection_uses_positive_income_negative_cost_for_each_side() -> None:
    long = funding_cost_estimate(_snapshot(), side="long", horizon_minutes=600)
    short = funding_cost_estimate(_snapshot(), side="short", horizon_minutes=600)

    assert long.signed_cashflow_pct == -0.2
    assert short.signed_cashflow_pct == 0.2
    assert short.adverse_cost_pct == 0.0


def test_projection_accepts_okx_native_epoch_millisecond_times() -> None:
    snapshot = _snapshot()
    snapshot["next_funding_time"] = "1786928400000"
    snapshot["funding_rate_observed_at"] = "1786924800000"

    result = funding_cost_estimate(snapshot, side="long", horizon_minutes=30)

    assert result.production_eligible is True
    assert result.estimated_settlement_count == 0
    assert result.reason == "current_direction_funding_cashflow_ready"


def test_production_snapshot_without_next_time_or_with_stale_rate_is_unavailable() -> None:
    missing_next = _snapshot()
    missing_next.pop("next_funding_time")
    stale = _snapshot()
    stale["timestamp"] = "2026-08-18T00:01:00+00:00"

    missing = funding_cost_estimate(missing_next, side="long", horizon_minutes=30)
    stale_result = funding_cost_estimate(stale, side="long", horizon_minutes=30)

    assert missing.production_eligible is False
    assert missing.reason == "next_funding_time_missing"
    assert stale_result.production_eligible is False
    assert stale_result.reason == "funding_rate_stale"


def test_production_snapshot_without_funding_rate_source_time_is_unavailable() -> None:
    snapshot = _snapshot()
    snapshot.pop("funding_rate_observed_at")

    result = funding_cost_estimate(snapshot, side="long", horizon_minutes=30)

    assert result.production_eligible is False
    assert result.reason == "funding_rate_observed_at_missing"


def test_projection_uses_decision_clock_and_excludes_entry_includes_exit_boundary() -> None:
    snapshot = _snapshot()
    snapshot["original_decision_timestamp"] = snapshot["timestamp"]
    snapshot["timestamp"] = "2026-08-18T00:00:00+00:00"
    exact_exit = funding_cost_estimate(snapshot, side="long", horizon_minutes=60)
    assert exact_exit.production_eligible is True
    assert exact_exit.estimated_settlement_count == 1
    snapshot["next_funding_time"] = snapshot["original_decision_timestamp"]
    at_entry = funding_cost_estimate(snapshot, side="long", horizon_minutes=60)
    assert at_entry.estimated_settlement_count == 0


def test_projection_does_not_use_future_funding_observation_or_invalid_clock() -> None:
    snapshot = _snapshot()
    snapshot["funding_rate_observed_at"] = "2026-08-17T00:00:01+00:00"
    result = funding_cost_estimate(snapshot, side="long", horizon_minutes=60)
    assert result.production_eligible is False
    assert result.reason == "funding_rate_observed_after_decision"
    snapshot["timestamp"] = "invalid"
    result = funding_cost_estimate(snapshot, side="long", horizon_minutes=60)
    assert result.production_eligible is False
    assert result.reason == "funding_as_of_invalid"


def test_projection_bounds_settlement_count_without_iterating_from_old_clock() -> None:
    snapshot = _snapshot()
    snapshot["next_funding_time"] = "1970-01-01T00:00:00+00:00"
    ordinary = funding_cost_estimate(snapshot, side="long", horizon_minutes=480)
    assert ordinary.production_eligible is True
    assert ordinary.estimated_settlement_count == 1
    snapshot["funding_interval_minutes"] = 0.001
    excessive = funding_cost_estimate(snapshot, side="long", horizon_minutes=480)
    assert excessive.production_eligible is False
    assert excessive.reason == "funding_settlement_count_exceeds_limit"
    assert excessive.signed_cashflow_pct is None
