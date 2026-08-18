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
