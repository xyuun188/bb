from datetime import UTC, datetime, timedelta

import pytest

from core.training_contracts import shadow_label_timing
from services.training_data_quality import assess_shadow_sample
from tests.test_training_data_quality import _shadow_sample


@pytest.mark.parametrize("lag,reason", [
    (-1, "result_source_before_schedule"),
    (0, None),
    (60, None),
    (61, "result_source_lag_exceeds_one_minute"),
])
def test_source_event_must_match_the_scheduled_label_window(lag, reason):
    due = datetime(2026, 10, 1, tzinfo=UTC)
    source = due + timedelta(seconds=lag)
    contract = {
        "result_source_timestamp_ms": int(source.timestamp() * 1000),
        "result_received_at": (source + timedelta(seconds=2)).isoformat(),
    }
    timing = shadow_label_timing(contract, due)
    assert timing["source_horizon_lag_seconds"] == lag
    assert timing["label_maturity_lag_seconds"] == lag + 2
    assert timing["reasons"] == ([reason] if reason else [])


def test_new_receipt_cannot_make_an_early_source_event_mature():
    due = datetime(2026, 10, 1, tzinfo=UTC)
    timing = shadow_label_timing({
        "result_source_timestamp_ms": int((due - timedelta(seconds=5)).timestamp() * 1000),
        "result_received_at": (due + timedelta(seconds=3)).isoformat(),
    }, due)
    assert timing["reasons"] == ["result_source_before_schedule"]


def test_native_receipt_clock_must_not_precede_the_scheduled_label():
    due = datetime(2026, 10, 1, tzinfo=UTC)
    timing = shadow_label_timing({
        "result_source_timestamp_ms": int((due + timedelta(seconds=1)).timestamp() * 1000),
        "result_received_at": (due - timedelta(seconds=1)).isoformat(),
    }, due)
    assert set(timing["reasons"]) == {
        "result_received_before_schedule",
        "result_received_before_source",
    }


@pytest.mark.parametrize("defect,reason", [
    ("missing_source", "result_source_time_missing"),
    ("early", "result_source_before_schedule"),
    ("late", "result_source_lag_exceeds_one_minute"),
    ("missing_receipt", "result_availability_time_missing"),
])
def test_shadow_quality_rejects_unverified_and_wrong_horizon_results(defect, reason):
    sample = _shadow_sample()
    market = sample["features"]["training_market_fact_contract"]
    if defect == "missing_source":
        market.pop("result_source_timestamp_ms")
    elif defect == "missing_receipt":
        market.pop("result_received_at")
    elif defect == "early":
        market["result_source_timestamp_ms"] -= 1000
    elif defect == "late":
        market["result_source_timestamp_ms"] += 61_000
    assessment = assess_shadow_sample(sample)
    assert assessment.exclude_from_training
    assert f"shadow_label_timing:{reason}" in assessment.reasons
