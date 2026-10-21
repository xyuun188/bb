import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

from scripts import audit_development_timing as audit


@pytest.fixture
def row():
    completed = datetime(2026, 9, 1, 12, 0, 10, tzinfo=UTC)
    timing = {
        "stage": "market",
        "analysis_started_at": (completed - timedelta(seconds=4)).isoformat(),
        "decision_completed_at": completed.isoformat(),
    }
    observed = (completed - timedelta(seconds=8)).isoformat()
    return {
        "id": 1, "decision_id": 7, "horizon_minutes": 5,
        "created_at": completed + timedelta(seconds=2),
        "original_created_at": completed + timedelta(milliseconds=10),
        "original_timing": timing, "compact_timing": dict(timing),
        "feature_at": observed, "original_feature_at": observed,
        "due_at": completed + timedelta(minutes=5, seconds=1),
        "market_source": int((completed - timedelta(seconds=9)).timestamp() * 1000),
        "market_received": observed,
    }


def test_original_clock_recovered_without_promoting_missing_component_evidence(row):
    result = audit.inspect_timing(row)
    assert result["recorded_original_clock_recovered"] is True
    assert result["feature_construction_before_completion"] is True
    assert result["offset_seconds"]["stored_creation_minus_completion_seconds"] == 2
    assert result["offset_seconds"]["scheduled_horizon_minus_recorded_horizon_seconds"] == 1
    assert result["component_clocks"]["market_source"] == "no_later_than_completion"
    assert result["component_clocks"]["book_received"] == "missing"
    assert result["component_clocks"]["technical_and_sentiment_inputs"] == (
        "individual_availability_not_recorded"
    )
    assert result["g2_row_accepted"] is False


@pytest.mark.parametrize("value", [
    1788264010000, "1788264010000", 1788264010, "1788264010",
    "2026-09-01T12:00:10+00:00",
])
def test_numeric_string_and_iso_clocks_share_runtime_timestamp_semantics(value):
    assert audit._clock(value).isoformat() == "2026-09-01T12:00:10+00:00"


@pytest.mark.parametrize("value", [None, True, False, "", "nan", "inf", "not-a-clock"])
def test_invalid_clock_does_not_create_evidence(value):
    assert audit._clock(value) is None


def test_post_decision_fee_overwrite_is_distinct_from_equal_numeric_rate(row):
    row["fee_observed"] = row["feature_at"]
    row["shadow_fee_observed"] = row["due_at"].isoformat()
    row["original_taker_fee"] = row["shadow_taker_fee"] = 0.0005
    row["contract_observation_window"] = json.dumps({
        "start": row["feature_at"], "end": row["due_at"].isoformat(),
    })
    result = audit.inspect_timing(row)
    assert "shadow_cost_observation_after_completion:fee_observed" in result["issues"]
    assert "shadow_original_cost_clock_mismatch:fee_observed" in result["issues"]
    assert "shadow_original_taker_fee_value_mismatch" not in result["issues"]
    assert result["offset_seconds"]["contract_entry_observed_minus_completion_seconds"] == -8
    assert result["offset_seconds"]["contract_result_observed_minus_completion_seconds"] == 301


def test_schedule_and_compact_copy_cannot_replace_missing_original_clock(row):
    row["original_timing"] = None
    result = audit.inspect_timing(row)
    assert result["recorded_original_clock_recovered"] is False
    assert "original_decision_clock_missing" in result["issues"]
    assert result["offset_seconds"] == {}


@pytest.mark.parametrize("change,issue", [
    ("reverse", "original_decision_clock_reversed"),
    ("insert", "completion_after_original_insertion"),
    ("compact", "original_compact_timing_conflict:decision_completed_at"),
    ("stage", "original_decision_stage_unverified"),
])
def test_clock_conflicts_cannot_be_accepted(row, change, issue):
    if change == "reverse":
        row["original_timing"]["analysis_started_at"] = (
            row["created_at"] + timedelta(seconds=1)
        ).isoformat()
    elif change == "insert":
        row["original_created_at"] -= timedelta(seconds=2)
    elif change == "compact":
        row["compact_timing"]["decision_completed_at"] = row["created_at"].isoformat()
    else:
        row["original_timing"]["stage"] = "position"
    result = audit.inspect_timing(row)
    assert result["recorded_original_clock_recovered"] is False
    assert issue in result["issues"]


def test_feature_and_component_timing_are_checked_against_original_completion(row):
    row["feature_at"] = row["created_at"]
    row["market_received"] = row["created_at"].isoformat()
    result = audit.inspect_timing(row)
    assert result["recorded_original_clock_recovered"] is True
    assert result["feature_construction_before_completion"] is False
    assert "feature_constructed_after_decision_completed" in result["issues"]
    assert "shadow_original_feature_clock_mismatch" in result["issues"]
    assert "component_after_completion:market_received" in result["issues"]


def test_historical_rows_and_replaced_cost_clocks_remain_explicit(row):
    row["historical"] = True
    row["shadow_fee_observed"] = row["feature_at"]
    row["result_source"] = int((row["due_at"] - timedelta(seconds=30)).timestamp() * 1000)
    result = audit.inspect_timing(row)
    assert "historical_bar_close_availability_unverified" in result["issues"]
    assert "shadow_original_cost_clock_mismatch:fee_observed" in result["issues"]
    assert "result_source_precedes_scheduled_label" in result["issues"]
    assert result["g2_row_accepted"] is False


def test_projection_omits_raw_payloads_and_label_values():
    compiled = " ".join(str(column.compile(dialect=postgresql.dialect()))
                        for column in audit.timing_columns())
    assert "long_return_pct" not in compiled
    assert "short_return_pct" not in compiled
    for column in audit.timing_columns():
        assert str(column) not in {
            "AIDecision.raw_llm_response", "AIDecision.feature_snapshot",
            "ShadowBacktest.feature_snapshot",
        }


def test_online_execution_is_bounded_cached_and_identity_verified(monkeypatch):
    from core import remote_ssh

    calls = []
    identity = audit.code_identity()
    monkeypatch.setattr(remote_ssh, "connect_remote_ssh", lambda *_a, **_kw: SimpleNamespace(
        close=lambda: calls.append("closed"),
    ))
    monkeypatch.setattr(remote_ssh, "run_remote_text", lambda _ssh, command, **_kw: (
        calls.append(command) or json.dumps({
            "audit_code_identity": identity, "source_row_budget": 10000,
        })
    ))
    report = audit.audit_online(max_rows=10000)
    assert "--property=MemoryMax=1G" in calls[0]
    assert "--property=RuntimeMaxSec=300" in calls[0]
    assert "timing audit already active" in calls[0]
    assert identity["sha256"][:12] in report["remote_unit"]
    assert calls[-1] == "closed"
    monkeypatch.setattr(remote_ssh, "run_remote_text", lambda *_a, **_kw: "{}")
    with pytest.raises(ValueError, match="code or budget mismatch"):
        audit.audit_online(max_rows=10000)
