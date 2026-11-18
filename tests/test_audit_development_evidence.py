import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import audit_development_evidence as audit
from services.ml_development_snapshot import load_development_snapshot


@pytest.fixture
def historical_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "frozen_development_v1.json"
    monkeypatch.setattr(audit, "SNAPSHOT", path)
    records = [{
        "id": 7, "decision_group": "decision:seven", "horizon_minutes": 5,
        "decision_timestamp": "2026-08-01T00:00:00+00:00",
        "label_timestamp": "2026-08-01T00:05:00.123456+00:00",
        "decision_confidence": 0.9, "spread_pct": 0.1, "sample_weight": 1.0,
        **dict.fromkeys(audit.ECONOMICS_COLUMNS, 0.0),
    }]
    legacy_input = [{
        **records[0],
        "decision_timestamp": "2026-08-01 00:00:00+00:00",
        "label_timestamp": "2026-08-01 00:05:00.123456+00:00",
    }]
    payload = {
        "version": "2026-10-21.frozen-development.v1",
        "development_cutoff": audit.CUTOFF,
        "row_count": 1,
        "decision_group_count": 1,
        "development_input_sha256": hashlib.sha256(
            json.dumps(legacy_input, sort_keys=True).encode()
        ).hexdigest(),
        "records": records,
    }

    def write():
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        path.write_text(json.dumps({**payload, "snapshot_sha256": digest}), encoding="utf-8")
        return digest

    monkeypatch.setattr(audit, "EXPECTED_SNAPSHOT_SHA256", write())
    return path, payload, write


def test_v1_snapshot_is_read_only_and_not_a_current_training_input(historical_snapshot):
    path, _, _ = historical_snapshot
    original = path.read_bytes()

    frame, snapshot = audit.load_historical_snapshot()

    assert frame["id"].tolist() == [7]
    assert frame.iloc[0]["label_timestamp"] == pd.Timestamp("2026-08-01T00:05:00.123456Z")
    comparison = audit.inspect_snapshot(frame, snapshot)["numeric_comparison"]
    assert comparison["frozen_only_columns"] == ["decision_confidence"]
    assert "technical_indicator_timeframe_minutes" in comparison["current_only_columns"]
    assert "spread_pct" in comparison["compared_columns"]
    with pytest.raises(ValueError, match="snapshot integrity failure"):
        load_development_snapshot(path, cutoff=audit.CUTOFF)
    assert path.read_bytes() == original


def test_mutated_v1_snapshot_cannot_replace_pinned_evidence(historical_snapshot):
    _, payload, write = historical_snapshot
    payload["records"][0]["sample_weight"] = 2.0
    write()
    with pytest.raises(ValueError, match="snapshot integrity failure"):
        audit.load_historical_snapshot()


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("development_input_sha256", "wrong", "frame fingerprint mismatch"),
        ("development_cutoff", "2026-09-27T00:00:00Z", "cutoff cannot change"),
        ("row_count", 2, "boundary or accounting mismatch"),
    ],
)
def test_v1_payload_integrity_does_not_bypass_other_checks(
    historical_snapshot, monkeypatch, field, value, error,
):
    _, payload, write = historical_snapshot
    payload[field] = value
    monkeypatch.setattr(audit, "EXPECTED_SNAPSHOT_SHA256", write())
    with pytest.raises(ValueError, match=error):
        audit.load_historical_snapshot()


@pytest.fixture
def source(monkeypatch):
    created = datetime(2026, 8, 1, tzinfo=UTC)
    expected = {
        **{column: 0.1 for column in audit.COMPARE_NUMERIC},
        "id": 1, "decision_group": "decision:one", "horizon_minutes": 5,
        "decision_timestamp": created, "label_timestamp": created + timedelta(minutes=5),
    }
    row = SimpleNamespace(
        id=1, created_at=created, due_at=expected["label_timestamp"],
        horizon_minutes=5, feature_snapshot={
            "timestamp": created.isoformat(),
            "training_label_contract": {
                "shadow_backtest_id": 1, "horizon_minutes": 5,
                "label_timestamp": expected["label_timestamp"].isoformat(),
                "label_fingerprint": "frozen-label",
            },
        },
    )
    monkeypatch.setattr(audit, "shadow_quality_sample", lambda _row: {})
    monkeypatch.setattr(audit, "assess_shadow_sample", lambda _sample: SimpleNamespace(
        exclude_from_training=False,
    ))
    monkeypatch.setattr(audit, "build_training_frame", lambda _rows: pd.DataFrame([expected]))
    return pd.DataFrame([deepcopy(expected)]), row, expected


def test_source_matches_exact_frozen_ids_without_reestimating_weights(source):
    frozen, row, _ = source
    issues, identities, ages, delays, decision_anchors = audit.inspect_source_batch(
        frozen, [row]
    )
    assert issues == {"decision_time_provenance_unresolved": 1}
    assert identities == [(1, "frozen-label")]
    assert ages == [0]
    assert delays == [0]
    assert decision_anchors[1] == pd.Timestamp(row.created_at)


def test_missing_source_cannot_be_replaced_by_unrelated_new_row(source):
    frozen, row, _ = source
    row.id = 999
    issues, identities, _, _, _ = audit.inspect_source_batch(frozen, [row])
    assert issues["source_row_missing"] == 1
    assert identities == []


def test_feature_after_schedule_anchor_does_not_prove_future_data_leakage(source):
    frozen, row, _ = source
    row.feature_snapshot["timestamp"] = (row.created_at + timedelta(minutes=1)).isoformat()
    issues, _, ages, delays, anchors = audit.inspect_source_batch(frozen, [row])
    assert not issues.get("feature_observed_after_decision")
    assert issues["decision_time_provenance_unresolved"] == 1
    assert ages == [-60]
    assert delays == [0]
    assert anchors[1] == pd.Timestamp(row.created_at)


def test_missing_feature_time_and_wrong_horizon_are_explicit(source):
    frozen, row, _ = source
    row.feature_snapshot.pop("timestamp")
    row.due_at += timedelta(minutes=1)
    issues, _, horizons, _, _ = audit.inspect_source_batch(frozen, [row])
    assert issues["feature_observation_time_missing"] == 1
    assert issues["source_label_time_mismatch"] == 1
    assert horizons == []


def test_due_minus_horizon_is_only_a_schedule_anchor(source):
    frozen, row, _ = source
    row.created_at += timedelta(seconds=30)
    row.due_at += timedelta(seconds=60)
    row.feature_snapshot["training_label_contract"]["label_timestamp"] = (
        row.due_at.isoformat()
    )
    issues, _, ages, delays, anchors = audit.inspect_source_batch(frozen, [row])
    assert issues["source_decision_time_mismatch"] == 1
    assert issues["source_label_time_mismatch"] == 1
    assert not issues.get("label_contract_timestamp_mismatch")
    assert ages == [60]
    assert delays == [-30]
    assert anchors[1] == pd.Timestamp(row.due_at - timedelta(minutes=5))
    assert issues["decision_time_provenance_unresolved"] == 1


def test_modified_feature_cost_and_label_are_not_silently_replaced(source):
    frozen, row, rebuilt = source
    rebuilt["spread_pct"] = 2
    rebuilt["long_return_pct"] = 3
    rebuilt["short_execution_cost_pct"] = 4
    issues, _, _, _, _ = audit.inspect_source_batch(frozen, [row])
    assert issues["source_value_mismatch:spread_pct"] == 1
    assert issues["source_value_mismatch:long_return_pct"] == 1
    assert issues["source_value_mismatch:short_execution_cost_pct"] == 1
    assert frozen.iloc[0]["long_return_pct"] == 0.1


def test_old_feature_set_does_not_impute_new_columns(source):
    frozen, row, rebuilt = source
    new_column = "technical_indicator_timeframe_minutes"
    frozen = frozen.drop(columns=new_column)
    rebuilt[new_column] = 5
    rebuilt["long_execution_cost_pct"] = 2
    issues, _, _, _, _ = audit.inspect_source_batch(frozen, [row])
    assert f"source_value_mismatch:{new_column}" not in issues
    assert issues["source_value_mismatch:long_execution_cost_pct"] == 1


def test_excluded_or_no_longer_cost_complete_source_blocks_research(source, monkeypatch):
    frozen, row, _ = source
    monkeypatch.setattr(audit, "assess_shadow_sample", lambda _sample: SimpleNamespace(
        exclude_from_training=True, reasons=["bad_lineage"],
    ))
    monkeypatch.setattr(audit, "build_training_frame", lambda _rows: pd.DataFrame())
    issues, _, _, _, _ = audit.inspect_source_batch(frozen, [row])
    assert issues["source_quality_rejected"] == 1
    assert issues["quality:bad_lineage"] == 1
    assert issues["source_not_in_current_cost_complete_frame"] == 1


def test_duplicate_source_ids_fail(source):
    frozen, row, _ = source
    with pytest.raises(ValueError, match="duplicate source identity"):
        audit.inspect_source_batch(frozen, [row, row])


def test_online_audit_is_named_bounded_and_checks_code_identity(monkeypatch):
    from core import remote_ssh

    calls = []
    source_hash = hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest()
    code_identity = audit.audit_code_identity()
    ssh = SimpleNamespace(close=lambda: calls.append("closed"))
    monkeypatch.setattr(remote_ssh, "connect_remote_ssh", lambda *_args, **_kw: ssh)
    monkeypatch.setattr(
        remote_ssh, "run_remote_text",
        lambda _ssh, command, **_kw: calls.append(command) or json.dumps({
            "audit_source_sha256": source_hash, "source_row_budget": 10000,
            "audit_code_identity": code_identity,
        }),
    )
    report = audit.audit_online(max_rows=10000)
    assert f"bb-frozen-v1-source-r10000-{code_identity['sha256'][:12]}" == report["remote_unit"]
    assert "--property=MemoryMax=1G" in calls[0]
    assert "--property=RuntimeMaxSec=300" in calls[0]
    assert "--setenv=DATABASE_POOL_SIZE=1" in calls[0]
    assert "audit already active" in calls[0]
    assert calls[-1] == "closed"
    monkeypatch.setattr(remote_ssh, "run_remote_text", lambda *_args, **_kw: "{}")
    with pytest.raises(ValueError, match="code or budget mismatch"):
        audit.audit_online(max_rows=10000)


def test_dependency_changes_invalidate_cached_report_identity(monkeypatch):
    before = audit.audit_code_identity()
    original = Path.read_bytes

    def changed(path):
        content = original(path)
        if path.as_posix().endswith("services/profit_supervision.py"):
            return content + b"\n# changed dependency\n"
        return content

    monkeypatch.setattr(Path, "read_bytes", changed)
    after = audit.audit_code_identity()
    assert before["sha256"] != after["sha256"]
    assert before["dependency_sha256"]["services/profit_supervision.py"] != (
        after["dependency_sha256"]["services/profit_supervision.py"]
    )
