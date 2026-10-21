import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import audit_development_evidence as audit


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
