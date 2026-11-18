from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scripts.audit_online_horizon_fold_coverage import REMOTE_SCRIPT, _row_clocks


def test_fold_coverage_projection_is_bounded_and_metadata_only() -> None:
    assert 'transaction_isolation="REPEATABLE READ READ ONLY"' in REMOTE_SCRIPT
    assert "statement_timeout_ms=15000" in REMOTE_SCRIPT
    assert "status='completed'" in REMOTE_SCRIPT
    assert "shadow_entry_result_and_okx_native_1m_path" in REMOTE_SCRIPT
    assert "fee_after_complete" in REMOTE_SCRIPT
    assert "returns_read" in REMOTE_SCRIPT
    assert "exam_labels_evaluated" in REMOTE_SCRIPT
    assert "database_mutated" in REMOTE_SCRIPT
    assert "orders_submitted" in REMOTE_SCRIPT
    assert "long_return_pct" not in REMOTE_SCRIPT
    assert "short_return_pct" not in REMOTE_SCRIPT
    assert "AND created_at >= :since" in REMOTE_SCRIPT
    assert "AND updated_at >= :since" not in REMOTE_SCRIPT
    assert "AND updated_at < :cutoff" in REMOTE_SCRIPT
    assert 'item[horizon] = (decision, available)' in REMOTE_SCRIPT


@pytest.fixture
def clocks():
    since = datetime(2026, 11, 18, tzinfo=UTC)
    cutoff = since + timedelta(hours=24)
    row = {
        "original_decision_at": since.isoformat(),
        "created_at": since + timedelta(seconds=2),
        "due_at": since + timedelta(minutes=5),
        "label_available_at": (since + timedelta(minutes=6)).isoformat(),
    }
    return row, since, cutoff


def test_fold_projection_uses_recorded_availability_after_due_time(clocks):
    row, since, cutoff = clocks
    assert _row_clocks(row, since=since, cutoff=cutoff) == (
        since, since + timedelta(minutes=6),
    )


def test_fold_projection_rejects_repaired_old_decision_and_unavailable_label(clocks):
    row, since, cutoff = clocks
    assert _row_clocks(
        {**row, "original_decision_at": (since - timedelta(seconds=1)).isoformat()},
        since=since, cutoff=cutoff,
    ) is None
    assert _row_clocks(
        {**row, "label_available_at": cutoff.isoformat()}, since=since, cutoff=cutoff,
    ) is None
    assert _row_clocks(
        {**row, "label_available_at": None}, since=since, cutoff=cutoff,
    ) is None
