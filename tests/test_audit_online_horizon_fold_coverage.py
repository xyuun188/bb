from __future__ import annotations

from scripts.audit_online_horizon_fold_coverage import REMOTE_SCRIPT


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
