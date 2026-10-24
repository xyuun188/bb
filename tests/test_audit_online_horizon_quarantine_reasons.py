from __future__ import annotations

from scripts.audit_online_horizon_quarantine_reasons import REMOTE_SCRIPT


def test_horizon_quarantine_audit_is_bounded_read_only() -> None:
    assert "REPEATABLE READ READ ONLY" in REMOTE_SCRIPT
    assert "statement_timeout_ms=10000" in REMOTE_SCRIPT
    assert "LIMIT :row_limit" in REMOTE_SCRIPT
    assert "database_mutated" in REMOTE_SCRIPT
    assert "orders_submitted" in REMOTE_SCRIPT
    assert "NATIVE_SOURCE" in REMOTE_SCRIPT
    assert "_group_horizon_state" in REMOTE_SCRIPT
    assert "market_fact_reason_codes" in REMOTE_SCRIPT
    assert "native_market_fact_invalid_samples" in REMOTE_SCRIPT
    assert "ORDER BY created_at DESC, id DESC" in REMOTE_SCRIPT
