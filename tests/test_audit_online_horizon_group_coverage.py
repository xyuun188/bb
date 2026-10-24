from __future__ import annotations

from scripts.audit_online_horizon_group_coverage import REMOTE_SCRIPT


def test_horizon_coverage_audit_is_read_only_and_uses_native_fee_after_rows() -> None:
    assert 'transaction_isolation="REPEATABLE READ READ ONLY"' in REMOTE_SCRIPT
    assert "statement_timeout_ms=10000" in REMOTE_SCRIPT
    assert "label_version='2026-07-27.market-opportunity-multitask.v2'" in REMOTE_SCRIPT
    assert "status='completed'" in REMOTE_SCRIPT
    assert "source='shadow_entry_result_and_okx_native_1m_path'" in REMOTE_SCRIPT
    assert "fee_after_complete='true'" in REMOTE_SCRIPT
    assert "all_four_rows_observed" in REMOTE_SCRIPT
    assert "all_four_rows_qualified" in REMOTE_SCRIPT
    assert "LIMIT 8" in REMOTE_SCRIPT
