from __future__ import annotations

import pytest

from scripts.audit_online_position_exit_queue import (
    REMOTE_SCRIPT,
    _render_remote_script,
)


def test_remote_position_exit_audit_is_bounded_and_read_only() -> None:
    script = _render_remote_script(hours=12, limit=250)

    assert "AIDecision.analysis_type == " in script
    assert "AIDecision.created_at >= since" in script
    assert "select(func.count(AIDecision.id))" in script
    assert "LIMIT = 250" in script
    assert "HOURS = 12" in script
    assert "get_read_session_ctx" in script
    assert 'transaction_isolation="REPEATABLE READ READ ONLY"' in script
    assert '"total_matching_count"' in script
    assert '"truncated"' in script
    assert "UPDATE " not in script
    assert "DELETE " not in script


def test_remote_position_exit_audit_reports_queue_and_execution_reasons() -> None:
    assert '"budget_deferred"' in REMOTE_SCRIPT
    assert '"close_not_executed"' in REMOTE_SCRIPT
    assert 'item["category"] == "model_hold"' in REMOTE_SCRIPT
    assert '"eligible_model_hold_count"' in REMOTE_SCRIPT
    assert '"dynamic_exit_execution_contract"' in REMOTE_SCRIPT
    assert '"read_only": True' in REMOTE_SCRIPT


@pytest.mark.parametrize(
    ("status", "quantity", "receipt", "expected"),
    [
        ("cancelled", 0, {"profit_lock_ioc_unfilled": True}, "profit_limit_unfilled"),
        ("cancelled", 0, {"native_close_cancelled_no_fill": True}, "native_close_cancelled_no_fill"),
        ("open", 0, {"exit_singleflight_wait": True, "singleflight_state": "native_close_cancelled_no_fill"},
         "native_close_cancelled_cooldown"),
        ("rejected", 0, {"profit_lock_ioc_unfilled": True}, "close_failed"),
        ("open", 0, {"exit_singleflight_wait": True, "singleflight_state": "profit_lock_no_fill"},
         "profit_limit_cooldown"),
        ("rejected", 0, {"execution_blocker": "profit_lock_execution_liquidity",
                         "system_pre_submit_rejection": True},
         "profit_limit_liquidity_deferred"),
        ("open", 0, {"exit_singleflight_wait": True, "singleflight_state": "profit_lock_liquidity_wait"},
         "profit_limit_liquidity_cooldown"),
        ("open", 0, {"exit_singleflight_wait": True}, "close_pending_confirmation"),
        ("pending", 0, {"execution_transport_unknown": True}, "close_result_unknown"),
        ("rejected", 0, {"execution_transport_unknown": True}, "close_result_unknown"),
    ],
)
def test_exit_audit_does_not_count_unfilled_and_unknown_as_failures(
    status, quantity, receipt, expected,
) -> None:
    namespace = {}
    script = _render_remote_script(hours=1, limit=10)
    exec(script.rsplit("\nasyncio.run(main())", 1)[0], namespace)  # noqa: S102
    classify = namespace["close_category"]
    execution = {"status": status, "quantity": quantity, "raw_response": receipt}

    assert classify({}, execution, {}) == expected
    assert classify(
        {"local_order_persistence": {"status": "failed"}}, execution, {},
    ) == "close_failed"
