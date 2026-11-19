from __future__ import annotations

import pytest

from scripts import audit_online_entry_recovery as audit


def test_entry_recovery_audit_skips_large_historical_dry_run_by_default() -> None:
    rendered = audit._render_remote_script(
        include_historical_requalification=False,
    )

    compile(rendered, "<entry-recovery-audit>", "exec")
    assert "INCLUDE_HISTORICAL_REQUALIFICATION = False" in rendered
    assert "max_batches=300" in rendered
    assert "enable with --include-historical-requalification" in rendered
    assert "if INCLUDE_HISTORICAL_REQUALIFICATION:" in rendered


def test_entry_recovery_audit_can_explicitly_include_historical_dry_run() -> None:
    rendered = audit._render_remote_script(
        include_historical_requalification=True,
    )

    compile(rendered, "<entry-recovery-audit>", "exec")
    assert "INCLUDE_HISTORICAL_REQUALIFICATION = True" in rendered
    assert "dry_run=True" in rendered


@pytest.mark.parametrize(
    ("state", "reason", "executed", "expected"),
    [
        ("analysis_only_execution_unverified", None, False, "analysis_only_execution_unverified"),
        ("execution_unavailable", None, False, "execution_instrument_unavailable"),
        (None, "same_symbol_reentry_cooldown_active", False, "same_symbol_reentry_cooldown"),
        ("analysis_only_execution_unverified", None, True, None),
    ],
)
def test_entry_recovery_distinguishes_pre_submit_states(state, reason, executed, expected) -> None:
    namespace = {"__LIMIT__": 8, "__SINCE_MINUTES__": 30}
    rendered = audit._render_remote_script(include_historical_requalification=False)
    exec(rendered.rsplit("asyncio.run(main())", 1)[0], namespace)  # noqa: S102
    fields = ("market_analysis_only_contract", "same_symbol_reentry_guard")
    row = {
        "action": "short",
        "was_executed": executed,
        "execution_reason": reason,
        "market_analysis_only_contract": (
            {"entry_permission": False, "execution_candidate_state": state} if state else None
        ),
    }

    assert namespace["_entry_funnel_reason"](row, fields) == expected


@pytest.mark.parametrize(
    ("stage", "status", "has_order", "expected"),
    [
        ("strategy_arbitration", "completed", False, "entry_queued"),
        ("risk_check", "pending", False, "entry_queued"),
        ("exchange_submit", "pending", False, "entry_result_pending"),
        ("exchange_confirm", "unknown", True, "entry_result_pending"),
        ("risk_check", "blocked", False, "execution_blocked"),
        ("exchange_confirm", "failed", True, "execution_blocked"),
    ],
)
def test_entry_recovery_does_not_count_queued_or_pending_as_blocked(
    stage, status, has_order, expected,
) -> None:
    namespace = {"__LIMIT__": 8, "__SINCE_MINUTES__": 30}
    rendered = audit._render_remote_script(include_historical_requalification=False)
    exec(rendered.rsplit("asyncio.run(main())", 1)[0], namespace)  # noqa: S102
    row = {
        "action": "short",
        "was_executed": False,
        "has_order": has_order,
        "execution_reason": "正在提交 OKX：开仓裁决已移交执行队列",
        "decision_state_summary": {"final_stage": stage, "final_status": status},
    }

    assert namespace["_entry_funnel_reason"](row, ()) == expected
