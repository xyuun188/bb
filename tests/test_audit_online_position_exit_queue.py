from __future__ import annotations

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
