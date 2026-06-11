from __future__ import annotations

from datetime import UTC, datetime

from services.analysis_budget import POSITION_REVIEW_FAST_EXIT_SCORE
from services.position_review_fast_scan_hold import PositionReviewFastScanHoldPolicy


def test_fast_scan_hold_plan_records_deferred_exit_signal() -> None:
    policy = PositionReviewFastScanHoldPolicy(
        clock=lambda: datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
    )

    plan = policy.plan(
        {
            "priority_score": 82.25,
            "exit_score": POSITION_REVIEW_FAST_EXIT_SCORE + 1,
            "add_score": 12.5,
            "reason": "profit_lock_candidate",
        },
        previous_defer_count=1,
        urgent_exit=True,
        portfolio_symbol_context={"active": True, "is_focus": True},
        agent_skill_dicts=[{"name": "risk"}],
        agent_skill_summary={"count": 1},
    )

    assert plan.defer_count == 2
    assert "需要复盘的平仓/锁盈信号" in plan.reason
    assert "紧急退出类" in plan.reason
    assert "已连续跳过 2 轮" in plan.reason
    assert "profit_lock_candidate" in plan.reason
    assert "组合利润保护已激活" in plan.reason
    assert plan.raw_response["position_fast_scan"] == {
        "skipped_llm": True,
        "priority_score": 82.25,
        "exit_score": POSITION_REVIEW_FAST_EXIT_SCORE + 1,
        "add_score": 12.5,
        "reason": "profit_lock_candidate",
    }
    assert plan.raw_response["portfolio_profit_protection"] == {
        "active": True,
        "is_focus": True,
    }
    phase = plan.raw_response["agent_skills"]["phases"]["position_fast_scan"]
    assert phase["recorded_at"] == "2026-06-10T12:00:00+00:00"
    assert "平仓/锁盈复盘信号" in phase["note"]
    assert phase["skills"] == [{"name": "risk"}]
    assert plan.raw_response["agent_skills"]["summary"] == {"count": 1}


def test_fast_scan_hold_plan_resets_defer_when_no_exit_signal() -> None:
    plan = PositionReviewFastScanHoldPolicy().plan(
        {"priority_score": 21.0, "exit_score": 0.0, "add_score": 0.0},
        previous_defer_count=3,
        urgent_exit=False,
        portfolio_symbol_context={},
        agent_skill_dicts=[],
        agent_skill_summary={},
    )

    assert plan.defer_count == 0
    assert "未发现必须立即交给慢专家" in plan.reason
    assert "优先级 21.0" in plan.reason
    assert "portfolio_profit_protection" not in plan.raw_response
