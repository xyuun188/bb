from __future__ import annotations

from datetime import UTC, datetime

from services.position_review_fast_scan_hold import PositionReviewFastScanHoldPolicy


def test_fast_scan_hold_records_deferred_governed_exit() -> None:
    policy = PositionReviewFastScanHoldPolicy(
        clock=lambda: datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
    )
    scan = {
        "dynamic_exit_eligible": True,
        "dynamic_exit_policy": {
            "eligible": True,
            "close_fraction": 0.42,
        },
        "reason": "dynamic_exit_policy_passed",
    }

    plan = policy.plan(
        scan,
        previous_defer_count=1,
        urgent_exit=True,
        portfolio_symbol_context={"active": True, "is_focus": True},
        agent_skill_dicts=[{"name": "risk"}],
        agent_skill_summary={"count": 1},
    )

    assert plan.defer_count == 2
    assert "Governed dynamic exit review was deferred" in plan.reason
    assert plan.raw_response["position_fast_scan"] == {
        "skipped_llm": True,
        "dynamic_exit_eligible": True,
        "close_fraction": 0.42,
        "reason": "dynamic_exit_policy_passed",
        "production_permission": False,
    }
    assert plan.raw_response["portfolio_profit_observation"]["production_permission"] is False
    assert plan.raw_response["dynamic_exit_policy"] == scan["dynamic_exit_policy"]
    phase = plan.raw_response["agent_skills"]["phases"]["position_fast_scan"]
    assert phase["recorded_at"] == "2026-06-10T12:00:00+00:00"
    assert phase["skills"] == [{"name": "risk"}]


def test_fast_scan_hold_resets_defer_without_dynamic_exit_contract() -> None:
    plan = PositionReviewFastScanHoldPolicy().plan(
        {"priority_score": 99.0, "exit_score": 99.0},
        previous_defer_count=3,
        urgent_exit=False,
        portfolio_symbol_context={},
        agent_skill_dicts=[],
        agent_skill_summary={},
    )

    assert plan.defer_count == 0
    assert "No governed dynamic exit permission" in plan.reason
    assert plan.raw_response["position_fast_scan"]["production_permission"] is False
    assert "portfolio_profit_observation" not in plan.raw_response
    assert "dynamic_exit_policy" not in plan.raw_response


def test_fast_scan_hold_persists_blocked_exit_evidence() -> None:
    assessment = {
        "eligible": False,
        "close_fraction": 0.0,
        "reason": "profit_lock_target_already_filled",
        "target_close_fraction": 0.3,
        "profit_lock_realized_fraction": 0.6,
    }
    plan = PositionReviewFastScanHoldPolicy().plan(
        {
            "dynamic_exit_eligible": False,
            "dynamic_exit_policy": assessment,
            "reason": assessment["reason"],
        },
        previous_defer_count=3,
        urgent_exit=False,
        portfolio_symbol_context={},
        agent_skill_dicts=[],
        agent_skill_summary={},
    )

    assert plan.defer_count == 0
    assert assessment["reason"] in plan.reason
    assert plan.raw_response["dynamic_exit_policy"] == assessment
    assert plan.raw_response["position_fast_scan"]["production_permission"] is False
    assert plan.raw_response["position_fast_scan"]["dynamic_exit_eligible"] is False
