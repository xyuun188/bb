from __future__ import annotations

import json
from datetime import UTC, datetime

from services.phase3_stage_handoff import (
    evaluate_phase3_stage_handoff_inputs,
)

NOW = datetime.now(UTC).isoformat()


def _go_no_go(
    *,
    status: str = "paper_resume_ready",
    can_start: bool = True,
    can_canary: bool = False,
    blockers: list[dict] | None = None,
) -> dict:
    return {
        "available": True,
        "status": status,
        "checked_at": NOW,
        "go_no_go": {
            "status": status,
            "next_step": "resume_paper_pending_operator_approval"
            if status == "paper_resume_ready"
            else "operator_review_for_canary",
            "can_start_paper_with_operator_approval": can_start,
            "can_enter_canary_with_operator_approval": can_canary,
            "can_enter_live": False,
            "blockers": blockers or [],
            "inputs": {"promotion_canary_ready": can_canary},
        },
    }


def _observation(*, status: str = "waiting_for_resume", paper_active: bool = False) -> dict:
    return {
        "available": True,
        "status": status,
        "checked_at": NOW,
        "paper_active": paper_active,
        "can_use_for_promotion": status == "healthy",
        "blockers": [],
        "warnings": [],
    }


def _specialist(*, promotion_ready_count: int = 1) -> dict:
    return {
        "available": True,
        "generated_at": NOW,
        "eligible_shadow_count": 0,
        "summary": {
            "promotion_ready_count": promotion_ready_count,
            "blocked_count": 0 if promotion_ready_count else 1,
            "top_blocked_reasons": [
                {"reason": "false_signal_loss_exceeds_floor", "count": 1}
            ]
            if not promotion_ready_count
            else [],
        },
        "models": [
            {
                "model": "timesfm_shadow_challenger",
                "tool": "time_series_prediction",
                "promotion_ready": bool(promotion_ready_count),
                "promotion_blockers": []
                if promotion_ready_count
                else ["false_signal_loss_exceeds_floor"],
                "tail_loss_count": 12 if not promotion_ready_count else 0,
                "worst_realized_return_pct": -8.7 if not promotion_ready_count else 0,
                "tail_loss_symbols": (
                    [{"symbol": "ACT/USDT", "count": 7}]
                    if not promotion_ready_count
                    else []
                ),
                "worst_samples": (
                    [
                        {
                            "shadow_backtest_id": 1996,
                            "symbol": "ACT/USDT",
                            "predicted_side": "short",
                            "actual_best_side": "long",
                            "actual_return_pct": -8.7,
                        }
                    ]
                    if not promotion_ready_count
                    else []
                ),
            }
        ],
        "live_mutation": False,
    }


def _rebuild(*, promotion_canary_ready: bool | None = None) -> dict:
    report = {"available": True, "status": "blocked", "checked_at": NOW}
    if promotion_canary_ready is not None:
        report["promotion_recommendation"] = {
            "canary_ready": promotion_canary_ready,
            "recommended_stage": "canary" if promotion_canary_ready else "shadow",
            "canary_blocking_reasons": []
            if promotion_canary_ready
            else ["time_series_prediction:chronos-2-shadow-primary_false_signal_loss_exceeds_floor"],
        }
    return report


def _okx_daily() -> dict:
    return {
        "available": True,
        "status": "ok",
        "generated_at": NOW,
        "issue_ledger_summary": {"unresolved": 0, "observing": 0, "fixed": 4},
    }


def _evaluate(**overrides):
    payload = {
        "go_no_go_report": _go_no_go(),
        "observation_report": _observation(),
        "specialist_shadow_report": _specialist(),
        "rebuild_preflight_report": _rebuild(),
        "okx_daily_report": _okx_daily(),
        "report_max_age_seconds": 24 * 3600,
    }
    payload.update(overrides)
    return evaluate_phase3_stage_handoff_inputs(**payload)


def test_phase3_stage_handoff_waits_for_operator_paper_start() -> None:
    report = _evaluate()

    assert report["status"] == "paper_start_ready"
    assert report["stage"] == "paper_start_pending_operator_approval"
    assert report["can_start_paper_with_operator_approval"] is True
    assert report["starts_trading_service"] is False
    assert report["can_enter_live"] is False


def test_phase3_stage_handoff_observes_after_paper_starts() -> None:
    report = _evaluate(
        go_no_go_report=_go_no_go(status="paper_resume_ready", can_start=False),
        observation_report=_observation(status="warming_up", paper_active=True),
    )

    assert report["status"] == "post_resume_observing"
    assert report["stage"] == "post_resume_observation_window"
    assert report["can_start_paper_with_operator_approval"] is False
    assert report["can_enter_canary_with_operator_approval"] is False


def test_phase3_stage_handoff_allows_canary_review_only_after_healthy_observation() -> None:
    report = _evaluate(
        go_no_go_report=_go_no_go(
            status="paper_observation_healthy",
            can_start=False,
            can_canary=True,
        ),
        observation_report=_observation(status="healthy", paper_active=True),
    )

    assert report["status"] == "canary_review_ready"
    assert report["stage"] == "operator_review_for_canary"
    assert report["can_enter_canary_with_operator_approval"] is True
    assert report["can_enter_live"] is False
    assert "specialist_shadow_has_promotion_ready_model" in report["passed_checks"]


def test_phase3_stage_handoff_blocks_canary_when_specialist_models_are_not_ready() -> None:
    report = _evaluate(
        go_no_go_report=_go_no_go(
            status="paper_observation_healthy",
            can_start=False,
            can_canary=True,
        ),
        observation_report=_observation(status="healthy", paper_active=True),
        specialist_shadow_report=_specialist(promotion_ready_count=0),
    )

    assert report["status"] == "paper_observation_healthy"
    assert report["stage"] == "stay_shadow_improve_specialists"
    assert report["can_enter_canary_with_operator_approval"] is False
    assert "specialist_shadow_no_promotion_ready_model" not in {
        item["code"] for item in report["blockers"]
    }
    assert "specialist_shadow_no_promotion_ready_model" in {
        item["code"] for item in report["warnings"]
    }
    warning = next(
        item
        for item in report["warnings"]
        if item["code"] == "specialist_shadow_no_promotion_ready_model"
    )
    assert warning["evidence"]["tail_risk_models"][0]["model"] == "timesfm_shadow_challenger"
    assert warning["evidence"]["tail_risk_models"][0]["tail_loss_symbols"][0]["symbol"] == "ACT/USDT"
    assert warning["evidence"]["tail_risk_models"][0]["worst_samples"][0]["shadow_backtest_id"] == 1996
    assert report["inputs"]["specialist_promotion_ready_count"] == 0
    assert report["inputs"]["specialist_canary_blocked"] is True
    assert report["inputs"]["specialist_tail_loss_total"] == 12


def test_phase3_stage_handoff_uses_rebuild_promotion_gate_over_stale_go_no_go_input() -> None:
    report = _evaluate(
        go_no_go_report=_go_no_go(
            status="paper_observation_healthy",
            can_start=False,
            can_canary=True,
        ),
        observation_report=_observation(status="healthy", paper_active=True),
        specialist_shadow_report=_specialist(promotion_ready_count=1),
        rebuild_preflight_report=_rebuild(promotion_canary_ready=False),
    )

    assert report["status"] == "paper_observation_healthy"
    assert report["can_enter_canary_with_operator_approval"] is False
    assert report["inputs"]["promotion_canary_ready"] is False
    assert "model_promotion_canary_not_ready" in {
        item["code"] for item in report["warnings"]
    }


def test_phase3_stage_handoff_keeps_paper_start_available_while_specialists_learn() -> None:
    report = _evaluate(specialist_shadow_report=_specialist(promotion_ready_count=0))

    assert report["status"] == "paper_start_ready"
    assert report["can_start_paper_with_operator_approval"] is True
    assert report["can_enter_canary_with_operator_approval"] is False
    assert "specialist_shadow_no_promotion_ready_model" not in {
        item["code"] for item in report["blockers"]
    }
    assert "specialist_shadow_no_promotion_ready_model" in {
        item["code"] for item in report["warnings"]
    }


def test_phase3_stage_handoff_blocks_when_go_no_go_has_blockers() -> None:
    report = _evaluate(
        go_no_go_report=_go_no_go(
            status="blocked",
            can_start=False,
            blockers=[{"code": "okx_difference"}],
        )
    )

    assert report["status"] == "blocked"
    assert report["stage"] == "fix_hard_blockers"
    assert "go_no_go_has_blockers" in {item["code"] for item in report["blockers"]}


def test_phase3_stage_handoff_report_is_json_serializable() -> None:
    report = _evaluate()
    json.dumps(report, ensure_ascii=False)
