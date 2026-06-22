from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from services.model_expert_competition import summarize_model_expert_competition


def _decision(
    *,
    decision_id: int,
    action: str,
    pnl: float | None,
    hours_ago: float,
    executed: bool = True,
    is_paper: bool = True,
    position_size_pct: float = 0.05,
    raw: dict | None = None,
) -> SimpleNamespace:
    now = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    return SimpleNamespace(
        id=decision_id,
        action=action,
        outcome_pnl_pct=pnl,
        was_executed=executed,
        is_paper=is_paper,
        position_size_pct=position_size_pct,
        raw_llm_response=raw or {},
        created_at=now - timedelta(hours=hours_ago),
    )


def test_competition_report_compares_components_to_baseline_without_live_weight_mutation() -> None:
    now = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    decisions = [
        _decision(
            decision_id=1,
            action="long",
            pnl=0.8,
            hours_ago=1,
            raw={
                "model_timings": [
                    {"name": "trend_expert", "status": "completed", "duration_sec": 1.2},
                    {"name": "risk_expert", "status": "completed", "duration_sec": 4.0},
                ],
                "experts": [
                    {"expert_name": "trend_expert", "action": "long", "confidence": 0.72},
                    {"expert_name": "risk_expert", "action": "hold", "confidence": 0.65},
                ],
            },
        ),
        _decision(
            decision_id=2,
            action="long",
            pnl=0.5,
            hours_ago=2,
            raw={
                "model_timings": [
                    {"name": "trend_expert", "status": "completed", "duration_sec": 1.3},
                    {"name": "risk_expert", "status": "completed", "duration_sec": 4.5},
                ],
                "experts": [
                    {"expert_name": "trend_expert", "action": "long", "confidence": 0.69},
                    {"expert_name": "risk_expert", "action": "hold", "confidence": 0.61},
                ],
            },
        ),
        _decision(
            decision_id=3,
            action="short",
            pnl=-1.0,
            hours_ago=3,
            raw={
                "model_timings": [
                    {"name": "trend_expert", "status": "completed", "duration_sec": 1.1},
                    {"name": "risk_expert", "status": "failed", "duration_sec": 18.0},
                ],
                "experts": [
                    {"expert_name": "trend_expert", "action": "hold", "confidence": 0.55},
                ],
            },
        ),
    ]

    report = summarize_model_expert_competition(decisions, [], now=now)

    assert report["audit_only"] is True
    assert report["live_weight_mutation"] is False
    assert report["can_apply_live_weight"] is False
    assert report["layers"]["offline_replay"]["baseline_available"] is True
    assert report["layers"]["sim_ab"]["available"] is False
    baseline = report["baseline"]
    assert baseline["sample_count"] == 3
    assert baseline["net_pnl_pct"] == 0.3
    trend = report["competitors"]["trend_expert"]
    assert trend["baseline_delta"]["net_pnl_pct"] > 0
    assert trend["recommended_weight_action"] == "increase_shadow_weight"
    assert trend["can_apply_live_weight"] is False
    risk = report["competitors"]["risk_expert"]
    assert risk["recommended_weight_action"] in {"reduce_shadow_weight", "pause_shadow"}
    assert "no_direct_live_weight_change" in report["safety_rules"]


def test_competition_report_refuses_actions_without_baseline_samples() -> None:
    report = summarize_model_expert_competition(
        [], [], now=datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    )

    assert report["baseline"]["sample_count"] == 0
    assert report["layers"]["offline_replay"]["baseline_available"] is False
    assert report["can_apply_live_weight"] is False
    assert report["competitors"] == {}
    assert "baseline_missing" in report["blocking_reasons"]
