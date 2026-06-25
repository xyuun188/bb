from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from services.strategy_signal_root_cause_audit import StrategySignalRootCauseAuditService


def _entry_decision(symbol: str, *, decision_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=decision_id,
        symbol=symbol,
        action="short",
        analysis_type="market",
        raw_llm_response={
            "analysis_type": "market",
            "opportunity_score": {
                "score": 0.45,
                "min_score_required": 0.95,
                "expected_net_return_pct": 0.22,
                "profit_quality_ratio": 0.18,
                "server_profit_loss_probability": 0.64,
                "tail_risk_score": 0.91,
                "server_profit_expected_return_pct": -0.12,
                "expected_net_breakdown": {
                    "components": [
                        {
                            "key": "ai",
                            "available": True,
                            "contribution_pct": 0.15,
                        },
                        {
                            "key": "local_ml",
                            "available": False,
                            "contribution_pct": 0.0,
                            "blocked_reasons": ["readiness_degraded"],
                        },
                        {
                            "key": "server_profit",
                            "available": True,
                            "contribution_pct": -0.024,
                        },
                        {
                            "key": "shadow_memory",
                            "available": True,
                            "contribution_pct": 0.30,
                        },
                        {"key": "fee", "available": True, "contribution_pct": -0.08},
                    ]
                },
                "evidence_score": {
                    "tier": "weak_conflict_probe",
                    "advisory_wait_reasons": [
                        "动态证据评分低于可交易底线，当前仅保留观望或极小探针"
                    ],
                    "components": [
                        {"source": "ai", "status": "aligned", "points": 8.0},
                        {
                            "source": "ml",
                            "status": "ignored",
                            "reason": "ML 当前处于学习观察中",
                        },
                        {
                            "source": "server_profit",
                            "status": "opposite",
                            "expected_return_pct": -0.12,
                        },
                        {"source": "timeseries", "status": "weak_opposite"},
                        {"source": "shadow_memory", "status": "aligned", "points": 3.0},
                    ],
                    "positive_net_probe_relief": {
                        "applied": True,
                        "shadow_only": True,
                        "tradeable_probe": False,
                    },
                },
            },
        },
        created_at=datetime(2026, 6, 25, tzinfo=UTC),
    )


def test_strategy_signal_root_cause_summarizes_ml_server_profit_and_shadow_blockers() -> None:
    service = StrategySignalRootCauseAuditService(
        now=lambda: datetime(2026, 6, 25, tzinfo=UTC),
        ml_status_provider=lambda: {
            "available": True,
            "status": "degraded",
            "readiness_state": "degraded",
            "allow_live_position_influence": False,
            "readiness": {
                "state": "degraded",
                "allow_live_position_influence": False,
                "blocking_reasons": [{"code": "long_pr_auc_below_threshold"}],
                "metrics": {"long_pr_auc": 0.34, "short_pr_auc": 0.39},
            },
        },
    )
    decisions = [_entry_decision(f"SYM{index % 5}/USDT", decision_id=index) for index in range(25)]
    shadows = [
        SimpleNamespace(
            symbol=f"SYM{index % 4}/USDT",
            missed_opportunity=True,
            best_action="short",
        )
        for index in range(30)
    ]

    report = service.summarize(
        decisions=decisions, shadows=shadows, ml_status=service._ml_status_provider()
    )

    assert report["status"] == "warning"
    assert report["audit_only"] is True
    assert report["can_force_open"] is False
    assert report["can_override_thresholds"] is False
    assert report["can_change_ml_readiness"] is False
    assert report["entry_decision_count"] == 25
    assert report["high_quality_entry_count"] == 0
    assert report["ml"]["usable_rate"] == 0.0
    assert report["server_profit"]["negative_or_opposite_count"] == 25
    assert report["shadow_missed_opportunity"]["missed_count"] == 30
    assert report["expected_net_component_stats"]["server_profit"]["negative_count"] == 25
    assert report["expected_net_component_stats"]["shadow_memory"]["positive_count"] == 25
    codes = {item["code"] for item in report["root_causes"]}
    assert {
        "ml_not_contributing",
        "server_profit_negative_or_opposite",
        "high_quality_entry_gap",
        "candidate_symbol_concentration",
        "shadow_missed_not_convertible",
        "positive_ev_still_below_evidence_quality",
        "weak_evidence_dominates",
    }.issubset(codes)


def test_strategy_signal_root_cause_is_ok_when_signal_chain_has_quality() -> None:
    service = StrategySignalRootCauseAuditService(
        now=lambda: datetime(2026, 6, 25, tzinfo=UTC),
        ml_status_provider=lambda: {"available": True, "status": "ready"},
    )
    decision = _entry_decision("BTC/USDT", decision_id=1)
    opportunity = decision.raw_llm_response["opportunity_score"]
    opportunity["score"] = 1.8
    opportunity["min_score_required"] = 0.95
    opportunity["evidence_score"]["tier"] = "small"
    opportunity["evidence_score"]["components"] = [
        {"source": "ai", "status": "aligned", "points": 12.0},
        {"source": "ml", "status": "aligned", "points": 5.0},
        {"source": "server_profit", "status": "aligned", "points": 5.0},
        {"source": "timeseries", "status": "aligned", "points": 5.0},
    ]
    opportunity["expected_net_breakdown"]["components"] = [
        {"key": "ai", "available": True, "contribution_pct": 0.15},
        {"key": "local_ml", "available": True, "contribution_pct": 0.08},
        {"key": "server_profit", "available": True, "contribution_pct": 0.12},
        {"key": "timeseries", "available": True, "contribution_pct": 0.04},
        {"key": "fee", "available": True, "contribution_pct": -0.04},
    ]

    report = service.summarize(
        decisions=[decision],
        shadows=[SimpleNamespace(symbol="BTC/USDT", missed_opportunity=False, best_action="hold")],
        ml_status={"available": True, "status": "ready", "allow_live_position_influence": True},
    )

    assert report["status"] == "ok"
    assert report["root_causes"] == []
    assert report["high_quality_entry_count"] == 1
