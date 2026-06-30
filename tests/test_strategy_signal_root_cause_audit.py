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


def _scheduler_decision(
    symbol: str,
    *,
    decision_id: int,
    action: str = "hold",
    strategy: str = "drawdown_clamp",
    cache_status: str = "baseline_timeout",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=decision_id,
        symbol=symbol,
        action=action,
        analysis_type="market",
        created_at=datetime(2026, 6, 25, tzinfo=UTC),
        raw_llm_response={
            "analysis_type": "market",
            "strategy_mode": {
                "strategy": strategy,
                "posture": "tight_selective",
                "reason": "Daily drawdown is active; allow only higher-quality opportunities.",
                "scheduler_reason": "profile drawdown guard selected tight entries",
                "risk_mode": "hard_recovery",
                "strategy_profile_id": "phase3-profit",
                "strategy_profile_version": "2026-06-26",
                "expert_integrity_mode": "strict",
                "strategy_learning_cache_status": cache_status,
                "strategy_learning_runtime_timeout_seconds": 0.01,
                "strategy_learning_entry_pause": True,
                "strategy_learning_entry_pause_reason": "profile pause after loss cluster",
                "strategy_learning_execution_guard_active": True,
                "strategy_learning_release_pressure_active": True,
                "strategy_learning_health_guard_active": True,
                "soft_avoided_directions": ["short"],
                "market_regime": {
                    "mode": "mixed",
                    "confidence": 0.28,
                },
                "dynamic_position_capacity": {
                    "base_limit": 25,
                    "target_limit": 25,
                    "effective_limit": 18,
                    "entry_limit": 18,
                    "open_group_count": 18,
                    "low_quality_count": 5,
                    "release_candidate_count": 2,
                    "reason": "drawdown and low-quality pressure",
                    "factors": {
                        "reason_codes": [
                            "drawdown",
                            "low_quality_pressure",
                            "release_rotation_slots",
                        ]
                    },
                },
            },
            "strategy_learning_context": {
                "strategy_profile_id": "phase3-profit",
                "strategy_profile_version": "2026-06-26",
                "scheduler_reason": "profile drawdown guard selected tight entries",
                "expert_integrity_mode": "strict",
                "strategy_learning_entry_pause": True,
                "strategy_learning_execution_guard_active": True,
                "strategy_learning_release_pressure_active": True,
                "strategy_learning_health_guard_active": True,
            },
        },
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


def test_strategy_signal_root_cause_summarizes_scheduler_posture_and_capacity() -> None:
    service = StrategySignalRootCauseAuditService(
        now=lambda: datetime(2026, 6, 25, tzinfo=UTC),
        ml_status_provider=lambda: {"available": True, "status": "ready"},
    )
    decisions = [
        _scheduler_decision(f"SYM{index}/USDT", decision_id=index)
        for index in range(3)
    ]

    report = service.summarize(decisions=decisions, shadows=[], ml_status={})
    scheduler = report["scheduler"]

    assert scheduler["read_only"] is True
    assert scheduler["audit_only"] is True
    assert scheduler["can_force_open"] is False
    assert scheduler["can_override_thresholds"] is False
    assert scheduler["can_bypass_risk_controls"] is False
    assert scheduler["sample_count"] == 3
    assert scheduler["strategy_counts"] == {"drawdown_clamp": 3}
    assert scheduler["risk_mode_counts"] == {"hard_recovery": 3}
    assert scheduler["cache_status_counts"] == {"baseline_timeout": 3}
    assert scheduler["flag_counts"]["strategy_learning_context_timeout"] == 3
    assert scheduler["flag_counts"]["strategy_learning_entry_pause_active"] == 3
    assert scheduler["flag_counts"]["drawdown_clamp_active"] == 3
    assert scheduler["flag_counts"]["market_regime_soft_bias_active"] == 3
    assert scheduler["dynamic_capacity"]["constrained_count"] == 3
    assert scheduler["dynamic_capacity"]["entry_blocked_count"] == 3
    assert scheduler["dynamic_capacity"]["reason_code_counts"]["drawdown"] == 3
    assert scheduler["latest_samples"][0]["dynamic_position_capacity"]["constrained"] is True
    assert scheduler["latest_samples"][0]["can_force_open"] is False
    codes = {item["code"] for item in report["root_causes"]}
    assert {
        "no_entry_candidates",
        "strategy_learning_context_timeout",
        "strategy_learning_entry_pause_active",
        "dynamic_capacity_constrained",
        "drawdown_clamp_active",
        "market_regime_soft_bias_active",
    }.issubset(codes)
    actions = " ".join(report["next_actions"])
    assert "dynamic capacity reason codes" in actions
    assert "strategy-learning context latency" in actions


def test_strategy_signal_root_cause_reports_ml_top_return_blocker() -> None:
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
                "blocking_reasons": [
                    {
                        "code": "long_top_return_below_threshold",
                        "actual": -0.05,
                        "required": 0.05,
                    },
                    {
                        "code": "short_top_return_below_threshold",
                        "actual": -0.02,
                        "required": 0.05,
                    },
                ],
                "thresholds": {"min_top_return_pct": 0.05},
                "metrics": {
                    "sample_count": 5900,
                    "test_count": 1400,
                    "top_long_avg_return_pct": -0.05,
                    "bottom_long_avg_return_pct": -0.12,
                    "top_short_avg_return_pct": -0.02,
                    "bottom_short_avg_return_pct": -0.08,
                },
            },
        },
    )

    report = service.summarize(
        decisions=[_entry_decision(f"SYM{index}/USDT", decision_id=index) for index in range(12)],
        shadows=[],
        ml_status=service._ml_status_provider(),
    )

    cause = next(
        item for item in report["root_causes"] if item["code"] == "ml_top_return_not_profitable"
    )
    assert cause["blocking_reason_codes"] == [
        "long_top_return_below_threshold",
        "short_top_return_below_threshold",
    ]
    assert cause["top_long_avg_return_pct"] == -0.05
    assert cause["top_short_avg_return_pct"] == -0.02
    assert cause["required_min_top_return_pct"] == 0.05
    assert report["ml"]["readiness"]["blocking_reasons"][0]["code"] == (
        "long_top_return_below_threshold"
    )
    assert (
        "Keep ML in observation until top-score buckets show positive fee-adjusted returns"
        in " ".join(report["next_actions"])
    )


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
