from __future__ import annotations

from ai_brain.ensemble_coordinator import EnsembleCoordinator
from ai_brain.model_registry import ModelRegistry
from data_feed.feature_vector import FeatureVector
from services.model_dynamic_routing import plan_dynamic_model_route, summarize_dynamic_model_routing


def test_dynamic_route_preserves_all_experts_without_baseline() -> None:
    route = plan_dynamic_model_route(
        FeatureVector(symbol="BTC/USDT", sentiment_data_available=False),
        {
            "candidate_quality": "low",
            "ml_signal": {"readiness": {"allow_live_position_influence": False}},
        },
        model_health={
            "components": {
                "sentiment_expert": {"recommended_state": "shadow_only"},
                "position_expert": {"recommended_state": "shadow_only"},
            }
        },
        competition={"blocking_reasons": ["baseline_missing"], "can_apply_live_weight": False},
        feature_coverage={"status": "warning", "missing_features": ["event_calendar"]},
    )

    assert route["mode"] == "shadow_only"
    assert route["applied_to_live_calls"] is False
    assert route["live_route_mutation"] is False
    assert route["can_apply_live_route"] is False
    assert route["selected_experts"] == [
        "trend_expert",
        "momentum_expert",
        "sentiment_expert",
        "position_expert",
        "risk_expert",
    ]
    assert route["skipped_experts"] == []
    assert "competition_baseline_missing" in route["blocking_reasons"]
    assert "ml_readiness_blocks_live_route" in route["blocking_reasons"]
    assert route["canary_ready"] is False
    assert route["live_ready"] is False
    assert "walk_forward_required" in route["live_blocking_reasons"]
    assert "live_mutation_not_enabled" in route["live_blocking_reasons"]
    assert route["estimated_call_reduction"] == 0
    assert route["expert_reasons"]["risk_expert"] == [
        "full_governed_expert_set_preserved"
    ]


def test_dynamic_route_keeps_sentiment_for_news_events_even_if_health_is_weak() -> None:
    features = FeatureVector(
        symbol="ETH/USDT",
        sentiment_data_available=True,
        direct_sentiment_data_available=True,
        news_article_count=3,
        direct_news_item_count=2,
        recent_news_items=[{"source": "okx_announcements", "event_type": "listing"}],
    )

    route = plan_dynamic_model_route(
        features,
        {"candidate_quality": "high"},
        model_health={"components": {"sentiment_expert": {"recommended_state": "shadow_only"}}},
        competition={"baseline": {"sample_count": 12}, "blocking_reasons": []},
        feature_coverage={"status": "ok", "missing_features": []},
    )

    assert "sentiment_expert" in route["selected_experts"]
    assert route["expert_reasons"]["sentiment_expert"] == [
        "full_governed_expert_set_preserved"
    ]
    assert "sentiment_expert" not in route["skipped_experts"]
    assert route["applied_to_live_calls"] is False
    assert route["mode"] == "canary_ready"
    assert route["canary_ready"] is True
    assert route["live_ready"] is False
    assert route["live_blocking_reasons"] == [
        "model_stage_not_live",
        "walk_forward_required",
        "live_mutation_not_enabled",
    ]


def test_dynamic_route_does_not_use_fixed_market_thresholds_to_change_experts() -> None:
    features = FeatureVector(
        symbol="SOL/USDT",
        abnormal_wick_count_72h=1,
        abnormal_wick_max_pct=9.5,
        volatility_20=0.09,
    )

    route = plan_dynamic_model_route(
        features,
        {"candidate_quality": "low", "market_risk_level": "high"},
        model_health={"components": {"risk_expert": {"recommended_state": "reduce"}}},
        competition={"baseline": {"sample_count": 20}, "blocking_reasons": []},
        feature_coverage={"status": "ok", "missing_features": []},
    )

    assert "risk_expert" in route["selected_experts"]
    assert route["expert_reasons"]["risk_expert"] == [
        "full_governed_expert_set_preserved"
    ]
    assert route["mandatory_safety_experts"] == ["risk_expert"]


def test_dynamic_route_live_requested_requires_walk_forward_and_explicit_enablement() -> None:
    features = FeatureVector(symbol="BTC/USDT")

    blocked = plan_dynamic_model_route(
        features,
        {"candidate_quality": "high"},
        competition={"baseline": {"sample_count": 20}, "blocking_reasons": []},
        feature_coverage={"status": "ok", "missing_features": []},
        requested_stage="live",
        training_governance={
            "training_mode": "shadow",
            "model_stage": "canary",
            "evaluation_policy": {
                "promotion_flow": "shadow_to_canary_to_live",
                "live_mutation": False,
                "requires_walk_forward": True,
            },
        },
    )

    assert blocked["mode"] == "live_blocked"
    assert blocked["canary_ready"] is True
    assert blocked["live_ready"] is False
    assert blocked["blocking_reasons"] == [
        "model_stage_not_live",
        "walk_forward_required",
        "live_mutation_not_enabled",
    ]

    ready = plan_dynamic_model_route(
        features,
        {"candidate_quality": "high"},
        competition={"baseline": {"sample_count": 20}, "blocking_reasons": []},
        feature_coverage={"status": "ok", "missing_features": []},
        requested_stage="live",
        training_governance={
            "training_mode": "walk_forward",
            "model_stage": "live",
            "evaluation_policy": {
                "promotion_flow": "shadow_to_canary_to_live",
                "live_mutation": True,
                "requires_walk_forward": True,
            },
        },
    )

    assert ready["mode"] == "live_ready"
    assert ready["live_ready"] is True
    assert ready["blocking_reasons"] == []
    assert ready["live_route_mutation"] is False
    assert ready["can_apply_live_route"] is False


def test_ensemble_coordinator_attaches_shadow_routing_without_live_mutation() -> None:
    coordinator = EnsembleCoordinator(ModelRegistry())
    raw: dict[str, object] = {}

    coordinator._attach_dynamic_model_routing(
        FeatureVector(symbol="BTC/USDT"),
        {
            "candidate_quality": "low",
            "model_expert_competition": {
                "blocking_reasons": ["baseline_missing"],
                "can_apply_live_weight": False,
            },
            "crypto_feature_coverage": {
                "status": "warning",
                "missing_features": ["event_calendar"],
            },
        },
        raw,
    )

    route = raw["dynamic_model_routing"]
    assert isinstance(route, dict)
    assert route["mode"] == "shadow_only"
    assert route["applied_to_live_calls"] is False
    assert route["live_route_mutation"] is False
    assert "competition_baseline_missing" in route["blocking_reasons"]


def test_dynamic_routing_report_summarizes_shadow_routes_and_safety_observations() -> None:
    decisions = [
        {
            "symbol": "BTC/USDT",
            "action": "long",
            "was_executed": False,
            "raw_llm_response": {
                "dynamic_model_routing": {
                    "mode": "shadow_only",
                    "selected_experts": ["trend_expert", "momentum_expert", "risk_expert"],
                    "skipped_experts": ["sentiment_expert", "position_expert"],
                    "estimated_call_reduction": 2,
                    "blocking_reasons": ["competition_baseline_missing"],
                    "live_blocking_reasons": [
                        "competition_baseline_missing",
                        "model_stage_not_live",
                    ],
                    "applied_to_live_calls": False,
                    "live_route_mutation": False,
                },
            },
        },
        {
            "symbol": "ETH/USDT",
            "action": "short",
            "was_executed": True,
            "outcome_pnl_pct": -0.4,
            "raw_llm_response": {
                "dynamic_model_routing": {
                    "mode": "shadow_only",
                    "selected_experts": ["trend_expert", "momentum_expert", "risk_expert"],
                    "skipped_experts": ["sentiment_expert", "position_expert"],
                    "estimated_call_reduction": 2,
                    "blocking_reasons": ["feature_coverage_missing"],
                    "live_ready": False,
                    "live_blocking_reasons": ["feature_coverage_missing"],
                    "applied_to_live_calls": True,
                    "live_route_mutation": True,
                },
                "production_return_policy": {
                    "eligible": False,
                    "return_lcb_pct": -0.2,
                },
            },
        },
    ]

    report = summarize_dynamic_model_routing(decisions)

    assert report["audit_only"] is True
    assert report["live_route_mutation"] is False
    assert report["can_apply_live_route"] is False
    assert report["summary"]["route_plan_count"] == 2
    assert report["summary"]["shadow_only_count"] == 2
    assert report["summary"]["estimated_call_reduction"] == 4
    assert report["summary"]["unsafe_live_mutation_attempts"] == 1
    assert report["summary"]["live_ready_count"] == 0
    assert report["summary"]["live_blocked_count"] == 2
    assert (
        report["safety_observations"]["ineligible_return_contract_executed_count"] == 1
    )
    assert report["blocking_reason_counts"]["competition_baseline_missing"] == 1
    assert report["blocking_reason_counts"]["feature_coverage_missing"] == 1
