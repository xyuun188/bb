from __future__ import annotations

from ai_brain.base_model import Action, DecisionOutput
from ai_brain.ensemble_coordinator import EnsembleCoordinator
from ai_brain.model_registry import ModelRegistry
from data_feed.feature_vector import FeatureVector

SYMBOL = "BTC/USDT"


def _coordinator() -> EnsembleCoordinator:
    return EnsembleCoordinator(ModelRegistry())


def _features(**kwargs) -> FeatureVector:
    values = {
        "symbol": SYMBOL,
        "volume_ratio": 1.2,
        "adx_14": 24.0,
        "price_vs_sma20": 0.03,
        "price_vs_sma50": 0.05,
        "volatility_20": 0.02,
        "spread_pct": 0.0004,
    }
    values.update(kwargs)
    return FeatureVector(**values)


def _decision(
    model_name: str,
    action: Action,
    *,
    confidence: float = 0.8,
    size: float = 0.08,
    reasoning: str = "test",
    provider_model: str | None = None,
) -> DecisionOutput:
    return DecisionOutput(
        model_name=model_name,
        symbol=SYMBOL,
        action=action,
        confidence=confidence,
        reasoning=reasoning,
        position_size_pct=size if action.is_entry() else 0.0,
        suggested_leverage=3.0,
        stop_loss_pct=0.035,
        take_profit_pct=0.08,
        raw_response=({"provider_model": provider_model} if provider_model else None),
    )


def _strong_long_opinions(
    *,
    risk_action: Action = Action.LONG,
    risk_confidence: float = 0.8,
    risk_reasoning: str = "risk cleared",
) -> dict[str, DecisionOutput]:
    return {
        "trend_expert": _decision("trend_expert", Action.LONG, confidence=0.86),
        "momentum_expert": _decision("momentum_expert", Action.LONG, confidence=0.84),
        "sentiment_expert": _decision("sentiment_expert", Action.LONG, confidence=0.70),
        "position_expert": _decision("position_expert", Action.LONG, confidence=0.90),
        "risk_expert": _decision(
            "risk_expert",
            risk_action,
            confidence=risk_confidence,
            reasoning=risk_reasoning,
        ),
    }


def _return_context(**extra: object) -> dict[str, object]:
    provenance = {
        "source": "test_authoritative_return_distribution",
        "observation_window": "current_test_round",
        "sample_count": 4,
        "generated_at": "2026-07-12T00:00:00+00:00",
        "strategy_version": "test",
        "fallback_reason": "",
    }
    context: dict[str, object] = {
        "entry_candidate_evidence": {
            "preferred_side_by_evidence": "long",
            "long": {
                "production_eligible": True,
                "expected_net_return_pct": 0.6,
                "return_lcb_pct": 0.4,
                "production_source_count": 2,
                "policy_provenance": provenance,
            },
            "short": {
                "production_eligible": False,
                "expected_net_return_pct": -0.2,
                "return_lcb_pct": -0.3,
                "production_source_count": 2,
                "policy_provenance": provenance,
            },
            "policy_provenance": provenance,
        }
    }
    context.update(extra)
    return context


def _paper_exploration_context(execution_mode: str = "paper") -> dict[str, object]:
    provenance = {
        "source": "test_cost_complete_return_distribution",
        "observation_window": "current_test_round",
        "sample_count": 3,
        "generated_at": "2026-07-21T00:00:00+00:00",
        "strategy_version": "test",
        "fallback_reason": "",
    }
    selected = {
        "eligible": True,
        "side": "long",
        "expected_net_return_pct": 0.3,
        "return_lcb_pct": -0.1,
        "lcb_gap_ratio": 1.0 / 3.0,
        "loss_probability": 0.3,
        "tail_risk_score": 0.2,
        "return_source_count": 3,
        "feature_opportunity_score": 8.0,
        "information_value_score": 0.04,
        "policy_provenance": provenance,
    }
    evidence = {
        "preferred_side_by_evidence": "neutral",
        "preferred_exploration_side": "long",
        "feature_opportunity_score": 8.0,
        "long": {
            "production_eligible": False,
            "expected_net_return_pct": 0.3,
            "return_lcb_pct": -0.1,
            "production_source_count": 3,
            "policy_provenance": provenance,
        },
        "paper_exploration": {
            "preferred_side": "long",
            "selected": selected,
            "eligible_side_count": 1,
            "reason": "bounded_paper_exploration_side_selected",
        },
        "policy_provenance": provenance,
    }
    return {
        "execution_mode": execution_mode,
        "entry_candidate_evidence": evidence,
    }


def test_no_position_overlay_keeps_position_tiny_and_risk_out_of_direction_vote() -> None:
    decision = _coordinator().combine(_features(), _return_context(), _strong_long_opinions())

    assert decision.action == Action.LONG
    assert "dynamic_expert_weights" not in decision.raw_response

    policy = decision.raw_response["expert_weight_policy"]
    assert policy["mode"] == "market_entry"

    candidate = decision.raw_response["authoritative_return_candidate"]
    assert candidate["production_eligible"] is True
    assert "legacy_expert_vote_permission_enabled" not in candidate


def test_authoritative_return_candidate_does_not_depend_on_expert_availability() -> None:
    decision = _coordinator().combine(_features(), _return_context(), {})

    assert decision.action == Action.LONG
    assert decision.raw_response["authoritative_return_candidate"]["production_eligible"] is True


def test_positive_mean_uncertain_candidate_can_only_create_bounded_paper_entry() -> None:
    decision = _coordinator().combine(
        _features(),
        _paper_exploration_context("paper"),
        _strong_long_opinions(),
    )

    assert decision.action == Action.LONG
    assert decision.suggested_leverage == 1.0
    contract = decision.raw_response["paper_exploration"]
    assert contract["execution_scope"] == "paper_only"
    assert contract["production_permission"] is False
    assert contract["trade_is_normal"] is True
    assert contract["sample_target"] is None
    assert contract["daily_sample_quota"] is None


def test_paper_exploration_candidate_remains_hold_in_live_mode() -> None:
    decision = _coordinator().combine(
        _features(),
        _paper_exploration_context("live"),
        _strong_long_opinions(),
    )

    assert decision.action == Action.HOLD
    assert "paper_exploration" not in decision.raw_response


def test_no_champion_uses_loss_tolerant_paper_training_direction() -> None:
    context = _return_context(
        execution_mode="paper",
        paper_training_mode="bootstrap",
        paper_strategy_champion={
            "active": False,
            "paper_execution_permission": False,
        },
        direction_competition={
            "preferred_side": "neutral",
            "training_preferred_side": "short",
            "training_short": {
                "score": -0.4,
                "raw_expected_return_pct": -0.1,
                "objective_expected_return_pct": -0.4,
                "horizon_minutes": 10,
                "observation_count": 1,
            },
        },
    )
    context["entry_candidate_evidence"]["preferred_side_by_evidence"] = "neutral"
    context["entry_candidate_evidence"]["preferred_exploration_side"] = "neutral"

    decision = _coordinator().combine(
        _features(),
        context,
        _strong_long_opinions(),
    )

    assert decision.action == Action.SHORT
    assert decision.raw_response["paper_training"]["loss_tolerant_for_training"] is True
    assert decision.raw_response["paper_training"]["production_permission"] is False
    assert decision.raw_response["paper_training"]["expected_net_return_pct"] == -0.4
    assert decision.raw_response["paper_training"]["valid_for_seconds"] == 600.0


def test_paper_training_route_is_never_created_for_live_execution() -> None:
    context = _return_context(
        execution_mode="live",
        paper_training_mode="bootstrap",
        direction_competition={
            "training_preferred_side": "long",
            "training_long": {"score": -1.0, "observation_count": 1},
        },
    )
    context["entry_candidate_evidence"]["preferred_side_by_evidence"] = "neutral"
    context["entry_candidate_evidence"]["preferred_exploration_side"] = "neutral"

    decision = _coordinator().combine(_features(), context, _strong_long_opinions())

    assert decision.action == Action.HOLD
    assert "paper_training" not in decision.raw_response


def test_expert_diversity_policy_is_carried_into_ensemble_raw() -> None:
    context = _return_context(
        _expert_diversity_policy={
            "should_retry": True,
            "reason": "strong entry-candidate evidence requires independent confirmation",
            "objective_evidence": {"side": "long", "score": 3.5},
        }
    )
    opinions = _strong_long_opinions()
    opinions["trend_expert"].raw_response = {
        "independent_expert_retry": True,
        "provider_independent_expert_mode": True,
    }

    decision = _coordinator().combine(_features(), context, opinions)

    assert decision.action == Action.LONG
    assert decision.raw_response["expert_diversity_policy"]["should_retry"] is True
    trend = next(
        item for item in decision.raw_response["opinions"] if item["model_name"] == "trend_expert"
    )
    assert trend["independent_expert_retry"] is True
    assert trend["provider_independent_expert_mode"] is True


def test_risk_expert_text_cannot_grant_or_veto_production_entry() -> None:
    opinions = _strong_long_opinions(
        risk_action=Action.HOLD,
        risk_confidence=0.82,
        risk_reasoning="hard veto: prohibit entry because exchange/liquidity risk is extreme",
    )

    decision = _coordinator().combine(_features(), _return_context(), opinions)

    assert decision.action == Action.LONG
    risk_policy = decision.raw_response["risk_expert_policy"]
    assert risk_policy["hard_veto"] is False
    assert decision.raw_response["authoritative_return_candidate"]["production_eligible"] is True


def test_non_hard_risk_caution_is_observation_only() -> None:
    coordinator = _coordinator()
    features = _features(volatility_20=0.07, spread_pct=0.0035)
    context = _return_context()
    baseline = coordinator.combine(features, context, _strong_long_opinions())
    cautious = coordinator.combine(
        features,
        context,
        _strong_long_opinions(
            risk_action=Action.HOLD,
            risk_confidence=0.84,
            risk_reasoning="liquidity and slippage caution, no hard veto",
        ),
    )

    assert cautious.action == Action.LONG
    risk_policy = cautious.raw_response["risk_expert_policy"]
    assert risk_policy["hard_veto"] is False
    assert "score_discount_pct" not in risk_policy
    assert "size_multiplier" not in risk_policy
    assert cautious.position_size_pct == baseline.position_size_pct == 0.0
    assert cautious.raw_response["authoritative_return_candidate"]["production_eligible"] is True


def test_local_fallback_is_trace_only_and_has_zero_effective_weight() -> None:
    opinions = _strong_long_opinions()
    opinions["trend_expert"].raw_response = {
        "local_fallback": True,
        "production_eligible": False,
    }

    decision = _coordinator().combine(_features(), _return_context(), opinions)

    trend = next(
        item for item in decision.raw_response["opinions"] if item["model_name"] == "trend_expert"
    )
    assert decision.action == Action.LONG
    assert trend["trace_only_fallback"] is True
    assert trend["effective_weight"] == 0.0
    assert trend["weight_policy"]["production_permission"] is False


def test_local_fallback_risk_opinion_cannot_veto_authoritative_return_entry() -> None:
    opinions = _strong_long_opinions(risk_action=Action.SHORT, risk_confidence=0.99)
    opinions["risk_expert"].raw_response = {
        "local_fallback": True,
        "production_eligible": False,
    }

    decision = _coordinator().combine(_features(), _return_context(), opinions)

    risk = next(
        item for item in decision.raw_response["opinions"] if item["model_name"] == "risk_expert"
    )
    assert decision.action == Action.LONG
    assert risk["effective_weight"] == 0.0
    assert decision.raw_response["risk_expert_policy"]["hard_veto"] is False
