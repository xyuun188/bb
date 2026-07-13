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


def test_no_position_overlay_keeps_position_tiny_and_risk_out_of_direction_vote() -> None:
    decision = _coordinator().combine(_features(), _return_context(), _strong_long_opinions())

    assert decision.action == Action.LONG
    assert "dynamic_expert_weights" not in decision.raw_response

    policy = decision.raw_response["expert_weight_policy"]
    assert policy["mode"] == "market_entry"

    candidate = decision.raw_response["authoritative_return_candidate"]
    assert candidate["production_eligible"] is True
    assert "legacy_expert_vote_permission_enabled" not in candidate


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
