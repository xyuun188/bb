from __future__ import annotations

import pytest

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


def test_no_position_overlay_keeps_position_tiny_and_risk_out_of_direction_vote() -> None:
    decision = _coordinator().combine(_features(), {}, _strong_long_opinions())

    assert decision.action == Action.LONG
    weights = decision.raw_response["dynamic_expert_weights"]
    assert weights["trend_expert"]["effective_weight"] == pytest.approx(0.33)
    assert weights["momentum_expert"]["effective_weight"] == pytest.approx(0.33)
    assert weights["sentiment_expert"]["effective_weight"] == pytest.approx(0.14)
    assert weights["position_expert"]["effective_weight"] == pytest.approx(0.05)
    assert weights["risk_expert"]["effective_weight"] == 0.0

    policy = decision.raw_response["expert_weight_policy"]
    assert policy["mode"] == "no_position_entry_overlay"
    assert "position_expert" in policy["entry_support_excluded_experts"]
    assert "risk_expert" in policy["entry_support_excluded_experts"]

    support = decision.raw_response["entry_signal_support"]
    assert "position_expert" in support["excluded_direction_experts"]
    assert "risk_expert" in support["excluded_direction_experts"]
    assert "position_expert" not in support["directional_support_experts"]
    assert "risk_expert" not in support["directional_support_experts"]


def test_risk_expert_hard_veto_blocks_new_entry_even_without_direction_weight() -> None:
    opinions = _strong_long_opinions(
        risk_action=Action.HOLD,
        risk_confidence=0.82,
        risk_reasoning="hard veto: prohibit entry because exchange/liquidity risk is extreme",
    )

    decision = _coordinator().combine(_features(), {}, opinions)

    assert decision.action == Action.HOLD
    risk_policy = decision.raw_response["risk_expert_policy"]
    assert risk_policy["hard_veto"] is True
    assert risk_policy["score_discount_pct"] == 1.0
    assert decision.raw_response["dynamic_expert_weights"]["risk_expert"]["effective_weight"] == 0.0


def test_non_hard_risk_caution_discounts_score_and_size_without_becoming_support() -> None:
    coordinator = _coordinator()
    features = _features(volatility_20=0.07, spread_pct=0.0035)
    baseline = coordinator.combine(features, {}, _strong_long_opinions())
    cautious = coordinator.combine(
        features,
        {},
        _strong_long_opinions(
            risk_action=Action.HOLD,
            risk_confidence=0.84,
            risk_reasoning="liquidity and slippage caution, no hard veto",
        ),
    )

    assert cautious.action == Action.LONG
    risk_policy = cautious.raw_response["risk_expert_policy"]
    assert risk_policy["hard_veto"] is False
    assert risk_policy["score_discount_pct"] > 0.0
    assert risk_policy["score_after_discount"] < risk_policy["score_before_discount"]
    assert cautious.position_size_pct < baseline.position_size_pct
    assert cautious.raw_response["risk_expert_size_discount"]["applied"] is True


def test_risk_expert_same_direction_does_not_unlock_entry_support_gate() -> None:
    coordinator = _coordinator()
    opinions = {
        "trend_expert": _decision("trend_expert", Action.LONG, confidence=0.70),
        "momentum_expert": _decision("momentum_expert", Action.HOLD, confidence=0.80),
        "sentiment_expert": _decision("sentiment_expert", Action.HOLD, confidence=0.80),
        "risk_expert": _decision("risk_expert", Action.LONG, confidence=0.95),
    }

    allowed = coordinator._entry_signal_allowed(
        Action.LONG,
        opinions,
        [{"consistency": "aligned"}, {"consistency": "aligned"}],
        validation_adjustment=0.20,
        disagreement=0.0,
        context={},
    )

    assert allowed is False


def test_quant_only_probe_does_not_hard_block_long_when_timeseries_opposes() -> None:
    result = _coordinator()._quant_only_probe_evidence(
        {},
        {
            "local_ai_tools": {
                "profit_prediction": {
                    "available": True,
                    "best_side": "long",
                    "adjusted_long_return_pct": 0.35,
                    "adjusted_short_return_pct": -0.10,
                    "long_loss_probability": 0.42,
                },
                "time_series_prediction": {
                    "available": True,
                    "best_side": "short",
                    "expected_return_pct": 0.20,
                    "confidence": 0.20,
                },
                "sentiment_analysis": {
                    "available": True,
                    "best_side": "long",
                    "expected_return_pct": 0.40,
                },
            },
            "market_regime": {"mode": "selloff_squeeze_down", "avoid_long": True},
        },
        [
            {"model_name": "trend_expert", "action": "hold", "confidence": 0.80},
            {"model_name": "momentum_expert", "action": "hold", "confidence": 0.78},
        ],
    )

    assert result["allow"] is True
    assert result["status"] == "quant_only_tiny_probe"
    assert result["side"] == "long"


def test_quant_only_probe_allows_tiny_short_when_quant_timeseries_and_direction_align() -> None:
    result = _coordinator()._quant_only_probe_evidence(
        {},
        {
            "local_ai_tools": {
                "profit_prediction": {
                    "available": True,
                    "best_side": "short",
                    "adjusted_short_return_pct": 0.28,
                    "adjusted_long_return_pct": -0.12,
                    "short_loss_probability": 0.46,
                },
                "time_series_prediction": {
                    "available": True,
                    "best_side": "short",
                    "expected_return_pct": 0.12,
                    "confidence": 0.12,
                },
            },
            "direction_competition": {
                "preferred_side": "short",
                "score_gap": 0.18,
                "short": {"score": 0.48},
                "long": {"score": 0.12},
            },
        },
        [
            {"model_name": "trend_expert", "action": "hold", "confidence": 0.80},
            {"model_name": "momentum_expert", "action": "hold", "confidence": 0.78},
        ],
    )

    assert result["allow"] is True
    assert result["status"] == "quant_only_tiny_probe"
    assert result["side"] == "short"
    assert result["supports"] == ["server_profit_model", "time_series_model"]
    assert result["direction_preferred_side"] == "short"
