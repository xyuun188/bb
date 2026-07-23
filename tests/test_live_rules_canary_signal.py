from __future__ import annotations

from copy import deepcopy

from ai_brain.base_model import Action, DecisionOutput
from services.live_rules_canary_signal import apply_live_rules_canary_signal
from services.production_trade_gate import PRODUCTION_TRADE_GATE_VERSION


def _gate() -> dict[str, object]:
    return {
        "version": PRODUCTION_TRADE_GATE_VERSION,
        "mode": "live_rules_canary",
        "can_trade": True,
        "decision_authority": "rules",
        "model_can_influence": False,
        "risk": {"max_notional_usdt": 10.0},
    }


def _decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.99,
        reasoning="model wanted long",
        position_size_pct=0.8,
        suggested_leverage=20.0,
        stop_loss_pct=0.5,
        take_profit_pct=0.8,
        suggested_holding_minutes=120.0,
        maximum_holding_minutes=240.0,
        raw_response={"ml_signal": {"action": "long", "score": 0.99}},
        feature_snapshot={
            "current_price": 100.0,
            "atr_14": 1.2,
            "volatility_20": 0.01,
            "returns_1": -0.01,
            "returns_5": -0.03,
            "returns_20": -0.05,
            "macd_diff": -0.2,
            "price_vs_sma20": -0.01,
            "price_vs_sma50": 0.0,
        },
    )


def test_rules_canary_overrides_model_direction_and_all_execution_parameters() -> None:
    decision = _decision()

    signal = apply_live_rules_canary_signal(decision, _gate())

    assert signal is not None and signal["production_eligible"] is True
    assert signal["action"] == "short"
    assert decision.action == Action.SHORT
    assert decision.position_size_pct == 0.0
    assert decision.suggested_leverage == 1.0
    assert decision.stop_loss_pct == 0.0
    assert decision.take_profit_pct == 0.0
    assert decision.suggested_holding_minutes == 0.0
    assert decision.maximum_holding_minutes == 0.0
    shadow = decision.raw_response["model_shadow_decision"]
    assert shadow["action"] == "long"
    assert shadow["position_size_pct"] == 0.8
    assert shadow["suggested_leverage"] == 20.0
    assert shadow["observation_only"] is True
    assert shadow["can_authorize_entry"] is False


def test_rules_canary_holds_when_technical_direction_is_not_proven() -> None:
    decision = _decision()
    decision.feature_snapshot.update(
        {
            "returns_1": 0.01,
            "returns_5": -0.01,
            "returns_20": 0.01,
            "macd_diff": -0.01,
            "price_vs_sma20": 0.0,
            "price_vs_sma50": 0.0,
        }
    )

    signal = apply_live_rules_canary_signal(decision, _gate())

    assert signal is not None and signal["production_eligible"] is False
    assert decision.action == Action.HOLD
    assert "rules_canary_signal_consensus_weak" in signal["blockers"]


def test_non_canary_gate_does_not_mutate_model_decision() -> None:
    decision = _decision()
    before = deepcopy(decision)
    gate = {**_gate(), "mode": "live_ml", "decision_authority": "model"}

    signal = apply_live_rules_canary_signal(decision, gate)

    assert signal is None
    assert decision == before
