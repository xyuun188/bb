from __future__ import annotations

from ai_brain.base_model import Action, DecisionOutput
from services.exit_intent import ExitIntent, classify_exit_intent


def _decision(raw_response: dict | None = None, reasoning: str = "exit") -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.8,
        reasoning=reasoning,
        position_size_pct=1.0,
        suggested_leverage=3.0,
        raw_response=raw_response or {},
        feature_snapshot={"current_price": 100.0},
    )


def test_exit_intent_uses_existing_structured_value() -> None:
    decision = _decision({"exit_intent": "trend_failure"})

    assert classify_exit_intent(decision) == ExitIntent.TREND_FAILURE
    assert decision.raw_response["exit_intent"] == "trend_failure"


def test_exit_intent_classifies_predictive_downside_from_close_evidence() -> None:
    decision = _decision(
        {
            "close_evidence": {
                "should_close": True,
                "moderate_opposite_pressure": True,
                "preventive_exit": True,
            }
        }
    )

    assert classify_exit_intent(decision) == ExitIntent.PREDICTIVE_DOWNSIDE
    assert decision.raw_response["exit_intent"] == "predictive_downside"
    assert decision.raw_response["close_evidence"]["exit_intent"] == "predictive_downside"


def test_exit_intent_keeps_loss_repair_separate_from_predictive_exit() -> None:
    decision = _decision(
        {
            "close_evidence": {
                "position_loss": True,
                "loss_repair": True,
                "moderate_opposite_pressure": True,
            }
        }
    )

    assert classify_exit_intent(decision) == ExitIntent.LOSS_REPAIR


def test_exit_intent_does_not_treat_profit_drawdown_fast_trigger_as_hard_risk() -> None:
    decision = _decision(
        {
            "fast_risk_exit": True,
            "fast_risk_trigger": "profit_drawdown_close",
        }
    )

    assert classify_exit_intent(decision) == ExitIntent.PROFIT_DRAWDOWN
    assert decision.raw_response["exit_intent"] == "profit_drawdown"


def test_exit_intent_does_not_treat_take_profit_fast_trigger_as_hard_risk() -> None:
    decision = _decision(
        {
            "fast_risk_exit": True,
            "fast_risk_trigger": "take_profit",
        }
    )

    assert classify_exit_intent(decision) == ExitIntent.PROFIT_PROTECTION
    assert decision.raw_response["exit_intent"] == "profit_protection"


def test_exit_intent_keeps_fast_adverse_move_as_hard_risk() -> None:
    decision = _decision(
        {
            "fast_risk_exit": True,
            "fast_risk_trigger": "fast_adverse_move",
        }
    )

    assert classify_exit_intent(decision) == ExitIntent.HARD_RISK
    assert decision.raw_response["exit_intent"] == "hard_risk"


def test_exit_intent_reclassifies_low_quality_release_as_capital_rotation() -> None:
    decision = _decision(
        {
            "forced_exit": True,
            "exit_intent": "hard_risk",
            "position_release_policy": {
                "source": "position_quality_capacity_release",
                "forced": True,
            },
            "close_evidence": {
                "forced_exit": True,
                "hard_risk": False,
                "source": "low_quality_position_release",
            },
        }
    )

    assert classify_exit_intent(decision) == ExitIntent.CAPITAL_ROTATION
    assert decision.raw_response["exit_intent"] == "capital_rotation"
    assert decision.raw_response["close_evidence"]["exit_intent"] == "capital_rotation"


def test_exit_intent_keeps_low_quality_release_hard_when_real_risk_is_present() -> None:
    decision = _decision(
        {
            "forced_exit": True,
            "position_release_policy": {
                "source": "position_quality_capacity_release",
                "forced": True,
            },
            "close_evidence": {
                "forced_exit": True,
                "hard_risk": True,
                "source": "low_quality_position_release",
            },
        }
    )

    assert classify_exit_intent(decision) == ExitIntent.HARD_RISK
