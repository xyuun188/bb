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
