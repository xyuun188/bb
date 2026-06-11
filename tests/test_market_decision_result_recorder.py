from __future__ import annotations

from ai_brain.base_model import Action, DecisionOutput
from services.market_decision_result_recorder import MarketDecisionResultRecorder


def test_market_decision_result_recorder_appends_standard_row_from_action() -> None:
    results = {"decisions": []}

    row = MarketDecisionResultRecorder().append_result(
        results=results,
        model_name="ensemble_trader",
        symbol="ALL",
        decision_or_action="hold",
        model_mode="paper",
        approved=False,
        execution_status="paused",
        reason="paused",
    )

    assert row == {
        "model": "ensemble_trader",
        "symbol": "ALL",
        "action": "hold",
        "approved": False,
        "executed": False,
        "execution_status": "paused",
        "reason": "paused",
        "is_paper": True,
    }
    assert results["decisions"] == [row]


def test_market_decision_result_recorder_uses_decision_action_and_confidence() -> None:
    decision = DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.73,
        reasoning="entry",
    )
    results = {"decisions": []}

    row = MarketDecisionResultRecorder().append_result(
        results=results,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        decision_or_action=decision,
        model_mode="live",
        reason="queued",
    )

    assert row["action"] == "long"
    assert row["confidence"] == 0.73
    assert row["is_paper"] is False
