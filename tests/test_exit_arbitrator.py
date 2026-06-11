from ai_brain.base_model import Action, DecisionOutput
from services.exit_arbitrator import ExitArbitrator
from services.exit_intent import ExitIntent


def _decision(raw_response: dict | None = None) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.8,
        reasoning="test",
        position_size_pct=0.5,
        suggested_leverage=3.0,
        raw_response=raw_response or {},
        feature_snapshot={"current_price": 100.0},
    )


def test_exit_arbitrator_gives_hard_risk_top_priority() -> None:
    decision = _decision({"close_evidence": {"hard_risk": True}})

    result = ExitArbitrator().arbitrate(decision)

    assert result.intent == ExitIntent.HARD_RISK
    assert result.priority == 100
    assert result.bypass_partial_guard is True
    assert result.bypass_cooldown is True
    assert result.bypass_profit_precheck is True
    assert result.bypass_fee_churn_guard is True
    assert decision.raw_response["exit_arbitration"]["intent"] == "hard_risk"


def test_exit_arbitrator_keeps_ordinary_exit_on_full_guard_chain() -> None:
    decision = _decision()

    result = ExitArbitrator().arbitrate(decision)

    assert result.intent == ExitIntent.ORDINARY
    assert result.priority == 10
    assert result.bypass_partial_guard is False
    assert result.bypass_cooldown is False
    assert result.bypass_profit_precheck is False
    assert result.bypass_fee_churn_guard is False
