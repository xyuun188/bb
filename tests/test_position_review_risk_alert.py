from __future__ import annotations

from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus
from services.position_review_risk_alert import PositionReviewRiskAlertPolicy


def _decision(action: Action = Action.HOLD) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.8,
        reasoning="test",
        position_size_pct=0.0,
        suggested_leverage=1.0,
        raw_response={},
    )


def _policy() -> PositionReviewRiskAlertPolicy:
    return PositionReviewRiskAlertPolicy(
        float_parser=lambda value, default: float(value if value is not None else default),
        text_shortener=lambda value, limit: str(value or "")[:limit],
        action_labeler=lambda value: value.value if hasattr(value, "value") else str(value),
    )


def test_position_review_risk_alert_requires_governed_dynamic_exit() -> None:
    policy = _policy()
    decision = _decision(Action.CLOSE_LONG)
    decision.raw_response = {
        "opinions": [
            {
                "model_name": "risk_expert",
                "action": "close_long",
                "confidence": 1.0,
            }
        ]
    }

    assert policy.build_alert(decision, [{"side": "long"}]) is None


def test_position_review_risk_alert_builds_observation_after_dynamic_exit() -> None:
    policy = _policy()
    decision = _decision(Action.CLOSE_LONG)
    decision.raw_response = {
        "close_evidence": {"dynamic_exit_policy": {"eligible": True}},
        "opinions": [
            {
                "model_name": "risk_expert",
                "action": "close_long",
                "confidence": 0.76,
                "reasoning": "risk observation",
            }
        ],
    }

    message = policy.build_alert(
        decision,
        [{"side": "long", "entry_price": 100.0, "quantity": 2, "unrealized_pnl": -8.5}],
    )

    assert message is not None
    assert "BTC/USDT" in message
    assert "(76%)" in message
    policy.attach(decision, message)
    assert policy.alert_context(decision) == {
        "message": message,
        "planned_action": "close_long",
        "production_permission": False,
    }


def test_position_review_risk_alert_formats_execution_result_text() -> None:
    policy = _policy()
    decision = _decision(Action.CLOSE_SHORT)
    filled = ExecutionResult(
        order_id="1",
        exchange_order_id="okx-1",
        symbol="BTC/USDT",
        side="close_short",
        order_type="market",
        quantity=3.0,
        price=99.5,
        status=OrderStatus.FILLED,
    )
    rejected = ExecutionResult(
        order_id=None,
        exchange_order_id=None,
        symbol="BTC/USDT",
        side="close_short",
        order_type="market",
        quantity=0.0,
        price=0.0,
        status=OrderStatus.REJECTED,
    )

    assert "Execution filled" in policy.execution_result_text(
        decision, filled, lambda _result: "unused"
    )
    assert "reason=blocked" in policy.execution_result_text(
        decision, rejected, lambda _result: "blocked"
    )


def test_position_review_risk_alert_detail_uses_original_alert_message() -> None:
    detail = _policy().risk_event_detail(
        _decision(Action.CLOSE_LONG),
        {"message": "risk message"},
        "done",
    )

    assert detail == "risk message System action=close_long. Execution result=done"
