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
        reasoning="测试",
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


def test_position_review_risk_alert_requires_urgent_risk_opinion() -> None:
    policy = _policy()
    decision = _decision()
    decision.raw_response = {
        "opinions": [
            {
                "model_name": "risk_expert",
                "action": "hold",
                "confidence": 0.4,
                "reasoning": "风险普通",
            }
        ]
    }

    assert policy.build_alert(decision, [{"side": "long"}]) is None


def test_position_review_risk_alert_builds_and_attaches_context() -> None:
    policy = _policy()
    decision = _decision(Action.CLOSE_LONG)
    decision.raw_response = {
        "opinions": [
            {
                "model_name": "risk_expert",
                "action": "close_long",
                "confidence": 0.76,
                "reasoning": "止损风险扩大，需要立即处理",
            }
        ]
    }

    message = policy.build_alert(
        decision,
        [{"side": "long", "entry_price": 100.0, "quantity": 2, "unrealized_pnl": -8.5}],
    )

    assert message is not None
    assert "BTC/USDT" in message
    assert "置信度=76%" in message
    policy.attach(decision, message)
    assert policy.alert_context(decision) == {
        "message": message,
        "planned_action": "close_long",
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

    assert "已执行完成" in policy.execution_result_text(
        decision,
        filled,
        lambda _result: "unused",
    )
    assert "原因=blocked" in policy.execution_result_text(
        decision,
        rejected,
        lambda _result: "blocked",
    )


def test_position_review_risk_alert_detail_uses_original_alert_message() -> None:
    policy = _policy()
    decision = _decision(Action.CLOSE_LONG)

    detail = policy.risk_event_detail(
        decision,
        {"message": "risk message"},
        "done",
    )

    assert detail == "risk message 系统动作=close_long。执行结果=done"
