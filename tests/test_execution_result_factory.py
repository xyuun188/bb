from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import OrderStatus
from services.execution_result_factory import ExecutionResultFactory


def _decision(action: Action) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.8,
        reasoning="test",
        position_size_pct=0.1,
        suggested_leverage=3.0,
    )


def test_execution_result_factory_maps_decision_to_exchange_side() -> None:
    factory = ExecutionResultFactory()

    assert factory.decision_side(_decision(Action.LONG)) == "buy"
    assert factory.decision_side(_decision(Action.SHORT)) == "sell"
    assert factory.decision_side(_decision(Action.CLOSE_LONG)) == "sell"
    assert factory.decision_side(_decision(Action.CLOSE_SHORT)) == "buy"
    assert factory.decision_side(_decision(Action.HOLD)) == "hold"


def test_execution_result_factory_builds_rejected_result() -> None:
    result = ExecutionResultFactory().rejected(_decision(Action.CLOSE_LONG), "blocked")

    assert result.order_id == "rejected"
    assert result.symbol == "BTC/USDT"
    assert result.side == "sell"
    assert result.quantity == 0.0
    assert result.price == 0.0
    assert result.status == OrderStatus.REJECTED
    assert result.raw_response == {"error": "blocked"}


def test_execution_result_factory_action_labels_match_dashboard_text() -> None:
    factory = ExecutionResultFactory()

    assert factory.action_label(Action.LONG) == "做多"
    assert factory.action_label("close_short") == "平空"
    assert factory.action_label(None) == "未知"
