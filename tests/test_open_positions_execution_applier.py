from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus
from services.open_positions_execution_applier import OpenPositionsExecutionApplier


def _decision(action: Action) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.8,
        reasoning="test",
        position_size_pct=0.1,
        suggested_leverage=3.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
    )


def _result(status: OrderStatus, *, quantity: float = 1.0) -> ExecutionResult:
    return ExecutionResult(
        order_id="local-1",
        exchange_order_id="okx-1",
        symbol="BTC/USDT",
        side="long",
        order_type="market",
        quantity=quantity,
        price=100.0,
        status=status,
    )


def _applier(exit_progress: bool = False) -> OpenPositionsExecutionApplier:
    return OpenPositionsExecutionApplier(
        normalize_symbol=lambda value: str(value).replace("-", "/"),
        is_exit_progress_execution=lambda _result: exit_progress,
    )


def test_open_positions_applier_adds_filled_entry() -> None:
    open_positions: list[dict] = []

    _applier().apply(
        open_positions,
        "ensemble_trader",
        _decision(Action.LONG),
        _result(OrderStatus.FILLED, quantity=2.0),
    )

    assert len(open_positions) == 1
    assert open_positions[0] == {
        "model_name": "ensemble_trader",
        "symbol": "BTC/USDT",
        "side": "long",
        "entry_price": 100.0,
        "current_price": 100.0,
        "quantity": 2.0,
        "unrealized_pnl": 0.0,
        "stop_loss": 98.0,
        "take_profit": 104.0,
        "is_open": True,
        "profit_first_trade_plan": {},
        "profit_first_exit_plan": {},
        "profit_first_exit_plan_id": "",
    }


def test_open_positions_applier_merges_same_symbol_side_add_entry() -> None:
    decision = _decision(Action.LONG)
    decision.raw_response = {
        "profit_first_trade_plan": {"exit_plan_id": "pfep-new"},
        "profit_first_exit_plan": {"exit_plan_id": "pfep-new"},
    }
    open_positions = [
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC-USDT",
            "side": "long",
            "entry_price": 90.0,
            "current_price": 95.0,
            "quantity": 1.0,
            "unrealized_pnl": 5.0,
            "is_open": True,
            "profit_first_trade_plan": {"exit_plan_id": "pfep-old"},
            "profit_first_exit_plan": {"exit_plan_id": "pfep-old"},
            "profit_first_exit_plan_id": "pfep-old",
        }
    ]

    _applier().apply(
        open_positions,
        "ensemble_trader",
        decision,
        _result(OrderStatus.FILLED, quantity=3.0),
    )

    assert len(open_positions) == 1
    merged = open_positions[0]
    assert merged["quantity"] == 4.0
    assert merged["entry_price"] == 97.5
    assert merged["current_price"] == 100.0
    assert merged["stop_loss"] == 95.55
    assert merged["take_profit"] == 101.4
    assert merged["profit_first_exit_plan_id"] == "pfep-new"
    assert merged["merged_entry_count"] == 2
    assert merged["entry_legs"][-1]["exchange_order_id"] == "okx-1"


def test_open_positions_applier_ignores_unconfirmed_entry() -> None:
    open_positions: list[dict] = []

    _applier(exit_progress=True).apply(
        open_positions,
        "ensemble_trader",
        _decision(Action.LONG),
        _result(OrderStatus.PARTIAL, quantity=2.0),
    )

    assert open_positions == []


def test_open_positions_applier_reduces_matching_exit_progress() -> None:
    open_positions = [
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC-USDT",
            "side": "long",
            "quantity": 3.0,
            "current_price": 95.0,
        },
        {
            "model_name": "other_model",
            "symbol": "BTC-USDT",
            "side": "long",
            "quantity": 3.0,
        },
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC-USDT",
            "side": "short",
            "quantity": 3.0,
        },
    ]

    _applier(exit_progress=True).apply(
        open_positions,
        "ensemble_trader",
        _decision(Action.CLOSE_LONG),
        _result(OrderStatus.PARTIAL, quantity=1.25),
    )

    assert open_positions[0]["quantity"] == 1.75
    assert open_positions[0]["current_price"] == 100.0
    assert open_positions[1]["quantity"] == 3.0
    assert open_positions[2]["quantity"] == 3.0


def test_open_positions_applier_removes_fully_closed_position() -> None:
    open_positions = [
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC/USDT",
            "side": "short",
            "quantity": 1.0,
        }
    ]

    _applier().apply(
        open_positions,
        "ensemble_trader",
        _decision(Action.CLOSE_SHORT),
        _result(OrderStatus.FILLED, quantity=1.0),
    )

    assert open_positions == []
