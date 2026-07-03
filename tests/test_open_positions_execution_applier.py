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
        "entry_exchange_order_id": "okx-1",
        "entry_legs": [
            {
                "quantity": 2.0,
                "price": 100.0,
                "exchange_order_id": "okx-1",
                "profit_first_exit_plan_id": "",
            }
        ],
    }


def test_open_positions_applier_keeps_same_symbol_side_entries_separate() -> None:
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
            "entry_exchange_order_id": "okx-old",
            "entry_legs": [
                {
                    "quantity": 1.0,
                    "price": 90.0,
                    "exchange_order_id": "okx-old",
                    "profit_first_exit_plan_id": "pfep-old",
                }
            ],
        }
    ]

    _applier().apply(
        open_positions,
        "ensemble_trader",
        decision,
        _result(OrderStatus.FILLED, quantity=3.0),
    )

    assert len(open_positions) == 2
    assert open_positions[0]["entry_exchange_order_id"] == "okx-old"
    assert open_positions[0]["profit_first_exit_plan_id"] == "pfep-old"
    newest = open_positions[1]
    assert newest["quantity"] == 3.0
    assert newest["entry_price"] == 100.0
    assert newest["current_price"] == 100.0
    assert newest["stop_loss"] == 98.0
    assert newest["take_profit"] == 104.0
    assert newest["profit_first_exit_plan_id"] == "pfep-new"
    assert newest["entry_exchange_order_id"] == "okx-1"
    assert newest["entry_legs"][0]["exchange_order_id"] == "okx-1"


def test_open_positions_applier_ignores_duplicate_entry_callback_for_same_order_id() -> None:
    decision = _decision(Action.LONG)
    open_positions = [
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC/USDT",
            "side": "long",
            "entry_price": 100.0,
            "current_price": 100.0,
            "quantity": 2.0,
            "unrealized_pnl": 0.0,
            "is_open": True,
            "profit_first_trade_plan": {},
            "profit_first_exit_plan": {},
            "profit_first_exit_plan_id": "",
            "entry_exchange_order_id": "okx-1",
            "entry_legs": [
                {
                    "quantity": 2.0,
                    "price": 100.0,
                    "exchange_order_id": "okx-1",
                    "profit_first_exit_plan_id": "",
                }
            ],
        }
    ]

    _applier().apply(
        open_positions,
        "ensemble_trader",
        decision,
        _result(OrderStatus.FILLED, quantity=2.0),
    )

    assert len(open_positions) == 1
    assert open_positions[0]["quantity"] == 2.0


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
