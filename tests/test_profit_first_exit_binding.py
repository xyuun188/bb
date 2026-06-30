from __future__ import annotations

from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus
from services.open_positions_execution_applier import OpenPositionsExecutionApplier
from services.profit_first_exit_binding import attach_profit_first_exit_reference


def test_exit_reference_attaches_from_matching_open_position() -> None:
    decision = DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.8,
        reasoning="close",
        raw_response={"close_evidence": {"hard_risk": True}},
    )

    raw = attach_profit_first_exit_reference(
        decision,
        [
            {
                "model_name": "ensemble_trader",
                "symbol": "BTC/USDT",
                "side": "long",
                "profit_first_exit_plan": {"exit_plan_id": "pfep-abc"},
                "profit_first_trade_plan": {
                    "plan_version": "profit-first-v3.1",
                    "decision_lane": "meaningful_entry",
                },
            }
        ],
        model_name="ensemble_trader",
    )

    assert raw["profit_first_exit_reference"]["exit_plan_id"] == "pfep-abc"
    assert raw["profit_first_exit_reference"]["missing_original_exit_plan_reference"] is False
    assert raw["close_evidence"]["profit_first_exit_plan_id"] == "pfep-abc"


def test_exit_reference_records_missing_plan_reason() -> None:
    decision = DecisionOutput(
        model_name="ensemble_trader",
        symbol="ETH/USDT",
        action=Action.CLOSE_SHORT,
        confidence=0.8,
        reasoning="close",
        raw_response={"plan_failure_reason": "exchange protection closed before local plan sync"},
    )

    raw = attach_profit_first_exit_reference(
        decision,
        [],
        model_name="ensemble_trader",
    )

    assert raw["profit_first_exit_reference"]["missing_original_exit_plan_reference"] is True
    assert raw["profit_first_exit_reference"]["plan_failure_reason"]


def test_open_position_snapshot_carries_profit_first_exit_plan() -> None:
    decision = DecisionOutput(
        model_name="ensemble_trader",
        symbol="SOL/USDT",
        action=Action.LONG,
        confidence=0.8,
        reasoning="entry",
        raw_response={
            "profit_first_trade_plan": {
                "exit_plan_id": "pfep-sol",
                "plan_version": "profit-first-v3.1",
            },
            "profit_first_exit_plan": {"exit_plan_id": "pfep-sol"},
        },
    )
    result = ExecutionResult(
        order_id="order-1",
        symbol="SOL/USDT",
        side="buy",
        order_type="market",
        quantity=1.0,
        price=100.0,
        status=OrderStatus.FILLED,
    )
    positions: list[dict] = []

    OpenPositionsExecutionApplier(
        normalize_symbol=lambda value: str(value),
        is_exit_progress_execution=lambda _result: False,
    ).apply(positions, "ensemble_trader", decision, result)

    assert positions[0]["profit_first_exit_plan_id"] == "pfep-sol"
    assert positions[0]["profit_first_exit_plan"]["exit_plan_id"] == "pfep-sol"
