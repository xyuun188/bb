from __future__ import annotations

from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus
from services.fast_risk_exit_execution import FastRiskExitExecutionProcessor


def _decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.CLOSE_LONG,
        confidence=1.0,
        reasoning="fast risk",
        position_size_pct=0.5,
    )


def _execution_result(status: OrderStatus = OrderStatus.FILLED) -> ExecutionResult:
    return ExecutionResult(
        order_id="order-1",
        exchange_order_id="okx-1",
        symbol="BTC/USDT",
        side="sell",
        order_type="market",
        quantity=2.0,
        price=99.0,
        status=status,
        pnl=1.25,
    )


def _processor(
    calls: list[tuple[str, Any]],
    *,
    execution_result: ExecutionResult | None = None,
    execute_error: Exception | None = None,
) -> FastRiskExitExecutionProcessor:
    async def log_decision(decision: DecisionOutput, is_paper: bool) -> int:
        calls.append(("log_decision", decision.action.value, is_paper))
        return 77

    def increment() -> None:
        calls.append(("increment", None))

    async def execute_candidate(*args: Any, **kwargs: Any) -> ExecutionResult | None:
        calls.append(
            (
                "execute",
                args[0],
                args[1],
                args[2].action.value,
                args[4],
                kwargs.get("refresh_exit_positions"),
            )
        )
        if execute_error is not None:
            raise execute_error
        return execution_result

    def exchange_confirmed(result: ExecutionResult | None) -> bool:
        return result is not None and result.status == OrderStatus.FILLED

    def exit_progress(_result: ExecutionResult | None) -> bool:
        return False

    def remember_profit(model: str, symbol: str, side: str) -> None:
        calls.append(("profit_exit", model, symbol, side))

    async def log_risk(level: str, symbol: str, message: str, model_name: str) -> None:
        calls.append(("risk_event", level, symbol, model_name, message))

    def rejected(_decision: DecisionOutput, _exc: Exception) -> ExecutionResult:
        calls.append(("rejected", str(_exc)))
        return _execution_result(OrderStatus.REJECTED)

    async def log_trade(
        result: ExecutionResult, model: str, decision: DecisionOutput, db_id: int
    ) -> None:
        calls.append(("trade", result.status.value, model, decision.action.value, db_id))

    async def mark_reason(decision_id: int, reason: str) -> None:
        calls.append(("reason", decision_id, reason))

    return FastRiskExitExecutionProcessor(
        model_execution_mode_provider=lambda _model: "paper",
        decision_logger=log_decision,
        decision_count_incrementer=increment,
        candidate_executor=execute_candidate,
        exchange_confirmed_checker=exchange_confirmed,
        exit_progress_checker=exit_progress,
        profit_exit_recorder=remember_profit,
        risk_event_logger=log_risk,
        rejected_execution_factory=rejected,
        trade_logger=log_trade,
        decision_reason_marker=mark_reason,
        execution_reason_provider=lambda result: f"status:{result.status.value}",
    )


@pytest.mark.asyncio
async def test_fast_risk_exit_execution_records_success_and_profit_peak() -> None:
    calls: list[tuple[str, Any]] = []
    result = await _processor(calls, execution_result=_execution_result()).execute(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        side="long",
        position={"symbol": "BTC/USDT"},
        decision=_decision(),
        trigger="profit_drawdown_reduce",
        reason="lock profit",
        close_fraction=0.5,
        entry_price=100.0,
        current_price=99.0,
    )

    assert result.auto_close == {
        "model_name": "ensemble_trader",
        "symbol": "BTC/USDT",
        "side": "long",
        "quantity": 2.0,
        "entry_price": 100.0,
        "exit_price": 99.0,
        "pnl": 1.25,
        "trigger": "profit_drawdown_reduce",
        "close_fraction": 0.5,
        "status": "filled",
    }
    assert ("profit_exit", "ensemble_trader", "BTC/USDT", "long") in calls
    assert ("execute", "BTC/USDT", "ensemble_trader", "close_long", 77, False) in calls


@pytest.mark.asyncio
async def test_fast_risk_exit_execution_reports_skipped_when_executor_returns_none() -> None:
    calls: list[tuple[str, Any]] = []
    result = await _processor(calls, execution_result=None).execute(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        side="long",
        position={},
        decision=_decision(),
        trigger="fast_adverse_reduce",
        reason="risk",
        close_fraction=0.5,
        entry_price=100.0,
        current_price=98.0,
    )

    assert result.skipped is True
    assert result.auto_close is None


@pytest.mark.asyncio
async def test_fast_risk_exit_execution_records_rejected_result_on_error() -> None:
    calls: list[tuple[str, Any]] = []
    result = await _processor(calls, execute_error=RuntimeError("submit failed")).execute(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        side="long",
        position={},
        decision=_decision(),
        trigger="fast_adverse_reduce",
        reason="risk",
        close_fraction=0.5,
        entry_price=100.0,
        current_price=98.0,
    )

    assert result.error == "submit failed"
    assert result.auto_close is not None
    assert result.auto_close["status"] == "rejected"
    assert ("rejected", "submit failed") in calls
    assert ("reason", 77, "status:rejected") in calls
