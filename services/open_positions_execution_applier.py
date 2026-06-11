"""Apply confirmed executions to an in-memory open-position snapshot."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus

SymbolNormalizer = Callable[[Any], str]
ExitProgressChecker = Callable[[ExecutionResult | None], bool]


class OpenPositionsExecutionApplier:
    """Update loop-local positions after a filled entry or progressing exit."""

    def __init__(
        self,
        *,
        normalize_symbol: SymbolNormalizer,
        is_exit_progress_execution: ExitProgressChecker,
    ) -> None:
        self._normalize_symbol = normalize_symbol
        self._is_exit_progress_execution = is_exit_progress_execution

    def apply(
        self,
        open_positions: list[dict[str, Any]],
        model_name: str,
        decision: DecisionOutput,
        execution_result: ExecutionResult,
    ) -> None:
        if execution_result.status != OrderStatus.FILLED and not self._is_exit_progress_execution(
            execution_result
        ):
            return

        if decision.action in (Action.LONG, Action.SHORT):
            self._apply_entry(open_positions, model_name, decision, execution_result)
            return

        if decision.action == Action.CLOSE_LONG:
            side = "long"
        elif decision.action == Action.CLOSE_SHORT:
            side = "short"
        else:
            return

        self._apply_exit(open_positions, model_name, decision, execution_result, side)

    def _apply_entry(
        self,
        open_positions: list[dict[str, Any]],
        model_name: str,
        decision: DecisionOutput,
        execution_result: ExecutionResult,
    ) -> None:
        if execution_result.status != OrderStatus.FILLED:
            return
        open_positions.append(
            {
                "model_name": model_name,
                "symbol": decision.symbol,
                "side": "long" if decision.action == Action.LONG else "short",
                "entry_price": execution_result.price,
                "current_price": execution_result.price,
                "quantity": execution_result.quantity,
                "unrealized_pnl": 0.0,
                "stop_loss": (
                    execution_result.price * (1 - decision.stop_loss_pct)
                    if decision.action == Action.LONG
                    else execution_result.price * (1 + decision.stop_loss_pct)
                ),
                "take_profit": (
                    execution_result.price * (1 + decision.take_profit_pct)
                    if decision.action == Action.LONG
                    else execution_result.price * (1 - decision.take_profit_pct)
                ),
                "is_open": True,
            }
        )

    def _apply_exit(
        self,
        open_positions: list[dict[str, Any]],
        model_name: str,
        decision: DecisionOutput,
        execution_result: ExecutionResult,
        side: str,
    ) -> None:
        remaining_qty = float(execution_result.quantity or 0.0)
        if remaining_qty <= 0:
            return

        for position in list(open_positions):
            if remaining_qty <= 0:
                break
            if not self._matches_position(position, model_name, decision, side):
                continue

            qty = float(position.get("quantity") or 0.0)
            if qty <= 0:
                continue

            close_qty = min(qty, remaining_qty)
            new_qty = qty - close_qty
            if new_qty <= 1e-12:
                open_positions.remove(position)
            else:
                position["quantity"] = new_qty
                position["current_price"] = execution_result.price
            remaining_qty -= close_qty

    def _matches_position(
        self,
        position: dict[str, Any],
        model_name: str,
        decision: DecisionOutput,
        side: str,
    ) -> bool:
        return bool(
            position.get("model_name") == model_name
            and self._normalize_symbol(position.get("symbol"))
            == self._normalize_symbol(decision.symbol)
            and position.get("side") == side
        )
