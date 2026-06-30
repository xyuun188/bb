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
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        trade_plan = (
            raw.get("profit_first_trade_plan")
            if isinstance(raw.get("profit_first_trade_plan"), dict)
            else {}
        )
        exit_plan = (
            raw.get("profit_first_exit_plan")
            if isinstance(raw.get("profit_first_exit_plan"), dict)
            else {}
        )
        side = "long" if decision.action == Action.LONG else "short"
        existing = self._matching_entry_position(open_positions, model_name, decision, side)
        if existing is not None:
            self._merge_entry(existing, decision, execution_result, trade_plan, exit_plan)
            return

        open_positions.append(
            {
                "model_name": model_name,
                "symbol": decision.symbol,
                "side": side,
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
                "profit_first_trade_plan": trade_plan,
                "profit_first_exit_plan": exit_plan,
                "profit_first_exit_plan_id": (
                    exit_plan.get("exit_plan_id") or trade_plan.get("exit_plan_id") or ""
                ),
            }
        )

    def _matching_entry_position(
        self,
        open_positions: list[dict[str, Any]],
        model_name: str,
        decision: DecisionOutput,
        side: str,
    ) -> dict[str, Any] | None:
        for position in open_positions:
            if self._matches_position(position, model_name, decision, side):
                return position
        return None

    def _merge_entry(
        self,
        position: dict[str, Any],
        decision: DecisionOutput,
        execution_result: ExecutionResult,
        trade_plan: dict[str, Any],
        exit_plan: dict[str, Any],
    ) -> None:
        old_qty = max(float(position.get("quantity") or 0.0), 0.0)
        add_qty = max(float(execution_result.quantity or 0.0), 0.0)
        if add_qty <= 0:
            return
        old_entry = float(position.get("entry_price") or execution_result.price or 0.0)
        add_entry = float(execution_result.price or old_entry or 0.0)
        total_qty = old_qty + add_qty
        entry_price = (
            ((old_entry * old_qty) + (add_entry * add_qty)) / total_qty
            if total_qty > 0
            else add_entry
        )
        side = "long" if decision.action == Action.LONG else "short"
        position["symbol"] = decision.symbol
        position["side"] = side
        position["quantity"] = total_qty
        position["entry_price"] = entry_price
        position["current_price"] = execution_result.price
        position["stop_loss"] = (
            entry_price * (1 - decision.stop_loss_pct)
            if side == "long"
            else entry_price * (1 + decision.stop_loss_pct)
        )
        position["take_profit"] = (
            entry_price * (1 + decision.take_profit_pct)
            if side == "long"
            else entry_price * (1 - decision.take_profit_pct)
        )
        position["unrealized_pnl"] = (
            (execution_result.price - entry_price) * total_qty
            if side == "long"
            else (entry_price - execution_result.price) * total_qty
        )
        position["is_open"] = True
        position["profit_first_trade_plan"] = trade_plan
        position["profit_first_exit_plan"] = exit_plan
        position["profit_first_exit_plan_id"] = (
            exit_plan.get("exit_plan_id") or trade_plan.get("exit_plan_id") or ""
        )
        history = position.get("entry_legs")
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "quantity": add_qty,
                "price": execution_result.price,
                "exchange_order_id": execution_result.exchange_order_id,
                "profit_first_exit_plan_id": position.get("profit_first_exit_plan_id") or "",
            }
        )
        position["entry_legs"] = history[-20:]
        position["merged_entry_count"] = int(position.get("merged_entry_count") or 1) + 1

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
