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
        side = "long" if decision.action == Action.LONG else "short"
        entry_exchange_order_id = self._entry_exchange_order_id(execution_result)
        existing = self._matching_entry_position(
            open_positions,
            model_name,
            decision,
            side,
            entry_exchange_order_id=entry_exchange_order_id,
        )
        if existing is not None:
            is_replay = self._position_has_entry_order_id(
                existing,
                entry_exchange_order_id,
            )
            self._refresh_existing_entry(
                existing,
                decision,
                execution_result,
                entry_exchange_order_id=entry_exchange_order_id,
                add_execution=not is_replay,
            )
            return

        entry_leg = {
            "quantity": execution_result.quantity,
            "price": execution_result.price,
            "exchange_order_id": entry_exchange_order_id,
        }
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
                "entry_exchange_order_id": entry_exchange_order_id,
                "entry_legs": [entry_leg],
            }
        )

    def _matching_entry_position(
        self,
        open_positions: list[dict[str, Any]],
        model_name: str,
        decision: DecisionOutput,
        side: str,
        *,
        entry_exchange_order_id: str,
    ) -> dict[str, Any] | None:
        fallback: dict[str, Any] | None = None
        for position in open_positions:
            if not self._matches_position(position, model_name, decision, side):
                continue
            if entry_exchange_order_id and self._position_has_entry_order_id(
                position,
                entry_exchange_order_id,
            ):
                return position
            if fallback is None:
                fallback = position
        return fallback

    def _refresh_existing_entry(
        self,
        position: dict[str, Any],
        decision: DecisionOutput,
        execution_result: ExecutionResult,
        *,
        entry_exchange_order_id: str,
        add_execution: bool,
    ) -> None:
        side = "long" if decision.action == Action.LONG else "short"
        if add_execution:
            old_quantity = max(float(position.get("quantity") or 0.0), 0.0)
            added_quantity = max(float(execution_result.quantity or 0.0), 0.0)
            total_quantity = old_quantity + added_quantity
            if total_quantity > 0:
                position["entry_price"] = (
                    old_quantity
                    * float(position.get("entry_price") or execution_result.price or 0.0)
                    + added_quantity * float(execution_result.price or 0.0)
                ) / total_quantity
                position["quantity"] = total_quantity
            legs = position.setdefault("entry_legs", [])
            if isinstance(legs, list):
                legs.append(
                    {
                        "quantity": execution_result.quantity,
                        "price": execution_result.price,
                        "exchange_order_id": entry_exchange_order_id,
                    }
                )
        position["symbol"] = decision.symbol
        position["side"] = side
        position["current_price"] = execution_result.price
        position["stop_loss"] = (
            float(position.get("entry_price") or execution_result.price or 0.0)
            * (1 - decision.stop_loss_pct)
            if side == "long"
            else float(position.get("entry_price") or execution_result.price or 0.0)
            * (1 + decision.stop_loss_pct)
        )
        position["take_profit"] = (
            float(position.get("entry_price") or execution_result.price or 0.0)
            * (1 + decision.take_profit_pct)
            if side == "long"
            else float(position.get("entry_price") or execution_result.price or 0.0)
            * (1 - decision.take_profit_pct)
        )
        position["is_open"] = True
        if entry_exchange_order_id:
            position["entry_exchange_order_id"] = self._merge_entry_order_ids(
                position.get("entry_exchange_order_id"),
                entry_exchange_order_id,
            )

    @staticmethod
    def _entry_exchange_order_id(execution_result: ExecutionResult) -> str:
        return str(
            getattr(execution_result, "exchange_order_id", None)
            or getattr(execution_result, "order_id", None)
            or ""
        ).strip()

    @staticmethod
    def _position_has_entry_order_id(position: dict[str, Any], target_order_id: str) -> bool:
        order_id = str(target_order_id or "").strip()
        if not order_id:
            return False
        top_level = str(position.get("entry_exchange_order_id") or "").strip()
        if top_level == order_id:
            return True
        legs = position.get("entry_legs")
        if not isinstance(legs, list):
            return False
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            if str(leg.get("exchange_order_id") or "").strip() == order_id:
                return True
        return False

    @staticmethod
    def _merge_entry_order_ids(*values: Any) -> str:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            for token in str(value or "").replace(";", ",").split(","):
                order_id = token.strip()
                if not order_id or order_id in seen:
                    continue
                candidate = ",".join([*result, order_id]) if result else order_id
                if len(candidate) > 500:
                    return ",".join(result)
                seen.add(order_id)
                result.append(order_id)
        return ",".join(result)

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
