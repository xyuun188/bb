from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any


def _default_float_parser(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class PositionSnapshotSyncer:
    """Apply an exchange open-position snapshot to local open position rows."""

    def __init__(self, float_parser: Callable[[Any, float], float] | None = None) -> None:
        self.float_parser = float_parser or _default_float_parser

    def sync(
        self,
        positions: list[Any],
        *,
        exchange_quantity: float,
        current_price: float,
        entry_price: float,
        leverage: float,
        exchange_unrealized: float,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> bool:
        open_positions = [position for position in positions if getattr(position, "is_open", False)]
        if not open_positions or exchange_quantity <= 0:
            return False

        local_total = sum(
            abs(self.float_parser(getattr(position, "quantity", 0.0), 0.0))
            for position in open_positions
        )
        tolerance = max(local_total * 0.001, exchange_quantity * 0.001, 1e-8)
        changed = abs(local_total - exchange_quantity) > tolerance
        stop_loss = self.float_parser(stop_loss_price, 0.0)
        take_profit = self.float_parser(take_profit_price, 0.0)

        if changed:
            self._sync_quantity(open_positions, local_total, exchange_quantity)

        total_after = sum(
            abs(self.float_parser(getattr(position, "quantity", 0.0), 0.0))
            for position in open_positions
        )
        for position in open_positions:
            position.current_price = current_price
            if not getattr(position, "entry_price", 0.0) and entry_price > 0:
                position.entry_price = entry_price
            if leverage > 0:
                position.leverage = leverage
            if (
                stop_loss > 0
                and abs(
                    self.float_parser(getattr(position, "stop_loss_price", 0.0), 0.0) - stop_loss
                )
                > 1e-12
            ):
                position.stop_loss_price = stop_loss
                changed = True
            if (
                take_profit > 0
                and abs(
                    self.float_parser(getattr(position, "take_profit_price", 0.0), 0.0)
                    - take_profit
                )
                > 1e-12
            ):
                position.take_profit_price = take_profit
                changed = True
            share = (
                abs(self.float_parser(getattr(position, "quantity", 0.0), 0.0)) / total_after
                if total_after > 0
                else 0.0
            )
            if exchange_unrealized:
                position.unrealized_pnl = exchange_unrealized * share
            elif getattr(position, "side", None) == "short":
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
            else:
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            position.updated_at = datetime.now(UTC)

        return changed

    def _sync_quantity(
        self,
        open_positions: list[Any],
        local_total: float,
        exchange_quantity: float,
    ) -> None:
        if len(open_positions) == 1:
            open_positions[0].quantity = exchange_quantity
            return

        ratio = exchange_quantity / local_total if local_total > 0 else 0.0
        remaining = exchange_quantity
        for position in open_positions[:-1]:
            new_quantity = max(
                abs(self.float_parser(getattr(position, "quantity", 0.0), 0.0)) * ratio,
                0.0,
            )
            position.quantity = new_quantity
            remaining -= new_quantity
        open_positions[-1].quantity = max(remaining, 0.0)
