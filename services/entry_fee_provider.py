from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import select

from executor.base_executor import OrderStatus
from models.trade import Order


def proportional_fee(fee: float | None, close_qty: float, total_qty: float) -> float:
    try:
        fee_value = abs(float(fee or 0.0))
        close_value = float(close_qty or 0.0)
        total_value = float(total_qty or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if fee_value <= 0 or close_value <= 0:
        return 0.0
    if total_value <= 0:
        return fee_value
    return fee_value * min(close_value / total_value, 1.0)


class EntryFeeProvider:
    """Find and prorate the entry order fee for a closing position."""

    @staticmethod
    def proportional_fee(fee: float | None, close_qty: float, total_qty: float) -> float:
        return proportional_fee(fee, close_qty, total_qty)

    async def entry_fee_for_position(self, session: Any, position: Any, close_qty: float) -> float:
        entry_side = "buy" if position.side == "long" else "sell"
        created_at = position.created_at
        statement = select(Order).where(
            Order.model_name == position.model_name,
            Order.execution_mode == position.execution_mode,
            Order.symbol == position.symbol,
            Order.side == entry_side,
            Order.status == OrderStatus.FILLED.value,
        )
        if created_at:
            window_start = created_at - timedelta(seconds=90)
            window_end = created_at + timedelta(seconds=90)
            window_statement = statement.where(
                Order.created_at >= window_start,
                Order.created_at <= window_end,
            )
            order = await self._first_order(
                session,
                window_statement.order_by(Order.created_at.asc()).limit(1),
            )
            if order:
                return proportional_fee(order.fee, close_qty, order.quantity)

            order = await self._first_order(
                session,
                statement.where(Order.created_at <= created_at)
                .order_by(Order.created_at.desc())
                .limit(1),
            )
            if order:
                return proportional_fee(order.fee, close_qty, order.quantity)

        order = await self._first_order(
            session,
            statement.order_by(Order.created_at.desc()).limit(1),
        )
        if order:
            return proportional_fee(order.fee, close_qty, order.quantity)
        return 0.0

    @staticmethod
    async def _first_order(session: Any, statement: Any) -> Any | None:
        result = await session.execute(statement)
        return result.scalar_one_or_none()
