from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta
from typing import Any

from sqlalchemy import select

from executor.base_executor import OrderStatus
from models.trade import Order


class ExchangeBackedPositionProvider:
    """Identify local positions that are backed by a filled exchange entry order."""

    def __init__(self, *, match_window_seconds: int = 30) -> None:
        self.match_window_seconds = match_window_seconds

    async def ids(self, session: Any, positions: Iterable[Any]) -> set[int]:
        backed_ids: set[int] = set()
        for position in positions:
            position_id = getattr(position, "id", None)
            if position_id is None:
                continue
            if await self._has_filled_exchange_entry_order(session, position):
                backed_ids.add(int(position_id))
        return backed_ids

    async def _has_filled_exchange_entry_order(self, session: Any, position: Any) -> bool:
        entry_side = "buy" if getattr(position, "side", None) == "long" else "sell"
        statement = select(Order.id).where(
            Order.model_name == getattr(position, "model_name", None),
            Order.execution_mode == getattr(position, "execution_mode", None),
            Order.symbol == getattr(position, "symbol", None),
            Order.side == entry_side,
            Order.status == OrderStatus.FILLED.value,
            Order.exchange_order_id.is_not(None),
            Order.exchange_order_id != "",
        )

        created_at = getattr(position, "created_at", None)
        if created_at:
            if created_at.tzinfo is not None:
                created_at = created_at.replace(tzinfo=None)
            window = timedelta(seconds=self.match_window_seconds)
            statement = statement.where(
                Order.created_at >= created_at - window,
                Order.created_at <= created_at + window,
            )

        result = await session.execute(statement.limit(1))
        return bool(result.scalar_one_or_none())
