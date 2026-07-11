from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select

from core.symbols import trading_symbol_variants
from db.repositories.base import BaseRepository
from models.trade import Order, Position


class TradeRepository(BaseRepository):
    """Repository for Orders and Positions."""

    model = Order

    async def create_order(self, data: dict) -> Order:
        order = Order(**data)
        self.session.add(order)
        await self.session.flush()
        return order

    async def update_order_status(
        self,
        order_id: int,
        status: str,
        exchange_order_id: str | None = None,
        filled_at: datetime | None = None,
        fee: float | None = None,
    ) -> Order | None:
        order = await self.get(order_id)
        if order:
            order.status = status
            if exchange_order_id:
                order.exchange_order_id = exchange_order_id
            if filled_at:
                order.filled_at = filled_at
            if fee is not None:
                order.fee = fee
            await self.session.flush()
        return order

    async def get_open_orders(
        self, model_name: str | None = None, symbol: str | None = None
    ) -> list[Order]:
        stmt = select(Order).where(Order.status.in_(["pending", "open", "partial"]))
        if model_name:
            stmt = stmt.where(Order.model_name == model_name)
        if symbol:
            stmt = stmt.where(Order.symbol == symbol)
        result = await self.session.execute(stmt.order_by(Order.created_at.desc()))
        return list(result.scalars().all())

    async def get_recent_orders(
        self,
        model_name: str | None = None,
        symbol: str | None = None,
        execution_mode: str | None = None,
        statuses: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Order]:
        stmt = (
            select(Order)
            .order_by(Order.created_at.desc())
            .offset(max(int(offset or 0), 0))
            .limit(limit)
        )
        if model_name:
            stmt = stmt.where(Order.model_name == model_name)
        if symbol:
            stmt = stmt.where(Order.symbol == symbol)
        if execution_mode:
            stmt = stmt.where(Order.execution_mode == execution_mode)
        if statuses:
            stmt = stmt.where(Order.status.in_(statuses))
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def open_position(self, data: dict) -> Position:
        data.setdefault("is_open", True)
        position = Position(**data)
        self.session.add(position)
        await self.session.flush()
        return position

    async def close_position(
        self, position_id: int, exit_price: float, realized_pnl: float
    ) -> Position | None:
        position = await self.session.get(Position, position_id)
        if position:
            position.is_open = False
            position.current_price = exit_price
            position.realized_pnl = realized_pnl
            position.closed_at = datetime.utcnow()
            await self.session.flush()
        return position

    async def get_open_positions(
        self, model_name: str | None = None, symbol: str | None = None
    ) -> list[Position]:
        stmt = select(Position).where(Position.is_open.is_(True))
        if model_name:
            stmt = stmt.where(Position.model_name == model_name)
        if symbol:
            stmt = stmt.where(Position.symbol == symbol)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_matching_open_positions(
        self,
        model_name: str,
        symbol: str,
        side: str,
        execution_mode: str,
    ) -> list[Position]:
        symbol_variants = trading_symbol_variants(symbol) or {symbol}
        stmt = select(Position).where(
            Position.model_name == model_name,
            Position.symbol.in_(symbol_variants),
            Position.side == side,
            Position.execution_mode == execution_mode,
            Position.is_open.is_(True),
        )
        result = await self.session.execute(stmt.order_by(Position.created_at.asc()))
        return list(result.scalars().all())

    async def get_exchange_matching_open_positions(
        self,
        symbol: str,
        side: str,
        execution_mode: str,
    ) -> list[Position]:
        symbol_variants = trading_symbol_variants(symbol) or {symbol}
        stmt = select(Position).where(
            Position.symbol.in_(symbol_variants),
            Position.side == side,
            Position.execution_mode == execution_mode,
            Position.is_open.is_(True),
        )
        result = await self.session.execute(stmt.order_by(Position.created_at.asc()))
        return list(result.scalars().all())

    async def get_position_records(
        self,
        execution_mode: str | None = None,
        model_name: str | None = None,
        symbol: str | None = None,
        limit: int = 500,
        offset: int = 0,
        is_open: bool | None = None,
    ) -> list[Position]:
        stmt = select(Position)
        if execution_mode:
            stmt = stmt.where(Position.execution_mode == execution_mode)
        if model_name:
            stmt = stmt.where(Position.model_name == model_name)
        if symbol:
            stmt = stmt.where(Position.symbol == symbol)
        if is_open is not None:
            stmt = stmt.where(Position.is_open.is_(is_open))
        stmt = (
            stmt.order_by(
                Position.closed_at.desc().nullslast(),
                Position.created_at.desc(),
            )
            .offset(max(int(offset or 0), 0))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        rows = list(result.scalars().all())
        return rows

    async def count_positions(
        self,
        execution_mode: str | None = None,
        model_name: str | None = None,
        symbol: str | None = None,
        is_open: bool | None = None,
    ) -> int:
        stmt = select(func.count(Position.id))
        if execution_mode:
            stmt = stmt.where(Position.execution_mode == execution_mode)
        if model_name:
            stmt = stmt.where(Position.model_name == model_name)
        if symbol:
            stmt = stmt.where(Position.symbol == symbol)
        if is_open is not None:
            stmt = stmt.where(Position.is_open.is_(is_open))
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def update_position_price(
        self, position_id: int, current_price: float, unrealized_pnl: float
    ) -> None:
        position = await self.session.get(Position, position_id)
        if position and position.is_open:
            position.current_price = current_price
            position.unrealized_pnl = unrealized_pnl
            await self.session.flush()

    async def update_open_position_prices(
        self,
        updates: list[tuple[Position, float, float]],
    ) -> int:
        """Flush price updates for already-loaded open positions as one unit of work."""

        changed = 0
        for position, current_price, unrealized_pnl in updates:
            if not position.is_open:
                continue
            position.current_price = current_price
            position.unrealized_pnl = unrealized_pnl
            changed += 1
        if changed:
            await self.session.flush()
        return changed

    async def count_orders(
        self,
        model_name: str | None = None,
        symbol: str | None = None,
        execution_mode: str | None = None,
        statuses: list[str] | None = None,
        require_exchange_order_id: bool = False,
    ) -> int:
        stmt = select(func.count(Order.id))
        if model_name:
            stmt = stmt.where(Order.model_name == model_name)
        if symbol:
            stmt = stmt.where(Order.symbol == symbol)
        if execution_mode:
            stmt = stmt.where(Order.execution_mode == execution_mode)
        if statuses:
            stmt = stmt.where(Order.status.in_(statuses))
        if require_exchange_order_id:
            stmt = stmt.where(Order.exchange_order_id.is_not(None), Order.exchange_order_id != "")
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def delete_all(self) -> int:
        """Delete all order records. Returns count of deleted rows."""
        from sqlalchemy import delete

        result = await self.session.execute(delete(Order))
        await self.session.flush()
        return result.rowcount

    async def get_daily_trade_pnl(self, model_name: str) -> float:
        """Sum realized PnL from today's closed positions."""
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        result = await self.session.execute(
            select(func.coalesce(func.sum(Position.realized_pnl), 0.0)).where(
                Position.model_name == model_name,
                Position.closed_at >= today,
                Position.is_open.is_(False),
            )
        )
        return result.scalar() or 0.0
