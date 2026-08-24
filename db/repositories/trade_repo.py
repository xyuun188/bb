from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select, text

from core.symbols import trading_symbol_variants
from db.repositories.base import BaseRepository
from models.decision import AIDecision
from models.trade import Order, Position


class TradeRepository(BaseRepository):
    """Repository for Orders and Positions."""

    model = Order

    async def create_order(self, data: dict) -> Order:
        order, _created = await self.create_order_fact(data)
        return order

    async def create_order_fact(self, data: dict) -> tuple[Order, bool]:
        """Create one exchange fact or reuse the row that already owns it."""

        # An exchange order id is the authoritative execution identity.  The
        # execution result and the recovery/sync path can both arrive for the
        # same fill, so always reuse the existing fact before inserting a new
        # projection.  The database migration adds a unique partial index; the
        # locked lookup keeps this boundary idempotent while older databases
        # are being repaired.
        exchange_order_id = str(data.get("exchange_order_id") or "").strip()
        execution_mode = str(data.get("execution_mode") or "").strip()
        if exchange_order_id and execution_mode and exchange_order_id not in {
            "hold",
            "rejected",
            "no_position",
        }:
            bind = self.session.get_bind()
            if bind is not None and bind.dialect.name == "postgresql":
                lock_identity = f"order-fact:{execution_mode}:{exchange_order_id}"
                await self.session.execute(
                    text("SELECT pg_advisory_xact_lock(hashtextextended(:identity, 0))"),
                    {"identity": lock_identity},
                )
            result = await self.session.execute(
                select(Order)
                .where(
                    Order.execution_mode == execution_mode,
                    Order.exchange_order_id == exchange_order_id,
                )
                .order_by(Order.id.asc())
                .with_for_update()
                .limit(1)
            )
            existing = result.scalar_one_or_none()
            if existing is not None:
                self._merge_order_fact(existing, data)
                await self._prefer_authoritative_decision(existing, data)
                await self.session.flush()
                return existing, False
        order = Order(**data)
        self.session.add(order)
        await self.session.flush()
        return order, True

    @staticmethod
    def _merge_order_fact(order: Order, data: dict) -> None:
        """Merge richer recovery data without replacing the original identity."""

        for field in (
            "okx_inst_id",
            "okx_trade_ids",
            "okx_fill_contracts",
            "okx_fill_pnl",
            "okx_state",
            "okx_sync_status",
            "okx_synced_at",
            "okx_last_error",
            "okx_raw_fills",
        ):
            incoming = data.get(field)
            if incoming is None or incoming == {}:
                continue
            current = getattr(order, field, None)
            if current in (None, "", {}, []):
                setattr(order, field, incoming)
                continue
            if field == "okx_raw_fills" and isinstance(current, dict) and isinstance(incoming, dict):
                setattr(order, field, {**current, **incoming})
        incoming_status = str(data.get("status") or "").lower()
        current_status = str(getattr(order, "status", "") or "").lower()
        status_rank = {"rejected": 0, "pending": 1, "open": 2, "partial": 3, "filled": 4}
        if status_rank.get(incoming_status, -1) > status_rank.get(current_status, -1):
            order.status = incoming_status
        for field in ("quantity", "price", "fee", "filled_at"):
            incoming = data.get(field)
            current = getattr(order, field, None)
            if incoming is not None and current in (None, 0, 0.0, ""):
                setattr(order, field, incoming)

    async def _prefer_authoritative_decision(self, order: Order, data: dict) -> None:
        incoming_id = int(data.get("decision_id") or 0)
        existing_id = int(getattr(order, "decision_id", 0) or 0)
        if incoming_id <= 0 or incoming_id == existing_id:
            return
        if existing_id <= 0:
            order.decision_id = incoming_id
            return
        existing = await self.session.get(AIDecision, existing_id)
        incoming = await self.session.get(AIDecision, incoming_id)
        if incoming is None:
            return
        existing_raw = (
            existing.raw_llm_response
            if existing is not None and isinstance(existing.raw_llm_response, dict)
            else {}
        )
        incoming_raw = (
            incoming.raw_llm_response if isinstance(incoming.raw_llm_response, dict) else {}
        )
        existing_is_sync = existing is None or existing_raw.get("system_sync") is True
        incoming_is_sync = incoming_raw.get("system_sync") is True
        if existing_is_sync and not incoming_is_sync:
            order.decision_id = incoming_id

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

    async def get_matching_open_positions_for_update(
        self,
        model_name: str,
        symbol: str,
        side: str,
        execution_mode: str,
    ) -> list[Position]:
        """Lock one local position lifecycle group before an exit submission."""

        symbol_variants = trading_symbol_variants(symbol) or {symbol}
        stmt = (
            select(Position)
            .where(
                Position.model_name == model_name,
                Position.symbol.in_(symbol_variants),
                Position.side == side,
                Position.execution_mode == execution_mode,
                Position.is_open.is_(True),
            )
            .order_by(Position.created_at.asc())
            .with_for_update()
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_positions_for_update(self, position_ids: tuple[int, ...]) -> list[Position]:
        if not position_ids:
            return []
        stmt = (
            select(Position)
            .where(Position.id.in_(position_ids))
            .order_by(Position.id.asc())
            .with_for_update()
        )
        result = await self.session.execute(stmt)
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
