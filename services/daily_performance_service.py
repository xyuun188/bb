"""Daily PnL telemetry used by strategy posture selection."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from config.settings import ENSEMBLE_TRADER_NAME
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx
from services.trade_fact_trust import closed_position_trade_fact_trusted

SessionFactory = Callable[[], Any]
TradeRepositoryFactory = Callable[[Any], TradeRepository]


class DailyPerformanceService:
    """Calculates daily performance without reintroducing daily profit targets."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory = get_session_ctx,
        trade_repository_factory: TradeRepositoryFactory = TradeRepository,
        model_name: str = ENSEMBLE_TRADER_NAME,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._trade_repository_factory = trade_repository_factory
        self._model_name = model_name
        self._clock = clock or (lambda: datetime.now(timezone(timedelta(hours=8))))

    async def state(self, mode: str) -> dict[str, float]:
        """Return today's realized and open PnL for the selected execution mode."""

        selected_mode = "live" if mode == "live" else "paper"
        async with self._session_factory() as session:
            rows = await self._trade_repository_factory(session).get_position_records(
                execution_mode=selected_mode,
                model_name=self._model_name,
                limit=5000,
            )
        return self.build_state(rows)

    def build_state(self, rows: list[Any]) -> dict[str, float]:
        """Build the daily posture state from an already-loaded position snapshot."""

        now_local = self._clock()
        if now_local.tzinfo is None:
            now_local = now_local.replace(tzinfo=timezone(timedelta(hours=8)))
        else:
            now_local = now_local.astimezone(timezone(timedelta(hours=8)))
        start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        start_utc = start_local.astimezone(UTC)
        realized_profit = 0.0
        realized_loss = 0.0
        trade_count = 0
        high_water = 0.0
        open_unrealized = 0.0

        closed_today = []
        for pos in rows:
            if pos.is_open:
                open_unrealized += float(pos.unrealized_pnl or 0.0)
                continue
            closed_at = pos.closed_at
            if not closed_at:
                continue
            if closed_at.tzinfo is None:
                closed_at = closed_at.replace(tzinfo=UTC)
            if closed_at < start_utc:
                continue
            if not closed_position_trade_fact_trusted(pos):
                continue
            closed_today.append(pos)

        closed_today.sort(key=lambda p: p.closed_at or datetime.min)
        running = 0.0
        for pos in closed_today:
            pnl = float(pos.realized_pnl or 0.0)
            running += pnl
            high_water = max(high_water, running)
            trade_count += 1
            if pnl >= 0:
                realized_profit += pnl
            else:
                realized_loss += abs(pnl)

        realized_pnl = realized_profit - realized_loss
        today_total = realized_pnl + open_unrealized
        high_water = max(high_water, today_total)
        return {
            "today_total_pnl": today_total,
            "today_realized_pnl": realized_pnl,
            "today_realized_profit": realized_profit,
            "today_realized_loss": realized_loss,
            "today_trade_count": float(trade_count),
            "today_high_water_pnl": high_water,
            "open_unrealized_pnl": open_unrealized,
        }
