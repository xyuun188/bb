"""One-read performance context for dynamic entry strategy selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from config.settings import ENSEMBLE_TRADER_NAME
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx
from services.daily_performance_service import DailyPerformanceService
from services.daily_side_performance import DailySidePerformanceService
from services.symbol_side_performance import SymbolSidePerformanceService

SessionFactory = Callable[[], Any]
TradeRepositoryFactory = Callable[[Any], TradeRepository]

STRATEGY_CONTEXT_POSITION_LIMIT = 5000


class StrategyContextPerformanceService:
    """Derive all position-based strategy context from one bounded repository read."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory = get_session_ctx,
        trade_repository_factory: TradeRepositoryFactory = TradeRepository,
        model_name: str = ENSEMBLE_TRADER_NAME,
        position_limit: int = STRATEGY_CONTEXT_POSITION_LIMIT,
        daily_performance: DailyPerformanceService | None = None,
        daily_side_performance: DailySidePerformanceService | None = None,
        symbol_side_performance: SymbolSidePerformanceService | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._trade_repository_factory = trade_repository_factory
        self._model_name = model_name
        self._position_limit = max(int(position_limit), 1)
        self._daily_performance = daily_performance or DailyPerformanceService()
        self._daily_side_performance = daily_side_performance or DailySidePerformanceService()
        self._symbol_side_performance = symbol_side_performance or SymbolSidePerformanceService()

    async def recent(self, mode: str) -> dict[str, dict[str, Any]]:
        selected_mode = "live" if mode == "live" else "paper"
        async with self._session_factory() as session:
            rows = await self._trade_repository_factory(session).get_position_records(
                execution_mode=selected_mode,
                model_name=self._model_name,
                limit=self._position_limit,
            )
        return {
            "daily_perf": self._daily_performance.build_state(rows),
            "today_side_perf": self._daily_side_performance.build(rows),
            "multiday_side_perf": self._daily_side_performance.build_window(rows, lookback_days=5.0),
            "symbol_side_perf": self._symbol_side_performance.build_profiles(rows),
        }
