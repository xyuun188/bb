from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from services.strategy_context_performance import StrategyContextPerformanceService


@pytest.mark.asyncio
async def test_strategy_context_performance_uses_one_position_read_for_all_metrics() -> None:
    calls: list[dict[str, object]] = []
    now = datetime.now(UTC)
    row = SimpleNamespace(
        is_open=False,
        closed_at=now,
        created_at=now,
        realized_pnl=2.0,
        unrealized_pnl=0.0,
        symbol="BTC/USDT",
        side="long",
    )

    class Repository:
        def __init__(self, _session: object) -> None:
            pass

        async def get_position_records(self, **kwargs: object) -> list[object]:
            calls.append(kwargs)
            return [row]

    @asynccontextmanager
    async def session_factory():
        yield object()

    service = StrategyContextPerformanceService(
        session_factory=session_factory,
        trade_repository_factory=Repository,
    )

    result = await service.recent("paper")

    assert set(result) == {
        "daily_perf",
        "today_side_perf",
        "multiday_side_perf",
        "symbol_side_perf",
    }
    assert result["daily_perf"]["today_realized_pnl"] == 2.0
    assert result["today_side_perf"]["long"]["pnl"] == 2.0
    assert result["multiday_side_perf"]["long"]["pnl"] == 2.0
    assert result["symbol_side_perf"]["BTC/USDT|long"]["pnl"] == 2.0
    assert calls == [
        {
            "execution_mode": "paper",
            "model_name": "ensemble_trader",
            "limit": 5000,
        }
    ]
