from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from services.daily_side_performance import DailySidePerformanceService
from services.trading_service import TradingService


@pytest.mark.asyncio
async def test_daily_side_performance_state_splits_today_pnl_by_side() -> None:
    now = datetime(2026, 6, 9, 8, 0, tzinfo=UTC)
    rows = [
        SimpleNamespace(side="long", realized_pnl=10.0, closed_at=now - timedelta(hours=1)),
        SimpleNamespace(side="long", realized_pnl=-4.0, closed_at=now),
        SimpleNamespace(side="short", realized_pnl=6.0, closed_at=now),
        SimpleNamespace(side="short", realized_pnl=-2.0, closed_at=now - timedelta(hours=2)),
        SimpleNamespace(side="long", realized_pnl=100.0, closed_at=now - timedelta(days=2)),
        SimpleNamespace(side="short", realized_pnl=5.0, closed_at=None),
    ]
    calls: list[dict[str, Any]] = []

    class FakeTradeRepository:
        def __init__(self, _session: object) -> None:
            pass

        async def get_position_records(self, **kwargs: Any) -> list[Any]:
            calls.append(kwargs)
            return rows

    @asynccontextmanager
    async def session_factory():
        yield object()

    service = DailySidePerformanceService(
        session_factory=session_factory,
        trade_repository_factory=FakeTradeRepository,
        model_name="ensemble_trader",
        clock=lambda: now,
    )

    state = await service.state("paper")

    assert calls == [
        {
            "execution_mode": "paper",
            "model_name": "ensemble_trader",
            "is_open": False,
            "limit": 5000,
        }
    ]
    assert state["long"]["count"] == 2
    assert state["long"]["wins"] == 1
    assert state["long"]["losses"] == 1
    assert state["long"]["pnl"] == pytest.approx(6.0)
    assert state["long"]["avg_pnl"] == pytest.approx(3.0)
    assert state["long"]["win_rate"] == pytest.approx(0.5)
    assert state["short"]["count"] == 2
    assert state["short"]["pnl"] == pytest.approx(4.0)
    assert state["short"]["profit"] == pytest.approx(6.0)
    assert state["short"]["loss"] == pytest.approx(2.0)


def test_daily_side_performance_build_returns_empty_buckets_without_today_rows() -> None:
    now = datetime(2026, 6, 9, 8, 0, tzinfo=UTC)
    service = DailySidePerformanceService(clock=lambda: now)

    state = service.build(
        [
            SimpleNamespace(
                side="long",
                realized_pnl=10.0,
                closed_at=now - timedelta(days=3),
            )
        ]
    )

    assert state["long"]["count"] == 0
    assert state["long"]["pnl"] == 0.0
    assert state["long"]["win_rate"] == 0.0
    assert state["short"]["count"] == 0


@pytest.mark.asyncio
async def test_trading_service_today_side_performance_delegates_to_service() -> None:
    service = object.__new__(TradingService)
    calls: list[str] = []

    class FakeDailySidePerformance:
        async def state(self, mode: str):
            calls.append(mode)
            return {"long": {"pnl": 1.0}, "short": {"pnl": -1.0}}

    service.daily_side_performance_service = FakeDailySidePerformance()

    assert await service._today_side_performance("paper") == {
        "long": {"pnl": 1.0},
        "short": {"pnl": -1.0},
    }
    assert calls == ["paper"]
