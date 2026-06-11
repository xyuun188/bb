from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from services.daily_performance_service import DailyPerformanceService


@pytest.mark.asyncio
async def test_daily_performance_state_uses_today_closed_and_open_unrealized() -> None:
    now = datetime(2026, 6, 9, 8, 0, tzinfo=UTC)
    rows = [
        SimpleNamespace(
            is_open=False,
            realized_pnl=10.0,
            unrealized_pnl=0.0,
            closed_at=now - timedelta(hours=1),
        ),
        SimpleNamespace(
            is_open=False,
            realized_pnl=-3.0,
            unrealized_pnl=0.0,
            closed_at=now,
        ),
        SimpleNamespace(
            is_open=True,
            realized_pnl=0.0,
            unrealized_pnl=2.0,
            closed_at=None,
        ),
        SimpleNamespace(
            is_open=False,
            realized_pnl=100.0,
            unrealized_pnl=0.0,
            closed_at=now - timedelta(days=2),
        ),
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

    service = DailyPerformanceService(
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
            "limit": 5000,
        }
    ]
    assert state["today_realized_pnl"] == pytest.approx(7.0)
    assert state["today_realized_profit"] == pytest.approx(10.0)
    assert state["today_realized_loss"] == pytest.approx(3.0)
    assert state["open_unrealized_pnl"] == pytest.approx(2.0)
    assert state["today_total_pnl"] == pytest.approx(9.0)
    assert state["today_high_water_pnl"] == pytest.approx(10.0)
    assert state["today_trade_count"] == pytest.approx(2.0)
