from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from services.symbol_side_performance import SymbolSidePerformanceService


@pytest.mark.asyncio
async def test_symbol_side_performance_loads_and_builds_recent_profiles() -> None:
    now = datetime(2026, 6, 9, 12, 0, tzinfo=UTC)
    rows = [
        SimpleNamespace(
            symbol="BTC-USDT",
            side="long",
            realized_pnl=-20.0,
            closed_at=now - timedelta(hours=1),
        ),
        SimpleNamespace(
            symbol="BTC-USDT",
            side="long",
            realized_pnl=-10.0,
            closed_at=now - timedelta(hours=2),
        ),
        SimpleNamespace(
            symbol="BTC-USDT",
            side="short",
            realized_pnl=6.0,
            closed_at=now - timedelta(days=1),
        ),
        SimpleNamespace(
            symbol="BTC-USDT",
            side="long",
            realized_pnl=999.0,
            closed_at=now - timedelta(days=20),
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

    service = SymbolSidePerformanceService(
        session_factory=session_factory,
        trade_repository_factory=FakeTradeRepository,
        normalize_symbol=lambda symbol: str(symbol or "").replace("-", "/"),
        model_name="ensemble_trader",
        lookback_limit=100,
        lookback_days=7.0,
        clock=lambda: now,
    )

    profiles = await service.recent("paper")

    assert calls == [
        {
            "execution_mode": "paper",
            "model_name": "ensemble_trader",
            "is_open": False,
            "limit": 100,
        }
    ]
    long_profile = profiles["BTC/USDT|long"]
    assert long_profile["count"] == 2
    assert long_profile["losses"] == 2
    assert long_profile["pnl"] == pytest.approx(-30.0)
    assert long_profile["today_loss"] == pytest.approx(30.0)
    assert long_profile["cooldown"] is True
    assert long_profile["cooldown_reason"] == "该币种这个方向的真实亏损已经超过限制"
    assert long_profile["cooldown_remaining_hours"] == pytest.approx(5.0)
    assert long_profile["largest_loss"] == pytest.approx(-20.0)
    assert profiles["BTC/USDT|short"]["cooldown"] is False
    assert profiles["BTC/USDT|all"]["count"] == 3
    assert profiles["BTC/USDT|all"]["pnl"] == pytest.approx(-24.0)


def test_symbol_side_performance_expires_time_based_cooldown_at_boundary() -> None:
    now = datetime(2026, 6, 9, 12, 0, tzinfo=UTC)
    service = SymbolSidePerformanceService(clock=lambda: now, lookback_days=7.0)

    profiles = service.build_profiles(
        [
            SimpleNamespace(
                symbol="BTC-USDT",
                side="long",
                realized_pnl=-30.0,
                closed_at=now - timedelta(hours=6),
            )
        ]
    )

    long_profile = profiles["BTC-USDT|long"]
    assert long_profile["cooldown"] is False
    assert long_profile["cooldown_time_based"] is False
    assert long_profile["cooldown_remaining_hours"] == 0.0


def test_symbol_side_performance_build_profiles_ignores_stale_or_empty_rows() -> None:
    now = datetime(2026, 6, 9, 12, 0, tzinfo=UTC)
    service = SymbolSidePerformanceService(clock=lambda: now, lookback_days=1.0)

    profiles = service.build_profiles(
        [
            SimpleNamespace(
                symbol="ETH/USDT",
                side="long",
                realized_pnl=-5.0,
                closed_at=now - timedelta(days=2),
            ),
            SimpleNamespace(
                symbol="",
                side="long",
                realized_pnl=5.0,
                closed_at=now,
            ),
            SimpleNamespace(
                symbol="SOL/USDT",
                side="long",
                realized_pnl=5.0,
                closed_at=None,
            ),
        ]
    )

    assert profiles == {}
