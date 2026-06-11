from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from web_dashboard.api import dashboard

FAKE_BEARER_ERROR = "Authorization: Bearer " + "dashboard-balance-secret failed"


class FailingExecutor:
    async def get_positions(self) -> list[dict[str, Any]]:
        raise RuntimeError("exchange unavailable")


class FailingBalanceExecutor(FailingExecutor):
    async def get_balance_snapshot(self, currency: str) -> dict[str, Any]:
        raise RuntimeError(FAKE_BEARER_ERROR)


class FailingRestClient:
    async def fetch_tickers(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        raise RuntimeError(f"ticker unavailable: {len(symbols)}")


class FakeTradingService:
    def __init__(self) -> None:
        self.okx_paper = FailingExecutor()
        self.okx_live = None

    def okx_executor_for_dashboard(self, mode: str) -> Any | None:
        return self.okx_live if mode == "live" else self.okx_paper


class FakeBalanceTradingService:
    def __init__(self) -> None:
        self.okx_paper = FailingBalanceExecutor()
        self.okx_live = None

    def okx_executor_for_dashboard(self, mode: str) -> Any | None:
        return self.okx_live if mode == "live" else self.okx_paper


class FakeDataService:
    rest_client = FailingRestClient()


@pytest.fixture
def dashboard_fallback_events(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    def record(event: str, exc: Exception, **fields: Any) -> None:
        events.append(
            {
                "event": event,
                "error": str(exc),
                **fields,
            }
        )

    monkeypatch.setattr(dashboard, "_log_dashboard_fallback", record)
    return events


async def test_exchange_open_symbols_fallback_logs_and_uses_stale_cache(
    monkeypatch: pytest.MonkeyPatch,
    dashboard_fallback_events: list[dict[str, Any]],
) -> None:
    stale_at = datetime.now(UTC) - timedelta(minutes=1)
    monkeypatch.setattr(dashboard, "_trading_service", FakeTradingService())
    monkeypatch.setattr(
        dashboard,
        "_exchange_open_symbol_cache",
        {"paper": (stale_at, {"BTC/USDT"})},
    )

    result = await dashboard._get_exchange_open_position_symbols("paper")

    assert result == {"BTC/USDT"}
    assert dashboard_fallback_events == [
        {
            "event": "exchange open position symbols fallback",
            "error": "exchange unavailable",
            "mode": "paper",
            "has_cached": True,
        }
    ]


async def test_exchange_mark_map_fallback_logs_and_uses_stale_cache(
    monkeypatch: pytest.MonkeyPatch,
    dashboard_fallback_events: list[dict[str, Any]],
) -> None:
    stale_at = datetime.now(UTC) - timedelta(minutes=1)
    cached_mark = {("BTC/USDT", "long"): {"mark_price": 100.0}}
    monkeypatch.setattr(dashboard, "_trading_service", FakeTradingService())
    monkeypatch.setattr(
        dashboard,
        "_exchange_mark_cache",
        {"paper": (stale_at, cached_mark)},
    )

    result = await dashboard._get_exchange_position_mark_map("paper")

    assert result == cached_mark
    assert dashboard_fallback_events == [
        {
            "event": "exchange mark map fallback",
            "error": "exchange unavailable",
            "mode": "paper",
            "has_cached": True,
        }
    ]


async def test_public_ticker_fallback_logs_and_uses_stale_cache(
    monkeypatch: pytest.MonkeyPatch,
    dashboard_fallback_events: list[dict[str, Any]],
) -> None:
    stale_at = datetime.now(UTC) - timedelta(minutes=1)
    cached_tickers = {"BTC/USDT": {"price": 100.0}}
    monkeypatch.setattr(dashboard, "_data_service", FakeDataService())
    monkeypatch.setattr(
        dashboard,
        "_public_ticker_cache",
        {"BTC/USDT": (stale_at, cached_tickers)},
    )

    result = await dashboard._get_public_ticker_map({"BTC/USDT"})

    assert result == cached_tickers
    assert dashboard_fallback_events == [
        {
            "event": "public ticker fallback",
            "error": "ticker unavailable: 1",
            "symbol_count": 1,
            "has_cached": True,
        }
    ]


async def test_dashboard_okx_balance_snapshot_fallback_logs(
    monkeypatch: pytest.MonkeyPatch,
    dashboard_fallback_events: list[dict[str, Any]],
) -> None:
    monkeypatch.setattr(dashboard, "_trading_service", FakeBalanceTradingService())

    result = await dashboard._get_dashboard_okx_account_snapshot("paper")

    assert result is None
    assert dashboard_fallback_events == [
        {
            "event": "dashboard summary okx balance fallback",
            "error": FAKE_BEARER_ERROR,
            "mode": "paper",
        }
    ]
