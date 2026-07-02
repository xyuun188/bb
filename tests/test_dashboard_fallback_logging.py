from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from web_dashboard.api import dashboard

FAKE_BEARER_ERROR = "Authorization: Bearer " + "dashboard-balance-secret failed"


class FailingExecutor:
    async def get_positions_strict(self) -> list[dict[str, Any]]:
        raise RuntimeError("exchange unavailable")


class FailingBalanceExecutor(FailingExecutor):
    async def get_balance_snapshot(self, currency: str) -> dict[str, Any]:
        raise RuntimeError(FAKE_BEARER_ERROR)


class FailingRestClient:
    async def fetch_tickers(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        raise RuntimeError(f"ticker unavailable: {len(symbols)}")


class SuccessfulRestClient:
    def __init__(self) -> None:
        self.closed = False

    async def fetch_tickers(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        return {
            symbol: {
                "symbol": f"{symbol}:USDT",
                "last": 100.0,
                "percentage": 1.5,
                "baseVolume": 1234.0,
                "bid": 99.9,
                "ask": 100.1,
                "info": {"instId": symbol.replace("/", "-") + "-SWAP", "sodUtc8": "98.5"},
            }
            for symbol in symbols
        }

    async def close(self) -> None:
        self.closed = True


class PartiallyFailingRestClient:
    async def fetch_tickers(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        raise RuntimeError(f"bad batch symbol: {','.join(symbols)}")

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        if symbol == "UNI/USDT":
            raise RuntimeError("okx does not have market symbol UNI/USDT:USDT")
        return {
            "symbol": f"{symbol}:USDT",
            "last": 25000.0,
            "percentage": 2.0,
            "baseVolume": 4321.0,
            "bid": 24999.0,
            "ask": 25001.0,
            "info": {"instId": symbol.replace("/", "-") + "-SWAP", "sodUtc8": "24500"},
        }


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


class PositionExecutor:
    def __init__(self, positions: list[dict[str, Any]]) -> None:
        self.positions = positions

    async def get_positions_strict(self) -> list[dict[str, Any]]:
        return self.positions


class PositionTradingService:
    def __init__(self, positions: list[dict[str, Any]]) -> None:
        self.okx_paper = PositionExecutor(positions)
        self.okx_live = None

    def okx_executor_for_dashboard(self, mode: str) -> Any | None:
        return self.okx_live if mode == "live" else self.okx_paper


class FakePartialDataService:
    rest_client = PartiallyFailingRestClient()


class SuccessfulStandaloneBalanceExecutor:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.closed = False

    async def initialize(self) -> None:
        return None

    async def get_balance_snapshot(self, currency: str) -> dict[str, Any]:
        return {
            "free": 5.0,
            "used": 1.0,
            "total": 6.0,
            "cash": 6.0,
            "equity": 7.0,
            "allocatable": 7.0,
        }

    async def shutdown(self) -> None:
        self.closed = True


class FailingStandaloneBalanceExecutor(SuccessfulStandaloneBalanceExecutor):
    async def get_balance_snapshot(self, currency: str) -> dict[str, Any]:
        raise RuntimeError("standalone balance unavailable")


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
            "event": "exchange open position symbols strict read unavailable",
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
            "event": "exchange mark map strict read unavailable",
            "error": "exchange unavailable",
            "mode": "paper",
            "has_cached": True,
        }
    ]


async def test_exchange_mark_map_uses_short_timeout(
    monkeypatch: pytest.MonkeyPatch,
    dashboard_fallback_events: list[dict[str, Any]],
) -> None:
    stale_at = datetime.now(UTC) - timedelta(minutes=1)
    cached_mark = {("BTC/USDT", "long"): {"mark_price": 100.0}}
    waits: list[float] = []

    async def timeout_wait_for(awaitable: Any, **kwargs: Any) -> Any:
        waits.append(float(kwargs["timeout"]))
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise TimeoutError

    monkeypatch.setattr(dashboard, "_trading_service", FakeTradingService())
    monkeypatch.setattr(dashboard.asyncio, "wait_for", timeout_wait_for)
    monkeypatch.setattr(
        dashboard,
        "_exchange_mark_cache",
        {"paper": (stale_at, cached_mark)},
    )

    result = await dashboard._get_exchange_position_mark_map("paper")

    assert result == cached_mark
    assert waits == [dashboard._DASHBOARD_OKX_POSITION_READ_TIMEOUT_SECONDS]
    assert dashboard_fallback_events[0]["event"] == "exchange mark map strict read unavailable"
    assert dashboard_fallback_events[0]["has_cached"] is True


async def test_exchange_mark_map_uses_okx_info_markpx_upl_and_contract_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dashboard, "_exchange_mark_cache", {})
    monkeypatch.setattr(
        dashboard,
        "_trading_service",
        PositionTradingService(
            [
                {
                    "symbol": "PROS/USDT:USDT",
                    "side": "long",
                    "contracts": 0,
                    "markPrice": 0,
                    "entryPrice": 0,
                    "info": {
                        "instId": "PROS-USDT-SWAP",
                        "pos": "46",
                        "ctVal": "1",
                        "avgPx": "0.4054",
                        "markPx": "0.4059",
                        "last": "0.5547",
                        "upl": "-0.82",
                    },
                }
            ]
        ),
    )

    result = await dashboard._get_exchange_position_mark_map("paper")
    snapshot = result[("PROS/USDT", "long")]

    assert snapshot["mark_price"] == pytest.approx(0.4059)
    assert snapshot["last_price"] == pytest.approx(0.5547)
    assert snapshot["entry_price"] == pytest.approx(0.4054)
    assert snapshot["quantity"] == pytest.approx(46.0)
    assert snapshot["upl"] == pytest.approx(-0.82)
    valuation = dashboard._exchange_position_display_valuation(
        snapshot,
        "long",
        fallback_current_price=0.5547,
        fallback_unrealized_pnl=6.8678,
        fallback_entry_price=0.4054,
        fallback_quantity=46,
    )
    assert valuation["current_price"] == pytest.approx(0.4059)
    assert valuation["unrealized_pnl"] == pytest.approx(-0.82)
    assert valuation["pnl_source"] == "okx_position_upl"


async def test_open_position_symbols_do_not_use_paper_executor_memory_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PaperExecutor:
        async def get_positions(self) -> list[dict[str, Any]]:
            raise AssertionError("dashboard open symbols must use OKX strict snapshots")

    class TradingService:
        paper_executor = PaperExecutor()

        def okx_executor_for_dashboard(self, mode: str) -> Any | None:
            return PositionExecutor(
                [
                    {
                        "symbol": "OKX/USDT:USDT",
                        "side": "long",
                        "contracts": 1,
                        "info": {
                            "instId": "OKX-USDT-SWAP",
                            "pos": "1",
                            "ctVal": "1",
                            "avgPx": "1",
                            "markPx": "1.1",
                        },
                    }
                ]
            )

    monkeypatch.setattr(dashboard, "_trading_service", TradingService())
    monkeypatch.setattr(dashboard, "_exchange_open_symbol_cache", {})
    monkeypatch.setattr(dashboard, "_dashboard_okx_position_cache", {})

    result = await dashboard._get_open_position_symbols("paper")

    assert result == {"OKX/USDT"}


async def test_open_position_ticker_prefers_okx_mark_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exchange_marks(mode: str | None = None) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            ("PROS/USDT", "long"): {
                "mark_price": 0.4059,
                "last_price": 0.5547,
                "entry_price": 0.4054,
                "upl": -0.82,
                "quantity": 46.0,
            }
        }

    async def public_tickers(symbols: set[str]) -> dict[str, dict[str, Any]]:
        return {}

    monkeypatch.setattr(dashboard, "_get_exchange_position_mark_map", exchange_marks)
    monkeypatch.setattr(dashboard, "_get_public_ticker_map", public_tickers)

    tickers = await dashboard._build_tickers_for_open_positions(
        {"PROS/USDT"},
        {"PROS/USDT": {"price": 0.5547, "change_24h": 1.0}},
        "paper",
    )

    assert tickers["PROS/USDT"]["price"] == pytest.approx(0.4059)
    assert tickers["PROS/USDT"]["mark_price"] == pytest.approx(0.4059)


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


async def test_public_ticker_bad_symbol_does_not_drop_valid_symbols(
    monkeypatch: pytest.MonkeyPatch,
    dashboard_fallback_events: list[dict[str, Any]],
) -> None:
    monkeypatch.setattr(dashboard, "_data_service", FakePartialDataService())
    monkeypatch.setattr(dashboard, "_public_ticker_cache", {})

    result = await dashboard._get_public_ticker_map({"BTC/USDT", "UNI/USDT"})

    assert set(result) == {"BTC/USDT"}
    assert result["BTC/USDT"]["price"] == 25000.0
    assert result["BTC/USDT"]["volume_24h"] == 4321.0
    assert dashboard_fallback_events == [
        {
            "event": "public ticker fallback",
            "error": "bad batch symbol: BTC/USDT,UNI/USDT",
            "symbol_count": 2,
            "has_cached": False,
        },
        {
            "event": "public ticker symbol fallback",
            "error": "okx does not have market symbol UNI/USDT:USDT",
            "symbol": "UNI/USDT",
        },
    ]


async def test_public_ticker_map_works_without_data_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dashboard, "_data_service", None)
    monkeypatch.setattr(dashboard, "_public_ticker_cache", {})
    monkeypatch.setattr(dashboard, "OKXRestClient", SuccessfulRestClient)

    result = await dashboard._get_public_ticker_map({"BTC/USDT"})

    assert result["BTC/USDT"]["price"] == 100.0
    assert result["BTC/USDT"]["change_24h"] == pytest.approx(1.5)
    assert result["BTC/USDT"]["volume_24h"] == 1234.0


async def test_dashboard_okx_balance_snapshot_fallback_logs(
    monkeypatch: pytest.MonkeyPatch,
    dashboard_fallback_events: list[dict[str, Any]],
) -> None:
    monkeypatch.setattr(dashboard, "_trading_service", FakeBalanceTradingService())
    monkeypatch.setattr(dashboard, "_dashboard_okx_balance_cache", {})
    monkeypatch.setattr(dashboard, "_dashboard_okx_balance_error_cache", {})
    monkeypatch.setattr(dashboard, "OKXExecutor", SuccessfulStandaloneBalanceExecutor)

    result = await dashboard._get_dashboard_okx_account_snapshot("paper")

    assert result == {
        "free": 5.0,
        "used": 1.0,
        "total": 6.0,
        "cash": 6.0,
        "equity": 7.0,
        "allocatable": 7.0,
    }
    assert dashboard_fallback_events == [
        {
            "event": "dashboard summary okx balance fallback",
            "error": FAKE_BEARER_ERROR,
            "mode": "paper",
            "source": "trading_service_executor",
        }
    ]


async def test_dashboard_okx_balance_snapshot_logs_standalone_failure(
    monkeypatch: pytest.MonkeyPatch,
    dashboard_fallback_events: list[dict[str, Any]],
) -> None:
    monkeypatch.setattr(dashboard, "_trading_service", FakeBalanceTradingService())
    monkeypatch.setattr(dashboard, "_dashboard_okx_balance_cache", {})
    monkeypatch.setattr(dashboard, "_dashboard_okx_balance_error_cache", {})
    monkeypatch.setattr(dashboard, "OKXExecutor", FailingStandaloneBalanceExecutor)

    result = await dashboard._get_dashboard_okx_account_snapshot("paper")

    assert result == {
        "error": "OKX 余额读取失败：standalone balance unavailable",
        "balance_error": "OKX 余额读取失败：standalone balance unavailable",
        "balance_source": "OKX 模拟盘账户",
        "source": "isolated_executor",
        "error_cached": True,
    }
    assert dashboard_fallback_events == [
        {
            "event": "dashboard summary okx balance fallback",
            "error": FAKE_BEARER_ERROR,
            "mode": "paper",
            "source": "trading_service_executor",
        },
        {
            "event": "dashboard summary okx balance fallback",
            "error": "standalone balance unavailable",
            "mode": "paper",
            "source": "isolated_executor",
        },
    ]


async def test_dashboard_okx_balance_failure_cache_prevents_retry(
    monkeypatch: pytest.MonkeyPatch,
    dashboard_fallback_events: list[dict[str, Any]],
) -> None:
    class CountingFailingStandaloneBalanceExecutor(FailingStandaloneBalanceExecutor):
        created = 0

        def __init__(self, mode: str) -> None:
            type(self).created += 1
            super().__init__(mode)

    monkeypatch.setattr(dashboard, "_trading_service", FakeBalanceTradingService())
    monkeypatch.setattr(dashboard, "_dashboard_okx_balance_cache", {})
    monkeypatch.setattr(dashboard, "_dashboard_okx_balance_error_cache", {})
    monkeypatch.setattr(dashboard, "OKXExecutor", CountingFailingStandaloneBalanceExecutor)

    first = await dashboard._get_dashboard_okx_account_snapshot("paper")
    second = await dashboard._get_dashboard_okx_account_snapshot("paper")

    assert first == second
    assert first["error_cached"] is True
    assert CountingFailingStandaloneBalanceExecutor.created == 1
    assert [event["source"] for event in dashboard_fallback_events] == [
        "trading_service_executor",
        "isolated_executor",
    ]


async def test_dashboard_okx_position_cache_is_bound_to_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SwitchingTradingService:
        def __init__(self) -> None:
            self.positions = [
                {
                    "symbol": "OLD/USDT:USDT",
                    "side": "long",
                    "contracts": 1,
                    "info": {"instId": "OLD-USDT-SWAP", "pos": "1"},
                }
            ]

        def okx_executor_for_dashboard(self, mode: str) -> Any | None:
            return PositionExecutor(self.positions)

    service = SwitchingTradingService()
    monkeypatch.setattr(dashboard, "_trading_service", service)
    monkeypatch.setattr(dashboard, "_dashboard_okx_position_cache", {})
    monkeypatch.setattr(dashboard, "_exchange_open_symbol_cache", {})

    first = await dashboard._get_exchange_open_position_symbols("paper")
    service.positions = [
        {
            "symbol": "NEW/USDT:USDT",
            "side": "long",
            "contracts": 1,
            "info": {"instId": "NEW-USDT-SWAP", "pos": "1"},
        }
    ]
    dashboard._exchange_open_symbol_cache.clear()

    second = await dashboard._get_exchange_open_position_symbols("paper")

    assert first == {"OLD/USDT"}
    assert second == {"NEW/USDT"}
