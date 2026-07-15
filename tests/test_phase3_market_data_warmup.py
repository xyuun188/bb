from __future__ import annotations

from typing import Any

import pytest

from scripts.install_phase3_market_data_warmup_timer import render_service
from scripts.run_phase3_market_data_warmup import _parse_symbols, warm_market_data
from services.data_service import KLINE_PERSIST_TIMEFRAME_LIMITS


class FakeDataService:
    def __init__(self) -> None:
        self.ticker_calls: list[str] = []
        self.kline_calls: list[tuple[str, str, int]] = []

        class RestClient:
            async def close(self) -> None:
                return None

        self.rest_client = RestClient()

    async def _get_ticker_snapshot(self, symbol: str) -> dict[str, Any]:
        self.ticker_calls.append(symbol)
        return {"last_price": 100.0, "timestamp": 1_700_000_000_000}

    async def _fetch_and_persist_klines(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> tuple[str, list[list[float]]]:
        self.kline_calls.append((symbol, timeframe, limit))
        return timeframe, [[1_700_000_000_000, 1, 2, 0.5, 1.5, 10] for _ in range(limit)]


class FakeFeatureService:
    async def report(self, *, hours: int = 24, limit: int = 1000) -> dict[str, Any]:
        return {"status": "warning", "missing_features": ["news"], "stale_features": []}


async def ready_db_coverage(symbols: list[str]) -> dict[str, Any]:
    return {
        "available": True,
        "ticker_ready_count": len(symbols),
        "kline_timeframe_ready_counts": {
            timeframe: len(symbols) for timeframe in KLINE_PERSIST_TIMEFRAME_LIMITS
        },
        "ticker_symbols": list(symbols),
        "kline_symbols_by_timeframe": {
            timeframe: list(symbols) for timeframe in KLINE_PERSIST_TIMEFRAME_LIMITS
        },
    }


async def empty_db_coverage(_symbols: list[str]) -> dict[str, Any]:
    return {
        "available": True,
        "ticker_ready_count": 0,
        "kline_timeframe_ready_counts": {
            timeframe: 0 for timeframe in KLINE_PERSIST_TIMEFRAME_LIMITS
        },
        "ticker_symbols": [],
        "kline_symbols_by_timeframe": {
            timeframe: [] for timeframe in KLINE_PERSIST_TIMEFRAME_LIMITS
        },
    }


def test_parse_symbols_normalizes_and_limits() -> None:
    assert _parse_symbols(["BTC-USDT-SWAP,ETH", "SOL/USDT"], limit=2) == [
        "BTC/USDT",
        "ETH/USDT",
    ]


@pytest.mark.asyncio
async def test_market_data_warmup_only_mutates_market_cache() -> None:
    data_service = FakeDataService()

    report = await warm_market_data(
        symbols=["BTC/USDT", "ETH/USDT"],
        symbol_limit=12,
        data_service=data_service,  # type: ignore[arg-type]
        feature_service=FakeFeatureService(),  # type: ignore[arg-type]
        db_coverage_loader=ready_db_coverage,
    )

    assert report["status"] == "ready"
    assert report["starts_trading_service"] is False
    assert report["submits_orders"] is False
    assert report["changes_model_routing"] is False
    assert report["changes_positions"] is False
    assert report["changes_orders"] is False
    assert report["mutation_scope"] == ["market_tickers", "market_klines"]
    assert report["db_verification_available"] is True
    assert data_service.ticker_calls == ["BTC/USDT", "ETH/USDT"]
    assert len(data_service.kline_calls) == 2 * len(KLINE_PERSIST_TIMEFRAME_LIMITS)
    assert report["feature_coverage_status_after_warmup"] == "warning"
    assert report["feature_missing_after_warmup"] == ["news"]


@pytest.mark.asyncio
async def test_market_data_warmup_blocks_when_core_market_fetch_fails() -> None:
    data_service = FakeDataService()

    async def empty_ticker(_symbol: str) -> dict[str, Any]:
        return {}

    async def empty_klines(_symbol: str, timeframe: str, _limit: int) -> tuple[str, list[Any]]:
        return timeframe, []

    data_service._get_ticker_snapshot = empty_ticker  # type: ignore[method-assign]
    data_service._fetch_and_persist_klines = empty_klines  # type: ignore[method-assign]

    report = await warm_market_data(
        symbols=["BTC/USDT"],
        symbol_limit=12,
        data_service=data_service,  # type: ignore[arg-type]
        feature_service=FakeFeatureService(),  # type: ignore[arg-type]
        db_coverage_loader=empty_db_coverage,
    )

    assert report["status"] == "blocked"
    assert report["ticker_ready_count"] == 0
    assert all(value == 0 for value in report["kline_timeframe_ready_counts"].values())


@pytest.mark.asyncio
async def test_market_data_warmup_discovers_symbols_when_not_provided() -> None:
    data_service = FakeDataService()

    async def fake_symbols() -> list[dict[str, str]]:
        return [
            {"symbol": "BTC/USDT"},
            {"symbol": "ETH-USDT-SWAP"},
            {"symbol": "SOL/USDT"},
        ]

    data_service.rest_client.get_available_symbols = fake_symbols  # type: ignore[attr-defined]

    report = await warm_market_data(
        symbols=[],
        symbol_limit=2,
        data_service=data_service,  # type: ignore[arg-type]
        feature_service=FakeFeatureService(),  # type: ignore[arg-type]
        db_coverage_loader=ready_db_coverage,
    )

    assert report["symbols"] == ["BTC/USDT", "ETH/USDT"]
    assert report["symbol_count"] == 2


@pytest.mark.asyncio
async def test_market_data_warmup_status_uses_verified_db_coverage() -> None:
    data_service = FakeDataService()

    async def partial_db_coverage(_symbols: list[str]) -> dict[str, Any]:
        return {
            "available": True,
            "ticker_ready_count": 1,
            "kline_timeframe_ready_counts": {
                timeframe: 1 for timeframe in KLINE_PERSIST_TIMEFRAME_LIMITS
            },
            "ticker_symbols": ["BTC/USDT"],
            "kline_symbols_by_timeframe": {
                timeframe: ["BTC/USDT"] for timeframe in KLINE_PERSIST_TIMEFRAME_LIMITS
            },
        }

    report = await warm_market_data(
        symbols=["BTC/USDT", "ETH/USDT"],
        symbol_limit=12,
        data_service=data_service,  # type: ignore[arg-type]
        feature_service=FakeFeatureService(),  # type: ignore[arg-type]
        db_coverage_loader=partial_db_coverage,
    )

    assert report["status"] == "partial"
    assert report["fetched_ticker_ready_count"] == 2
    assert report["ticker_ready_count"] == 1
    assert report["db_ticker_symbols"] == ["BTC/USDT"]


@pytest.mark.asyncio
async def test_market_data_warmup_reports_rest_client_close_failure() -> None:
    data_service = FakeDataService()

    async def broken_close() -> None:
        raise RuntimeError("close failed")

    data_service.rest_client.close = broken_close
    report = await warm_market_data(
        symbols=["BTC/USDT"],
        data_service=data_service,  # type: ignore[arg-type]
        feature_service=FakeFeatureService(),  # type: ignore[arg-type]
        db_coverage_loader=ready_db_coverage,
    )

    assert report["status"] == "ready"
    assert report["rest_client_close_error"] == "close failed"


def test_market_data_warmup_timer_contract_does_not_start_trading() -> None:
    service = render_service(symbol_limit=7)

    assert "run_phase3_market_data_warmup.py" in service
    assert "--symbol-limit 7" in service
    assert "bb-paper-trading" not in service
    assert "trading_service" not in service
    assert "systemctl start" not in service
