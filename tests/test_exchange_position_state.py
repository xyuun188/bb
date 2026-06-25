from __future__ import annotations

import asyncio

import pytest

from core.symbols import normalize_trading_symbol
from services.exchange_position_state import (
    ExchangeProtectionMapProvider,
    exchange_position_display_valuation,
    exchange_snapshot_price,
    exchange_snapshot_unrealized,
    parse_exchange_position_snapshot,
)


def test_parse_okx_position_snapshot_prefers_info_markpx_upl_and_ctval() -> None:
    snapshot = parse_exchange_position_snapshot(
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
        },
        symbol_normalizer=normalize_trading_symbol,
    )

    assert snapshot is not None
    assert snapshot["symbol"] == "PROS/USDT"
    assert snapshot["side"] == "long"
    assert snapshot["mark_price"] == pytest.approx(0.4059)
    assert snapshot["last_price"] == pytest.approx(0.5547)
    assert snapshot["entry_price"] == pytest.approx(0.4054)
    assert snapshot["quantity"] == pytest.approx(46.0)
    assert snapshot["upl"] == pytest.approx(-0.82)
    assert exchange_snapshot_price(snapshot) == pytest.approx(0.4059)
    assert exchange_snapshot_unrealized(snapshot, "long") == pytest.approx(-0.82)


def test_parse_okx_position_snapshot_treats_quantity_as_contracts_when_contracts_exist() -> None:
    snapshot = parse_exchange_position_snapshot(
        {
            "symbol": "AUCTION/USDT:USDT",
            "side": "short",
            "quantity": 99.5,
            "contracts": 99.5,
            "entryPrice": 3.532,
            "markPrice": 3.528,
            "info": {
                "instId": "AUCTION-USDT-SWAP",
                "pos": "99.5",
                "ctVal": "0.1",
                "avgPx": "3.532",
                "markPx": "3.528",
                "upl": "0.0398",
            },
        },
        symbol_normalizer=normalize_trading_symbol,
    )

    assert snapshot is not None
    assert snapshot["contracts"] == pytest.approx(99.5)
    assert snapshot["contract_size"] == pytest.approx(0.1)
    assert snapshot["quantity"] == pytest.approx(9.95)
    assert snapshot["raw_quantity"] == pytest.approx(99.5)


def test_parse_okx_position_snapshot_infers_missing_contract_size_from_upl() -> None:
    snapshot = parse_exchange_position_snapshot(
        {
            "symbol": "LAB-USDT-SWAP",
            "side": "long",
            "contracts": 9.0,
            "markPrice": 17.787,
            "entryPrice": 16.865555555555556,
            "unrealizedPnl": 0.8292999999999989,
            "info": {
                "instId": "LAB-USDT-SWAP",
                "pos": "9",
                "avgPx": "16.8655555555555556",
                "markPx": "17.787",
                "upl": "0.8292999999999989",
            },
        },
        symbol_normalizer=normalize_trading_symbol,
    )

    assert snapshot is not None
    assert snapshot["symbol"] == "LAB/USDT"
    assert snapshot["contracts"] == pytest.approx(9.0)
    assert snapshot["contract_size"] == pytest.approx(0.1)
    assert snapshot["quantity"] == pytest.approx(0.9)


def test_exchange_position_display_valuation_does_not_use_stale_local_profit() -> None:
    valuation = exchange_position_display_valuation(
        {
            "mark_price": 0.4059,
            "last_price": 0.5547,
            "entry_price": 0.4054,
            "quantity": 46.0,
            "upl": -0.82,
        },
        "long",
        fallback_current_price=0.5547,
        fallback_unrealized_pnl=6.8678,
        fallback_entry_price=0.4054,
        fallback_quantity=46,
    )

    assert valuation["current_price"] == pytest.approx(0.4059)
    assert valuation["unrealized_pnl"] == pytest.approx(-0.82)
    assert valuation["pnl_source"] == "okx_position_upl"


@pytest.mark.asyncio
async def test_exchange_protection_map_fetches_account_wide_once() -> None:
    class FakeExecutor:
        def __init__(self) -> None:
            self.calls: list[str | None] = []

        async def get_position_protection_orders(self, symbol=None):
            self.calls.append(symbol)
            return [
                {
                    "symbol": "BTC/USDT",
                    "position_side": "long",
                    "take_profit_price": 125.0,
                    "updated_at_ms": 2,
                },
                {
                    "symbol": "ETH/USDT",
                    "position_side": "short",
                    "stop_loss_price": 2420.0,
                    "updated_at_ms": 1,
                },
                {
                    "symbol": "DOGE/USDT",
                    "position_side": "long",
                    "take_profit_price": 0.2,
                    "updated_at_ms": 3,
                },
            ]

    executor = FakeExecutor()
    provider = ExchangeProtectionMapProvider(
        symbol_normalizer=normalize_trading_symbol,
        position_open_checker=lambda position: float(position.get("contracts") or 0) > 0,
        timeout_seconds=1.0,
        cache_ttl_seconds=30.0,
    )

    result = await provider.fetch(
        executor,
        [
            {"symbol": "BTC/USDT:USDT", "contracts": "1"},
            {"symbol": "ETH/USDT:USDT", "contracts": "2"},
            {"symbol": "SOL/USDT:USDT", "contracts": "0"},
        ],
    )

    assert executor.calls == [None]
    assert set(result) == {("BTC/USDT", "long"), ("ETH/USDT", "short")}
    assert result[("BTC/USDT", "long")]["take_profit_price"] == 125.0


@pytest.mark.asyncio
async def test_exchange_protection_map_does_not_fan_out_after_account_wide_timeout() -> None:
    class SlowExecutor:
        def __init__(self) -> None:
            self.calls: list[str | None] = []

        async def get_position_protection_orders(self, symbol=None):
            self.calls.append(symbol)
            await asyncio.sleep(0.05)
            return []

    executor = SlowExecutor()
    provider = ExchangeProtectionMapProvider(
        symbol_normalizer=normalize_trading_symbol,
        position_open_checker=lambda position: True,
        timeout_seconds=0.001,
        cache_ttl_seconds=0.0,
    )

    result = await provider.fetch(
        executor,
        [
            {"symbol": "BTC/USDT", "contracts": "1"},
            {"symbol": "ETH/USDT", "contracts": "1"},
            {"symbol": "SOL/USDT", "contracts": "1"},
        ],
    )

    assert result == {}
    assert executor.calls == [None]
