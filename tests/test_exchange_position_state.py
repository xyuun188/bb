from __future__ import annotations

import asyncio

import pytest

from core.symbols import normalize_trading_symbol
from services.exchange_position_state import (
    ExchangeProtectionMapProvider,
    exchange_position_display_valuation,
    exchange_snapshot_price,
    exchange_snapshot_quantity,
    exchange_snapshot_unrealized,
    parse_exchange_position_snapshot,
)


def test_parse_position_does_not_infer_contract_size_from_notional() -> None:
    snapshot = parse_exchange_position_snapshot(
        {
            "symbol": "INJ-USDT-SWAP",
            "side": "long",
            "contracts": 77.0,
            "contractSize": None,
            "markPrice": 5.078,
            "entryPrice": 5.055376623376623,
            "notional": 39.1006,
            "info": {
                "instId": "INJ-USDT-SWAP",
                "pos": "77",
                "posSide": "net",
                "avgPx": "5.055376623376623",
                "markPx": "5.078",
                "upl": "0.015436575875486344",
                "notionalUsd": "39.1006",
            },
        },
        symbol_normalizer=lambda value: str(value).replace("-USDT-SWAP", "/USDT"),
    )

    assert snapshot is not None
    assert snapshot["contract_size"] == 0.0
    assert snapshot["quantity"] == 0.0


def test_parse_okx_position_snapshot_uses_explicit_public_contract_size() -> None:
    snapshot = parse_exchange_position_snapshot(
        {
            "symbol": "PROS/USDT:USDT",
            "side": "long",
            "contracts": 0,
            "contractSize": 1.0,
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


def test_parse_okx_position_snapshot_preserves_exchange_identity_gap() -> None:
    snapshot = parse_exchange_position_snapshot(
        {
            "symbol": "STRK-USDT-SWAP",
            "side": "short",
            "contracts": 10.0,
            "contractSize": 1.0,
            "markPrice": 0.023,
            "entryPrice": 0.024,
            "info": {
                "instId": "STRK-USDT-SWAP",
                "uly": "TSLA-USDT",
                "pos": "-10",
            },
        },
        symbol_normalizer=normalize_trading_symbol,
    )

    assert snapshot is not None
    assert "exchange_position_underlying_mismatch" in snapshot["identity_gaps"]


def test_parse_okx_position_snapshot_blocks_notional_price_unit_mismatch() -> None:
    snapshot = parse_exchange_position_snapshot(
        {
            "symbol": "STRK-USDT-SWAP",
            "side": "short",
            "contracts": 3949.0,
            "contractSize": 1.0,
            "markPrice": 376.6,
            "entryPrice": 375.95,
            "notional": 14856.0,
            "info": {
                "instId": "STRK-USDT-SWAP",
                "pos": "-3949",
                "markPx": "376.6",
                "avgPx": "375.95",
                "notionalUsd": "14856",
            },
        },
        symbol_normalizer=normalize_trading_symbol,
    )

    assert snapshot is not None
    assert "exchange_position_notional_mismatch" in snapshot["identity_gaps"]
    assert "contract_spec_mismatch" in snapshot["identity_gaps"]
    assert snapshot["contract_valuation"]["notional_consistent"] is False


def test_parse_okx_linear_contract_uses_ct_val_currency_and_multiplier() -> None:
    snapshot = parse_exchange_position_snapshot(
        {
            "symbol": "STRK-USDT-SWAP",
            "side": "short",
            "contracts": 3949.0,
            "contractSize": 0.01,
            "contractSizeSource": "okx_public_instruments",
            "contractSpec": {
                "instId": "STRK-USDT-SWAP",
                "ctVal": "0.01",
                "ctMult": "1",
                "ctValCcy": "STRK",
                "settleCcy": "USDT",
                "ctType": "linear",
                "source": "okx_public_instruments",
            },
            "markPrice": 379.29,
            "entryPrice": 375.95,
            "notional": 14962.62,
            "timestamp": 1786950000000,
            "info": {
                "instId": "STRK-USDT-SWAP",
                "pos": "-3949",
                "markPx": "379.29",
                "notionalUsd": "14962.62",
                "uTime": "1786950000000",
            },
        },
        symbol_normalizer=normalize_trading_symbol,
    )

    assert snapshot is not None
    assert snapshot["quantity"] == pytest.approx(39.49)
    assert snapshot["calculated_notional"] == pytest.approx(14978.1621)
    assert snapshot["notional_consistent"] is True
    assert "contract_spec_mismatch" not in snapshot["identity_gaps"]


def test_parse_okx_quote_denominated_contract_does_not_multiply_mark_price() -> None:
    snapshot = parse_exchange_position_snapshot(
        {
            "symbol": "BTC-USD-SWAP",
            "side": "long",
            "contracts": 10.0,
            "contractSpec": {
                "instId": "BTC-USD-SWAP",
                "ctVal": "100",
                "ctMult": "1",
                "ctValCcy": "USD",
                "settleCcy": "BTC",
                "ctType": "inverse",
                "source": "okx_public_instruments",
            },
            "markPrice": 100000.0,
            "notional": 1000.0,
            "info": {"instId": "BTC-USD-SWAP", "pos": "10"},
        },
        symbol_normalizer=lambda value: str(value),
    )

    assert snapshot is not None
    assert snapshot["calculated_notional"] == pytest.approx(1000.0)
    assert snapshot["quantity"] == pytest.approx(0.01)
    assert exchange_snapshot_quantity(snapshot) == pytest.approx(0.01)
    assert snapshot["notional_consistent"] is True


def test_parse_okx_position_snapshot_prefers_inst_id_over_ccxt_alias_symbol() -> None:
    snapshot = parse_exchange_position_snapshot(
        {
            "symbol": "SAHARA/USDT:USDT",
            "side": "short",
            "contracts": 100,
            "contractSize": 1.0,
            "markPrice": 0.01751,
            "entryPrice": 0.0179,
            "info": {
                "instId": "SPK-USDT-SWAP",
                "uly": "SAHARA-USDT",
                "ctValCcy": "SPK",
                "pos": "-100",
                "ctVal": "1",
                "avgPx": "0.0179",
                "markPx": "0.01751",
                "upl": "0.039",
            },
        },
        symbol_normalizer=normalize_trading_symbol,
    )

    assert snapshot is not None
    assert snapshot["symbol"] == "SPK/USDT"
    assert snapshot["ccxt_symbol"] == "SAHARA/USDT:USDT"
    assert snapshot["raw_symbol"] == "SPK-USDT-SWAP"
    assert snapshot["side"] == "short"
    assert snapshot["quantity"] == pytest.approx(100.0)


def test_parse_okx_net_position_snapshot_infers_side_from_signed_pos() -> None:
    snapshot = parse_exchange_position_snapshot(
        {
            "contracts": 200,
            "contractSize": 1.0,
            "info": {
                "instId": "SPK-USDT-SWAP",
                "posSide": "net",
                "pos": "-200",
                "ctVal": "1",
                "avgPx": "0.01785",
                "markPx": "0.0177",
                "last": "0.01762",
                "upl": "0.0300000000000002",
                "posId": "3688338318498172929",
            },
        },
        symbol_normalizer=normalize_trading_symbol,
    )

    assert snapshot is not None
    assert snapshot["symbol"] == "SPK/USDT"
    assert snapshot["side"] == "short"
    assert snapshot["contracts"] == pytest.approx(200.0)
    assert snapshot["quantity"] == pytest.approx(200.0)
    assert snapshot["entry_price"] == pytest.approx(0.01785)
    assert snapshot["mark_price"] == pytest.approx(0.0177)
    assert snapshot["upl"] == pytest.approx(0.0300000000000002)
    assert snapshot["raw_pos_side"] == "net"
    assert snapshot["raw_pos"] == "-200"
    assert snapshot["signed_position_size"] == pytest.approx(-200.0)
    assert snapshot["side_inference"] == "okx_net_signed_pos"


def test_parse_okx_net_position_snapshot_infers_long_from_positive_pos() -> None:
    snapshot = parse_exchange_position_snapshot(
        {
            "info": {
                "instId": "ENA-USDT-SWAP",
                "posSide": "net",
                "pos": "45",
                "ctVal": "1",
                "avgPx": "0.0876",
                "markPx": "0.0881",
            },
        },
        symbol_normalizer=normalize_trading_symbol,
    )

    assert snapshot is not None
    assert snapshot["symbol"] == "ENA/USDT"
    assert snapshot["side"] == "long"
    assert snapshot["contracts"] == pytest.approx(45.0)


def test_parse_okx_position_snapshot_treats_quantity_as_contracts_when_contracts_exist() -> None:
    snapshot = parse_exchange_position_snapshot(
        {
            "symbol": "AUCTION/USDT:USDT",
            "side": "short",
            "quantity": 99.5,
            "contracts": 99.5,
            "contractSize": 0.1,
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


def test_parse_okx_position_snapshot_does_not_infer_contract_size_from_upl() -> None:
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
    assert snapshot["contract_size"] == 0.0
    assert snapshot["quantity"] == 0.0


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
async def test_exchange_protection_map_propagates_account_wide_timeout_without_stale_cache() -> (
    None
):
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
        account_wide_timeout_seconds=0.001,
        cache_ttl_seconds=0.0,
    )

    with pytest.raises(TimeoutError):
        await provider.fetch(
            executor,
            [
                {"symbol": "BTC/USDT", "contracts": "1"},
                {"symbol": "ETH/USDT", "contracts": "1"},
                {"symbol": "SOL/USDT", "contracts": "1"},
            ],
        )

    assert executor.calls == [None]


@pytest.mark.asyncio
async def test_exchange_protection_map_retains_split_group_inventory() -> None:
    class FakeExecutor:
        async def get_position_protection_orders(self, symbol=None):
            return [
                {
                    "symbol": "BTC/USDT",
                    "position_side": "long",
                    "algo_id": "oco-a",
                    "contracts": 2.0,
                    "stop_loss_price": 95.0,
                    "take_profit_price": 110.0,
                    "updated_at_ms": 1,
                },
                {
                    "symbol": "BTC/USDT",
                    "position_side": "long",
                    "algo_id": "oco-b",
                    "contracts": 1.0,
                    "stop_loss_price": 96.0,
                    "take_profit_price": 112.0,
                    "updated_at_ms": 2,
                },
            ]

    provider = ExchangeProtectionMapProvider(
        symbol_normalizer=normalize_trading_symbol,
        position_open_checker=lambda position: float(position.get("contracts") or 0) > 0,
        timeout_seconds=1.0,
        cache_ttl_seconds=0.0,
    )

    result = await provider.fetch(
        FakeExecutor(),
        [{"symbol": "BTC/USDT", "contracts": "3"}],
    )

    orders = result[("BTC/USDT", "long")]["protection_orders"]
    assert {order["algo_id"] for order in orders} == {"oco-a", "oco-b"}
    assert sum(order["contracts"] for order in orders) == pytest.approx(3.0)
