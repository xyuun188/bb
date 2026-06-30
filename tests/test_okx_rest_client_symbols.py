from __future__ import annotations

import pytest

from data_feed.okx_rest_client import OKXRestClient


class _AliasMarketExchange:
    def __init__(self) -> None:
        self.markets = {
            "H-USDT-SWAP": {
                "active": True,
                "type": "swap",
                "swap": True,
                "linear": True,
                "settle": "USDT",
                "quote": "USDT",
                "base": "WLFI",
                "symbol": "WLFI/USDT:USDT",
                "id": "H-USDT-SWAP",
                "info": {"instId": "H-USDT-SWAP", "ctValCcy": "H"},
            }
        }


class _PepeMarketExchange:
    def __init__(self) -> None:
        self.markets = {
            "PEPE-USDT-SWAP": {
                "active": True,
                "type": "swap",
                "swap": True,
                "linear": True,
                "settle": "USDT",
                "quote": "USDT",
                "base": "PEPE",
                "symbol": "PEPE/USDT:USDT",
                "id": "PEPE-USDT-SWAP",
                "info": {"instId": "PEPE-USDT-SWAP", "ctValCcy": "PEPE"},
            }
        }


@pytest.mark.asyncio
async def test_available_symbols_use_okx_inst_id_over_ccxt_alias(monkeypatch) -> None:
    client = OKXRestClient()
    exchange = _AliasMarketExchange()

    async def fake_exchange():
        return exchange

    async def fake_ccxt_call(method_name: str, *args, **kwargs):
        assert method_name == "publicGetMarketTickers"
        assert args == ({"instType": "SWAP"},)
        return {"data": []}

    monkeypatch.setattr(client, "_get_exchange", fake_exchange)
    monkeypatch.setattr(client, "_ccxt_call", fake_ccxt_call)

    symbols = await client.get_available_symbols()

    assert symbols[0]["symbol"] == "H/USDT"
    assert symbols[0]["id"] == "H-USDT-SWAP"
    assert symbols[0]["ccxt_symbol"] == "WLFI/USDT:USDT"


@pytest.mark.asyncio
async def test_available_symbols_exposes_okx_swap_base_and_notional_volume(monkeypatch) -> None:
    client = OKXRestClient()
    exchange = _PepeMarketExchange()

    async def fake_exchange():
        return exchange

    async def fake_ccxt_call(method_name: str, *args, **kwargs):
        assert method_name == "publicGetMarketTickers"
        assert args == ({"instType": "SWAP"},)
        return {
            "data": [
                {
                    "instId": "PEPE-USDT-SWAP",
                    "last": "0.000002355",
                    "vol24h": "5357584.8",
                    "volCcy24h": "53575848000000",
                }
            ]
        }

    monkeypatch.setattr(client, "_get_exchange", fake_exchange)
    monkeypatch.setattr(client, "_ccxt_call", fake_ccxt_call)

    symbols = await client.get_available_symbols()

    assert symbols[0]["symbol"] == "PEPE/USDT"
    assert symbols[0]["volume_24h"] == pytest.approx(53_575_848_000_000)
    assert symbols[0]["volume_24h_contracts"] == pytest.approx(5_357_584.8)
    assert symbols[0]["notional_24h_usdt"] == pytest.approx(126_171_122.04)


@pytest.mark.asyncio
async def test_fetch_ticker_uses_okx_native_inst_id(monkeypatch) -> None:
    client = OKXRestClient()
    calls: list[tuple[str, tuple]] = []

    async def fake_ccxt_call(method_name: str, *args, **kwargs):
        calls.append((method_name, args))
        return {
            "data": [
                {
                    "instId": "SPK-USDT-SWAP",
                    "last": "0.0123",
                    "open24h": "0.011",
                    "high24h": "0.013",
                    "low24h": "0.010",
                    "bidPx": "0.0122",
                    "askPx": "0.0124",
                    "vol24h": "1000",
                    "volCcy24h": "100000",
                    "ts": "1780000000000",
                }
            ]
        }

    monkeypatch.setattr(client, "_ccxt_call", fake_ccxt_call)

    ticker = await client.fetch_ticker("SPK/USDT")

    assert calls == [
        ("publicGetMarketTicker", ({"instId": "SPK-USDT-SWAP"},)),
    ]
    assert ticker["symbol"] == "SPK/USDT"
    assert ticker["id"] == "SPK-USDT-SWAP"
    assert ticker["last"] == pytest.approx(0.0123)
    assert ticker["bid"] == pytest.approx(0.0122)
    assert ticker["ask"] == pytest.approx(0.0124)
    assert ticker["volume_24h_base"] == pytest.approx(100000.0)


@pytest.mark.asyncio
async def test_fetch_tickers_filters_by_okx_native_inst_ids(monkeypatch) -> None:
    client = OKXRestClient()
    calls: list[tuple[str, tuple]] = []

    async def fake_ccxt_call(method_name: str, *args, **kwargs):
        calls.append((method_name, args))
        return {
            "data": [
                {"instId": "SPK-USDT-SWAP", "last": "0.0123", "volCcy24h": "100"},
                {"instId": "HOME-USDT-SWAP", "last": "0.0222", "volCcy24h": "200"},
            ]
        }

    monkeypatch.setattr(client, "_ccxt_call", fake_ccxt_call)

    tickers = await client.fetch_tickers(["SPK/USDT"])

    assert calls == [
        ("publicGetMarketTickers", ({"instType": "SWAP"},)),
    ]
    assert set(tickers) == {"SPK/USDT", "SPK-USDT-SWAP"}
    assert tickers["SPK/USDT"]["id"] == "SPK-USDT-SWAP"


@pytest.mark.asyncio
async def test_fetch_positions_uses_okx_native_positions(monkeypatch) -> None:
    client = OKXRestClient()
    calls: list[tuple[str, tuple]] = []

    async def fake_ccxt_call(method_name: str, *args, **kwargs):
        calls.append((method_name, args))
        return {
            "data": [
                {
                    "instId": "SPK-USDT-SWAP",
                    "posSide": "net",
                    "pos": "-200",
                    "ctVal": "1",
                    "avgPx": "0.012",
                    "markPx": "0.011",
                    "upl": "0.2",
                    "lever": "3",
                },
                {
                    "instId": "HOME-USDT-SWAP",
                    "posSide": "long",
                    "pos": "0",
                    "ctVal": "1",
                },
            ]
        }

    monkeypatch.setattr(client, "_ccxt_call", fake_ccxt_call)

    positions = await client.fetch_positions(["SPK/USDT"])

    assert calls == [
        ("privateGetAccountPositions", ({"instType": "SWAP", "instId": "SPK-USDT-SWAP"},)),
    ]
    assert len(positions) == 1
    assert positions[0]["symbol"] == "SPK-USDT-SWAP"
    assert positions[0]["side"] == "short"
    assert positions[0]["contracts"] == pytest.approx(200.0)
    assert positions[0]["markPrice"] == pytest.approx(0.011)


@pytest.mark.asyncio
async def test_fetch_open_orders_uses_okx_native_pending_orders(monkeypatch) -> None:
    client = OKXRestClient()
    calls: list[tuple[str, tuple]] = []

    async def fake_ccxt_call(method_name: str, *args, **kwargs):
        calls.append((method_name, args))
        return {
            "data": [
                {
                    "instId": "SPK-USDT-SWAP",
                    "ordId": "spk-order-1",
                    "clOrdId": "local-1",
                    "side": "buy",
                    "ordType": "market",
                    "state": "live",
                    "sz": "120",
                    "accFillSz": "20",
                    "reduceOnly": "true",
                    "cTime": "1780000000000",
                    "uTime": "1780000001000",
                }
            ]
        }

    monkeypatch.setattr(client, "_ccxt_call", fake_ccxt_call)

    orders = await client.fetch_open_orders("SPK/USDT")

    assert calls == [
        (
            "privateGetTradeOrdersPending",
            ({"instType": "SWAP", "instId": "SPK-USDT-SWAP", "limit": "100"},),
        ),
    ]
    assert len(orders) == 1
    assert orders[0]["id"] == "spk-order-1"
    assert orders[0]["symbol"] == "SPK-USDT-SWAP"
    assert orders[0]["side"] == "buy"
    assert orders[0]["status"] == "open"
    assert orders[0]["remaining"] == pytest.approx(100.0)
    assert orders[0]["reduceOnly"] is True


@pytest.mark.asyncio
async def test_fetch_order_uses_okx_native_order_detail(monkeypatch) -> None:
    client = OKXRestClient()
    calls: list[tuple[str, tuple]] = []

    async def fake_ccxt_call(method_name: str, *args, **kwargs):
        calls.append((method_name, args))
        return {
            "data": [
                {
                    "instId": "SPK-USDT-SWAP",
                    "ordId": "spk-order-1",
                    "clOrdId": "local-1",
                    "side": "buy",
                    "ordType": "market",
                    "state": "filled",
                    "sz": "120",
                    "accFillSz": "120",
                    "avgPx": "0.012",
                    "fee": "-0.01",
                    "cTime": "1780000000000",
                    "uTime": "1780000001000",
                }
            ]
        }

    monkeypatch.setattr(client, "_ccxt_call", fake_ccxt_call)

    order = await client.fetch_order("spk-order-1", "SPK/USDT")

    assert calls == [
        (
            "privateGetTradeOrder",
            ({"instId": "SPK-USDT-SWAP", "ordId": "spk-order-1"},),
        ),
    ]
    assert order["id"] == "spk-order-1"
    assert order["symbol"] == "SPK-USDT-SWAP"
    assert order["status"] == "closed"
    assert order["filled"] == pytest.approx(120.0)


@pytest.mark.asyncio
async def test_cancel_order_uses_okx_native_cancel_order(monkeypatch) -> None:
    client = OKXRestClient()
    calls: list[tuple[str, tuple]] = []

    async def fake_ccxt_call(method_name: str, *args, **kwargs):
        calls.append((method_name, args))
        return {"code": "0", "data": [{"instId": "SPK-USDT-SWAP", "ordId": "spk-order-1"}]}

    monkeypatch.setattr(client, "_ccxt_call", fake_ccxt_call)

    result = await client.cancel_order("spk-order-1", "SPK/USDT")

    assert result["code"] == "0"
    assert calls == [
        (
            "privatePostTradeCancelOrder",
            ({"instId": "SPK-USDT-SWAP", "ordId": "spk-order-1"},),
        ),
    ]
