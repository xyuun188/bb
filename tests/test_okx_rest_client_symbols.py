from __future__ import annotations

import pytest

from data_feed.feature_vector import build_feature_vector
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
async def test_fetch_instrument_spec_keeps_native_contract_identity(monkeypatch) -> None:
    client = OKXRestClient()

    async def fake_ccxt_call(method_name: str, *args, **kwargs):
        assert method_name == "publicGetPublicInstruments"
        assert args == ({"instType": "SWAP", "instId": "ROBO-USDT-SWAP"},)
        return {
            "data": [
                {
                    "instId": "ROBO-USDT-SWAP",
                    "instType": "SWAP",
                    "uly": "ROBO-USDT",
                    "instFamily": "ROBO-USDT",
                    "instCategory": "1",
                    "ctType": "linear",
                    "ctVal": "1",
                    "ctMult": "1",
                    "ctValCcy": "ROBO",
                    "settleCcy": "USDT",
                    "lotSz": "1",
                    "minSz": "1",
                    "tickSz": "0.00001",
                    "state": "live",
                }
            ]
        }

    monkeypatch.setattr(client, "_ccxt_call", fake_ccxt_call)

    spec = await client.fetch_instrument_spec("ROBO/USDT")

    assert spec["instId"] == "ROBO-USDT-SWAP"
    assert spec["uly"] == "ROBO-USDT"
    assert spec["ctVal"] == "1"
    assert spec["source"] == "okx_public_instruments"


@pytest.mark.asyncio
async def test_reference_prices_keep_swap_and_index_native_identities(monkeypatch) -> None:
    client = OKXRestClient()
    calls: list[tuple[str, tuple]] = []

    async def fake_ccxt_call(method_name: str, *args, **kwargs):
        calls.append((method_name, args))
        if method_name == "publicGetPublicMarkPrice":
            return {
                "data": [
                    {
                        "instId": "ROBO-USDT-SWAP",
                        "instType": "SWAP",
                        "markPx": "0.01291",
                        "ts": "1783990800000",
                    }
                ]
            }
        return {
            "data": [
                {
                    "instId": "ROBO-USDT",
                    "idxPx": "0.01290",
                    "ts": "1783990800000",
                }
            ]
        }

    monkeypatch.setattr(client, "_ccxt_call", fake_ccxt_call)

    prices = await client.fetch_reference_prices(
        "ROBO/USDT",
        contract_spec={"instId": "ROBO-USDT-SWAP", "uly": "ROBO-USDT"},
    )

    assert calls == [
        (
            "publicGetPublicMarkPrice",
            ({"instType": "SWAP", "instId": "ROBO-USDT-SWAP"},),
        ),
        ("publicGetMarketIndexTickers", ({"instId": "ROBO-USDT"},)),
    ]
    assert prices["mark_price_fact"]["inst_id"] == "ROBO-USDT-SWAP"
    assert prices["index_price_fact"]["inst_id"] == "ROBO-USDT"
    assert prices["mark_price"] == pytest.approx(0.01291)
    assert prices["index_price"] == pytest.approx(0.01290)


@pytest.mark.asyncio
async def test_orderbook_depth_uses_native_contract_value(monkeypatch) -> None:
    client = OKXRestClient()

    async def fake_ccxt_call(method_name: str, *args, **kwargs):
        assert method_name == "fetch_order_book"
        return {
            "timestamp": 1_783_990_800_000,
            "bids": [[100.0, 2.0]],
            "asks": [[100.1, 3.0]],
        }

    monkeypatch.setattr(client, "_ccxt_call", fake_ccxt_call)

    metrics = await client.fetch_order_book_metrics(
        "BTC/USDT",
        contract_spec={
            "instId": "BTC-USDT-SWAP",
            "instType": "SWAP",
            "uly": "BTC-USDT",
            "ctVal": "0.01",
            "ctMult": "1",
        },
    )

    assert metrics["orderbook_bid_depth"] == pytest.approx(2.0)
    assert metrics["orderbook_ask_depth"] == pytest.approx(3.003)
    assert metrics["orderbook_fact"]["source_timestamp_ms"] == 1_783_990_800_000


@pytest.mark.asyncio
async def test_funding_snapshot_derives_interval_from_okx_times(monkeypatch) -> None:
    client = OKXRestClient()

    async def fake_ccxt_call(method_name: str, *args, **kwargs):
        assert method_name == "fetch_funding_rate"
        assert args == ("BTC/USDT:USDT",)
        return {
            "fundingRate": -0.00001,
            "fundingDatetime": "1783958400000",
            "nextFundingDatetime": "1783987200000",
            "info": {
                "fundingRate": "-0.00001",
                "fundingTime": "1783958400000",
                "nextFundingTime": "1783987200000",
                "ts": "1783931769300",
            },
        }

    monkeypatch.setattr(client, "_ccxt_call", fake_ccxt_call)

    funding = await client.fetch_funding_rate("BTC/USDT")
    vector = build_feature_vector("BTC/USDT", derivatives=funding)

    assert funding["funding_data_available"] is True
    assert funding["funding_interval_minutes"] == pytest.approx(480.0)
    assert vector.funding_interval_minutes == pytest.approx(480.0)
    assert vector.funding_data_available is True
    assert vector.to_dict()["funding_rate_observed_at"] == "1783931769300"


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
async def test_fetch_tickers_without_targets_keeps_loaded_market_universe(monkeypatch) -> None:
    client = OKXRestClient()
    exchange = _PepeMarketExchange()
    calls: list[tuple[str, tuple]] = []

    async def fake_exchange():
        return exchange

    async def fake_ccxt_call(method_name: str, *args, **kwargs):
        calls.append((method_name, args))
        return {
            "data": [
                {"instId": "PEPE-USDT-SWAP", "last": "0.000002", "volCcy24h": "100"},
                {"instId": "PLTR-USDT-SWAP", "last": "64.46", "volCcy24h": "200"},
            ]
        }

    monkeypatch.setattr(client, "_get_exchange", fake_exchange)
    monkeypatch.setattr(client, "_ccxt_call", fake_ccxt_call)

    tickers = await client.fetch_tickers()

    assert calls == [
        ("publicGetMarketTickers", ({"instType": "SWAP"},)),
    ]
    assert set(tickers) == {"PEPE/USDT", "PEPE-USDT-SWAP"}


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
