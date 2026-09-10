from __future__ import annotations

import json
from typing import Any

import pytest

from data_feed import okx_ws_client
from data_feed.okx_ticker_volume import okx_swap_volume_fields
from data_feed.okx_ws_client import OKXWebSocketClient


def test_okx_swap_volume_fields_uses_base_currency_not_contract_count() -> None:
    fields = okx_swap_volume_fields(
        {
            "last": "0.000002355",
            "vol24h": "5357584.8",
            "volCcy24h": "53575848000000",
        }
    )

    assert fields["volume_24h_contracts"] == pytest.approx(5_357_584.8)
    assert fields["volume_24h_base"] == pytest.approx(53_575_848_000_000)
    assert fields["notional_24h_usdt"] == pytest.approx(126_171_122.04)
    assert fields["volume_24h_source"] == "quote"


@pytest.mark.asyncio
async def test_okx_ws_ticker_keeps_contracts_and_base_volume_separate() -> None:
    client = OKXWebSocketClient()

    await client._handle_message(
        json.dumps(
            {
                "arg": {"channel": "tickers", "instId": "PEPE-USDT-SWAP"},
                "data": [
                    {
                        "last": "0.000002355",
                        "open24h": "0.000002533",
                        "bidPx": "0.000002354",
                        "askPx": "0.000002356",
                        "high24h": "0.000002563",
                        "low24h": "0.000002301",
                        "vol24h": "5357584.8",
                        "volCcy24h": "53575848000000",
                        "ts": "1782432000000",
                    }
                ],
            }
        )
    )

    ticker = client.latest_tickers["PEPE/USDT"]
    assert ticker["volume_24h_contracts"] == pytest.approx(5_357_584.8)
    assert ticker["volume_24h"] == pytest.approx(53_575_848_000_000)
    assert ticker["notional_24h_usdt"] == pytest.approx(126_171_122.04)


@pytest.mark.asyncio
async def test_okx_ws_ticker_processing_is_bounded_per_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OKXWebSocketClient()
    monotonic_times = iter((100.0, 100.1, 100.6))
    current_time = [100.6]

    def fake_monotonic() -> float:
        try:
            current_time[0] = next(monotonic_times)
        except StopIteration:
            pass
        return current_time[0]

    monkeypatch.setattr(okx_ws_client.time, "monotonic", fake_monotonic)

    def payload(last: str, timestamp: str) -> str:
        return json.dumps(
            {
                "arg": {"channel": "tickers", "instId": "BTC-USDT-SWAP"},
                "data": [
                    {
                        "last": last,
                        "open24h": "63000",
                        "bidPx": last,
                        "askPx": last,
                        "high24h": "65000",
                        "low24h": "62000",
                        "vol24h": "100",
                        "volCcy24h": "10",
                        "ts": timestamp,
                    }
                ],
            }
        )

    await client._handle_message(payload("64000", "1782432000000"))
    await client._handle_message(payload("64001", "1782432000100"))
    await client._handle_message(payload("64002", "1782432000600"))

    assert client.latest_tickers["BTC/USDT"]["last_price"] == pytest.approx(64002.0)
    stats = client.get_stats()
    assert stats["throttled_ticker_updates"] == 1
    assert stats["ticker_process_interval_seconds"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_okx_ws_books_and_mark_price_include_native_source_facts() -> None:
    client = OKXWebSocketClient()

    await client._handle_message(
        json.dumps(
            {
                "arg": {"channel": "books5", "instId": "PEPE-USDT-SWAP"},
                "data": [
                    {
                        "bids": [["0.000002354", "5000000", "0", "1"]],
                        "asks": [["0.000002356", "6000000", "0", "1"]],
                        "ts": "1782432000000",
                    }
                ],
            }
        )
    )
    await client._handle_message(
        json.dumps(
            {
                "arg": {"channel": "mark-price", "instId": "PEPE-USDT-SWAP"},
                "data": [{"markPx": "0.000002355", "ts": "1782432000100"}],
            }
        )
    )

    book = client.latest_orderbooks["PEPE/USDT"]
    mark = client.latest_mark_prices["PEPE/USDT"]
    assert book["orderbook_bid_depth"] > 0
    assert book["orderbook_ask_depth"] > 0
    assert book["orderbook_fact"] == {
        "inst_id": "PEPE-USDT-SWAP",
        "inst_type": "SWAP",
        "source_endpoint": "okx_ws_public",
        "source_channel": "books5",
        "source_timestamp_ms": 1_782_432_000_000,
        "bid": pytest.approx(0.000002354),
        "ask": pytest.approx(0.000002356),
        "bid_depth_usdt": pytest.approx(11.77),
        "ask_depth_usdt": pytest.approx(14.136),
    }
    assert mark["mark_price_fact"] == {
        "inst_id": "PEPE-USDT-SWAP",
        "inst_type": "SWAP",
        "source_endpoint": "okx_ws_public",
        "source_channel": "mark-price",
        "source_timestamp_ms": 1_782_432_000_100,
        "price": pytest.approx(0.000002355),
    }


@pytest.mark.asyncio
async def test_okx_ws_connect_uses_unified_sdk_stream(monkeypatch) -> None:
    instances: list[Any] = []

    class _FakeSdkStream:
        def __init__(self, url: str) -> None:
            self.url = url
            self.sent: list[dict] = []
            instances.append(self)

        async def connect(self) -> None:
            return None

        async def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

    monkeypatch.setattr(okx_ws_client, "OkxPublicWebSocketSdkStream", _FakeSdkStream)
    monkeypatch.setattr(okx_ws_client.settings, "symbols", ["BTC/USDT", "ETH/USDT"])

    client = OKXWebSocketClient()
    await client.connect()

    assert len(instances) == 1
    assert instances[0].url == okx_ws_client.WS_PUBLIC_URL
    assert instances[0].sent == [
        {
            "op": "subscribe",
            "args": [
                {"channel": "tickers", "instId": "BTC-USDT-SWAP"},
                {"channel": "books5", "instId": "BTC-USDT-SWAP"},
                {"channel": "mark-price", "instId": "BTC-USDT-SWAP"},
                {"channel": "tickers", "instId": "ETH-USDT-SWAP"},
                {"channel": "books5", "instId": "ETH-USDT-SWAP"},
                {"channel": "mark-price", "instId": "ETH-USDT-SWAP"},
            ],
        }
    ]
