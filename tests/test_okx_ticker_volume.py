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
    assert book["orderbook_bid_depth"] == 0
    assert book["orderbook_ask_depth"] == 0
    assert book["bid_depth_quote_contracts"] > 0
    assert book["ask_depth_quote_contracts"] > 0
    assert book["orderbook_fact"] == {
        "inst_id": "PEPE-USDT-SWAP",
        "inst_type": "SWAP",
        "source_endpoint": "okx_ws_public",
        "source_channel": "books5",
        "source_timestamp_ms": 1_782_432_000_000,
        "received_at": book["received_at"],
        "bid": pytest.approx(0.000002354),
        "ask": pytest.approx(0.000002356),
        "bid_depth_quote_contracts": pytest.approx(11.77),
        "ask_depth_quote_contracts": pytest.approx(14.136),
    }
    assert mark["mark_price_fact"] == {
        "inst_id": "PEPE-USDT-SWAP",
        "inst_type": "SWAP",
        "source_endpoint": "okx_ws_public",
        "source_channel": "mark-price",
        "source_timestamp_ms": 1_782_432_000_100,
        "received_at": mark["received_at"],
        "price": pytest.approx(0.000002355),
    }


@pytest.mark.asyncio
async def test_okx_ws_index_ticker_is_kept_as_distinct_native_fact() -> None:
    client = OKXWebSocketClient()
    await client._handle_message(
        json.dumps(
            {
                "arg": {"channel": "index-tickers", "instId": "PEPE-USDT"},
                "data": [{"idxPx": "0.000002355", "ts": "1782432000200"}],
            }
        )
    )

    row = client.latest_index_prices["PEPE/USDT"]
    assert row["index_price"] == pytest.approx(0.000002355)
    assert row["index_price_fact"] == {
        "inst_id": "PEPE-USDT",
        "inst_type": "INDEX",
        "source_endpoint": "okx_ws_public",
        "source_channel": "index-tickers",
        "source_timestamp_ms": 1_782_432_000_200,
        "received_at": row["received_at"],
        "price": pytest.approx(0.000002355),
    }
    assert client.get_stats()["channel_message_counts"] == {"index-tickers": 1}


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
    assert ":8443" not in instances[0].url
    assert instances[0].sent == [
        {
            "op": "subscribe",
            "args": [
                {"channel": "tickers", "instId": "BTC-USDT-SWAP"},
                {"channel": "books5", "instId": "BTC-USDT-SWAP"},
                {"channel": "mark-price", "instId": "BTC-USDT-SWAP"},
                {"channel": "index-tickers", "instId": "BTC-USDT"},
                {"channel": "tickers", "instId": "ETH-USDT-SWAP"},
                {"channel": "books5", "instId": "ETH-USDT-SWAP"},
                {"channel": "mark-price", "instId": "ETH-USDT-SWAP"},
                {"channel": "index-tickers", "instId": "ETH-USDT"},
            ],
        }
    ]


@pytest.mark.asyncio
async def test_okx_ws_subscription_batches_are_rate_limited(monkeypatch) -> None:
    client = OKXWebSocketClient()
    sent: list[dict[str, Any]] = []
    delays: list[float] = []

    class _FakeStream:
        async def send(self, payload: str) -> None:
            sent.append(json.loads(payload))

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    client._ws = _FakeStream()
    client._running = True
    monkeypatch.setattr(okx_ws_client.asyncio, "sleep", fake_sleep)

    channels = client._subscription_args(
        [f"COIN{index}/USDT" for index in range(26)]
    )
    request_count = await client._send_subscription_operation("subscribe", channels)

    assert request_count == 2
    assert [len(request["args"]) for request in sent] == [100, 4]
    assert len(delays) == 1
    assert 0.49 <= delays[0] <= 0.51
    assert client.get_stats()["subscription_request_count"] == 2


@pytest.mark.asyncio
async def test_okx_ws_subscription_error_is_counted_and_logged(monkeypatch) -> None:
    client = OKXWebSocketClient()
    errors: list[dict[str, Any]] = []
    monkeypatch.setattr(
        okx_ws_client.logger,
        "error",
        lambda _message, **kwargs: errors.append(kwargs),
    )

    await client._handle_message(
        json.dumps(
            {
                "event": "error",
                "code": "60018",
                "msg": "Requests too frequent",
                "arg": {"channel": "books5", "instId": "BTC-USDT-SWAP"},
            }
        )
    )

    stats = client.get_stats()
    assert stats["subscription_error_count"] == 1
    assert stats["subscription_coverage"]["unattributed_error_count"] == 1
    assert errors[0]["code"] == "60018"


@pytest.mark.asyncio
async def test_okx_ws_subscription_coverage_tracks_argument_level_acknowledgements() -> None:
    client = OKXWebSocketClient()
    channels = [
        {"channel": "tickers", "instId": "BTC-USDT-SWAP"},
        {"channel": "books5", "instId": "ETH-USDT-SWAP"},
        {"channel": "mark-price", "instId": "XRP-USDT-SWAP"},
    ]
    sent: list[dict[str, Any]] = []

    class _FakeStream:
        async def send(self, payload: str) -> None:
            sent.append(json.loads(payload))

    client._ws = _FakeStream()
    client._running = True
    await client._send_subscription_operation("subscribe", channels)
    await client._handle_message(
        json.dumps(
            {
                "event": "subscribe",
                "arg": {"channel": "tickers", "instId": "BTC-USDT-SWAP"},
            }
        )
    )
    await client._handle_message(
        json.dumps(
            {
                "event": "subscribe",
                "arg": {"channel": "books5", "instId": "ETH-USDT-SWAP"},
            }
        )
    )
    await client._handle_message(
        json.dumps(
            {
                "event": "error",
                "code": "60018",
                "msg": "Requests too frequent",
                "arg": {"channel": "mark-price", "instId": "XRP-USDT-SWAP"},
            }
        )
    )

    stats = client.get_stats()
    coverage = stats["subscription_coverage"]
    assert len(sent) == 1
    assert coverage["expected_argument_count"] == 3
    assert coverage["acknowledged_argument_count"] == 2
    assert coverage["rejected_argument_count"] == 1
    assert coverage["missing_ack_count"] == 0
    assert coverage["by_channel"] == {
        "books5": {
            "expected": 1,
            "acknowledged": 1,
            "rejected": 0,
            "missing_ack": 0,
        },
        "mark-price": {
            "expected": 1,
            "acknowledged": 0,
            "rejected": 1,
            "missing_ack": 0,
        },
        "tickers": {
            "expected": 1,
            "acknowledged": 1,
            "rejected": 0,
            "missing_ack": 0,
        },
    }
