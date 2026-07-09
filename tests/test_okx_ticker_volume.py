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
                {"channel": "tickers", "instId": "ETH-USDT-SWAP"},
            ],
        }
    ]
