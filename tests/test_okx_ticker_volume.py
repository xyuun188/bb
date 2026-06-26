from __future__ import annotations

import json

import pytest

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
