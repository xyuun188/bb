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


@pytest.mark.asyncio
async def test_available_symbols_use_okx_inst_id_over_ccxt_alias(monkeypatch) -> None:
    client = OKXRestClient()
    exchange = _AliasMarketExchange()

    async def fake_exchange():
        return exchange

    async def fake_ccxt_call(method_name: str, *args, **kwargs):
        assert method_name == "fetch_tickers"
        return {}

    monkeypatch.setattr(client, "_get_exchange", fake_exchange)
    monkeypatch.setattr(client, "_ccxt_call", fake_ccxt_call)

    symbols = await client.get_available_symbols()

    assert symbols[0]["symbol"] == "H/USDT"
    assert symbols[0]["id"] == "H-USDT-SWAP"
    assert symbols[0]["ccxt_symbol"] == "WLFI/USDT:USDT"
