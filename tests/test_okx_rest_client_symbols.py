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
        assert method_name == "fetch_tickers"
        return {}

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
        assert method_name == "fetch_tickers"
        return {
            "PEPE/USDT:USDT": {
                "symbol": "PEPE/USDT:USDT",
                "last": 0.000002355,
                "info": {
                    "instId": "PEPE-USDT-SWAP",
                    "vol24h": "5357584.8",
                    "volCcy24h": "53575848000000",
                },
            }
        }

    monkeypatch.setattr(client, "_get_exchange", fake_exchange)
    monkeypatch.setattr(client, "_ccxt_call", fake_ccxt_call)

    symbols = await client.get_available_symbols()

    assert symbols[0]["symbol"] == "PEPE/USDT"
    assert symbols[0]["volume_24h"] == pytest.approx(53_575_848_000_000)
    assert symbols[0]["volume_24h_contracts"] == pytest.approx(5_357_584.8)
    assert symbols[0]["notional_24h_usdt"] == pytest.approx(126_171_122.04)
