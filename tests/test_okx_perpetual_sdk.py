from __future__ import annotations

from typing import Any

import pytest

from core.exceptions import ExchangeAPIError
from services.okx_perpetual_sdk import OkxPerpetualSdkExchange


class _PublicApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_instruments(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_instruments", dict(kwargs)))
        return {"code": "0", "data": []}

    def get_position_tiers(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_position_tiers", dict(kwargs)))
        return {"code": "0", "data": [{"maxLever": "20"}]}


class _MarketApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_tickers(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_tickers", dict(kwargs)))
        return {"code": "0", "data": []}


class _TradeApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def place_order(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("place_order", dict(kwargs)))
        return {"code": "0", "data": [{"ordId": "okx-1", "sCode": "0"}]}


class _AccountApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def set_leverage(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("set_leverage", dict(kwargs)))
        return {"code": "0", "data": [{"sCode": "0"}]}


@pytest.mark.asyncio
async def test_sdk_adapter_forces_public_tickers_to_swap() -> None:
    exchange = OkxPerpetualSdkExchange("paper")
    market_api = _MarketApi()
    exchange._market_api = market_api

    await exchange.publicGetMarketTickers({"instType": "SWAP"})

    assert market_api.calls == [("get_tickers", {"instType": "SWAP", "uly": "", "instFamily": ""})]


@pytest.mark.asyncio
async def test_sdk_adapter_rejects_non_swap_public_tickers() -> None:
    exchange = OkxPerpetualSdkExchange("paper")
    exchange._market_api = _MarketApi()

    with pytest.raises(ExchangeAPIError, match="Only OKX SWAP"):
        await exchange.publicGetMarketTickers({"instType": "SPOT"})


@pytest.mark.asyncio
async def test_sdk_adapter_places_perpetual_order_through_sdk() -> None:
    exchange = OkxPerpetualSdkExchange("paper")
    trade_api = _TradeApi()
    exchange._trade_api = trade_api

    order = await exchange.create_order(
        "BTC/USDT:USDT",
        "market",
        "buy",
        2,
        None,
        {"tdMode": "cross", "reduceOnly": False},
    )

    assert order["id"] == "okx-1"
    assert trade_api.calls == [
        (
            "place_order",
            {
                "instId": "BTC-USDT-SWAP",
                "tdMode": "cross",
                "side": "buy",
                "ordType": "market",
                "sz": "2",
                "ccy": "",
                "clOrdId": "",
                "tag": "",
                "posSide": "",
                "px": "",
                "reduceOnly": "false",
                "tgtCcy": "",
                "stpMode": "",
                "attachAlgoOrds": None,
            },
        )
    ]


@pytest.mark.asyncio
async def test_sdk_adapter_rejects_spot_order_shape() -> None:
    exchange = OkxPerpetualSdkExchange("paper")
    exchange._trade_api = _TradeApi()

    with pytest.raises(ExchangeAPIError, match="Only OKX USDT perpetual swaps"):
        await exchange.privatePostTradeOrder(
            {"instId": "BTC-USDT", "tdMode": "cross", "side": "buy", "ordType": "market", "sz": "1"}
        )


@pytest.mark.asyncio
async def test_sdk_adapter_set_leverage_uses_swap_inst_id() -> None:
    exchange = OkxPerpetualSdkExchange("paper")
    account_api = _AccountApi()
    exchange._account_api = account_api

    await exchange.set_leverage(3, "ETH/USDT", {"mgnMode": "cross"})

    assert account_api.calls == [
        (
            "set_leverage",
            {
                "lever": "3",
                "mgnMode": "cross",
                "instId": "ETH-USDT-SWAP",
                "posSide": "",
            },
        )
    ]


@pytest.mark.asyncio
async def test_sdk_adapter_leverage_tiers_normalizes_max_leverage() -> None:
    exchange = OkxPerpetualSdkExchange("paper")
    public_api = _PublicApi()
    exchange._public_api = public_api

    tiers = await exchange.fetch_market_leverage_tiers("SOL/USDT")

    assert tiers[0]["maxLeverage"] == 20.0
    assert public_api.calls == [
        (
            "get_position_tiers",
            {
                "instType": "SWAP",
                "tdMode": "cross",
                "instId": "SOL-USDT-SWAP",
            },
        )
    ]
