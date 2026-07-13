from __future__ import annotations

from typing import Any

import pytest

from core.exceptions import ExchangeAPIError
from services import okx_perpetual_sdk
from services.okx_perpetual_sdk import OkxPerpetualSdkExchange


def test_sdk_error_preserves_top_level_okx_code_and_payload() -> None:
    payload = {"code": "50026", "msg": "System error. Try again later.", "data": []}

    with pytest.raises(ExchangeAPIError) as captured:
        okx_perpetual_sdk.raise_if_okx_error(payload)

    assert captured.value.code == "50026"
    assert captured.value.payload == payload


def test_sdk_error_preserves_order_item_code_and_payload() -> None:
    item = {"ordId": "", "sCode": "51008", "sMsg": "Insufficient USDT margin"}
    payload = {"code": "0", "msg": "", "data": [item]}

    with pytest.raises(ExchangeAPIError) as captured:
        okx_perpetual_sdk.raise_if_okx_error(payload, check_data_code=True)

    assert captured.value.code == "51008"
    assert captured.value.payload == {"response": payload, "item": item}


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

    def get_fee_rates(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_fee_rates", dict(kwargs)))
        return {"code": "0", "data": [{"taker": "-0.0005", "ts": "1783931709453"}]}


class _ServerTimeResponse:
    def __init__(self, server_ms: int) -> None:
        self.server_ms = server_ms

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"code": "0", "data": [{"ts": str(self.server_ms)}]}


class _PrivateApiForTime:
    def __init__(self, server_ms: int) -> None:
        self.server_ms = server_ms
        self.use_server_time = False
        self.urls: list[str] = []

    def get(self, url: str) -> _ServerTimeResponse:
        self.urls.append(url)
        return _ServerTimeResponse(self.server_ms)


def test_sdk_adapter_enables_cached_okx_server_time_for_private_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = OkxPerpetualSdkExchange("paper")
    api = exchange._configure_private_api(_PrivateApiForTime(1_783_592_000_123))
    clock = {"wall": 1000.0, "mono": 10.0}
    monkeypatch.setattr(okx_perpetual_sdk.time, "time", lambda: clock["wall"])
    monkeypatch.setattr(okx_perpetual_sdk.time, "monotonic", lambda: clock["mono"])

    first = api._get_timestamp()
    clock["wall"] = 1001.0
    clock["mono"] = 11.0
    second = api._get_timestamp()

    assert api.use_server_time is True
    assert first == okx_perpetual_sdk._timestamp_from_epoch_ms(1_783_592_000_123)
    assert second == okx_perpetual_sdk._timestamp_from_epoch_ms(1_783_592_001_123)
    assert api.urls == [f"{okx_perpetual_sdk.OKX_DOMAIN}{okx_perpetual_sdk.OKX_SERVER_TIME_PATH}"]


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
async def test_sdk_adapter_reads_account_level_swap_fee_rate() -> None:
    exchange = OkxPerpetualSdkExchange("paper")
    account_api = _AccountApi()
    exchange._account_api = account_api

    response = await exchange.privateGetAccountFeeRates({"instType": "SWAP"})

    assert response["data"][0]["taker"] == "-0.0005"
    assert account_api.calls == [
        (
            "get_fee_rates",
            {
                "instType": "SWAP",
                "instId": "",
                "uly": "",
                "category": "",
                "instFamily": "",
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
