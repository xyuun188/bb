from __future__ import annotations

import inspect
import re
from typing import Any

import pytest

from core.exceptions import ExchangeAPIError
from data_feed import okx_sdk_client


class _FailingMarketApi:
    def __init__(self, message: str) -> None:
        self.message = message

    async def publicGetMarketTickers(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"code": "51000", "msg": self.message}


class _TickerMarketApi:
    async def publicGetMarketTickers(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {
            "code": "0",
            "data": [
                {"instId": "BTC-USDT-SWAP", "last": "50000", "open24h": "49000", "bidPx": "", "askPx": ""},
                {"instId": "PLTR-USDT-SWAP", "last": "64.46", "open24h": "63.00"},
            ],
        }


class _FailingAccountApi:
    def __init__(self, message: str) -> None:
        self.message = message

    async def privateGetAccountBalance(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"code": "51001", "msg": self.message}


class _InstrumentPublicApi:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    def get_instruments(
        self,
        instType: str,  # noqa: N803
        uly: str = "",
        instId: str = "",  # noqa: N803
        instFamily: str = "",  # noqa: N803
    ) -> dict[str, Any]:
        self.calls.append(
            {"instType": instType, "uly": uly, "instId": instId, "instFamily": instFamily}
        )
        return {"code": "0", "data": list(self.rows)}


class _Exchange:
    def __init__(
        self,
        *,
        market_api: Any | None = None,
        public_api: Any | None = None,
        account_api: Any | None = None,
        ohlcv_rows: list[list[float]] | None = None,
    ) -> None:
        self.market_api = market_api
        self.public_api = public_api
        self.account_api = account_api
        self.ohlcv_rows = ohlcv_rows or []
        self.ohlcv_calls: list[dict[str, Any]] = []

    async def publicGetMarketTickers(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self.market_api.publicGetMarketTickers(params)

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        result = self.public_api.get_instruments(
            instType=params.get("instType", ""),
            uly=params.get("uly", ""),
            instId=params.get("instId", ""),
            instFamily=params.get("instFamily", ""),
        )
        return result

    async def privateGetAccountBalance(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self.account_api.privateGetAccountBalance(params)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> list[list[float]]:
        self.ohlcv_calls.append({"symbol": symbol, "timeframe": timeframe, "limit": limit})
        return list(self.ohlcv_rows)


def _leaking_okx_message() -> tuple[str, str, str]:
    leaked_value = "abcdefghi" + "jklmnopqrst" + "uvwxyz123456"
    hidden_value = "plain-credential-value"
    message = f"Authorization: Bearer {leaked_value} failed password={hidden_value}"
    return leaked_value, hidden_value, message


def _instrument(inst_id: str, inst_category: str) -> dict[str, str]:
    return {
        "instType": "SWAP",
        "state": "live",
        "ctType": "linear",
        "settleCcy": "USDT",
        "instId": inst_id,
        "ctVal": "1",
        "minSz": "0.01",
        "tickSz": "0.0001",
        "instCategory": inst_category,
    }


def test_raise_okx_api_error_redacts_secret_bearing_message() -> None:
    leaked_value, hidden_value, message = _leaking_okx_message()

    with pytest.raises(ExchangeAPIError) as exc_info:
        okx_sdk_client._raise_okx_api_error({"code": "50011", "msg": message})

    rendered = str(exc_info.value)
    assert leaked_value not in rendered
    assert hidden_value not in rendered
    assert "Authorization: ***" in rendered
    assert "password=***" in rendered


def test_okx_sdk_client_uses_unified_adapter_only() -> None:
    source = inspect.getsource(okx_sdk_client)

    assert not re.search(r"^\s*from\s+okx(\.|\s)", source, flags=re.MULTILINE)
    assert not re.search(r"^\s*import\s+okx(\.|\s|$)", source, flags=re.MULTILINE)
    assert "MarketAPI(" not in source
    assert "PublicAPI(" not in source
    assert "AccountAPI(" not in source


@pytest.mark.asyncio
async def test_fetch_klines_routes_through_unified_exchange(monkeypatch) -> None:
    exchange = _Exchange(ohlcv_rows=[[1_783_592_000_000, 1.0, 1.2, 0.9, 1.1, 42.0]])
    monkeypatch.setattr(okx_sdk_client, "_make_exchange", lambda _mode: exchange)

    rows = await okx_sdk_client.fetch_klines("BTC/USDT", bar="1H", limit=1)

    assert exchange.ohlcv_calls == [{"symbol": "BTC/USDT", "timeframe": "1H", "limit": 1}]
    assert rows == [
        {
            "time": "2026-07-09T10:13:20+00:00",
            "open": 1.0,
            "high": 1.2,
            "low": 0.9,
            "close": 1.1,
            "volume": 42.0,
        }
    ]


@pytest.mark.asyncio
async def test_fetch_tickers_raises_typed_redacted_exchange_error(monkeypatch) -> None:
    leaked_value, hidden_value, message = _leaking_okx_message()

    monkeypatch.setattr(
        okx_sdk_client,
        "_make_exchange",
        lambda _mode: _Exchange(market_api=_FailingMarketApi(message), public_api=_InstrumentPublicApi([])),
    )

    with pytest.raises(ExchangeAPIError) as exc_info:
        await okx_sdk_client.fetch_tickers()

    rendered = str(exc_info.value)
    assert leaked_value not in rendered
    assert hidden_value not in rendered
    assert "Authorization: ***" in rendered
    assert "password=***" in rendered


@pytest.mark.asyncio
async def test_fetch_swap_tickers_filters_to_supported_crypto_instruments(monkeypatch) -> None:
    public_api = _InstrumentPublicApi(
        [
            _instrument("BTC-USDT-SWAP", "1"),
            _instrument("PLTR-USDT-SWAP", "3"),
        ]
    )

    monkeypatch.setattr(
        okx_sdk_client,
        "_make_exchange",
        lambda _mode: _Exchange(market_api=_TickerMarketApi(), public_api=public_api),
    )

    tickers = await okx_sdk_client.fetch_tickers(instType="SWAP")

    assert set(tickers) == {"BTC/USDT/SWAP"}
    assert tickers["BTC/USDT/SWAP"]["bid"] == 0.0
    assert tickers["BTC/USDT/SWAP"]["ask"] == 0.0
    assert public_api.calls == [
        {"instType": "SWAP", "uly": "", "instId": "", "instFamily": ""}
    ]


@pytest.mark.asyncio
async def test_get_available_symbols_uses_demo_instrument_universe_for_paper(
    monkeypatch,
) -> None:
    public_api = _InstrumentPublicApi(
        [
            _instrument("BTC-USDT-SWAP", "1"),
            _instrument("BCH-USDT-SWAP", "1"),
            {**_instrument("LINEA-USDT-SWAP", "1"), "uly": "LINEA-USDT"},
            {**_instrument("SKY-USDT-SWAP", "1"), "uly": "LINEA-USDT"},
            _instrument("PLTR-USDT-SWAP", "3"),
        ]
    )

    monkeypatch.setattr(
        okx_sdk_client,
        "_make_exchange",
        lambda _mode: _Exchange(public_api=public_api),
    )

    symbols = await okx_sdk_client.get_available_symbols(mode="paper")

    assert [symbol["id"] for symbol in symbols] == [
        "BTC-USDT-SWAP",
        "BCH-USDT-SWAP",
        "SKY-USDT-SWAP",
    ]
    assert public_api.calls == [
        {"instType": "SWAP", "uly": "", "instId": "", "instFamily": ""}
    ]


@pytest.mark.asyncio
async def test_fetch_usdt_balance_keeps_none_fallback_on_typed_api_error(monkeypatch) -> None:
    _leaked_value, _hidden_value, message = _leaking_okx_message()

    monkeypatch.setattr(
        okx_sdk_client,
        "_make_exchange",
        lambda _mode: _Exchange(account_api=_FailingAccountApi(message)),
    )

    assert await okx_sdk_client.fetch_usdt_balance() is None
