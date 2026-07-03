from __future__ import annotations

from typing import Any

import pytest

from core.exceptions import ExchangeAPIError
from data_feed import okx_sdk_client


class _FailingMarketApi:
    def __init__(self, message: str) -> None:
        self.message = message

    def get_tickers(self, instType: str) -> dict[str, Any]:  # noqa: N803
        return {"code": "51000", "msg": self.message}


class _TickerMarketApi:
    def get_tickers(self, instType: str) -> dict[str, Any]:  # noqa: N803
        return {
            "code": "0",
            "data": [
                {"instId": "BTC-USDT-SWAP", "last": "50000", "open24h": "49000"},
                {"instId": "PLTR-USDT-SWAP", "last": "64.46", "open24h": "63.00"},
            ],
        }


class _FailingAccountApi:
    def __init__(self, message: str) -> None:
        self.message = message

    def get_account_balance(self, ccy: str) -> dict[str, Any]:
        return {"code": "51001", "msg": self.message}


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


@pytest.mark.asyncio
async def test_fetch_tickers_raises_typed_redacted_exchange_error(monkeypatch) -> None:
    leaked_value, hidden_value, message = _leaking_okx_message()

    monkeypatch.setattr(
        okx_sdk_client,
        "_make_market_api",
        lambda _mode: _FailingMarketApi(message),
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
    calls: list[dict[str, Any]] = []

    class _Response:
        def json(self) -> dict[str, Any]:
            return {
                "code": "0",
                "data": [
                    _instrument("BTC-USDT-SWAP", "1"),
                    _instrument("PLTR-USDT-SWAP", "3"),
                ],
            }

    import requests

    def fake_get(*args, **kwargs):
        calls.append(kwargs)
        return _Response()

    monkeypatch.setattr(okx_sdk_client, "_make_market_api", lambda _mode: _TickerMarketApi())
    monkeypatch.setattr(requests, "get", fake_get)

    tickers = await okx_sdk_client.fetch_tickers(instType="SWAP")

    assert set(tickers) == {"BTC/USDT/SWAP"}
    assert calls[0]["headers"] == {"x-simulated-trading": "1"}


@pytest.mark.asyncio
async def test_get_available_symbols_uses_demo_instrument_universe_for_paper(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class _Response:
        def json(self) -> dict[str, Any]:
            return {
                "code": "0",
                "data": [
                    _instrument("BTC-USDT-SWAP", "1"),
                    _instrument("BCH-USDT-SWAP", "1"),
                    {**_instrument("LINEA-USDT-SWAP", "1"), "uly": "LINEA-USDT"},
                    {**_instrument("SKY-USDT-SWAP", "1"), "uly": "LINEA-USDT"},
                    _instrument("PLTR-USDT-SWAP", "3"),
                ],
            }

    import requests

    def fake_get(*args, **kwargs):
        calls.append(kwargs)
        return _Response()

    monkeypatch.setattr(requests, "get", fake_get)

    symbols = await okx_sdk_client.get_available_symbols(mode="paper")

    assert [symbol["id"] for symbol in symbols] == [
        "BTC-USDT-SWAP",
        "BCH-USDT-SWAP",
        "SKY-USDT-SWAP",
    ]
    assert calls[0]["headers"] == {"x-simulated-trading": "1"}


@pytest.mark.asyncio
async def test_fetch_usdt_balance_keeps_none_fallback_on_typed_api_error(monkeypatch) -> None:
    _leaked_value, _hidden_value, message = _leaking_okx_message()

    monkeypatch.setattr(okx_sdk_client.settings, "okx_paper_api_key", "configured")
    monkeypatch.setattr(okx_sdk_client.settings, "okx_paper_api_secret", "configured")
    monkeypatch.setattr(okx_sdk_client.settings, "okx_paper_passphrase", "configured")
    monkeypatch.setattr(
        okx_sdk_client,
        "_make_account_api",
        lambda _mode: _FailingAccountApi(message),
    )

    assert await okx_sdk_client.fetch_usdt_balance() is None
