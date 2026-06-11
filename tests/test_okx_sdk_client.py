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
