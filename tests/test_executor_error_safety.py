from __future__ import annotations

from typing import Any

import pytest

import executor.okx_executor as okx_module
from ai_brain.base_model import Action, DecisionOutput
from core.exceptions import OrderPlacementError
from executor.okx_executor import OKXExecutor


class _FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def warning(self, message: str, **kwargs: Any) -> None:
        self.events.append(("warning", message, kwargs))


class _FailingBalanceCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self, error_text: str) -> None:
        self.error_text = error_text

    async def fetch_balance(self) -> dict[str, Any]:
        raise RuntimeError(self.error_text)


class _FailingCancelCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self, error_text: str) -> None:
        self.error_text = error_text

    async def cancel_order(self, _order_id: str, _symbol: str) -> dict[str, Any]:
        raise RuntimeError(self.error_text)


class _FailingOpenOrdersCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self, error_text: str) -> None:
        self.error_text = error_text

    async def fetch_open_orders(self, _symbol: str | None = None) -> list[dict[str, Any]]:
        raise RuntimeError(self.error_text)


class _FailingPositionsForExitCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self, error_text: str) -> None:
        self.error_text = error_text

    def market(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "contractSize": 1.0,
            "limits": {"amount": {"min": 1.0}},
        }

    async def fetch_ticker(self, _symbol: str) -> dict[str, Any]:
        return {"last": 100.0}

    async def fetch_positions(self, _symbols: list[str] | None = None) -> list[dict[str, Any]]:
        raise RuntimeError(self.error_text)


class _FailingPositionsAfterExitSubmitCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self, error_text: str) -> None:
        self.error_text = error_text
        self.fetch_positions_calls = 0

    def market(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "contractSize": 1.0,
            "limits": {"amount": {"min": 1.0}},
        }

    def amount_to_precision(self, _symbol: str, amount: float) -> str:
        return str(float(amount))

    async def fetch_ticker(self, _symbol: str) -> dict[str, Any]:
        return {"last": 100.0}

    async def fetch_positions(self, _symbols: list[str] | None = None) -> list[dict[str, Any]]:
        self.fetch_positions_calls += 1
        if self.fetch_positions_calls == 1:
            return [
                {
                    "symbol": "BTC/USDT:USDT",
                    "side": "long",
                    "contracts": 2.0,
                    "contractSize": 1.0,
                    "info": {"posSide": "long", "pos": "2"},
                }
            ]
        raise RuntimeError(self.error_text)

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: float | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "id": "exit-1",
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": quantity,
            "filled": 0.0,
            "price": price or 100.0,
            "average": 100.0,
            "status": "open",
            "info": {"state": "live", "ordId": "exit-1", "side": side},
        }

    async def fetch_order(self, _order_id: str, _symbol: str) -> dict[str, Any]:
        return {
            "id": "exit-1",
            "filled": 0.0,
            "average": 100.0,
            "status": "open",
            "info": {"state": "live", "ordId": "exit-1"},
        }


class _LeverageUnknownAfterOpenOrderLimitCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self, error_text: str) -> None:
        self.error_text = error_text

    def market(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "contractSize": 1.0,
            "limits": {"amount": {"min": 1.0}},
            "info": {"instId": "BTC-USDT-SWAP"},
        }

    async def fetch_market_leverage_tiers(self, _symbol: str) -> list[dict[str, Any]]:
        return [{"maxLeverage": 20}]

    async def privateGetAccountAdjustLeverageInfo(
        self,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        return {"data": [{"maxLever": "20"}]}

    async def fetch_leverage(
        self,
        _symbol: str,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        raise RuntimeError(self.error_text)

    async def set_leverage(
        self,
        _leverage: int,
        _symbol: str,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        raise RuntimeError(f"OKX 59670 open order limit: {self.error_text}")

    async def fetch_open_orders(self, _symbol: str | None = None) -> list[dict[str, Any]]:
        return []

    async def fetch_positions(self, _symbols: list[str] | None = None) -> list[dict[str, Any]]:
        raise RuntimeError(self.error_text)


class _PrecisionEntryCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self) -> None:
        self.create_calls: list[tuple[Any, ...]] = []

    def market(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "contractSize": 1.0,
            "limits": {"amount": {"min": 1.0}},
            "info": {"instId": "SHIB-USDT-SWAP"},
        }

    def amount_to_precision(self, _symbol: str, amount: float) -> str:
        return str(float(amount))

    def price_to_precision(self, _symbol: str, price: float) -> str:
        return f"{float(price):.9f}"

    async def fetch_ticker(self, _symbol: str) -> dict[str, Any]:
        return {"last": 0.000008789, "bid": 0.000008788, "ask": 0.00000879}

    async def fetch_open_orders(self, _symbol: str | None = None) -> list[dict[str, Any]]:
        return []

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: float | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self.create_calls.append((symbol, order_type, side, quantity, price, params))
        return {
            "id": "entry-shib",
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": quantity,
            "filled": quantity,
            "price": price or 0.000008789,
            "average": 0.000008789,
            "status": "closed",
            "info": {"state": "filled", "ordId": "entry-shib", "side": side},
        }

    async def fetch_order(self, _order_id: str, _symbol: str) -> dict[str, Any]:
        return {
            "id": "entry-shib",
            "filled": 1.0,
            "average": 0.000008789,
            "status": "closed",
            "info": {"state": "filled", "ordId": "entry-shib"},
        }


def _executor(exchange: Any) -> OKXExecutor:
    executor = OKXExecutor(mode="paper")
    executor._connected = True
    executor._exchange = exchange
    return executor


def _exit_decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.8,
        reasoning="test exit",
        position_size_pct=1.0,
        suggested_leverage=3.0,
        raw_response={},
        feature_snapshot={"current_price": 100.0},
    )


def _entry_decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.8,
        reasoning="test entry",
        position_size_pct=0.1,
        suggested_leverage=5.0,
        raw_response={},
        feature_snapshot={"current_price": 100.0},
    )


def _shib_entry_decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="SHIB/USDT",
        action=Action.LONG,
        confidence=0.8,
        reasoning="test shib entry",
        position_size_pct=0.1,
        suggested_leverage=3.0,
        stop_loss_pct=0.012,
        take_profit_pct=0.024,
        raw_response={},
        feature_snapshot={"current_price": 0.000008789},
    )


def _secret_bearing_error() -> tuple[str, str, str]:
    token = "abcdefghi" + "jklmnopqrst" + "uvwxyz123456"
    hidden_value = "plain-credential-value"
    return token, hidden_value, f"Authorization: Bearer {token} failed password={hidden_value}"


@pytest.mark.asyncio
async def test_okx_balance_snapshot_error_is_redacted() -> None:
    token, hidden_value, error_text = _secret_bearing_error()
    result = await _executor(_FailingBalanceCcxt(error_text)).get_balance_snapshot()

    rendered = str(result)
    assert token not in rendered
    assert hidden_value not in rendered
    assert "Authorization: ***" in result["error"]
    assert "password=***" in result["error"]


@pytest.mark.asyncio
async def test_okx_cancel_replace_error_is_redacted() -> None:
    token, hidden_value, error_text = _secret_bearing_error()
    result = await _executor(_FailingCancelCcxt(error_text))._cancel_stale_exit_order(
        _FailingCancelCcxt(error_text),
        {},
        "BTC/USDT:USDT",
        "order-1",
        30.0,
    )

    rendered = str(result)
    assert result["cancel_success"] is False
    assert token not in rendered
    assert hidden_value not in rendered
    assert "Authorization: ***" in result["cancel_error"]
    assert "password=***" in result["cancel_error"]


@pytest.mark.asyncio
async def test_okx_open_orders_failure_is_logged_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token, hidden_value, error_text = _secret_bearing_error()
    fake_logger = _FakeLogger()
    monkeypatch.setattr(okx_module, "logger", fake_logger)

    result = await _executor(_FailingOpenOrdersCcxt(error_text)).get_open_orders("BTC/USDT")

    assert result == []
    assert fake_logger.events
    level, message, fields = fake_logger.events[-1]
    assert level == "warning"
    assert message == "fetch open orders failed"
    assert fields["symbol"] == "BTC/USDT:USDT"
    rendered = str(fields)
    assert token not in rendered
    assert hidden_value not in rendered
    assert "Authorization: ***" in fields["error"]
    assert "password=***" in fields["error"]


@pytest.mark.asyncio
async def test_okx_exit_position_lookup_failure_does_not_return_no_position() -> None:
    token, hidden_value, error_text = _secret_bearing_error()
    executor = _executor(_FailingPositionsForExitCcxt(error_text))

    with pytest.raises(OrderPlacementError) as exc_info:
        await executor.place_order(_exit_decision(), account_id="ensemble_trader")

    message = str(exc_info.value)
    assert token not in message
    assert hidden_value not in message
    assert "Authorization: ***" in message
    assert "password=***" in message
    assert "no_position" not in message


@pytest.mark.asyncio
async def test_okx_exit_after_submit_position_refresh_failure_is_tracked_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token, hidden_value, error_text = _secret_bearing_error()
    exchange = _FailingPositionsAfterExitSubmitCcxt(error_text)
    executor = _executor(exchange)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(okx_module.asyncio, "sleep", no_sleep)

    result = await executor.place_order(_exit_decision(), account_id="ensemble_trader")

    assert result.order_id == "exit-1"
    assert result.status == okx_module.OrderStatus.PENDING
    assert result.raw_response is not None
    assert result.raw_response["position_snapshot_unknown"] is True
    assert result.raw_response["position_contracts_after"] is None
    assert result.raw_response["remaining_contracts"] is None
    rendered = str(result.raw_response)
    assert token not in rendered
    assert hidden_value not in rendered
    assert "Authorization: ***" in rendered
    assert "password=***" in rendered


def test_okx_attached_protection_uses_market_price_precision() -> None:
    exchange = _PrecisionEntryCcxt()
    executor = _executor(exchange)
    decision = _shib_entry_decision()
    stop_loss, take_profit = executor._attached_sl_tp_prices(
        decision,
        0.000008789,
        ticker={"last": 0.000008789, "bid": 0.000008788, "ask": 0.00000879},
    )

    result = executor._format_attached_sl_tp_prices(
        exchange,
        "SHIB/USDT:USDT",
        decision,
        stop_loss,
        take_profit,
        0.000008789,
    )

    assert result["ok"] is True
    assert result["stop_loss_price"] == "0.000008683"
    assert result["take_profit_price"] == "0.000009001"


def test_okx_attached_protection_rejects_invalid_direction_after_precision() -> None:
    class RoundedToReferenceCcxt(_PrecisionEntryCcxt):
        def price_to_precision(self, _symbol: str, _price: float) -> str:
            return "0.000008789"

    exchange = RoundedToReferenceCcxt()
    executor = _executor(exchange)
    result = executor._format_attached_sl_tp_prices(
        exchange,
        "SHIB/USDT:USDT",
        _shib_entry_decision(),
        0.00000868,
        0.000009,
        0.000008789,
    )

    assert result["ok"] is False
    assert result["stop_loss_price"] == "0.000008789"
    assert result["take_profit_price"] == "0.000008789"


@pytest.mark.asyncio
async def test_okx_leverage_open_order_limit_with_unknown_actual_rejects_entry() -> None:
    token, hidden_value, error_text = _secret_bearing_error()
    executor = _executor(_LeverageUnknownAfterOpenOrderLimitCcxt(error_text))

    result = await executor._set_leverage_if_needed(_entry_decision())

    rendered = str(result)
    assert result["ok"] is False
    assert result["target_leverage"] == 5
    assert result["actual_leverage"] is None
    assert "未知杠杆" in result["error"]
    assert "59670" in result["open_order_limit_error"]
    assert token not in rendered
    assert hidden_value not in rendered
    assert "Authorization: ***" in rendered
    assert "password=***" in rendered
