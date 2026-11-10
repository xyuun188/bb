from __future__ import annotations

from typing import Any

import pytest

from config.settings import settings
from core.exceptions import ExchangeAPIError, OrderPlacementError
from executor.base_executor import OrderStatus
from executor.okx_executor import OKXExecutor

SYMBOL = "BTC-USDT-SWAP"


class _FakeCCXT:
    """Minimal OKX adapter double with a per-order fill ledger."""

    def __init__(self) -> None:
        self.create_order_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[dict[str, Any]] = []
        # Fill size reported by each successive poll of the post-only order.
        self.detail_sequence: list[float] = []
        # Reject the first post-only submit when set (OKX 51016 style).
        self.reject_post_only = False
        self.amount = 10.0
        # ordId -> {"type", "amount", "filled", "cancelled"}
        self.orders: dict[str, dict[str, Any]] = {}
        self._polled_post_only: str | None = None

    def price_to_precision(self, symbol: str, value: float) -> str:
        return f"{value:.1f}"

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self.amount = amount
        self.create_order_calls.append(
            {
                "symbol": symbol,
                "type": order_type,
                "side": side,
                "amount": amount,
                "price": price,
                "params": dict(params),
            }
        )
        if order_type == "post_only" and self.reject_post_only:
            self.reject_post_only = False
            raise ExchangeAPIError("OKX API error [51016]: post only order would cross")
        # A market order is acknowledged as fully filled, matching OKX.
        filled = 0.0 if order_type == "post_only" else amount
        order_id = f"order-{len(self.create_order_calls)}"
        self.orders[order_id] = {
            "type": order_type,
            "amount": amount,
            "filled": filled,
            "cancelled": False,
        }
        if order_type == "post_only":
            self._polled_post_only = order_id
        status = "live" if order_type == "post_only" else "filled"
        return {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "status": status,
            "amount": amount,
            "filled": filled,
            "price": price,
            "average": price,
            "fee": {"cost": 0.0},
            "info": {"ordId": order_id, "state": status},
        }

    async def privateGetTradeOrder(self, params: dict[str, Any]) -> dict[str, Any]:
        order_id = str(params.get("ordId") or "")
        record = self.orders.setdefault(
            order_id,
            {"type": "post_only", "amount": self.amount, "filled": 0.0, "cancelled": False},
        )
        if record["type"] == "post_only":
            # Each poll advances the resting order's cumulative fill.
            record["filled"] = self.detail_sequence.pop(0) if self.detail_sequence else 0.0
        filled = record["filled"]
        amount = record["amount"]
        if record["cancelled"]:
            state = "canceled" if filled <= 0 else "partially_filled"
        elif filled >= amount and filled > 0:
            state = "filled"
        elif filled > 0:
            state = "partially_filled"
        else:
            state = "live"
        return {
            "data": [
                {
                    "instId": SYMBOL,
                    "ordId": order_id,
                    "side": "buy",
                    "ordType": record["type"],
                    "state": state,
                    "sz": str(amount),
                    "accFillSz": str(filled),
                    "avgPx": "100.0" if filled > 0 else "0",
                    "fee": "0",
                }
            ]
        }

    async def privatePostTradeCancelOrder(self, params: dict[str, Any]) -> dict[str, Any]:
        self.cancel_calls.append(dict(params))
        record = self.orders.get(str(params.get("ordId") or ""))
        if record is not None:
            record["cancelled"] = True
        return {"code": "0", "data": [{"ordId": params.get("ordId"), "sCode": "0"}]}


def _executor() -> OKXExecutor:
    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    executor._connected = True
    return executor


@pytest.fixture(autouse=True)
def _enable_post_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "okx_entry_post_only_enabled", True, raising=False)
    monkeypatch.setattr(settings, "okx_entry_post_only_timeout_seconds", 1.0, raising=False)
    monkeypatch.setattr(settings, "okx_entry_post_only_poll_seconds", 0.01, raising=False)


async def _place(
    executor: OKXExecutor,
    ccxt: _FakeCCXT,
    *,
    side: str = "buy",
    quantity: float = 10.0,
    bid: float = 99.0,
    ask: float = 101.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return await executor._place_entry_with_post_only(
        ccxt,
        okx_symbol=SYMBOL,
        side=side,
        quantity=quantity,
        params={"tdMode": "cross", "clOrdId": "BBNP123456"},
        market={"symbol": SYMBOL, "info": {"tickSz": "0.1"}},
        ticker={"bid": bid, "ask": ask},
    )


@pytest.mark.asyncio
async def test_post_only_full_fill_never_touches_the_market() -> None:
    executor = _executor()
    ccxt = _FakeCCXT()
    ccxt.detail_sequence = [10.0]

    order, fact = await _place(executor, ccxt)

    assert [call["type"] for call in ccxt.create_order_calls] == ["post_only"]
    assert ccxt.create_order_calls[0]["price"] == pytest.approx(99.0)
    assert ccxt.cancel_calls == []
    assert fact["post_only_outcome"] == "filled"
    assert fact["maker_fill_contracts"] == pytest.approx(10.0)
    assert fact["taker_fill_contracts"] == pytest.approx(0.0)
    assert order["status"] == "filled"


@pytest.mark.asyncio
async def test_post_only_rejection_falls_back_to_market() -> None:
    executor = _executor()
    ccxt = _FakeCCXT()
    ccxt.reject_post_only = True

    order, fact = await _place(executor, ccxt)

    assert [call["type"] for call in ccxt.create_order_calls] == ["post_only", "market"]
    assert fact["post_only_outcome"] == "rejected"
    assert fact["fallback_reason"] == "post_only_rejected_51016"
    assert fact["taker_fill_contracts"] == pytest.approx(10.0)
    assert order["status"] == "filled"


@pytest.mark.asyncio
async def test_post_only_timeout_cancels_then_market_fills_everything() -> None:
    executor = _executor()
    ccxt = _FakeCCXT()
    ccxt.detail_sequence = [0.0, 0.0, 0.0]

    order, fact = await _place(executor, ccxt)

    assert [call["type"] for call in ccxt.create_order_calls] == ["post_only", "market"]
    assert ccxt.create_order_calls[1]["amount"] == pytest.approx(10.0)
    assert ccxt.cancel_calls, "the resting order must be cancelled before the fallback"
    assert fact["post_only_outcome"] == "market_fallback"
    assert fact["fallback_reason"] == "post_only_timeout"
    assert fact["maker_fill_contracts"] == pytest.approx(0.0)
    assert fact["taker_fill_contracts"] == pytest.approx(10.0)
    assert order["status"] == "filled"


@pytest.mark.asyncio
async def test_partial_maker_fill_only_markets_the_remainder() -> None:
    executor = _executor()
    ccxt = _FakeCCXT()
    ccxt.detail_sequence = [4.0, 0.0, 0.0]

    order, fact = await _place(executor, ccxt)

    assert [call["type"] for call in ccxt.create_order_calls] == ["post_only", "market"]
    assert ccxt.create_order_calls[1]["amount"] == pytest.approx(6.0)
    assert fact["post_only_outcome"] == "partial_then_market_fallback"
    assert fact["fallback_reason"] == "post_only_partial_timeout"
    assert fact["maker_fill_contracts"] == pytest.approx(4.0)
    assert fact["taker_fill_contracts"] == pytest.approx(6.0)
    assert order["filled"] == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_disabled_post_only_submits_a_single_market_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "okx_entry_post_only_enabled", False, raising=False)
    executor = _executor()
    ccxt = _FakeCCXT()

    order, fact = await _place(executor, ccxt)

    assert [call["type"] for call in ccxt.create_order_calls] == ["market"]
    assert fact["entry_execution_style"] == "market"
    assert fact["post_only_attempted"] is False
    assert order["status"] == "filled"


@pytest.mark.asyncio
async def test_a_touch_that_would_cross_skips_straight_to_market() -> None:
    executor = _executor()
    ccxt = _FakeCCXT()
    # The ask sits below the tick-rounded bid, so resting there would cross.
    order, fact = await _place(executor, ccxt, bid=100.0, ask=99.9)

    assert [call["type"] for call in ccxt.create_order_calls] == ["market"]
    assert fact["post_only_attempted"] is False
    assert order["status"] == "filled"


@pytest.mark.asyncio
async def test_missing_book_skips_straight_to_market() -> None:
    executor = _executor()
    ccxt = _FakeCCXT()

    order, fact = await _place(executor, ccxt, bid=0.0, ask=0.0)

    assert [call["type"] for call in ccxt.create_order_calls] == ["market"]
    assert fact["post_only_attempted"] is False


@pytest.mark.asyncio
async def test_passive_leg_uses_its_own_client_order_id() -> None:
    executor = _executor()
    ccxt = _FakeCCXT()
    ccxt.detail_sequence = [10.0]

    await _place(executor, ccxt)

    assert ccxt.create_order_calls[0]["params"]["clOrdId"] == "BBNP123456P"


@pytest.mark.asyncio
async def test_uncancellable_passive_order_never_submits_a_duplicate() -> None:
    executor = _executor()
    ccxt = _FakeCCXT()
    ccxt.detail_sequence = [0.0, 0.0]

    async def cancel_fails(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ExchangeAPIError("OKX native cancel failed")

    executor._cancel_order_native = cancel_fails  # type: ignore[method-assign]

    with pytest.raises(OrderPlacementError, match="could not be cancelled"):
        await _place(executor, ccxt)

    assert [call["type"] for call in ccxt.create_order_calls] == ["post_only"]


@pytest.mark.asyncio
async def test_sell_side_rests_above_the_touch() -> None:
    executor = _executor()
    ccxt = _FakeCCXT()
    ccxt.detail_sequence = [10.0]

    await _place(executor, ccxt, side="sell", bid=99.0, ask=101.0)

    assert ccxt.create_order_calls[0]["type"] == "post_only"
    assert ccxt.create_order_calls[0]["price"] == pytest.approx(101.0)
    assert ccxt.create_order_calls[0]["side"] == "sell"


@pytest.mark.asyncio
async def test_exit_orders_still_use_a_plain_market_submission() -> None:
    executor = _executor()
    ccxt = _FakeCCXT()

    order = await executor._create_order_with_client_recovery(
        ccxt, SYMBOL, "sell", 5.0, {"tdMode": "cross", "reduceOnly": True}
    )

    assert [call["type"] for call in ccxt.create_order_calls] == ["market"]
    assert ccxt.create_order_calls[0]["params"]["reduceOnly"] is True
    assert executor._order_status_from_ccxt(order["status"]) == OrderStatus.FILLED
