from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import OrderStatus
from executor.okx_executor import OKXExecutor


def _entry_decision(symbol: str = "HOME/USDT") -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol=symbol,
        action=Action.LONG,
        confidence=0.8,
        reasoning="test entry",
        position_size_pct=0.1,
        suggested_leverage=3.0,
        raw_response={},
        feature_snapshot={"current_price": 1.0},
    )


def _manual_exit_decision(symbol: str = "HOME/USDT") -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol=symbol,
        action=Action.CLOSE_LONG,
        confidence=1.0,
        reasoning="manual close",
        position_size_pct=0.01,
        suggested_leverage=3.0,
        raw_response={"manual_close": True},
        feature_snapshot={"current_price": 1.0},
    )


class FakeCcxt:
    def __init__(self, *, open_orders=None, created_order=None, confirmed_order=None):
        self.urls = {"api": {"rest": "https://www.okx.com"}}
        self.hostname = "www.okx.com"
        self.open_orders = list(open_orders or [])
        self.positions = [
            {
                "symbol": "HOME/USDT:USDT",
                "side": "long",
                "contracts": 100.0,
                "contractSize": 1.0,
                "info": {"posSide": "long", "pos": "100"},
            }
        ]
        self.position_snapshots = None
        self.created_order = created_order or {
            "id": "entry-1",
            "symbol": "HOME/USDT:USDT",
            "side": "buy",
            "type": "market",
            "status": "open",
            "amount": 30.0,
            "filled": 0.0,
            "price": 1.0,
            "average": None,
            "info": {"state": "live", "ordId": "entry-1", "side": "buy", "ordType": "market"},
        }
        self.confirmed_order = confirmed_order or self.created_order
        self.create_calls: list[tuple[Any, ...]] = []

    def market(self, symbol):
        return {
            "symbol": symbol,
            "contractSize": 1.0,
            "limits": {"amount": {"min": 1.0}},
        }

    def amount_to_precision(self, _symbol, amount):
        return str(float(amount))

    async def fetch_ticker(self, _symbol):
        return {"last": 1.0}

    async def fetch_open_orders(self, _symbol, *args, **kwargs):
        return list(self.open_orders)

    async def fetch_positions(self, _symbols=None):
        if self.position_snapshots:
            index = min(len(self.position_snapshots) - 1, getattr(self, "_position_call", 0))
            self._position_call = index + 1
            return list(self.position_snapshots[index])
        return list(self.positions)

    async def create_order(self, symbol, order_type, side, quantity, price, params):
        self.create_calls.append((symbol, order_type, side, quantity, price, params))
        return dict(self.created_order)

    async def fetch_order(self, _order_id, _symbol):
        return dict(self.confirmed_order)


def _executor(fake_ccxt: FakeCcxt) -> OKXExecutor:
    executor = OKXExecutor(mode="paper")
    executor._connected = True
    executor._exchange = fake_ccxt

    async def fake_leverage(_decision):
        return {
            "ok": True,
            "target_leverage": 3.0,
            "actual_leverage": 3.0,
            "okx_max_leverage": 10.0,
        }

    executor._set_leverage_if_needed = fake_leverage  # type: ignore[method-assign]
    return executor


@pytest.mark.asyncio
async def test_open_entry_order_is_not_treated_as_filled_position():
    fake_ccxt = FakeCcxt()
    executor = _executor(fake_ccxt)

    result = await executor.place_order(
        _entry_decision(),
        account_id="ensemble_trader",
        override_balance=10.0,
    )

    assert len(fake_ccxt.create_calls) == 1
    assert result.status == OrderStatus.OPEN
    assert result.quantity == 0.0
    assert result.exchange_order_id == "entry-1"
    assert result.raw_response is not None
    assert result.raw_response["entry_tracking"] is True
    assert "尚未确认成交" in result.raw_response["message"]


@pytest.mark.asyncio
async def test_filled_entry_without_filled_quantity_stays_pending_tracking():
    fake_ccxt = FakeCcxt(
        created_order={
            "id": "entry-missing-fill",
            "symbol": "HOME/USDT:USDT",
            "side": "buy",
            "type": "market",
            "status": "closed",
            "amount": 30.0,
            "filled": 0.0,
            "price": 1.0,
            "average": 1.0,
            "info": {"state": "filled", "ordId": "entry-missing-fill", "side": "buy"},
        },
        confirmed_order={
            "id": "entry-missing-fill",
            "symbol": "HOME/USDT:USDT",
            "side": "buy",
            "type": "market",
            "status": "closed",
            "amount": 30.0,
            "filled": 0.0,
            "price": 1.0,
            "average": 1.0,
            "info": {"state": "filled", "ordId": "entry-missing-fill", "side": "buy"},
        },
    )
    executor = _executor(fake_ccxt)

    result = await executor.place_order(
        _entry_decision(),
        account_id="ensemble_trader",
        override_balance=10.0,
    )

    assert result.status == OrderStatus.PENDING
    assert result.quantity == 0.0
    assert result.exchange_order_id == "entry-missing-fill"
    assert result.raw_response is not None
    assert result.raw_response["entry_tracking"] is True
    assert result.raw_response["fill_quantity_missing"] is True
    assert "不会用下单数量冒充成交数量" in result.raw_response["message"]


@pytest.mark.asyncio
async def test_existing_active_entry_order_blocks_duplicate_submit():
    existing = {
        "id": "existing-entry",
        "symbol": "HOME/USDT:USDT",
        "side": "buy",
        "type": "market",
        "status": "open",
        "amount": 30.0,
        "filled": 0.0,
        "price": 1.0,
        "info": {"state": "live", "ordId": "existing-entry", "side": "buy", "ordType": "market"},
    }
    fake_ccxt = FakeCcxt(open_orders=[existing])
    executor = _executor(fake_ccxt)

    result = await executor.place_order(
        _entry_decision(),
        account_id="ensemble_trader",
        override_balance=10.0,
    )

    assert fake_ccxt.create_calls == []
    assert result.status == OrderStatus.OPEN
    assert result.quantity == 0.0
    assert result.exchange_order_id == "existing-entry"
    assert result.raw_response is not None
    assert result.raw_response["existing_entry_order"] is True
    assert "不会重复提交新的开仓单" in result.raw_response["message"]


@pytest.mark.asyncio
async def test_manual_close_exit_fraction_is_not_forced_to_five_percent():
    fake_ccxt = FakeCcxt(
        confirmed_order={
            "id": "manual-exit",
            "symbol": "HOME/USDT:USDT",
            "side": "sell",
            "type": "market",
            "status": "closed",
            "amount": 1.0,
            "filled": 1.0,
            "price": 1.0,
            "average": 1.0,
            "info": {"state": "filled", "ordId": "manual-exit", "side": "sell"},
        }
    )
    fake_ccxt.positions = [
        {
            "symbol": "HOME/USDT:USDT",
            "side": "long",
            "contracts": 100.0,
            "contractSize": 1.0,
            "info": {"posSide": "long", "pos": "100"},
        }
    ]
    fake_ccxt.position_snapshots = [
        list(fake_ccxt.positions),
        [
            {
                "symbol": "HOME/USDT:USDT",
                "side": "long",
                "contracts": 99.0,
                "contractSize": 1.0,
                "info": {"posSide": "long", "pos": "99"},
            }
        ],
    ]
    executor = _executor(fake_ccxt)

    result = await executor.place_order(
        _manual_exit_decision(),
        account_id="ensemble_trader",
    )

    assert fake_ccxt.create_calls[0][3] == pytest.approx(1.0)
    assert result.status == OrderStatus.FILLED
    assert result.raw_response is not None
    assert result.raw_response["requested_exit_fraction"] == pytest.approx(0.01)
