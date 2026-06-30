from pathlib import Path
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from core.exceptions import ExchangeAPIError
from core.symbols import okx_inst_id_from_symbol
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
    def __init__(
        self,
        *,
        open_orders=None,
        created_order=None,
        confirmed_order=None,
        amount_min=1.0,
        contract_size=1.0,
    ):
        self.urls = {"api": {"rest": "https://www.okx.com"}}
        self.hostname = "www.okx.com"
        self.open_orders = list(open_orders or [])
        self.amount_min = amount_min
        self.contract_size = contract_size
        self.positions = [
            {
                "symbol": "HOME/USDT:USDT",
                "side": "long",
                "contracts": 100.0,
                "contractSize": contract_size,
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
        self.native_order_detail_requests: list[dict[str, Any]] = []
        self.native_cancel_order_requests: list[dict[str, Any]] = []

    def market(self, symbol):
        return {
            "symbol": symbol,
            "id": okx_inst_id_from_symbol(symbol),
            "contractSize": self.contract_size,
            "limits": {"amount": {"min": self.amount_min}},
            "info": {
                "instId": okx_inst_id_from_symbol(symbol),
                "ctVal": str(self.contract_size),
                "minSz": str(self.amount_min),
                "lotSz": str(self.amount_min),
            },
        }

    def amount_to_precision(self, _symbol, amount):
        return str(float(amount))

    async def fetch_ticker(self, _symbol):
        raise AssertionError("execution sizing must use OKX native ticker API")

    async def publicGetMarketTicker(self, params):
        inst_id = str(params.get("instId") or "HOME-USDT-SWAP").strip().upper()
        return {
            "data": [
                {
                    "instId": inst_id,
                    "last": "1.0",
                    "bidPx": "0.999",
                    "askPx": "1.001",
                    "ts": "1780000000000",
                }
            ]
        }

    async def fetch_open_orders(self, _symbol, *args, **kwargs):
        raise AssertionError("entry guards must use OKX native pending-orders API")

    async def fetch_positions(self, _symbols=None):
        raise AssertionError("position reads must use OKX native positions API")

    async def privateGetTradeOrdersPending(self, params):
        inst_id = str(params.get("instId") or "").strip().upper()
        rows = []
        for order in self.open_orders:
            info = dict(order.get("info") or {})
            order_inst_id = str(
                info.get("instId")
                or order.get("instId")
                or okx_inst_id_from_symbol(order.get("symbol"))
                or ""
            ).strip().upper()
            if inst_id and order_inst_id and order_inst_id != inst_id:
                continue
            rows.append(
                {
                    "instId": order_inst_id or inst_id or "HOME-USDT-SWAP",
                    "ordId": order.get("id") or info.get("ordId"),
                    "side": order.get("side") or info.get("side"),
                    "ordType": order.get("type") or info.get("ordType") or "market",
                    "state": info.get("state") or order.get("status") or "live",
                    "sz": str(order.get("amount") or info.get("sz") or "0"),
                    "accFillSz": str(order.get("filled") or info.get("accFillSz") or "0"),
                    "reduceOnly": str(info.get("reduceOnly") or "false").lower(),
                }
            )
        return {"data": rows}

    async def privateGetAccountPositions(self, params):
        if self.position_snapshots:
            index = min(len(self.position_snapshots) - 1, getattr(self, "_position_call", 0))
            self._position_call = index + 1
            positions = list(self.position_snapshots[index])
        else:
            positions = list(self.positions)
        inst_id = str(params.get("instId") or "").strip().upper()
        rows = []
        for position in positions:
            info = dict(position.get("info") or {})
            pos_inst_id = str(
                info.get("instId")
                or position.get("instId")
                or okx_inst_id_from_symbol(position.get("symbol"))
                or ""
            ).strip().upper()
            if inst_id and pos_inst_id and pos_inst_id != inst_id:
                continue
            rows.append(
                {
                    "instId": pos_inst_id or inst_id or "HOME-USDT-SWAP",
                    "posSide": info.get("posSide") or position.get("side") or "net",
                    "pos": str(position.get("contracts") or info.get("pos") or "0"),
                    "ctVal": str(position.get("contractSize") or self.contract_size),
                    "avgPx": str(position.get("entryPrice") or info.get("avgPx") or "1"),
                    "markPx": str(position.get("markPrice") or info.get("markPx") or "1"),
                    "upl": str(position.get("unrealizedPnl") or info.get("upl") or "0"),
                }
            )
        return {"data": rows}

    async def legacy_fetch_positions_fixture(self, _symbols=None):
        if self.position_snapshots:
            index = min(len(self.position_snapshots) - 1, getattr(self, "_position_call", 0))
            self._position_call = index + 1
            return list(self.position_snapshots[index])
        return list(self.positions)

    async def create_order(self, symbol, order_type, side, quantity, price, params):
        self.create_calls.append((symbol, order_type, side, quantity, price, params))
        return dict(self.created_order)

    async def privateGetTradeOrder(self, params):
        self.native_order_detail_requests.append(dict(params))
        order = dict(self.confirmed_order)
        info = dict(order.get("info") or {})
        inst_id = str(
            info.get("instId")
            or order.get("instId")
            or okx_inst_id_from_symbol(order.get("symbol"))
            or params.get("instId")
            or ""
        ).strip().upper()
        return {
            "data": [
                {
                    "instId": inst_id or "HOME-USDT-SWAP",
                    "ordId": order.get("id") or info.get("ordId") or params.get("ordId"),
                    "side": order.get("side") or info.get("side") or "buy",
                    "ordType": order.get("type") or info.get("ordType") or "market",
                    "state": info.get("state") or order.get("status") or "live",
                    "sz": str(order.get("amount") or info.get("sz") or "0"),
                    "accFillSz": str(order.get("filled") or info.get("accFillSz") or "0"),
                    "avgPx": str(order.get("average") or info.get("avgPx") or "0"),
                    "px": str(order.get("price") or info.get("px") or "0"),
                    "fee": str(
                        (order.get("fee") or {}).get("cost")
                        if isinstance(order.get("fee"), dict)
                        else "0"
                    ),
                }
            ]
        }

    async def fetch_order(self, _order_id, _symbol):
        raise AssertionError("order confirmation must use OKX native privateGetTradeOrder")

    async def privatePostTradeCancelOrder(self, params):
        self.native_cancel_order_requests.append(dict(params))
        return {"code": "0", "data": [{"instId": params.get("instId"), "ordId": params.get("ordId")}]}

    async def cancel_order(self, _order_id, _symbol):
        raise AssertionError("order cancellation must use OKX native privatePostTradeCancelOrder")


class RejectingCreateOrderCcxt(FakeCcxt):
    async def create_order(self, symbol, order_type, side, quantity, price, params):
        self.create_calls.append((symbol, order_type, side, quantity, price, params))
        raise ExchangeAPIError("51008 Insufficient USDT margin")


class MissingNativeTickerCcxt(FakeCcxt):
    async def publicGetMarketTicker(self, params):
        raise ExchangeAPIError("OKX native ticker unavailable")


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
async def test_entry_rejects_before_submit_when_okx_native_ticker_unavailable():
    fake_ccxt = MissingNativeTickerCcxt()
    executor = _executor(fake_ccxt)

    result = await executor.place_order(
        _entry_decision(),
        account_id="ensemble_trader",
        override_balance=10.0,
    )

    assert fake_ccxt.create_calls == []
    assert result.status == OrderStatus.REJECTED
    assert result.order_id == "ticker_unavailable"
    assert result.raw_response is not None
    assert result.raw_response["execution_blocker"] == "okx_native_ticker_unavailable"
    assert result.raw_response["system_pre_submit_rejection"] is True
    assert result.raw_response["okx_rejection"] is False


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
async def test_entry_size_lifts_to_okx_min_contracts_when_affordable():
    fake_ccxt = FakeCcxt(amount_min=10.0, contract_size=1.0)
    executor = _executor(fake_ccxt)
    decision = _entry_decision()
    decision.position_size_pct = 0.001
    decision.suggested_leverage = 1.0

    result = await executor.place_order(
        decision,
        account_id="ensemble_trader",
        override_balance=1_000.0,
    )

    assert fake_ccxt.create_calls
    assert fake_ccxt.create_calls[0][3] == pytest.approx(10.0)
    assert result.raw_response is not None
    rules = result.raw_response["okx_order_rules"]
    assert rules["amount_min_contracts"] == pytest.approx(10.0)
    assert rules["planned_contracts_raw"] == pytest.approx(3.0)
    assert rules["final_contracts"] == pytest.approx(10.0)
    assert rules["system_adjusted_to_min_contracts"] is True


@pytest.mark.asyncio
async def test_entry_confirmation_uses_okx_native_inst_id_over_ccxt_alias():
    fake_ccxt = FakeCcxt(
        created_order={
            "id": "spk-entry",
            "symbol": "SAHARA/USDT:USDT",
            "side": "buy",
            "type": "market",
            "status": "closed",
            "amount": 10.0,
            "filled": 10.0,
            "price": 0.012,
            "average": 0.012,
            "info": {
                "state": "filled",
                "ordId": "spk-entry",
                "side": "buy",
                "instId": "SAHARA-USDT-SWAP",
            },
        },
        confirmed_order={
            "id": "spk-entry",
            "symbol": "SAHARA/USDT:USDT",
            "side": "buy",
            "type": "market",
            "status": "filled",
            "amount": 10.0,
            "filled": 10.0,
            "price": 0.012,
            "average": 0.012,
            "info": {
                "state": "filled",
                "ordId": "spk-entry",
                "side": "buy",
                "instId": "SPK-USDT-SWAP",
            },
        },
    )
    executor = _executor(fake_ccxt)

    result = await executor.place_order(
        _entry_decision("SPK/USDT"),
        account_id="ensemble_trader",
        override_balance=1_000.0,
    )

    assert fake_ccxt.native_order_detail_requests == [
        {"instId": "SPK-USDT-SWAP", "ordId": "spk-entry"}
    ]
    assert result.status == OrderStatus.FILLED
    assert result.symbol == "SPK/USDT"
    assert result.exchange_order_id == "spk-entry"
    assert result.raw_response is not None
    assert result.raw_response["okx_native_order_detail"] is True
    assert result.raw_response["canonical_exchange_symbol"] == "SPK/USDT"


@pytest.mark.asyncio
async def test_cancel_order_uses_okx_native_inst_id_and_ord_id():
    fake_ccxt = FakeCcxt()
    executor = _executor(fake_ccxt)

    cancelled = await executor.cancel_order("spk-order-1", "SPK/USDT")

    assert cancelled is True
    assert fake_ccxt.native_cancel_order_requests == [
        {"instId": "SPK-USDT-SWAP", "ordId": "spk-order-1"}
    ]


@pytest.mark.asyncio
async def test_leverage_retry_cleanup_uses_okx_native_cancel_order():
    open_orders = [
        {
            "id": f"entry-{index}",
            "symbol": "SPK/USDT:USDT",
            "side": "buy",
            "type": "limit",
            "status": "open",
            "amount": 1.0,
            "filled": 0.0,
            "info": {
                "instId": "SPK-USDT-SWAP",
                "ordId": f"entry-{index}",
                "side": "buy",
                "ordType": "limit",
                "state": "live",
                "reduceOnly": "false",
                "cTime": str(1780000000000 + index),
            },
        }
        for index in range(7)
    ]
    fake_ccxt = FakeCcxt(open_orders=open_orders)
    executor = _executor(fake_ccxt)

    result = await executor._reduce_open_orders_for_leverage_retry(
        fake_ccxt,
        "SPK-USDT-SWAP",
    )

    assert result["checked"] is True
    assert result["cancelled"] == 2
    assert fake_ccxt.native_cancel_order_requests == [
        {"instId": "SPK-USDT-SWAP", "ordId": "entry-0"},
        {"instId": "SPK-USDT-SWAP", "ordId": "entry-1"},
    ]


@pytest.mark.asyncio
async def test_entry_size_rejects_before_okx_when_min_contracts_unaffordable():
    fake_ccxt = FakeCcxt(amount_min=10.0, contract_size=1.0)
    executor = _executor(fake_ccxt)
    decision = _entry_decision()
    decision.position_size_pct = 0.001
    decision.suggested_leverage = 1.0

    result = await executor.place_order(
        decision,
        account_id="ensemble_trader",
        override_balance=1.0,
    )

    assert fake_ccxt.create_calls == []
    assert result.status == OrderStatus.REJECTED
    assert result.raw_response is not None
    assert "\u63d0\u4ea4\u524d\u62e6\u622a" in result.raw_response["error"]
    assert result.raw_response["okx_min_order_notional_usdt"] > 0
    assert result.raw_response["system_pre_submit_rejection"] is True
    assert result.raw_response["okx_rejection"] is False
    rules = result.raw_response["okx_order_rules"]
    assert rules["pre_submit_valid"] is False
    assert rules["amount_min_contracts"] == pytest.approx(10.0)
    assert rules["affordable_notional_usdt"] == pytest.approx(1.0)
    assert rules["min_notional_usdt"] == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_okx_entry_exchange_rejection_is_structured_not_raised():
    fake_ccxt = RejectingCreateOrderCcxt(amount_min=1.0, contract_size=1.0)
    executor = _executor(fake_ccxt)

    result = await executor.place_order(
        _entry_decision(),
        account_id="ensemble_trader",
        override_balance=10.0,
    )

    assert fake_ccxt.create_calls
    assert result.status == OrderStatus.REJECTED
    assert result.order_id == "okx_rejected"
    assert result.raw_response is not None
    assert result.raw_response["okx_rejection"] is True
    assert result.raw_response["system_pre_submit_rejection"] is False
    assert result.raw_response["execution_blocker"] == "okx_exchange_rejection"
    assert "51008" in result.raw_response["raw_error"]
    assert result.raw_response["okx_order_rules"]["pre_submit_valid"] is True


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


def test_okx_executor_no_english_min_contract_rejection_message() -> None:
    source = Path("executor/okx_executor.py").read_text(encoding="utf-8")

    assert "Order size is below OKX minimum contract size" not in source
    assert "\u63d0\u4ea4\u524d\u62e6\u622a" in source
