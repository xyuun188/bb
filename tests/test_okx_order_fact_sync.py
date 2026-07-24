from __future__ import annotations

import asyncio
import inspect
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import select

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.trade import Order
from services import okx_order_fact_sync as order_fact_sync_module
from services.okx_execution_slippage import (
    OKX_FILL_MARK_SLIPPAGE_VERSION,
    build_okx_fill_mark_slippage,
)
from services.okx_native_facts import OkxNativeFillGroup
from services.okx_order_fact_sync import (
    OKX_SYNC_CONFIRMED,
    OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
    OKX_SYNC_ORDER_DETAIL_CONFIRMED,
    OkxOrderFactSyncService,
    _build_contract_size_catalog,
    _dedupe_fills_by_order_id,
    _prioritized_exchange_order_ids,
    _rebuild_stored_slippage_fact,
    _repair_stored_fill_contract_size_from_instruments,
    _stored_slippage_fact_needs_refresh,
)


class _ScalarResult:
    def __init__(self, value: Any) -> None:
        self.value = value

    def scalar(self) -> Any:
        return self.value


class _AdvisoryLockSession:
    def __init__(self, values: list[Any]) -> None:
        self.values = list(values)
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    async def execute(self, statement: Any, params: dict[str, Any]) -> _ScalarResult:
        self.calls.append((statement, params))
        return _ScalarResult(self.values.pop(0) if self.values else None)


class _FakeCcxt:
    def __init__(
        self,
        *,
        fills: list[dict[str, Any]] | None = None,
        orders: list[dict[str, Any]] | None = None,
        instruments: list[dict[str, Any]] | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.fills = list(fills or [])
        self.orders = list(orders or [])
        self.instruments = list(instruments) if instruments is not None else [
            {
                "instId": "BTC-USDT-SWAP",
                "instType": "SWAP",
                "ctVal": "0.01",
                "ctMult": "1",
                "lotSz": "1",
                "minSz": "1",
                "settleCcy": "USDT",
            }
        ]
        self.delay_seconds = delay_seconds
        self.calls: list[str] = []

    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("fills")
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        order_id = str(params.get("ordId") or "")
        rows = [row for row in self.fills if not order_id or row.get("ordId") == order_id]
        return {"data": rows}

    async def privateGetTradeFills(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self.privateGetTradeFillsHistory(params)

    async def privateGetTradeOrdersHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("orders")
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        order_id = str(params.get("ordId") or "")
        rows = [row for row in self.orders if not order_id or row.get("ordId") == order_id]
        return {"data": rows}

    async def privateGetTradeOrdersAlgoHistory(self, _params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("protection")
        return {"data": []}

    async def publicGetPublicInstruments(self, _params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("contract_specs")
        return {"data": self.instruments}


class _RecentOnlyCcxt(_FakeCcxt):
    async def privateGetTradeFills(self, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("fills")
        order_id = str(params.get("ordId") or "")
        rows = [row for row in self.fills if not order_id or row.get("ordId") == order_id]
        return {"data": rows}

    async def privateGetTradeFillsHistory(
        self,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append("fills")
        return {"data": []}


class _FakeExecutor:
    def __init__(self, ccxt: _FakeCcxt) -> None:
        self.ccxt = ccxt
        self.closed = False

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        self.closed = True

    async def _get_ccxt(self) -> _FakeCcxt:
        return self.ccxt

    async def _with_retry(self, fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result


def _executor_factory(ccxt: _FakeCcxt):
    def factory(*_args, **_kwargs) -> _FakeExecutor:
        return _FakeExecutor(ccxt)

    return factory


def _ms(value: datetime) -> str:
    return str(int(value.timestamp() * 1000))


def _fill_row(now: datetime, *, order_id: str = "okx-order-1") -> dict[str, Any]:
    return {
        "instId": "BTC-USDT-SWAP",
        "ordId": order_id,
        "tradeId": f"trade-{order_id}",
        "side": "buy",
        "posSide": "long",
        "fillSz": "2",
        "fillPx": "60000",
        "fillMarkPx": "59990",
        "fee": "-0.12",
        "fillPnl": "0",
        "ts": _ms(now),
    }


def _order_row(now: datetime, *, order_id: str = "okx-order-1") -> dict[str, Any]:
    return {
        "instId": "BTC-USDT-SWAP",
        "ordId": order_id,
        "side": "buy",
        "state": "filled",
        "ordType": "market",
        "sz": "2",
        "avgPx": "60000",
        "cTime": _ms(now),
        "fillTime": _ms(now),
    }


def _act_fill_row(now: datetime, *, order_id: str = "act-order-1") -> dict[str, Any]:
    return {
        "instId": "ACT-USDT-SWAP",
        "ordId": order_id,
        "tradeId": f"trade-{order_id}",
        "side": "buy",
        "posSide": "long",
        "fillSz": "4",
        "fillPx": "0.00895",
        "fillMarkPx": "0.00890",
        "fee": "-0.001",
        "fillPnl": "0",
        "ts": _ms(now),
    }


def _act_order_row(now: datetime, *, order_id: str = "act-order-1") -> dict[str, Any]:
    return {
        "instId": "ACT-USDT-SWAP",
        "ordId": order_id,
        "side": "buy",
        "state": "filled",
        "ordType": "market",
        "sz": "4",
        "avgPx": "0.00895",
        "cTime": _ms(now),
        "fillTime": _ms(now),
    }


def _act_instrument_row() -> dict[str, Any]:
    return {
        "instId": "ACT-USDT-SWAP",
        "instType": "SWAP",
        "ctVal": "1",
        "ctMult": "1",
        "lotSz": "1",
        "minSz": "1",
        "settleCcy": "USDT",
    }


def _act_slippage_fact(now: datetime, *, order_id: str) -> dict[str, Any]:
    row = _act_fill_row(now, order_id=order_id)
    return build_okx_fill_mark_slippage(
        order_id=order_id,
        inst_id="ACT-USDT-SWAP",
        side="buy",
        contracts=4.0,
        average_price=0.00895,
        contract_size=1.0,
        rows=[row],
    )


async def _init_test_db(tmp_path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / name).as_posix()}",
    )
    await init_db()


@pytest.mark.asyncio
async def test_postgres_single_writer_lock_defers_overlapping_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_session = _AdvisoryLockSession([False])

    @asynccontextmanager
    async def fake_session_ctx():
        yield lock_session

    service = OkxOrderFactSyncService(mode="paper")
    sync_called = False

    async def fake_sync_single_writer() -> dict[str, Any]:
        nonlocal sync_called
        sync_called = True
        return {"status": "ok"}

    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://test")
    monkeypatch.setattr(order_fact_sync_module, "get_session_ctx", fake_session_ctx)
    monkeypatch.setattr(service, "_sync_single_writer", fake_sync_single_writer)

    report = await service.sync()

    assert sync_called is False
    assert report["status"] == "deferred"
    assert report["deferred_stages"] == ["single_writer_lock"]
    assert len(lock_session.calls) == 1


@pytest.mark.asyncio
async def test_postgres_single_writer_lock_is_released_after_sync_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_session = _AdvisoryLockSession([True, True])

    @asynccontextmanager
    async def fake_session_ctx():
        yield lock_session

    service = OkxOrderFactSyncService(mode="live")

    async def failing_sync_single_writer() -> dict[str, Any]:
        raise RuntimeError("sync failed")

    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://test")
    monkeypatch.setattr(order_fact_sync_module, "get_session_ctx", fake_session_ctx)
    monkeypatch.setattr(service, "_sync_single_writer", failing_sync_single_writer)

    with pytest.raises(RuntimeError, match="sync failed"):
        await service.sync()

    assert len(lock_session.calls) == 2
    assert "pg_advisory_unlock" in str(lock_session.calls[1][0])


@pytest.mark.asyncio
async def test_order_fact_sync_only_calls_order_fact_endpoints_and_confirms_fill(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "order-fact-confirm.db")
    now = datetime.now(UTC)
    ccxt = _FakeCcxt(fills=[_fill_row(now)], orders=[_order_row(now)])
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="rule_strategy",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    side="buy",
                    order_type="market",
                    quantity=0.02,
                    price=60000.0,
                    status="filled",
                    fee=0.0,
                    exchange_order_id="okx-order-1",
                    created_at=now,
                    filled_at=now,
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            timeout_seconds=5.0,
            executor_factory=_executor_factory(ccxt),
        ).sync()

        assert report["confirmed_count"] == 1
        assert report["unverified_count"] == 0
        assert "position_history" not in report["completed_stages"]
        assert "account_bills" not in report["completed_stages"]
        assert set(ccxt.calls) <= {"fills", "orders", "protection", "contract_specs"}
        async with get_session_ctx() as session:
            order = (await session.execute(select(Order))).scalar_one()
        assert order.okx_sync_status == OKX_SYNC_CONFIRMED
        assert order.okx_trade_ids == "trade-okx-order-1"
        assert order.fee == pytest.approx(0.12)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_timeout_defers_without_marking_missing_fill_unverified(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "order-fact-timeout.db")
    now = datetime.now(UTC)
    ccxt = _FakeCcxt(delay_seconds=2.0)
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="rule_strategy",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    side="buy",
                    order_type="market",
                    quantity=0.02,
                    price=60000.0,
                    status="filled",
                    fee=0.0,
                    exchange_order_id="slow-order-1",
                    created_at=now,
                    filled_at=now,
                )
            )

        started = time.monotonic()
        report = await OkxOrderFactSyncService(
            mode="paper",
            timeout_seconds=0.5,
            executor_factory=_executor_factory(ccxt),
        ).sync()
        elapsed = time.monotonic() - started

        assert elapsed < 1.2
        assert report["status"] == "deferred"
        assert "fills_history_account" in report["deferred_stages"]
        assert report["unverified_count"] == 0
        async with get_session_ctx() as session:
            order = (await session.execute(select(Order))).scalar_one()
        assert order.okx_sync_status is None
    finally:
        await close_db()


def test_fill_deduplication_keeps_complete_cumulative_order_fact() -> None:
    partial = OkxNativeFillGroup(
        order_id="order-1",
        trade_ids=("fill-1",),
        inst_id="BTC-USDT-SWAP",
        symbol="BTC/USDT",
        side="sell",
        pos_side="net",
        contracts=1.0,
        avg_price=60000.0,
        fee_abs=0.01,
        fill_pnl=0.0,
        timestamp_ms=1_000.0,
        timestamp=datetime.fromtimestamp(1, tz=UTC),
        raw_count=1,
    )
    complete = OkxNativeFillGroup(
        order_id="order-1",
        trade_ids=("fill-1", "fill-2"),
        inst_id="BTC-USDT-SWAP",
        symbol="BTC/USDT",
        side="sell",
        pos_side="net",
        contracts=2.0,
        avg_price=60001.0,
        fee_abs=0.02,
        fill_pnl=0.5,
        timestamp_ms=2_000.0,
        timestamp=datetime.fromtimestamp(2, tz=UTC),
        raw_count=2,
    )

    assert _dedupe_fills_by_order_id([partial, complete]) == [complete]


def test_targeted_fill_queries_prioritize_incomplete_slippage_over_recency() -> None:
    now = datetime.now(UTC)
    recent = SimpleNamespace(
        exchange_order_id="recent-generic",
        filled_at=now,
        created_at=now,
        okx_raw_fills={"execution_result_confirmed": True},
    )
    incomplete_slippage = SimpleNamespace(
        exchange_order_id="older-slippage-gap",
        filled_at=now - timedelta(days=3),
        created_at=now - timedelta(days=3),
        okx_raw_fills={
            "fills_history_confirmed": True,
            "execution_slippage": {
                "version": OKX_FILL_MARK_SLIPPAGE_VERSION,
                "complete": False,
                "recovery_terminal": False,
            },
        },
    )

    assert _prioritized_exchange_order_ids(
        [recent, incomplete_slippage],
        limit=1,
    ) == ["older-slippage-gap"]


def test_contract_catalog_keeps_only_public_specification() -> None:
    catalog = _build_contract_size_catalog(
        public_sizes={"BTC-USDT-SWAP": 0.01},
    )

    assert catalog["BTC-USDT-SWAP"] == pytest.approx(0.01)


def test_stored_fill_cannot_recover_missing_raw_contracts_from_local_columns() -> None:
    now = datetime.now(UTC)
    order = Order(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ACT/USDT",
        side="buy",
        order_type="market",
        quantity=4.0,
        price=0.00895,
        status="filled",
        exchange_order_id="act-local-only",
        okx_inst_id="ACT-USDT-SWAP",
        okx_fill_contracts=4.0,
        okx_sync_status=OKX_SYNC_CONFIRMED,
        okx_raw_fills={
            "fills_history_confirmed": True,
            "order_id": "act-local-only",
            "trade_ids": ["trade-act-local-only"],
            "inst_id": "ACT-USDT-SWAP",
        },
    )

    changed = _repair_stored_fill_contract_size_from_instruments(
        order,
        contract_sizes={"ACT-USDT-SWAP": 1.0},
        now=now,
    )

    assert changed is False
    assert order.okx_raw_fills.get("contract_size_verified") is not True


def test_public_contract_spec_repairs_account_derived_order_quantity() -> None:
    now = datetime.now(UTC)
    order = Order(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ACT/USDT",
        side="buy",
        order_type="market",
        quantity=40.0,
        price=0.00895,
        status="filled",
        exchange_order_id="act-order-1",
        okx_inst_id="ACT-USDT-SWAP",
        okx_fill_contracts=4.0,
        okx_sync_status=OKX_SYNC_CONFIRMED,
        okx_raw_fills={
            "fills_history_confirmed": True,
            "order_id": "act-order-1",
            "trade_ids": ["trade-act-order-1"],
            "inst_id": "ACT-USDT-SWAP",
            "contracts": 4.0,
            "avg_price": 0.00895,
            "contract_size": 10.0,
            "contract_size_verified": True,
            "contract_size_source": "okx_account_position_history_pnl_fill_crosscheck",
            "base_quantity": 40.0,
        },
    )

    changed = _repair_stored_fill_contract_size_from_instruments(
        order,
        contract_sizes={"ACT-USDT-SWAP": 1.0},
        now=now,
    )

    assert changed is True
    assert order.quantity == pytest.approx(4.0)
    assert order.okx_raw_fills["contract_size"] == pytest.approx(1.0)
    assert order.okx_raw_fills["contract_size_source"] == "okx_public_instruments"


def test_authoritative_stored_fill_is_repaired_before_decision_recovery() -> None:
    now = datetime.now(UTC)
    order = Order(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ACT/USDT",
        side="buy",
        order_type="market",
        quantity=40.0,
        price=0.00895,
        status="filled",
        decision_id=7,
        exchange_order_id="act-order-priority",
        okx_inst_id="ACT-USDT-SWAP",
        okx_fill_contracts=4.0,
        okx_sync_status=OKX_SYNC_CONFIRMED,
        okx_raw_fills={
            "fills_history_confirmed": True,
            "order_id": "act-order-priority",
            "trade_ids": ["trade-act-order-priority"],
            "inst_id": "ACT-USDT-SWAP",
            "contracts": 4.0,
            "avg_price": 0.00895,
            "contract_size": 10.0,
            "contract_size_verified": True,
            "contract_size_source": "okx_account_position_history_pnl_fill_crosscheck",
            "base_quantity": 40.0,
        },
        created_at=now,
        filled_at=now,
    )
    decision = SimpleNamespace(
        id=7,
        raw_llm_response={
            "close_fill": {
                "order_id": "act-order-priority",
                "instId": "ACT-USDT-SWAP",
                "contracts": 4.0,
                "price": 0.00895,
                "contract_size": 10.0,
                "quantity": 40.0,
            }
        },
    )

    confirmed, unverified, skipped, deferred, samples = OkxOrderFactSyncService(
        mode="paper"
    )._apply_local_order_facts(
        [order],
        fills=[],
        fills_by_order_id={},
        order_rows_by_id={},
        protection_execution_by_order_id={},
        contract_sizes={"ACT-USDT-SWAP": 1.0},
        decisions_by_id={7: decision},
        now=now,
        since=now - timedelta(minutes=1),
        authoritative_absence_order_ids=set(),
    )

    assert (confirmed, unverified, skipped, deferred) == (1, 0, 0, 0)
    assert samples[0]["kind"] == "local_order_contract_size_repaired"
    assert order.quantity == pytest.approx(4.0)
    assert order.okx_raw_fills["contract_size"] == pytest.approx(1.0)
    assert order.okx_raw_fills["contract_size_source"] == "okx_public_instruments"
    assert "recovered_from_decision" not in order.okx_raw_fills


def test_already_verified_stored_fill_is_not_counted_as_missing_contract_size() -> None:
    now = datetime.now(UTC)
    order = Order(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ACT/USDT",
        side="buy",
        order_type="market",
        quantity=4.0,
        price=0.00895,
        status="filled",
        exchange_order_id="act-order-verified",
        okx_inst_id="ACT-USDT-SWAP",
        okx_fill_contracts=4.0,
        okx_sync_status=OKX_SYNC_CONFIRMED,
        okx_raw_fills={
            "fills_history_confirmed": True,
            "order_id": "act-order-verified",
            "trade_ids": ["trade-act-order-verified"],
            "inst_id": "ACT-USDT-SWAP",
            "contracts": 4.0,
            "avg_price": 0.00895,
            "contract_size": 1.0,
            "contract_size_verified": True,
            "contract_size_source": "okx_public_instruments",
            "base_quantity": 4.0,
            "rows": [_act_fill_row(now, order_id="act-order-verified")],
            "execution_slippage": _act_slippage_fact(
                now,
                order_id="act-order-verified",
            ),
        },
        created_at=now,
        filled_at=now,
    )

    confirmed, unverified, skipped, deferred, samples = OkxOrderFactSyncService(
        mode="paper"
    )._apply_local_order_facts(
        [order],
        fills=[],
        fills_by_order_id={},
        order_rows_by_id={},
        protection_execution_by_order_id={},
        contract_sizes={"ACT-USDT-SWAP": 1.0},
        decisions_by_id={},
        now=now,
        since=now - timedelta(minutes=1),
        authoritative_absence_order_ids=set(),
    )

    assert (confirmed, unverified, skipped, deferred) == (0, 0, 0, 0)
    assert samples[0]["kind"] == "local_order_stored_fill_already_verified"


def test_matching_native_fill_still_refreshes_missing_slippage_fact() -> None:
    now = datetime.now(UTC)
    order_id = "act-matching-fill"
    row = _act_fill_row(now, order_id=order_id)
    fill = OkxNativeFillGroup(
        order_id=order_id,
        trade_ids=(f"trade-{order_id}",),
        inst_id="ACT-USDT-SWAP",
        symbol="ACT/USDT",
        side="buy",
        pos_side="net",
        contracts=4.0,
        avg_price=0.00895,
        fee_abs=0.001,
        fill_pnl=0.0,
        timestamp_ms=now.timestamp() * 1000.0,
        timestamp=now,
        raw_count=1,
        rows=(row,),
    )
    order = Order(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ACT/USDT",
        side="buy",
        order_type="market",
        quantity=4.0,
        price=0.00895,
        status="filled",
        fee=0.001,
        exchange_order_id=order_id,
        okx_inst_id="ACT-USDT-SWAP",
        okx_trade_ids=f"trade-{order_id}",
        okx_fill_contracts=4.0,
        okx_fill_pnl=0.0,
        okx_sync_status=OKX_SYNC_CONFIRMED,
        okx_raw_fills={
            "fills_history_confirmed": True,
            "order_id": order_id,
            "trade_ids": [f"trade-{order_id}"],
            "inst_id": "ACT-USDT-SWAP",
            "contracts": 4.0,
            "avg_price": 0.00895,
            "fee_abs": 0.001,
            "fill_pnl": 0.0,
            "contract_size": 1.0,
            "contract_size_verified": True,
            "contract_size_source": "okx_public_instruments",
            "base_quantity": 4.0,
            "rows": [row],
        },
        created_at=now,
        filled_at=now,
    )

    confirmed, unverified, skipped, deferred, samples = OkxOrderFactSyncService(
        mode="paper"
    )._apply_local_order_facts(
        [order],
        fills=[fill],
        fills_by_order_id={order_id: fill},
        order_rows_by_id={},
        protection_execution_by_order_id={},
        contract_sizes={"ACT-USDT-SWAP": 1.0},
        decisions_by_id={},
        now=now,
        since=now - timedelta(minutes=1),
        authoritative_absence_order_ids=set(),
    )

    assert (confirmed, unverified, skipped, deferred) == (1, 0, 0, 0)
    assert samples[0]["kind"] == "local_order_slippage_fact_refreshed"
    assert order.okx_raw_fills["execution_slippage"]["complete"] is True


def test_fill_storage_keeps_every_compact_row_without_twenty_row_truncation() -> None:
    now = datetime.now(UTC)
    order_id = "act-many-fills"
    rows: list[dict[str, Any]] = []
    for index in range(21):
        row = _act_fill_row(now, order_id=order_id)
        row["tradeId"] = f"trade-{index}"
        row["fillSz"] = "1"
        row["fee"] = "-0.00025"
        rows.append(row)
    fill = OkxNativeFillGroup(
        order_id=order_id,
        trade_ids=tuple(f"trade-{index}" for index in range(21)),
        inst_id="ACT-USDT-SWAP",
        symbol="ACT/USDT",
        side="buy",
        pos_side="net",
        contracts=21.0,
        avg_price=0.00895,
        fee_abs=0.00525,
        fill_pnl=0.0,
        timestamp_ms=now.timestamp() * 1000.0,
        timestamp=now,
        raw_count=21,
        rows=tuple(rows),
    )
    order = Order(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ACT/USDT",
        side="buy",
        order_type="market",
        quantity=21.0,
        price=0.00895,
        status="filled",
        fee=0.00525,
        exchange_order_id=order_id,
    )

    OkxOrderFactSyncService(mode="paper")._apply_fill_to_order(
        order,
        fill,
        now=now,
        sync_status=OKX_SYNC_CONFIRMED,
        contract_size=1.0,
        contract_size_source="okx_public_instruments",
    )

    assert len(order.okx_raw_fills["rows"]) == 21
    assert set(order.okx_raw_fills["rows"][0]) == {
        "ordId",
        "instId",
        "tradeId",
        "billId",
        "clOrdId",
        "side",
        "posSide",
        "fillSz",
        "fillPx",
        "fillMarkPx",
        "fee",
        "feeCcy",
        "fillPnl",
        "ts",
        "fillTime",
    }
    assert order.okx_raw_fills["execution_slippage"]["complete"] is True


def test_complete_official_pull_marks_missing_fill_mark_terminal() -> None:
    now = datetime.now(UTC)
    order_id = "act-no-fill-mark"
    row = _act_fill_row(now, order_id=order_id)
    row["fillMarkPx"] = ""
    fill = OkxNativeFillGroup(
        order_id=order_id,
        trade_ids=(f"trade-{order_id}",),
        inst_id="ACT-USDT-SWAP",
        symbol="ACT/USDT",
        side="buy",
        pos_side="net",
        contracts=4.0,
        avg_price=0.00895,
        fee_abs=0.001,
        fill_pnl=0.0,
        timestamp_ms=now.timestamp() * 1000.0,
        timestamp=now,
        raw_count=1,
        rows=(row,),
    )
    order = Order(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ACT/USDT",
        side="buy",
        order_type="market",
        quantity=4.0,
        price=0.00895,
        status="filled",
        fee=0.001,
        exchange_order_id=order_id,
    )

    OkxOrderFactSyncService(mode="paper")._apply_fill_to_order(
        order,
        fill,
        now=now,
        sync_status=OKX_SYNC_CONFIRMED,
        contract_size=1.0,
        contract_size_source="okx_public_instruments",
    )

    slippage = order.okx_raw_fills["execution_slippage"]
    assert slippage["complete"] is False
    assert slippage["recovery_terminal"] is True
    assert slippage["recovery_source"] == "okx_fills_history_current_pull"
    assert _stored_slippage_fact_needs_refresh(order) is False


def test_stored_rows_upgrade_slippage_version_without_false_quantity_gap() -> None:
    now = datetime.now(UTC)
    order_id = "act-old-slippage-contract"
    row = _act_fill_row(now, order_id=order_id)
    row["fillMarkPx"] = ""
    old_slippage = build_okx_fill_mark_slippage(
        order_id=order_id,
        inst_id="ACT-USDT-SWAP",
        side="buy",
        contracts=4.0,
        average_price=0.00895,
        contract_size=1.0,
        rows=[row],
    )
    old_slippage.update(
        {
            "version": "2026-07-24.okx-fill-mark-slippage.v1",
            "reasons": [
                "fill_row_mark_price_invalid",
                "fill_row_contract_total_mismatch",
            ],
            "contracts": 0.0,
            "fill_vwap": None,
            "recovery_terminal": True,
            "recovery_source": "okx_fills_history_current_pull",
        }
    )
    order = Order(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ACT/USDT",
        side="buy",
        order_type="market",
        quantity=4.0,
        price=0.00895,
        status="filled",
        fee=0.001,
        exchange_order_id=order_id,
        okx_inst_id="ACT-USDT-SWAP",
        okx_fill_contracts=4.0,
        okx_raw_fills={
            "fills_history_confirmed": True,
            "order_id": order_id,
            "trade_ids": [f"trade-{order_id}"],
            "inst_id": "ACT-USDT-SWAP",
            "contracts": 4.0,
            "avg_price": 0.00895,
            "fee_abs": 0.001,
            "contract_size": 1.0,
            "contract_size_verified": True,
            "contract_size_source": "okx_public_instruments",
            "base_quantity": 4.0,
            "rows": [row],
            "execution_slippage": old_slippage,
        },
        created_at=now,
        filled_at=now,
    )

    assert _rebuild_stored_slippage_fact(order, now=now) is True

    slippage = order.okx_raw_fills["execution_slippage"]
    assert slippage["version"] == OKX_FILL_MARK_SLIPPAGE_VERSION
    assert slippage["reasons"] == ["fill_row_mark_price_invalid"]
    assert slippage["contracts"] == 4.0
    assert slippage["fill_vwap"] == pytest.approx(0.00895)
    assert slippage["recovery_terminal"] is True
    assert slippage["recovery_source"] == "stored_okx_fill_rows_contract_upgrade"
    assert _stored_slippage_fact_needs_refresh(order) is False


@pytest.mark.asyncio
async def test_confirmed_fill_queries_public_spec_and_repairs_polluted_quantity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "confirmed-contract-repair.db")
    now = datetime.now(UTC)
    ccxt = _FakeCcxt(
        fills=[_act_fill_row(now)],
        orders=[_act_order_row(now)],
        instruments=[_act_instrument_row()],
    )
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="ACT/USDT",
                    side="buy",
                    order_type="market",
                    quantity=40.0,
                    price=0.00895,
                    status="filled",
                    fee=0.001,
                    exchange_order_id="act-order-1",
                    okx_inst_id="ACT-USDT-SWAP",
                    okx_fill_contracts=4.0,
                    okx_fill_pnl=0.0,
                    okx_sync_status=OKX_SYNC_CONFIRMED,
                    okx_raw_fills={
                        "fills_history_confirmed": True,
                        "order_id": "act-order-1",
                        "trade_ids": ["trade-act-order-1"],
                        "inst_id": "ACT-USDT-SWAP",
                        "contracts": 4.0,
                        "avg_price": 0.00895,
                        "fee_abs": 0.001,
                        "fill_pnl": 0.0,
                        "contract_size": 10.0,
                        "contract_size_verified": True,
                        "contract_size_source": (
                            "okx_account_position_history_pnl_fill_crosscheck"
                        ),
                        "base_quantity": 40.0,
                    },
                    created_at=now,
                    filled_at=now,
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            timeout_seconds=5.0,
            executor_factory=_executor_factory(ccxt),
        ).sync()

        async with get_session_ctx() as session:
            order = (await session.execute(select(Order))).scalar_one()
        assert ccxt.calls.count("fills") == 2
        assert "contract_specs" in ccxt.calls
        assert report["confirmed_count"] == 1
        assert report["contract_size_deferred_count"] == 0
        assert order.quantity == pytest.approx(4.0)
        assert order.okx_fill_contracts == pytest.approx(4.0)
        assert order.okx_raw_fills["contract_size"] == pytest.approx(1.0)
        assert order.okx_raw_fills["contract_size_source"] == "okx_public_instruments"
        assert order.okx_raw_fills["contract_size_verified"] is True
        assert order.okx_raw_fills["base_quantity"] == pytest.approx(4.0)
        assert order.okx_raw_fills["execution_slippage"]["complete"] is True
        assert order.okx_raw_fills["execution_slippage"][
            "adverse_slippage_usdt"
        ] == pytest.approx(0.0002)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_stored_fill_slippage_is_refreshed_outside_recent_window(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "stored-slippage-refresh.db")
    now = datetime.now(UTC)
    filled_at = now - timedelta(days=10)
    order_id = "act-stored-slippage"
    raw_row = _act_fill_row(filled_at, order_id=order_id)
    ccxt = _FakeCcxt(
        fills=[],
        orders=[],
        instruments=[_act_instrument_row()],
    )
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="ACT/USDT",
                    side="buy",
                    order_type="market",
                    quantity=4.0,
                    price=0.00895,
                    status="filled",
                    fee=0.001,
                    exchange_order_id=order_id,
                    okx_inst_id="ACT-USDT-SWAP",
                    okx_trade_ids=f"trade-{order_id}",
                    okx_fill_contracts=4.0,
                    okx_fill_pnl=0.0,
                    okx_sync_status=OKX_SYNC_CONFIRMED,
                    okx_raw_fills={
                        "fills_history_confirmed": True,
                        "order_id": order_id,
                        "trade_ids": [f"trade-{order_id}"],
                        "inst_id": "ACT-USDT-SWAP",
                        "contracts": 4.0,
                        "avg_price": 0.00895,
                        "fee_abs": 0.001,
                        "fill_pnl": 0.0,
                        "contract_size": 1.0,
                        "contract_size_verified": True,
                        "contract_size_source": "okx_public_instruments",
                        "base_quantity": 4.0,
                        "rows": [raw_row],
                    },
                    created_at=filled_at,
                    filled_at=filled_at,
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            timeout_seconds=5.0,
            executor_factory=_executor_factory(ccxt),
        ).sync()

        async with get_session_ctx() as session:
            order = (await session.execute(select(Order))).scalar_one()
        assert report["confirmed_count"] == 1
        assert report["skipped_old_count"] == 0
        assert order.okx_raw_fills["execution_slippage"]["complete"] is True
        assert order.okx_raw_fills["execution_slippage"][
            "adverse_slippage_usdt"
        ] == pytest.approx(0.0002)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_missing_public_spec_preserves_existing_confirmed_fill_fact(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "missing-contract-preserves-order.db")
    now = datetime.now(UTC)
    raw_fill = {
        "fills_history_confirmed": True,
        "order_id": "act-order-2",
        "trade_ids": ["trade-act-order-2"],
        "inst_id": "ACT-USDT-SWAP",
        "contracts": 4.0,
        "avg_price": 0.00895,
        "fee_abs": 0.001,
        "fill_pnl": 0.0,
        "contract_size": 1.0,
        "contract_size_verified": True,
        "contract_size_source": "okx_public_instruments",
        "base_quantity": 4.0,
    }
    ccxt = _FakeCcxt(
        fills=[_act_fill_row(now, order_id="act-order-2")],
        orders=[_act_order_row(now, order_id="act-order-2")],
        instruments=[],
    )
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="ACT/USDT",
                    side="buy",
                    order_type="market",
                    quantity=4.0,
                    price=0.00895,
                    status="filled",
                    fee=0.001,
                    exchange_order_id="act-order-2",
                    okx_inst_id="ACT-USDT-SWAP",
                    okx_fill_contracts=4.0,
                    okx_fill_pnl=0.0,
                    okx_sync_status=OKX_SYNC_CONFIRMED,
                    okx_raw_fills=raw_fill,
                    created_at=now,
                    filled_at=now,
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            timeout_seconds=5.0,
            executor_factory=_executor_factory(ccxt),
        ).sync()

        async with get_session_ctx() as session:
            order = (await session.execute(select(Order))).scalar_one()
        assert report["contract_size_deferred_count"] == 1
        assert "order_facts_missing_public_contract_size" in report["deferred_stages"]
        assert order.quantity == pytest.approx(4.0)
        assert order.okx_fill_contracts == pytest.approx(4.0)
        assert order.okx_raw_fills == raw_fill
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_only_fill_without_public_spec_is_not_persisted(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "missing-contract-skips-backfill.db")
    now = datetime.now(UTC)
    ccxt = _FakeCcxt(
        fills=[_act_fill_row(now, order_id="act-order-3")],
        orders=[_act_order_row(now, order_id="act-order-3")],
        instruments=[],
    )
    try:
        report = await OkxOrderFactSyncService(
            mode="paper",
            timeout_seconds=5.0,
            executor_factory=_executor_factory(ccxt),
        ).sync()

        async with get_session_ctx() as session:
            orders = list((await session.execute(select(Order))).scalars().all())
        assert orders == []
        assert report["backfilled_count"] == 0
        assert report["contract_size_deferred_count"] == 1
        assert "order_facts_missing_public_contract_size" in report["deferred_stages"]
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_recent_fill_ledger_upgrades_execution_result_to_native_fill_fact(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "recent-fill-upgrades-execution-result.db")
    now = datetime.now(UTC)
    ccxt = _RecentOnlyCcxt(
        fills=[_act_fill_row(now, order_id="act-recent-order")],
        orders=[_act_order_row(now, order_id="act-recent-order")],
        instruments=[_act_instrument_row()],
    )
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="ACT/USDT",
                    side="buy",
                    order_type="market",
                    quantity=4.0,
                    price=0.00895,
                    status="filled",
                    fee=0.001,
                    exchange_order_id="act-recent-order",
                    okx_inst_id="ACT-USDT-SWAP",
                    okx_fill_contracts=4.0,
                    okx_fill_pnl=0.0,
                    okx_sync_status=OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
                    okx_raw_fills={
                        "fills_history_confirmed": False,
                        "execution_result_confirmed": True,
                        "order_id": "act-recent-order",
                        "trade_ids": ["trade-act-recent-order"],
                        "inst_id": "ACT-USDT-SWAP",
                        "contracts": 4.0,
                        "avg_price": 0.00895,
                        "fee_abs": 0.001,
                        "fill_pnl": 0.0,
                        "contract_size": 1.0,
                        "contract_size_verified": True,
                        "contract_size_source": "okx_public_instruments",
                        "base_quantity": 4.0,
                    },
                    created_at=now,
                    filled_at=now,
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            timeout_seconds=5.0,
            executor_factory=_executor_factory(ccxt),
        ).sync()

        async with get_session_ctx() as session:
            order = (await session.execute(select(Order))).scalar_one()
        assert report["confirmed_count"] == 1
        assert report["contract_size_deferred_count"] == 0
        assert order.okx_sync_status == OKX_SYNC_CONFIRMED
        assert order.okx_raw_fills["fills_history_confirmed"] is True
        assert order.okx_raw_fills["contract_size_source"] == "okx_public_instruments"
        assert order.okx_raw_fills["base_quantity"] == pytest.approx(4.0)
        assert order.okx_raw_fills["execution_slippage"]["complete"] is True
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_complete_embedded_okx_order_detail_is_promoted_from_execution_result(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "embedded-order-detail.db")
    now = datetime.now(UTC)
    order_id = "act-order-detail"
    trade_id = "trade-act-order-detail"
    detail_row = {
        "state": "filled",
        "ordId": order_id,
        "instId": "ACT-USDT-SWAP",
        "side": "buy",
        "tradeId": trade_id,
        "accFillSz": "4",
        "fillSz": "4",
        "avgPx": "0.00895",
        "fillPx": "0.00895",
        "fee": "-0.001",
        "fillTime": _ms(now),
        "uTime": _ms(now),
    }
    ccxt = _FakeCcxt(
        fills=[],
        orders=[{**detail_row, "ordType": "market", "cTime": _ms(now)}],
        instruments=[_act_instrument_row()],
    )
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="ACT/USDT",
                    side="buy",
                    order_type="market",
                    quantity=4.0,
                    price=0.00895,
                    status="filled",
                    fee=0.001,
                    exchange_order_id=order_id,
                    okx_inst_id="ACT-USDT-SWAP",
                    okx_fill_contracts=4.0,
                    okx_fill_pnl=0.0,
                    okx_sync_status=OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
                    okx_raw_fills={
                        "source": "okx_execution_result",
                        "fills_history_confirmed": False,
                        "execution_result_confirmed": True,
                        "order_id": order_id,
                        "trade_ids": [trade_id],
                        "inst_id": "ACT-USDT-SWAP",
                        "contracts": 4.0,
                        "avg_price": 0.00895,
                        "fee_abs": 0.001,
                        "fill_pnl": 0.0,
                        "contract_size": 1.0,
                        "contract_size_verified": True,
                        "contract_size_source": "okx_public_instruments",
                        "base_quantity": 4.0,
                        "rows": [detail_row],
                    },
                    created_at=now,
                    filled_at=now,
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            timeout_seconds=5.0,
            executor_factory=_executor_factory(ccxt),
        ).sync()

        async with get_session_ctx() as session:
            order = (await session.execute(select(Order))).scalar_one()
        assert report["confirmed_count"] == 1
        assert order.okx_sync_status == OKX_SYNC_ORDER_DETAIL_CONFIRMED
        assert order.okx_raw_fills["source"] == "okx_order_detail"
        assert order.okx_raw_fills["order_detail_confirmed"] is True
        assert order.okx_raw_fills["execution_result_confirmed"] is False
        assert order.okx_raw_fills["fills_history_confirmed"] is False
    finally:
        await close_db()
