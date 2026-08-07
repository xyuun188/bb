from __future__ import annotations

import asyncio
import inspect
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from ai_brain.base_model import Action, DecisionOutput
from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from models.trade import Order, Position
from services import okx_order_fact_sync as order_fact_sync_module
from services.normal_paper_trade import (
    attach_normal_paper_order_identity,
    build_normal_paper_trade_contract,
)
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
    _configure_order_fact_write_transaction,
    _database_timeout_stage,
    _decision_for_order_fact,
    _dedupe_fills_by_order_id,
    _prioritized_exchange_order_ids,
    _rebuild_stored_slippage_fact,
    _repair_stored_fill_contract_size_from_instruments,
    _stored_slippage_fact_needs_refresh,
    _target_fill_query_requires_historical,
)


def test_normal_paper_client_identity_recovers_exact_decision_lineage() -> None:
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="strategy_edge_selected",
        direction_support={
            "eligible": True,
            "selected_side": "long",
            "prediction_horizon_minutes": 30.0,
            "expected_net_return_pct": 0.2,
            "objective_net_return_pct": 0.1,
            "loss_probability": 0.4,
            "quant_evidence_families": ["local_ml"],
            "strong_expert_opposition": False,
        },
    )
    output = DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.7,
        reasoning="test",
        position_size_pct=0.1,
        raw_response={"normal_paper_trade": contract},
    )
    identity = attach_normal_paper_order_identity(
        output,
        model_mode="paper",
        decision_id=88,
    )
    decision = AIDecision(
        id=88,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action="long",
        confidence=0.7,
        is_paper=True,
        raw_llm_response=output.raw_response,
    )
    fill = OkxNativeFillGroup(
        order_id="okx-entry-88",
        trade_ids=("trade-88",),
        inst_id="BTC-USDT-SWAP",
        symbol="BTC/USDT",
        side="buy",
        pos_side="long",
        contracts=2.0,
        avg_price=100.0,
        fee_abs=0.01,
        fill_pnl=0.0,
        timestamp_ms=0.0,
        timestamp=datetime.now(UTC),
        raw_count=1,
        rows=({"clOrdId": identity["client_order_id"]},),
    )

    assert _decision_for_order_fact(
        fill=fill,
        order_row={"clOrdId": identity["client_order_id"]},
        decisions_by_id={88: decision},
    ) is decision

    wrong_side_fill = replace(fill, side="sell")
    assert _decision_for_order_fact(
        fill=wrong_side_fill,
        order_row={"clOrdId": identity["client_order_id"]},
        decisions_by_id={88: decision},
    ) is None


class _ScalarResult:
    def __init__(self, value: Any) -> None:
        self.value = value

    def scalar(self) -> Any:
        return self.value


class _EmptyRowsResult:
    def scalars(self) -> _EmptyRowsResult:
        return self

    def all(self) -> list[Any]:
        return []


class _StatementCaptureSession:
    def __init__(self) -> None:
        self.statements: list[Any] = []

    async def execute(
        self,
        statement: Any,
        params: dict[str, Any] | None = None,
    ) -> _EmptyRowsResult:
        self.statements.append(statement)
        return _EmptyRowsResult()


class _AdvisoryLockConnection:
    def __init__(self, values: list[Any]) -> None:
        self.values = list(values)
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.commit_count = 0
        self.invalidate_count = 0

    async def execute(self, statement: Any, params: dict[str, Any]) -> _ScalarResult:
        self.calls.append((statement, params))
        return _ScalarResult(self.values.pop(0) if self.values else None)

    async def commit(self) -> None:
        self.commit_count += 1

    async def invalidate(self) -> None:
        self.invalidate_count += 1


class _AdvisoryLockEngine:
    def __init__(self, connection: Any) -> None:
        self.connection = connection
        self.connect_count = 0

    @asynccontextmanager
    async def _connect(self):
        self.connect_count += 1
        yield self.connection

    def connect(self):
        return self._connect()


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


class _HistoricalOnlyTargetCcxt(_FakeCcxt):
    async def privateGetTradeFills(self, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("fills_recent_targeted")
        return {"data": []}

    async def privateGetTradeFillsHistory(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append("fills_historical_targeted")
        order_id = str(params.get("ordId") or "")
        rows = [row for row in self.fills if not order_id or row.get("ordId") == order_id]
        return {"data": rows}


class _SlowAccountFastTargetCcxt(_FakeCcxt):
    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        order_id = str(params.get("ordId") or "")
        self.calls.append("fills_targeted" if order_id else "fills_account")
        if not order_id:
            await asyncio.sleep(2.0)
            return {"data": []}
        return {"data": [row for row in self.fills if row.get("ordId") == order_id]}

    async def privateGetTradeFills(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self.privateGetTradeFillsHistory(params)


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


def _confirmed_order_fact(
    *,
    order_id: str,
    inst_id: str,
    quantity: float,
    price: float,
    fee: float,
    confirmed: bool = True,
) -> dict[str, Any]:
    return {
        "fills_history_confirmed": confirmed,
        "order_id": order_id,
        "trade_ids": [f"trade-{order_id}"],
        "inst_id": inst_id,
        "contracts": quantity,
        "avg_price": price,
        "contract_size": 1.0,
        "contract_size_verified": True,
        "contract_size_source": "okx_public_instruments",
        "base_quantity": quantity,
        "fee_abs": fee,
    }


@pytest.mark.asyncio
async def test_order_fact_sync_auto_recovers_exact_exit_decision_lineage(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "exit-lineage-auto-recovery.db")
    now = datetime.now(UTC)
    order_id = "act-raced-close"
    fill = OkxNativeFillGroup(
        order_id=order_id,
        trade_ids=("trade-act-raced-close",),
        inst_id="ACT-USDT-SWAP",
        symbol="ACT/USDT",
        side="buy",
        pos_side="short",
        contracts=4.0,
        avg_price=0.00895,
        fee_abs=0.001,
        fill_pnl=-0.02,
        timestamp_ms=float(_ms(now)),
        timestamp=now,
        raw_count=1,
        rows=(_act_fill_row(now, order_id=order_id),),
    )
    try:
        async with get_session_ctx() as session:
            original = AIDecision(
                model_name="ensemble_trader",
                symbol="ACT/USDT",
                action="close_short",
                confidence=0.8,
                is_paper=True,
                was_executed=True,
                raw_llm_response={
                    "dynamic_exit_policy": {"eligible": True, "close_fraction": 1.0},
                    "execution_result": {"exchange_order_id": order_id},
                },
            )
            synthetic = AIDecision(
                model_name="ensemble_trader",
                symbol="ACT/USDT",
                action="close_short",
                confidence=1.0,
                is_paper=True,
                was_executed=True,
                executed_at=now,
                raw_llm_response={
                    "system_sync": True,
                    "source": "okx_position_reconcile",
                    "close_fill": {"order_id": order_id},
                },
            )
            session.add_all([original, synthetic])
            await session.flush()
            original_id = int(original.id)
            synthetic_id = int(synthetic.id)
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
                    decision_id=synthetic_id,
                    exchange_order_id=order_id,
                    filled_at=now,
                    created_at=now,
                )
            )
            session.add(
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="ACT/USDT",
                    side="short",
                    quantity=4.0,
                    entry_price=0.009,
                    realized_pnl=-0.02,
                    is_open=False,
                    closed_at=now,
                    close_exchange_order_id=order_id,
                )
            )

        service = OkxOrderFactSyncService(mode="paper")
        samples: list[dict[str, Any]] = []
        async with get_session_ctx() as session:
            order = (
                await session.execute(
                    select(Order).where(Order.exchange_order_id == order_id)
                )
            ).scalar_one()
            recovered_count, errors = await service._recover_exit_decision_lineages(
                session,
                orders=[order],
                fills=[fill],
                samples=samples,
            )

        assert recovered_count == 1
        assert errors == []
        assert samples[-1]["kind"] == "exit_decision_lineage_auto_recovered"
        async with get_session_ctx() as session:
            order = (
                await session.execute(
                    select(Order).where(Order.exchange_order_id == order_id)
                )
            ).scalar_one()
            original = await session.get(AIDecision, original_id)
            synthetic = await session.get(AIDecision, synthetic_id)
            assert order.decision_id == original_id
            assert original.was_executed is True
            assert synthetic.was_executed is False
            recovered_count, errors = await service._recover_exit_decision_lineages(
                session,
                orders=[order],
                fills=[fill],
                samples=[],
            )
            assert recovered_count == 0
            assert errors == []
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_recovers_only_exact_confirmed_missing_closed_position(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "missing-position-auto-recovery.db")
    opened_at = datetime.now(UTC) - timedelta(minutes=5)
    closed_at = opened_at + timedelta(minutes=3)
    try:
        async with get_session_ctx() as session:
            entry_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="OP/USDT",
                action="long",
                confidence=0.8,
                is_paper=True,
                was_executed=True,
            )
            close_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="OP/USDT",
                action="close_long",
                confidence=0.8,
                is_paper=True,
                was_executed=True,
            )
            session.add_all([entry_decision, close_decision])
            await session.flush()
            entry_order = Order(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="OP/USDT",
                side="buy",
                order_type="market",
                quantity=1.0,
                price=0.71,
                status="filled",
                fee=0.0001,
                decision_id=entry_decision.id,
                exchange_order_id="okx-op-entry",
                okx_inst_id="OP-USDT-SWAP",
                okx_raw_fills=_confirmed_order_fact(
                    order_id="okx-op-entry",
                    inst_id="OP-USDT-SWAP",
                    quantity=1.0,
                    price=0.71,
                    fee=0.0001,
                ),
                filled_at=opened_at,
                created_at=opened_at,
            )
            close_order = Order(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="OP/USDT",
                side="sell",
                order_type="market",
                quantity=1.0,
                price=0.7099,
                status="filled",
                fee=0.0001,
                decision_id=close_decision.id,
                exchange_order_id="okx-op-close",
                okx_inst_id="OP-USDT-SWAP",
                okx_raw_fills=_confirmed_order_fact(
                    order_id="okx-op-close",
                    inst_id="OP-USDT-SWAP",
                    quantity=1.0,
                    price=0.7099,
                    fee=0.0001,
                ),
                filled_at=closed_at,
                created_at=closed_at,
            )
            session.add_all([entry_order, close_order])

        service = OkxOrderFactSyncService(mode="paper")
        samples: list[dict[str, Any]] = []
        since_naive = (opened_at - timedelta(minutes=1)).replace(tzinfo=None)
        async with get_session_ctx() as session:
            candidates = await service._load_missing_closed_position_recovery_orders(
                session,
                since_naive,
            )
            recovered = await service._recover_missing_closed_positions(
                session,
                orders=candidates,
                samples=samples,
            )

        assert recovered == 1
        assert samples[-1]["kind"] == "missing_closed_position_auto_recovered"
        async with get_session_ctx() as session:
            position = (await session.execute(select(Position))).scalar_one()
            candidates = await service._load_missing_closed_position_recovery_orders(
                session,
                since_naive,
            )
            repeated = await service._recover_missing_closed_positions(
                session,
                orders=candidates,
                samples=[],
            )
        assert position.close_exchange_order_id == "okx-op-close"
        assert position.entry_exchange_order_id == "okx-op-entry"
        assert position.settlement_source == "missing_closed_position_repair"
        assert repeated == 0
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_does_not_recover_with_unconfirmed_entry_fact(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "missing-position-unconfirmed-entry.db")
    opened_at = datetime.now(UTC) - timedelta(minutes=5)
    closed_at = opened_at + timedelta(minutes=3)
    try:
        async with get_session_ctx() as session:
            entry_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="OP/USDT",
                action="long",
                confidence=0.8,
                is_paper=True,
                was_executed=True,
            )
            close_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="OP/USDT",
                action="close_long",
                confidence=0.8,
                is_paper=True,
                was_executed=True,
            )
            session.add_all([entry_decision, close_decision])
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="OP/USDT",
                        side="buy",
                        order_type="market",
                        quantity=1.0,
                        price=0.71,
                        status="filled",
                        fee=0.0001,
                        decision_id=entry_decision.id,
                        exchange_order_id="okx-unconfirmed-entry",
                        okx_inst_id="OP-USDT-SWAP",
                        okx_raw_fills=_confirmed_order_fact(
                            order_id="okx-unconfirmed-entry",
                            inst_id="OP-USDT-SWAP",
                            quantity=1.0,
                            price=0.71,
                            fee=0.0001,
                            confirmed=False,
                        ),
                        filled_at=opened_at,
                        created_at=opened_at,
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="OP/USDT",
                        side="sell",
                        order_type="market",
                        quantity=1.0,
                        price=0.7099,
                        status="filled",
                        fee=0.0001,
                        decision_id=close_decision.id,
                        exchange_order_id="okx-confirmed-close",
                        okx_inst_id="OP-USDT-SWAP",
                        okx_raw_fills=_confirmed_order_fact(
                            order_id="okx-confirmed-close",
                            inst_id="OP-USDT-SWAP",
                            quantity=1.0,
                            price=0.7099,
                            fee=0.0001,
                        ),
                        filled_at=closed_at,
                        created_at=closed_at,
                    ),
                ]
            )

        service = OkxOrderFactSyncService(mode="paper")
        async with get_session_ctx() as session:
            candidates = await service._load_missing_closed_position_recovery_orders(
                session,
                (opened_at - timedelta(minutes=1)).replace(tzinfo=None),
            )
            recovered = await service._recover_missing_closed_positions(
                session,
                orders=candidates,
                samples=[],
            )
        assert recovered == 0
        async with get_session_ctx() as session:
            positions = list((await session.execute(select(Position))).scalars().all())
        assert positions == []
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_postgres_single_writer_lock_defers_overlapping_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_connection = _AdvisoryLockConnection([False])
    lock_engine = _AdvisoryLockEngine(lock_connection)

    async def fake_get_engine():
        return lock_engine

    service = OkxOrderFactSyncService(mode="paper")
    sync_called = False

    async def fake_sync_single_writer() -> dict[str, Any]:
        nonlocal sync_called
        sync_called = True
        return {"status": "ok"}

    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://test")
    monkeypatch.setattr(order_fact_sync_module, "get_engine", fake_get_engine)
    monkeypatch.setattr(service, "_sync_single_writer", fake_sync_single_writer)

    report = await service.sync()

    assert sync_called is False
    assert report["status"] == "deferred"
    assert report["deferred_stages"] == ["single_writer_lock"]
    assert len(lock_connection.calls) == 1
    assert lock_connection.commit_count == 0
    assert lock_engine.connect_count == 1


@pytest.mark.asyncio
async def test_postgres_single_writer_lock_is_released_after_sync_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_connection = _AdvisoryLockConnection([True, True])
    lock_engine = _AdvisoryLockEngine(lock_connection)

    async def fake_get_engine():
        return lock_engine

    service = OkxOrderFactSyncService(mode="live")

    async def failing_sync_single_writer() -> dict[str, Any]:
        raise RuntimeError("sync failed")

    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://test")
    monkeypatch.setattr(order_fact_sync_module, "get_engine", fake_get_engine)
    monkeypatch.setattr(service, "_sync_single_writer", failing_sync_single_writer)

    with pytest.raises(RuntimeError, match="sync failed"):
        await service.sync()

    assert len(lock_connection.calls) == 2
    assert "pg_advisory_unlock" in str(lock_connection.calls[1][0])
    assert lock_connection.commit_count == 1
    assert lock_connection.invalidate_count == 0


@pytest.mark.asyncio
async def test_postgres_hard_deadline_cancels_sync_and_releases_writer_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_connection = _AdvisoryLockConnection([True, True])
    lock_engine = _AdvisoryLockEngine(lock_connection)
    cancelled = asyncio.Event()
    events: list[str] = []
    execute = lock_connection.execute

    async def tracking_execute(
        statement: Any,
        params: dict[str, Any],
    ) -> _ScalarResult:
        if "pg_advisory_unlock" in str(statement):
            events.append("unlock")
        return await execute(statement, params)

    lock_connection.execute = tracking_execute  # type: ignore[method-assign]

    async def fake_get_engine():
        return lock_engine

    service = OkxOrderFactSyncService(mode="paper")
    service.timeout_seconds = 0.01

    async def never_finish() -> dict[str, Any]:
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            events.append("cancelled")
            cancelled.set()

    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://test")
    monkeypatch.setattr(
        order_fact_sync_module,
        "ORDER_FACT_SYNC_HARD_DEADLINE_GRACE_SECONDS",
        0.01,
    )
    monkeypatch.setattr(order_fact_sync_module, "get_engine", fake_get_engine)
    monkeypatch.setattr(service, "_sync_single_writer", never_finish)

    started = time.monotonic()
    report = await service.sync()
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert cancelled.is_set() is True
    assert report["status"] == "deferred"
    assert report["okx_pull_available"] is False
    assert report["deferred_stages"] == ["hard_deadline"]
    assert report["error"] == "order_fact_sync_hard_deadline_exceeded"
    assert len(lock_connection.calls) == 2
    assert "pg_advisory_unlock" in str(lock_connection.calls[1][0])
    assert events == ["cancelled", "unlock"]
    assert lock_connection.commit_count == 1
    assert lock_connection.invalidate_count == 0


@pytest.mark.asyncio
async def test_postgres_hard_deadline_includes_writer_lock_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_wait_cancelled = asyncio.Event()

    class SlowLockSession:
        async def execute(self, _statement: Any, _params: dict[str, Any]) -> _ScalarResult:
            try:
                await asyncio.Event().wait()
            finally:
                lock_wait_cancelled.set()
            return _ScalarResult(False)

    lock_engine = _AdvisoryLockEngine(SlowLockSession())

    async def fake_get_engine():
        return lock_engine

    service = OkxOrderFactSyncService(mode="paper")
    service.timeout_seconds = 0.01
    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://test")
    monkeypatch.setattr(
        order_fact_sync_module,
        "ORDER_FACT_SYNC_HARD_DEADLINE_GRACE_SECONDS",
        0.01,
    )
    monkeypatch.setattr(order_fact_sync_module, "get_engine", fake_get_engine)

    report = await service.sync()

    assert lock_wait_cancelled.is_set() is True
    assert report["status"] == "deferred"
    assert report["deferred_stages"] == ["hard_deadline"]


@pytest.mark.asyncio
async def test_postgres_order_fact_writes_configure_bounded_database_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _StatementCaptureSession()
    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://test")

    await _configure_order_fact_write_transaction(session)

    statements = [str(statement) for statement in session.statements]
    assert any("lock_timeout" in statement for statement in statements)
    assert any("statement_timeout" in statement for statement in statements)


def test_database_timeout_stage_classifies_postgres_lock_and_statement_timeouts() -> None:
    assert _database_timeout_stage(RuntimeError("canceling statement due to lock timeout")) == (
        "database_lock_timeout"
    )
    assert _database_timeout_stage(
        RuntimeError("canceling statement due to statement timeout")
    ) == "database_statement_timeout"
    assert _database_timeout_stage(RuntimeError("unrelated failure")) is None


@pytest.mark.asyncio
async def test_non_postgres_sync_uses_same_hard_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = asyncio.Event()
    service = OkxOrderFactSyncService(mode="paper")
    service.timeout_seconds = 0.01

    async def never_finish() -> dict[str, Any]:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(settings, "database_url", "sqlite+aiosqlite:///test.db")
    monkeypatch.setattr(
        order_fact_sync_module,
        "ORDER_FACT_SYNC_HARD_DEADLINE_GRACE_SECONDS",
        0.01,
    )
    monkeypatch.setattr(service, "_sync_single_writer", never_finish)

    report = await service.sync()

    assert cancelled.is_set() is True
    assert report["status"] == "deferred"
    assert report["deferred_stages"] == ["hard_deadline"]
    assert report["error"] == "order_fact_sync_hard_deadline_exceeded"


@pytest.mark.asyncio
async def test_background_fact_queries_never_wait_on_busy_order_rows() -> None:
    service = OkxOrderFactSyncService(mode="paper")
    session = _StatementCaptureSession()

    await service._load_stored_slippage_refresh_orders(
        session,
        for_update=False,
    )
    await service._load_stored_slippage_refresh_orders(
        session,
        for_update=True,
    )
    await service._load_writable_refresh_orders(
        session,
        datetime.now(UTC).replace(tzinfo=None),
    )

    statements = [
        str(
            statement.compile(
                dialect=postgresql.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        ).upper()
        for statement in session.statements
    ]
    assert "FOR UPDATE" not in statements[0]
    assert all("FOR UPDATE SKIP LOCKED" in statement for statement in statements[1:])


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
async def test_recovery_order_continues_through_protection_history(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "order-fact-recovery-stages.db")
    now = datetime.now(UTC)
    order_id = "okx-recovery-order"
    ccxt = _FakeCcxt(
        fills=[_fill_row(now, order_id=order_id)],
        orders=[_order_row(now, order_id=order_id)],
    )
    try:
        report = await OkxOrderFactSyncService(
            mode="paper",
            timeout_seconds=5.0,
            recovery_order_ids=(order_id,),
            executor_factory=_executor_factory(ccxt),
        ).sync()

        assert "protection" in ccxt.calls
        assert report["source"] == "okx_native_orders_and_fills"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_stored_fact_fallback_recovers_confirmed_protection_position(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "stored-protection-recovery.db")
    now = datetime.now(UTC)
    entry_order_id = "stored-entry"
    close_order_id = "stored-protection-close"
    try:
        async with get_session_ctx() as session:
            position = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="ETH/USDT",
                side="long",
                quantity=0.5,
                entry_price=1900.0,
                current_price=1920.0,
                is_open=False,
                closed_at=now - timedelta(minutes=10),
                okx_inst_id="ETH-USDT-SWAP",
                entry_exchange_order_id=entry_order_id,
                close_exchange_order_id="okx_orphan_quarantine:5721",
                settlement_status="settlement_quarantined",
                current_management_contract={
                    "original_entry_order_ids": [entry_order_id],
                    "protection_orders": [
                        {"algo_id": "stored-protection-algo", "reduce_only": True}
                    ],
                },
            )
            close_order = Order(
                model_name="okx_authoritative_sync",
                execution_mode="paper",
                symbol="ETH/USDT",
                side="sell",
                order_type="market",
                quantity=0.5,
                price=1930.0,
                status="filled",
                fee=0.2,
                exchange_order_id=close_order_id,
                filled_at=now,
                okx_inst_id="ETH-USDT-SWAP",
                okx_fill_pnl=15.0,
                okx_sync_status=OKX_SYNC_CONFIRMED,
                okx_raw_fills={
                    "fills_history_confirmed": True,
                    "order_id": close_order_id,
                    "base_quantity": 0.5,
                    "protection_execution": {
                        "lifecycle_complete": True,
                        "source_authority": "okx_algo_history_plus_fills_history",
                        "generated_order_id": close_order_id,
                        "algo_id": "stored-protection-algo",
                        "inst_id": "ETH-USDT-SWAP",
                        "position_side": "long",
                        "close_side": "sell",
                        "reduce_only": True,
                    },
                },
                created_at=now,
            )
            session.add_all([position, close_order])

        since = now - timedelta(days=1)
        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_executor_factory(_FakeCcxt()),
        )._sync_from_stored_facts(
            since=since,
            since_naive=since.replace(tzinfo=None),
            started_at=now,
            completed_stages=[],
            deferred_stages=["initialize"],
            stage_errors=[],
            pull_error="test OKX initialization failure",
            initial_confirmed_count=0,
            initial_samples=[],
        )

        async with get_session_ctx() as session:
            recovered = (
                await session.execute(select(Position).where(Position.symbol == "ETH/USDT"))
            ).scalar_one()
        assert report["status"] == "degraded"
        assert report["position_lifecycle_recovered_count"] == 1
        assert recovered.close_exchange_order_id == close_order_id
        assert recovered.settlement_status == "settlement_quarantined"
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


def test_targeted_fill_queries_prioritize_unconfirmed_recent_fill_over_slippage_refresh() -> None:
    now = datetime.now(UTC)
    recent = SimpleNamespace(
        exchange_order_id="recent-generic",
        filled_at=now,
        created_at=now,
        okx_raw_fills={"execution_result_confirmed": True},
    )
    incomplete_slippage = SimpleNamespace(
        exchange_order_id="older-slippage-gap",
        okx_inst_id="BTC-USDT-SWAP",
        filled_at=now - timedelta(days=3),
        created_at=now - timedelta(days=3),
        okx_raw_fills={
            "fills_history_confirmed": True,
            "order_id": "older-slippage-gap",
            "trade_ids": ["trade-older-slippage-gap"],
            "inst_id": "BTC-USDT-SWAP",
            "contracts": 2.0,
            "avg_price": 60000.0,
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
    ) == ["recent-generic"]


def test_targeted_fill_query_uses_history_outside_okx_recent_retention() -> None:
    now = datetime.now(UTC)
    recent = SimpleNamespace(
        exchange_order_id="recent-order",
        filled_at=now - timedelta(hours=2),
        created_at=now - timedelta(hours=2),
    )
    old = SimpleNamespace(
        exchange_order_id="old-order",
        filled_at=now - timedelta(days=4),
        created_at=now - timedelta(days=4),
    )

    assert not _target_fill_query_requires_historical(
        ["recent-order"],
        orders=[recent, old],
        now=now,
    )
    assert _target_fill_query_requires_historical(
        ["old-order"],
        orders=[recent, old],
        now=now,
    )
    assert _target_fill_query_requires_historical(
        ["explicit-recovery-without-local-row"],
        orders=[recent, old],
        now=now,
    )


@pytest.mark.asyncio
async def test_old_targeted_fill_recovers_from_okx_historical_endpoint(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "old-targeted-historical-fill.db")
    now = datetime.now(UTC)
    filled_at = now - timedelta(days=4)
    order_id = "historical-fill-order"
    ccxt = _HistoricalOnlyTargetCcxt(
        fills=[_fill_row(filled_at, order_id=order_id)],
    )
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    side="buy",
                    order_type="market",
                    quantity=0.02,
                    price=60000.0,
                    status="filled",
                    fee=0.0,
                    exchange_order_id=order_id,
                    created_at=filled_at,
                    filled_at=filled_at,
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            timeout_seconds=3.0,
            priority_only=True,
            executor_factory=_executor_factory(ccxt),
        ).sync()

        async with get_session_ctx() as session:
            order = (await session.execute(select(Order))).scalar_one()
        assert ccxt.calls[0] == "fills_historical_targeted"
        assert "fills_recent_targeted" not in ccxt.calls
        assert report["status"] == "ok"
        assert report["confirmed_count"] == 1
        assert report["unverified_count"] == 0
        assert order.okx_sync_status == OKX_SYNC_CONFIRMED
        assert order.okx_raw_fills["fills_history_confirmed"] is True
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_priority_only_sync_is_idle_without_pending_local_order(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "priority-only-idle.db")
    ccxt = _FakeCcxt()
    try:
        report = await OkxOrderFactSyncService(
            mode="paper",
            priority_only=True,
            executor_factory=_executor_factory(ccxt),
        ).sync()

        assert report["status"] == "ok"
        assert report["source"] == "okx_native_order_priority_sync"
        assert report["completed_stages"] == ["priority_queue_idle"]
        assert ccxt.calls == []
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_recovery_order_id_backfills_okx_only_fill_before_account_history(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "targeted-recovery-fast-lane.db")
    now = datetime.now(UTC)
    recovery_order_id = "3785286327831597056"
    ccxt = _SlowAccountFastTargetCcxt(
        fills=[_act_fill_row(now, order_id=recovery_order_id)],
        orders=[_act_order_row(now, order_id=recovery_order_id)],
        instruments=[_act_instrument_row()],
    )
    try:
        report = await OkxOrderFactSyncService(
            mode="paper",
            timeout_seconds=2.0,
            priority_only=True,
            recovery_order_ids=[recovery_order_id],
            executor_factory=_executor_factory(ccxt),
        ).sync()

        async with get_session_ctx() as session:
            order = (await session.execute(select(Order))).scalar_one()
        assert ccxt.calls[0] == "fills_targeted"
        assert "fills_account" not in ccxt.calls
        assert "recovery_order_facts_persisted" in report["completed_stages"]
        assert report["backfilled_count"] == 1
        assert order.exchange_order_id == recovery_order_id
        assert order.model_name == "okx_authoritative_sync"
        assert order.decision_id is None
        assert order.okx_raw_fills["contract_size_verified"] is True
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_recovery_order_id_defers_without_public_contract_size(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "targeted-recovery-no-contract.db")
    now = datetime.now(UTC)
    recovery_order_id = "3785286327831597057"
    ccxt = _SlowAccountFastTargetCcxt(
        fills=[_act_fill_row(now, order_id=recovery_order_id)],
        instruments=[],
    )
    try:
        report = await OkxOrderFactSyncService(
            mode="paper",
            timeout_seconds=2.0,
            priority_only=True,
            recovery_order_ids=[recovery_order_id],
            executor_factory=_executor_factory(ccxt),
        ).sync()

        async with get_session_ctx() as session:
            orders = list((await session.execute(select(Order))).scalars().all())
        assert orders == []
        assert report["backfilled_count"] == 0
        assert report["contract_size_deferred_count"] == 1
        assert report["status"] == "deferred"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_targeted_recent_fill_is_persisted_before_slow_account_history(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "targeted-fill-fast-lane.db")
    now = datetime.now(UTC)
    ccxt = _SlowAccountFastTargetCcxt(
        fills=[_act_fill_row(now, order_id="act-fast-lane")],
        orders=[_act_order_row(now, order_id="act-fast-lane")],
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
                    exchange_order_id="act-fast-lane",
                    okx_inst_id="ACT-USDT-SWAP",
                    okx_fill_contracts=4.0,
                    okx_fill_pnl=0.0,
                    okx_sync_status=OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
                    okx_raw_fills={
                        "fills_history_confirmed": False,
                        "execution_result_confirmed": True,
                        "order_id": "act-fast-lane",
                        "trade_ids": ["trade-act-fast-lane"],
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

        service = OkxOrderFactSyncService(
            mode="paper",
            timeout_seconds=0.8,
            priority_only=True,
            executor_factory=_executor_factory(ccxt),
        )
        slippage_refresh_called = False

        async def blocked_slippage_refresh() -> tuple[int, list[dict[str, Any]]]:
            nonlocal slippage_refresh_called
            slippage_refresh_called = True
            await asyncio.Event().wait()
            return 0, []

        monkeypatch.setattr(
            service,
            "_refresh_stored_slippage_from_rows",
            blocked_slippage_refresh,
        )

        report = await service.sync()

        async with get_session_ctx() as session:
            order = (await session.execute(select(Order))).scalar_one()
        assert slippage_refresh_called is False
        assert ccxt.calls[0] == "fills_targeted"
        assert "fills_account" not in ccxt.calls
        assert "account_history_after_priority_fast_lane" in report["deferred_stages"]
        assert report["confirmed_count"] == 1
        assert order.okx_sync_status == OKX_SYNC_CONFIRMED
        assert order.okx_raw_fills["fills_history_confirmed"] is True
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_completed_submit_stage_recovers_fill_overwritten_by_outer_timeout(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "completed-submit-fill-recovery.db")
    now = datetime.now(UTC)
    ccxt = _SlowAccountFastTargetCcxt(
        fills=[_fill_row(now, order_id="okx-timeout-fill")],
        orders=[_order_row(now, order_id="okx-timeout-fill")],
    )
    try:
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="BTC/USDT",
                action="long",
                confidence=0.5,
                reasoning="test",
                is_paper=True,
                was_executed=False,
                execution_reason="outer timeout",
                raw_llm_response={
                    "decision_state_machine": {
                        "stages": [
                            {
                                "stage": "exchange_submit",
                                "status": "passed",
                                "data": {
                                    "has_execution_result": True,
                                    "status": "filled",
                                    "exchange_order_id": "okx-timeout-fill",
                                },
                            },
                            {
                                "stage": "exchange_submit",
                                "status": "failed",
                                "data": {"error_type": "cancelled"},
                            },
                        ]
                    },
                    "execution_result": {
                        "source": "exchange_not_confirmed",
                        "status": "rejected",
                        "exchange_order_id": None,
                    },
                },
            )
            session.add(decision)
            await session.flush()
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    side="buy",
                    order_type="market",
                    quantity=0.0,
                    price=0.0,
                    status="rejected",
                    decision_id=decision.id,
                    exchange_order_id=None,
                    created_at=now,
                    filled_at=now,
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            timeout_seconds=0.8,
            executor_factory=_executor_factory(ccxt),
        ).sync()

        async with get_session_ctx() as session:
            order = (await session.execute(select(Order))).scalar_one()
            recovered_decision = (
                await session.execute(
                    select(AIDecision).where(AIDecision.id == order.decision_id)
                )
            ).scalar_one()
        assert ccxt.calls[0] == "fills_targeted"
        assert "fills_account" not in ccxt.calls
        assert report["confirmed_count"] == 1
        assert report["samples"][0]["kind"] == (
            "local_order_confirmed_from_completed_submit_stage"
        )
        assert order.status == "filled"
        assert order.exchange_order_id == "okx-timeout-fill"
        assert order.quantity == pytest.approx(0.02)
        assert order.okx_sync_status == OKX_SYNC_CONFIRMED
        assert recovered_decision.was_executed is True
        assert recovered_decision.raw_llm_response["execution_result"][
            "exchange_confirmed"
        ] is True
        assert recovered_decision.raw_llm_response["completed_submit_fill_recovery"][
            "exchange_order_id"
        ] == "okx-timeout-fill"
    finally:
        await close_db()


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


def test_matching_native_fill_refreshes_missing_protection_execution() -> None:
    now = datetime.now(UTC)
    order_id = "act-protection-fill"
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
        created_at=now,
        filled_at=now,
    )
    service = OkxOrderFactSyncService(mode="paper")
    service._apply_fill_to_order(
        order,
        fill,
        now=now,
        sync_status=OKX_SYNC_CONFIRMED,
        contract_size=1.0,
        contract_size_source="okx_public_instruments",
    )
    lifecycle = {
        "lifecycle_complete": True,
        "source_authority": "okx_algo_history_plus_fills_history",
        "actual_side": "tp",
    }

    confirmed, unverified, skipped, deferred, samples = service._apply_local_order_facts(
        [order],
        fills=[fill],
        fills_by_order_id={order_id: fill},
        order_rows_by_id={},
        protection_execution_by_order_id={order_id: lifecycle},
        contract_sizes={"ACT-USDT-SWAP": 1.0},
        decisions_by_id={},
        now=now,
        since=now - timedelta(minutes=1),
        authoritative_absence_order_ids=set(),
    )

    assert (confirmed, unverified, skipped, deferred) == (1, 0, 0, 0)
    assert samples[0]["kind"] == "local_order_protection_execution_refreshed"
    assert order.okx_raw_fills["protection_execution"] == lifecycle


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
        assert ccxt.calls.count("fills") >= 2
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
