from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import event, select

from config.settings import settings
from db.session import close_db, get_engine, get_session_ctx, init_db
from models.account import OkxAccountBill
from models.trade import OkxPositionHistory, Position
from services import okx_position_history_store as position_history_store
from services import okx_settlement_fact_sync as settlement_fact_sync_module
from services.okx_position_history_store import (
    load_okx_position_history_watermark,
    publish_okx_position_history_watermark,
)
from services.okx_settlement_fact_sync import OkxSettlementFactSyncService


class _FakeCcxt:
    def __init__(
        self,
        *,
        history_rows: list[dict[str, Any]],
        bills: list[dict[str, Any]],
        history_rows_by_inst_id: dict[str, list[dict[str, Any]]] | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.history_rows = history_rows
        self.history_rows_by_inst_id = history_rows_by_inst_id or {}
        self.bills = bills
        self.delay_seconds = delay_seconds
        self.calls: list[str] = []
        self.history_params: list[dict[str, Any]] = []
        self.active_private_calls = 0
        self.max_active_private_calls = 0

    async def _private_response(
        self,
        call_name: str,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.calls.append(call_name)
        self.active_private_calls += 1
        self.max_active_private_calls = max(
            self.max_active_private_calls,
            self.active_private_calls,
        )
        try:
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            return {"data": rows}
        finally:
            self.active_private_calls -= 1

    async def privateGetAccountPositionsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.history_params.append(dict(params))
        rows = self.history_rows_by_inst_id.get(
            str(params.get("instId") or "").upper(),
            self.history_rows,
        )
        return await self._private_response("position_history", rows)

    async def privateGetAccountBills(self, _params: dict[str, Any]) -> dict[str, Any]:
        return await self._private_response("account_bills", self.bills)

    async def publicGetPublicInstruments(self, _params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("contract_specs")
        return {
            "data": [
                {
                    "instId": "ADA-USDT-SWAP",
                    "instType": "SWAP",
                    "ctVal": "1",
                    "ctMult": "1",
                    "lotSz": "1",
                    "minSz": "1",
                    "settleCcy": "USDT",
                }
            ]
        }


class _FakeExecutor:
    def __init__(
        self,
        ccxt: _FakeCcxt,
        *,
        circuit_status: dict[str, Any] | None = None,
    ) -> None:
        self.ccxt = ccxt
        self.circuit_status = circuit_status
        self.initialize_calls = 0
        self.shutdown_calls = 0

    async def initialize(self) -> None:
        self.initialize_calls += 1
        return None

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        return None

    def private_api_circuit_status(self) -> dict[str, Any]:
        return self.circuit_status or {
            "state": "closed",
            "background_calls_allowed": True,
        }

    async def _get_ccxt(self) -> _FakeCcxt:
        return self.ccxt

    async def _with_retry(self, fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result


def _executor_factory(ccxt: _FakeCcxt):
    return lambda *_args, **_kwargs: _FakeExecutor(ccxt)


def _ms(value: datetime) -> str:
    return str(int(value.timestamp() * 1000))


async def _init_test_db(tmp_path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / name).as_posix()}",
    )
    monkeypatch.setattr(
        settlement_fact_sync_module,
        "load_okx_position_history_watermark",
        lambda _mode: datetime.now(UTC),
    )
    await init_db()


def test_position_history_watermark_is_atomic_and_monotonic(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "history.watermark"
    monkeypatch.setattr(
        position_history_store,
        "okx_position_history_watermark_path",
        lambda _mode: path,
    )
    first = datetime(2026, 7, 29, 5, 0, tzinfo=UTC)
    second = first + timedelta(minutes=1)

    assert publish_okx_position_history_watermark("paper", changed_at=second) == second
    assert publish_okx_position_history_watermark("paper", changed_at=first) == second
    assert load_okx_position_history_watermark("paper") == second
    assert list(tmp_path.glob("*.tmp.*")) == []


@pytest.mark.asyncio
async def test_settlement_fact_sync_mirrors_history_and_funding_bills(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "settlement-facts.db")
    now = datetime.now(UTC)
    history_row = {
        "instId": "ADA-USDT-SWAP",
        "posId": "ada-pos-1",
        "posSide": "long",
        "type": "2",
        "cTime": _ms(now - timedelta(minutes=20)),
        "uTime": _ms(now - timedelta(minutes=5)),
        "openAvgPx": "0.6",
        "closeAvgPx": "0.64",
        "openMaxPos": "100",
        "closeTotalPos": "100",
        "realizedPnl": "4.2",
        "fundingFee": "-0.03",
        "fee": "-0.11",
    }
    bill = {
        "billId": "funding-1",
        "instId": "ADA-USDT-SWAP",
        "posSide": "long",
        "type": "8",
        "subType": "173",
        "balChg": "-0.03",
        "ts": _ms(now - timedelta(minutes=10)),
    }
    ccxt = _FakeCcxt(history_rows=[history_row], bills=[bill])
    try:
        report = await OkxSettlementFactSyncService(
            mode="paper",
            executor_factory=_executor_factory(ccxt),
        ).sync_once()

        assert report["status"] == "ok"
        assert report["position_history_count"] == 1
        assert report["targeted_pending_settlement_inst_count"] == 0
        assert report["position_history_inserted_count"] == 1
        assert report["account_bill_count"] == 1
        assert report["account_bill_inserted_count"] == 1
        assert ccxt.max_active_private_calls == 1
        assert {"position_history", "account_bills"} <= set(report["completed_stages"])
        async with get_session_ctx() as session:
            history = (await session.execute(select(OkxPositionHistory))).scalar_one()
            stored_bill = (await session.execute(select(OkxAccountBill))).scalar_one()
        assert history.source == "okx_settlement_fact_mirror"
        assert history.realized_pnl == pytest.approx(4.2)
        assert history.raw_row["_bb_contract_spec"]["ctVal"] == "1"
        assert stored_bill.source == "okx_settlement_fact_mirror"
        assert stored_bill.funding_fee == pytest.approx(-0.03)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_settlement_fact_sync_targeted_pull_covers_pending_local_instrument(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "settlement-facts-targeted.db")
    now = datetime.now(UTC)
    history_row = {
        "instId": "SUI-USDT-SWAP",
        "posId": "sui-pos-1",
        "posSide": "short",
        "type": "2",
        "cTime": _ms(now - timedelta(hours=30)),
        "uTime": _ms(now - timedelta(hours=29)),
        "openAvgPx": "1.1",
        "closeAvgPx": "1.0",
        "openMaxPos": "100",
        "closeTotalPos": "100",
        "realizedPnl": "10.0",
        "fundingFee": "0.2",
        "fee": "-0.1",
    }
    ccxt = _FakeCcxt(
        history_rows=[],
        history_rows_by_inst_id={"SUI-USDT-SWAP": [history_row]},
        bills=[],
    )
    async with get_session_ctx() as session:
        session.add(
            Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="SUI/USDT",
                side="short",
                quantity=100.0,
                entry_price=1.1,
                current_price=1.0,
                is_open=False,
                okx_inst_id="SUI-USDT-SWAP",
                okx_pos_id="sui-pos-1",
                settlement_status="pending_okx_authority",
                closed_at=now - timedelta(hours=28),
            )
        )
        session.add(
            Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="ADA/USDT",
                side="long",
                quantity=10.0,
                entry_price=0.6,
                current_price=0.61,
                is_open=False,
                okx_inst_id="ADA-USDT-SWAP",
                settlement_status="okx_position_history",
                closed_at=now - timedelta(hours=28),
            )
        )
        await session.flush()
    try:
        report = await OkxSettlementFactSyncService(
            mode="paper",
            executor_factory=_executor_factory(ccxt),
        ).sync_once()

        assert report["status"] == "ok"
        assert report["position_history_count"] == 1
        assert report["targeted_pending_settlement_inst_count"] == 1
        assert "position_history_targeted_pending_settlements" in report["completed_stages"]
        assert ccxt.history_params[0].get("instId") == "SUI-USDT-SWAP"
        assert ccxt.history_params[0].get("posId") == "sui-pos-1"
        assert all(
            params.get("instId") != "ADA-USDT-SWAP" for params in ccxt.history_params
        )
        assert any(
            params.get("instId") == "SUI-USDT-SWAP" for params in ccxt.history_params
        )
        async with get_session_ctx() as session:
            stored = (await session.execute(select(OkxPositionHistory))).scalar_one()
        assert stored.inst_id == "SUI-USDT-SWAP"
        assert stored.pos_id == "sui-pos-1"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_pending_settlement_targets_retry_quarantine_and_skip_terminal_rows(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "settlement-facts-pending-targets.db")
    now = datetime.now(UTC)
    async with get_session_ctx() as session:
        session.add_all(
            [
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="OLD/USDT",
                    side="short",
                    quantity=10.0,
                    entry_price=1.0,
                    current_price=1.0,
                    is_open=False,
                    okx_inst_id="OLD-USDT-SWAP",
                    okx_pos_id=None,
                    settlement_status="settling",
                    closed_at=now - timedelta(days=30),
                ),
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="STALE/USDT",
                    side="short",
                    quantity=10.0,
                    entry_price=1.0,
                    current_price=1.0,
                    is_open=False,
                    okx_inst_id="STALE-USDT-SWAP",
                    okx_pos_id="stale-pos-1",
                    settlement_status="settling",
                    closed_at=now - timedelta(days=7),
                ),
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="NEW/USDT",
                    side="long",
                    quantity=10.0,
                    entry_price=1.0,
                    current_price=1.0,
                    is_open=False,
                    okx_inst_id="NEW-USDT-SWAP",
                    okx_pos_id="new-pos-1",
                    settlement_status="settling",
                    closed_at=now - timedelta(minutes=5),
                ),
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="RESIDUAL/USDT",
                    side="short",
                    quantity=10.0,
                    entry_price=1.0,
                    current_price=1.0,
                    is_open=False,
                    okx_inst_id="RESIDUAL-USDT-SWAP",
                    okx_pos_id="residual-pos-1",
                    settlement_status="superseded_position_residual",
                    closed_at=now - timedelta(minutes=4),
                ),
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="QUARANTINED/USDT",
                    side="short",
                    quantity=10.0,
                    entry_price=1.0,
                    current_price=1.0,
                    is_open=False,
                    okx_inst_id="QUARANTINED-USDT-SWAP",
                    okx_pos_id="quarantined-pos-1",
                    settlement_status="settlement_quarantined",
                    closed_at=now - timedelta(minutes=3),
                ),
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="TERMINAL/USDT",
                    side="short",
                    quantity=10.0,
                    entry_price=1.0,
                    current_price=1.0,
                    is_open=False,
                    okx_inst_id="TERMINAL-USDT-SWAP",
                    okx_pos_id="terminal-pos-1",
                    settlement_status="settlement_unresolved",
                    closed_at=now - timedelta(minutes=2),
                ),
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="PARTIAL/USDT",
                    side="long",
                    quantity=4.0,
                    entry_price=1.0,
                    current_price=1.0,
                    is_open=False,
                    okx_inst_id="PARTIAL-USDT-SWAP",
                    okx_pos_id="partial-pos-1",
                    settlement_status="settling",
                    closed_at=now - timedelta(minutes=2),
                ),
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="PARTIAL/USDT",
                    side="long",
                    quantity=6.0,
                    entry_price=1.0,
                    current_price=1.0,
                    is_open=True,
                    okx_inst_id="PARTIAL-USDT-SWAP",
                    okx_pos_id="partial-pos-1",
                ),
            ]
        )
    try:
        targets = await OkxSettlementFactSyncService(mode="paper")._pending_settlement_targets(
            since=now - timedelta(hours=96)
        )
        assert targets == (
            ("NEW-USDT-SWAP", "new-pos-1"),
            ("QUARANTINED-USDT-SWAP", "quarantined-pos-1"),
        )
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_settlement_fact_sync_bulk_loads_existing_rows_once(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "settlement-facts-bulk.db")
    now = datetime.now(UTC)
    history_rows = [
        {
            "instId": "ADA-USDT-SWAP",
            "posId": f"ada-pos-{index}",
            "posSide": "long",
            "type": "2",
            "cTime": _ms(now - timedelta(minutes=30 + index)),
            "uTime": _ms(now - timedelta(minutes=5 + index)),
            "openAvgPx": "0.6",
            "closeAvgPx": "0.64",
            "openMaxPos": "100",
            "closeTotalPos": "100",
            "realizedPnl": "4.2",
            "fundingFee": "-0.03",
            "fee": "-0.11",
        }
        for index in range(3)
    ]
    bills = [
        {
            "billId": f"funding-{index}",
            "instId": "ADA-USDT-SWAP",
            "posSide": "long",
            "type": "8",
            "subType": "173",
            "balChg": "-0.03",
            "ts": _ms(now - timedelta(minutes=10 + index)),
        }
        for index in range(3)
    ]
    service = OkxSettlementFactSyncService(
        mode="paper",
        executor_factory=_executor_factory(_FakeCcxt(history_rows=history_rows, bills=bills)),
    )
    try:
        first = await service.sync_once()
        assert first["position_history_inserted_count"] == 3
        assert first["account_bill_inserted_count"] == 3

        engine = await get_engine()
        target_selects: list[str] = []

        def record_statement(
            _connection,
            _cursor,
            statement: str,
            _parameters,
            _context,
            _executemany,
        ) -> None:
            normalized = statement.lstrip().lower()
            if normalized.startswith("select") and (
                "okx_position_history" in normalized
                or "okx_account_bills" in normalized
            ):
                target_selects.append(normalized)

        event.listen(engine.sync_engine, "before_cursor_execute", record_statement)
        try:
            second = await service.sync_once()
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", record_statement)

        assert second["position_history_unchanged_count"] == 3
        assert (
            second["account_bill_updated_count"]
            + second["account_bill_unchanged_count"]
            == 3
        )
        assert len(target_selects) == 3
        assert sum("okx_position_history" in statement for statement in target_selects) == 2
        assert sum("okx_account_bills" in statement for statement in target_selects) == 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_settlement_fact_sync_defers_slow_private_pulls_under_one_budget(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "settlement-facts-timeout.db")
    ccxt = _FakeCcxt(history_rows=[], bills=[], delay_seconds=2.0)
    try:
        report = await OkxSettlementFactSyncService(
            mode="paper",
            timeout_seconds=0.5,
            executor_factory=_executor_factory(ccxt),
        ).sync_once()

        assert report["status"] == "deferred"
        assert {"position_history", "account_bills"} <= set(report["deferred_stages"])
        assert report["stage_errors"] == []
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_settlement_fact_sync_reuses_shared_executor_without_owning_lifecycle(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "settlement-facts-shared-executor.db")
    ccxt = _FakeCcxt(history_rows=[], bills=[])
    executor = _FakeExecutor(ccxt)

    async def executor_provider(mode: str) -> _FakeExecutor:
        assert mode == "paper"
        return executor

    try:
        report = await OkxSettlementFactSyncService(
            mode="paper",
            executor_provider=executor_provider,
        ).sync_once()

        assert report["status"] == "ok"
        assert report["okx_pull_available"] is True
        assert "shared_executor_reused" in report["completed_stages"]
        assert executor.initialize_calls == 0
        assert executor.shutdown_calls == 0
        assert {"position_history", "account_bills"} <= set(ccxt.calls)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_settlement_fact_sync_defers_shared_private_work_while_circuit_is_open(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "settlement-facts-open-circuit.db")
    ccxt = _FakeCcxt(history_rows=[], bills=[])
    executor = _FakeExecutor(
        ccxt,
        circuit_status={
            "state": "open",
            "background_calls_allowed": False,
            "retry_after_seconds": 12.0,
        },
    )

    async def executor_provider(_mode: str) -> _FakeExecutor:
        return executor

    try:
        report = await OkxSettlementFactSyncService(
            mode="paper",
            executor_provider=executor_provider,
        ).sync_once()

        assert report["status"] == "deferred"
        assert report["okx_pull_available"] is False
        assert "private_api_circuit_deferred" in report["completed_stages"]
        assert {"position_history", "account_bills"} <= set(report["deferred_stages"])
        assert ccxt.calls == []
        assert executor.initialize_calls == 0
        assert executor.shutdown_calls == 0
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_settlement_fact_sync_allows_shared_half_open_recovery_probe(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "settlement-facts-half-open.db")
    ccxt = _FakeCcxt(history_rows=[], bills=[])
    executor = _FakeExecutor(
        ccxt,
        circuit_status={
            "state": "half_open_ready",
            "background_calls_allowed": True,
            "retry_after_seconds": 0.0,
        },
    )

    async def executor_provider(_mode: str) -> _FakeExecutor:
        return executor

    try:
        report = await OkxSettlementFactSyncService(
            mode="paper",
            executor_provider=executor_provider,
        ).sync_once()

        assert report["status"] == "ok"
        assert report["okx_pull_available"] is True
        assert {"position_history", "account_bills"} <= set(ccxt.calls)
        assert executor.initialize_calls == 0
        assert executor.shutdown_calls == 0
    finally:
        await close_db()
