from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.account import ExecutionEquitySnapshot
from models.trade import Order, Position
from scripts import run_phase3_okx_fact_sync as script
from services.okx_order_fact_sync import PHASE3_DEFAULT_ORDER_SYNC_START


def _report(status: str = "ok") -> dict:
    return {
        "status": status,
        "can_open_new_entries": status == "ok",
        "can_refresh_training": status == "ok",
        "requires_attention": status != "ok",
        "issue_ledger": {"summary": {"unresolved": 0 if status == "ok" else 1}},
    }


@pytest.mark.asyncio
async def test_phase3_okx_fact_sync_default_is_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    async def fake_collect_report(*, allow_cache: bool = False) -> dict:
        calls.append(allow_cache)
        return _report()

    monkeypatch.setattr(script, "collect_report", fake_collect_report)

    result = await script.run(mode="paper", apply_order_sync=False, allow_cache=True)

    assert result["mutated_database"] is False
    assert result["order_sync_applied"] is False
    assert result["order_sync_result"] is None
    assert calls == [True, False]


@pytest.mark.asyncio
async def test_phase3_okx_fact_sync_apply_runs_order_fact_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_collect_report(*, allow_cache: bool = False) -> dict:
        return _report()

    async def fake_cleanup(*, mode: str) -> dict:
        calls.append(f"cleanup:{mode}")
        return {"execution_equity_snapshots_deleted": 1}

    async def fake_equity_snapshot(*, mode: str) -> dict:
        calls.append(f"equity:{mode}")
        return {"status": "created", "equity": 4998.15}

    class FakeOrderSync:
        def __init__(self, *, mode: str) -> None:
            self.mode = mode

        async def sync(self) -> dict:
            calls.append(f"sync:{self.mode}")
            return {
                "status": "ok",
                "mode": self.mode,
                "confirmed_count": 2,
                "unverified_count": 0,
            }

    monkeypatch.setattr(script, "collect_report", fake_collect_report)
    monkeypatch.setattr(script, "_cleanup_phase3_local_okx_cache", fake_cleanup)
    monkeypatch.setattr(script, "_sync_okx_equity_snapshot", fake_equity_snapshot)
    monkeypatch.setattr(script, "OkxOrderFactSyncService", FakeOrderSync)

    result = await script.run(mode="paper", apply_order_sync=True, allow_cache=False)

    assert result["mutated_database"] is True
    assert result["order_sync_applied"] is True
    assert result["cleanup_result"]["execution_equity_snapshots_deleted"] == 1
    assert result["equity_snapshot_result"]["status"] == "created"
    assert result["order_sync_result"]["confirmed_count"] == 2
    assert result["after_reconciliation"]["status"] == "ok"
    assert calls == ["cleanup:paper", "equity:paper", "sync:paper"]


@pytest.mark.asyncio
async def test_cleanup_phase3_local_okx_cache_removes_phase3_orders_positions_and_equity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'phase3-cleanup.db').as_posix()}",
    )
    await init_db()
    phase3_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=10)
    ).astimezone(UTC).replace(tzinfo=None)
    old_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START - timedelta(hours=1)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SPK/USDT",
                        side="buy",
                        order_type="market",
                        quantity=1.0,
                        price=1.0,
                        status="filled",
                        fee=0.0,
                        exchange_order_id="phase3-order",
                        filled_at=phase3_time,
                        created_at=phase3_time,
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="OLD/USDT",
                        side="buy",
                        order_type="market",
                        quantity=1.0,
                        price=1.0,
                        status="filled",
                        fee=0.0,
                        exchange_order_id="old-order",
                        filled_at=old_time,
                        created_at=old_time,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SPK/USDT",
                        side="long",
                        quantity=1.0,
                        entry_price=1.0,
                        current_price=1.1,
                        is_open=False,
                        closed_at=phase3_time,
                        created_at=phase3_time,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="OLD/USDT",
                        side="long",
                        quantity=1.0,
                        entry_price=1.0,
                        current_price=1.1,
                        is_open=False,
                        closed_at=old_time,
                        created_at=old_time,
                    ),
                    ExecutionEquitySnapshot(
                        mode="paper",
                        model_name="ensemble_trader",
                        snapshot_date="2026-06-28",
                        snapshot_at=phase3_time,
                        equity=4000.0,
                        total_pnl=9.22,
                    ),
                    ExecutionEquitySnapshot(
                        mode="paper",
                        model_name="ensemble_trader",
                        snapshot_date="2026-06-27",
                        snapshot_at=old_time,
                        equity=4000.0,
                        total_pnl=0.0,
                        source="reconstructed",
                    ),
                    ExecutionEquitySnapshot(
                        mode="paper",
                        model_name="ensemble_trader",
                        snapshot_date="2026-06-29",
                        snapshot_at=phase3_time,
                        equity=4998.15,
                        total_pnl=0.0,
                        source="okx_snapshot",
                    ),
                    ExecutionEquitySnapshot(
                        mode="paper",
                        model_name="legacy_local_model",
                        snapshot_date="2026-06-29",
                        snapshot_at=phase3_time,
                        equity=7777.0,
                        total_pnl=0.0,
                        source="okx_snapshot",
                    ),
                ]
            )

        result = await script._cleanup_phase3_local_okx_cache(mode="paper")

        async with get_session_ctx() as session:
            order_rows = (await session.execute(Order.__table__.select())).all()
            position_rows = (await session.execute(Position.__table__.select())).all()
            equity_rows = (await session.execute(ExecutionEquitySnapshot.__table__.select())).all()
    finally:
        await close_db()

    assert result["orders_deleted"] == 1
    assert result["positions_deleted"] == 1
    assert result["execution_equity_snapshots_deleted"] == 3
    assert [row._mapping["exchange_order_id"] for row in order_rows] == ["old-order"]
    assert [row._mapping["symbol"] for row in position_rows] == ["OLD/USDT"]
    assert len(equity_rows) == 1
    assert equity_rows[0]._mapping["snapshot_date"] == "2026-06-29"
    assert equity_rows[0]._mapping["source"] == "okx_snapshot"


@pytest.mark.asyncio
async def test_sync_okx_equity_snapshot_creates_first_daily_okx_snapshot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'phase3-equity-sync.db').as_posix()}",
    )
    await init_db()

    class FakeExecutor:
        def __init__(self, *, mode: str, load_markets_on_initialize: bool) -> None:
            self.mode = mode
            self.load_markets_on_initialize = load_markets_on_initialize

        async def initialize(self) -> None:
            return None

        async def get_balance_snapshot(self, asset: str) -> dict:
            assert asset == "USDT"
            return {"equity": 4998.15, "total": 4998.15, "free": 4998.15}

        async def shutdown(self) -> None:
            return None

    monkeypatch.setattr(script, "OKXExecutor", FakeExecutor)
    try:
        result = await script._sync_okx_equity_snapshot(
            mode="paper",
            now=datetime(2026, 6, 28, 16, 30, tzinfo=UTC),
        )
        second = await script._sync_okx_equity_snapshot(
            mode="paper",
            now=datetime(2026, 6, 28, 17, 30, tzinfo=UTC),
        )

        async with get_session_ctx() as session:
            rows = (await session.execute(ExecutionEquitySnapshot.__table__.select())).all()
    finally:
        await close_db()

    assert result["status"] == "created"
    assert result["snapshot_date"] == "2026-06-29"
    assert result["equity"] == pytest.approx(4998.15)
    assert second["status"] == "kept_existing_okx_snapshot"
    assert len(rows) == 1
    assert rows[0]._mapping["source"] == "okx_snapshot"
    assert rows[0]._mapping["equity"] == pytest.approx(4998.15)


@pytest.mark.asyncio
async def test_sync_okx_equity_snapshot_audits_bills_without_writing_equity_baseline(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'phase3-equity-bills.db').as_posix()}",
    )
    await init_db()

    class FakeCcxt:
        async def privateGetAccountBills(self, params: dict) -> dict:
            assert params["ccy"] == "USDT"
            return {
                "data": [
                    {
                        "billId": "1",
                        "ccy": "USDT",
                        "balChg": "-1.85",
                        "ts": str(int(script.PHASE3_DEFAULT_ORDER_SYNC_START.timestamp() * 1000) + 1000),
                    }
                ]
            }

    class FakeExecutor:
        def __init__(self, *, mode: str, load_markets_on_initialize: bool) -> None:
            self.mode = mode

        async def initialize(self) -> None:
            return None

        async def get_balance_snapshot(self, asset: str) -> dict:
            assert asset == "USDT"
            return {"equity": 4998.15, "total": 4998.15, "free": 4998.15}

        async def _get_ccxt(self) -> FakeCcxt:
            return FakeCcxt()

        async def _with_retry(self, method, params: dict) -> dict:
            return await method(params)

        async def shutdown(self) -> None:
            return None

    monkeypatch.setattr(script, "OKXExecutor", FakeExecutor)
    try:
        result = await script._sync_okx_equity_snapshot(
            mode="paper",
            now=datetime(2026, 6, 28, 16, 30, tzinfo=UTC),
        )
        async with get_session_ctx() as session:
            rows = (await session.execute(ExecutionEquitySnapshot.__table__.select())).all()
    finally:
        await close_db()

    by_date = {row._mapping["snapshot_date"]: row._mapping for row in rows}
    assert result["account_bill_audit"]["available"] is True
    assert result["account_bill_audit"]["audit_only"] is True
    assert result["account_bill_audit"]["usable_for_equity_baseline"] is False
    assert result["account_bill_audit"]["net_balance_change_since_phase3"] == pytest.approx(-1.85)
    assert "phase3_start_equity" not in result["account_bill_audit"]
    assert "2026-06-28" not in by_date
    assert by_date["2026-06-29"]["equity"] == pytest.approx(4998.15)


@pytest.mark.asyncio
async def test_sync_okx_equity_snapshot_removes_previous_bill_derived_phase3_snapshot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'phase3-equity-bill-cleanup.db').as_posix()}",
    )
    await init_db()

    class FakeCcxt:
        async def privateGetAccountBills(self, params: dict) -> dict:
            return {
                "data": [
                    {
                        "billId": "1",
                        "ccy": "USDT",
                        "balChg": "-8.35",
                        "ts": str(int(script.PHASE3_DEFAULT_ORDER_SYNC_START.timestamp() * 1000) + 1000),
                    }
                ]
            }

    class FakeExecutor:
        def __init__(self, *, mode: str, load_markets_on_initialize: bool) -> None:
            self.mode = mode

        async def initialize(self) -> None:
            return None

        async def get_balance_snapshot(self, asset: str) -> dict:
            assert asset == "USDT"
            return {"equity": 5000.15, "total": 5000.15, "free": 5000.15}

        async def _get_ccxt(self) -> FakeCcxt:
            return FakeCcxt()

        async def _with_retry(self, method, params: dict) -> dict:
            return await method(params)

        async def shutdown(self) -> None:
            return None

    monkeypatch.setattr(script, "OKXExecutor", FakeExecutor)
    try:
        async with get_session_ctx() as session:
            session.add(
                ExecutionEquitySnapshot(
                    mode="paper",
                    model_name="ensemble_trader",
                    snapshot_date="2026-06-28",
                    snapshot_at=script.PHASE3_DEFAULT_ORDER_SYNC_START.astimezone(UTC),
                    equity=5008.50,
                    total_pnl=0.0,
                    realized_pnl=0.0,
                    unrealized_pnl=0.0,
                    source="okx_snapshot",
                )
            )

        result = await script._sync_okx_equity_snapshot(
            mode="paper",
            now=datetime(2026, 6, 28, 18, 30, tzinfo=UTC),
        )
        async with get_session_ctx() as session:
            rows = (await session.execute(ExecutionEquitySnapshot.__table__.select())).all()
    finally:
        await close_db()

    by_date = {row._mapping["snapshot_date"]: row._mapping for row in rows}
    assert result["legacy_phase3_snapshot_cleanup"]["deleted"] == 1
    assert "2026-06-28" not in by_date
    assert by_date["2026-06-29"]["equity"] == pytest.approx(5000.15)
