from __future__ import annotations

import sys
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx
import paramiko
import pytest

from config.settings import ENSEMBLE_TRADER_NAME, settings
from core.safe_output import safe_error_text
from core.server_monitor_probe import SERVER_MONITOR_REMOTE_COMMAND_TIMEOUT_SECONDS
from core.trading_mode import TradingMode
from db.session import close_db, get_session_ctx, init_db
from models.account import ExecutionEquitySnapshot
from models.decision import AIDecision
from models.trade import Order, Position
from services import server_monitor_status
from services.okx_order_fact_sync import OKX_SYNC_CONFIRMED
from web_dashboard.api import dashboard, symbols


class _FakeServerInfo:
    host = "203.0.113.17"
    access_host = "203.0.113.17"
    port = 22
    username = "root"
    source_path = "model_server_info.txt"

    def redacted(self) -> dict[str, object]:
        return {
            "host": self.host,
            "access_host": self.access_host,
            "port": self.port,
            "username": self.username,
            "password": "***",
            "source_path": self.source_path,
        }


class _FakeSSH:
    def __init__(self) -> None:
        self.closed = False
        self.exec_calls: list[dict[str, Any]] = []

    def close(self) -> None:
        self.closed = True


def _confirmed_okx_order(
    *,
    symbol: str,
    exchange_order_id: str,
    side: str,
    quantity: float,
    price: float,
    filled_at: datetime,
    pnl: float = 0.0,
) -> Order:
    inst_id = symbol.replace("/", "-") + "-SWAP"
    return Order(
        model_name="okx_authoritative_sync",
        execution_mode="paper",
        symbol=symbol,
        side=side,
        order_type="market",
        quantity=quantity,
        price=price,
        status="filled",
        fee=0.0,
        exchange_order_id=exchange_order_id,
        filled_at=filled_at,
        created_at=filled_at,
        okx_inst_id=inst_id,
        okx_trade_ids=f"trade-{exchange_order_id}",
        okx_fill_contracts=quantity,
        okx_fill_pnl=pnl,
        okx_sync_status=OKX_SYNC_CONFIRMED,
        okx_raw_fills={
            "order_id": exchange_order_id,
            "trade_ids": [f"trade-{exchange_order_id}"],
            "inst_id": inst_id,
            "contracts": quantity,
            "contract_size": 1.0,
            "base_quantity": quantity,
            "avg_price": price,
            "fee_abs": 0.0,
            "fill_pnl": pnl,
            "timestamp": filled_at.isoformat(),
        },
    )


def _build_server_monitor_service(
    *,
    status: int = 0,
    stdout: str = "{}",
    stderr: str = "",
    raise_exc: Exception | None = None,
    clock=None,
) -> tuple[server_monitor_status.ServerMonitorStatusService, _FakeSSH]:
    ssh = _FakeSSH()

    def fake_exec_remote_command(*args: Any, **kwargs: Any) -> SimpleNamespace:
        ssh.exec_calls.append({"args": args, **kwargs})
        if raise_exc is not None:
            raise raise_exc
        return SimpleNamespace(status=status, stdout=stdout, stderr=stderr)

    service = server_monitor_status.ServerMonitorStatusService(
        model_id_provider=lambda: "qwen3-32b-trade",
        info_loader=lambda _root: _FakeServerInfo(),
        ssh_connector=lambda *args, **kwargs: ssh,
        command_executor=fake_exec_remote_command,
        clock=clock or server_monitor_status.monotonic,
    )
    return service, ssh


def test_safe_error_text_redacts_and_truncates_secret_bearing_text() -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    text = f"Authorization: Bearer {leaked_value} failed. " "x" * 120

    result = safe_error_text(text, limit=40)

    assert leaked_value not in result
    assert "Authorization: ***" in result
    assert len(result) == 43
    assert result.endswith("...")


def test_dashboard_execution_account_payload_separates_allocation_from_okx_equity(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dashboard.settings,
        "execution_account_balances",
        {**dashboard.settings.execution_account_balances, "paper": 123.45},
    )
    monkeypatch.setattr(dashboard, "_trading_service", None)

    payload = dashboard._build_execution_account_status(
        "paper",
        paper_summary={"available_balance": 100.0, "positions": []},
        okx_account={"free": 200.0, "used": 10.0, "total": 250.0, "equity": 260.0},
        pnl_summary={},
    )

    assert payload["allocated_balance"] is None
    assert payload["account_balance_source_value"] == 260.0
    assert payload["account_equity"] == 260.0


def test_dashboard_execution_account_refuses_synthetic_balance_when_okx_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dashboard.settings,
        "execution_account_balances",
        {**dashboard.settings.execution_account_balances, "paper": 4000.0},
    )
    monkeypatch.setattr(dashboard, "_trading_service", None)

    payload = dashboard._build_execution_account_status(
        "paper",
        paper_summary={"available_balance": 4000.0, "equity": 4000.0, "positions": []},
        okx_account=None,
        pnl_summary={"total_pnl": 9.22, "today_total_pnl": 9.22},
    )

    assert payload["account_equity"] == 0.0
    assert payload["allocated_balance"] is None
    assert payload["available_balance"] is None
    assert payload["total_pnl"] is None
    assert payload["cumulative_total_pnl"] is None
    assert payload["local_trade_total_pnl"] == 9.22
    assert payload["account_pnl_source"] == "okx_unavailable"
    assert payload["risk_paused"] is True


def test_dashboard_execution_account_does_not_promote_local_trade_pnl_to_account_pnl(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dashboard, "_trading_service", None)

    payload = dashboard._build_execution_account_status(
        "paper",
        paper_summary={"available_balance": 4000.0, "equity": 4000.0, "positions": []},
        okx_account={"free": 4998.15, "total": 4998.15, "equity": 4998.15},
        pnl_summary={
            "total_pnl": 9.22,
            "today_total_pnl": 9.22,
            "today_closed_realized_pnl": 1.23,
        },
    )

    assert payload["account_equity"] == 4998.15
    assert payload["total_pnl"] is None
    assert payload["today_total_pnl"] is None
    assert payload["cumulative_total_pnl"] is None
    assert payload["local_trade_total_pnl"] == 9.22
    assert payload["local_trade_today_pnl"] == 1.23


def test_dashboard_execution_account_uses_okx_equity_not_cash_for_today_pnl(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dashboard, "_trading_service", None)

    payload = dashboard._build_execution_account_status(
        "paper",
        paper_summary={"available_balance": 4965.95, "positions": []},
        okx_account={
            "free": 4965.95,
            "used": 24.74,
            "total": 4990.70,
            "cash": 4991.26,
            "equity": 4990.70,
            "allocatable": 4990.70,
        },
        pnl_summary={"today_equity_baseline": 4990.90},
    )

    assert payload["account_equity"] == pytest.approx(4990.70)
    assert payload["okx_equity_balance"] == pytest.approx(4990.70)
    assert payload["today_equity_pnl"] == pytest.approx(-0.20)
    assert payload["today_total_pnl"] == pytest.approx(-0.20)


@pytest.mark.asyncio
async def test_dashboard_pnl_history_uses_only_okx_equity_snapshots(
    tmp_path,
    monkeypatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-pnl-history.db').as_posix()}",
    )
    await init_db()

    class PaperExecutor:
        async def get_account_summary(self, _model_name):
            raise AssertionError("pnl history must not read paper executor virtual balances")

    class TradingService:
        paper_executor = PaperExecutor()
        models = {"ensemble_trader": object()}

        def get_pnl_history(self):
            raise AssertionError("pnl history must not read in-memory local PnL")

    monkeypatch.setattr(dashboard, "_trading_service", TradingService())
    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    ExecutionEquitySnapshot(
                        mode="paper",
                        model_name="ensemble_trader",
                        snapshot_date="2026-06-28",
                        snapshot_at=datetime(2026, 6, 28, 0, 0, tzinfo=UTC),
                        equity=4998.15,
                        total_pnl=0.0,
                        realized_pnl=0.0,
                        unrealized_pnl=0.0,
                        source="okx_snapshot",
                    ),
                    ExecutionEquitySnapshot(
                        mode="paper",
                        model_name="ensemble_trader",
                        snapshot_date="2026-06-29",
                        snapshot_at=datetime(2026, 6, 29, 0, 0, tzinfo=UTC),
                        equity=5001.15,
                        total_pnl=0.0,
                        realized_pnl=0.0,
                        unrealized_pnl=0.0,
                        source="okx_snapshot",
                    ),
                ]
            )

        payload = await dashboard.get_pnl_history("paper")
        history = payload["history"]["ensemble_trader"]
    finally:
        await close_db()

    assert history["source"] == "okx_equity_snapshots"
    assert history["pnl_curve"] == [0.0, pytest.approx(0.060022)]


@pytest.mark.asyncio
async def test_opening_funnel_limits_market_rows_before_position_rows(
    tmp_path,
    monkeypatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'opening-funnel.db').as_posix()}",
    )
    await init_db()

    try:
        now = datetime.now(UTC)
        async with get_session_ctx() as session:
            session.add_all(
                [
                    AIDecision(
                        model_name=ENSEMBLE_TRADER_NAME,
                        symbol=f"P{i}/USDT",
                        action="hold",
                        confidence=0.2,
                        analysis_type="position",
                        is_paper=True,
                        created_at=now,
                    )
                    for i in range(600)
                ]
            )
            session.add_all(
                [
                    AIDecision(
                        model_name=ENSEMBLE_TRADER_NAME,
                        symbol="BTC/USDT",
                        action="short",
                        confidence=0.72,
                        analysis_type="market",
                        is_paper=True,
                        was_executed=False,
                        execution_reason="entry_evidence_wait",
                        created_at=now,
                    ),
                    AIDecision(
                        model_name=ENSEMBLE_TRADER_NAME,
                        symbol="ETH/USDT",
                        action="long",
                        confidence=0.69,
                        analysis_type="entry_candidate",
                        is_paper=True,
                        was_executed=False,
                        execution_reason="expected_net_return_not_positive",
                        created_at=now,
                    ),
                ]
            )
            await session.flush()

        payload = await dashboard._build_opening_funnel_payload(
            mode="paper",
            hours=24,
            limit=500,
        )

        assert payload["sampled_decisions"] == 2
        assert payload["market_scans"] == 2
        assert payload["stages"]["ai_entry_signals"] == 2
        assert payload["bottleneck"] != "no_data"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_model_contribution_stats_reports_lineage_gap_when_orders_lack_decision_id(
    tmp_path,
    monkeypatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'model-contribution-lineage.db').as_posix()}",
    )
    await init_db()

    try:
        now = datetime.now(UTC)
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="okx_authoritative_sync",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    side="long",
                    quantity=0.01,
                    entry_price=100.0,
                    current_price=110.0,
                    leverage=1.0,
                    unrealized_pnl=0.0,
                    realized_pnl=2.5,
                    is_open=False,
                    closed_at=now,
                    created_at=now,
                    entry_exchange_order_id="entry-ok",
                    close_exchange_order_id="close-ok",
                )
            )
            session.add(
                Order(
                    model_name=ENSEMBLE_TRADER_NAME,
                    execution_mode="paper",
                    symbol="BTC-USDT-SWAP",
                    side="buy",
                    order_type="market",
                    quantity=0.01,
                    price=100.0,
                    status="filled",
                    decision_id=None,
                    exchange_order_id="entry-ok",
                    filled_at=now,
                    created_at=now,
                )
            )

        payload = await dashboard.get_model_contribution_stats(mode="paper", days=7)
    finally:
        await close_db()

    assert payload["total_positions"] == 1
    assert payload["lineage"]["total_closed_positions"] == 1
    assert payload["lineage"]["filled_order_count"] == 1
    assert payload["lineage"]["orders_with_decision_id"] == 0
    assert payload["lineage"]["matched_position_count"] == 0
    assert payload["lineage"]["reason"] == "filled_orders_missing_decision_id"
    assert sum(int(row["count"]) for row in payload["stats"]) == 0


@pytest.mark.asyncio
async def test_model_contribution_stats_uses_okx_authoritative_order_decision_links(
    tmp_path,
    monkeypatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'model-contribution-okx-ledger.db').as_posix()}",
    )
    await init_db()

    try:
        now = datetime.now(UTC)
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name=ENSEMBLE_TRADER_NAME,
                symbol="BTC/USDT",
                action="long",
                confidence=0.82,
                position_size_pct=2.0,
                suggested_leverage=2.0,
                is_paper=True,
                was_executed=True,
                executed_at=now,
                execution_price=100.0,
                created_at=now,
                raw_llm_response={
                    "ml_signal": {
                        "available": True,
                        "predictions": [{"best_side": "long", "best_expected_return_pct": 1.2}],
                    }
                },
            )
            session.add(decision)
            await session.flush()
            session.add_all(
                [
                    Position(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        side="long",
                        quantity=0.01,
                        entry_price=100.0,
                        current_price=110.0,
                        leverage=1.0,
                        unrealized_pnl=0.0,
                        realized_pnl=2.5,
                        is_open=False,
                        closed_at=now,
                        created_at=now,
                        entry_exchange_order_id="entry-ok",
                        close_exchange_order_id="close-ok",
                    ),
                    Order(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="BTC-USDT-SWAP",
                        side="buy",
                        order_type="market",
                        quantity=0.01,
                        price=100.0,
                        status="filled",
                        decision_id=int(decision.id),
                        exchange_order_id="entry-ok",
                        filled_at=now,
                        created_at=now,
                    ),
                ]
            )

        payload = await dashboard.get_model_contribution_stats(mode="paper", days=7)
    finally:
        await close_db()

    nonzero_rows = [row for row in payload["stats"] if int(row["count"]) > 0]
    assert payload["lineage"]["filled_order_count"] == 1
    assert payload["lineage"]["orders_with_decision_id"] == 1
    assert payload["lineage"]["matched_position_count"] == 1
    assert payload["lineage"]["reason"] == "ok"
    assert len(nonzero_rows) == 1
    assert nonzero_rows[0]["count"] == 1
    assert nonzero_rows[0]["pnl"] == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_dashboard_execution_pnl_summary_gets_okx_equity_without_trading_service(
    tmp_path,
    monkeypatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-okx-equity.db').as_posix()}",
    )
    await init_db()

    async def okx_snapshot(_mode: str):
        return {"equity": 5001.0, "total": 5001.0, "free": 5001.0}

    monkeypatch.setattr(dashboard, "_trading_service", None)
    monkeypatch.setattr(dashboard, "_dashboard_okx_balance_snapshot_for_mode", okx_snapshot)
    monkeypatch.setattr(dashboard, "_get_exchange_position_mark_map", lambda _mode: _async_value({}))
    monkeypatch.setattr(dashboard, "_get_exchange_open_position_symbols", lambda _mode: _async_value(set()))

    try:
        summary = await dashboard._get_execution_pnl_summary("paper")
    finally:
        await close_db()

    assert summary["today_equity_baseline"] == pytest.approx(5001.0)
    assert summary["today_equity_baseline_source"] == "okx_snapshot"
    assert summary["today_equity_pnl"] == pytest.approx(0.0)
    assert summary["today_total_pnl"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_execution_pnl_summary_includes_okx_authoritative_ledger_positions(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-okx-ledger-positions.db').as_posix()}",
    )
    await init_db()

    async def okx_snapshot(_mode: str):
        return {"equity": 4998.15, "total": 4998.15, "free": 4998.15}

    exchange_marks = {
        ("XPL/USDT", "short"): {
            "quantity": 120.0,
            "entry_price": 0.1019,
            "mark_price": 0.0985,
            "unrealized_pnl": 0.408,
            "margin": 3.94,
        }
    }

    monkeypatch.setattr(dashboard, "_trading_service", None)
    monkeypatch.setattr(dashboard, "_dashboard_okx_balance_snapshot_for_mode", okx_snapshot)
    monkeypatch.setattr(dashboard, "_get_exchange_position_mark_map", lambda _mode: _async_value(exchange_marks))
    monkeypatch.setattr(dashboard, "_get_exchange_open_position_symbols", lambda _mode: _async_value({"XPL/USDT"}))

    try:
        async with get_session_ctx() as session:
            act_entry_at = datetime(2026, 6, 28, 5, 0, 59, tzinfo=UTC)
            act_close_at = datetime(2026, 6, 28, 7, 46, 45, tzinfo=UTC)
            session.add_all(
                [
                    Position(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="ACT/USDT",
                        side="short",
                        quantity=1270.0,
                        entry_price=0.01049,
                        current_price=0.00908,
                        leverage=3.0,
                        realized_pnl=1.77827305,
                        is_open=False,
                        entry_exchange_order_id="3695029485428248576",
                        close_exchange_order_id="3695363208212353024",
                        created_at=act_entry_at,
                        closed_at=act_close_at,
                    ),
                    Position(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="XPL/USDT",
                        side="short",
                        quantity=120.0,
                        entry_price=0.1019,
                        current_price=0.0985,
                        leverage=3.0,
                        unrealized_pnl=0.408,
                        realized_pnl=0.0,
                        is_open=True,
                        entry_exchange_order_id="3694821774166036480",
                        created_at=datetime(2026, 6, 28, 3, 17, 49, tzinfo=UTC),
                    ),
                    _confirmed_okx_order(
                        symbol="ACT/USDT",
                        exchange_order_id="3695029485428248576",
                        side="sell",
                        quantity=1270.0,
                        price=0.01049,
                        filled_at=act_entry_at,
                        pnl=0.0,
                    ),
                    _confirmed_okx_order(
                        symbol="ACT/USDT",
                        exchange_order_id="3695363208212353024",
                        side="buy",
                        quantity=1270.0,
                        price=0.00908,
                        filled_at=act_close_at,
                        pnl=1.77827305,
                    ),
                ]
            )

        summary = await dashboard._get_execution_pnl_summary("paper")
    finally:
        await close_db()

    assert summary["realized_profit"] == pytest.approx(1.77827305)
    assert summary["realized_loss"] == pytest.approx(0.0)
    assert summary["realized_pnl"] == pytest.approx(1.77827305)
    assert summary["open_positions"] == 1
    assert summary["unrealized_pnl"] == pytest.approx(0.408)
    assert summary["used_margin"] == pytest.approx(120.0 * 0.1019 / 3.0)


@pytest.mark.asyncio
async def test_display_open_positions_snapshot_groups_same_symbol_side_fragments(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-open-position-merge.db').as_posix()}",
    )
    await init_db()

    exchange_marks = {
        ("LAB/USDT", "short"): {
            "quantity": 11.0,
            "entry_price": 0.094,
            "mark_price": 0.091,
            "unrealized_pnl": 0.033,
            "margin": 0.35,
        }
    }

    monkeypatch.setattr(dashboard, "_data_service", None)
    monkeypatch.setattr(
        dashboard,
        "_get_exchange_position_mark_map",
        lambda _mode: _async_value(exchange_marks),
    )

    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="LAB/USDT",
                        side="short",
                        quantity=5.0,
                        entry_price=0.1,
                        current_price=0.095,
                        leverage=3.0,
                        unrealized_pnl=0.025,
                        realized_pnl=0.0,
                        is_open=True,
                        entry_exchange_order_id="lab-entry-1",
                        created_at=datetime(2026, 6, 29, 8, 0, tzinfo=UTC),
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="LAB-USDT",
                        side="short",
                        quantity=6.0,
                        entry_price=0.09,
                        current_price=0.095,
                        leverage=3.0,
                        unrealized_pnl=-0.03,
                        realized_pnl=0.0,
                        is_open=True,
                        entry_exchange_order_id="lab-entry-2",
                        created_at=datetime(2026, 6, 29, 8, 1, tzinfo=UTC),
                    ),
                ]
            )
        positions = await dashboard._get_display_open_positions_snapshot("paper")
    finally:
        await close_db()

    assert len(positions) == 1
    row = positions[0]
    assert row["symbol"] == "LAB/USDT"
    assert row["side"] == "short"
    assert row["quantity"] == pytest.approx(11.0)
    assert row["entry_price"] == pytest.approx(0.094)
    assert row["unrealized_pnl"] == pytest.approx(0.033)
    assert row["local_quantity"] == pytest.approx(11.0)
    assert row["local_entry_price"] == pytest.approx((5.0 * 0.1 + 6.0 * 0.09) / 11.0)
    assert row["merged_local_position_count"] == 2
    assert len(row["local_position_ids"]) == 2


@pytest.mark.asyncio
async def test_execution_pnl_summary_does_not_use_local_open_position_when_okx_has_none(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-no-local-open-fallback.db').as_posix()}",
    )
    await init_db()

    async def okx_snapshot(_mode: str):
        return {"equity": 4998.15, "total": 4998.15, "free": 4998.15}

    monkeypatch.setattr(dashboard, "_trading_service", None)
    monkeypatch.setattr(dashboard, "_dashboard_okx_balance_snapshot_for_mode", okx_snapshot)
    monkeypatch.setattr(dashboard, "_get_exchange_position_mark_map", lambda _mode: _async_value({}))
    monkeypatch.setattr(dashboard, "_get_exchange_open_position_symbols", lambda _mode: _async_value(set()))

    try:
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="HOME/USDT",
                    side="long",
                    quantity=100.0,
                    entry_price=1.0,
                    current_price=1.2,
                    leverage=2.0,
                    unrealized_pnl=20.0,
                    realized_pnl=0.0,
                    is_open=True,
                    created_at=datetime(2026, 6, 28, 1, 0, tzinfo=UTC),
                )
            )

        summary = await dashboard._get_execution_pnl_summary("paper")
    finally:
        await close_db()

    assert summary["open_positions"] == 0
    assert summary["unrealized_pnl"] == 0.0
    assert summary["used_margin"] == 0.0
    assert summary["total_pnl"] == 0.0


@pytest.mark.asyncio
async def test_daily_pnl_records_include_okx_authoritative_ledger_positions(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-daily-okx-ledger.db').as_posix()}",
    )
    await init_db()

    async def exchange_marks(_mode: str):
        return {}

    monkeypatch.setattr(dashboard, "_get_exchange_position_mark_map", exchange_marks)
    monkeypatch.setattr(dashboard, "_get_exchange_open_position_symbols", lambda _mode: _async_value(set()))

    try:
        async with get_session_ctx() as session:
            act_entry_at = datetime(2026, 6, 28, 5, 0, 59, tzinfo=UTC)
            act_close_at = datetime(2026, 6, 28, 7, 46, 45, tzinfo=UTC)
            session.add_all(
                [
                    Position(
                    model_name="okx_authoritative_sync",
                    execution_mode="paper",
                    symbol="ACT/USDT",
                    side="short",
                    quantity=1270.0,
                    entry_price=0.01049,
                    current_price=0.00908,
                    leverage=3.0,
                    realized_pnl=1.77827305,
                    is_open=False,
                    entry_exchange_order_id="3695029485428248576",
                    close_exchange_order_id="3695363208212353024",
                        created_at=act_entry_at,
                        closed_at=act_close_at,
                    ),
                    _confirmed_okx_order(
                        symbol="ACT/USDT",
                        exchange_order_id="3695029485428248576",
                        side="sell",
                        quantity=1270.0,
                        price=0.01049,
                        filled_at=act_entry_at,
                    ),
                    _confirmed_okx_order(
                        symbol="ACT/USDT",
                        exchange_order_id="3695363208212353024",
                        side="buy",
                        quantity=1270.0,
                        price=0.00908,
                        filled_at=act_close_at,
                        pnl=1.77827305,
                    ),
                ]
            )

        payload = await dashboard.get_daily_pnl_records(mode="paper", days=30)
    finally:
        await close_db()

    act_day = next(row for row in payload["records"] if row["date"] == "2026-06-28")
    assert act_day["trade_count"] == 1
    assert act_day["realized_profit"] == pytest.approx(1.77827305)
    assert act_day["realized_pnl"] == pytest.approx(1.77827305)
    assert act_day["symbols"] == ["ACT/USDT"]


@pytest.mark.asyncio
async def test_daily_pnl_records_include_phase3_closed_position_even_if_opened_before_phase3(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-daily-okx-closed-window.db').as_posix()}",
    )
    await init_db()

    monkeypatch.setattr(dashboard, "_get_exchange_position_mark_map", lambda _mode: _async_value({}))
    monkeypatch.setattr(
        dashboard, "_get_exchange_open_position_symbols", lambda _mode: _async_value(set())
    )

    try:
        async with get_session_ctx() as session:
            act_entry_at = datetime(2026, 6, 27, 15, 59, 59, tzinfo=UTC)
            act_close_at = datetime(2026, 6, 28, 5, 0, 59, tzinfo=UTC)
            session.add_all(
                [
                    Position(
                    model_name="okx_authoritative_sync",
                    execution_mode="paper",
                    symbol="ACT/USDT",
                    side="short",
                    quantity=1270.0,
                    entry_price=0.01049,
                    current_price=0.00908,
                    leverage=3.0,
                    realized_pnl=1.77827305,
                    is_open=False,
                    entry_exchange_order_id="3695029485428248576",
                    close_exchange_order_id="3695363208212353024",
                        created_at=act_entry_at,
                        closed_at=act_close_at,
                    ),
                    _confirmed_okx_order(
                        symbol="ACT/USDT",
                        exchange_order_id="3695029485428248576",
                        side="sell",
                        quantity=1270.0,
                        price=0.01049,
                        filled_at=act_entry_at,
                    ),
                    _confirmed_okx_order(
                        symbol="ACT/USDT",
                        exchange_order_id="3695363208212353024",
                        side="buy",
                        quantity=1270.0,
                        price=0.00908,
                        filled_at=act_close_at,
                        pnl=1.77827305,
                    ),
                ]
            )

        payload = await dashboard.get_daily_pnl_records(mode="paper", days=30)
    finally:
        await close_db()

    act_day = next(row for row in payload["records"] if row["date"] == "2026-06-28")
    assert act_day["trade_count"] == 1
    assert act_day["realized_profit"] == pytest.approx(1.77827305)
    assert act_day["symbols"] == ["ACT/USDT"]


@pytest.mark.asyncio
async def test_daily_pnl_records_exclude_local_position_without_okx_confirmed_orders(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-daily-local-only.db').as_posix()}",
    )
    await init_db()

    monkeypatch.setattr(dashboard, "_get_exchange_position_mark_map", lambda _mode: _async_value({}))
    monkeypatch.setattr(
        dashboard, "_get_exchange_open_position_symbols", lambda _mode: _async_value(set())
    )

    try:
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="okx_authoritative_sync",
                    execution_mode="paper",
                    symbol="FAKE/USDT",
                    side="long",
                    quantity=100.0,
                    entry_price=1.0,
                    current_price=1.1,
                    leverage=3.0,
                    realized_pnl=10.0,
                    is_open=False,
                    entry_exchange_order_id="local-entry",
                    close_exchange_order_id="local-close",
                    created_at=datetime(2026, 6, 28, 5, 0, tzinfo=UTC),
                    closed_at=datetime(2026, 6, 28, 6, 0, tzinfo=UTC),
                )
            )

        payload = await dashboard.get_daily_pnl_records(mode="paper", days=30)
    finally:
        await close_db()

    day = next(row for row in payload["records"] if row["date"] == "2026-06-28")
    assert day["trade_count"] == 0
    assert day["realized_pnl"] == 0.0
    assert day["symbols"] == []


@pytest.mark.asyncio
async def test_daily_pnl_records_do_not_emit_future_rows_after_phase3_clamp(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-daily-window.db').as_posix()}",
    )
    await init_db()

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = cls(2026, 6, 29, 1, 0, tzinfo=tz or UTC)
            return value

    monkeypatch.setattr(dashboard, "datetime", FrozenDatetime)
    monkeypatch.setattr(dashboard, "_get_exchange_position_mark_map", lambda _mode: _async_value({}))
    monkeypatch.setattr(dashboard, "_get_exchange_open_position_symbols", lambda _mode: _async_value(set()))

    try:
        payload = await dashboard.get_daily_pnl_records(mode="paper", days=30)
    finally:
        await close_db()

    assert payload["start_date"] == "2026-06-28"
    assert payload["end_date"] == "2026-06-29"
    assert [row["date"] for row in payload["records"]] == ["2026-06-29", "2026-06-28"]


@pytest.mark.asyncio
async def test_daily_pnl_today_row_uses_current_okx_equity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-daily-current-okx.db').as_posix()}",
    )
    await init_db()

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 6, 29, 12, 0, tzinfo=tz or UTC)

    async def okx_snapshot(_mode: str):
        return {"equity": 4997.95, "total": 4997.95, "free": 4931.0}

    monkeypatch.setattr(dashboard, "datetime", FrozenDatetime)
    monkeypatch.setattr(dashboard, "_get_exchange_position_mark_map", lambda _mode: _async_value({}))
    monkeypatch.setattr(dashboard, "_get_exchange_open_position_symbols", lambda _mode: _async_value(set()))
    monkeypatch.setattr(dashboard, "_get_dashboard_okx_account_snapshot", okx_snapshot)

    try:
        async with get_session_ctx() as session:
            session.add(
                ExecutionEquitySnapshot(
                    mode="paper",
                    model_name="ensemble_trader",
                    snapshot_date="2026-06-29",
                    snapshot_at=datetime(2026, 6, 28, 16, 0, tzinfo=UTC),
                    equity=4999.58,
                    source="okx_snapshot",
                )
            )
        payload = await dashboard.get_daily_pnl_records(mode="paper", days=3)
    finally:
        await close_db()

    today = next(row for row in payload["records"] if row["date"] == "2026-06-29")
    assert today["okx_equity"] == pytest.approx(4997.95)
    assert today["okx_equity_source"] == "okx_current_balance"
    assert today["okx_today_baseline_equity"] == pytest.approx(4999.58)
    assert today["okx_current_equity"] == pytest.approx(4997.95)
    assert today["okx_equity_pnl_source"] == "current_equity_minus_today_baseline"
    assert today["okx_equity_pnl"] == pytest.approx(-1.63)
    assert today["okx_cumulative_equity_pnl"] == pytest.approx(-1.63)
    assert today["total_pnl"] == pytest.approx(-1.63)


async def _async_value(value):
    return value


def test_dashboard_fallback_logger_redacts_exception_text(
    monkeypatch,
) -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    events: list[dict[str, Any]] = []

    class FakeLogger:
        def debug(self, event: str, **fields: Any) -> None:
            events.append({"event": event, **fields})

    monkeypatch.setattr(dashboard, "logger", FakeLogger())

    dashboard._log_dashboard_fallback(
        "unit fallback",
        RuntimeError(f"Authorization: Bearer {leaked_value} failed"),
        mode="paper",
    )

    assert events == [
        {
            "event": "unit fallback",
            "error": "Authorization: *** failed",
            "mode": "paper",
        }
    ]


def test_dashboard_execution_reason_uses_raw_llm_response_for_orm_decision() -> None:
    decision = SimpleNamespace(
        was_executed=False,
        action="long",
        confidence=0.9,
        feature_snapshot={"volume_ratio": 0.01},
        raw_llm_response={"entry_filters": {"min_entry_volume_ratio": 2.0}},
    )

    reason = dashboard._fallback_execution_reason_clean(decision)

    assert reason


def test_dashboard_exit_reason_prefers_raw_exchange_error_without_order() -> None:
    decision = SimpleNamespace(
        was_executed=False,
        action="close_long",
        confidence=0.9,
        reasoning="AI 建议平仓",
        feature_snapshot={},
        raw_llm_response={
            "execution_result": {
                "status": "rejected",
                "raw_response": {
                    "error": (
                        'okx {"code":"1","data":[{"ordId":"","sCode":"51028",'
                        '"sMsg":"Contract under delivery."}],"msg":"All operations failed"}'
                    )
                },
            }
        },
    )

    reason = dashboard._fallback_execution_reason_clean(decision)

    assert "OKX 51028" in reason
    assert "Contract under delivery" in reason
    assert "本地平仓委托记录" not in reason


def test_dashboard_exit_reason_recovers_from_generic_local_order_fallback() -> None:
    decision = SimpleNamespace(
        was_executed=False,
        action="close_short",
        confidence=0.9,
        reasoning="AI 建议平仓",
        feature_snapshot={},
        execution_reason=("这条平仓决策没有找到对应的本地平仓委托记录，因此系统未把它视为已执行。"),
        raw_llm_response={
            "untradable_exit_execution_error": {
                "reason": "okx {'sCode': '51028', 'sMsg': 'Contract under delivery.'}"
            }
        },
    )

    reason = dashboard._display_execution_reason(decision)

    assert "OKX 51028" in reason
    assert "本地平仓委托记录" not in reason


@pytest.mark.asyncio
async def test_dashboard_account_balance_is_mode_aware(monkeypatch) -> None:
    async def fake_okx_snapshot(mode: str) -> dict[str, Any]:
        assert mode == "live"
        return {"equity": 456.0, "total": 456.0, "free": 123.0}

    monkeypatch.setattr(dashboard, "_get_dashboard_okx_account_snapshot", fake_okx_snapshot)
    monkeypatch.setattr(dashboard.mode_manager, "_mode", TradingMode.LIVE)

    payload = await dashboard.get_account_balance()

    assert payload["mode"] == "live"
    assert payload["virtual_accounts"] == []
    assert payload["live_balance"] == 456.0
    assert payload["account_equity"] == 456.0
    assert payload["available_balance"] == 123.0
    assert payload["balance_source"] == "okx_authoritative"


def test_server_monitor_uses_probe_timeout_budget() -> None:
    service, ssh = _build_server_monitor_service()

    result = service.collect_sync()

    assert result["available"] is True
    assert ssh.closed is True
    assert ssh.exec_calls
    assert ssh.exec_calls[0]["timeout"] == SERVER_MONITOR_REMOTE_COMMAND_TIMEOUT_SECONDS
    assert ssh.exec_calls[0]["timeout"] > 12


def test_server_monitor_status_can_use_async_loaded_info_override() -> None:
    def unexpected_loader(_root: object) -> object:
        raise AssertionError("sync info loader should not run")

    service, ssh = _build_server_monitor_service(stdout='{"hostname": "model-host"}')
    service.info_loader = unexpected_loader

    result = service.get_status_sync(info_override=_FakeServerInfo())

    assert ssh.exec_calls
    assert result["available"] is True
    assert result["hostname"] == "model-host"


def test_server_monitor_defaults_to_model_server_info_loader() -> None:
    service = server_monitor_status.ServerMonitorStatusService(
        model_id_provider=lambda: "qwen3-32b-trade",
    )

    assert service.info_loader is server_monitor_status.load_model_server_info_for_monitor


def test_server_monitor_command_timeout_is_classified_and_redacted() -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    service, ssh = _build_server_monitor_service(
        raise_exc=TimeoutError(f"Authorization: Bearer {leaked_value} timed out"),
    )

    result = service.collect_sync()

    assert ssh.closed is True
    assert result["available"] is False
    assert result["status"] == "remote_command_timeout"
    assert leaked_value not in result["message"]
    assert "Authorization: ***" in result["message"]


def test_server_monitor_ssh_auth_failure_is_classified_without_leaking_secret() -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"

    def failing_connector(*_args: Any, **_kwargs: Any) -> Any:
        raise paramiko.AuthenticationException(f"Authentication failed {leaked_value}")

    service = server_monitor_status.ServerMonitorStatusService(
        model_id_provider=lambda: "qwen3-32b-trade",
        info_loader=lambda _root: _FakeServerInfo(),
        ssh_connector=failing_connector,
    )

    result = service.collect_sync()

    assert result["available"] is False
    assert result["remote_monitor_available"] is False
    assert result["status"] == "ssh_auth_failed"
    assert leaked_value not in result["message"]
    assert result["credential_source"]["host"] == "203.0.113.17"
    assert result["credential_source"]["password"] == "***"


def test_server_monitor_status_uses_short_cache() -> None:
    clock = [100.0]
    service, ssh = _build_server_monitor_service(
        stdout='{"hostname": "model-host"}',
        clock=lambda: clock[0],
    )

    first = service.get_status_sync()
    first["hostname"] = "mutated-by-caller"
    second = service.get_status_sync()

    assert ssh.exec_calls
    assert len(ssh.exec_calls) == 1
    assert second["hostname"] == "model-host"
    assert second["available"] is True
    assert second["cache"]["status"] == "fresh"


def test_server_monitor_status_cache_expires() -> None:
    clock = [200.0]
    service, ssh = _build_server_monitor_service(
        stdout='{"hostname": "model-host"}',
        clock=lambda: clock[0],
    )

    service.get_status_sync()
    clock[0] += service.cache_ttl_seconds + 0.1
    service.get_status_sync()

    assert len(ssh.exec_calls) == 2


def test_server_monitor_status_keeps_stale_cache_when_refresh_fails() -> None:
    clock = [250.0]
    service, good_ssh = _build_server_monitor_service(
        stdout='{"hostname": "model-host"}',
        clock=lambda: clock[0],
    )

    service.get_status_sync()
    clock[0] += service.cache_ttl_seconds + 0.1
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    bad_ssh = _FakeSSH()

    def failing_exec_remote_command(*args: Any, **kwargs: Any) -> SimpleNamespace:
        bad_ssh.exec_calls.append({"args": args, **kwargs})
        return SimpleNamespace(
            status=1,
            stdout="",
            stderr=f"Authorization: Bearer {leaked_value} failed",
        )

    service.ssh_connector = lambda *args, **kwargs: bad_ssh
    service.command_executor = failing_exec_remote_command

    result = service.get_status_sync()

    assert len(good_ssh.exec_calls) == 1
    assert len(bad_ssh.exec_calls) == 1
    assert result["available"] is True
    assert result["hostname"] == "model-host"
    assert result["cache"]["status"] == "stale_refresh_failed"
    assert result["refresh_error"]["status"] == "remote_command_failed"
    assert leaked_value not in result["refresh_error"]["message"]
    assert "Authorization: ***" in result["refresh_error"]["message"]


def test_server_monitor_status_returns_stale_cache_while_refreshing() -> None:
    clock = [300.0]
    service, ssh = _build_server_monitor_service(
        stdout='{"hostname": "model-host"}',
        clock=lambda: clock[0],
    )

    service.get_status_sync()
    clock[0] += service.cache_ttl_seconds + 0.1

    assert service._refresh_lock.acquire(blocking=False) is True
    try:
        result = service.get_status_sync()
    finally:
        service._refresh_lock.release()

    assert len(ssh.exec_calls) == 1
    assert result["hostname"] == "model-host"
    assert result["cache"]["status"] == "stale_refreshing"
    assert result["cache"]["age_seconds"] > service.cache_ttl_seconds


def test_server_monitor_status_returns_refreshing_without_cache() -> None:
    service, ssh = _build_server_monitor_service(stdout='{"hostname": "model-host"}')

    assert service._refresh_lock.acquire(blocking=False) is True
    try:
        result = service.get_status_sync()
    finally:
        service._refresh_lock.release()

    assert len(ssh.exec_calls) == 0
    assert result["available"] is False
    assert result["status"] == "server_monitor_refreshing"
    assert result["cache"]["status"] == "initial_refreshing"


def test_server_monitor_invalid_json_payload_is_not_reported_as_ssh_failed() -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    service, ssh = _build_server_monitor_service(
        stdout=f"boot log Authorization: Bearer {leaked_value}",
    )

    result = service.collect_sync()

    assert ssh.closed is True
    assert result["available"] is False
    assert result["status"] == "remote_payload_invalid"
    assert leaked_value not in result["message"]
    assert "Authorization: ***" in result["message"]


def test_server_monitor_non_object_json_payload_is_classified_without_exception() -> None:
    service, _ssh = _build_server_monitor_service(stdout="[]")

    result = service.collect_sync()

    assert result["available"] is False
    assert result["status"] == "remote_payload_invalid"
    assert "non-object JSON payload" in result["message"]


def test_server_monitor_remote_command_failure_is_redacted() -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    service, _ssh = _build_server_monitor_service(
        status=1,
        stdout='{"token": "stdout-secret-value"}',
        stderr=f"password=stderr-secret Authorization: Bearer {leaked_value}",
    )

    result = service.collect_sync()

    assert result["available"] is False
    assert result["status"] == "remote_command_failed"
    assert leaked_value not in result["message"]
    assert "stderr-secret" not in result["message"]
    assert "Authorization: ***" in result["message"]
    assert "password=***" in result["message"]


async def test_collect_platform_runtime_status_probes_real_local_tool_endpoints(
    monkeypatch,
) -> None:
    requests: list[tuple[str, str, str]] = []

    fake_settings = SimpleNamespace(
        get_fixed_ai_models=lambda include_empty=False: [
            {
                "name": "main",
                "label": "主模型",
                "api_base": "http://llm.test/v1",
                "api_key": "hidden-llm-key",
                "model": "qwen3-14b-trade",
                "enabled": True,
            }
        ],
        local_ai_tools_api_base="http://local-ai.test",
        local_ai_tools_api_key="hidden-tools-key",
        ai_api_key="",
    )
    monkeypatch.setattr(server_monitor_status, "settings", fake_settings)

    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def request(
            self,
            method: str,
            url: str,
            headers: dict[str, str] | None = None,
            json: dict[str, Any] | None = None,
        ) -> httpx.Response:
            requests.append((method, url, str((headers or {}).get("Authorization") or "")))
            request = httpx.Request(method, url)
            if url == "http://llm.test/v1/models":
                return httpx.Response(
                    200,
                    json={"data": [{"id": "qwen3-14b-trade"}]},
                    request=request,
                )
            if url == "http://local-ai.test/health":
                return httpx.Response(500, json={"detail": "booting"}, request=request)
            if url == "http://local-ai.test/models/status":
                return httpx.Response(503, json={"available": False}, request=request)
            if url.endswith("/profit/predict"):
                return httpx.Response(200, json={"available": True}, request=request)
            return httpx.Response(500, json={"available": False}, request=request)

    monkeypatch.setattr(server_monitor_status.httpx, "AsyncClient", FakeAsyncClient)

    result = await server_monitor_status.collect_platform_runtime_status()

    assert result["ai_models"][0]["available"] is True
    tools = result["local_ai_tools"]
    assert tools["available"] is True
    assert tools["health"]["ok"] is False
    assert tools["status"]["ok"] is False
    assert tools["model_bundle_available"] is False
    assert tools["child_endpoints"]["profit_prediction"]["available"] is True
    assert tools["child_endpoints"]["exit_advice"]["available"] is False
    assert tools["expected_platform_api_base"] == "http://127.0.0.1:18001"
    assert tools["tunnel_contract"]["status"] == "external_or_dev_endpoint"
    assert ("POST", "http://local-ai.test/profit/predict", "Bearer hidden-tools-key") in requests


async def test_collect_platform_runtime_status_uses_env_local_tools_key_when_settings_empty(
    monkeypatch,
) -> None:
    requests: list[tuple[str, str, str]] = []

    fake_settings = SimpleNamespace(
        get_fixed_ai_models=lambda include_empty=False: [],
        local_ai_tools_api_base="http://local-ai.test",
        local_ai_tools_api_key="",
        ai_api_key="",
    )
    monkeypatch.setattr(server_monitor_status, "settings", fake_settings)
    monkeypatch.setenv("LOCAL_AI_TOOLS_API_KEY", "env-tools-key")

    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def request(
            self,
            method: str,
            url: str,
            headers: dict[str, str] | None = None,
            json: dict[str, Any] | None = None,
        ) -> httpx.Response:
            requests.append((method, url, str((headers or {}).get("Authorization") or "")))
            return httpx.Response(200, json={"available": True}, request=httpx.Request(method, url))

    monkeypatch.setattr(server_monitor_status.httpx, "AsyncClient", FakeAsyncClient)

    result = await server_monitor_status.collect_platform_runtime_status()

    assert result["local_ai_tools"]["available"] is True
    assert ("GET", "http://local-ai.test/health", "Bearer env-tools-key") in requests
    assert ("GET", "http://local-ai.test/models/status", "Bearer env-tools-key") in requests
    assert ("POST", "http://local-ai.test/profit/predict", "Bearer env-tools-key") in requests


async def test_collect_platform_runtime_status_defaults_to_phase3_tunnel_when_unconfigured(
    monkeypatch,
) -> None:
    requests: list[tuple[str, str]] = []

    fake_settings = SimpleNamespace(
        get_fixed_ai_models=lambda include_empty=False: [],
        local_ai_tools_api_base="",
        local_ai_tools_api_key="",
        ai_api_key="",
    )
    monkeypatch.setattr(server_monitor_status, "settings", fake_settings)

    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def request(
            self,
            method: str,
            url: str,
            headers: dict[str, str] | None = None,
            json: dict[str, Any] | None = None,
        ) -> httpx.Response:
            requests.append((method, url))
            request = httpx.Request(method, url)
            if url == "http://127.0.0.1:18001/health":
                return httpx.Response(
                    200,
                    json={"ok": True, "service": "phase3_quant_api", "root": "/data/BB"},
                    request=request,
                )
            if url == "http://127.0.0.1:18001/models/status":
                return httpx.Response(200, json={"available": True}, request=request)
            return httpx.Response(200, json={"available": True}, request=request)

    monkeypatch.setattr(server_monitor_status.httpx, "AsyncClient", FakeAsyncClient)

    result = await server_monitor_status.collect_platform_runtime_status()

    tools = result["local_ai_tools"]
    assert tools["available"] is True
    assert tools["configured"] is False
    assert tools["using_default_phase3_tunnel"] is True
    assert tools["api_base"] == "http://127.0.0.1:18001"
    assert tools["configured_api_base"] == ""
    assert ("GET", "http://127.0.0.1:18001/health") in requests


async def test_collect_platform_runtime_status_flags_wrong_local_ai_loopback_port(
    monkeypatch,
) -> None:
    fake_settings = SimpleNamespace(
        get_fixed_ai_models=lambda include_empty=False: [],
        local_ai_tools_api_base="http://127.0.0.1:8001",
        local_ai_tools_api_key="hidden-tools-key",
        ai_api_key="",
    )
    monkeypatch.setattr(server_monitor_status, "settings", fake_settings)

    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def request(
            self,
            method: str,
            url: str,
            headers: dict[str, str] | None = None,
            json: dict[str, Any] | None = None,
        ) -> httpx.Response:
            request = httpx.Request(method, url)
            return httpx.Response(200, json={"available": True}, request=request)

    monkeypatch.setattr(server_monitor_status.httpx, "AsyncClient", FakeAsyncClient)

    result = await server_monitor_status.collect_platform_runtime_status()

    tools = result["local_ai_tools"]
    assert tools["available"] is False
    assert tools["tunnel_contract"]["status"] == "wrong_loopback_port"
    assert tools["tunnel_contract"]["expected"] == "http://127.0.0.1:18001"
    assert "18001" in tools["config_issue"]


async def test_symbols_available_error_response_is_redacted(
    monkeypatch,
) -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"

    async def failing_sdk_symbols() -> list[str]:
        raise RuntimeError("SDK unavailable")

    class FakeDataService:
        async def get_available_symbols(self) -> list[str]:
            raise RuntimeError(f"Authorization: Bearer {leaked_value} failed")

    monkeypatch.setitem(
        sys.modules,
        "data_feed.okx_sdk_client",
        SimpleNamespace(get_available_symbols=failing_sdk_symbols),
    )
    monkeypatch.setattr(dashboard, "_data_service", FakeDataService())

    result = await symbols.get_available_symbols()

    assert result["count"] == 0
    assert result["symbols"] == []
    assert leaked_value not in result["error"]
    assert result["error"] == "Authorization: *** failed"
