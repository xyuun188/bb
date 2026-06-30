from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

import db.session as session_module
from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.account import ExecutionEquitySnapshot, VirtualAccount
from models.dashboard_auth import DashboardUser
from models.decision import AIDecision
from models.learning import ShadowBacktest, StrategyLearningEvent, TradeReflection
from models.secure_config import SecureSetting, SecureSettingAudit
from models.trade import Order, Position
from scripts import phase3_cold_start_reset as reset_script


async def _reset_test_db(tmp_path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    await close_db()
    session_module._engine = None
    session_module._sessionmaker = None
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / name).as_posix()}",
    )
    await init_db()


async def _seed_rows() -> None:
    now = datetime.now(UTC)
    async with get_session_ctx() as session:
        session.add_all(
            [
                DashboardUser(username="admin", password_hash="hash", role="admin"),
                SecureSetting(key="okx.paper.key", ciphertext="secret", nonce="n", aad="a"),
                SecureSettingAudit(key="okx.paper.key", action="update", actor="test"),
                VirtualAccount(
                    model_name="ensemble_trader",
                    initial_balance=1000.0,
                    current_balance=880.0,
                    realized_pnl=-120.0,
                    unrealized_pnl=5.0,
                    total_trades=3,
                    winning_trades=1,
                ),
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    side="buy",
                    order_type="market",
                    quantity=1,
                    price=100,
                    status="filled",
                    exchange_order_id="paper-order",
                ),
                Order(
                    model_name="ensemble_trader",
                    execution_mode="live",
                    symbol="ETH/USDT",
                    side="buy",
                    order_type="market",
                    quantity=1,
                    price=100,
                    status="filled",
                    exchange_order_id="live-order",
                ),
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    side="long",
                    quantity=1,
                    entry_price=100,
                    is_open=False,
                    closed_at=now,
                ),
                Position(
                    model_name="ensemble_trader",
                    execution_mode="live",
                    symbol="ETH/USDT",
                    side="long",
                    quantity=1,
                    entry_price=100,
                    is_open=False,
                    closed_at=now,
                ),
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="BTC/USDT",
                    action="long",
                    confidence=0.8,
                    is_paper=True,
                ),
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="ETH/USDT",
                    action="long",
                    confidence=0.8,
                    is_paper=False,
                ),
                TradeReflection(
                    position_id=1,
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    side="long",
                ),
                ShadowBacktest(
                    decision_id=1,
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    due_at=now + timedelta(minutes=10),
                ),
                StrategyLearningEvent(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    event_type="execution",
                ),
                ExecutionEquitySnapshot(
                    mode="paper",
                    model_name="ensemble_trader",
                    snapshot_date="2026-06-26",
                    snapshot_at=now,
                    equity=880,
                ),
            ]
        )


async def _count(model) -> int:
    async with get_session_ctx() as session:
        result = await session.execute(select(model))
        return len(result.scalars().all())


@pytest.mark.asyncio
async def test_phase3_cold_start_dry_run_does_not_mutate(tmp_path, monkeypatch) -> None:
    await _reset_test_db(tmp_path, monkeypatch, "cold-start-dry-run.db")
    await _seed_rows()
    monkeypatch.setattr(
        reset_script,
        "check_trading_service_stopped",
        lambda _name=reset_script.TRADING_SERVICE_NAME: {"ok": True, "status": "inactive"},
    )
    monkeypatch.setattr(reset_script, "check_okx_paper_empty", lambda: _async_okx_empty())

    report = await reset_script.run(
        apply=False,
        confirm="",
        backup_dir=tmp_path / "backups",
    )

    assert report["apply"] is False
    assert report["plan"]["delete_counts"]["orders"] == 1
    assert await _count(Order) == 2
    assert await _count(DashboardUser) == 1
    assert await _count(SecureSetting) == 1
    await close_db()


@pytest.mark.asyncio
async def test_phase3_cold_start_apply_requires_confirmation(tmp_path, monkeypatch) -> None:
    await _reset_test_db(tmp_path, monkeypatch, "cold-start-confirm.db")
    await _seed_rows()
    monkeypatch.setattr(
        reset_script,
        "check_trading_service_stopped",
        lambda _name=reset_script.TRADING_SERVICE_NAME: {"ok": True, "status": "inactive"},
    )
    monkeypatch.setattr(reset_script, "check_okx_paper_empty", lambda: _async_okx_empty())

    with pytest.raises(SystemExit):
        await reset_script.run(apply=True, confirm="", backup_dir=tmp_path / "backups")
    await close_db()


@pytest.mark.asyncio
async def test_phase3_cold_start_apply_clears_paper_and_preserves_secrets(
    tmp_path,
    monkeypatch,
) -> None:
    await _reset_test_db(tmp_path, monkeypatch, "cold-start-apply.db")
    await _seed_rows()
    monkeypatch.setattr(
        reset_script,
        "check_trading_service_stopped",
        lambda _name=reset_script.TRADING_SERVICE_NAME: {"ok": True, "status": "inactive"},
    )
    monkeypatch.setattr(reset_script, "check_okx_paper_empty", lambda: _async_okx_empty())

    report = await reset_script.run(
        apply=True,
        confirm=reset_script.CONFIRMATION,
        backup_dir=tmp_path / "backups",
        marker_path=tmp_path / "phase3_cold_start_reset_marker.json",
    )

    assert report["result"]["deleted"]["orders"] == 1
    assert report["result"]["deleted"]["positions"] == 1
    assert report["post_plan"]["delete_counts"]["orders"] == 0
    assert await _count(DashboardUser) == 1
    assert await _count(SecureSetting) == 1
    assert await _count(SecureSettingAudit) == 1

    async with get_session_ctx() as session:
        orders = list((await session.execute(select(Order))).scalars().all())
        decisions = list((await session.execute(select(AIDecision))).scalars().all())
        account = (await session.execute(select(VirtualAccount))).scalar_one()
    assert [(row.execution_mode, row.exchange_order_id) for row in orders] == [
        ("live", "live-order")
    ]
    assert [(row.symbol, row.is_paper) for row in decisions] == [("ETH/USDT", False)]
    assert account.current_balance == pytest.approx(account.initial_balance)
    assert account.realized_pnl == pytest.approx(0.0)
    assert account.total_trades == 0
    assert (tmp_path / "backups").exists()
    marker_path = tmp_path / "phase3_cold_start_reset_marker.json"
    assert marker_path.exists()
    marker = marker_path.read_text(encoding="utf-8")
    assert reset_script.CONFIRMATION in marker
    assert "okx_authoritative_sync_ignores_pre_reset_fills" in marker
    await close_db()


async def _async_okx_empty() -> dict:
    return {
        "ok": True,
        "mode": "paper",
        "open_position_count": 0,
        "open_order_count": 0,
    }
