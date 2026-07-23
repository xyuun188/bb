from datetime import UTC, datetime

import pytest
from sqlalchemy import select

import db.session as session_module
from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.account import ExecutionEquitySnapshot, OkxAccountBill
from models.decision import AIDecision
from models.learning import ShadowBacktest, StrategyLearningEvent, TradeReflection
from models.trade import OkxPositionHistory, Order, Position
from scripts import reset_training_derived_state as reset_script


async def _reset_db(tmp_path, monkeypatch, name: str) -> None:
    await close_db()
    session_module._engine = None
    session_module._sessionmaker = None
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{(tmp_path / name).as_posix()}")
    await init_db()


async def _count(model) -> int:
    async with get_session_ctx() as session:
        return len((await session.execute(select(model))).scalars().all())


@pytest.mark.asyncio
async def test_reset_only_deletes_derived_state_and_starts_new_epoch(tmp_path, monkeypatch) -> None:
    await _reset_db(tmp_path, monkeypatch, "derived-reset.db")
    now = datetime.now(UTC)
    async with get_session_ctx() as session:
        session.add_all(
            [
                ShadowBacktest(
                    model_name="model",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    due_at=now,
                ),
                TradeReflection(
                    position_id=1,
                    model_name="model",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    side="long",
                ),
                ExecutionEquitySnapshot(
                    mode="paper",
                    model_name="model",
                    snapshot_date="2026-07-24",
                    snapshot_at=now,
                    equity=1000,
                ),
                Order(
                    model_name="model",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    side="buy",
                    order_type="market",
                    quantity=1,
                    status="filled",
                    exchange_order_id="okx-order-1",
                ),
                Position(
                    model_name="model",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    side="long",
                    quantity=1,
                    entry_price=100,
                    is_open=False,
                    realized_pnl=-2,
                    funding_fee=-0.1,
                ),
                AIDecision(
                    model_name="model",
                    symbol="BTC/USDT",
                    action="long",
                    confidence=0.8,
                    is_paper=True,
                ),
                StrategyLearningEvent(
                    model_name="model",
                    execution_mode="paper",
                    event_type="execution",
                ),
                OkxPositionHistory(
                    mode="paper",
                    row_identity="history-1",
                    inst_id="BTC-USDT-SWAP",
                    symbol="BTC/USDT",
                    realized_pnl=-2,
                    fee=-0.1,
                    funding_fee=-0.1,
                ),
                OkxAccountBill(
                    mode="paper",
                    bill_id="bill-1",
                    bill_ts=now,
                    pnl=-2,
                    fee=-0.1,
                    funding_fee=-0.1,
                ),
            ]
        )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "ml_signal").mkdir()
    (data_dir / "ml_signal" / "old.joblib").write_bytes(b"old")
    (data_dir / "model_training_scheduler_state.json").write_text("{}", encoding="utf-8")
    (data_dir / "system_audit_latest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(reset_script, "check_services_stopped", lambda: {"ok": True})

    report = await reset_script.run(
        apply=True,
        confirm=reset_script.CONFIRMATION,
        data_dir=data_dir,
        skip_service_gate=False,
    )

    assert report["result"]["deleted_tables"]["shadow_backtests"] == 1
    assert await _count(ShadowBacktest) == 0
    assert await _count(TradeReflection) == 0
    assert await _count(ExecutionEquitySnapshot) == 0
    assert await _count(Order) == 1
    assert await _count(Position) == 1
    assert await _count(AIDecision) == 1
    assert await _count(StrategyLearningEvent) == 1
    assert await _count(OkxPositionHistory) == 1
    assert await _count(OkxAccountBill) == 1
    assert not (data_dir / "ml_signal").exists()
    assert (data_dir / "training_epoch.json").exists()
    assert (data_dir / "training_reset_manifest.json").exists()
    assert report["result"]["preserved_table_counts_match"] is True
    assert report["post_plan"]["preserved_table_counts"] == report["plan"][
        "preserved_table_counts"
    ]
    await close_db()


@pytest.mark.asyncio
async def test_reset_permission_preflight_runs_before_database_delete(
    tmp_path,
    monkeypatch,
) -> None:
    await _reset_db(tmp_path, monkeypatch, "derived-reset-permission.db")
    async with get_session_ctx() as session:
        session.add(
            ShadowBacktest(
                model_name="model",
                execution_mode="paper",
                symbol="BTC/USDT",
                due_at=datetime.now(UTC),
            )
        )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    blocked = data_dir / "model_artifacts"
    blocked.mkdir()
    monkeypatch.setattr(reset_script, "check_services_stopped", lambda: {"ok": True})
    monkeypatch.setattr(
        reset_script,
        "_deletion_blocker",
        lambda path: "directory_not_writable:test" if path == blocked else None,
    )

    with pytest.raises(RuntimeError, match="permission preflight"):
        await reset_script.run(
            apply=True,
            confirm=reset_script.CONFIRMATION,
            data_dir=data_dir,
        )

    assert await _count(ShadowBacktest) == 1
    await close_db()
