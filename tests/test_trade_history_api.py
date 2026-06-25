from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from config.settings import settings
from db.repositories.trade_repo import TradeRepository
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from web_dashboard.api.dashboard import get_positions as get_dashboard_positions
from web_dashboard.api.trades import (
    _execution_status_label,
    _readable_execution_reason,
    _repair_position_reason_hold_hours,
    _translate_execution_text,
    get_trade_detail,
    get_trades,
)


@pytest.mark.asyncio
async def test_trade_history_uses_matched_position_leverage_for_close_order(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trade-history.db').as_posix()}",
    )
    await init_db()
    filled_at = datetime(2026, 6, 23, 6, 55, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="WLFI/USDT",
                action="close_short",
                confidence=0.88,
                reasoning="fast exit",
                position_size_pct=1.0,
                suggested_leverage=1.0,
                raw_llm_response={},
                is_paper=True,
                was_executed=True,
                created_at=filled_at,
            )
            session.add(decision)
            await session.flush()
            repo = TradeRepository(session)
            order = await repo.create_order(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "WLFI/USDT",
                    "side": "buy",
                    "order_type": "market",
                    "quantity": 152.0,
                    "price": 0.1226,
                    "status": "filled",
                    "fee": 0.01,
                    "decision_id": decision.id,
                    "exchange_order_id": "close-order-1",
                    "filled_at": filled_at,
                    "created_at": filled_at,
                }
            )
            await repo.open_position(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "WLFI/USDT",
                    "side": "short",
                    "quantity": 152.0,
                    "entry_price": 0.1223,
                    "current_price": 0.1226,
                    "leverage": 3.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": -0.0556,
                    "is_open": False,
                    "closed_at": filled_at + timedelta(seconds=1),
                    "created_at": filled_at - timedelta(minutes=14),
                }
            )

        trades = await get_trades(model_name=None, symbol=None, mode="paper", limit=10, page=1)
        detail = await get_trade_detail(order.id)
    finally:
        await close_db()

    assert trades["trades"][0]["action"] == "close_short"
    assert trades["trades"][0]["leverage"] == pytest.approx(3.0)
    assert trades["trades"][0]["actual_leverage"] == pytest.approx(3.0)
    assert trades["trades"][0]["ai_suggested_leverage"] == pytest.approx(1.0)
    assert trades["trades"][0]["execution_source"] == "system"
    assert trades["trades"][0]["execution_source_label"] == "系统执行"
    assert detail["matched_positions"][0]["leverage"] == pytest.approx(3.0)
    assert detail["execution_source"] == "system"
    assert detail["execution_source_label"] == "系统执行"


@pytest.mark.asyncio
async def test_dashboard_position_history_marks_same_open_group_slices_as_partial(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-positions.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 6, 24, 5, 20, tzinfo=UTC)
    closed_at = datetime(2026, 6, 24, 5, 26, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            full_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="USAR/USDT",
                action="close_long",
                confidence=1.0,
                reasoning="exchange sync",
                position_size_pct=1.0,
                suggested_leverage=6.0,
                raw_llm_response={},
                is_paper=True,
                was_executed=True,
                created_at=closed_at,
            )
            session.add(full_decision)
            await session.flush()
            for offset, quantity, price in [
                (0, 10.0, 3.85),
                (20, 6.0, 4.26),
            ]:
                await repo.open_position(
                    {
                        "model_name": "ensemble_trader",
                        "execution_mode": "paper",
                        "symbol": "USAR/USDT",
                        "side": "long",
                        "quantity": quantity,
                        "entry_price": 2.31,
                        "current_price": price,
                        "leverage": 6.0,
                        "unrealized_pnl": 0.0,
                        "realized_pnl": (price - 2.31) * quantity,
                        "is_open": False,
                        "closed_at": closed_at + timedelta(seconds=offset),
                        "created_at": opened_at,
                    }
                )
                await repo.create_order(
                    {
                        "model_name": "ensemble_trader",
                        "execution_mode": "paper",
                        "symbol": "USAR/USDT",
                        "side": "sell",
                        "order_type": "market",
                        "quantity": quantity,
                        "price": price,
                        "status": "filled",
                        "fee": 0.0,
                        "decision_id": full_decision.id,
                        "exchange_order_id": f"usar-close-{int(quantity)}",
                        "filled_at": closed_at + timedelta(seconds=offset),
                        "created_at": closed_at + timedelta(seconds=offset),
                    }
                )

        payload = await get_dashboard_positions(mode="paper", closed_only=True)
    finally:
        await close_db()

    assert payload["positions"]
    assert {item["close_status"] for item in payload["positions"]} == {"partial"}


@pytest.mark.asyncio
async def test_dashboard_position_history_uses_synced_position_realized_pnl(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-realized-pnl.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 6, 24, 10, 38, tzinfo=UTC)
    closed_at = datetime(2026, 6, 24, 10, 47, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="WLFI/USDT",
                action="close_short",
                confidence=1.0,
                reasoning="exchange sync",
                position_size_pct=1.0,
                suggested_leverage=4.0,
                raw_llm_response={},
                is_paper=True,
                was_executed=True,
                created_at=closed_at,
            )
            session.add(decision)
            await session.flush()
            await repo.open_position(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "WLFI/USDT",
                    "side": "short",
                    "quantity": 1964.0,
                    "entry_price": 0.0818,
                    "current_price": 0.08149516293279023,
                    "leverage": 4.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.43834414999998383,
                    "is_open": False,
                    "closed_at": closed_at,
                    "created_at": opened_at,
                }
            )
            await repo.create_order(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "WLFI/USDT",
                    "side": "buy",
                    "order_type": "market",
                    "quantity": 1964.0,
                    "price": 0.08149516293279023,
                    "status": "filled",
                    "fee": 0.08002825,
                    "decision_id": decision.id,
                    "exchange_order_id": "wlfi-close",
                    "filled_at": closed_at,
                    "created_at": closed_at,
                }
            )

        payload = await get_dashboard_positions(mode="paper", closed_only=True)
    finally:
        await close_db()

    assert payload["positions"][0]["realized_pnl"] == pytest.approx(0.43834414999998383)
    assert payload["positions"][0]["pnl_source"] == "position_realized_pnl"


@pytest.mark.asyncio
async def test_trade_history_only_marks_explicit_exchange_sync_as_okx_execution(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trade-source.db').as_posix()}",
    )
    await init_db()
    filled_at = datetime(2026, 6, 24, 11, 10, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            await repo.open_position(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "MASK/USDT",
                    "side": "long",
                    "quantity": 47.0,
                    "entry_price": 1.68,
                    "current_price": 1.67,
                    "leverage": 3.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": -0.47,
                    "is_open": False,
                    "closed_at": filled_at,
                    "created_at": filled_at - timedelta(minutes=30),
                }
            )
            system_order = await repo.create_order(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "MASK/USDT",
                    "side": "sell",
                    "order_type": "market",
                    "quantity": 47.0,
                    "price": 1.67,
                    "status": "filled",
                    "fee": 0.02,
                    "decision_id": None,
                    "exchange_order_id": "367-system-okx-fill",
                    "filled_at": filled_at,
                    "created_at": filled_at,
                }
            )
            sync_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="H/USDT",
                action="close_short",
                confidence=1.0,
                reasoning="exchange reconcile",
                position_size_pct=1.0,
                suggested_leverage=4.0,
                raw_llm_response={
                    "system_sync": True,
                    "source": "okx_position_reconcile",
                    "close_fill": {"order_id": "367-okx-sync", "price": 0.08, "quantity": 100},
                },
                feature_snapshot={"source": "okx_position_reconcile"},
                is_paper=True,
                was_executed=True,
                created_at=filled_at + timedelta(seconds=1),
            )
            session.add(sync_decision)
            await session.flush()
            await repo.create_order(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "H/USDT",
                    "side": "buy",
                    "order_type": "market",
                    "quantity": 100.0,
                    "price": 0.08,
                    "status": "filled",
                    "fee": 0.02,
                    "decision_id": sync_decision.id,
                    "exchange_order_id": "367-okx-sync",
                    "filled_at": filled_at + timedelta(seconds=1),
                    "created_at": filled_at + timedelta(seconds=1),
                }
            )
            await repo.open_position(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "LINK/USDT",
                    "side": "short",
                    "quantity": 2.29,
                    "entry_price": 6.95,
                    "current_price": 6.902,
                    "leverage": 3.0,
                    "take_profit_price": 6.9,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.10,
                    "is_open": False,
                    "closed_at": filled_at + timedelta(seconds=2),
                    "created_at": filled_at - timedelta(minutes=20),
                }
            )
            protected_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="LINK/USDT",
                action="close_short",
                confidence=1.0,
                reasoning="system protection reconcile",
                position_size_pct=1.0,
                suggested_leverage=3.0,
                raw_llm_response={
                    "system_sync": True,
                    "source": "okx_position_reconcile",
                    "close_fill": {
                        "order_id": "protected-close",
                        "price": 6.902,
                        "quantity": 2.29,
                    },
                },
                feature_snapshot={"source": "okx_position_reconcile"},
                is_paper=True,
                was_executed=True,
                created_at=filled_at + timedelta(seconds=2),
            )
            session.add(protected_decision)
            await session.flush()
            protected_order = await repo.create_order(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "LINK/USDT",
                    "side": "buy",
                    "order_type": "market",
                    "quantity": 2.29,
                    "price": 6.902,
                    "status": "filled",
                    "fee": 0.01,
                    "decision_id": protected_decision.id,
                    "exchange_order_id": "protected-close",
                    "filled_at": filled_at + timedelta(seconds=2),
                    "created_at": filled_at + timedelta(seconds=2),
                }
            )

        trades = await get_trades(model_name=None, symbol=None, mode="paper", limit=10, page=1)
        by_id = {item["id"]: item for item in trades["trades"]}
        detail = await get_trade_detail(system_order.id)
        protected_detail = await get_trade_detail(protected_order.id)
    finally:
        await close_db()

    assert by_id[system_order.id]["execution_source"] == "system"
    assert by_id[system_order.id]["execution_source_label"] == "系统执行"
    sync_rows = [item for item in trades["trades"] if item["exchange_order_id"] == "367-okx-sync"]
    assert sync_rows and sync_rows[0]["execution_source"] == "okx"
    assert sync_rows[0]["execution_source_label"] == "OKX同步"
    protected_rows = [
        item for item in trades["trades"] if item["exchange_order_id"] == "protected-close"
    ]
    assert protected_rows and protected_rows[0]["execution_source"] == "system"
    assert protected_rows[0]["execution_source_label"] == "系统保护单"
    assert "系统保护单" in protected_rows[0]["reason"]
    assert protected_detail["execution_source"] == "system"
    assert protected_detail["execution_source_label"] == "系统保护单"
    assert detail["execution_source"] == "system"
    assert detail["execution_source_label"] == "系统执行"
    assert "OKX 平仓成交已同步" not in detail["reason"]


def test_repair_position_reason_hold_hours_replaces_stale_zero_value() -> None:
    reason = "策略纪律触发低质量旧仓释放：hard_loss_pressure；质量分层=watch，质量分=56.0，持仓小时=0.0。"

    repaired = _repair_position_reason_hold_hours(reason, 68.5883 * 60)

    assert "持仓小时=68.5883" in repaired
    assert "持仓小时=0.0" not in repaired


def test_repair_position_reason_hold_hours_keeps_valid_existing_value() -> None:
    reason = "策略纪律触发低质量旧仓释放：loss_watch；持仓小时=70.1184。"

    repaired = _repair_position_reason_hold_hours(reason, 68.0 * 60)

    assert repaired == reason


def test_trade_history_translates_okx_50001_as_temporary_exchange_failure() -> None:
    reason = (
        'Max retries exceeded: okx {"code":"50001","data":[],'
        '"msg":"Service temporarily unavailable. Please try again later."}'
    )

    translated = _translate_execution_text(reason)

    assert "交易所服务临时不可用" in translated
    assert _execution_status_label("rejected", translated) == "交易所临时不可用"


def test_trade_detail_does_not_use_numeric_order_id_as_reason() -> None:
    reason = _readable_execution_reason(
        execution_reason="",
        reasoning="策略纪律触发低质量旧仓释放：signal_reversal。",
        exchange_order_id="3670054929945042944",
        status="filled",
    )

    assert "3670054929945042944" not in reason
    assert "策略纪律触发" in reason


def test_trade_detail_numeric_only_reason_falls_back_to_readable_success() -> None:
    reason = _readable_execution_reason(
        execution_reason="3670054929945042944",
        reasoning="",
        exchange_order_id="3670054929945042944",
        status="filled",
    )

    assert "3670054929945042944" not in reason
    assert "订单已成交" in reason
