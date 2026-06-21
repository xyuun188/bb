from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from db.session import close_db, get_session_ctx, init_db
from config.settings import settings
from models.decision import AIDecision
from models.trade import Order, Position
from services.order_position_reconciliation import (
    apply_missing_closed_position_plan,
    plan_missing_closed_position,
)


@pytest.mark.asyncio
async def test_reconciles_missing_closed_short_position_from_order_pair(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trading.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime(2026, 6, 21, 10, 43, tzinfo=UTC)
        closed_at = opened_at + timedelta(minutes=4)
        async with get_session_ctx() as session:
            entry_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="PROS/USDT",
                action="short",
                confidence=0.88,
                reasoning="entry",
                position_size_pct=0.02,
                suggested_leverage=3.0,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
                is_paper=True,
                was_executed=True,
            )
            close_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="PROS/USDT",
                action="close_short",
                confidence=0.92,
                reasoning="close",
                position_size_pct=1.0,
                suggested_leverage=1.0,
                is_paper=True,
                was_executed=True,
            )
            session.add_all([entry_decision, close_decision])
            await session.flush()
            entry_order = Order(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="PROS/USDT:USDT",
                side="sell",
                order_type="market",
                quantity=9.0,
                price=0.7316,
                status="filled",
                fee=0.001,
                decision_id=entry_decision.id,
                exchange_order_id="entry-1",
                filled_at=opened_at,
            )
            close_order = Order(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="PROS/USDT:USDT",
                side="buy",
                order_type="market",
                quantity=9.0,
                price=0.3879,
                status="filled",
                fee=0.002,
                decision_id=close_decision.id,
                exchange_order_id="close-1",
                filled_at=closed_at,
            )
            session.add_all([entry_order, close_order])
            await session.flush()
            close_id = close_order.id

        async with get_session_ctx() as session:
            close_order = await session.get(Order, close_id)
            plan = await plan_missing_closed_position(session, close_order)
            assert plan is not None
            assert plan.symbol == "PROS/USDT"
            assert plan.side == "short"
            assert plan.quantity == pytest.approx(9.0)
            assert plan.entry_price == pytest.approx(0.7316)
            assert plan.exit_price == pytest.approx(0.3879)
            assert plan.realized_pnl == pytest.approx((0.7316 - 0.3879) * 9.0 - 0.003)
            position = await apply_missing_closed_position_plan(session, plan)
            assert position.is_open is False
            assert position.realized_pnl == pytest.approx(3.0903)

        async with get_session_ctx() as session:
            close_order = await session.get(Order, close_id)
            duplicate_plan = await plan_missing_closed_position(session, close_order)
            result = await session.execute(select(Position))
            positions = list(result.scalars().all())

        assert duplicate_plan is None
        assert len(positions) == 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_existing_close_row_is_not_duplicated_when_entry_price_was_legacy_wrong(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trading.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime(2026, 6, 21, 10, 43, tzinfo=UTC)
        closed_at = opened_at + timedelta(minutes=4)
        async with get_session_ctx() as session:
            close_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="PROS/USDT",
                action="close_short",
                confidence=0.92,
                reasoning="close",
                is_paper=True,
                was_executed=True,
            )
            session.add(close_decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="PROS/USDT:USDT",
                        side="sell",
                        order_type="market",
                        quantity=9.0,
                        price=0.7316,
                        status="filled",
                        fee=0.001,
                        exchange_order_id="entry-1",
                        filled_at=opened_at,
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="PROS/USDT:USDT",
                        side="buy",
                        order_type="market",
                        quantity=9.0,
                        price=0.3879,
                        status="filled",
                        fee=0.002,
                        decision_id=close_decision.id,
                        exchange_order_id="close-1",
                        filled_at=closed_at,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="PROS/USDT",
                        side="short",
                        quantity=9.0,
                        entry_price=0.6546,
                        current_price=0.3879,
                        leverage=3.0,
                        unrealized_pnl=0.0,
                        realized_pnl=3.08,
                        is_open=False,
                        closed_at=closed_at,
                        created_at=opened_at,
                    ),
                ]
            )
            await session.flush()

        async with get_session_ctx() as session:
            order_result = await session.execute(select(Order).where(Order.side == "buy"))
            close_order = order_result.scalar_one()
            plan = await plan_missing_closed_position(session, close_order)

        assert plan is None
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_existing_split_close_rows_are_not_duplicated(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trading.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime(2026, 6, 21, 12, 1, tzinfo=UTC)
        closed_at = opened_at + timedelta(minutes=3)
        async with get_session_ctx() as session:
            close_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="MET/USDT",
                action="close_short",
                confidence=0.92,
                reasoning="close",
                is_paper=True,
                was_executed=True,
            )
            session.add(close_decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="MET/USDT:USDT",
                        side="sell",
                        order_type="market",
                        quantity=90.0,
                        price=0.02,
                        status="filled",
                        fee=0.001,
                        exchange_order_id="entry-met-1",
                        filled_at=opened_at,
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="MET/USDT:USDT",
                        side="buy",
                        order_type="market",
                        quantity=90.0,
                        price=0.015,
                        status="filled",
                        fee=0.002,
                        decision_id=close_decision.id,
                        exchange_order_id="close-met-1",
                        filled_at=closed_at,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="MET/USDT",
                        side="short",
                        quantity=10.0,
                        entry_price=0.021,
                        current_price=0.015,
                        leverage=3.0,
                        realized_pnl=0.06,
                        is_open=False,
                        closed_at=closed_at,
                        created_at=opened_at - timedelta(hours=2),
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="MET/USDT",
                        side="short",
                        quantity=80.0,
                        entry_price=0.019,
                        current_price=0.015,
                        leverage=3.0,
                        realized_pnl=0.32,
                        is_open=False,
                        closed_at=closed_at + timedelta(seconds=2),
                        created_at=opened_at - timedelta(hours=1),
                    ),
                ]
            )
            await session.flush()

        async with get_session_ctx() as session:
            order_result = await session.execute(select(Order).where(Order.side == "buy"))
            close_order = order_result.scalar_one()
            plan = await plan_missing_closed_position(session, close_order)

        assert plan is None
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_does_not_pair_order_when_decision_action_conflicts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trading.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime(2026, 6, 21, 10, 43, tzinfo=UTC)
        closed_at = opened_at + timedelta(minutes=4)
        async with get_session_ctx() as session:
            wrong_entry_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="PROS/USDT",
                action="long",
                confidence=0.88,
                reasoning="wrong side",
                is_paper=True,
                was_executed=True,
            )
            close_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="PROS/USDT",
                action="close_short",
                confidence=0.92,
                reasoning="close",
                is_paper=True,
                was_executed=True,
            )
            session.add_all([wrong_entry_decision, close_decision])
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="PROS/USDT:USDT",
                        side="sell",
                        order_type="market",
                        quantity=9.0,
                        price=0.7316,
                        status="filled",
                        fee=0.001,
                        decision_id=wrong_entry_decision.id,
                        exchange_order_id="entry-1",
                        filled_at=opened_at,
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="PROS/USDT:USDT",
                        side="buy",
                        order_type="market",
                        quantity=9.0,
                        price=0.3879,
                        status="filled",
                        fee=0.002,
                        decision_id=close_decision.id,
                        exchange_order_id="close-1",
                        filled_at=closed_at,
                    ),
                ]
            )
            await session.flush()
            close_order_id = close_decision.id

        async with get_session_ctx() as session:
            order_result = await session.execute(select(Order).where(Order.side == "buy"))
            close_order = order_result.scalar_one()
            plan = await plan_missing_closed_position(session, close_order)

        assert plan is None
    finally:
        await close_db()
