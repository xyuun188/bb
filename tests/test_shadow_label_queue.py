from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.repositories.memory_repo import MemoryRepository
from models.decision import AIDecision
from models.learning import ShadowBacktest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limit", "expected"),
    [(1, [1]), (4, [1, 2, 9, 10]), (5, [1, 2, 3, 9, 10]), (25, list(range(1, 11)))],
)
async def test_due_label_queue_keeps_oldest_progress_and_fresh_evidence(limit, expected):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    now = datetime.now(UTC)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(ShadowBacktest.__table__.create)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        async with sessions() as session:
            session.add_all(
                [
                    ShadowBacktest(
                        id=index,
                        model_name="test",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        due_at=now - timedelta(minutes=11 - index),
                        status="pending",
                    )
                    for index in range(1, 11)
                ]
            )
            session.add_all(
                [
                    ShadowBacktest(
                        id=11,
                        model_name="test",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        due_at=now + timedelta(minutes=5),
                        status="pending",
                    ),
                    ShadowBacktest(
                        id=12,
                        model_name="test",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        due_at=now - timedelta(days=2),
                        status="completed",
                    ),
                    ShadowBacktest(
                        id=13,
                        model_name="test",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        due_at=now - timedelta(days=1),
                        status="quarantined",
                    ),
                ]
            )
            await session.commit()
            rows = await MemoryRepository(session).get_due_shadow_backtests(limit=limit)
            ids = [row.id for row in rows]
            assert ids == expected
            assert len(set(ids)) == len(ids)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_shadow_label_batch_is_idempotent_for_all_horizons():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    now = datetime.now(UTC)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(AIDecision.__table__.create)
            await connection.run_sync(ShadowBacktest.__table__.create)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        payload = {
            "decision_id": 7002,
            "model_name": "ensemble_trader",
            "execution_mode": "paper",
            "symbol": "ETH/USDT",
            "analysis_type": "market",
            "decision_action": "hold",
            "decision_confidence": 0.7,
            "entry_price": 100.0,
            "feature_snapshot": {},
            "raw_llm_response": {},
            "status": "pending",
        }
        async with sessions() as session:
            repo = MemoryRepository(session)
            rows = await repo.create_shadow_backtests(
                [
                    {
                        **payload,
                        "horizon_minutes": horizon,
                        "due_at": now + timedelta(minutes=horizon),
                    }
                    for horizon in (5, 15, 60, 240)
                ]
            )
            duplicate_rows = await repo.create_shadow_backtests(
                [
                    {
                        **payload,
                        "horizon_minutes": horizon,
                        "due_at": now + timedelta(minutes=horizon),
                    }
                    for horizon in (5, 15, 60, 240)
                ]
            )
            assert [row.horizon_minutes for row in rows] == [5, 15, 60, 240]
            assert [row.id for row in duplicate_rows] == [row.id for row in rows]
            assert len({row.id for row in rows}) == 4
    finally:
        await engine.dispose()
