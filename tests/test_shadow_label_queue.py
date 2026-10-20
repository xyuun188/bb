from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.repositories.memory_repo import MemoryRepository
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
