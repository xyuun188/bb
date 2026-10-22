from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.training_contracts import SHADOW_LABEL_VERSION
from db.repositories.memory_repo import MemoryRepository
from models.decision import AIDecision
from models.learning import ShadowBacktest
from services.shadow_backtest_service import ShadowBacktestService


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        (1, [1]),
        (4, [1, 10, 2, 9]),
        (5, [1, 10, 2, 9, 3]),
        (25, [1, 10, 2, 9, 3, 8, 4, 7, 5, 6]),
    ],
)
async def test_due_label_queue_interleaves_oldest_and_fresh_evidence(limit, expected):
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
            assert all("raw_llm_response" in inspect(row).unloaded for row in rows)
            assert all("feature_snapshot" not in inspect(row).unloaded for row in rows)
            excluded_rows = await MemoryRepository(session).get_due_shadow_backtests(
                limit=limit, exclude_ids=[1, 10],
            )
            assert not {row.id for row in excluded_rows}.intersection({1, 10})
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


@pytest.mark.asyncio
async def test_real_repository_recovers_partial_current_version_at_original_observation_time():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    observed_at = datetime.now(UTC) - timedelta(minutes=90)

    @asynccontextmanager
    async def session_ctx():
        async with sessions() as session:
            yield session
            await session.commit()

    async def forbidden_quote(_symbol):
        raise AssertionError("recovery must not substitute a current quote")

    try:
        async with engine.begin() as connection:
            await connection.run_sync(AIDecision.__table__.create)
            await connection.run_sync(ShadowBacktest.__table__.create)
        async with session_ctx() as session:
            session.add(AIDecision(
                id=123, model_name="ensemble_trader", symbol="BTC/USDT",
                analysis_type="market", action="hold", confidence=0.7,
                feature_snapshot={"current_price": 100.0},
                raw_llm_response={"original": True}, created_at=observed_at,
            ))
            session.add_all([
                ShadowBacktest(
                    id=index, decision_id=123, model_name="ensemble_trader",
                    execution_mode="paper", symbol="BTC/USDT",
                    due_at=observed_at+timedelta(minutes=horizon),
                    horizon_minutes=horizon, label_version=version,
                    status="pending", created_at=observed_at,
                )
                for index, horizon, version in (
                    (1, 5, SHADOW_LABEL_VERSION), (2, 15, "obsolete-label-version"),
                )
            ])
        async with session_ctx() as session:
            rows = await MemoryRepository(session).get_market_decisions_missing_shadow_samples(
                since=observed_at-timedelta(minutes=1), horizon_count=2, limit=10,
            )
            assert [row.id for row in rows] == [123]
            assert rows[0].raw_llm_response == {"original": True}
            assert "model_health_timings" in inspect(rows[0]).unloaded
        service = ShadowBacktestService(
            latest_price_provider=forbidden_quote,
            symbol_normalizer=lambda x: x,
            float_parser=lambda x, fallback=0: float(x) if x is not None else fallback,
            session_factory=session_ctx, horizons_minutes=(5, 15),
        )
        assert await service.recover_missing_market_samples(limit=10) == {
            "scanned": 1, "recovered": 1, "failed": 0,
        }
        async with session_ctx() as session:
            rows = (await session.execute(select(ShadowBacktest).where(
                ShadowBacktest.label_version == SHADOW_LABEL_VERSION,
            ).order_by(ShadowBacktest.horizon_minutes))).scalars().all()
            assert [row.horizon_minutes for row in rows] == [5, 15]
            assert all(row.decision_id == 123 for row in rows)
            assert rows[-1].entry_price == 100.0
            assert rows[-1].raw_llm_response == {"original": True}
            for row in rows:
                assert row.created_at.replace(tzinfo=UTC) == observed_at
                assert row.due_at.replace(tzinfo=UTC) == (
                    observed_at+timedelta(minutes=row.horizon_minutes)
                )
        assert await service.recover_missing_market_samples(limit=10) == {
            "scanned": 0, "recovered": 0, "failed": 0,
        }
    finally:
        await engine.dispose()
