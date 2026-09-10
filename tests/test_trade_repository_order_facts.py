from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.repositories.trade_repo import TradeRepository
from models.decision import AIDecision
from models.trade import Order


class _RaceResult:
    def __init__(self, row) -> None:
        self.row = row

    def scalar_one_or_none(self):
        return self.row


class _RaceSession:
    def __init__(self, existing) -> None:
        self.existing = existing
        self.execute_count = 0
        self.flush_count = 0
        self.expunge_count = 0

    def get_bind(self):
        return None

    async def execute(self, _statement):
        self.execute_count += 1
        return _RaceResult(None if self.execute_count == 1 else self.existing)

    @asynccontextmanager
    async def begin_nested(self):
        yield

    @contextmanager
    def no_autoflush(self):
        yield

    def add(self, _order) -> None:
        pass

    def expunge(self, _order) -> None:
        self.expunge_count += 1

    async def flush(self) -> None:
        self.flush_count += 1
        if self.flush_count == 1:
            raise IntegrityError("insert", {}, Exception("duplicate"))

    async def get(self, _model, _id):
        return None


async def test_same_exchange_order_from_different_decisions_reuses_one_fact() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(AIDecision.__table__.create)
            await conn.run_sync(Order.__table__.create)

        async with sessions() as session:
            sync_decision = AIDecision(
                model_name="okx_authoritative_sync",
                symbol="GRAM/USDT",
                action="close_short",
                confidence=1.0,
                raw_llm_response={"system_sync": True},
            )
            production_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="GRAM/USDT",
                action="close_short",
                confidence=0.8,
                raw_llm_response={"system_sync": False},
            )
            session.add_all([sync_decision, production_decision])
            await session.flush()

            repo = TradeRepository(session)
            common = {
                "execution_mode": "paper",
                "symbol": "GRAM/USDT",
                "side": "buy",
                "order_type": "market",
                "quantity": 100.0,
                "price": 0.0042,
                "status": "filled",
                "fee": 0.01,
                "exchange_order_id": "3859763988972404736",
            }
            first, first_created = await repo.create_order_fact(
                {
                    **common,
                    "model_name": "okx_authoritative_sync",
                    "decision_id": sync_decision.id,
                }
            )
            second, second_created = await repo.create_order_fact(
                {
                    **common,
                    "model_name": "ensemble_trader",
                    "decision_id": production_decision.id,
                    "okx_fill_contracts": 100.0,
                    "okx_sync_status": "okx_confirmed",
                }
            )

            assert first_created is True
            assert second_created is False
            assert second.id == first.id
            assert second.decision_id == production_decision.id
            assert second.okx_fill_contracts == 100.0
            assert second.okx_sync_status == "okx_confirmed"
            assert (
                await session.scalar(
                    select(func.count(Order.id)).where(
                        Order.execution_mode == "paper",
                        Order.exchange_order_id == common["exchange_order_id"],
                    )
                )
            ) == 1
    finally:
        await engine.dispose()


async def test_exchange_fact_insert_race_reuses_committed_row_after_savepoint() -> None:
    existing = SimpleNamespace(
        execution_mode="paper",
        exchange_order_id="race-order",
        status="pending",
        quantity=0.0,
        price=0.0,
        fee=0.0,
        filled_at=None,
        decision_id=None,
        okx_raw_fills={},
    )
    session = _RaceSession(existing)

    order, created = await TradeRepository(session).create_order_fact(
        {
            "model_name": "okx_authoritative_sync",
            "execution_mode": "paper",
            "symbol": "ADA/USDT",
            "side": "sell",
            "order_type": "market",
            "quantity": 43.0,
            "price": 0.2166,
            "status": "filled",
            "fee": 0.0046,
            "exchange_order_id": "race-order",
        }
    )

    assert order is existing
    assert created is False
    assert existing.status == "filled"
    assert existing.quantity == 43.0
    assert existing.price == 0.2166
    assert session.flush_count == 2
    assert session.expunge_count == 1
