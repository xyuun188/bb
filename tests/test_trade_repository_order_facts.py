from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.repositories.trade_repo import TradeRepository
from models.decision import AIDecision
from models.trade import Order


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
