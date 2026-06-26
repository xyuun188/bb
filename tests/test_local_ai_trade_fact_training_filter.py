from datetime import UTC, datetime, timedelta

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.learning import TradeReflection
from models.trade import Position
from scripts.train_local_ai_tools_models import (
    _completed_trade_sample_count,
    _load_closed_position_samples,
    _load_trade_reflection_samples,
)


@pytest.mark.asyncio
async def test_local_ai_trade_samples_exclude_unlinked_okx_position_facts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trade-fact-training.db').as_posix()}",
    )
    await init_db()
    closed_at = datetime(2026, 6, 26, 5, 30, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            trusted = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="BTC/USDT",
                side="long",
                quantity=1.0,
                entry_price=100.0,
                current_price=110.0,
                realized_pnl=9.9,
                is_open=False,
                closed_at=closed_at,
                created_at=closed_at - timedelta(minutes=10),
                okx_inst_id="BTC-USDT-SWAP",
                entry_exchange_order_id="entry-ok",
                close_exchange_order_id="close-ok",
            )
            dirty = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="USAR/USDT",
                side="long",
                quantity=10.0,
                entry_price=2.31,
                current_price=4.05,
                realized_pnl=17.0,
                is_open=False,
                closed_at=closed_at + timedelta(seconds=1),
                created_at=closed_at - timedelta(minutes=20),
                okx_inst_id="USAR-USDT-SWAP",
                entry_exchange_order_id="entry-dirty",
            )
            session.add_all([trusted, dirty])
            await session.flush()
            session.add_all(
                [
                    _reflection(trusted.id, "BTC/USDT", 9.9),
                    _reflection(dirty.id, "USAR/USDT", 17.0),
                ]
            )

        reflections = await _load_trade_reflection_samples(10)
        closed_positions = await _load_closed_position_samples(10)
        completed_count = await _completed_trade_sample_count()
    finally:
        await close_db()

    assert [sample["position_id"] for sample in reflections] == [trusted.id]
    assert [sample["position_id"] for sample in closed_positions] == [trusted.id]
    assert completed_count == 1


def _reflection(position_id: int, symbol: str, realized_pnl: float) -> TradeReflection:
    return TradeReflection(
        position_id=position_id,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol=symbol,
        side="long",
        entry_price=1.0,
        exit_price=2.0,
        quantity=1.0,
        realized_pnl=realized_pnl,
        fee_estimate=0.1,
        hold_minutes=10.0,
        outcome="profit",
        source="unit_test",
    )
