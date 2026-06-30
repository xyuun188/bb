from datetime import UTC, datetime, timedelta

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.learning import TradeReflection
from models.trade import Order, Position
from scripts.train_local_ai_tools_models import (
    _completed_trade_sample_count,
    _load_closed_position_samples,
    _load_trade_reflection_samples,
    _merge_trade_samples,
)
from services.okx_order_fact_sync import OKX_SYNC_CONFIRMED
from services.training_data_quality import annotate_training_payload


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

    assert sorted(sample["position_id"] for sample in reflections) == sorted(
        [dirty.id, trusted.id]
    )
    assert sorted(sample["position_id"] for sample in closed_positions) == sorted(
        [dirty.id, trusted.id]
    )
    assert completed_count == 1

    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=_merge_trade_samples(reflections, closed_positions),
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert [sample["position_id"] for sample in payload["trade_samples"]] == [trusted.id]
    top_reasons = {item["reason"]: item["count"] for item in payload["quality_report"]["top_reasons"]}
    assert top_reasons["trade:untrusted_trade_fact:missing_close_exchange_order_id"] >= 1


@pytest.mark.asyncio
async def test_local_ai_closed_position_samples_require_okx_confirmed_orders(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trade-okx-confirmed-training.db').as_posix()}",
    )
    await init_db()
    closed_at = datetime(2026, 6, 28, 5, 30, tzinfo=UTC)
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
            unconfirmed = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="ETH/USDT",
                side="long",
                quantity=1.0,
                entry_price=100.0,
                current_price=101.0,
                realized_pnl=1.0,
                is_open=False,
                closed_at=closed_at + timedelta(seconds=1),
                created_at=closed_at - timedelta(minutes=9),
                okx_inst_id="ETH-USDT-SWAP",
                entry_exchange_order_id="entry-local",
                close_exchange_order_id="close-local",
            )
            session.add_all([trusted, unconfirmed])
            await session.flush()
            session.add_all(
                [
                    _order("BTC/USDT", "entry-ok", OKX_SYNC_CONFIRMED, closed_at),
                    _order("BTC/USDT", "close-ok", OKX_SYNC_CONFIRMED, closed_at),
                    _order("ETH/USDT", "entry-local", None, closed_at),
                    _order("ETH/USDT", "close-local", OKX_SYNC_CONFIRMED, closed_at),
                ]
            )

        closed_positions = await _load_closed_position_samples(10)
    finally:
        await close_db()

    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=closed_positions,
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert [sample["position_id"] for sample in payload["trade_samples"]] == [trusted.id]
    top_reasons = {item["reason"]: item["count"] for item in payload["quality_report"]["top_reasons"]}
    assert top_reasons["trade:untrusted_trade_fact:entry_order_not_okx_confirmed"] >= 1


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


def _order(
    symbol: str,
    exchange_order_id: str,
    okx_sync_status: str | None,
    filled_at: datetime,
) -> Order:
    return Order(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol=symbol,
        side="buy",
        order_type="market",
        quantity=1.0,
        price=1.0,
        status="filled",
        fee=0.05,
        exchange_order_id=exchange_order_id,
        okx_sync_status=okx_sync_status,
        filled_at=filled_at,
        created_at=filled_at,
    )


@pytest.mark.asyncio
async def test_local_ai_trade_repair_reflections_are_quarantined_from_training(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trade-repair-training.db').as_posix()}",
    )
    await init_db()
    closed_at = datetime(2026, 6, 26, 6, 0, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repaired = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="PROS/USDT",
                side="long",
                quantity=9.0,
                entry_price=0.42,
                current_price=0.45,
                realized_pnl=1.2,
                is_open=False,
                closed_at=closed_at,
                created_at=closed_at - timedelta(minutes=15),
                okx_inst_id="PROS-USDT-SWAP",
                entry_exchange_order_id="entry-repair",
                close_exchange_order_id="close-repair",
            )
            trusted = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="BTC/USDT",
                side="long",
                quantity=1.0,
                entry_price=100.0,
                current_price=103.0,
                realized_pnl=2.7,
                is_open=False,
                closed_at=closed_at + timedelta(seconds=1),
                created_at=closed_at - timedelta(minutes=25),
                okx_inst_id="BTC-USDT-SWAP",
                entry_exchange_order_id="entry-ok",
                close_exchange_order_id="close-ok",
            )
            session.add_all([repaired, trusted])
            await session.flush()
            session.add_all(
                [
                    TradeReflection(
                        position_id=repaired.id,
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="PROS/USDT",
                        side="long",
                        entry_price=0.42,
                        exit_price=0.45,
                        quantity=9.0,
                        realized_pnl=1.2,
                        fee_estimate=0.02,
                        hold_minutes=15.0,
                        outcome="profit",
                        source="okx_order_pair_repair",
                        expert_lessons={"source": "missing_closed_position_repair"},
                    ),
                    TradeReflection(
                        position_id=trusted.id,
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        side="long",
                        entry_price=100.0,
                        exit_price=103.0,
                        quantity=1.0,
                        realized_pnl=2.7,
                        fee_estimate=0.08,
                        hold_minutes=25.0,
                        outcome="profit",
                        source="unit_test",
                    ),
                ]
            )

        reflections = await _load_trade_reflection_samples(10)
    finally:
        await close_db()

    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=reflections,
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert sorted(sample["position_id"] for sample in reflections) == [repaired.id, trusted.id]
    assert [sample["position_id"] for sample in payload["trade_samples"]] == [trusted.id]
    top_reasons = {item["reason"]: item["count"] for item in payload["quality_report"]["top_reasons"]}
    assert top_reasons["trade:historical_reconciliation_repair:missing_closed_position_repair"] >= 1


@pytest.mark.asyncio
async def test_local_ai_position_link_repairs_remain_quarantined_from_training(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'position-link-repair-training.db').as_posix()}",
    )
    await init_db()
    closed_at = datetime(2026, 6, 26, 6, 30, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repaired = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="HOME/USDT",
                side="long",
                quantity=12.0,
                entry_price=0.031,
                current_price=0.034,
                realized_pnl=0.036,
                is_open=False,
                closed_at=closed_at,
                created_at=closed_at - timedelta(minutes=12),
                okx_inst_id="HOME-USDT-SWAP",
                entry_exchange_order_id="entry-home",
                close_exchange_order_id="close-home",
            )
            trusted = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="BTC/USDT",
                side="long",
                quantity=1.0,
                entry_price=100.0,
                current_price=104.0,
                realized_pnl=3.9,
                is_open=False,
                closed_at=closed_at + timedelta(seconds=1),
                created_at=closed_at - timedelta(minutes=20),
                okx_inst_id="BTC-USDT-SWAP",
                entry_exchange_order_id="entry-ok",
                close_exchange_order_id="close-ok",
            )
            session.add_all([repaired, trusted])
            await session.flush()
            session.add_all(
                [
                    TradeReflection(
                        position_id=repaired.id,
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="HOME/USDT",
                        side="long",
                        entry_price=0.031,
                        exit_price=0.034,
                        quantity=12.0,
                        realized_pnl=0.036,
                        fee_estimate=0.0,
                        hold_minutes=12.0,
                        outcome="profit",
                        source="okx_position_link_repair",
                        expert_lessons={"source": "okx_position_link_repair"},
                    ),
                    TradeReflection(
                        position_id=trusted.id,
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        side="long",
                        entry_price=100.0,
                        exit_price=104.0,
                        quantity=1.0,
                        realized_pnl=3.9,
                        fee_estimate=0.08,
                        hold_minutes=20.0,
                        outcome="profit",
                        source="unit_test",
                    ),
                ]
            )

        reflections = await _load_trade_reflection_samples(10)
        closed_positions = await _load_closed_position_samples(10)
    finally:
        await close_db()

    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=_merge_trade_samples(reflections, closed_positions),
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert [sample["position_id"] for sample in payload["trade_samples"]] == [trusted.id]
    top_reasons = {item["reason"]: item["count"] for item in payload["quality_report"]["top_reasons"]}
    assert top_reasons["trade:historical_reconciliation_repair:okx_position_link_repair"] >= 1
