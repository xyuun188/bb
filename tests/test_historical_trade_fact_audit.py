from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.learning import TradeReflection
from models.trade import Position
from services.historical_trade_fact_audit import HistoricalTradeFactAuditService


@pytest.mark.asyncio
async def test_historical_trade_fact_audit_classifies_clean_quarantined_and_repaired(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'historical-facts.db').as_posix()}",
    )
    await init_db()
    now = datetime(2026, 6, 26, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            clean = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="BTC/USDT",
                side="long",
                quantity=1.0,
                entry_price=100.0,
                current_price=110.0,
                realized_pnl=10.0,
                is_open=False,
                created_at=now - timedelta(days=2),
                closed_at=now - timedelta(days=1),
                entry_exchange_order_id="entry-ok",
                close_exchange_order_id="close-ok",
            )
            missing_entry = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="HOME/USDT",
                side="long",
                quantity=1.0,
                entry_price=1.0,
                current_price=0.9,
                realized_pnl=-0.1,
                is_open=False,
                created_at=now - timedelta(days=2),
                closed_at=now - timedelta(days=1),
                entry_exchange_order_id="",
                close_exchange_order_id="close-home",
            )
            manual_close = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="SPK/USDT",
                side="short",
                quantity=10.0,
                entry_price=0.02,
                current_price=0.019,
                realized_pnl=0.01,
                is_open=False,
                created_at=now - timedelta(days=2),
                closed_at=now - timedelta(days=1),
                entry_exchange_order_id="entry-spk",
                close_exchange_order_id="manual_close:local-only",
            )
            repaired = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="ETH/USDT",
                side="short",
                quantity=2.0,
                entry_price=100.0,
                current_price=90.0,
                realized_pnl=20.0,
                is_open=False,
                created_at=now - timedelta(days=2),
                closed_at=now - timedelta(days=1),
                entry_exchange_order_id="entry-eth",
                close_exchange_order_id="close-eth",
            )
            session.add_all([clean, missing_entry, manual_close, repaired])
            await session.flush()
            session.add(
                TradeReflection(
                    position_id=repaired.id,
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="ETH/USDT",
                    side="short",
                    realized_pnl=20.0,
                    source="okx_position_link_repair",
                    expert_lessons={"training_policy": "exclude_until_manual_trust"},
                )
            )

        report = await HistoricalTradeFactAuditService(lookback_days=30, limit=100).report()
    finally:
        await close_db()

    assert report["status"] == "dirty"
    assert report["read_only"] is True
    assert report["can_delete_history"] is False
    assert report["can_apply_repair"] is False
    assert report["checked_closed_positions"] == 4
    assert report["trainable_closed_positions"] == 1
    assert report["quarantined_closed_positions"] == 3
    assert report["repairable_candidate_count"] == 1
    assert report["manual_close_marker_count"] == 1
    assert report["historical_repair_provenance_count"] == 1
    assert report["reason_counts"]["missing_entry_exchange_order_id"] == 1
    assert report["reason_counts"]["manual_close_exchange_order_id"] == 1
    assert report["reason_counts"]["historical_repair_provenance"] == 1
    assert {sample["symbol"] for sample in report["samples"]} == {
        "HOME/USDT",
        "SPK/USDT",
        "ETH/USDT",
    }
