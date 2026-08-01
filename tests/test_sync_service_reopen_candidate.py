from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.trade import Position
from services.sync_service import (
    ORPHAN_QUARANTINE_CLOSE_PREFIX,
    _find_reopenable_closed_position,
)


@pytest.mark.asyncio
async def test_reopen_candidate_selects_quarantined_remainder_not_real_partial_close(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'sync-reopen-candidate.db').as_posix()}",
    )
    await init_db()
    created_at = datetime(2026, 8, 1, 2, 0, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            partial_close = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="LTC/USDT",
                side="short",
                quantity=0.2,
                entry_price=45.0,
                current_price=44.0,
                is_open=False,
                created_at=created_at,
                closed_at=created_at + timedelta(minutes=5),
                entry_exchange_order_id="ltc-entry",
                close_exchange_order_id="ltc-real-partial-close",
            )
            quarantined_remainder = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="LTC/USDT",
                side="short",
                quantity=3.8,
                entry_price=45.0,
                current_price=44.0,
                is_open=False,
                created_at=created_at,
                closed_at=created_at + timedelta(minutes=10),
                entry_exchange_order_id="ltc-entry",
                close_exchange_order_id=f"{ORPHAN_QUARANTINE_CLOSE_PREFIX}5708",
            )
            session.add_all([partial_close, quarantined_remainder])
            await session.flush()

            candidate = await _find_reopenable_closed_position(
                session,
                symbol_variants={"LTC/USDT", "LTC/USDT:USDT"},
                side="short",
                entry_exchange_order_id="ltc-entry",
                quantity=3.8,
            )

            assert candidate is quarantined_remainder
            assert partial_close.close_exchange_order_id == "ltc-real-partial-close"
            assert partial_close.quantity == pytest.approx(0.2)
    finally:
        await close_db()
