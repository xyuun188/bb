from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from scripts.export_quant_optimization_baseline import _position_metrics


@pytest.mark.asyncio
async def test_position_metrics_quarantines_untrusted_closed_trade_facts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    from db.session import close_db, get_session_ctx, init_db
    from models.trade import Position
    from web_dashboard.api import system_audit

    await close_db()
    db_path = tmp_path / "baseline.db"
    monkeypatch.setattr(
        system_audit.settings,
        "database_url",
        f"sqlite+aiosqlite:///{db_path.as_posix()}",
    )
    now = datetime(2026, 6, 22, 4, 0, tzinfo=UTC)
    since = now - timedelta(hours=24)

    await init_db()
    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Position(
                        model_name="test_model",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        side="long",
                        quantity=1,
                        entry_price=100,
                        current_price=110,
                        is_open=False,
                        realized_pnl=10,
                        entry_exchange_order_id="entry-ok",
                        close_exchange_order_id="close-ok",
                        created_at=(now - timedelta(minutes=30)).replace(tzinfo=None),
                        closed_at=(now - timedelta(minutes=5)).replace(tzinfo=None),
                    ),
                    Position(
                        model_name="test_model",
                        execution_mode="paper",
                        symbol="ETH/USDT",
                        side="short",
                        quantity=1,
                        entry_price=100,
                        current_price=90,
                        is_open=False,
                        realized_pnl=-90,
                        created_at=(now - timedelta(minutes=20)).replace(tzinfo=None),
                        closed_at=(now - timedelta(minutes=3)).replace(tzinfo=None),
                    ),
                ]
            )

        metrics = await _position_metrics(since.replace(tzinfo=None))

        assert metrics["raw_closed_count"] == 2
        assert metrics["closed_count"] == 1
        assert metrics["trade_fact_quarantined_closed_position_count"] == 1
        assert metrics["trade_fact_quarantine_reasons"] == {
            "missing_entry_exchange_order_id": 1
        }
        assert metrics["realized_pnl_distribution"]["count"] == 1
        assert metrics["realized_pnl_distribution"]["avg"] == 10
        assert metrics["raw_realized_pnl_distribution"]["count"] == 2
        assert metrics["realized_pnl_policy"] == "trusted_closed_positions_only"
    finally:
        await close_db()
