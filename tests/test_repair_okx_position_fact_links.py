from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.trade import Order, Position
from scripts.repair_okx_position_fact_links import collect_scan_report


@pytest.mark.asyncio
async def test_position_fact_link_scan_reports_repairable_entry_link(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trading.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime.now(UTC) - timedelta(minutes=5)
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="PROS/USDT:USDT",
                        side="buy",
                        order_type="market",
                        quantity=9.0,
                        price=0.7316,
                        status="filled",
                        fee=0.001,
                        exchange_order_id="entry-pros-1",
                        filled_at=opened_at,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="PROS/USDT",
                        side="long",
                        quantity=9.0,
                        entry_price=0.7316,
                        current_price=0.7316,
                        leverage=3.0,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        is_open=True,
                        created_at=opened_at,
                    ),
                ]
            )
            await session.flush()

        report = await collect_scan_report(days=14)

        assert report.candidate_link_count == 1
        assert report.repairable_count == 1
        assert report.manual_review_count == 0
        assert report.classification_counts == {"repairable": 1, "manual_review": 0}
        assert report.scanned_position_count == 1
        assert report.max_positions == 500
        assert report.truncated is False
        assert len(report.plans) == 1
        assert report.plans[0].link_kind == "entry"
        assert report.diagnostics[0]["status"] == "repairable"
        assert report.diagnostics[0]["reason"] == "deterministic_position_order_match"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_fact_link_scan_reports_manual_review_when_close_order_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trading.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime.now(UTC) - timedelta(hours=1)
        closed_at = opened_at + timedelta(minutes=30)
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="PROS/USDT",
                    side="long",
                    quantity=9.0,
                    entry_price=0.7316,
                    current_price=0.7879,
                    leverage=3.0,
                    unrealized_pnl=0.0,
                    realized_pnl=0.5067,
                    is_open=False,
                    created_at=opened_at,
                    closed_at=closed_at,
                    entry_exchange_order_id="entry-pros-1",
                )
            )
            await session.flush()

        report = await collect_scan_report(days=14)

        assert report.candidate_link_count == 1
        assert report.repairable_count == 0
        assert report.manual_review_count == 1
        assert report.classification_counts == {"repairable": 0, "manual_review": 1}
        assert report.scanned_position_count == 1
        assert report.truncated is False
        assert report.plans == []
        assert report.diagnostics[0]["status"] == "manual_review"
        assert report.diagnostics[0]["reason"] == "missing_matching_close_order"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_fact_link_scan_does_not_treat_manual_close_as_okx_link(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trading.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime.now(UTC) - timedelta(hours=1)
        closed_at = opened_at + timedelta(minutes=30)
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="PROS/USDT:USDT",
                        side="sell",
                        order_type="market",
                        quantity=9.0,
                        price=0.7879,
                        status="filled",
                        fee=0.001,
                        exchange_order_id="manual_close:local-only",
                        filled_at=closed_at,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="PROS/USDT",
                        side="long",
                        quantity=9.0,
                        entry_price=0.7316,
                        current_price=0.7879,
                        leverage=3.0,
                        unrealized_pnl=0.0,
                        realized_pnl=0.5067,
                        is_open=False,
                        created_at=opened_at,
                        closed_at=closed_at,
                        entry_exchange_order_id="entry-pros-1",
                    ),
                ]
            )
            await session.flush()

        report = await collect_scan_report(days=14)

        assert report.plans == []
        assert report.manual_review_count == 1
        assert report.diagnostics[0]["reason"] == "missing_matching_close_order"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_fact_link_scan_respects_existing_okx_inst_id(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trading.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime.now(UTC) - timedelta(minutes=5)
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SAHARA/USDT:USDT",
                        side="buy",
                        order_type="market",
                        quantity=9.0,
                        price=0.7316,
                        status="filled",
                        fee=0.001,
                        exchange_order_id="entry-sahara-1",
                        filled_at=opened_at,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SPK/USDT",
                        side="long",
                        quantity=9.0,
                        entry_price=0.7316,
                        current_price=0.7316,
                        leverage=3.0,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        is_open=True,
                        okx_inst_id="SPK-USDT-SWAP",
                        created_at=opened_at,
                    ),
                ]
            )
            await session.flush()

        report = await collect_scan_report(days=14)

        assert report.plans == []
        assert report.manual_review_count == 1
        assert report.diagnostics[0]["reason"] == "missing_matching_entry_order"
    finally:
        await close_db()
