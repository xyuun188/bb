from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from models.trade import OkxPositionHistory, Position
from web_dashboard.api import dashboard


@pytest.mark.asyncio
async def test_closed_ledger_read_model_reuses_same_watermark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard._clear_dashboard_heavy_cache("closed-position-ledger")
    watermark = {"value": ("v1",)}
    builds = {"count": 0}

    async def fake_watermark(
        _session: Any,
        *,
        mode: str | None,
        model_names: tuple[str, ...] | None,
    ) -> tuple[Any, ...]:
        assert mode == "paper"
        assert model_names is None
        return watermark["value"]

    async def fake_build(*_args: Any, **_kwargs: Any) -> tuple[list[dict[str, Any]], int, int, int, str]:
        builds["count"] += 1
        return ([{"build": builds["count"]}], 1, 1, 1, "test")

    monkeypatch.setattr(dashboard, "_dashboard_closed_ledger_watermark", fake_watermark)
    monkeypatch.setattr(dashboard, "_dashboard_closed_position_ledger_rows_uncached", fake_build)

    first = await dashboard._dashboard_closed_position_ledger_rows(
        object(),
        object(),
        mode="paper",
    )
    second = await dashboard._dashboard_closed_position_ledger_rows(
        object(),
        object(),
        mode="paper",
    )

    assert first == second
    assert builds["count"] == 1

    watermark["value"] = ("v2",)
    refreshed = await dashboard._dashboard_closed_position_ledger_rows(
        object(),
        object(),
        mode="paper",
    )

    assert refreshed[0][0]["build"] == 2
    assert builds["count"] == 2


@pytest.mark.asyncio
async def test_closed_ledger_watermark_changes_when_history_row_updates_in_place(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'closed-ledger-watermark.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    try:
        async with get_session_ctx() as session:
            session.add(
                OkxPositionHistory(
                    mode="paper",
                    row_identity="paper|XRP-USDT-SWAP|xrp-pos|net|1",
                    inst_id="XRP-USDT-SWAP",
                    symbol="XRP/USDT",
                    pos_id="xrp-pos",
                    pos_side="net",
                    side="short",
                    close_type="1",
                    close_status="partial",
                    raw_row={"instId": "XRP-USDT-SWAP", "type": "1"},
                    sync_status="synced",
                    synced_at=now - timedelta(minutes=1),
                )
            )

        async with get_session_ctx() as session:
            before = await dashboard._dashboard_closed_ledger_watermark(
                session,
                mode="paper",
                model_names=None,
            )

        async with get_session_ctx() as session:
            record = (
                await session.execute(
                    select(OkxPositionHistory).where(
                        OkxPositionHistory.mode == "paper"
                    )
                )
            ).scalars().one()
            record.close_type = "2"
            record.close_status = "full"
            record.synced_at = now

        async with get_session_ctx() as session:
            after = await dashboard._dashboard_closed_ledger_watermark(
                session,
                mode="paper",
                model_names=None,
            )

        assert after != before
    finally:
        await close_db()


def test_analysis_payload_bounds_transcripts_and_nested_collections() -> None:
    payload = {
        "reasoning": "x" * 5000,
        "opinions": [{"reasoning": "y" * 5000} for _ in range(120)],
        "nested": {"rows": list(range(120))},
    }

    bounded = dashboard._bounded_dashboard_payload(payload)

    assert len(bounded["reasoning"]) < 1700
    assert bounded["reasoning"].endswith("...")
    assert len(bounded["opinions"]) == 80
    assert len(bounded["opinions"][0]["reasoning"]) < 1700
    assert len(bounded["nested"]["rows"]) == 80


@pytest.mark.asyncio
async def test_profit_attribution_watermark_ignores_unrelated_new_decisions(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'profit-watermark.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    since = now - timedelta(hours=24)
    try:
        async with get_session_ctx() as session:
            position = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="BTC/USDT",
                side="long",
                quantity=0.1,
                entry_price=100.0,
                current_price=101.0,
                realized_pnl=0.1,
                is_open=False,
                closed_at=now - timedelta(minutes=10),
                created_at=now - timedelta(hours=1),
                updated_at=now - timedelta(minutes=9),
            )
            session.add(position)

        async with get_session_ctx() as session:
            before = await dashboard._profit_attribution_watermark(
                session,
                selected_mode="paper",
                since=since,
            )

        async with get_session_ctx() as session:
            session.add(
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="ETH/USDT",
                    action="hold",
                    confidence=0.5,
                    is_paper=True,
                    created_at=now,
                )
            )

        async with get_session_ctx() as session:
            after_decision = await dashboard._profit_attribution_watermark(
                session,
                selected_mode="paper",
                since=since,
            )
            persisted_position = await session.get(Position, position.id)
            assert persisted_position is not None
            persisted_position.updated_at = now

        async with get_session_ctx() as session:
            after_position_update = await dashboard._profit_attribution_watermark(
                session,
                selected_mode="paper",
                since=since,
            )

        assert after_decision == before
        assert after_position_update != before
    finally:
        await close_db()
