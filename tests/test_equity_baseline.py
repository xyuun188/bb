from __future__ import annotations

from datetime import UTC, datetime

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.account import ExecutionEquitySnapshot
from services.equity_baseline import apply_daily_equity_baseline, phase3_equity_change_from_snapshots


@pytest.mark.asyncio
async def test_daily_equity_baseline_uses_okx_current_equity_when_available(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'equity-baseline.db').as_posix()}",
    )
    await init_db()
    try:
        async with get_session_ctx() as session:
            result = await apply_daily_equity_baseline(
                session,
                mode="paper",
                model_name="ensemble_trader",
                allocated=4000.0,
                positions=[],
                realized_pnl=9.22,
                unrealized_pnl=0.0,
                total_pnl=9.22,
                current_equity=4998.15,
                now=datetime(2026, 6, 28, 12, 0, tzinfo=UTC),
            )

        assert result["today_equity_baseline"] == pytest.approx(4998.15)
        assert result["today_equity_baseline_total_pnl"] is None
        assert result["today_equity_baseline_source"] == "okx_snapshot"
        assert result["today_equity_pnl"] == pytest.approx(0.0)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_daily_equity_baseline_refuses_local_estimate_without_okx_equity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'equity-baseline-no-okx.db').as_posix()}",
    )
    await init_db()
    try:
        async with get_session_ctx() as session:
            result = await apply_daily_equity_baseline(
                session,
                mode="paper",
                model_name="ensemble_trader",
                allocated=4000.0,
                positions=[],
                realized_pnl=9.22,
                unrealized_pnl=0.0,
                total_pnl=9.22,
                current_equity=None,
                now=datetime(2026, 6, 28, 12, 0, tzinfo=UTC),
            )

        assert result["today_equity_baseline"] is None
        assert result["today_equity_pnl"] is None
        assert result["today_equity_baseline_source"] == "okx_unavailable"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_daily_equity_baseline_replaces_legacy_local_baseline_with_okx_snapshot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'equity-baseline-replace.db').as_posix()}",
    )
    await init_db()
    try:
        async with get_session_ctx() as session:
            session.add(
                ExecutionEquitySnapshot(
                    mode="paper",
                    model_name="ensemble_trader",
                    snapshot_date="2026-06-28",
                    snapshot_at=datetime(2026, 6, 28, 0, 0, tzinfo=UTC),
                    equity=4009.22,
                    total_pnl=9.22,
                    realized_pnl=9.22,
                    unrealized_pnl=0.0,
                    source="estimated",
                )
            )

        async with get_session_ctx() as session:
            result = await apply_daily_equity_baseline(
                session,
                mode="paper",
                model_name="ensemble_trader",
                allocated=0.0,
                positions=[],
                realized_pnl=9.22,
                unrealized_pnl=0.0,
                total_pnl=9.22,
                current_equity=4998.15,
                now=datetime(2026, 6, 28, 12, 0, tzinfo=UTC),
            )

        assert result["today_equity_baseline"] == pytest.approx(4998.15)
        assert result["today_equity_baseline_total_pnl"] is None
        assert result["today_equity_baseline_source"] == "okx_snapshot"
        assert result["today_equity_pnl"] == pytest.approx(0.0)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_daily_equity_baseline_rebuilds_stale_phase3_okx_snapshot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'equity-baseline-stale-okx.db').as_posix()}",
    )
    await init_db()
    try:
        async with get_session_ctx() as session:
            session.add(
                ExecutionEquitySnapshot(
                    mode="paper",
                    model_name="ensemble_trader",
                    snapshot_date="2026-06-28",
                    snapshot_at=datetime(2026, 6, 28, 0, 0, tzinfo=UTC),
                    equity=4000.0,
                    total_pnl=0.0,
                    realized_pnl=0.0,
                    unrealized_pnl=0.0,
                    source="okx_snapshot",
                )
            )

        async with get_session_ctx() as session:
            result = await apply_daily_equity_baseline(
                session,
                mode="paper",
                model_name="ensemble_trader",
                allocated=0.0,
                positions=[],
                realized_pnl=9.22,
                unrealized_pnl=0.0,
                total_pnl=9.22,
                current_equity=4998.15,
                now=datetime(2026, 6, 28, 12, 0, tzinfo=UTC),
            )

        assert result["today_equity_baseline"] == pytest.approx(4998.15)
        assert result["today_equity_pnl"] == pytest.approx(0.0)
        assert result["today_equity_baseline_source"] == "okx_snapshot"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_phase3_equity_change_uses_first_okx_snapshot_not_fixed_balance(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'phase3-equity-change.db').as_posix()}",
    )
    await init_db()
    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    ExecutionEquitySnapshot(
                        mode="paper",
                        model_name="ensemble_trader",
                        snapshot_date="2026-06-27",
                        snapshot_at=datetime(2026, 6, 27, 0, 0, tzinfo=UTC),
                        equity=4000.0,
                        total_pnl=0.0,
                        realized_pnl=0.0,
                        unrealized_pnl=0.0,
                        source="okx_snapshot",
                    ),
                    ExecutionEquitySnapshot(
                        mode="paper",
                        model_name="ensemble_trader",
                        snapshot_date="2026-06-28",
                        snapshot_at=datetime(2026, 6, 28, 0, 0, tzinfo=UTC),
                        equity=4998.15,
                        total_pnl=0.0,
                        realized_pnl=0.0,
                        unrealized_pnl=0.0,
                        source="okx_snapshot",
                    ),
                ]
            )

        async with get_session_ctx() as session:
            result = await phase3_equity_change_from_snapshots(
                session,
                mode="paper",
                model_name="ensemble_trader",
                current_equity=4999.15,
            )

        assert result["phase3_equity_baseline"] == pytest.approx(4998.15)
        assert result["phase3_equity_pnl"] == pytest.approx(1.0)
        assert result["phase3_equity_start_date"] == "2026-06-28"
    finally:
        await close_db()
