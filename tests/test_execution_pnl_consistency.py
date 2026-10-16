from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.trade import Position
from services.okx_position_history_store import upsert_okx_position_history_row
from web_dashboard.api import dashboard


class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        value = datetime(2026, 10, 16, 4, tzinfo=UTC)
        return value.astimezone(tz) if tz else value.replace(tzinfo=None)


async def _value(value):
    return value


@pytest.fixture
async def pnl_database(tmp_path, monkeypatch):
    await close_db()
    monkeypatch.setattr(
        settings, "database_url", f"sqlite+aiosqlite:///{(tmp_path / 'pnl.db').as_posix()}"
    )
    monkeypatch.setattr(dashboard, "datetime", FrozenDatetime)
    monkeypatch.setattr(dashboard, "_dashboard_heavy_cache", {})
    monkeypatch.setattr(dashboard, "_dashboard_heavy_cache_locks", {})
    monkeypatch.setattr(dashboard, "_dashboard_okx_position_error_cache", {})
    monkeypatch.setattr(dashboard, "_trading_service", None)
    await init_db()
    yield
    await close_db()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["paper", "live"])
async def test_dashboard_and_daily_pnl_share_official_settlements_and_beijing_days(
    pnl_database, monkeypatch, mode
):
    snapshot = {"equity": 1005.0, "total": 1005.0, "free": 900.0}
    marks = {
        ("BTC/USDT", "long"): {
            "quantity": 1.0,
            "entry_price": 100.0,
            "upl": 1.25,
            "margin_used": 20.0,
        }
    }
    monkeypatch.setattr(
        dashboard, "_dashboard_okx_balance_snapshot_for_mode", lambda _: _value(snapshot)
    )
    monkeypatch.setattr(
        dashboard, "_get_dashboard_okx_account_snapshot", lambda _: _value(snapshot)
    )
    monkeypatch.setattr(
        dashboard, "_get_exchange_position_mark_map", lambda _: _value(marks)
    )
    monkeypatch.setattr(
        dashboard, "_get_exchange_open_position_symbols", lambda _: _value({"BTC/USDT"})
    )

    settlements = [
        ("older", "2026-10-13T08:00:00+00:00", 10.0),
        ("before-midnight", "2026-10-15T15:59:59+00:00", -2.0),
        ("after-midnight", "2026-10-15T16:00:00+00:00", 4.0),
    ]
    async with get_session_ctx() as session:
        for pos_id, timestamp, pnl in settlements:
            closed_at = datetime.fromisoformat(timestamp)
            await upsert_okx_position_history_row(
                session,
                mode=mode,
                source="okx_settlement_fact_mirror",
                row={
                    "instId": "BTC-USDT-SWAP",
                    "posId": pos_id,
                    "posSide": "long",
                    "type": "2",
                    "openAvgPx": "100",
                    "closeAvgPx": "104",
                    "openMaxPos": "1",
                    "closeTotalPos": "1",
                    "realizedPnl": str(pnl),
                    "pnl": str(pnl),
                    "cTime": str(int((closed_at - timedelta(hours=1)).timestamp() * 1000)),
                    "uTime": str(int(closed_at.timestamp() * 1000)),
                },
            )
        for model_name in ("ensemble_trader", "okx_authoritative_sync"):
            session.add(
                Position(
                    model_name=model_name,
                    execution_mode=mode,
                    symbol="BTC/USDT",
                    side="long",
                    quantity=1.0,
                    entry_price=100.0,
                    realized_pnl=9999.0,
                    settlement_status="reconciled",
                    is_open=False,
                    okx_pos_id="after-midnight",
                    okx_inst_id="BTC-USDT-SWAP",
                    created_at=datetime(2026, 10, 15, 15, tzinfo=UTC),
                    closed_at=datetime(2026, 10, 15, 16, tzinfo=UTC),
                )
            )

    summary = await dashboard._get_execution_pnl_summary(mode)
    account = dashboard._build_execution_account_status(
        mode=mode,
        paper_summary={"positions": [{}]},
        pnl_summary=summary,
        okx_account=snapshot,
    )
    for days in (1, 30):
        payload = await dashboard.get_daily_pnl_records(mode=mode, days=days)
        today = payload["records"][0]
        assert today["date"] == "2026-10-16"
        assert today["realized_pnl"] == pytest.approx(4.0)
        assert today["trade_count"] == 1
        assert today["daily_total_pnl"] == pytest.approx(5.25)
        assert today["cumulative_total_pnl"] == pytest.approx(13.25)
        assert account["today_trade_total_pnl"] == today["daily_total_pnl"]
        assert account["cumulative_trade_total_pnl"] == today["cumulative_total_pnl"]
        assert payload["today_trade_total_pnl"] == today["daily_total_pnl"]
        assert payload["cumulative_trade_total_pnl"] == today["cumulative_total_pnl"]
        assert payload["trade_pnl_start_date"] == account["trade_pnl_start_date"] == "2026-10-13"
    assert summary["realized_pnl"] == pytest.approx(12.0)
    assert summary["today_equity_pnl"] == pytest.approx(0.0)
    assert summary["today_trade_total_pnl"] != summary["today_equity_pnl"]


@pytest.mark.asyncio
async def test_both_pnl_views_share_local_marks_only_during_okx_outage(
    pnl_database, monkeypatch
):
    monkeypatch.setattr(
        dashboard, "_dashboard_okx_balance_snapshot_for_mode", lambda _: _value({})
    )
    monkeypatch.setattr(
        dashboard, "_get_dashboard_okx_account_snapshot", lambda _: _value({})
    )
    monkeypatch.setattr(
        dashboard, "_get_exchange_position_mark_map", lambda _: _value({})
    )
    monkeypatch.setattr(
        dashboard, "_get_exchange_open_position_symbols", lambda _: _value(None)
    )
    monkeypatch.setattr(
        dashboard, "_dashboard_okx_positions_temporarily_unavailable", lambda _: True
    )
    async with get_session_ctx() as session:
        session.add(
            Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="BTC/USDT",
                side="long",
                quantity=1.0,
                entry_price=100.0,
                leverage=5.0,
                unrealized_pnl=2.5,
                is_open=True,
                created_at=datetime(2026, 10, 15, 16, tzinfo=UTC),
            )
        )
    summary = await dashboard._get_execution_pnl_summary("paper")
    payload = await dashboard.get_daily_pnl_records(mode="paper", days=1)
    assert summary["today_trade_total_pnl"] == payload["records"][0]["daily_total_pnl"] == 2.5

    monkeypatch.setattr(
        dashboard, "_get_exchange_open_position_symbols", lambda _: _value(set())
    )
    summary = await dashboard._get_execution_pnl_summary("paper")
    payload = await dashboard.get_daily_pnl_records(mode="paper", days=1)
    assert summary["today_trade_total_pnl"] == payload["records"][0]["daily_total_pnl"] == 0.0


@pytest.mark.asyncio
async def test_unavailable_ledger_does_not_show_a_zero_trading_result(pnl_database, monkeypatch):
    async def unavailable(*args, **kwargs):
        raise TimeoutError("settlement ledger unavailable")

    monkeypatch.setattr(dashboard, "_dashboard_closed_position_ledger_rows", unavailable)
    monkeypatch.setattr(
        dashboard, "_dashboard_okx_balance_snapshot_for_mode", lambda _: _value({})
    )
    monkeypatch.setattr(
        dashboard, "_get_exchange_position_mark_map", lambda _: _value({})
    )
    monkeypatch.setattr(
        dashboard, "_get_exchange_open_position_symbols", lambda _: _value(set())
    )
    summary = await dashboard._get_execution_pnl_summary("paper")
    assert summary["trade_pnl_available"] is False
    assert summary["today_trade_total_pnl"] is None
    assert summary["cumulative_trade_total_pnl"] is None
