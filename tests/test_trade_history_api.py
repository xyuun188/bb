from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from config.settings import settings
from db.repositories.trade_repo import TradeRepository
from db.session import close_db, get_session_ctx, init_db
from models.account import OkxAccountBill
from models.decision import AIDecision
from services.okx_order_fact_sync import OKX_SYNC_CONFIRMED, OKX_SYNC_EXECUTION_RESULT_CONFIRMED
from services.okx_position_history_store import upsert_okx_position_history_row
from web_dashboard.api.dashboard import get_positions as get_dashboard_positions
from web_dashboard.api.trades import (
    _execution_status_label,
    _readable_execution_reason,
    _repair_position_reason_hold_hours,
    _translate_execution_text,
    get_trade_detail,
    get_trades,
)
from web_dashboard.api.trades import get_positions as get_trade_positions


@pytest.fixture(autouse=True)
def _isolate_dashboard_exchange_position_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    from web_dashboard.api import dashboard as dashboard_api

    async def empty_exchange_mark_map(_mode):
        return {}

    async def empty_position_history_rows(*_args, **_kwargs):
        return []

    monkeypatch.setattr(dashboard_api, "_exchange_mark_cache", {})
    monkeypatch.setattr(dashboard_api, "_exchange_open_symbol_cache", {})
    monkeypatch.setattr(dashboard_api, "_dashboard_okx_position_cache", {})
    monkeypatch.setattr(dashboard_api, "_dashboard_okx_position_error_cache", {})
    monkeypatch.setattr(dashboard_api, "_get_exchange_position_mark_map", empty_exchange_mark_map)
    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_okx_position_history_rows",
        empty_position_history_rows,
        raising=False,
    )


async def _seed_okx_position_history_rows(
    rows,
    *,
    entry_order_ids=None,
    close_order_ids=None,
) -> None:
    async with get_session_ctx() as session:
        for row in rows:
            await upsert_okx_position_history_row(
                session,
                row,
                mode="paper",
                source="okx_settlement_fact_mirror",
                entry_order_ids=(entry_order_ids or {}).get(row.get("posId")),
                close_order_ids=(close_order_ids or {}).get(row.get("posId")),
                match_status="test_seed",
            )


@pytest.mark.asyncio
async def test_trade_history_uses_matched_position_leverage_for_close_order(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trade-history.db').as_posix()}",
    )
    await init_db()
    filled_at = datetime(2026, 6, 23, 6, 55, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="WLFI/USDT",
                action="close_short",
                confidence=0.88,
                reasoning="fast exit",
                position_size_pct=1.0,
                suggested_leverage=1.0,
                raw_llm_response={},
                is_paper=True,
                was_executed=True,
                created_at=filled_at,
            )
            session.add(decision)
            await session.flush()
            repo = TradeRepository(session)
            order = await repo.create_order(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "WLFI/USDT",
                    "side": "buy",
                    "order_type": "market",
                    "quantity": 152.0,
                    "price": 0.1226,
                    "status": "filled",
                    "fee": 0.01,
                    "decision_id": decision.id,
                    "exchange_order_id": "close-order-1",
                    "filled_at": filled_at,
                    "created_at": filled_at,
                }
            )
            await repo.open_position(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "WLFI/USDT",
                    "side": "short",
                    "quantity": 152.0,
                    "entry_price": 0.1223,
                    "current_price": 0.1226,
                    "leverage": 3.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": -0.0556,
                    "settlement_status": "reconciled",
                    "is_open": False,
                    "closed_at": filled_at + timedelta(seconds=1),
                    "created_at": filled_at - timedelta(minutes=14),
                }
            )

        trades = await get_trades(model_name=None, symbol=None, mode="paper", limit=10, page=1)
        detail = await get_trade_detail(order.id)
    finally:
        await close_db()

    assert trades["trades"][0]["action"] == "close_short"
    assert trades["trades"][0]["leverage"] == pytest.approx(3.0)
    assert trades["trades"][0]["actual_leverage"] == pytest.approx(3.0)
    assert trades["trades"][0]["ai_suggested_leverage"] == pytest.approx(1.0)
    assert trades["trades"][0]["execution_source"] == "system"
    assert trades["trades"][0]["execution_source_label"] == "系统执行"
    assert detail["matched_positions"][0]["leverage"] == pytest.approx(3.0)
    assert detail["execution_source"] == "system"
    assert detail["execution_source_label"] == "系统执行"


@pytest.mark.asyncio
async def test_trade_history_accepts_okx_execution_result_confirmed_order(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trade-history-execution-confirmed.db').as_posix()}",
    )
    await init_db()
    filled_at = datetime(2026, 7, 1, 6, 46, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            order = await repo.create_order(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "ACT/USDT",
                    "side": "buy",
                    "order_type": "market",
                    "quantity": 1580.0,
                    "price": 0.0097,
                    "status": "filled",
                    "fee": 0.007663,
                    "exchange_order_id": "3703940352525967360",
                    "filled_at": filled_at,
                    "created_at": filled_at,
                    "okx_inst_id": "ACT-USDT-SWAP",
                    "okx_trade_ids": "535631715",
                    "okx_fill_contracts": 158.0,
                    "okx_fill_pnl": 0.4582,
                    "okx_sync_status": OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
                    "okx_raw_fills": {
                        "source": "okx_execution_result",
                        "fills_history_confirmed": False,
                        "execution_result_confirmed": True,
                        "order_id": "3703940352525967360",
                        "trade_ids": ["535631715"],
                        "inst_id": "ACT-USDT-SWAP",
                        "contracts": 158.0,
                        "base_quantity": 1580.0,
                        "avg_price": 0.0097,
                        "fee_abs": 0.007663,
                        "fill_pnl": 0.4582,
                        "timestamp": filled_at.isoformat(),
                    },
                }
            )

        trades = await get_trades(model_name=None, symbol=None, mode="paper", limit=10, page=1)
        detail = await get_trade_detail(order.id)
    finally:
        await close_db()

    row = next(item for item in trades["trades"] if item["id"] == order.id)
    assert row["okx_confirmed"] is True
    assert row["success"] is True
    assert detail["okx_confirmed"] is True
    assert detail["success"] is True
    assert detail["okx_sync_status"] == OKX_SYNC_EXECUTION_RESULT_CONFIRMED


@pytest.mark.asyncio
async def test_dashboard_position_history_marks_same_open_group_slices_as_partial(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-positions.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 6, 24, 5, 20, tzinfo=UTC)
    closed_at = datetime(2026, 6, 24, 5, 26, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            full_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="USAR/USDT",
                action="close_long",
                confidence=1.0,
                reasoning="exchange sync",
                position_size_pct=1.0,
                suggested_leverage=6.0,
                raw_llm_response={},
                is_paper=True,
                was_executed=True,
                created_at=closed_at,
            )
            session.add(full_decision)
            await session.flush()
            for offset, quantity, price in [
                (0, 10.0, 3.85),
                (20, 6.0, 4.26),
            ]:
                await repo.open_position(
                    {
                        "model_name": "ensemble_trader",
                        "execution_mode": "paper",
                        "symbol": "USAR/USDT",
                        "side": "long",
                        "quantity": quantity,
                        "entry_price": 2.31,
                        "current_price": price,
                        "leverage": 6.0,
                        "unrealized_pnl": 0.0,
                        "realized_pnl": (price - 2.31) * quantity,
                        "settlement_status": "reconciled",
                        "is_open": False,
                        "closed_at": closed_at + timedelta(seconds=offset),
                        "created_at": opened_at,
                    }
                )
                await repo.create_order(
                    {
                        "model_name": "ensemble_trader",
                        "execution_mode": "paper",
                        "symbol": "USAR/USDT",
                        "side": "sell",
                        "order_type": "market",
                        "quantity": quantity,
                        "price": price,
                        "status": "filled",
                        "fee": 0.0,
                        "decision_id": full_decision.id,
                        "exchange_order_id": f"usar-close-{int(quantity)}",
                        "filled_at": closed_at + timedelta(seconds=offset),
                        "created_at": closed_at + timedelta(seconds=offset),
                    }
                )

        payload = await get_dashboard_positions(mode="paper", closed_only=True)
    finally:
        await close_db()

    assert payload["ledger_source"] == "okx_positions_history_official_unavailable"
    assert payload["positions"] == []


@pytest.mark.asyncio
async def test_dashboard_position_history_uses_synced_position_realized_pnl(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-realized-pnl.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 6, 24, 10, 38, tzinfo=UTC)
    closed_at = datetime(2026, 6, 24, 10, 47, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="WLFI/USDT",
                action="close_short",
                confidence=1.0,
                reasoning="exchange sync",
                position_size_pct=1.0,
                suggested_leverage=4.0,
                raw_llm_response={},
                is_paper=True,
                was_executed=True,
                created_at=closed_at,
            )
            session.add(decision)
            await session.flush()
            await repo.open_position(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "WLFI/USDT",
                    "side": "short",
                    "quantity": 1964.0,
                    "entry_price": 0.0818,
                    "current_price": 0.08149516293279023,
                    "leverage": 4.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.43834414999998383,
                    "settlement_status": "reconciled",
                    "is_open": False,
                    "closed_at": closed_at,
                    "created_at": opened_at,
                }
            )
            await repo.create_order(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "WLFI/USDT",
                    "side": "buy",
                    "order_type": "market",
                    "quantity": 1964.0,
                    "price": 0.08149516293279023,
                    "status": "filled",
                    "fee": 0.08002825,
                    "decision_id": decision.id,
                    "exchange_order_id": "wlfi-close",
                    "filled_at": closed_at,
                    "created_at": closed_at,
                }
            )

        payload = await get_dashboard_positions(mode="paper", closed_only=True)
    finally:
        await close_db()

    assert payload["ledger_source"] == "okx_positions_history_official_unavailable"
    assert payload["positions"] == []


@pytest.mark.asyncio
async def test_dashboard_position_history_prefers_closed_position_settlement_snapshot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-settlement-snapshot.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 7, 5, 1, 0, tzinfo=UTC)
    closed_at = datetime(2026, 7, 5, 1, 20, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            await repo.open_position(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "SNAP/USDT",
                    "side": "long",
                    "quantity": 10.0,
                    "entry_price": 1.0,
                    "current_price": 1.4,
                    "leverage": 2.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 3.67,
                    "close_fill_pnl": 4.0,
                    "entry_fee": 0.1,
                    "close_fee": 0.2,
                    "funding_fee": -0.03,
                    "settlement_status": "reconciled",
                    "settlement_source": "okx_position_history_settlement",
                    "settlement_synced_at": closed_at,
                    "settlement_raw": {
                        "formula": "close_fill_pnl + funding_fee - entry_fee - close_fee"
                    },
                    "is_open": False,
                    "okx_inst_id": "SNAP-USDT-SWAP",
                    "entry_exchange_order_id": "snap-entry",
                    "close_exchange_order_id": "snap-close",
                    "closed_at": closed_at,
                    "created_at": opened_at,
                }
            )

        payload = await get_dashboard_positions(mode="paper", closed_only=True)
    finally:
        await close_db()

    assert payload["ledger_source"] == "okx_positions_history_official_unavailable"
    assert payload["positions"] == []


@pytest.mark.asyncio
async def test_dashboard_position_history_uses_persisted_okx_official_snapshot_when_live_empty(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-okx-official-stale-cache.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 7, 5, 1, 0, tzinfo=UTC)
    closed_at = datetime(2026, 7, 5, 1, 20, tzinfo=UTC)
    official_row = {
        "instId": "BAND-USDT-SWAP",
        "posId": "3714144287308087298",
        "posSide": "net",
        "openAvgPx": "1.0",
        "closeAvgPx": "1.12",
        "openMaxPos": "20",
        "closeTotalPos": "20",
        "realizedPnl": "2.34",
        "pnl": "2.34",
        "fundingFee": "-0.01",
        "type": "2",
        "cTime": str(int(opened_at.timestamp() * 1000)),
        "uTime": str(int(closed_at.timestamp() * 1000)),
    }
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            for index in range(2):
                await repo.open_position(
                    {
                        "model_name": "okx_authoritative_sync",
                        "execution_mode": "paper",
                        "symbol": "BAND/USDT",
                        "side": "long",
                        "quantity": 20.0,
                        "entry_price": 1.0,
                        "current_price": 1.12,
                        "leverage": 2.0,
                        "realized_pnl": -999.0,
                        "settlement_status": "reconciled",
                        "settlement_source": "okx_position_history_settlement",
                        "settlement_raw": {
                            "okx_position_history_row": dict(official_row),
                            "source": "okx_position_history_settlement",
                        },
                        "is_open": False,
                        "okx_inst_id": "BAND-USDT-SWAP",
                        "okx_pos_id": "band-pos-3714144287308087298",
                        "entry_exchange_order_id": f"local-entry-{index}",
                        "close_exchange_order_id": f"local-close-{index}",
                        "closed_at": closed_at,
                        "created_at": opened_at,
                    }
                )
        payload = await get_dashboard_positions(mode="paper", closed_only=True)
    finally:
        await close_db()

    assert payload["ledger_source"] == "okx_position_history_snapshot_backfill_pending"
    assert payload["total"] == 1
    row = payload["positions"][0]
    assert row["symbol"] == "BAND/USDT"
    assert row["okx_pos_id"] == "3714144287308087298"
    assert row["realized_pnl"] == pytest.approx(2.34)
    assert row["funding_fee"] == pytest.approx(-0.01)
    assert row["quantity"] == pytest.approx(20.0)
    assert len(row["position_ids"]) == 2


@pytest.mark.asyncio
async def test_dashboard_position_history_prefers_confirmed_okx_close_fill_net_pnl(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-okx-net-pnl.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 7, 1, 0, 10, tzinfo=UTC)
    closed_at = datetime(2026, 7, 1, 0, 22, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            await repo.open_position(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "AI16Z/USDT",
                    "side": "short",
                    "quantity": 100.0,
                    "entry_price": 0.08,
                    "current_price": 0.072,
                    "leverage": 1.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 15.1136,
                    "settlement_status": "reconciled",
                    "is_open": False,
                    "okx_inst_id": "AI16Z-USDT-SWAP",
                    "entry_exchange_order_id": "ai16z-entry-okx",
                    "close_exchange_order_id": "ai16z-close-okx",
                    "closed_at": closed_at,
                    "created_at": opened_at,
                }
            )
            for order_id, side, price, fee, fill_pnl, ts in (
                ("ai16z-entry-okx", "sell", 0.08, 0.01, 0.0, opened_at),
                ("ai16z-close-okx", "buy", 0.072, 0.01, 7.95, closed_at),
            ):
                await repo.create_order(
                    {
                        "model_name": "ensemble_trader",
                        "execution_mode": "paper",
                        "symbol": "AI16Z/USDT",
                        "side": side,
                        "order_type": "market",
                        "quantity": 100.0,
                        "price": price,
                        "status": "filled",
                        "fee": fee,
                        "exchange_order_id": order_id,
                        "filled_at": ts,
                        "created_at": ts,
                        "okx_inst_id": "AI16Z-USDT-SWAP",
                        "okx_trade_ids": f"trade-{order_id}",
                        "okx_fill_contracts": 10.0,
                        "okx_fill_pnl": fill_pnl,
                        "okx_sync_status": OKX_SYNC_CONFIRMED,
                        "okx_raw_fills": {
                            "order_id": order_id,
                            "trade_ids": [f"trade-{order_id}"],
                            "inst_id": "AI16Z-USDT-SWAP",
                            "contracts": 10.0,
                            "contract_size": 10.0,
                            "base_quantity": 100.0,
                            "avg_price": price,
                            "fee_abs": fee,
                            "fill_pnl": fill_pnl,
                            "timestamp": ts.isoformat(),
                            "fills_history_confirmed": True,
                        },
                    }
                )

        payload = await get_dashboard_positions(mode="paper", closed_only=True)
    finally:
        await close_db()

    assert payload["ledger_source"] == "okx_positions_history_official_unavailable"
    assert payload["positions"] == []


@pytest.mark.asyncio
async def test_dashboard_position_history_uses_okx_contract_units_when_local_quantity_differs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-okx-contract-unit-pnl.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 7, 3, 18, 16, tzinfo=UTC)
    closed_at = datetime(2026, 7, 3, 18, 21, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            await repo.open_position(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "PEPE/USDT",
                    "side": "short",
                    "quantity": 1.8,
                    "entry_price": 0.00000257,
                    "current_price": 0.00000257,
                    "leverage": 1.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.0,
                    "settlement_status": "reconciled",
                    "is_open": False,
                    "okx_inst_id": "PEPE-USDT-SWAP",
                    "entry_exchange_order_id": "pepe-entry",
                    "close_exchange_order_id": "pepe-close",
                    "closed_at": closed_at,
                    "created_at": opened_at,
                }
            )
            for order_id, side, fee, fill_pnl, ts in (
                ("pepe-entry", "sell", 0.023094, 0.0, opened_at),
                ("pepe-close", "buy", 0.023166, -0.144, closed_at),
            ):
                await repo.create_order(
                    {
                        "model_name": "ensemble_trader",
                        "execution_mode": "paper",
                        "symbol": "PEPE/USDT",
                        "side": side,
                        "order_type": "market",
                        "quantity": 18_000_000.0,
                        "price": 0.00000257,
                        "status": "filled",
                        "fee": fee,
                        "exchange_order_id": order_id,
                        "filled_at": ts,
                        "created_at": ts,
                        "okx_inst_id": "PEPE-USDT-SWAP",
                        "okx_trade_ids": f"trade-{order_id}",
                        "okx_fill_contracts": 1.8,
                        "okx_fill_pnl": fill_pnl,
                        "okx_sync_status": OKX_SYNC_CONFIRMED,
                        "okx_raw_fills": {
                            "order_id": order_id,
                            "trade_ids": [f"trade-{order_id}"],
                            "inst_id": "PEPE-USDT-SWAP",
                            "contracts": 1.8,
                            "base_quantity": 18_000_000.0,
                            "avg_price": 0.00000257,
                            "fee_abs": fee,
                            "fill_pnl": fill_pnl,
                            "timestamp": ts.isoformat(),
                            "fills_history_confirmed": True,
                        },
                    }
                )

        payload = await get_dashboard_positions(mode="paper", closed_only=True)
    finally:
        await close_db()

    assert payload["ledger_source"] == "okx_positions_history_official_unavailable"
    assert payload["positions"] == []


@pytest.mark.asyncio
async def test_dashboard_position_history_prefers_okx_net_pnl_for_fragment_quantity_mismatch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-okx-fragment-pnl.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 7, 2, 1, 14, tzinfo=UTC)
    closed_at = datetime(2026, 7, 3, 11, 47, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            await repo.open_position(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "YFI/USDT",
                    "side": "short",
                    "quantity": 0.0781,
                    "entry_price": 1758.0,
                    "current_price": 1810.0,
                    "leverage": 1.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": -4.2005304,
                    "settlement_status": "reconciled",
                    "is_open": False,
                    "okx_inst_id": "YFI-USDT-SWAP",
                    "entry_exchange_order_id": "yfi-entry",
                    "close_exchange_order_id": "yfi-close",
                    "closed_at": closed_at,
                    "created_at": opened_at,
                }
            )
            for order_id, side, price, fee, fill_pnl, ts in (
                ("yfi-entry", "sell", 1758.0, 0.1372998, 0.0, opened_at),
                ("yfi-close", "buy", 1810.0, 0.141361, -8.1224, closed_at),
            ):
                await repo.create_order(
                    {
                        "model_name": "ensemble_trader",
                        "execution_mode": "paper",
                        "symbol": "YFI/USDT",
                        "side": side,
                        "order_type": "market",
                        "quantity": 0.1562,
                        "price": price,
                        "status": "filled",
                        "fee": fee,
                        "exchange_order_id": order_id,
                        "filled_at": ts,
                        "created_at": ts,
                        "okx_inst_id": "YFI-USDT-SWAP",
                        "okx_trade_ids": f"trade-{order_id}",
                        "okx_fill_contracts": 1562.0,
                        "okx_fill_pnl": fill_pnl,
                        "okx_sync_status": OKX_SYNC_CONFIRMED,
                        "okx_raw_fills": {
                            "order_id": order_id,
                            "trade_ids": [f"trade-{order_id}"],
                            "inst_id": "YFI-USDT-SWAP",
                            "contracts": 1562.0,
                            "base_quantity": 0.1562,
                            "avg_price": price,
                            "fee_abs": fee,
                            "fill_pnl": fill_pnl,
                            "timestamp": ts.isoformat(),
                            "fills_history_confirmed": True,
                        },
                    }
                )

        payload = await get_dashboard_positions(mode="paper", closed_only=True)
    finally:
        await close_db()

    assert payload["ledger_source"] == "okx_positions_history_official_unavailable"
    assert payload["positions"] == []


@pytest.mark.asyncio
async def test_dashboard_position_history_uses_okx_grouped_ledger_with_linked_fills(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web_dashboard.api import dashboard as dashboard_api

    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-okx-ledger.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 6, 28, 0, 38, tzinfo=UTC)
    closed_at = datetime(2026, 6, 28, 12, 40, tzinfo=UTC)

    async def official_position_history_rows(*_args, **_kwargs):
        return [
            {
                "instId": "INJ-USDT-SWAP",
                "posId": "inj-pos",
                "posSide": "net",
                "openAvgPx": "4.813",
                "closeAvgPx": "4.758",
                "openMaxPos": "381.2",
                "closeTotalPos": "381.2",
                "realizedPnl": "19.63",
                "pnl": "19.63",
                "fundingFee": "0",
                "type": "2",
                "cTime": str(int(opened_at.timestamp() * 1000)),
                "uTime": str(int(closed_at.timestamp() * 1000)),
            }
        ]

    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_okx_position_history_rows",
        official_position_history_rows,
    )
    await _seed_okx_position_history_rows(
        await official_position_history_rows(),
        entry_order_ids={"inj-pos": ["inj-entry"]},
        close_order_ids={"inj-pos": ["inj-close-a", "inj-close-b"]},
    )
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            await repo.open_position(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "INJ/USDT",
                    "side": "long",
                    "quantity": 381.2,
                    "entry_price": 4.813,
                    "current_price": 4.758,
                    "leverage": 2.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 19.63,
                    "settlement_status": "reconciled",
                    "is_open": False,
                    "okx_inst_id": "INJ-USDT-SWAP",
                    "entry_exchange_order_id": "inj-entry",
                    "close_exchange_order_id": "inj-close-a,inj-close-b",
                    "closed_at": closed_at,
                    "created_at": opened_at,
                }
            )
            await repo.create_order(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "INJ/USDT",
                    "side": "buy",
                    "order_type": "market",
                    "quantity": 381.2,
                    "price": 4.813,
                    "status": "filled",
                    "fee": 0.1,
                    "exchange_order_id": "inj-entry",
                    "filled_at": opened_at,
                    "created_at": opened_at,
                    "okx_inst_id": "INJ-USDT-SWAP",
                    "okx_trade_ids": "trade-entry",
                    "okx_fill_contracts": 381.2,
                    "okx_sync_status": OKX_SYNC_CONFIRMED,
                    "okx_raw_fills": {
                        "order_id": "inj-entry",
                        "trade_ids": ["trade-entry"],
                        "inst_id": "INJ-USDT-SWAP",
                        "contracts": 381.2,
                        "contract_size": 1.0,
                        "base_quantity": 381.2,
                        "avg_price": 4.813,
                        "fee_abs": 0.1,
                        "timestamp": opened_at.isoformat(),
                    },
                }
            )
            for order_id, qty, price, pnl, minutes in (
                ("inj-close-a", 20.2, 4.713, -0.03, 2),
                ("inj-close-b", 361.0, 4.758, 19.66, 0),
            ):
                await repo.create_order(
                    {
                        "model_name": "ensemble_trader",
                        "execution_mode": "paper",
                        "symbol": "INJ/USDT",
                        "side": "sell",
                        "order_type": "market",
                        "quantity": qty,
                        "price": price,
                        "status": "filled",
                        "fee": 0.05,
                        "exchange_order_id": order_id,
                        "filled_at": closed_at - timedelta(minutes=minutes),
                        "created_at": closed_at - timedelta(minutes=minutes),
                        "okx_inst_id": "INJ-USDT-SWAP",
                        "okx_trade_ids": f"trade-{order_id}",
                        "okx_fill_contracts": qty,
                        "okx_fill_pnl": pnl,
                        "okx_sync_status": OKX_SYNC_CONFIRMED,
                        "okx_raw_fills": {
                            "order_id": order_id,
                            "trade_ids": [f"trade-{order_id}"],
                            "inst_id": "INJ-USDT-SWAP",
                            "contracts": qty,
                            "contract_size": 1.0,
                            "base_quantity": qty,
                            "avg_price": price,
                            "fee_abs": 0.05,
                            "fill_pnl": pnl,
                            "timestamp": (closed_at - timedelta(minutes=minutes)).isoformat(),
                        },
                    }
                )

        payload = await get_dashboard_positions(mode="paper", closed_only=True)
    finally:
        await close_db()

    assert payload["ledger_source"] == "okx_settlement_fact_mirror"
    assert payload["total"] == 1
    row = payload["positions"][0]
    assert row["symbol"] == "INJ/USDT"
    assert row["quantity"] == pytest.approx(381.2)
    assert row["average_entry_price"] == pytest.approx(4.813)
    assert row["linked_order_count"] == 3
    assert row["evidence_complete"] is True
    assert row["trainable"] is True
    assert {item["order_id"] for item in row["linked_fills"]} == {
        "inj-entry",
        "inj-close-a",
        "inj-close-b",
    }


@pytest.mark.asyncio
async def test_dashboard_position_history_uses_okx_official_rows_over_polluted_local_fragments(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web_dashboard.api import dashboard as dashboard_api

    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-okx-official-authority.db').as_posix()}",
    )
    await init_db()

    def ms(value: datetime) -> str:
        return str(int(value.timestamp() * 1000))

    axs_opened = datetime(2026, 7, 9, 6, 58, 39, tzinfo=UTC)
    axs_closed = datetime(2026, 7, 9, 18, 11, 59, tzinfo=UTC)
    xpl_opened = datetime(2026, 7, 9, 11, 13, 17, tzinfo=UTC)
    xpl_updated = datetime(2026, 7, 9, 17, 50, 0, tzinfo=UTC)

    async def official_position_history_rows(*_args, **_kwargs):
        return [
            {
                "instId": "AXS-USDT-SWAP",
                "posId": "axs-pos",
                "posSide": "net",
                "openAvgPx": "1.0",
                "closeAvgPx": "0.6",
                "openMaxPos": "293.1",
                "closeTotalPos": "293.1",
                "realizedPnl": "116.14",
                "pnl": "116.42",
                "fee": "-0.28",
                "fundingFee": "0",
                "lever": "1",
                "type": "2",
                "cTime": ms(axs_opened),
                "uTime": ms(axs_closed),
            },
            {
                "instId": "XPL-USDT-SWAP",
                "posId": "xpl-pos",
                "posSide": "net",
                "openAvgPx": "0.1",
                "closeAvgPx": "0.104",
                "openMaxPos": "420",
                "closeTotalPos": "340",
                "realizedPnl": "1.10",
                "pnl": "1.31",
                "fee": "-0.21",
                "fundingFee": "0",
                "lever": "1",
                "type": "1",
                "cTime": ms(xpl_opened),
                "uTime": ms(xpl_updated),
            },
        ]

    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_okx_position_history_rows",
        official_position_history_rows,
    )
    await _seed_okx_position_history_rows(
        await official_position_history_rows(),
        entry_order_ids={
            "axs-pos": ["axs-entry-a", "axs-entry-b"],
            "xpl-pos": ["xpl-entry"],
        },
        close_order_ids={
            "axs-pos": ["axs-close-a", "axs-close-b"],
            "xpl-pos": ["xpl-close-a", "xpl-close-b", "xpl-close-c"],
        },
    )

    async def add_order(
        repo: TradeRepository,
        *,
        symbol: str,
        inst_id: str,
        order_id: str,
        side: str,
        quantity: float,
        price: float,
        filled_at: datetime,
        pnl: float = 0.0,
    ) -> None:
        await repo.create_order(
            {
                "model_name": "okx_authoritative_sync",
                "execution_mode": "paper",
                "symbol": symbol,
                "side": side,
                "order_type": "market",
                "quantity": quantity,
                "price": price,
                "status": "filled",
                "fee": 0.01,
                "exchange_order_id": order_id,
                "filled_at": filled_at,
                "created_at": filled_at,
                "okx_inst_id": inst_id,
                "okx_trade_ids": f"trade-{order_id}",
                "okx_fill_contracts": quantity,
                "okx_fill_pnl": pnl,
                "okx_sync_status": OKX_SYNC_CONFIRMED,
                "okx_raw_fills": {
                    "order_id": order_id,
                    "trade_ids": [f"trade-{order_id}"],
                    "inst_id": inst_id,
                    "contracts": quantity,
                    "contract_size": 1.0,
                    "base_quantity": quantity,
                    "avg_price": price,
                    "fee_abs": 0.01,
                    "fill_pnl": pnl,
                    "timestamp": filled_at.isoformat(),
                    "fills_history_confirmed": True,
                },
            }
        )

    async def add_closed_position(
        repo: TradeRepository,
        *,
        symbol: str,
        inst_id: str,
        pos_id: str,
        side: str,
        quantity: float,
        entry_order_id: str,
        close_order_id: str,
        opened_at: datetime,
        closed_at: datetime,
        realized_pnl: float,
    ) -> None:
        await repo.open_position(
            {
                "model_name": "okx_authoritative_sync",
                "execution_mode": "paper",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "entry_price": 1.0,
                "current_price": 1.1,
                "leverage": 1.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": realized_pnl,
                "settlement_status": "reconciled",
                "settlement_source": "okx_position_history_settlement",
                "is_open": False,
                "okx_inst_id": inst_id,
                "okx_pos_id": pos_id,
                "entry_exchange_order_id": entry_order_id,
                "close_exchange_order_id": close_order_id,
                "created_at": opened_at,
                "closed_at": closed_at,
            }
        )

    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            for order_id, qty, ts, pnl in (
                ("axs-entry-a", 201.0, axs_opened, 0.0),
                ("axs-entry-b", 92.1, axs_opened + timedelta(minutes=4), 0.0),
                ("axs-close-a", 201.0, axs_closed - timedelta(minutes=1), 83.39),
                ("axs-close-b", 92.1, axs_closed, 33.03),
            ):
                await add_order(
                    repo,
                    symbol="AXS/USDT",
                    inst_id="AXS-USDT-SWAP",
                    order_id=order_id,
                    side="sell" if "entry" in order_id else "buy",
                    quantity=qty,
                    price=1.0 if "entry" in order_id else 0.6,
                    filled_at=ts,
                    pnl=pnl,
                )
            for qty, entry_id, close_id in (
                (287.48316479, "axs-entry-a", "axs-close-a"),
                (5.61683521, "axs-entry-b", "axs-close-b"),
            ):
                await add_closed_position(
                    repo,
                    symbol="AXS/USDT",
                    inst_id="AXS-USDT-SWAP",
                    pos_id="axs-pos",
                    side="short",
                    quantity=qty,
                    entry_order_id=entry_id,
                    close_order_id=close_id,
                    opened_at=axs_opened,
                    closed_at=axs_closed,
                    realized_pnl=116.14,
                )

            await add_order(
                repo,
                symbol="XPL/USDT",
                inst_id="XPL-USDT-SWAP",
                order_id="xpl-entry",
                side="buy",
                quantity=420.0,
                price=0.1,
                filled_at=xpl_opened,
            )
            for order_id, qty, minutes, pnl in (
                ("xpl-close-a", 180.0, 10, 0.52),
                ("xpl-close-b", 100.0, 20, 0.36),
                ("xpl-close-c", 60.0, 30, 0.22),
                ("xpl-generated-extra-close", 62.67283785, 40, 1.11),
            ):
                await add_order(
                    repo,
                    symbol="XPL/USDT",
                    inst_id="XPL-USDT-SWAP",
                    order_id=order_id,
                    side="sell",
                    quantity=qty,
                    price=0.104,
                    filled_at=xpl_opened + timedelta(minutes=minutes),
                    pnl=pnl,
                )
                await add_closed_position(
                    repo,
                    symbol="XPL/USDT",
                    inst_id="XPL-USDT-SWAP",
                    pos_id="xpl-pos",
                    side="long",
                    quantity=qty,
                    entry_order_id="xpl-entry",
                    close_order_id=order_id,
                    opened_at=xpl_opened,
                    closed_at=xpl_opened + timedelta(minutes=minutes),
                    realized_pnl=pnl,
                )

        payload = await get_dashboard_positions(mode="paper", closed_only=True)
    finally:
        await close_db()

    assert payload["ledger_source"] == "okx_settlement_fact_mirror"
    assert payload["total"] == 2
    by_symbol = {row["symbol"]: row for row in payload["positions"]}

    axs = by_symbol["AXS/USDT"]
    assert axs["quantity"] == pytest.approx(293.1)
    assert axs["realized_pnl"] == pytest.approx(116.14)
    assert axs["linked_order_count"] == 4
    assert set(axs["entry_order_ids"]) == {"axs-entry-a", "axs-entry-b"}
    assert set(axs["close_order_ids"]) == {"axs-close-a", "axs-close-b"}

    xpl = by_symbol["XPL/USDT"]
    assert xpl["close_status"] == "partial"
    assert xpl["quantity"] == pytest.approx(340.0)
    assert xpl["max_position_quantity"] == pytest.approx(420.0)
    assert xpl["realized_pnl"] == pytest.approx(1.10)
    assert set(xpl["close_order_ids"]) == {"xpl-close-a", "xpl-close-b", "xpl-close-c"}
    assert "xpl-generated-extra-close" not in xpl["close_order_ids"]


@pytest.mark.asyncio
async def test_dashboard_position_history_groups_okx_partial_closes_by_position_id(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web_dashboard.api import dashboard as dashboard_api

    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-okx-partial-lifecycle.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 7, 4, 19, 15, 25, tzinfo=UTC)
    entry_order_id = "band-entry"
    pos_id = "band-pos-3714144287308087298"
    entry_fee = 0.0366182
    official_realized_pnl = 1.245640902389385
    partial_closes = (
        ("band-close-203", 203.0, 0.1650, 0.6034309734513242, 0.0167475, 16, 59, 55),
        ("band-close-112", 112.0, 0.1645, 0.2769274336283168, 0.0092120, 17, 16, 10),
        ("band-close-061", 61.0, 0.1663, 0.2606265486725654, 0.00507215, 21, 7, 27),
        ("band-close-034", 34.0, 0.1677, 0.1928672566371676, 0.0028509, 23, 22, 36),
    )

    async def official_position_history_rows(*_args, **_kwargs):
        return [
            {
                "instId": "BAND-USDT-SWAP",
                "posId": pos_id,
                "posSide": "net",
                "openAvgPx": "0.1620274336283186",
                "closeAvgPx": "0.1652807317073171",
                "closeTotalPos": "410",
                "realizedPnl": str(official_realized_pnl),
                "fee": "-0.07050075",
                "fundingFee": "-0.01771056",
                "pnl": "1.333852212389385",
                "lever": "2.0",
                "cTime": "1783192525444",
                "uTime": str(
                    int(datetime(2026, 7, 5, 23, 22, 36, tzinfo=UTC).timestamp() * 1000)
                ),
            }
        ]

    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_okx_position_history_rows",
        official_position_history_rows,
    )
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            await repo.create_order(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "BAND/USDT",
                    "side": "buy",
                    "order_type": "market",
                    "quantity": 452.0,
                    "price": 0.1620274336283186,
                    "status": "filled",
                    "fee": entry_fee,
                    "exchange_order_id": entry_order_id,
                    "filled_at": opened_at,
                    "created_at": opened_at,
                    "okx_inst_id": "BAND-USDT-SWAP",
                    "okx_trade_ids": "band-entry-trade-a,band-entry-trade-b",
                    "okx_fill_contracts": 452.0,
                    "okx_fill_pnl": 0.0,
                    "okx_sync_status": OKX_SYNC_CONFIRMED,
                    "okx_raw_fills": {
                        "order_id": entry_order_id,
                        "trade_ids": ["band-entry-trade-a", "band-entry-trade-b"],
                        "inst_id": "BAND-USDT-SWAP",
                        "contracts": 452.0,
                        "contract_size": 1.0,
                        "base_quantity": 452.0,
                        "avg_price": 0.1620274336283186,
                        "fee_abs": entry_fee,
                        "fill_pnl": 0.0,
                        "timestamp": opened_at.isoformat(),
                        "fills_history_confirmed": True,
                    },
                }
            )
            for order_id, qty, price, fill_pnl, close_fee, hour, minute, second in partial_closes:
                closed_at = datetime(2026, 7, 5, hour, minute, second, tzinfo=UTC)
                allocated_entry_fee = entry_fee * qty / 452.0
                await repo.open_position(
                    {
                        "model_name": "ensemble_trader",
                        "execution_mode": "paper",
                        "symbol": "BAND/USDT",
                        "side": "long",
                        "quantity": qty,
                        "entry_price": 0.1620274336283186,
                        "current_price": price,
                        "leverage": 2.0,
                        "unrealized_pnl": 0.0,
                        "realized_pnl": fill_pnl - allocated_entry_fee - close_fee,
                        "close_fill_pnl": fill_pnl,
                        "entry_fee": allocated_entry_fee,
                        "close_fee": close_fee,
                        "funding_fee": 0.0,
                        "settlement_status": "reconciled",
                        "settlement_source": "okx_position_history_settlement",
                        "is_open": False,
                        "okx_inst_id": "BAND-USDT-SWAP",
                        "okx_pos_id": pos_id,
                        "entry_exchange_order_id": entry_order_id,
                        "close_exchange_order_id": order_id,
                        "closed_at": closed_at,
                        "created_at": opened_at,
                    }
                )
                await repo.create_order(
                    {
                        "model_name": "ensemble_trader",
                        "execution_mode": "paper",
                        "symbol": "BAND/USDT",
                        "side": "sell",
                        "order_type": "market",
                        "quantity": qty,
                        "price": price,
                        "status": "filled",
                        "fee": close_fee,
                        "exchange_order_id": order_id,
                        "filled_at": closed_at,
                        "created_at": closed_at,
                        "okx_inst_id": "BAND-USDT-SWAP",
                        "okx_trade_ids": f"trade-{order_id}",
                        "okx_fill_contracts": qty,
                        "okx_fill_pnl": fill_pnl,
                        "okx_sync_status": OKX_SYNC_CONFIRMED,
                        "okx_raw_fills": {
                            "order_id": order_id,
                            "trade_ids": [f"trade-{order_id}"],
                            "inst_id": "BAND-USDT-SWAP",
                            "contracts": qty,
                            "contract_size": 1.0,
                            "base_quantity": qty,
                            "avg_price": price,
                            "fee_abs": close_fee,
                            "fill_pnl": fill_pnl,
                            "timestamp": closed_at.isoformat(),
                            "fills_history_confirmed": True,
                        },
                    }
                )

        await _seed_okx_position_history_rows(
            await official_position_history_rows(),
            entry_order_ids={pos_id: [entry_order_id]},
            close_order_ids={pos_id: [item[0] for item in partial_closes]},
        )
        payload = await get_dashboard_positions(mode="paper", closed_only=True)
    finally:
        await close_db()

    assert payload["ledger_source"] == "okx_settlement_fact_mirror"
    assert payload["total"] == 1
    row = payload["positions"][0]
    assert row["symbol"] == "BAND/USDT"
    assert row["close_status"] == "partial"
    assert row["close_status_label"] == "部分平仓"
    assert row["quantity"] == pytest.approx(410.0)
    assert row["max_position_quantity"] == pytest.approx(452.0)
    assert row["average_close_price"] == pytest.approx(0.1652807317073171)
    assert row["realized_pnl"] == pytest.approx(official_realized_pnl)
    assert row["funding_fee"] == pytest.approx(-0.01771056)
    assert row["close_fill_pnl"] == pytest.approx(1.333852212389385)
    assert row["pnl_source"] == "okx_position_history_realized_pnl"
    assert row["linked_order_count"] == 5
    assert row["entry_order_ids"] == [entry_order_id]
    assert set(row["close_order_ids"]) == {item[0] for item in partial_closes}
    assert {item["order_id"] for item in row["linked_fills"]} == {
        entry_order_id,
        *(item[0] for item in partial_closes),
    }


@pytest.mark.asyncio
async def test_dashboard_position_history_keeps_reused_posid_lifecycles_and_partial_updates_visible(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web_dashboard.api import dashboard as dashboard_api

    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-okx-visible-partials.db').as_posix()}",
    )
    await init_db()

    def ms(value: datetime) -> str:
        return str(int(value.timestamp() * 1000))

    jup_pos_id = "jup-pos"
    jup_open = datetime(2026, 7, 4, 9, 57, 45, tzinfo=UTC)
    jup_close_a = datetime(2026, 7, 4, 11, 52, 28, tzinfo=UTC)
    jup_close_b = datetime(2026, 7, 4, 12, 8, 59, tzinfo=UTC)
    jup_close_c = datetime(2026, 7, 5, 15, 15, 10, tzinfo=UTC)
    jup_close_d = datetime(2026, 7, 5, 16, 25, 18, tzinfo=UTC)
    icp_pos_id = "reused-icp-pos"
    icp_old_open = datetime(2026, 7, 4, 11, 24, 30, tzinfo=UTC)
    icp_old_close = datetime(2026, 7, 4, 15, 57, 51, tzinfo=UTC)
    icp_new_open = datetime(2026, 7, 5, 1, 9, 32, tzinfo=UTC)
    icp_close_a = datetime(2026, 7, 5, 12, 54, 34, tzinfo=UTC)
    icp_close_b = datetime(2026, 7, 5, 13, 50, 39, tzinfo=UTC)
    icp_close_c = datetime(2026, 7, 5, 16, 21, 38, tzinfo=UTC)
    filler_open = datetime(2026, 7, 5, 0, 0, tzinfo=UTC)
    filler_close = datetime(2026, 7, 5, 1, 0, tzinfo=UTC)

    async def official_position_history_rows(*_args, **_kwargs):
        return [
            {
                "instId": "JUP-USDT-SWAP",
                "posId": jup_pos_id,
                "posSide": "net",
                "openAvgPx": "0.2324363636363637",
                "closeAvgPx": "0.2374444444444444",
                "closeTotalPos": "9",
                "realizedPnl": "0.423237822727263",
                "fee": "-0.023469",
                "fundingFee": "-0.00402045",
                "pnl": "0.450727272727263",
                "cTime": ms(jup_open),
                "uTime": ms(jup_close_d),
            },
            {
                "instId": "ICP-USDT-SWAP",
                "posId": icp_pos_id,
                "posSide": "net",
                "openAvgPx": "2.2060082778193859",
                "closeAvgPx": "2.2451956456042151",
                "closeTotalPos": "7610.7",
                "realizedPnl": "-3.1433918435999959",
                "fee": "-0.1693838885",
                "fundingFee": "0.0084250449",
                "pnl": "-2.9824329999999959",
                "cTime": ms(icp_old_open),
                "uTime": ms(icp_old_close),
            },
            {
                "instId": "ICP-USDT-SWAP",
                "posId": icp_pos_id,
                "posSide": "net",
                "openAvgPx": "2.2201334723380596",
                "closeAvgPx": "2.1942609725186766",
                "closeTotalPos": "2998.4",
                "realizedPnl": "0.7160030259843799",
                "fee": "-0.072825461",
                "fundingFee": "0.0130674524",
                "pnl": "0.7757610345843799",
                "cTime": ms(icp_new_open),
                "uTime": ms(icp_close_c),
            },
        ]

    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_okx_position_history_rows",
        official_position_history_rows,
    )

    async def add_order(
        repo: TradeRepository,
        *,
        symbol: str,
        inst_id: str,
        order_id: str,
        side: str,
        quantity: float,
        contracts: float,
        contract_size: float,
        price: float,
        fee: float,
        filled_at: datetime,
        pnl: float = 0.0,
    ) -> None:
        await repo.create_order(
            {
                "model_name": "ensemble_trader",
                "execution_mode": "paper",
                "symbol": symbol,
                "side": side,
                "order_type": "market",
                "quantity": quantity,
                "price": price,
                "status": "filled",
                "fee": fee,
                "exchange_order_id": order_id,
                "filled_at": filled_at,
                "created_at": filled_at,
                "okx_inst_id": inst_id,
                "okx_trade_ids": f"trade-{order_id}",
                "okx_fill_contracts": contracts,
                "okx_fill_pnl": pnl,
                "okx_sync_status": OKX_SYNC_CONFIRMED,
                "okx_raw_fills": {
                    "order_id": order_id,
                    "trade_ids": [f"trade-{order_id}"],
                    "inst_id": inst_id,
                    "contracts": contracts,
                    "contract_size": contract_size,
                    "base_quantity": quantity,
                    "avg_price": price,
                    "fee_abs": fee,
                    "fill_pnl": pnl,
                    "timestamp": filled_at.isoformat(),
                    "fills_history_confirmed": True,
                },
            }
        )

    async def add_closed_position(
        repo: TradeRepository,
        *,
        model_name: str,
        symbol: str,
        inst_id: str,
        pos_id: str | None,
        side: str,
        quantity: float,
        entry_price: float,
        close_price: float,
        leverage: float,
        entry_order: str,
        close_order: str,
        opened_at: datetime,
        closed_at: datetime,
        realized_pnl: float,
    ) -> None:
        await repo.open_position(
            {
                "model_name": model_name,
                "execution_mode": "paper",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "entry_price": entry_price,
                "current_price": close_price,
                "leverage": leverage,
                "unrealized_pnl": 0.0,
                "realized_pnl": realized_pnl,
                "settlement_status": "reconciled",
                "settlement_source": "okx_position_history_settlement",
                "is_open": False,
                "okx_inst_id": inst_id,
                "okx_pos_id": pos_id,
                "entry_exchange_order_id": entry_order,
                "close_exchange_order_id": close_order,
                "closed_at": closed_at,
                "created_at": opened_at,
            }
        )

    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            for order_id, qty, contracts, price, fee, ts, pnl in (
                ("jup-entry", 110.0, 11.0, 0.23243636363636364, 0.012784, jup_open, 0.0),
                (
                    "jup-close-a",
                    40.0,
                    4.0,
                    0.2359,
                    0.004718,
                    jup_close_a,
                    0.138545454545452,
                ),
                (
                    "jup-close-b",
                    30.0,
                    3.0,
                    0.2367,
                    0.0035505,
                    jup_close_b,
                    0.127909090909089,
                ),
                (
                    "jup-close-c",
                    10.0,
                    1.0,
                    0.2408,
                    0.001204,
                    jup_close_c,
                    0.083636363636363,
                ),
                (
                    "jup-close-d",
                    10.0,
                    1.0,
                    0.2425,
                    0.0012125,
                    jup_close_d,
                    0.100636363636363,
                ),
            ):
                await add_order(
                    repo,
                    symbol="JUP/USDT",
                    inst_id="JUP-USDT-SWAP",
                    order_id=order_id,
                    side="buy" if order_id == "jup-entry" else "sell",
                    quantity=qty,
                    contracts=contracts,
                    contract_size=10.0,
                    price=price,
                    fee=fee,
                    filled_at=ts,
                    pnl=pnl,
                )
            for model_name, qty, close_order, closed_at, close_price, pnl in (
                ("okx_authoritative_sync", 40.0, "jup-close-a", jup_close_a, 0.2359, 0.12),
                ("okx_authoritative_sync", 30.0, "jup-close-b", jup_close_b, 0.2367, 0.12),
                ("ensemble_trader", 10.0, "jup-close-c", jup_close_c, 0.2408, 0.08),
                ("ensemble_trader", 10.0, "jup-close-d", jup_close_d, 0.2425, 0.09),
            ):
                await add_closed_position(
                    repo,
                    model_name=model_name,
                    symbol="JUP/USDT",
                    inst_id="JUP-USDT-SWAP",
                    pos_id=jup_pos_id,
                    side="long",
                    quantity=qty,
                    entry_price=0.23243636363636364,
                    close_price=close_price,
                    leverage=1.0,
                    entry_order="jup-entry",
                    close_order=close_order,
                    opened_at=jup_open,
                    closed_at=closed_at,
                    realized_pnl=pnl,
                )

            for order_id, side, qty, contracts, price, fee, ts, pnl in (
                (
                    "icp-old-entry",
                    "sell",
                    76.107,
                    7610.7,
                    2.2060082778193855,
                    0.083946336,
                    icp_old_open,
                    0.0,
                ),
                (
                    "icp-old-close",
                    "buy",
                    75.433,
                    7543.3,
                    2.245,
                    0.0846735425,
                    icp_old_close,
                    -2.9412625792502634,
                ),
                (
                    "icp-new-entry",
                    "sell",
                    35.97,
                    3597.0,
                    2.22013347233806,
                    0.0399291005,
                    icp_new_open,
                    0.0,
                ),
                (
                    "icp-close-a",
                    "buy",
                    16.186,
                    1618.6,
                    2.1946103422710985,
                    0.0177609815,
                    icp_close_a,
                    0.4131173832638329,
                ),
                (
                    "icp-close-b",
                    "buy",
                    8.902,
                    890.2,
                    2.19596922039991,
                    0.009774259,
                    icp_close_b,
                    0.2151101707534066,
                ),
                (
                    "icp-close-c",
                    "buy",
                    4.896,
                    489.6,
                    2.19,
                    0.00536112,
                    icp_close_c,
                    0.1475334805671398,
                ),
            ):
                await add_order(
                    repo,
                    symbol="ICP/USDT",
                    inst_id="ICP-USDT-SWAP",
                    order_id=order_id,
                    side=side,
                    quantity=qty,
                    contracts=contracts,
                    contract_size=0.01,
                    price=price,
                    fee=fee,
                    filled_at=ts,
                    pnl=pnl,
                )
            await add_closed_position(
                repo,
                model_name="okx_authoritative_sync",
                symbol="ICP/USDT",
                inst_id="ICP-USDT-SWAP",
                pos_id=icp_pos_id,
                side="short",
                quantity=0.674,
                entry_price=2.2060082778193855,
                close_price=2.2670919881305633,
                leverage=2.0,
                entry_order="icp-old-entry",
                close_order="icp-old-close",
                opened_at=icp_old_open,
                closed_at=icp_old_close,
                realized_pnl=-3.143391843599996,
            )
            for qty, close_order, closed_at, close_price, pnl in (
                (16.186, "icp-close-a", icp_close_a, 2.1946103422710985, 0.37),
                (8.902, "icp-close-b", icp_close_b, 2.19596922039991, 0.19),
                (4.896, "icp-close-c", icp_close_c, 2.19, 0.13),
            ):
                await add_closed_position(
                    repo,
                    model_name="ensemble_trader",
                    symbol="ICP/USDT",
                    inst_id="ICP-USDT-SWAP",
                    pos_id=icp_pos_id,
                    side="short",
                    quantity=qty,
                    entry_price=2.22013347233806,
                    close_price=close_price,
                    leverage=1.0,
                    entry_order="icp-new-entry",
                    close_order=close_order,
                    opened_at=icp_new_open,
                    closed_at=closed_at,
                    realized_pnl=pnl,
                )

            await add_order(
                repo,
                symbol="ZZZ/USDT",
                inst_id="ZZZ-USDT-SWAP",
                order_id="filler-entry",
                side="buy",
                quantity=1.0,
                contracts=1.0,
                contract_size=1.0,
                price=1.0,
                fee=0.0,
                filled_at=filler_open,
            )
            await add_order(
                repo,
                symbol="ZZZ/USDT",
                inst_id="ZZZ-USDT-SWAP",
                order_id="filler-close",
                side="sell",
                quantity=1.0,
                contracts=1.0,
                contract_size=1.0,
                price=1.1,
                fee=0.0,
                filled_at=filler_close,
                pnl=0.1,
            )
            await add_closed_position(
                repo,
                model_name="ensemble_trader",
                symbol="ZZZ/USDT",
                inst_id="ZZZ-USDT-SWAP",
                pos_id=None,
                side="long",
                quantity=1.0,
                entry_price=1.0,
                close_price=1.1,
                leverage=1.0,
                entry_order="filler-entry",
                close_order="filler-close",
                opened_at=filler_open,
                closed_at=filler_close,
                realized_pnl=0.1,
            )

        await _seed_okx_position_history_rows(await official_position_history_rows())
        payload = await get_dashboard_positions(
            mode="paper",
            closed_only=True,
            page=1,
            page_size=2,
        )
    finally:
        await close_db()

    rows = payload["positions"]
    assert [row["symbol"] for row in rows] == ["JUP/USDT", "ICP/USDT"]

    jup_row = rows[0]
    assert jup_row["quantity"] == pytest.approx(90.0)
    assert jup_row["max_position_quantity"] == pytest.approx(110.0)
    assert jup_row["realized_pnl"] == pytest.approx(0.423237822727263)
    assert set(jup_row["close_order_ids"]) == {
        "jup-close-a",
        "jup-close-b",
        "jup-close-c",
        "jup-close-d",
    }

    icp_row = rows[1]
    assert icp_row["quantity"] == pytest.approx(29.984)
    assert icp_row["max_position_quantity"] == pytest.approx(35.97)
    assert icp_row["realized_pnl"] == pytest.approx(0.7160030259843799)
    assert icp_row["entry_order_ids"] == ["icp-new-entry"]
    assert set(icp_row["close_order_ids"]) == {
        "icp-close-a",
        "icp-close-b",
        "icp-close-c",
    }
    assert "icp-old-entry" not in icp_row["entry_order_ids"]


@pytest.mark.asyncio
async def test_dashboard_position_history_adds_okx_funding_fee_from_account_bills(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-okx-funding-pnl.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 7, 1, 0, 10, tzinfo=UTC)
    funding_at = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)
    closed_at = datetime(2026, 7, 1, 9, 20, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            await repo.open_position(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "INJ/USDT",
                    "side": "long",
                    "quantity": 20.0,
                    "entry_price": 4.0,
                    "current_price": 4.6,
                    "leverage": 2.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 99.0,
                    "settlement_status": "reconciled",
                    "is_open": False,
                    "okx_inst_id": "INJ-USDT-SWAP",
                    "entry_exchange_order_id": "inj-funding-entry",
                    "close_exchange_order_id": "inj-funding-close",
                    "closed_at": closed_at,
                    "created_at": opened_at,
                }
            )
            for order_id, side, price, fee, fill_pnl, ts in (
                ("inj-funding-entry", "buy", 4.0, 0.1, 0.0, opened_at),
                ("inj-funding-close", "sell", 4.6, 0.2, 10.0, closed_at),
            ):
                await repo.create_order(
                    {
                        "model_name": "ensemble_trader",
                        "execution_mode": "paper",
                        "symbol": "INJ/USDT",
                        "side": side,
                        "order_type": "market",
                        "quantity": 20.0,
                        "price": price,
                        "status": "filled",
                        "fee": fee,
                        "exchange_order_id": order_id,
                        "filled_at": ts,
                        "created_at": ts,
                        "okx_inst_id": "INJ-USDT-SWAP",
                        "okx_trade_ids": f"trade-{order_id}",
                        "okx_fill_contracts": 20.0,
                        "okx_fill_pnl": fill_pnl,
                        "okx_sync_status": OKX_SYNC_CONFIRMED,
                        "okx_raw_fills": {
                            "order_id": order_id,
                            "trade_ids": [f"trade-{order_id}"],
                            "inst_id": "INJ-USDT-SWAP",
                            "pos_side": "long",
                            "contracts": 20.0,
                            "contract_size": 1.0,
                            "base_quantity": 20.0,
                            "avg_price": price,
                            "fee_abs": fee,
                            "fill_pnl": fill_pnl,
                            "timestamp": ts.isoformat(),
                            "fills_history_confirmed": True,
                        },
                    }
                )
            session.add(
                OkxAccountBill(
                    mode="paper",
                    bill_id="inj-funding-bill",
                    inst_id="INJ-USDT-SWAP",
                    pos_side="long",
                    ccy="USDT",
                    bill_type="8",
                    bill_sub_type="173",
                    bill_ts=funding_at,
                    balance_change=-0.2,
                    pnl=-0.2,
                    fee=0.0,
                    funding_fee=-0.2,
                    raw_bill={
                        "billId": "inj-funding-bill",
                        "instId": "INJ-USDT-SWAP",
                        "posSide": "long",
                        "subType": "173",
                        "ts": str(int(funding_at.timestamp() * 1000)),
                        "pnl": "-0.2",
                    },
                )
            )

        payload = await get_dashboard_positions(mode="paper", closed_only=True)
    finally:
        await close_db()

    assert payload["ledger_source"] == "okx_positions_history_official_unavailable"
    assert payload["positions"] == []


@pytest.mark.asyncio
async def test_position_history_prefers_final_okx_realized_pnl_over_local_fragments(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-final-history-pnl.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 6, 30, 23, 57, 8, tzinfo=UTC)
    first_close_at = datetime(2026, 7, 1, 0, 2, 32, tzinfo=UTC)
    second_close_at = datetime(2026, 7, 1, 0, 5, 13, tzinfo=UTC)
    final_close_at = datetime(2026, 7, 1, 0, 5, 56, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            for row in (
                {
                    "model_name": "okx_authoritative_sync",
                    "quantity": 732.0,
                    "realized_pnl": 7.937940890773545,
                    "close_exchange_order_id": "ai16z-close-a,ai16z-close-b,ai16z-close-c",
                    "closed_at": final_close_at,
                },
                {
                    "model_name": "okx_authoritative_sync",
                    "quantity": 423.0,
                    "realized_pnl": 3.7719024514292054,
                    "close_exchange_order_id": "ai16z-close-a",
                    "closed_at": first_close_at,
                },
                {
                    "model_name": "ensemble_trader",
                    "quantity": 423.0,
                    "realized_pnl": 3.403710916475388,
                    "close_exchange_order_id": "ai16z-close-a",
                    "closed_at": first_close_at - timedelta(seconds=20),
                },
            ):
                await repo.open_position(
                    {
                        "execution_mode": "paper",
                        "symbol": "AI16Z/USDT",
                        "side": "short",
                        "entry_price": 0.0737920765027324,
                        "current_price": 0.063397868852459,
                        "leverage": 1.0,
                        "unrealized_pnl": 0.0,
                        "settlement_status": "reconciled",
                        "is_open": False,
                        "okx_inst_id": "AI16Z-USDT-SWAP",
                        "okx_pos_id": "3703019077645344768",
                        "entry_exchange_order_id": "ai16z-entry",
                        "created_at": opened_at,
                        **row,
                    }
                )

            for order_id, side, qty, price, ts, pnl in (
                ("ai16z-entry", "sell", 732.0, 0.0737920765027324, opened_at, 0.0),
                ("ai16z-close-a", "buy", 423.0, 0.0656757446808511, first_close_at, 3.4332083606558053),
                ("ai16z-close-b", "buy", 216.0, 0.0604861111111111, second_close_at, 2.8740885245901984),
                ("ai16z-close-c", "buy", 93.0, 0.0598, final_close_at, 1.3012631147541132),
            ):
                await repo.create_order(
                    {
                        "model_name": "okx_authoritative_sync",
                        "execution_mode": "paper",
                        "symbol": "AI16Z/USDT",
                        "side": side,
                        "order_type": "market",
                        "quantity": qty,
                        "price": price,
                        "status": "filled",
                        "fee": 0.01,
                        "exchange_order_id": order_id,
                        "filled_at": ts,
                        "created_at": ts,
                        "okx_inst_id": "AI16Z-USDT-SWAP",
                        "okx_trade_ids": f"trade-{order_id}",
                        "okx_fill_contracts": qty / 10.0,
                        "okx_fill_pnl": pnl,
                        "okx_sync_status": OKX_SYNC_CONFIRMED,
                        "okx_raw_fills": {
                            "order_id": order_id,
                            "trade_ids": [f"trade-{order_id}"],
                            "inst_id": "AI16Z-USDT-SWAP",
                            "contracts": qty / 10.0,
                            "contract_size": 10.0,
                            "base_quantity": qty,
                            "avg_price": price,
                            "fee_abs": 0.01,
                            "fill_pnl": pnl,
                            "timestamp": ts.isoformat(),
                        },
                    }
                )

        payload = await get_dashboard_positions(mode="paper", closed_only=True)
    finally:
        await close_db()

    assert payload["ledger_source"] == "okx_positions_history_official_unavailable"
    assert payload["positions"] == []


@pytest.mark.asyncio
async def test_position_history_splits_polluted_reused_okx_posid_lifecycles(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'sky-reused-posid-ledger.db').as_posix()}",
    )
    await init_db()
    first_opened_at = datetime(2026, 6, 30, 9, 22, 36, tzinfo=UTC)
    first_closed_at = datetime(2026, 6, 30, 10, 43, 50, tzinfo=UTC)
    second_opened_at = datetime(2026, 6, 30, 10, 48, 57, tzinfo=UTC)
    second_closed_at = datetime(2026, 6, 30, 11, 56, 21, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            await repo.open_position(
                {
                    "model_name": "okx_authoritative_sync",
                    "execution_mode": "paper",
                    "symbol": "SKY/USDT",
                    "side": "short",
                    "quantity": 900.0,
                    "entry_price": 0.0526,
                    "current_price": 0.0529,
                    "leverage": 3.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": -0.0325,
                    "settlement_status": "reconciled",
                    "is_open": False,
                    "okx_inst_id": "SKY-USDT-SWAP",
                    "okx_pos_id": "sky-reused-pos",
                    "entry_exchange_order_id": "sky-entry-a,sky-entry-b",
                    "close_exchange_order_id": "sky-close-a,sky-close-b",
                    "closed_at": second_closed_at,
                    "created_at": first_opened_at,
                }
            )
            await repo.open_position(
                {
                    "model_name": "okx_authoritative_sync",
                    "execution_mode": "paper",
                    "symbol": "SKY/USDT",
                    "side": "short",
                    "quantity": 500.0,
                    "entry_price": 0.05235,
                    "current_price": 0.05286,
                    "leverage": 3.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": -0.280615,
                    "settlement_status": "reconciled",
                    "is_open": False,
                    "okx_inst_id": "SKY-USDT-SWAP",
                    "okx_pos_id": "sky-reused-pos",
                    "entry_exchange_order_id": "sky-entry-b",
                    "close_exchange_order_id": "sky-close-b",
                    "closed_at": first_closed_at,
                    "created_at": first_opened_at,
                }
            )
            await repo.open_position(
                {
                    "model_name": "okx_authoritative_sync",
                    "execution_mode": "paper",
                    "symbol": "SKY/USDT",
                    "side": "short",
                    "quantity": 500.0,
                    "entry_price": 0.05235,
                    "current_price": 0.05286,
                    "leverage": 3.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.11,
                    "settlement_status": "reconciled",
                    "is_open": False,
                    "okx_inst_id": "SKY-USDT-SWAP",
                    "okx_pos_id": "sky-reused-pos",
                    "entry_exchange_order_id": "sky-entry-b",
                    "close_exchange_order_id": "sky-close-b",
                    "closed_at": first_closed_at,
                    "created_at": first_opened_at,
                }
            )
            for order_id, side, quantity, price, filled_at, pnl in (
                ("sky-entry-a", "sell", 400.0, 0.05294, second_opened_at, None),
                ("sky-close-a", "buy", 400.0, 0.05306, second_closed_at, 0.24),
                ("sky-entry-b", "sell", 500.0, 0.05235, first_opened_at, None),
                ("sky-close-b", "buy", 500.0, 0.05286, first_closed_at, -0.255),
            ):
                raw_fills = {
                    "order_id": order_id,
                    "trade_ids": [f"trade-{order_id}"],
                    "inst_id": "SKY-USDT-SWAP",
                    "contracts": quantity,
                    "contract_size": 1.0,
                    "base_quantity": quantity,
                    "avg_price": price,
                    "fee_abs": 0.01,
                    "timestamp": filled_at.isoformat(),
                }
                if pnl is not None:
                    raw_fills["fill_pnl"] = pnl
                await repo.create_order(
                    {
                        "model_name": "okx_authoritative_sync",
                        "execution_mode": "paper",
                        "symbol": "SKY/USDT",
                        "side": side,
                        "order_type": "market",
                        "quantity": quantity,
                        "price": price,
                        "status": "filled",
                        "fee": 0.01,
                        "exchange_order_id": order_id,
                        "filled_at": filled_at,
                        "created_at": filled_at,
                        "okx_inst_id": "SKY-USDT-SWAP",
                        "okx_trade_ids": f"trade-{order_id}",
                        "okx_fill_contracts": quantity,
                        "okx_fill_pnl": pnl,
                        "okx_sync_status": OKX_SYNC_CONFIRMED,
                        "okx_raw_fills": raw_fills,
                    }
                )

        dashboard_payload = await get_dashboard_positions(mode="paper", closed_only=True)
        trade_payload = await get_trade_positions(mode="paper")
    finally:
        await close_db()

    assert dashboard_payload["ledger_source"] == "okx_positions_history_official_unavailable"
    assert dashboard_payload["positions"] == []
    sky_rows = [row for row in trade_payload["positions"] if row["symbol"] == "SKY/USDT"]
    assert len(sky_rows) == 2
    assert {row["quantity"] for row in sky_rows} == {400.0, 500.0}
    assert all(row["linked_order_count"] == 2 for row in sky_rows)
    assert all(len(row["entry_order_ids"]) == 1 for row in sky_rows)
    assert all(len(row["close_order_ids"]) == 1 for row in sky_rows)
    by_quantity = {row["quantity"]: row for row in sky_rows}
    assert by_quantity[400.0]["entry_order_ids"] == ["sky-entry-a"]
    assert by_quantity[400.0]["close_order_ids"] == ["sky-close-a"]
    assert by_quantity[400.0]["realized_pnl"] == pytest.approx(0.22)
    assert by_quantity[500.0]["entry_order_ids"] == ["sky-entry-b"]
    assert by_quantity[500.0]["close_order_ids"] == ["sky-close-b"]
    assert by_quantity[500.0]["realized_pnl"] == pytest.approx(-0.275)


@pytest.mark.asyncio
async def test_position_history_rebuilds_reused_posid_from_confirmed_order_lifecycles(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'lab-reused-posid-ledger.db').as_posix()}",
    )
    await init_db()
    pos_id = "lab-reused-pos"
    first_open_a = datetime(2026, 7, 4, 9, 18, 29, tzinfo=UTC)
    first_open_b = datetime(2026, 7, 4, 9, 25, 32, tzinfo=UTC)
    first_close_a = datetime(2026, 7, 4, 9, 30, 30, tzinfo=UTC)
    first_close_b = datetime(2026, 7, 4, 9, 32, 17, tzinfo=UTC)
    first_close_c = datetime(2026, 7, 4, 9, 33, 57, tzinfo=UTC)
    second_open = datetime(2026, 7, 4, 9, 34, 38, tzinfo=UTC)
    second_close = datetime(2026, 7, 4, 9, 39, 37, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            for row in (
                {
                    "model_name": "okx_authoritative_sync",
                    "quantity": 1.9,
                    "entry_price": 9.69473684,
                    "current_price": 8.97,
                    "realized_pnl": 1.176,
                    "entry_exchange_order_id": "lab-entry-a",
                    "close_exchange_order_id": "lab-close-a",
                    "created_at": first_open_a,
                    "closed_at": first_close_a,
                },
                {
                    "model_name": "okx_authoritative_sync",
                    "quantity": 16.0,
                    "entry_price": 9.67694678,
                    "current_price": 8.7,
                    "realized_pnl": 14.845,
                    "entry_exchange_order_id": "lab-entry-b",
                    "close_exchange_order_id": "lab-close-b",
                    "created_at": first_open_a,
                    "closed_at": first_close_b,
                },
                {
                    "model_name": "okx_authoritative_sync",
                    "quantity": 40.1,
                    "entry_price": 8.48,
                    "current_price": 8.71,
                    "realized_pnl": -9.075,
                    "entry_exchange_order_id": "lab-entry-b,lab-entry-c",
                    "close_exchange_order_id": "lab-close-d",
                    "created_at": second_open,
                    "closed_at": second_close,
                },
            ):
                await repo.open_position(
                    {
                        "execution_mode": "paper",
                        "symbol": "LAB/USDT",
                        "side": "short",
                        "leverage": 1.0,
                        "unrealized_pnl": 0.0,
                        "settlement_status": "reconciled",
                        "is_open": False,
                        "okx_inst_id": "LAB-USDT-SWAP",
                        "okx_pos_id": pos_id,
                        **row,
                    }
                )

            for order_id, side, quantity, price, fee, ts, pnl in (
                ("lab-entry-a", "sell", 1.9, 9.69473684, 0.00921, first_open_a, 0.0),
                ("lab-entry-b", "sell", 35.7, 9.67694678, 0.1727335, first_open_b, 0.0),
                ("lab-close-a", "buy", 1.9, 8.97, 0.0085215, first_close_a, 1.34490691),
                ("lab-close-b", "buy", 16.0, 8.7, 0.0696, first_close_b, 15.64553191),
                ("lab-close-c", "buy", 19.7, 8.03, 0.0790955, first_close_c, 32.46256117),
                ("lab-entry-c", "sell", 40.1, 8.48, 0.0680096, second_open, 0.0),
                ("lab-close-d", "buy", 40.1, 8.71, 0.1746355, second_close, -9.223),
            ):
                raw_fills = {
                    "order_id": order_id,
                    "trade_ids": [f"trade-{order_id}"],
                    "inst_id": "LAB-USDT-SWAP",
                    "contracts": quantity * 10.0,
                    "contract_size": 0.1,
                    "base_quantity": quantity,
                    "avg_price": price,
                    "fee_abs": fee,
                    "fill_pnl": pnl,
                    "timestamp": ts.isoformat(),
                    "fills_history_confirmed": True,
                }
                await repo.create_order(
                    {
                        "model_name": "okx_authoritative_sync",
                        "execution_mode": "paper",
                        "symbol": "LAB/USDT",
                        "side": side,
                        "order_type": "market",
                        "quantity": quantity,
                        "price": price,
                        "status": "filled",
                        "fee": fee,
                        "exchange_order_id": order_id,
                        "filled_at": ts,
                        "created_at": ts,
                        "okx_inst_id": "LAB-USDT-SWAP",
                        "okx_trade_ids": f"trade-{order_id}",
                        "okx_fill_contracts": quantity * 10.0,
                        "okx_fill_pnl": pnl,
                        "okx_sync_status": OKX_SYNC_CONFIRMED,
                        "okx_raw_fills": raw_fills,
                    }
                )

        payload = await get_dashboard_positions(mode="paper", closed_only=True)
    finally:
        await close_db()

    assert payload["ledger_source"] == "okx_positions_history_official_unavailable"
    assert payload["positions"] == []


@pytest.mark.asyncio
async def test_trade_positions_api_groups_closed_positions_with_okx_ledger(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trade-positions-okx-ledger.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 6, 28, 1, 0, tzinfo=UTC)
    closed_at = datetime(2026, 6, 28, 1, 20, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            for _idx, qty, pnl in ((1, 10.0, 0.5), (2, 15.0, 0.75)):
                await repo.open_position(
                    {
                        "model_name": "okx_authoritative_sync",
                        "execution_mode": "paper",
                        "symbol": "SPK/USDT",
                        "side": "short",
                        "quantity": qty,
                        "entry_price": 0.039,
                        "current_price": 0.0345,
                        "leverage": 3.0,
                        "unrealized_pnl": 0.0,
                        "realized_pnl": pnl,
                        "settlement_status": "reconciled",
                        "is_open": False,
                        "okx_inst_id": "SPK-USDT-SWAP",
                        "okx_pos_id": "spk-phase3-pos",
                        "entry_exchange_order_id": "spk-entry",
                        "close_exchange_order_id": "spk-close",
                        "closed_at": closed_at,
                        "created_at": opened_at,
                    }
                )
            for order_id, side, ts in (
                ("spk-entry", "sell", opened_at),
                ("spk-close", "buy", closed_at),
            ):
                await repo.create_order(
                    {
                        "model_name": "okx_authoritative_sync",
                        "execution_mode": "paper",
                        "symbol": "SPK/USDT",
                        "side": side,
                        "order_type": "market",
                        "quantity": 25.0,
                        "price": 0.0345 if side == "buy" else 0.039,
                        "status": "filled",
                        "fee": 0.01,
                        "exchange_order_id": order_id,
                        "filled_at": ts,
                        "created_at": ts,
                        "okx_inst_id": "SPK-USDT-SWAP",
                        "okx_trade_ids": f"trade-{order_id}",
                        "okx_fill_contracts": 25.0,
                        "okx_fill_pnl": 1.25 if side == "buy" else 0.0,
                        "okx_sync_status": OKX_SYNC_CONFIRMED,
                        "okx_raw_fills": {
                            "order_id": order_id,
                            "trade_ids": [f"trade-{order_id}"],
                            "inst_id": "SPK-USDT-SWAP",
                            "contracts": 25.0,
                            "contract_size": 1.0,
                            "base_quantity": 25.0,
                            "avg_price": 0.0345 if side == "buy" else 0.039,
                            "fee_abs": 0.01,
                            "fill_pnl": 1.25 if side == "buy" else 0.0,
                            "timestamp": ts.isoformat(),
                        },
                    }
                )

        payload = await get_trade_positions(mode="paper")
    finally:
        await close_db()

    assert payload["ledger_source"] == "okx_current_positions_plus_grouped_closed_cache"
    assert payload["count"] == 1
    assert payload["closed_count"] == 1
    row = payload["positions"][0]
    assert row["okx_inst_id"] == "SPK-USDT-SWAP"
    assert row["is_open"] is False
    assert row["quantity"] == pytest.approx(25.0)
    assert row["realized_pnl"] == pytest.approx(1.23)
    assert row["linked_order_count"] == 2


@pytest.mark.asyncio
async def test_trade_positions_api_groups_add_entry_fragments_closed_by_same_okx_order(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trade-positions-add-entry-ledger.db').as_posix()}",
    )
    await init_db()
    first_opened_at = datetime(2026, 6, 29, 8, 0, tzinfo=UTC)
    second_opened_at = datetime(2026, 6, 29, 8, 3, tzinfo=UTC)
    closed_at = datetime(2026, 6, 29, 10, 46, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            for entry_id, opened_at, qty, entry_price, pnl in (
                ("bnb-entry-1", first_opened_at, 5.0, 610.0, -0.10),
                ("bnb-entry-2", second_opened_at, 6.0, 608.0, -0.12),
            ):
                await repo.open_position(
                    {
                        "model_name": "ensemble_trader",
                        "execution_mode": "paper",
                        "symbol": "BNB/USDT",
                        "side": "short",
                        "quantity": qty,
                        "entry_price": entry_price,
                        "current_price": 609.5,
                        "leverage": 3.0,
                        "unrealized_pnl": 0.0,
                        "realized_pnl": pnl,
                        "settlement_status": "reconciled",
                        "is_open": False,
                        "okx_inst_id": "BNB-USDT-SWAP",
                        "entry_exchange_order_id": entry_id,
                        "close_exchange_order_id": "bnb-close",
                        "closed_at": closed_at,
                        "created_at": opened_at,
                    }
                )
                await repo.create_order(
                    {
                        "model_name": "ensemble_trader",
                        "execution_mode": "paper",
                        "symbol": "BNB/USDT",
                        "side": "sell",
                        "order_type": "market",
                        "quantity": qty,
                        "price": entry_price,
                        "status": "filled",
                        "fee": 0.01,
                        "exchange_order_id": entry_id,
                        "filled_at": opened_at,
                        "created_at": opened_at,
                        "okx_inst_id": "BNB-USDT-SWAP",
                        "okx_trade_ids": f"trade-{entry_id}",
                        "okx_fill_contracts": qty,
                        "okx_sync_status": OKX_SYNC_CONFIRMED,
                        "okx_raw_fills": {
                            "order_id": entry_id,
                            "trade_ids": [f"trade-{entry_id}"],
                            "inst_id": "BNB-USDT-SWAP",
                            "contracts": qty,
                            "contract_size": 1.0,
                            "base_quantity": qty,
                            "avg_price": entry_price,
                            "fee_abs": 0.01,
                            "timestamp": opened_at.isoformat(),
                        },
                    }
                )
            await repo.create_order(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "BNB/USDT",
                    "side": "buy",
                    "order_type": "market",
                    "quantity": 11.0,
                    "price": 609.5,
                    "status": "filled",
                    "fee": 0.02,
                    "exchange_order_id": "bnb-close",
                    "filled_at": closed_at,
                    "created_at": closed_at,
                    "okx_inst_id": "BNB-USDT-SWAP",
                    "okx_trade_ids": "trade-bnb-close",
                    "okx_fill_contracts": 11.0,
                    "okx_fill_pnl": -0.22,
                    "okx_sync_status": OKX_SYNC_CONFIRMED,
                    "okx_raw_fills": {
                        "order_id": "bnb-close",
                        "trade_ids": ["trade-bnb-close"],
                        "inst_id": "BNB-USDT-SWAP",
                        "contracts": 11.0,
                        "contract_size": 1.0,
                        "base_quantity": 11.0,
                        "avg_price": 609.5,
                        "fee_abs": 0.02,
                        "fill_pnl": -0.22,
                        "timestamp": closed_at.isoformat(),
                    },
                }
            )

        payload = await get_trade_positions(mode="paper")
    finally:
        await close_db()

    assert payload["ledger_source"] == "okx_current_positions_plus_grouped_closed_cache"
    assert payload["count"] == 1
    assert payload["closed_count"] == 1
    row = payload["positions"][0]
    assert row["symbol"] == "BNB/USDT"
    assert row["side"] == "short"
    assert row["quantity"] == pytest.approx(11.0)
    assert row["average_entry_price"] == pytest.approx((5.0 * 610.0 + 6.0 * 608.0) / 11.0)
    assert row["average_close_price"] == pytest.approx(609.5)
    assert row["realized_pnl"] == pytest.approx(-0.26)
    assert row["pnl_source"] == "okx_linked_order_net_pnl"
    assert row["position_ids"] and len(row["position_ids"]) == 2
    assert set(row["entry_order_ids"]) == {"bnb-entry-1", "bnb-entry-2"}
    assert row["close_order_ids"] == ["bnb-close"]
    assert row["linked_order_count"] == 3


@pytest.mark.asyncio
async def test_trade_positions_api_hides_zero_quantity_residual_and_dedupes_canonical_row(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trade-positions-residual-ledger.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 6, 29, 8, 0, tzinfo=UTC)
    closed_at = datetime(2026, 6, 29, 10, 46, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            for qty, pnl, entry_ids, close_id in (
                (0.45, -0.5487, "bnb-entry-a,bnb-entry-b", "bnb-close"),
                (0.45, -0.4632, "bnb-entry-a,bnb-entry-b", "bnb-close"),
                (0.05, -0.10, None, "bnb-close"),
                (0.0, 0.0, None, None),
            ):
                await repo.open_position(
                    {
                        "model_name": "okx_authoritative_sync",
                        "execution_mode": "paper",
                        "symbol": "BNB/USDT",
                        "side": "short",
                        "quantity": qty,
                        "entry_price": 553.2911,
                        "current_price": 554.0,
                        "leverage": 3.0,
                        "unrealized_pnl": 0.0,
                        "realized_pnl": pnl,
                        "settlement_status": "reconciled",
                        "is_open": False,
                        "okx_inst_id": "BNB-USDT-SWAP",
                        "okx_pos_id": "bnb-pos-1",
                        "entry_exchange_order_id": entry_ids,
                        "close_exchange_order_id": close_id,
                        "closed_at": closed_at,
                        "created_at": opened_at,
                    }
                )
            for order_id, side, qty, price, ts in (
                ("bnb-entry-a", "sell", 0.1, 551.58, opened_at),
                ("bnb-entry-b", "sell", 0.35, 553.78, opened_at + timedelta(minutes=2)),
                ("bnb-close", "buy", 0.45, 554.0, closed_at),
            ):
                await repo.create_order(
                    {
                        "model_name": "ensemble_trader",
                        "execution_mode": "paper",
                        "symbol": "BNB/USDT",
                        "side": side,
                        "order_type": "market",
                        "quantity": qty,
                        "price": price,
                        "status": "filled",
                        "fee": 0.01,
                        "exchange_order_id": order_id,
                        "filled_at": ts,
                        "created_at": ts,
                        "okx_inst_id": "BNB-USDT-SWAP",
                        "okx_trade_ids": f"trade-{order_id}",
                        "okx_fill_contracts": qty / 0.01,
                        "okx_fill_pnl": -0.319 if order_id == "bnb-close" else 0.0,
                        "okx_sync_status": OKX_SYNC_CONFIRMED,
                        "okx_raw_fills": {
                            "order_id": order_id,
                            "trade_ids": [f"trade-{order_id}"],
                            "inst_id": "BNB-USDT-SWAP",
                            "contracts": qty / 0.01,
                            "contract_size": 0.01,
                            "base_quantity": qty,
                            "avg_price": price,
                            "fee_abs": 0.01,
                            "fill_pnl": -0.319 if order_id == "bnb-close" else 0.0,
                            "timestamp": ts.isoformat(),
                        },
                    }
                )

        payload = await get_trade_positions(mode="paper")
    finally:
        await close_db()

    assert payload["count"] == 1
    assert payload["closed_count"] == 1
    row = payload["positions"][0]
    assert row["symbol"] == "BNB/USDT"
    assert row["quantity"] == pytest.approx(0.45)
    assert row["realized_pnl"] == pytest.approx(-0.349)
    assert len(row["position_ids"]) == 1
    assert row["entry_order_ids"] == ["bnb-entry-a", "bnb-entry-b"]
    assert row["close_order_ids"] == ["bnb-close"]


@pytest.mark.asyncio
async def test_dashboard_positions_default_uses_okx_grouped_ledger_for_closed_rows(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-default-ledger.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 6, 29, 8, 36, tzinfo=UTC)
    closed_at = datetime(2026, 6, 29, 14, 13, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            for pnl in (-1.676530735, -1.67521975):
                await repo.open_position(
                    {
                        "model_name": "okx_authoritative_sync",
                        "execution_mode": "paper",
                        "symbol": "ETHW/USDT",
                        "side": "short",
                        "quantity": 117.0,
                        "entry_price": 0.2227000854700855,
                        "current_price": 0.23679965811965814,
                        "leverage": 3.0,
                        "unrealized_pnl": 0.0,
                        "realized_pnl": pnl,
                        "settlement_status": "reconciled",
                        "is_open": False,
                        "okx_inst_id": "ETHW-USDT-SWAP",
                        "okx_pos_id": "3698362284486926337",
                        "entry_exchange_order_id": "ethw-entry",
                        "close_exchange_order_id": "ethw-close",
                        "closed_at": closed_at,
                        "created_at": opened_at,
                    }
                )
            for order_id, side, qty, price, ts, pnl in (
                ("ethw-entry", "sell", 117.0, 0.2227, opened_at, 0.0),
                ("ethw-close", "buy", 117.0, 0.2368, closed_at, -1.676530735),
            ):
                await repo.create_order(
                    {
                        "model_name": "okx_authoritative_sync",
                        "execution_mode": "paper",
                        "symbol": "ETHW/USDT",
                        "side": side,
                        "order_type": "market",
                        "quantity": qty,
                        "price": price,
                        "status": "filled",
                        "fee": 0.01,
                        "exchange_order_id": order_id,
                        "filled_at": ts,
                        "created_at": ts,
                        "okx_inst_id": "ETHW-USDT-SWAP",
                        "okx_trade_ids": f"trade-{order_id}",
                        "okx_fill_contracts": qty,
                        "okx_fill_pnl": pnl,
                        "okx_sync_status": OKX_SYNC_CONFIRMED,
                        "okx_raw_fills": {
                            "order_id": order_id,
                            "trade_ids": [f"trade-{order_id}"],
                            "inst_id": "ETHW-USDT-SWAP",
                            "contracts": qty,
                            "contract_size": 1.0,
                            "base_quantity": qty,
                            "avg_price": price,
                            "fee_abs": 0.01,
                            "fill_pnl": pnl,
                            "timestamp": ts.isoformat(),
                        },
                    }
                )

        payload = await get_dashboard_positions(mode="paper", page_size=20)
    finally:
        await close_db()

    assert (
        payload["ledger_source"]
        == "okx_current_positions_plus_okx_positions_history_official_unavailable"
    )
    assert payload["total"] == 0
    assert payload["count"] == 0
    assert payload["open_count"] == 0
    assert payload["closed_count"] == 0
    assert payload["positions"] == []


@pytest.mark.asyncio
async def test_trade_positions_api_hides_pending_zero_quantity_residual(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trade-positions-pending-residual.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 6, 28, 3, 27, tzinfo=UTC)
    closed_at = datetime(2026, 6, 29, 10, 46, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            await repo.open_position(
                {
                    "model_name": "okx_authoritative_sync",
                    "execution_mode": "paper",
                    "symbol": "BNB/USDT",
                    "side": "short",
                    "quantity": 0.9,
                    "entry_price": 553.2911,
                    "current_price": 554.0,
                    "leverage": 3.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": -1.012,
                    "settlement_status": "reconciled",
                    "is_open": False,
                    "okx_inst_id": "BNB-USDT-SWAP",
                    "okx_pos_id": "bnb-pos-merged",
                    "entry_exchange_order_id": "bnb-entry-a,bnb-entry-b",
                    "close_exchange_order_id": "bnb-close",
                    "closed_at": closed_at,
                    "created_at": opened_at,
                }
            )
            await repo.open_position(
                {
                    "model_name": "okx_authoritative_sync",
                    "execution_mode": "paper",
                    "symbol": "BNB/USDT",
                    "side": "short",
                    "quantity": 0.0,
                    "entry_price": 0.0,
                    "current_price": 0.0,
                    "leverage": 3.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.0,
                    "settlement_status": "reconciled",
                    "is_open": False,
                    "okx_inst_id": "BNB-USDT-SWAP",
                    "okx_pos_id": "bnb-pos-merged",
                    "entry_exchange_order_id": "0",
                    "close_exchange_order_id": "pending",
                    "closed_at": closed_at - timedelta(minutes=3),
                    "created_at": opened_at,
                }
            )
            for order_id, side, qty, price, ts, pnl in (
                ("bnb-entry-a", "sell", 0.3, 552.0, opened_at, 0.0),
                ("bnb-entry-b", "sell", 0.6, 554.0, opened_at + timedelta(minutes=1), 0.0),
                ("bnb-close", "buy", 0.9, 554.0, closed_at, -1.012),
            ):
                await repo.create_order(
                    {
                        "model_name": "okx_authoritative_sync",
                        "execution_mode": "paper",
                        "symbol": "BNB/USDT",
                        "side": side,
                        "order_type": "market",
                        "quantity": qty,
                        "price": price,
                        "status": "filled",
                        "fee": 0.01,
                        "exchange_order_id": order_id,
                        "filled_at": ts,
                        "created_at": ts,
                        "okx_inst_id": "BNB-USDT-SWAP",
                        "okx_trade_ids": f"trade-{order_id}",
                        "okx_fill_contracts": qty / 0.01,
                        "okx_fill_pnl": pnl,
                        "okx_sync_status": OKX_SYNC_CONFIRMED,
                        "okx_raw_fills": {
                            "order_id": order_id,
                            "trade_ids": [f"trade-{order_id}"],
                            "inst_id": "BNB-USDT-SWAP",
                            "contracts": qty / 0.01,
                            "contract_size": 0.01,
                            "base_quantity": qty,
                            "avg_price": price,
                            "fee_abs": 0.01,
                            "fill_pnl": pnl,
                            "timestamp": ts.isoformat(),
                        },
                    }
                )

        payload = await get_trade_positions(mode="paper")
        dashboard_payload = await get_dashboard_positions(mode="paper", closed_only=True)
    finally:
        await close_db()

    assert payload["count"] == 1
    row = payload["positions"][0]
    assert row["symbol"] == "BNB/USDT"
    assert row["quantity"] == pytest.approx(0.9)
    assert row["realized_pnl"] == pytest.approx(-1.042)
    assert row["entry_order_ids"] == ["bnb-entry-a", "bnb-entry-b"]
    assert row["close_order_ids"] == ["bnb-close"]
    assert dashboard_payload["ledger_source"] == "okx_positions_history_official_unavailable"
    assert dashboard_payload["positions"] == []


@pytest.mark.asyncio
async def test_trade_positions_api_uses_okx_current_positions_when_db_open_rows_missing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web_dashboard.api import dashboard as dashboard_api

    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trade-positions-okx-current.db').as_posix()}",
    )
    await init_db()
    monkeypatch.setattr(dashboard_api, "_data_service", None)

    async def fake_exchange_mark_map(_mode):
        return {
            ("XPL/USDT", "short"): {
                "symbol": "XPL/USDT",
                "side": "short",
                "contracts": 12.0,
                "contract_size": 10.0,
                "entry_price": 0.1019,
                "mark_price": 0.0986,
                "quantity": 120.0,
                "upl": 0.396,
                "info": {
                    "instId": "XPL-USDT-SWAP",
                    "posId": "xpl-pos",
                },
            }
        }

    monkeypatch.setattr(dashboard_api, "_get_exchange_position_mark_map", fake_exchange_mark_map)

    try:
        payload = await get_trade_positions(mode="paper")
    finally:
        await close_db()

    assert payload["open_count"] == 1
    row = payload["positions"][0]
    assert row["symbol"] == "XPL/USDT"
    assert row["side"] == "short"
    assert row["quantity"] == pytest.approx(120.0)
    assert row["current_price"] == pytest.approx(0.0986)
    assert row["unrealized_pnl"] == pytest.approx(0.396)
    assert row["exchange_synced"] is True
    assert row["db_is_open"] is False
    assert row["close_status_source"] == "okx_current_position"
    assert row["okx_inst_id"] == "XPL-USDT-SWAP"


@pytest.mark.asyncio
async def test_trade_history_only_marks_explicit_exchange_sync_as_okx_execution(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trade-source.db').as_posix()}",
    )
    await init_db()
    filled_at = datetime(2026, 6, 24, 11, 10, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            await repo.open_position(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "MASK/USDT",
                    "side": "long",
                    "quantity": 47.0,
                    "entry_price": 1.68,
                    "current_price": 1.67,
                    "leverage": 3.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": -0.47,
                    "settlement_status": "reconciled",
                    "is_open": False,
                    "closed_at": filled_at,
                    "created_at": filled_at - timedelta(minutes=30),
                }
            )
            system_order = await repo.create_order(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "MASK/USDT",
                    "side": "sell",
                    "order_type": "market",
                    "quantity": 47.0,
                    "price": 1.67,
                    "status": "filled",
                    "fee": 0.02,
                    "decision_id": None,
                    "exchange_order_id": "367-system-okx-fill",
                    "filled_at": filled_at,
                    "created_at": filled_at,
                }
            )
            sync_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="H/USDT",
                action="close_short",
                confidence=1.0,
                reasoning="exchange reconcile",
                position_size_pct=1.0,
                suggested_leverage=4.0,
                raw_llm_response={
                    "system_sync": True,
                    "source": "okx_position_reconcile",
                    "close_fill": {"order_id": "367-okx-sync", "price": 0.08, "quantity": 100},
                },
                feature_snapshot={"source": "okx_position_reconcile"},
                is_paper=True,
                was_executed=True,
                created_at=filled_at + timedelta(seconds=1),
            )
            session.add(sync_decision)
            await session.flush()
            await repo.create_order(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "H/USDT",
                    "side": "buy",
                    "order_type": "market",
                    "quantity": 100.0,
                    "price": 0.08,
                    "status": "filled",
                    "fee": 0.02,
                    "decision_id": sync_decision.id,
                    "exchange_order_id": "367-okx-sync",
                    "filled_at": filled_at + timedelta(seconds=1),
                    "created_at": filled_at + timedelta(seconds=1),
                }
            )
            await repo.open_position(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "LINK/USDT",
                    "side": "short",
                    "quantity": 2.29,
                    "entry_price": 6.95,
                    "current_price": 6.902,
                    "leverage": 3.0,
                    "take_profit_price": 6.9,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.10,
                    "settlement_status": "reconciled",
                    "is_open": False,
                    "closed_at": filled_at + timedelta(seconds=2),
                    "created_at": filled_at - timedelta(minutes=20),
                }
            )
            protected_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="LINK/USDT",
                action="close_short",
                confidence=1.0,
                reasoning="system protection reconcile",
                position_size_pct=1.0,
                suggested_leverage=3.0,
                raw_llm_response={
                    "system_sync": True,
                    "source": "okx_position_reconcile",
                    "close_fill": {
                        "order_id": "protected-close",
                        "price": 6.902,
                        "quantity": 2.29,
                    },
                },
                feature_snapshot={"source": "okx_position_reconcile"},
                is_paper=True,
                was_executed=True,
                created_at=filled_at + timedelta(seconds=2),
            )
            session.add(protected_decision)
            await session.flush()
            protected_order = await repo.create_order(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "LINK/USDT",
                    "side": "buy",
                    "order_type": "market",
                    "quantity": 2.29,
                    "price": 6.902,
                    "status": "filled",
                    "fee": 0.01,
                    "decision_id": protected_decision.id,
                    "exchange_order_id": "protected-close",
                    "filled_at": filled_at + timedelta(seconds=2),
                    "created_at": filled_at + timedelta(seconds=2),
                }
            )

        trades = await get_trades(model_name=None, symbol=None, mode="paper", limit=10, page=1)
        by_id = {item["id"]: item for item in trades["trades"]}
        detail = await get_trade_detail(system_order.id)
        protected_detail = await get_trade_detail(protected_order.id)
    finally:
        await close_db()

    assert by_id[system_order.id]["execution_source"] == "system"
    assert by_id[system_order.id]["execution_source_label"] == "系统执行"
    sync_rows = [item for item in trades["trades"] if item["exchange_order_id"] == "367-okx-sync"]
    assert sync_rows and sync_rows[0]["execution_source"] == "okx"
    assert sync_rows[0]["execution_source_label"] == "OKX同步"
    protected_rows = [
        item for item in trades["trades"] if item["exchange_order_id"] == "protected-close"
    ]
    assert protected_rows and protected_rows[0]["execution_source"] == "system"
    assert protected_rows[0]["execution_source_label"] == "系统保护单"
    assert "系统保护单" in protected_rows[0]["reason"]
    assert protected_detail["execution_source"] == "system"
    assert protected_detail["execution_source_label"] == "系统保护单"
    assert detail["execution_source"] == "system"
    assert detail["execution_source_label"] == "系统执行"
    assert "OKX 平仓成交已同步" not in detail["reason"]


def test_repair_position_reason_hold_hours_replaces_stale_zero_value() -> None:
    reason = "策略纪律触发低质量旧仓释放：hard_loss_pressure；质量分层=watch，质量分=56.0，持仓小时=0.0。"

    repaired = _repair_position_reason_hold_hours(reason, 68.5883 * 60)

    assert "持仓小时=68.5883" in repaired
    assert "持仓小时=0.0" not in repaired


def test_repair_position_reason_hold_hours_keeps_valid_existing_value() -> None:
    reason = "策略纪律触发低质量旧仓释放：loss_watch；持仓小时=70.1184。"

    repaired = _repair_position_reason_hold_hours(reason, 68.0 * 60)

    assert repaired == reason


def test_trade_history_translates_okx_50001_as_temporary_exchange_failure() -> None:
    reason = (
        'Max retries exceeded: okx {"code":"50001","data":[],'
        '"msg":"Service temporarily unavailable. Please try again later."}'
    )

    translated = _translate_execution_text(reason)

    assert "交易所服务临时不可用" in translated
    assert _execution_status_label("rejected", translated) == "交易所临时不可用"


def test_trade_detail_does_not_use_numeric_order_id_as_reason() -> None:
    reason = _readable_execution_reason(
        execution_reason="",
        reasoning="策略纪律触发低质量旧仓释放：signal_reversal。",
        exchange_order_id="3670054929945042944",
        status="filled",
    )

    assert "3670054929945042944" not in reason
    assert "策略纪律触发" in reason


@pytest.mark.asyncio
async def test_xrp_final_close_cannot_be_downgraded_by_stale_local_order_links(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web_dashboard.api import dashboard as dashboard_api

    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'xrp-final-close-links.db').as_posix()}",
    )
    await init_db()
    dashboard_api._clear_dashboard_heavy_cache("closed-position-ledger")
    opened_at = datetime(2026, 7, 22, 11, 9, 44, 608000, tzinfo=UTC)
    first_close_at = datetime(2026, 7, 22, 11, 13, 53, 949000, tzinfo=UTC)
    final_close_at = datetime(2026, 7, 22, 11, 20, 49, 921000, tzinfo=UTC)
    pos_id = "3765222187788378112"
    entry_id = "3765350335720955904"
    first_close_id = "3765358702216585216"
    final_close_id = "3765372659920773120"
    official_row = {
        "instId": "XRP-USDT-SWAP",
        "posId": pos_id,
        "posSide": "net",
        "direction": "short",
        "type": "2",
        "cTime": str(int(opened_at.timestamp() * 1000)),
        "uTime": str(int(final_close_at.timestamp() * 1000)),
        "openAvgPx": "1.1372",
        "closeAvgPx": "1.137525",
        "openMaxPos": "0.04",
        "closeTotalPos": "0.04",
        "realizedPnl": "-0.00584945",
        "pnl": "-0.0013",
        "fee": "-0.00454945",
        "fundingFee": "0",
        "_bb_contract_spec": {"ctVal": "100.0", "ctMult": "1", "lotSz": "0.01"},
        "_bb_contract_spec_source": "okx_public_instruments",
    }

    async def add_order(
        repo: TradeRepository,
        *,
        order_id: str,
        side: str,
        base_quantity: float,
        contracts: float,
        price: float,
        fee: float,
        pnl: float,
        filled_at: datetime,
    ) -> None:
        await repo.create_order(
            {
                "model_name": "ensemble_trader",
                "execution_mode": "paper",
                "symbol": "XRP/USDT",
                "side": side,
                "order_type": "market",
                "quantity": base_quantity,
                "price": price,
                "status": "filled",
                "fee": fee,
                "exchange_order_id": order_id,
                "filled_at": filled_at,
                "created_at": filled_at,
                "okx_inst_id": "XRP-USDT-SWAP",
                "okx_trade_ids": f"trade-{order_id}",
                "okx_fill_contracts": contracts,
                "okx_fill_pnl": pnl,
                "okx_sync_status": OKX_SYNC_CONFIRMED,
                "okx_raw_fills": {
                    "order_id": order_id,
                    "trade_ids": [f"trade-{order_id}"],
                    "inst_id": "XRP-USDT-SWAP",
                    "contracts": contracts,
                    "contract_size": 100.0,
                    "base_quantity": base_quantity,
                    "avg_price": price,
                    "fee_abs": fee,
                    "fill_pnl": pnl,
                    "timestamp": filled_at.isoformat(),
                    "fills_history_confirmed": True,
                },
            }
        )

    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            await add_order(
                repo,
                order_id=entry_id,
                side="sell",
                base_quantity=4.0,
                contracts=0.04,
                price=1.1372,
                fee=0.0022744,
                pnl=0.0,
                filled_at=opened_at,
            )
            await add_order(
                repo,
                order_id=first_close_id,
                side="buy",
                base_quantity=3.0,
                contracts=0.03,
                price=1.1375,
                fee=0.00170625,
                pnl=-0.0009,
                filled_at=first_close_at,
            )
        await _seed_okx_position_history_rows(
            [official_row],
            entry_order_ids={pos_id: [entry_id]},
            close_order_ids={pos_id: [first_close_id]},
        )

        before_final_link = await get_dashboard_positions(mode="paper", closed_only=True)
        stale_link_row = before_final_link["positions"][0]
        assert stale_link_row["close_status"] == "full"
        assert stale_link_row["closed_at"] == final_close_at.isoformat()
        assert stale_link_row["quantity"] == pytest.approx(4.0)
        assert stale_link_row["max_position_quantity"] == pytest.approx(4.0)
        assert stale_link_row["close_order_ids"] == [first_close_id]
        assert stale_link_row["evidence_complete"] is False
        assert stale_link_row["trainable"] is False
        assert "position_history_close_quantity_not_matched_to_orders" in stale_link_row[
            "evidence_gaps"
        ]

        async with get_session_ctx() as session:
            await add_order(
                TradeRepository(session),
                order_id=final_close_id,
                side="buy",
                base_quantity=1.0,
                contracts=0.01,
                price=1.1376,
                fee=0.0005688,
                pnl=-0.0004,
                filled_at=final_close_at,
            )

        after_final_link = await get_dashboard_positions(mode="paper", closed_only=True)
        repaired_row = after_final_link["positions"][0]
        assert repaired_row["close_status"] == "full"
        assert repaired_row["quantity"] == pytest.approx(4.0)
        assert repaired_row["max_position_quantity"] == pytest.approx(4.0)
        assert repaired_row["entry_order_ids"] == [entry_id]
        assert repaired_row["close_order_ids"] == [first_close_id, final_close_id]
        assert repaired_row["linked_order_count"] == 3
        assert repaired_row["evidence_complete"] is True
        assert repaired_row["trainable"] is True
    finally:
        dashboard_api._clear_dashboard_heavy_cache("closed-position-ledger")
        await close_db()


def test_trade_detail_numeric_only_reason_falls_back_to_readable_success() -> None:
    reason = _readable_execution_reason(
        execution_reason="3670054929945042944",
        reasoning="",
        exchange_order_id="3670054929945042944",
        status="filled",
    )

    assert "3670054929945042944" not in reason
    assert "订单已成交" in reason
