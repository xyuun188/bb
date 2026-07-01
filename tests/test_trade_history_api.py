from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from config.settings import settings
from db.repositories.trade_repo import TradeRepository
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from services.okx_order_fact_sync import OKX_SYNC_CONFIRMED, OKX_SYNC_EXECUTION_RESULT_CONFIRMED
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

    assert payload["positions"]
    assert {item["close_status"] for item in payload["positions"]} == {"partial"}


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

    assert payload["positions"][0]["realized_pnl"] == pytest.approx(0.43834414999998383)
    assert payload["positions"][0]["pnl_source"] == "position_realized_pnl"


@pytest.mark.asyncio
async def test_dashboard_position_history_uses_okx_grouped_ledger_with_linked_fills(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-okx-ledger.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 6, 28, 0, 38, tzinfo=UTC)
    closed_at = datetime(2026, 6, 28, 12, 40, tzinfo=UTC)
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

    assert payload["ledger_source"] == "okx_native_grouped_cache"
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

    assert payload["total"] == 1
    row = payload["positions"][0]
    assert row["symbol"] == "AI16Z/USDT"
    assert row["quantity"] == pytest.approx(732.0)
    assert row["realized_pnl"] == pytest.approx(7.937940890773545)
    assert row["pnl_source"] == "okx_position_history_realized_pnl"
    assert row["close_order_ids"] == ["ai16z-close-a", "ai16z-close-b", "ai16z-close-c"]
    assert row["linked_order_count"] == 4
    assert len(row["position_ids"]) == 1


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

    for payload in (dashboard_payload, trade_payload):
        sky_rows = [row for row in payload["positions"] if row["symbol"] == "SKY/USDT"]
        assert len(sky_rows) == 2
        assert {row["quantity"] for row in sky_rows} == {400.0, 500.0}
        assert all(row["linked_order_count"] == 2 for row in sky_rows)
        assert all(len(row["entry_order_ids"]) == 1 for row in sky_rows)
        assert all(len(row["close_order_ids"]) == 1 for row in sky_rows)
        by_quantity = {row["quantity"]: row for row in sky_rows}
        assert by_quantity[400.0]["entry_order_ids"] == ["sky-entry-a"]
        assert by_quantity[400.0]["close_order_ids"] == ["sky-close-a"]
        assert by_quantity[400.0]["realized_pnl"] == pytest.approx(0.24)
        assert by_quantity[500.0]["entry_order_ids"] == ["sky-entry-b"]
        assert by_quantity[500.0]["close_order_ids"] == ["sky-close-b"]
        assert by_quantity[500.0]["realized_pnl"] == pytest.approx(0.11)


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
    assert row["realized_pnl"] == pytest.approx(1.25)
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
    assert row["realized_pnl"] == pytest.approx(-0.22)
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
    assert row["realized_pnl"] == pytest.approx(-0.4632)
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

    assert payload["ledger_source"] == "okx_current_positions_plus_grouped_closed_cache"
    assert payload["total"] == 1
    assert payload["count"] == 1
    assert payload["open_count"] == 0
    assert payload["closed_count"] == 1
    row = payload["positions"][0]
    assert row["symbol"] == "ETHW/USDT"
    assert row["position_ids"] and len(row["position_ids"]) == 1
    assert row["entry_order_ids"] == ["ethw-entry"]
    assert row["close_order_ids"] == ["ethw-close"]
    assert row["quantity"] == pytest.approx(117.0)


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

    for payload_rowset in (payload, dashboard_payload):
        assert payload_rowset["count"] == 1
        row = payload_rowset["positions"][0]
        assert row["symbol"] == "BNB/USDT"
        assert row["quantity"] == pytest.approx(0.9)
        assert row["realized_pnl"] == pytest.approx(-1.012)
        assert row["entry_order_ids"] == ["bnb-entry-a", "bnb-entry-b"]
        assert row["close_order_ids"] == ["bnb-close"]


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


def test_trade_detail_numeric_only_reason_falls_back_to_readable_success() -> None:
    reason = _readable_execution_reason(
        execution_reason="3670054929945042944",
        reasoning="",
        exchange_order_id="3670054929945042944",
        status="filled",
    )

    assert "3670054929945042944" not in reason
    assert "订单已成交" in reason
