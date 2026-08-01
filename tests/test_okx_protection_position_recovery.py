from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.trade import Order, Position
from services.okx_protection_position_recovery import (
    PROTECTION_POSITION_RECOVERY_SOURCE,
    confirmed_protection_lifecycle,
    recover_protection_position_lifecycles,
)


def _management_contract(*, algo_id: str, entry_order_id: str, entry_fee: float) -> dict:
    return {
        "entry_fee_usdt": entry_fee,
        "original_entry_order_ids": [entry_order_id],
        "protection_orders": [
            {
                "algo_id": algo_id,
                "reduce_only": True,
            }
        ],
    }


def _protection_order(
    *,
    exchange_order_id: str,
    entry_decision_id: int | None,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    fill_pnl: float,
    fee: float,
    algo_id: str,
    position_side: str,
    filled_at: datetime,
) -> Order:
    return Order(
        model_name="okx_authoritative_sync",
        execution_mode="paper",
        symbol=symbol,
        side=side,
        order_type="market",
        quantity=quantity,
        price=price,
        status="filled",
        fee=fee,
        decision_id=entry_decision_id,
        exchange_order_id=exchange_order_id,
        filled_at=filled_at,
        okx_inst_id=symbol.replace("/", "-") + "-SWAP",
        okx_fill_pnl=fill_pnl,
        okx_sync_status="okx_confirmed",
        okx_raw_fills={
            "fills_history_confirmed": True,
            "order_id": exchange_order_id,
            "base_quantity": quantity,
            "protection_execution": {
                "lifecycle_complete": True,
                "source_authority": "okx_algo_history_plus_fills_history",
                "generated_order_id": exchange_order_id,
                "algo_id": algo_id,
                "inst_id": symbol.replace("/", "-") + "-SWAP",
                "position_side": position_side,
                "close_side": side,
                "reduce_only": True,
            },
        },
        created_at=filled_at,
    )


def test_recovery_rejects_non_reduce_only_lifecycle() -> None:
    order = _protection_order(
        exchange_order_id="unsafe-close",
        entry_decision_id=704,
        symbol="ETH/USDT",
        side="sell",
        quantity=0.5,
        price=1930.0,
        fill_pnl=15.0,
        fee=0.2,
        algo_id="unsafe-algo",
        position_side="long",
        filled_at=datetime(2026, 8, 1, 5, 0, tzinfo=UTC),
    )
    order.okx_raw_fills["protection_execution"]["reduce_only"] = False
    order.okx_raw_fills["protection_execution"]["algo_row"] = {
        "reduceOnly": "true"
    }

    assert confirmed_protection_lifecycle(order) is None


@pytest.mark.asyncio
async def test_recovery_links_unique_quarantined_protection_remainder(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'quarantined-recovery.db').as_posix()}",
    )
    await init_db()
    filled_at = datetime(2026, 8, 1, 4, 0, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="LTC/USDT",
                    side="sell",
                    order_type="market",
                    quantity=4.0,
                    price=45.0,
                    status="filled",
                    decision_id=701,
                    exchange_order_id="ltc-entry",
                    filled_at=filled_at - timedelta(days=1),
                )
            )
            position = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="LTC/USDT",
                side="short",
                quantity=3.8,
                entry_price=45.0,
                current_price=44.8,
                is_open=False,
                closed_at=filled_at - timedelta(hours=4),
                okx_inst_id="LTC-USDT-SWAP",
                entry_exchange_order_id="ltc-entry",
                close_exchange_order_id="okx_orphan_quarantine:11",
                settlement_status="settlement_quarantined",
                settlement_source="okx_position_history_identity_quarantine",
                current_management_contract=_management_contract(
                    algo_id="ltc-algo",
                    entry_order_id="ltc-entry",
                    entry_fee=0.08,
                ),
            )
            close_order = _protection_order(
                exchange_order_id="ltc-protection-close",
                entry_decision_id=None,
                symbol="LTC/USDT",
                side="buy",
                quantity=3.8,
                price=44.0,
                fill_pnl=3.2,
                fee=0.08,
                algo_id="ltc-algo",
                position_side="short",
                filled_at=filled_at,
            )
            legacy_lifecycle = close_order.okx_raw_fills["protection_execution"]
            legacy_lifecycle.pop("reduce_only")
            legacy_lifecycle["algo_row"] = {"reduceOnly": "true"}
            session.add_all([position, close_order])
            await session.flush()

            result = await recover_protection_position_lifecycles(
                session,
                orders=[close_order],
                mode="paper",
                now=filled_at + timedelta(minutes=1),
            )

            assert result[0]["kind"] == "quarantined_protection_position_lifecycle_recovered"
            assert position.close_exchange_order_id == "ltc-protection-close"
            assert position.closed_at == filled_at
            assert position.settlement_status == "settlement_quarantined"
            assert close_order.decision_id == 701
            assert close_order.okx_raw_fills["protection_execution"]["reduce_only"] is True
            evidence = position.settlement_raw["protection_position_lifecycle_recovery"]
            assert evidence["okx_algo_id"] == "ltc-algo"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_recovery_closes_unique_open_position_from_confirmed_protection_fill(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'open-recovery.db').as_posix()}",
    )
    await init_db()
    filled_at = datetime(2026, 8, 1, 5, 0, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            position = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="ETH/USDT",
                side="long",
                quantity=0.5,
                entry_price=1900.0,
                current_price=1920.0,
                unrealized_pnl=10.0,
                entry_fee=0.1,
                funding_fee=0.05,
                is_open=True,
                okx_inst_id="ETH-USDT-SWAP",
                entry_exchange_order_id="eth-entry",
                current_management_contract=_management_contract(
                    algo_id="eth-algo",
                    entry_order_id="eth-entry",
                    entry_fee=0.1,
                ),
            )
            close_order = _protection_order(
                exchange_order_id="eth-protection-close",
                entry_decision_id=702,
                symbol="ETH/USDT",
                side="sell",
                quantity=0.5,
                price=1930.0,
                fill_pnl=15.0,
                fee=0.2,
                algo_id="eth-algo",
                position_side="long",
                filled_at=filled_at,
            )
            session.add_all([position, close_order])
            await session.flush()

            result = await recover_protection_position_lifecycles(
                session,
                orders=[close_order],
                mode="paper",
                now=filled_at + timedelta(minutes=1),
            )

            assert result[0]["kind"] == "open_protection_position_lifecycle_recovered"
            assert position.is_open is False
            assert position.close_exchange_order_id == "eth-protection-close"
            assert position.closed_at == filled_at
            assert position.settlement_status == "settling"
            assert position.settlement_source == PROTECTION_POSITION_RECOVERY_SOURCE
            assert position.close_fill_pnl == pytest.approx(15.0)
            assert position.realized_pnl == pytest.approx(14.75)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_recovery_refuses_ambiguous_quarantined_candidates(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'ambiguous-recovery.db').as_posix()}",
    )
    await init_db()
    filled_at = datetime(2026, 8, 1, 6, 0, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            positions = [
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="SAND/USDT",
                    side="short",
                    quantity=100.0,
                    entry_price=0.04,
                    current_price=0.04,
                    is_open=False,
                    closed_at=filled_at - timedelta(hours=1),
                    okx_inst_id="SAND-USDT-SWAP",
                    entry_exchange_order_id="sand-entry",
                    close_exchange_order_id=f"okx_orphan_quarantine:{position_id}",
                    current_management_contract=_management_contract(
                        algo_id="sand-algo",
                        entry_order_id="sand-entry",
                        entry_fee=0.01,
                    ),
                )
                for position_id in (21, 22)
            ]
            close_order = _protection_order(
                exchange_order_id="sand-protection-close",
                entry_decision_id=703,
                symbol="SAND/USDT",
                side="buy",
                quantity=100.0,
                price=0.041,
                fill_pnl=-0.1,
                fee=0.01,
                algo_id="sand-algo",
                position_side="short",
                filled_at=filled_at,
            )
            session.add_all([*positions, close_order])
            await session.flush()

            result = await recover_protection_position_lifecycles(
                session,
                orders=[close_order],
                mode="paper",
            )

            assert result == []
            assert all(
                str(position.close_exchange_order_id).startswith("okx_orphan_quarantine:")
                for position in positions
            )
    finally:
        await close_db()
