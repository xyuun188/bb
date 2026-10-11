from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.trade import Order, Position
from services.okx_lifecycle_fragment_repair import (
    REPAIR_SOURCE,
    _history_lifecycle,
    _repair_linked_lifecycle_identity,
    repair_missing_okx_lifecycle_fragments,
)
from services.okx_position_settlement_sync import (
    OkxPositionSettlementSyncService,
    _group_candidates_by_lifecycle,
    _prepare_lifecycle_allocations,
)


def _raw_fill(
    *,
    order_id: str,
    contracts: float,
    quantity: float,
    pnl: float,
    side: str,
    position_side: str | None = None,
) -> dict:
    raw = {
        "source": "okx_authoritative_sync",
        "contracts": contracts,
        "base_quantity": quantity,
        "contract_size": 10.0,
        "contract_size_verified": True,
        "fills_history_confirmed": True,
    }
    if position_side:
        raw["protection_execution"] = {
            "lifecycle_complete": True,
            "source_authority": "okx_algo_history_plus_fills_history",
            "generated_order_id": order_id,
            "algo_id": f"algo-{order_id}",
            "position_side": position_side,
            "close_side": side,
            "reduce_only": True,
        }
    return raw


def _order(
    *,
    order_id: str,
    side: str,
    quantity: float,
    contracts: float,
    price: float,
    filled_at: datetime,
    pnl: float,
    protection: bool = False,
) -> Order:
    return Order(
        model_name="okx_authoritative_sync",
        execution_mode="paper",
        symbol="SLX/USDT",
        side=side,
        order_type="market",
        quantity=quantity,
        price=price,
        status="filled",
        fee=0.0,
        exchange_order_id=order_id,
        filled_at=filled_at,
        created_at=filled_at,
        okx_inst_id="SLX-USDT-SWAP",
        okx_fill_contracts=contracts,
        okx_fill_pnl=pnl,
        okx_sync_status="okx_only_backfilled" if protection else "okx_confirmed",
        okx_raw_fills=_raw_fill(
            order_id=order_id,
            contracts=contracts,
            quantity=quantity,
            pnl=pnl,
            side=side,
            position_side="short" if protection else None,
        ),
    )


@pytest.mark.parametrize("evidence_gap", ["", "ambiguous_close", "quantity_mismatch", "unconfirmed"])
@pytest.mark.parametrize("settlement_status", ["settlement_quarantined", "reconciled"])
def test_stale_entry_identity_requires_unique_conserved_fill_evidence(
    evidence_gap, settlement_status,
) -> None:
    opened_at = datetime(2026, 10, 8, 11, 0, tzinfo=UTC)
    closed_at = opened_at + timedelta(hours=4)
    row = {
        "instId": "SLX-USDT-SWAP", "posId": "reused-pos", "direction": "short",
        "cTime": str(int(opened_at.timestamp() * 1000)),
        "uTime": str(int(closed_at.timestamp() * 1000)),
        "openMaxPos": "100", "closeTotalPos": "100", "openAvgPx": "0.07",
        "_dashboard_entry_order_ids": ["entry-current"],
        "_dashboard_close_order_ids": ["close-current"],
        "_dashboard_history_record_id": 123,
    }
    entry = _order(
        order_id="entry-current", side="sell", quantity=1000, contracts=100,
        price=0.07, filled_at=opened_at, pnl=0,
    )
    close = _order(
        order_id="close-current", side="buy", quantity=1000,
        contracts=90 if evidence_gap == "quantity_mismatch" else 100,
        price=0.066, filled_at=closed_at, pnl=4,
    )
    if evidence_gap == "unconfirmed":
        close.okx_raw_fills["fills_history_confirmed"] = False
    position = Position(
        id=1, symbol="SLX/USDT", side="short", quantity=1000,
        okx_inst_id="SLX-USDT-SWAP", okx_pos_id="reused-pos",
        entry_exchange_order_id="entry-previous", close_exchange_order_id="close-current",
        created_at=opened_at - timedelta(days=1), entry_price=0.09,
        settlement_status=settlement_status, realized_pnl=4.0,
        settlement_raw={"original_fact": "preserved"},
    )
    lifecycle = _history_lifecycle(row, mode="paper")
    owners = {"close-current": {"current", "other"} if evidence_gap == "ambiguous_close" else {"current"}}
    result = _repair_linked_lifecycle_identity(
        lifecycle, positions_by_close_id={"close-current": [position]},
        orders_by_id={"entry-current": entry, "close-current": close},
        close_owners=owners, now=closed_at,
    )
    assert len(result) == (0 if evidence_gap else 1)
    assert position.realized_pnl == 4.0
    assert position.settlement_raw["original_fact"] == "preserved"
    if not evidence_gap:
        assert position.entry_exchange_order_id == "entry-current"
        assert position.created_at == opened_at
        assert position.entry_price == 0.07
        assert position.settlement_raw["entry_identity_repair"]["previous_entry_order_ids"] == "entry-previous"
        assert position.settlement_status == settlement_status
        assert _repair_linked_lifecycle_identity(
            lifecycle, positions_by_close_id={"close-current": [position]},
            orders_by_id={"entry-current": entry, "close-current": close},
            close_owners=owners, now=closed_at,
        ) == []
    else:
        assert position.entry_exchange_order_id == "entry-previous"


@pytest.mark.asyncio
async def test_missing_authoritative_close_fill_creates_idempotent_fragment(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'fragment-repair.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 8, 26, 0, 0, tzinfo=UTC)
    closed_at = opened_at + timedelta(hours=5)
    entry_id = "entry-104"
    try:
        async with get_session_ctx() as session:
            session.add(
                _order(
                    order_id=entry_id,
                    side="sell",
                    quantity=1040.0,
                    contracts=104.0,
                    price=0.07,
                    filled_at=opened_at,
                    pnl=0.0,
                )
            )
            close_specs = (
                ("close-10", 100.0, 10.0, 0.01),
                ("close-73", 730.0, 73.0, 0.02),
                ("close-5", 50.0, 5.0, 0.03),
                ("close-7", 70.0, 7.0, 0.04),
                ("close-9", 90.0, 9.0, 0.05),
            )
            for index, (order_id, quantity, contracts, pnl) in enumerate(close_specs):
                session.add(
                    _order(
                        order_id=order_id,
                        side="buy",
                        quantity=quantity,
                        contracts=contracts,
                        price=0.066,
                        filled_at=closed_at - timedelta(minutes=4 - index),
                        pnl=pnl,
                        protection=order_id == "close-9",
                    )
                )
            for order_id, quantity in (
                ("close-10", 100.0),
                ("close-73", 730.0),
                ("close-5", 50.0),
                ("close-7", 70.0),
            ):
                session.add(
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SLX/USDT",
                        side="short",
                        quantity=quantity,
                        entry_price=0.07,
                        current_price=0.066,
                        leverage=1.0,
                        is_open=False,
                        closed_at=closed_at,
                        okx_inst_id="SLX-USDT-SWAP",
                        okx_pos_id="pos-104",
                        entry_exchange_order_id=entry_id,
                        close_exchange_order_id=order_id,
                        settlement_status="settling",
                    )
                )
            await session.flush()
            history = {
                "instId": "SLX-USDT-SWAP",
                "posId": "pos-104",
                "posSide": "net",
                "direction": "short",
                "cTime": str(int(opened_at.timestamp() * 1000)),
                "uTime": str(int(closed_at.timestamp() * 1000)),
                "openAvgPx": "0.07",
                "openMaxPos": "104",
                "closeTotalPos": "104",
                "realizedPnl": "0.15",
                "_bb_contract_spec": {"ctVal": "10", "ctMult": "1"},
            }
            result = await repair_missing_okx_lifecycle_fragments(
                session,
                mode="paper",
                position_history_rows=[history],
                now=closed_at + timedelta(minutes=1),
            )
            assert result["created_count"] == 1
            repaired = (
                await session.execute(
                    select(Position).where(Position.close_exchange_order_id == "close-9")
                )
            ).scalar_one()
            assert repaired.quantity == pytest.approx(90.0)
            assert repaired.settlement_source == REPAIR_SOURCE
            assert repaired.settlement_raw["training_policy"] == (
                "exclude_until_authoritative_settlement"
            )

            second = await repair_missing_okx_lifecycle_fragments(
                session,
                mode="paper",
                position_history_rows=[history],
                now=closed_at + timedelta(minutes=2),
            )
            assert second["created_count"] == 0

        candidates = await OkxPositionSettlementSyncService(mode="paper")._load_candidates(
            closed_at + timedelta(minutes=3)
        )
        group = next(iter(_group_candidates_by_lifecycle(candidates).values()))
        _prepare_lifecycle_allocations(group, [history])
        assert sum(member.close_contracts for member in group) == pytest.approx(104.0)
        assert all(member.allocation_complete for member in group)
    finally:
        await close_db()
