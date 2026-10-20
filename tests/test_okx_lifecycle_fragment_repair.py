from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from models.trade import OkxPositionHistory, Order, Position
from services.okx_lifecycle_fragment_repair import (
    REPAIR_SOURCE,
    _history_lifecycle,
    _repair_linked_lifecycle_identity,
    repair_missing_okx_lifecycle_fragments,
)
from services.okx_position_history_store import (
    okx_position_history_records_to_rows,
    upsert_okx_position_history_row,
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
    model_name: str = "okx_authoritative_sync",
    decision_id: int | None = None,
) -> Order:
    return Order(
        model_name=model_name,
        execution_mode="paper",
        symbol="SLX/USDT",
        side=side,
        order_type="market",
        quantity=quantity,
        price=price,
        status="filled",
        fee=0.0,
        decision_id=decision_id,
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


async def _seed_missing_full_lifecycle(session, *, failure: str | None = None):
    opened_at = datetime(2026, 10, 10, 16, 55, 53, 410000, tzinfo=UTC)
    closed_at = datetime(2026, 10, 10, 16, 58, 6, 464000, tzinfo=UTC)
    entry_id = "entry-full-lifecycle"
    close_id = "close-full-lifecycle"
    entry_decision_id = 607397 if failure != "missing_decision" else 999999
    close_contracts = 103.0 if failure == "partial" else 104.0
    close_quantity = close_contracts * 10.0
    if failure != "missing_decision":
        session.add(
            AIDecision(
                id=607397,
                model_name="ensemble_trader",
                symbol="SLX/USDT",
                action="long",
                confidence=0.8,
                is_paper=True,
                was_executed=True,
                executed_at=opened_at,
            )
        )
    session.add(
        _order(
            order_id=entry_id,
            side="buy",
            quantity=1040.0,
            contracts=104.0,
            price=0.07,
            filled_at=opened_at,
            pnl=0.0,
            model_name="ensemble_trader",
            decision_id=entry_decision_id,
        )
    )
    close_order = _order(
        order_id=close_id,
        side="sell",
        quantity=close_quantity,
        contracts=close_contracts,
        price=0.072,
        filled_at=closed_at,
        pnl=close_contracts * 10.0 * (0.072 - 0.07),
    )
    if failure == "unconfirmed":
        close_order.okx_raw_fills["fills_history_confirmed"] = False
    session.add(close_order)
    official_close_contracts = close_contracts
    history_record = OkxPositionHistory(
        mode="paper",
        row_identity=f"paper|SLX-USDT-SWAP|pos-full|net|{int(opened_at.timestamp() * 1000)}",
        inst_id="SLX-USDT-SWAP",
        symbol="SLX/USDT",
        pos_id="pos-full",
        pos_side="net",
        side="long",
        close_status="partial" if failure == "not_full" else "full",
        opened_at=opened_at,
        updated_at_okx=closed_at,
        open_avg_px=0.07,
        close_avg_px=0.072,
        open_max_pos=104.0,
        close_total_pos=official_close_contracts,
        leverage=2.0,
        realized_pnl=official_close_contracts * 10.0 * (0.072 - 0.07),
        pnl=official_close_contracts * 10.0 * (0.072 - 0.07),
        funding_fee=0.0,
        fee=0.0,
        entry_order_ids=[entry_id],
        close_order_ids=[close_id],
        linked_order_ids=[entry_id, close_id],
        position_ids=[],
        match_status="okx_account_position_history",
        evidence_gaps=[],
        source="okx_position_history",
        sync_status="synced",
        raw_row={},
    )
    session.add(history_record)
    await session.flush()
    row = {
        "instId": "SLX-USDT-SWAP",
        "posId": "pos-full",
        "posSide": "net",
        "direction": "long",
        "cTime": str(int(opened_at.timestamp() * 1000)),
        "uTime": str(int(closed_at.timestamp() * 1000)),
        "openAvgPx": "0.07",
        "closeAvgPx": "0.072",
        "openMaxPos": "104",
        "closeTotalPos": str(official_close_contracts),
        "lever": "2.0",
        "_bb_contract_spec": {"ctVal": "10", "ctMult": "1"},
        "_dashboard_history_record_id": history_record.id,
        "_dashboard_history_row_identity": history_record.row_identity,
        "_dashboard_entry_order_ids": [entry_id],
        "_dashboard_close_order_ids": [close_id],
        "_dashboard_position_ids": [],
    }
    return row, history_record, opened_at, closed_at


@pytest.mark.asyncio
async def test_missing_full_lifecycle_is_recovered_once_and_waits_for_settlement(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'full-lifecycle-repair.db').as_posix()}",
    )
    await init_db()
    try:
        async with get_session_ctx() as session:
            row, history_record, opened_at, closed_at = await _seed_missing_full_lifecycle(session)
            session.add(
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="ALGO/USDT",
                    side="long",
                    quantity=50.0,
                    entry_price=0.07,
                    current_price=0.071,
                    leverage=1.0,
                    is_open=False,
                    closed_at=opened_at + timedelta(hours=1),
                    okx_inst_id="ALGO-USDT-SWAP",
                    okx_pos_id="pos-full",
                    entry_exchange_order_id="unrelated-entry-order",
                    close_exchange_order_id="unrelated-close-order",
                    created_at=opened_at + timedelta(hours=1),
                )
            )
            await session.flush()
            result = await repair_missing_okx_lifecycle_fragments(
                session,
                mode="paper",
                position_history_rows=[row],
                now=closed_at + timedelta(minutes=1),
            )
            assert result["created_count"] == 1
            position = (
                await session.execute(
                    select(Position).where(
                        Position.okx_pos_id == "pos-full",
                        Position.settlement_source == REPAIR_SOURCE,
                    )
                )
            ).scalar_one()
            assert position.is_open is False
            assert position.settlement_status == "settling"
            assert position.quantity == pytest.approx(1040.0)
            assert position.entry_price == pytest.approx(0.07)
            assert position.current_price == pytest.approx(0.072)
            assert position.created_at.replace(tzinfo=UTC) == opened_at
            assert position.closed_at.replace(tzinfo=UTC) == closed_at
            assert history_record.position_ids == [str(position.id)]
            assert history_record.match_status == "okx_full_lifecycle_recovery_pending_settlement"
            assert position.settlement_raw["lifecycle_recovery"]["decision_id"] == 607397

            repeat = await repair_missing_okx_lifecycle_fragments(
                session,
                mode="paper",
                position_history_rows=[row],
                now=closed_at + timedelta(minutes=2),
            )
            assert repeat["created_count"] == 0
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_concurrent_repairs_read_positions_after_prior_transaction_commits(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings, "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'concurrent-lifecycle-repair.db').as_posix()}",
    )
    await init_db()
    transaction_lock = asyncio.Lock()
    second_waiting = asyncio.Event()
    first_repaired = asyncio.Event()
    statements: dict[int, list[str]] = {1: [], 2: []}
    lock_keys: list[str] = []

    class PostgresRepairSession:
        def __init__(self, session, worker):
            self.session = session
            self.worker = worker

        def __getattr__(self, name):
            return getattr(self.session, name)

        def get_bind(self):
            return SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

        async def execute(self, statement, params=None):
            query = str(statement)
            statements[self.worker].append(query)
            if "pg_advisory_xact_lock" in query:
                lock_keys.append(params["lock_key"])
                if self.worker == 2:
                    second_waiting.set()
                await transaction_lock.acquire()
                return None
            assert transaction_lock.locked()
            return await self.session.execute(statement, params)

    try:
        async with get_session_ctx() as session:
            row, _, _, closed_at = await _seed_missing_full_lifecycle(session)

        async def repair(worker):
            if worker == 2:
                await first_repaired.wait()
            acquired = False
            try:
                async with get_session_ctx() as session:
                    proxy = PostgresRepairSession(session, worker)
                    result = await repair_missing_okx_lifecycle_fragments(
                        proxy, mode="paper", position_history_rows=[deepcopy(row)],
                        now=closed_at + timedelta(minutes=worker),
                    )
                    acquired = True
                    if worker == 1:
                        first_repaired.set()
                        await second_waiting.wait()
                return result
            finally:
                if acquired:
                    transaction_lock.release()

        results = await asyncio.wait_for(asyncio.gather(repair(1), repair(2)), timeout=10)
        assert [result["created_count"] for result in results] == [1, 0]
        assert lock_keys == ["bb:okx_lifecycle_fragment_repair:paper"] * 2
        for worker in (1, 2):
            assert "pg_advisory_xact_lock" in statements[worker][0]
            assert "FROM positions" in statements[worker][1]
        async with get_session_ctx() as session:
            positions = (await session.execute(select(Position))).scalars().all()
            assert len(positions) == 1
    finally:
        await close_db()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gap",
    [
        "",
        "millisecond_boundary",
        "missing_identity_close_fragment",
        "superseded_projection",
        "unconfirmed",
        "duplicate_order",
        "quantity",
        "fee",
        "price",
        "other_lifecycle",
        "shared_position",
        "after_history",
    ],
)
async def test_partial_lifecycle_links_catch_up_without_closing_or_relabeling_open_position(
    tmp_path, monkeypatch, gap,
) -> None:
    await close_db()
    database_path = tmp_path / f"partial-links-{gap or 'complete'}.db"
    monkeypatch.setattr(
        settings, "database_url",
        f"sqlite+aiosqlite:///{database_path.as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    opened_at = now - timedelta(hours=3)
    close_at = (now - timedelta(hours=1)).replace(microsecond=0)
    repair_expected = gap in {
        "", "millisecond_boundary", "missing_identity_close_fragment", "superseded_projection",
    }
    try:
        async with get_session_ctx() as session:
            session.add(_order(
                order_id="partial-entry", side="buy", quantity=1000, contracts=100,
                price=0.07, filled_at=opened_at, pnl=0,
            ))
            session.add(_order(
                order_id="partial-close-old", side="sell", quantity=300, contracts=30,
                price=0.072, filled_at=opened_at + timedelta(hours=1), pnl=0.6,
            ))
            missing_close = _order(
                order_id="partial-close-new", side="sell", quantity=300, contracts=30,
                price=0.075 if gap == "price" else 0.072,
                filled_at=(
                    close_at + timedelta(seconds=1) if gap == "after_history"
                    else close_at + timedelta(milliseconds=1) if gap == "millisecond_boundary"
                    else close_at
                ),
                pnl=0.6,
            )
            if gap == "unconfirmed":
                missing_close.okx_raw_fills["fills_history_confirmed"] = False
            session.add(missing_close)
            if gap == "duplicate_order":
                session.add(_order(
                    order_id="partial-close-new", side="sell", quantity=300, contracts=30,
                    price=0.072, filled_at=close_at, pnl=0.6,
                ))
            position = Position(
                model_name="ensemble_trader", execution_mode="paper",
                symbol="SLX/USDT", side="long", quantity=500 if gap == "quantity" else 400,
                entry_price=0.07, current_price=0.072, leverage=2,
                realized_pnl=1.2, is_open=True, created_at=opened_at,
                okx_inst_id="SLX-USDT-SWAP", okx_pos_id="partial-pos",
                entry_exchange_order_id="partial-entry",
            )
            session.add(position)
            conflicting_position = None
            if gap in {
                "missing_identity_close_fragment", "shared_position", "superseded_projection",
            }:
                conflicting_position = Position(
                    model_name="ensemble_trader", execution_mode="paper",
                    symbol="SLX/USDT", side="long", quantity=300,
                    entry_price=0.07, leverage=2, is_open=False, closed_at=close_at,
                    created_at=(
                        opened_at if gap == "missing_identity_close_fragment"
                        else opened_at - timedelta(days=1)
                    ),
                    okx_inst_id="SLX-USDT-SWAP",
                    okx_pos_id=(
                        None if gap in {"missing_identity_close_fragment", "superseded_projection"}
                        else "previous-pos"
                    ),
                    entry_exchange_order_id=(
                        "partial-entry"
                        if gap in {"missing_identity_close_fragment", "superseded_projection"}
                        else "previous-entry"
                    ),
                    close_exchange_order_id="partial-close-new",
                    settlement_status=(
                        "superseded_position_residual" if gap == "superseded_projection" else None
                    ),
                )
                session.add(conflicting_position)
                if gap == "missing_identity_close_fragment":
                    session.add(
                        Position(
                            model_name="ensemble_trader",
                            execution_mode="paper",
                            symbol="SLX/USDT",
                            side="long",
                            quantity=300,
                            entry_price=0.07,
                            leverage=2,
                            is_open=False,
                            closed_at=opened_at + timedelta(hours=1),
                            created_at=opened_at,
                            okx_inst_id="SLX-USDT-SWAP",
                            okx_pos_id="partial-pos",
                            entry_exchange_order_id="partial-entry",
                            close_exchange_order_id="partial-close-old",
                        )
                    )
            await session.flush()
            raw = {
                "instId": "SLX-USDT-SWAP", "posId": "partial-pos",
                "posSide": "net", "direction": "long", "type": "1",
                "cTime": str(int(opened_at.timestamp() * 1000)),
                "uTime": str(int(close_at.timestamp() * 1000)),
                "openMaxPos": "100", "closeTotalPos": "60",
                "openAvgPx": "0.07", "closeAvgPx": "0.072",
                "pnl": "1.2", "realizedPnl": "1.2", "fundingFee": "0",
                "fee": "-0.1" if gap == "fee" else "0",
            }
            history = await upsert_okx_position_history_row(
                session, raw, mode="paper", source="okx_settlement_fact_mirror",
                entry_order_ids=["partial-entry"], close_order_ids=["partial-close-old"],
                match_status="okx_account_position_history", position_ids=[], synced_at=now,
            )
            await session.flush()
            rows = okx_position_history_records_to_rows([history])
            if gap == "other_lifecycle":
                other = {
                    **rows[0], "posId": "another-pos",
                    "cTime": str(int((opened_at - timedelta(minutes=1)).timestamp() * 1000)),
                    "_dashboard_entry_order_ids": ["another-entry"],
                    "_dashboard_close_order_ids": ["partial-close-new"],
                }
                rows.append(other)
            original_raw = dict(history.raw_row)
            result = await repair_missing_okx_lifecycle_fragments(
                session, mode="paper", position_history_rows=rows, now=now,
            )
            assert result["created_count"] == 0
            assert result["repaired_order_link_count"] == int(repair_expected)
            assert position.is_open is True
            assert position.quantity == (500 if gap == "quantity" else 400)
            assert position.realized_pnl == pytest.approx(1.2)
            assert history.close_status == "partial"
            assert history.realized_pnl == pytest.approx(1.2)
            if gap == "missing_identity_close_fragment":
                assert result["repaired_identity_count"] == 1
                assert conflicting_position.okx_pos_id == "partial-pos"
                assert conflicting_position.entry_exchange_order_id == "partial-entry"
                assert abs(
                    (
                        conflicting_position.created_at.replace(tzinfo=UTC)
                        - opened_at
                    ).total_seconds()
                ) < 0.001
            if gap == "superseded_projection":
                assert conflicting_position.settlement_status == "superseded_position_residual"
                assert conflicting_position.quantity == 300
                assert conflicting_position.okx_pos_id is None
            if not repair_expected:
                assert history.close_order_ids == ["partial-close-old"]
                assert history.position_ids == []
                assert history.raw_row == original_raw
            else:
                assert set(history.close_order_ids) == {"partial-close-old", "partial-close-new"}
                assert history.position_ids == [str(position.id)]
                for key, value in original_raw.items():
                    assert history.raw_row[key] == value
                assert history.raw_row["_bb_partial_lifecycle_link_recovery"][
                    "added_close_order_ids"
                ] == ["partial-close-new"]
                assert (await repair_missing_okx_lifecycle_fragments(
                    session, mode="paper",
                    position_history_rows=okx_position_history_records_to_rows([history]),
                    now=now + timedelta(seconds=1),
                ))["repaired_order_link_count"] == 0
    finally:
        await close_db()


@pytest.mark.parametrize(
    "failure",
    [
        "partial",
        "not_full",
        "unconfirmed",
        "missing_decision",
        "duplicate_history",
        "nearby_position",
        "superseded_close_link",
    ],
)
@pytest.mark.asyncio
async def test_missing_full_lifecycle_recovery_rejects_incomplete_or_ambiguous_evidence(
    failure,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / f'full-lifecycle-{failure}.db').as_posix()}",
    )
    await init_db()
    try:
        async with get_session_ctx() as session:
            row, history_record, _, closed_at = await _seed_missing_full_lifecycle(
                session,
                failure=None if failure == "duplicate_history" else failure,
            )
            rows = [row]
            if failure == "duplicate_history":
                duplicate_record = OkxPositionHistory(
                    mode=history_record.mode,
                    row_identity=f"{history_record.row_identity}|duplicate",
                    inst_id=history_record.inst_id,
                    symbol=history_record.symbol,
                    pos_id=history_record.pos_id,
                    pos_side=history_record.pos_side,
                    side=history_record.side,
                    close_status=history_record.close_status,
                    opened_at=history_record.opened_at,
                    updated_at_okx=history_record.updated_at_okx,
                    open_avg_px=history_record.open_avg_px,
                    close_avg_px=history_record.close_avg_px,
                    open_max_pos=history_record.open_max_pos,
                    close_total_pos=history_record.close_total_pos,
                    leverage=history_record.leverage,
                    realized_pnl=history_record.realized_pnl,
                    pnl=history_record.pnl,
                    funding_fee=history_record.funding_fee,
                    fee=history_record.fee,
                    entry_order_ids=history_record.entry_order_ids,
                    close_order_ids=history_record.close_order_ids,
                    linked_order_ids=history_record.linked_order_ids,
                    position_ids=[],
                    match_status=history_record.match_status,
                    evidence_gaps=[],
                    source=history_record.source,
                    sync_status=history_record.sync_status,
                    raw_row={},
                )
                session.add(duplicate_record)
                await session.flush()
                duplicate_row = dict(row)
                duplicate_row["_dashboard_history_record_id"] = duplicate_record.id
                duplicate_row["_dashboard_history_row_identity"] = duplicate_record.row_identity
                rows.append(duplicate_row)
            elif failure == "nearby_position":
                session.add(
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SLX/USDT",
                        side="long",
                        quantity=1040.0,
                        entry_price=0.07,
                        current_price=0.072,
                        leverage=1.0,
                        is_open=False,
                        closed_at=closed_at,
                        okx_inst_id="SLX-USDT-SWAP",
                        okx_pos_id="pos-full",
                        entry_exchange_order_id="conflicting-entry-order",
                        close_exchange_order_id="conflicting-close-order",
                        created_at=closed_at - timedelta(seconds=30),
                    )
                )
                await session.flush()
            elif failure == "superseded_close_link":
                session.add(
                    Position(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="SLX/USDT",
                        side="long",
                        quantity=1040.0,
                        entry_price=0.07,
                        current_price=0.072,
                        leverage=2.0,
                        is_open=False,
                        closed_at=closed_at,
                        okx_inst_id="SLX-USDT-SWAP",
                        okx_pos_id="pos-full",
                        entry_exchange_order_id="entry-full-lifecycle",
                        close_exchange_order_id="close-full-lifecycle",
                        settlement_status="superseded_position_residual",
                        settlement_source="legacy_recovery",
                        realized_pnl=2.08,
                        created_at=history_record.opened_at,
                    )
                )
                await session.flush()

            result = await repair_missing_okx_lifecycle_fragments(
                session,
                mode="paper",
                position_history_rows=rows,
                now=closed_at + timedelta(minutes=1),
            )
            assert result["created_count"] == 0
            positions = (
                await session.execute(select(Position).where(Position.okx_pos_id == "pos-full"))
            ).scalars().all()
            assert len(positions) == (
                1 if failure in {"nearby_position", "superseded_close_link"} else 0
            )
            if failure == "superseded_close_link":
                assert positions[0].settlement_status == "superseded_position_residual"
                assert positions[0].quantity == pytest.approx(1040.0)
                assert positions[0].realized_pnl == pytest.approx(2.08)
    finally:
        await close_db()


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


@pytest.mark.parametrize("ambiguous_position_link", [False, True])
def test_stale_close_time_is_repaired_from_unique_authoritative_fills(
    ambiguous_position_link,
) -> None:
    opened_at = datetime(2026, 10, 8, 11, 0, tzinfo=UTC)
    closed_at = opened_at + timedelta(hours=4)
    old_closed_at = closed_at + timedelta(days=3)
    row = {
        "instId": "SLX-USDT-SWAP", "posId": "pos-time", "direction": "short",
        "cTime": str(int(opened_at.timestamp() * 1000)),
        "uTime": str(int(closed_at.timestamp() * 1000)),
        "openMaxPos": "100", "closeTotalPos": "100", "openAvgPx": "0.07",
        "_dashboard_entry_order_ids": ["entry-time"],
        "_dashboard_close_order_ids": ["close-time"],
        "_dashboard_history_record_id": 123,
    }
    entry = _order(
        order_id="entry-time", side="sell", quantity=1000, contracts=100,
        price=0.07, filled_at=opened_at, pnl=0,
    )
    close = _order(
        order_id="close-time", side="buy", quantity=1000, contracts=100,
        price=0.066, filled_at=closed_at, pnl=4,
    )
    position = Position(
        id=2, symbol="SLX/USDT", side="short", quantity=1000,
        okx_inst_id="SLX-USDT-SWAP", okx_pos_id="pos-time",
        entry_exchange_order_id="entry-time", close_exchange_order_id="close-time",
        created_at=opened_at, closed_at=old_closed_at, entry_price=0.07,
        settlement_status="settlement_unresolved", realized_pnl=4.0,
        settlement_raw={"original_fact": "preserved"},
    )
    lifecycle = _history_lifecycle(row, mode="paper")
    linked_positions = [position]
    if ambiguous_position_link:
        linked_positions.append(Position(
            id=3, symbol="SLX/USDT", side="short", quantity=1000,
            okx_inst_id="SLX-USDT-SWAP", okx_pos_id="pos-time",
            entry_exchange_order_id="entry-time", close_exchange_order_id="close-time",
            created_at=opened_at, closed_at=old_closed_at, entry_price=0.07,
            settlement_status="settlement_unresolved",
        ))
    result = _repair_linked_lifecycle_identity(
        lifecycle, positions_by_close_id={"close-time": linked_positions},
        orders_by_id={"entry-time": entry, "close-time": close},
        close_owners={"close-time": {"current"}}, now=closed_at,
    )
    if ambiguous_position_link:
        assert result == []
        assert position.closed_at == old_closed_at
        assert position.settlement_raw == {"original_fact": "preserved"}
        return
    assert len(result) == 1
    assert result[0]["kind"] == "okx_lifecycle_close_time_repaired"
    assert position.closed_at == closed_at
    assert position.settlement_raw["original_fact"] == "preserved"
    assert position.settlement_raw["close_time_repair"]["previous_closed_at"] == old_closed_at.isoformat()
    assert _repair_linked_lifecycle_identity(
        lifecycle, positions_by_close_id={"close-time": [position]},
        orders_by_id={"entry-time": entry, "close-time": close},
        close_owners={"close-time": {"current"}}, now=closed_at,
    ) == []


@pytest.mark.asyncio
async def test_terminal_partial_close_with_stale_time_recovers_and_conserves_lifecycle(
    tmp_path, monkeypatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings, "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'stale-fragment-time.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    opened_at = now - timedelta(days=5)
    closed_at = opened_at + timedelta(minutes=20)
    try:
        async with get_session_ctx() as session:
            for order_id, side, contracts, filled_at, pnl in (
                ("entry-time", "sell", 100, opened_at, 0),
                ("close-primary", "buy", 90, closed_at, 0.1),
                ("close-terminal", "buy", 10, closed_at, 0.2),
            ):
                session.add(_order(
                    order_id=order_id, side=side, quantity=contracts * 10,
                    contracts=contracts, price=0.07 if side == "sell" else 0.066,
                    filled_at=filled_at, pnl=pnl,
                ))
            positions = []
            for close_id, quantity, status, delta in (
                ("close-primary", 1000, "reconciled", timedelta(seconds=5)),
                ("close-terminal", 100, "settlement_unresolved", timedelta(minutes=44)),
            ):
                position = Position(
                    model_name="ensemble_trader", execution_mode="paper",
                    symbol="SLX/USDT", side="short", quantity=quantity,
                    entry_price=0.07, current_price=0.066, leverage=1,
                    entry_fee=0, close_fee=0, realized_pnl=0,
                    is_open=False, created_at=opened_at, closed_at=closed_at + delta,
                    okx_inst_id="SLX-USDT-SWAP", okx_pos_id="pos-time",
                    entry_exchange_order_id="entry-time", close_exchange_order_id=close_id,
                    settlement_status=status, settlement_raw={"original_fact": "preserved"},
                )
                session.add(position)
                positions.append(position)
            await session.flush()
            position_ids = [position.id for position in positions]
            await upsert_okx_position_history_row(
                session,
                {
                    "instId": "SLX-USDT-SWAP", "posId": "pos-time", "direction": "short",
                    "type": "2", "cTime": str(int(opened_at.timestamp() * 1000)),
                    "uTime": str(int(closed_at.timestamp() * 1000)),
                    "openMaxPos": "100", "closeTotalPos": "100", "openAvgPx": "0.07",
                    "closeAvgPx": "0.066", "realizedPnl": "0.3", "pnl": "0.3",
                    "fee": "0", "fundingFee": "0", "_bb_contract_spec": {"ctVal": "10"},
                },
                mode="paper", source="okx_settlement_fact_mirror",
                position_ids=position_ids, entry_order_ids=["entry-time"],
                close_order_ids=["close-primary", "close-terminal"],
                match_status="okx_account_position_history", synced_at=now,
            )
        report = await OkxPositionSettlementSyncService(mode="paper").sync_once()
        assert report["reconciled_count"] == 2
        assert report["exception_count"] == 0
        async with get_session_ctx() as session:
            positions = list((await session.execute(select(Position))).scalars())
            assert {position.settlement_status for position in positions} == {"reconciled"}
            assert sorted(position.quantity for position in positions) == [100, 900]
            assert sum(position.realized_pnl for position in positions) == pytest.approx(0.3)
            for position in positions:
                assert position.closed_at.replace(tzinfo=UTC) == closed_at
                assert "close_time_repair" in position.settlement_raw
        assert (await OkxPositionSettlementSyncService(mode="paper").sync_once())[
            "reconciled_count"
        ] == 0
    finally:
        await close_db()


@pytest.mark.asyncio
@pytest.mark.parametrize("superseded_close_exists", [False, True])
async def test_missing_authoritative_close_fill_creates_idempotent_fragment(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    superseded_close_exists: bool,
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
            if superseded_close_exists:
                session.add(
                    Position(
                        model_name="okx_authoritative_sync", execution_mode="paper",
                        symbol="SLX/USDT", side="short", quantity=90.0,
                        entry_price=0.07, current_price=0.066, leverage=1.0,
                        is_open=False, closed_at=closed_at, created_at=opened_at,
                        okx_inst_id="SLX-USDT-SWAP", okx_pos_id="pos-104",
                        entry_exchange_order_id=entry_id,
                        close_exchange_order_id="close-9",
                        settlement_status="superseded_position_residual",
                        realized_pnl=0.05, settlement_raw={"audit": "preserved"},
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
            assert result["created_count"] == int(not superseded_close_exists)
            repaired = (
                await session.execute(
                    select(Position).where(Position.close_exchange_order_id == "close-9")
                )
            ).scalar_one()
            assert repaired.quantity == pytest.approx(90.0)
            if superseded_close_exists:
                assert repaired.settlement_status == "superseded_position_residual"
                assert repaired.realized_pnl == pytest.approx(0.05)
                assert repaired.settlement_raw == {"audit": "preserved"}
            else:
                assert repaired.settlement_source == REPAIR_SOURCE
                assert repaired.settlement_raw["lifecycle_recovery"]["training_policy"] == (
                    "exclude_until_authoritative_settlement"
                )

            second = await repair_missing_okx_lifecycle_fragments(
                session,
                mode="paper",
                position_history_rows=[history],
                now=closed_at + timedelta(minutes=2),
            )
            assert second["created_count"] == 0

        if superseded_close_exists:
            return
        candidates = await OkxPositionSettlementSyncService(mode="paper")._load_candidates(
            closed_at + timedelta(minutes=3)
        )
        group = next(iter(_group_candidates_by_lifecycle(candidates).values()))
        _prepare_lifecycle_allocations(group, [history])
        assert sum(member.close_contracts for member in group) == pytest.approx(104.0)
        assert all(member.allocation_complete for member in group)
    finally:
        await close_db()
