from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.trade import OkxPositionHistory, Position
from services.okx_position_history_store import upsert_okx_position_history_row
from services.okx_position_settlement_sync import (
    DISTINCT_CLOSED_FRAGMENT_REACTIVATED_REASON,
    DUPLICATE_CLOSED_POSITION_REASON,
    POSITION_HISTORY_MATCH_MAX_ATTEMPTS,
    POSITION_HISTORY_QUARANTINE_RETRY_SECONDS,
    SETTLEMENT_LIFECYCLE_OPEN_REASON,
    SETTLEMENT_LIFECYCLE_OPEN_SOURCE,
    SETTLEMENT_QUARANTINE_SOURCE,
    SETTLEMENT_STATUS_QUARANTINED,
    SUPERSEDED_POSITION_STATUS,
    OkxPositionSettlementSyncService,
    SettlementCandidate,
    SettlementFailure,
    _claim_history_row_for_position,
    _final_fragment_requires_quantity_repair,
    _group_candidates_by_lifecycle,
    _match_position_history_row,
    _prepare_lifecycle_allocations,
    _reactivate_distinct_superseded_fragment,
)


@pytest.mark.asyncio
async def test_position_settlement_loads_history_mirror_once_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = OkxPositionSettlementSyncService(mode="paper", limit=2)
    candidates = [
        SettlementCandidate(
            position_id=position_id,
            symbol="ADA/USDT",
            side="long",
            quantity=100.0,
            entry_price=0.6,
            current_price=0.64,
            leverage=1.0,
            entry_fee=0.05,
            close_fee=0.06,
            okx_inst_id="ADA-USDT-SWAP",
            okx_pos_id=f"ada-pos-{position_id}",
            entry_exchange_order_id=f"entry-{position_id}",
            close_exchange_order_id=f"close-{position_id}",
            created_at=datetime.now(UTC) - timedelta(minutes=20),
            closed_at=datetime.now(UTC),
            settlement_status="settling",
            settlement_raw={},
        )
        for position_id in (1, 2)
    ]
    history_rows = [{"posId": "shared-history"}]
    load_calls = 0
    seen_rows: list[list[dict]] = []
    failure_batches: list[list[int]] = []

    async def backfill(_now: datetime) -> list[dict]:
        return []

    async def load_candidates(_now: datetime) -> list[SettlementCandidate]:
        return candidates

    async def load_rows() -> list[dict]:
        nonlocal load_calls
        load_calls += 1
        return history_rows

    async def settle_candidate(
        _candidate: SettlementCandidate,
        _now: datetime,
        *,
        position_history_rows: list[dict] | None = None,
    ) -> SettlementFailure:
        seen_rows.append(position_history_rows or [])
        return SettlementFailure("missing", "missing", {})

    async def apply_failures(
        failures: list[tuple[SettlementCandidate, SettlementFailure]],
        _now: datetime,
    ) -> dict[int, bool]:
        failure_batches.append([candidate.position_id for candidate, _failure in failures])
        return {candidate.position_id: False for candidate, _failure in failures}

    monkeypatch.setattr(service, "_backfill_decision_outcomes", backfill)
    monkeypatch.setattr(service, "_load_candidates", load_candidates)
    monkeypatch.setattr(service, "_load_position_history_rows", load_rows)
    monkeypatch.setattr(service, "_settle_candidate", settle_candidate)
    monkeypatch.setattr(service, "_apply_failures", apply_failures)

    report = await service.sync_once()

    assert report["checked_count"] == 2
    assert load_calls == 1
    assert seen_rows == [history_rows, history_rows]
    assert failure_batches == [[1, 2]]


def _ms(value: datetime) -> str:
    return str(int(value.timestamp() * 1000))


async def _init_test_db(tmp_path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / name).as_posix()}",
    )
    await init_db()


async def _seed_closed_position(
    now: datetime,
    *,
    settlement_raw: dict | None = None,
    settlement_status: str = "settling",
    settlement_source: str = "system_execution",
) -> int:
    async with get_session_ctx() as session:
        position = Position(
            model_name="rule_strategy",
            execution_mode="paper",
            symbol="ADA/USDT",
            side="long",
            quantity=100.0,
            entry_price=0.6,
            current_price=0.64,
            leverage=1.0,
            realized_pnl=0.0,
            close_fill_pnl=0.0,
            entry_fee=0.05,
            close_fee=0.06,
            funding_fee=0.0,
            settlement_status=settlement_status,
            settlement_source=settlement_source,
            settlement_raw=settlement_raw or {},
            is_open=False,
            closed_at=now,
            okx_inst_id="ADA-USDT-SWAP",
            okx_pos_id="ada-pos-1",
            entry_exchange_order_id="entry-1",
            close_exchange_order_id="close-1",
            created_at=now - timedelta(minutes=20),
        )
        session.add(position)
        await session.flush()
        return int(position.id)


@pytest.mark.asyncio
async def test_position_settlement_reads_only_local_settlement_fact_mirror(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "position-settlement.db")
    now = datetime.now(UTC)
    try:
        position_id = await _seed_closed_position(now)
        async with get_session_ctx() as session:
            await upsert_okx_position_history_row(
                session,
                {
                    "instId": "ADA-USDT-SWAP",
                    "posId": "ada-pos-1",
                    "posSide": "long",
                    "type": "2",
                    "cTime": _ms(now - timedelta(minutes=20)),
                    "uTime": _ms(now),
                    "openAvgPx": "0.6",
                    "closeAvgPx": "0.64",
                    "openMaxPos": "100",
                    "closeTotalPos": "100",
                    "realizedPnl": "4.09",
                    "pnl": "4.2",
                    "fundingFee": "-0.01",
                    "fee": "-0.1",
                },
                mode="paper",
                source="okx_settlement_fact_mirror",
                match_status="okx_account_position_history",
                synced_at=now,
            )

        report = await OkxPositionSettlementSyncService(mode="paper").sync_once()

        assert report["status"] == "ok"
        assert report["reconciled_count"] == 1
        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            positions = (await session.execute(select(Position))).scalars().all()
            history = (
                await session.execute(select(OkxPositionHistory))
            ).scalars().one()
        assert position is not None
        assert position.settlement_source == "okx_position_history_settlement"
        assert position.settlement_status == "reconciled"
        assert position.realized_pnl == pytest.approx(4.09)
        assert len(positions) == 1
        assert history.position_ids == [str(position_id)]
        assert history.entry_order_ids == ["entry-1"]
        assert history.close_order_ids == ["close-1"]
        assert history.linked_order_ids == ["close-1", "entry-1"]
        assert history.match_status == "okx_position_settlement_linked"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_settlement_retires_final_duplicate_lifecycle_with_clock_drift(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "position-settlement-dedup.db")
    now = datetime.now(UTC)
    try:
        canonical_id = await _seed_closed_position(
            now,
            settlement_status="reconciled",
            settlement_source="okx_position_history_settlement",
        )
        duplicate_id = await _seed_closed_position(
            now + timedelta(seconds=3),
            settlement_status="reconciled",
            settlement_source="okx_position_history_settlement",
        )

        candidates = await OkxPositionSettlementSyncService(mode="paper")._load_candidates(
            now + timedelta(minutes=1)
        )

        assert candidates == []
        async with get_session_ctx() as session:
            canonical = await session.get(Position, canonical_id)
            duplicate = await session.get(Position, duplicate_id)
        assert canonical is not None
        assert duplicate is not None
        assert canonical.settlement_status == "reconciled"
        assert duplicate.settlement_status == SUPERSEDED_POSITION_STATUS
        assert duplicate.settlement_raw["reason"] == DUPLICATE_CLOSED_POSITION_REASON
        assert duplicate.settlement_raw["canonical_position_id"] == canonical_id
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_settlement_prefers_complete_close_order_projection(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "position-settlement-complete-projection.db")
    now = datetime.now(UTC)
    try:
        canonical_id = await _seed_closed_position(now)
        duplicate_id = await _seed_closed_position(now + timedelta(seconds=3))
        async with get_session_ctx() as session:
            canonical = await session.get(Position, canonical_id)
            duplicate = await session.get(Position, duplicate_id)
            assert canonical is not None
            assert duplicate is not None
            canonical.close_exchange_order_id = "close-1,close-2,close-3"
            duplicate.close_exchange_order_id = "close-1"
            await session.flush()

        candidates = await OkxPositionSettlementSyncService(mode="paper")._load_candidates(
            now + timedelta(minutes=1)
        )

        assert [candidate.position_id for candidate in candidates] == [canonical_id]
        async with get_session_ctx() as session:
            canonical = await session.get(Position, canonical_id)
            duplicate = await session.get(Position, duplicate_id)
        assert canonical is not None
        assert duplicate is not None
        assert canonical.settlement_status == "settling"
        assert duplicate.settlement_status == SUPERSEDED_POSITION_STATUS
        assert duplicate.settlement_raw["canonical_position_id"] == canonical_id
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_settlement_does_not_merge_different_quantity_lifecycles(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "position-settlement-distinct-quantity.db")
    now = datetime.now(UTC)
    try:
        first_id = await _seed_closed_position(now)
        second_id = await _seed_closed_position(now + timedelta(seconds=3))
        async with get_session_ctx() as session:
            second = await session.get(Position, second_id)
            assert second is not None
            second.quantity = 50.0
            await session.flush()

        candidates = await OkxPositionSettlementSyncService(mode="paper")._load_candidates(
            now + timedelta(minutes=1)
        )

        assert {candidate.position_id for candidate in candidates} == {first_id, second_id}
        async with get_session_ctx() as session:
            first = await session.get(Position, first_id)
            second = await session.get(Position, second_id)
        assert first is not None
        assert second is not None
        assert first.settlement_status == "settling"
        assert second.settlement_status == "settling"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_settlement_rejects_reused_pos_id_from_older_lifecycle(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "position-settlement-reused-pos-id.db")
    now = datetime.now(UTC)
    try:
        position_id = await _seed_closed_position(now)
        async with get_session_ctx() as session:
            await upsert_okx_position_history_row(
                session,
                {
                    "instId": "ADA-USDT-SWAP",
                    "posId": "ada-pos-1",
                    "posSide": "long",
                    "type": "2",
                    "cTime": _ms(now - timedelta(days=1, minutes=20)),
                    "uTime": _ms(now - timedelta(days=1)),
                    "openAvgPx": "0.6",
                    "closeAvgPx": "0.64",
                    "openMaxPos": "100",
                    "closeTotalPos": "100",
                    "realizedPnl": "4.09",
                    "pnl": "4.2",
                    "fundingFee": "-0.01",
                    "fee": "-0.1",
                },
                mode="paper",
                source="okx_settlement_fact_mirror",
                match_status="okx_account_position_history",
                synced_at=now,
            )

        report = await OkxPositionSettlementSyncService(
            mode="paper",
            retry_seconds=30.0,
        ).sync_once()

        assert report["status"] == "warning"
        assert report["reconciled_count"] == 0
        assert report["samples"][0]["error_code"] == "positions_history_no_matching_row"
        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            history = (
                await session.execute(select(OkxPositionHistory))
            ).scalars().one()
        assert position is not None
        assert position.realized_pnl == pytest.approx(0.0)
        assert position.settlement_status == "settlement_exception"
        assert history.position_ids == []
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_settlement_marks_missing_mirror_fact_for_retry_without_network(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "position-settlement-retry.db")
    now = datetime.now(UTC)
    try:
        position_id = await _seed_closed_position(now)

        report = await OkxPositionSettlementSyncService(
            mode="paper",
            retry_seconds=30.0,
        ).sync_once()

        assert report["status"] == "warning"
        assert report["exception_count"] == 1
        assert report["samples"][0]["error_code"] == "position_history_mirror_no_rows"
        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
        assert position is not None
        assert position.settlement_status == "settlement_exception"
        assert position.settlement_raw["last_error_code"] == "position_history_mirror_no_rows"
        assert "next_settlement_retry_at" in position.settlement_raw
    finally:
        await close_db()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("closed_age_hours", "previous_attempts", "expected_trigger"),
    (
        (7.0, 0, "closed_age_limit"),
        (0.0, POSITION_HISTORY_MATCH_MAX_ATTEMPTS - 1, "attempt_limit"),
    ),
)
async def test_position_settlement_quarantines_persistent_identity_mismatch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    closed_age_hours: float,
    previous_attempts: int,
    expected_trigger: str,
) -> None:
    await _init_test_db(
        tmp_path,
        monkeypatch,
        f"position-settlement-quarantine-{expected_trigger}.db",
    )
    now = datetime.now(UTC)
    closed_at = now - timedelta(hours=closed_age_hours)
    try:
        position_id = await _seed_closed_position(
            closed_at,
            settlement_raw={"settlement_attempt_count": previous_attempts},
        )
        async with get_session_ctx() as session:
            await upsert_okx_position_history_row(
                session,
                {
                    "instId": "ADA-USDT-SWAP",
                    "posId": "different-lifecycle",
                    "posSide": "long",
                    "type": "2",
                    "cTime": _ms(closed_at - timedelta(minutes=20)),
                    "uTime": _ms(closed_at),
                    "openAvgPx": "0.6",
                    "closeAvgPx": "0.64",
                    "openMaxPos": "100",
                    "closeTotalPos": "100",
                    "realizedPnl": "4.09",
                    "pnl": "4.2",
                    "fundingFee": "-0.01",
                    "fee": "-0.1",
                },
                mode="paper",
                source="okx_settlement_fact_mirror",
                match_status="okx_account_position_history",
                synced_at=now,
            )

        service = OkxPositionSettlementSyncService(mode="paper", retry_seconds=30.0)
        report = await service.sync_once()

        assert report["status"] == "warning"
        assert report["exception_count"] == 1
        assert report["samples"][0]["kind"] == "okx_position_settlement_quarantined"
        assert report["samples"][0]["next_retry_seconds"] == pytest.approx(
            POSITION_HISTORY_QUARANTINE_RETRY_SECONDS
        )
        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
        assert position is not None
        assert position.settlement_status == SETTLEMENT_STATUS_QUARANTINED
        assert position.settlement_source == SETTLEMENT_QUARANTINE_SOURCE
        assert position.settlement_raw["retry_policy"].startswith(
            f"quarantined from authority; retry every {POSITION_HISTORY_QUARANTINE_RETRY_SECONDS:g}s"
        )
        assert "next_settlement_retry_at" in position.settlement_raw
        assert expected_trigger in position.settlement_raw["quarantine_evidence"]["triggers"]
        assert len(await service._load_candidates(now + timedelta(minutes=1))) == 0
        assert len(await service._load_candidates(now + timedelta(days=1))) == 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_settlement_recovers_quarantined_position_when_official_row_arrives(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "position-settlement-quarantine-recovery.db")
    now = datetime.now(UTC)
    closed_at = now - timedelta(minutes=20)
    try:
        position_id = await _seed_closed_position(
            closed_at,
            settlement_status=SETTLEMENT_STATUS_QUARANTINED,
            settlement_source=SETTLEMENT_QUARANTINE_SOURCE,
            settlement_raw={
                "settlement_attempt_count": POSITION_HISTORY_MATCH_MAX_ATTEMPTS,
                "next_settlement_retry_at": (now - timedelta(seconds=1)).isoformat(),
                "retry_policy": "quarantined from authority; retry until available",
            },
        )
        async with get_session_ctx() as session:
            await upsert_okx_position_history_row(
                session,
                {
                    "instId": "ADA-USDT-SWAP",
                    "posId": "ada-pos-1",
                    "posSide": "long",
                    "type": "2",
                    "cTime": _ms(closed_at - timedelta(minutes=20)),
                    "uTime": _ms(closed_at),
                    "openAvgPx": "0.6",
                    "closeAvgPx": "0.64",
                    "openMaxPos": "100",
                    "closeTotalPos": "100",
                    "realizedPnl": "4.09",
                    "pnl": "4.2",
                    "fundingFee": "-0.01",
                    "fee": "-0.1",
                },
                mode="paper",
                source="okx_settlement_fact_mirror",
                match_status="okx_account_position_history",
                synced_at=now,
            )

        report = await OkxPositionSettlementSyncService(
            mode="paper",
            retry_seconds=30.0,
        ).sync_once()

        assert report["status"] == "ok"
        assert report["reconciled_count"] == 1
        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
        assert position is not None
        assert position.settlement_status == "reconciled"
        assert position.settlement_source == "okx_position_history_settlement"
        assert position.realized_pnl == pytest.approx(4.09)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_open_okx_lifecycle_keeps_closed_child_settling_instead_of_quarantine(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "position-settlement-open-lifecycle.db")
    now = datetime.now(UTC)
    closed_at = now - timedelta(hours=8)
    try:
        position_id = await _seed_closed_position(
            closed_at,
            settlement_status=SETTLEMENT_STATUS_QUARANTINED,
            settlement_source=SETTLEMENT_QUARANTINE_SOURCE,
            settlement_raw={
                "settlement_attempt_count": POSITION_HISTORY_MATCH_MAX_ATTEMPTS,
                "quarantine_reason": "official_position_history_identity_unresolved",
            },
        )
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="rule_strategy",
                    execution_mode="paper",
                    symbol="ADA/USDT",
                    side="long",
                    quantity=50.0,
                    entry_price=0.6,
                    current_price=0.64,
                    leverage=1.0,
                    is_open=True,
                    created_at=closed_at - timedelta(minutes=20),
                    okx_inst_id="ADA-USDT-SWAP",
                    okx_pos_id="ada-pos-1",
                    entry_exchange_order_id="entry-1",
                )
            )
            await session.flush()

        candidates = await OkxPositionSettlementSyncService(mode="paper")._load_candidates(now)

        assert candidates == []
        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
        assert position is not None
        assert position.settlement_status == "settling"
        assert position.settlement_source == SETTLEMENT_LIFECYCLE_OPEN_SOURCE
        assert position.settlement_raw["reason"] == SETTLEMENT_LIFECYCLE_OPEN_REASON
        assert position.settlement_raw["previous_settlement_status"] == (
            SETTLEMENT_STATUS_QUARANTINED
        )
        assert position.settlement_raw["lifecycle_open_original_settlement_status"] == (
            SETTLEMENT_STATUS_QUARANTINED
        )
        assert position.settlement_raw["lifecycle_open_previous_attempt_count"] == (
            POSITION_HISTORY_MATCH_MAX_ATTEMPTS
        )
        assert position.settlement_raw["settlement_attempt_count"] == 0
        assert "next_settlement_retry_at" in position.settlement_raw

        candidates = await OkxPositionSettlementSyncService(mode="paper")._load_candidates(
            now + timedelta(hours=1)
        )
        assert candidates == []
        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
        assert position is not None
        assert position.settlement_raw["lifecycle_open_original_settlement_status"] == (
            SETTLEMENT_STATUS_QUARANTINED
        )
        assert position.settlement_raw["lifecycle_open_previous_attempt_count"] == (
            POSITION_HISTORY_MATCH_MAX_ATTEMPTS
        )
        assert position.settlement_raw["settlement_attempt_count"] == 0
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_different_open_okx_pos_id_does_not_block_closed_lifecycle(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "position-settlement-different-open-lifecycle.db")
    now = datetime.now(UTC)
    try:
        position_id = await _seed_closed_position(now)
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="rule_strategy",
                    execution_mode="paper",
                    symbol="ADA/USDT",
                    side="long",
                    quantity=50.0,
                    entry_price=0.6,
                    current_price=0.64,
                    leverage=1.0,
                    is_open=True,
                    created_at=now - timedelta(minutes=20),
                    okx_inst_id="ADA-USDT-SWAP",
                    okx_pos_id="different-pos-id",
                    entry_exchange_order_id="entry-2",
                )
            )
            await session.flush()

        candidates = await OkxPositionSettlementSyncService(mode="paper")._load_candidates(
            now + timedelta(minutes=1)
        )

        assert [candidate.position_id for candidate in candidates] == [position_id]
    finally:
        await close_db()


def test_official_history_row_is_not_reused_by_another_local_position() -> None:
    now = datetime.now(UTC)
    rows = [
        {
            "instId": "ADA-USDT-SWAP",
            "posId": "ada-pos-1",
            "posSide": "long",
            "cTime": _ms(now - timedelta(minutes=20)),
            "uTime": _ms(now),
            "realizedPnl": "4.09",
        }
    ]
    first = SettlementCandidate(
        position_id=101,
        symbol="ADA/USDT",
        side="long",
        quantity=50.0,
        entry_price=0.6,
        current_price=0.64,
        leverage=1.0,
        entry_fee=0.05,
        close_fee=0.06,
        okx_inst_id="ADA-USDT-SWAP",
        okx_pos_id="ada-pos-1",
        entry_exchange_order_id="entry-1",
        close_exchange_order_id="close-1",
        created_at=now - timedelta(minutes=20),
        closed_at=now,
        settlement_status="settling",
        settlement_raw={},
    )
    second = replace(first, position_id=102)

    matched = _match_position_history_row(first, rows, inst_id="ADA-USDT-SWAP")
    assert not isinstance(matched, SettlementFailure)
    _claim_history_row_for_position(matched[0], first.position_id)

    reused = _match_position_history_row(second, rows, inst_id="ADA-USDT-SWAP")
    assert isinstance(reused, SettlementFailure)
    assert reused.code == "positions_history_no_matching_row"


def test_shared_official_lifecycle_allocates_pnl_only_by_close_contracts() -> None:
    now = datetime.now(UTC)
    first = SettlementCandidate(
        position_id=201,
        symbol="ALGO/USDT",
        side="short",
        quantity=8300.0,
        entry_price=0.2,
        current_price=0.19,
        leverage=1.0,
        entry_fee=0.04,
        close_fee=0.03,
        okx_inst_id="ALGO-USDT-SWAP",
        okx_pos_id="shared-pos",
        entry_exchange_order_id="entry-1",
        close_exchange_order_id="close-1",
        created_at=now - timedelta(minutes=20),
        closed_at=now - timedelta(minutes=2),
        settlement_status="settling",
            settlement_raw={},
            close_contracts=83.0,
            entry_fee_authoritative=0.04,
            close_fee_authoritative=0.0444374,
            close_fill_pnl_authoritative=-1.654,
    )
    second = replace(
        first,
        position_id=202,
        quantity=5300.0,
        close_exchange_order_id="close-2",
        closed_at=now,
        close_contracts=53.0,
        close_fee_authoritative=0.0444374,
        close_fill_pnl_authoritative=-2.486,
    )
    groups = _group_candidates_by_lifecycle([first, second])
    group = next(iter(groups.values()))
    _prepare_lifecycle_allocations(
        group,
        [
            {
                "instId": "ALGO-USDT-SWAP",
                "posId": "shared-pos",
                "posSide": "short",
                "cTime": str(int((now - timedelta(minutes=20)).timestamp() * 1000)),
                "uTime": str(int(now.timestamp() * 1000)),
                "openMaxPos": "136",
                "closeTotalPos": "136",
                "realizedPnl": "-4.2688748",
            }
        ],
    )
    assert [member.allocation_complete for member in group] == [True, True]
    assert [member.allocation_ratio for member in group] == pytest.approx([83 / 136, 53 / 136])


def test_single_lifecycle_prefers_authoritative_order_economics() -> None:
    from services.okx_position_settlement_sync import (
        SettlementSuccess,
        _scale_settlement_success,
    )
    from services.position_settlement import build_position_settlement_snapshot

    now = datetime.now(UTC)
    candidate = SettlementCandidate(
        position_id=204,
        symbol="ALGO/USDT",
        side="short",
        quantity=8300.0,
        entry_price=0.2,
        current_price=0.19,
        leverage=1.0,
        entry_fee=0.03,
        close_fee=0.03,
        okx_inst_id="ALGO-USDT-SWAP",
        okx_pos_id="single-pos",
        entry_exchange_order_id="entry-1",
        close_exchange_order_id="close-1",
        created_at=now - timedelta(minutes=20),
        closed_at=now,
        settlement_status="settling",
        settlement_raw={},
        entry_fee_authoritative=0.021,
        close_fee_authoritative=0.017,
        close_fill_pnl_authoritative=-1.654,
    )
    success = SettlementSuccess(
        row={},
        snapshot=build_position_settlement_snapshot(
            close_fill_pnl=-9.0,
            entry_fee=0.3,
            close_fee=0.3,
            funding_fee=0.12,
        ),
        match_reason="test",
        fee_source="test",
        funding_fee_source="test",
    )
    scaled = _scale_settlement_success(success, candidate)
    assert scaled.snapshot.close_fill_pnl == pytest.approx(-1.654)
    assert scaled.snapshot.entry_fee == pytest.approx(0.021)
    assert scaled.snapshot.close_fee == pytest.approx(0.017)
    assert scaled.snapshot.funding_fee == pytest.approx(0.12)


def test_shared_lifecycle_is_quarantined_when_fragment_contracts_do_not_conserve() -> None:
    now = datetime.now(UTC)
    first = SettlementCandidate(
        position_id=301,
        symbol="ALGO/USDT",
        side="short",
        quantity=5200.0,
        entry_price=0.2,
        current_price=0.19,
        leverage=1.0,
        entry_fee=0.04,
        close_fee=0.03,
        okx_inst_id="ALGO-USDT-SWAP",
        okx_pos_id="bad-shared-pos",
        entry_exchange_order_id="entry-1",
        close_exchange_order_id="close-1",
        created_at=now - timedelta(minutes=20),
        closed_at=now,
        settlement_status="settling",
        settlement_raw={},
        close_contracts=52.0,
    )
    second = replace(first, position_id=302, quantity=4000.0, close_exchange_order_id="close-2", close_contracts=40.0)
    group = next(iter(_group_candidates_by_lifecycle([first, second]).values()))
    _prepare_lifecycle_allocations(
        group,
        [{"instId": "ALGO-USDT-SWAP", "posId": "bad-shared-pos", "posSide": "short", "cTime": str(int((now - timedelta(minutes=20)).timestamp() * 1000)), "uTime": str(int(now.timestamp() * 1000)), "closeTotalPos": "136", "realizedPnl": "-4.2"}],
    )
    assert [member.allocation_complete for member in group] == [False, False]


def test_authoritative_contract_size_flags_local_fragment_quantity_drift() -> None:
    position = Position(
        quantity=520.0,
        settlement_raw={
            "lifecycle_allocation": {"allocated_contracts": 53.0},
            "okx_position_history_row": {
                "_bb_contract_spec": {"ctVal": "10"},
            },
        },
    )
    assert _final_fragment_requires_quantity_repair(position) is True


def test_legacy_superseded_disjoint_close_fragment_is_reactivated() -> None:
    now = datetime.now(UTC)
    canonical = Position(id=10, close_exchange_order_id="close-a")
    fragment = Position(
        id=11,
        settlement_status=SUPERSEDED_POSITION_STATUS,
        close_exchange_order_id="close-b",
    )
    raw = {
        "reason": DUPLICATE_CLOSED_POSITION_REASON,
        "canonical_position_id": 10,
    }
    assert _reactivate_distinct_superseded_fragment(
        fragment,
        [canonical, fragment],
        raw=raw,
        now=now,
    )
    assert fragment.settlement_status == "settling"
    assert fragment.settlement_raw["reason"] == DISTINCT_CLOSED_FRAGMENT_REACTIVATED_REASON


def test_legacy_superseded_overlapping_projection_stays_retired() -> None:
    now = datetime.now(UTC)
    canonical = Position(id=10, close_exchange_order_id="close-a,close-b")
    duplicate = Position(id=11, close_exchange_order_id="close-a")
    assert not _reactivate_distinct_superseded_fragment(
        duplicate,
        [canonical, duplicate],
        raw={
            "reason": DUPLICATE_CLOSED_POSITION_REASON,
            "canonical_position_id": 10,
        },
        now=now,
    )
