from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.trade import OkxPositionHistory, Position
from services.okx_position_history_store import upsert_okx_position_history_row
from services.okx_position_settlement_sync import (
    DUPLICATE_CLOSED_POSITION_REASON,
    POSITION_HISTORY_MATCH_MAX_ATTEMPTS,
    POSITION_HISTORY_QUARANTINE_RETRY_SECONDS,
    SETTLEMENT_QUARANTINE_SOURCE,
    SETTLEMENT_STATUS_QUARANTINED,
    SUPERSEDED_POSITION_STATUS,
    OkxPositionSettlementSyncService,
    SettlementCandidate,
    SettlementFailure,
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
