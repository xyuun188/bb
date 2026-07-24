from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.trade import Position
from services.okx_position_history_store import upsert_okx_position_history_row
from services.okx_position_settlement_sync import (
    POSITION_HISTORY_MATCH_MAX_ATTEMPTS,
    SETTLEMENT_QUARANTINE_SOURCE,
    SETTLEMENT_STATUS_QUARANTINED,
    OkxPositionSettlementSyncService,
)


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
            settlement_status="settling",
            settlement_source="system_execution",
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
            histories = (await session.execute(select(Position))).scalars().all()
        assert position is not None
        assert position.settlement_source == "okx_position_history_settlement"
        assert position.settlement_status == "reconciled"
        assert position.realized_pnl == pytest.approx(4.09)
        assert len(histories) == 1
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
        assert "next_retry_seconds" not in report["samples"][0]
        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
        assert position is not None
        assert position.settlement_status == SETTLEMENT_STATUS_QUARANTINED
        assert position.settlement_source == SETTLEMENT_QUARANTINE_SOURCE
        assert position.settlement_raw["retry_policy"] == "permanent_no_retry"
        assert "next_settlement_retry_at" not in position.settlement_raw
        assert expected_trigger in position.settlement_raw["quarantine_evidence"]["triggers"]
        assert await service._load_candidates(now + timedelta(days=1)) == []
    finally:
        await close_db()
