"""Account-level OKX positions-history mirror sync.

The dashboard reads closed positions from the local OKX positions-history
mirror. This service keeps that mirror current even when there are no matching
local Position rows and no order-fact repair is currently due.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import select

from core.safe_output import safe_error_text
from db.session import get_session_ctx
from executor.okx_executor import OKXExecutor
from models.trade import OkxPositionHistory
from services.okx_native_facts import OkxNativeFactsClient
from services.okx_position_history_store import (
    okx_position_history_row_identity,
    upsert_okx_position_history_row,
)

logger = structlog.get_logger(__name__)

DEFAULT_POSITION_HISTORY_LOOKBACK_HOURS = 72
DEFAULT_POSITION_HISTORY_LIMIT = 100
DEFAULT_POSITION_HISTORY_MAX_PAGES = 5
DEFAULT_POSITION_HISTORY_TIMEOUT_SECONDS = 8.0

SessionContextFactory = Callable[[], AbstractAsyncContextManager[Any]]


@dataclass(frozen=True, slots=True)
class OkxPositionHistoryMirrorSyncSummary:
    status: str
    mode: str
    source: str
    checked_at: datetime
    since: datetime
    okx_pull_available: bool
    live_count: int = 0
    upserted_count: int = 0
    inserted_count: int = 0
    updated_count: int = 0
    unchanged_count: int = 0
    skipped_count: int = 0
    latest_u_time_ms: float = 0.0
    latest_inst_id: str = ""
    latest_pos_id: str = ""
    error: str | None = None
    samples: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.mode,
            "source": self.source,
            "checked_at": self.checked_at.astimezone(UTC).isoformat(),
            "since": self.since.astimezone(UTC).isoformat(),
            "okx_pull_available": self.okx_pull_available,
            "live_count": self.live_count,
            "upserted_count": self.upserted_count,
            "inserted_count": self.inserted_count,
            "updated_count": self.updated_count,
            "unchanged_count": self.unchanged_count,
            "skipped_count": self.skipped_count,
            "latest_u_time_ms": self.latest_u_time_ms,
            "latest_inst_id": self.latest_inst_id,
            "latest_pos_id": self.latest_pos_id,
            "error": self.error,
            "samples": list(self.samples),
        }


class OkxPositionHistoryMirrorSyncService:
    """Continuously mirror OKX's official historical position rows."""

    def __init__(
        self,
        *,
        mode: str = "paper",
        lookback_hours: int = DEFAULT_POSITION_HISTORY_LOOKBACK_HOURS,
        limit: int = DEFAULT_POSITION_HISTORY_LIMIT,
        max_pages: int = DEFAULT_POSITION_HISTORY_MAX_PAGES,
        timeout_seconds: float = DEFAULT_POSITION_HISTORY_TIMEOUT_SECONDS,
        executor_factory: Any | None = None,
        session_context_factory: SessionContextFactory = get_session_ctx,
    ) -> None:
        self.mode = "live" if str(mode or "").lower() == "live" else "paper"
        self.lookback_hours = max(1, min(int(lookback_hours or 1), 24 * 30))
        self.limit = max(1, min(int(limit or DEFAULT_POSITION_HISTORY_LIMIT), 100))
        self.max_pages = max(1, min(int(max_pages or DEFAULT_POSITION_HISTORY_MAX_PAGES), 20))
        self.timeout_seconds = max(1.0, float(timeout_seconds or 1.0))
        self.executor_factory = executor_factory or OKXExecutor
        self.session_context_factory = session_context_factory

    async def sync_once(self) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        since = started_at - timedelta(hours=self.lookback_hours)
        executor = self.executor_factory(mode=self.mode, load_markets_on_initialize=False)
        rows: list[dict[str, Any]] = []
        contract_specs: dict[str, dict[str, Any]] = {}
        contract_spec_error = ""
        fatal_error: str | None = None
        try:
            await asyncio.wait_for(executor.initialize(), timeout=min(self.timeout_seconds, 3.0))
            native_facts = OkxNativeFactsClient(executor)
            try:
                rows = await asyncio.wait_for(
                    native_facts.fetch_position_history_rows(
                        inst_ids=None,
                        pos_ids=None,
                        since=since,
                        limit=self.limit,
                        max_pages=self.max_pages,
                        strict=True,
                    ),
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                fatal_error = safe_error_text(exc, limit=220)
                logger.warning(
                    "OKX position history mirror sync failed",
                    mode=self.mode,
                    error=fatal_error,
                )
            inst_ids = {
                str(row.get("instId") or "").strip().upper()
                for row in rows
                if str(row.get("instId") or "").strip()
            }
            if not inst_ids:
                inst_ids = await self._stored_inst_ids()
            if inst_ids:
                try:
                    contract_specs = await asyncio.wait_for(
                        native_facts.fetch_contract_specs(inst_ids=inst_ids),
                        timeout=min(self.timeout_seconds, 5.0),
                    )
                except Exception as exc:
                    contract_spec_error = safe_error_text(exc, limit=220)
                    logger.warning(
                        "OKX position history contract spec enrichment failed",
                        mode=self.mode,
                        error=contract_spec_error,
                    )
        except Exception as exc:
            if not fatal_error:
                fatal_error = safe_error_text(exc, limit=220)
            logger.warning(
                "OKX position history mirror sync failed",
                mode=self.mode,
                error=fatal_error,
            )
        finally:
            try:
                await executor.shutdown()
            except Exception as exc:
                logger.debug(
                    "OKX position history mirror sync shutdown failed",
                    error=safe_error_text(exc, limit=120),
                )

        if contract_specs:
            await self._persist_contract_specs(contract_specs, synced_at=started_at)

        if fatal_error:
            return OkxPositionHistoryMirrorSyncSummary(
                status="degraded",
                mode=self.mode,
                source="okx_position_history_account_sync",
                checked_at=started_at,
                since=since,
                okx_pull_available=False,
                error=fatal_error,
            ).as_dict()

        for row in rows:
            inst_id = str(row.get("instId") or "").strip().upper()
            spec = contract_specs.get(inst_id)
            if spec:
                row["_bb_contract_spec"] = dict(spec)
            elif contract_spec_error:
                row["_bb_contract_spec_error"] = contract_spec_error

        return (
            await self._persist_rows(rows, checked_at=started_at, since=since)
        ).as_dict()

    async def _stored_inst_ids(self) -> set[str]:
        async with self.session_context_factory() as session:
            result = await session.execute(
                select(OkxPositionHistory.inst_id).where(
                    OkxPositionHistory.mode == self.mode
                )
            )
            return {
                str(value or "").strip().upper()
                for value in result.scalars().all()
                if str(value or "").strip()
            }

    async def _persist_contract_specs(
        self,
        specs: dict[str, dict[str, Any]],
        *,
        synced_at: datetime,
    ) -> int:
        if not specs:
            return 0
        async with self.session_context_factory() as session:
            result = await session.execute(
                select(OkxPositionHistory).where(
                    OkxPositionHistory.mode == self.mode,
                    OkxPositionHistory.inst_id.in_(sorted(specs)),
                )
            )
            updated = 0
            for record in result.scalars().all():
                spec = specs.get(str(record.inst_id or "").strip().upper())
                if not spec:
                    continue
                raw = dict(record.raw_row or {})
                if str(raw.get("_bb_contract_spec_source") or "").startswith(
                    "okx_account_position_"
                ):
                    continue
                if raw.get("_bb_contract_spec") == spec:
                    continue
                raw["_bb_contract_spec"] = dict(spec)
                raw.pop("_bb_contract_spec_error", None)
                record.raw_row = raw
                record.synced_at = synced_at
                updated += 1
            return updated

    async def _persist_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        checked_at: datetime,
        since: datetime,
    ) -> OkxPositionHistoryMirrorSyncSummary:
        identities = [
            identity
            for row in rows
            if (identity := okx_position_history_row_identity(row, mode=self.mode)).strip("|")
        ]
        existing_fingerprints: dict[str, tuple[Any, ...]] = {}
        if identities:
            async with self.session_context_factory() as session:
                result = await session.execute(
                    select(OkxPositionHistory).where(
                        OkxPositionHistory.mode == self.mode,
                        OkxPositionHistory.row_identity.in_(identities),
                    )
                )
                existing_fingerprints = {
                    str(record.row_identity): _history_record_fact_fingerprint(record)
                    for record in result.scalars().all()
                }

        inserted = 0
        updated = 0
        unchanged = 0
        upserted = 0
        skipped = 0
        samples: list[dict[str, Any]] = []
        latest_row = _latest_row(rows)
        async with self.session_context_factory() as session:
            for row in rows:
                identity = okx_position_history_row_identity(row, mode=self.mode)
                if not identity.strip("|"):
                    skipped += 1
                    continue
                record = await upsert_okx_position_history_row(
                    session,
                    row,
                    mode=self.mode,
                    source="okx_position_history_account_sync",
                    match_status="okx_account_position_history",
                    synced_at=checked_at,
                )
                if record is None:
                    skipped += 1
                    continue
                upserted += 1
                previous_fingerprint = existing_fingerprints.get(identity)
                was_existing = previous_fingerprint is not None
                operation = "inserted"
                if was_existing and previous_fingerprint != _history_record_fact_fingerprint(record):
                    updated += 1
                    operation = "updated"
                elif was_existing:
                    unchanged += 1
                    operation = "unchanged"
                else:
                    inserted += 1
                existing_fingerprints[identity] = _history_record_fact_fingerprint(record)
                if len(samples) < 10:
                    samples.append(_sample_from_row(row, operation=operation))

        return OkxPositionHistoryMirrorSyncSummary(
            status="ok",
            mode=self.mode,
            source="okx_position_history_account_sync",
            checked_at=checked_at,
            since=since,
            okx_pull_available=True,
            live_count=len(rows),
            upserted_count=upserted,
            inserted_count=inserted,
            updated_count=updated,
            unchanged_count=unchanged,
            skipped_count=skipped,
            latest_u_time_ms=_row_u_time_ms(latest_row),
            latest_inst_id=str((latest_row or {}).get("instId") or ""),
            latest_pos_id=str((latest_row or {}).get("posId") or ""),
            samples=tuple(samples),
        )


def _latest_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=_row_u_time_ms)


def _row_u_time_ms(row: dict[str, Any] | None) -> float:
    if not row:
        return 0.0
    try:
        return float(str(row.get("uTime") or row.get("cTime") or 0).strip() or 0)
    except (TypeError, ValueError):
        return 0.0


def _history_record_fact_fingerprint(record: OkxPositionHistory) -> tuple[Any, ...]:
    """Return only exchange facts whose change should trigger downstream work."""

    return (
        str(record.inst_id or ""),
        str(record.pos_id or ""),
        str(record.pos_side or ""),
        str(record.side or ""),
        str(record.close_type or ""),
        str(record.close_status or ""),
        _history_datetime_fingerprint(record.opened_at),
        _history_datetime_fingerprint(record.updated_at_okx),
        float(record.open_avg_px or 0.0),
        float(record.close_avg_px or 0.0),
        float(record.open_max_pos or 0.0),
        float(record.close_total_pos or 0.0),
        float(record.leverage or 0.0),
        float(record.realized_pnl or 0.0),
        float(record.pnl or 0.0),
        record.pnl_ratio,
        float(record.funding_fee or 0.0),
        float(record.fee or 0.0),
    )


def _history_datetime_fingerprint(value: datetime | None) -> float | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return round(value.astimezone(UTC).timestamp(), 6)


def _sample_from_row(row: dict[str, Any], *, operation: str) -> dict[str, Any]:
    return {
        "inst_id": str(row.get("instId") or ""),
        "pos_id": str(row.get("posId") or ""),
        "pos_side": str(row.get("posSide") or ""),
        "type": str(row.get("type") or ""),
        "u_time": str(row.get("uTime") or ""),
        "realized_pnl": str(row.get("realizedPnl") or ""),
        "operation": operation,
    }
