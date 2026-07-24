"""Mirror the OKX facts required for authoritative position settlement.

This is the sole owner of OKX positions-history and funding-bill API pulls.
Position settlement reads these local mirrors and never competes for the same
private endpoints.
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
from models.account import OkxAccountBill
from models.trade import OkxPositionHistory
from services.okx_native_facts import (
    OkxNativeAccountBill,
    OkxNativeFactsClient,
)
from services.okx_position_history_store import (
    okx_position_history_row_identity,
    upsert_okx_position_history_row,
)

logger = structlog.get_logger(__name__)

DEFAULT_SETTLEMENT_FACT_LOOKBACK_HOURS = 72
DEFAULT_SETTLEMENT_FACT_LIMIT = 100
DEFAULT_SETTLEMENT_FACT_MAX_PAGES = 5
DEFAULT_SETTLEMENT_FACT_TIMEOUT_SECONDS = 8.0

SessionContextFactory = Callable[[], AbstractAsyncContextManager[Any]]


@dataclass(frozen=True, slots=True)
class OkxSettlementFactSyncSummary:
    status: str
    mode: str
    source: str
    checked_at: datetime
    since: datetime
    okx_pull_available: bool
    position_history_count: int = 0
    position_history_inserted_count: int = 0
    position_history_updated_count: int = 0
    position_history_unchanged_count: int = 0
    account_bill_count: int = 0
    account_bill_inserted_count: int = 0
    account_bill_updated_count: int = 0
    account_bill_unchanged_count: int = 0
    completed_stages: tuple[str, ...] = ()
    deferred_stages: tuple[str, ...] = ()
    stage_errors: tuple[str, ...] = ()
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
            "position_history_count": self.position_history_count,
            "position_history_inserted_count": self.position_history_inserted_count,
            "position_history_updated_count": self.position_history_updated_count,
            "position_history_unchanged_count": self.position_history_unchanged_count,
            "account_bill_count": self.account_bill_count,
            "account_bill_inserted_count": self.account_bill_inserted_count,
            "account_bill_updated_count": self.account_bill_updated_count,
            "account_bill_unchanged_count": self.account_bill_unchanged_count,
            "completed_stages": list(self.completed_stages),
            "deferred_stages": list(self.deferred_stages),
            "stage_errors": list(self.stage_errors),
            "latest_u_time_ms": self.latest_u_time_ms,
            "latest_inst_id": self.latest_inst_id,
            "latest_pos_id": self.latest_pos_id,
            "error": self.error,
            "samples": list(self.samples),
        }


class OkxSettlementFactSyncService:
    """Continuously mirror official position-history and funding bills."""

    def __init__(
        self,
        *,
        mode: str = "paper",
        lookback_hours: int = DEFAULT_SETTLEMENT_FACT_LOOKBACK_HOURS,
        limit: int = DEFAULT_SETTLEMENT_FACT_LIMIT,
        max_pages: int = DEFAULT_SETTLEMENT_FACT_MAX_PAGES,
        timeout_seconds: float = DEFAULT_SETTLEMENT_FACT_TIMEOUT_SECONDS,
        executor_factory: Any | None = None,
        session_context_factory: SessionContextFactory = get_session_ctx,
    ) -> None:
        self.mode = "live" if str(mode or "").lower() == "live" else "paper"
        self.lookback_hours = max(1, min(int(lookback_hours or 1), 24 * 30))
        self.limit = max(1, min(int(limit or DEFAULT_SETTLEMENT_FACT_LIMIT), 100))
        self.max_pages = max(1, min(int(max_pages or DEFAULT_SETTLEMENT_FACT_MAX_PAGES), 20))
        self.timeout_seconds = max(1.0, float(timeout_seconds or 1.0))
        self.executor_factory = executor_factory or OKXExecutor
        self.session_context_factory = session_context_factory

    async def sync_once(self) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        since = started_at - timedelta(hours=self.lookback_hours)
        deadline = asyncio.get_running_loop().time() + self.timeout_seconds
        executor = self.executor_factory(mode=self.mode, load_markets_on_initialize=False)
        completed_stages: list[str] = []
        deferred_stages: list[str] = []
        stage_errors: list[str] = []
        history_rows: list[dict[str, Any]] = []
        account_bills: list[OkxNativeAccountBill] = []
        contract_specs: dict[str, dict[str, Any]] = {}
        initialized = False

        async def run_stage(
            stage: str,
            operation: Any,
            *,
            cap_seconds: float,
        ) -> tuple[Any, bool]:
            remaining = deadline - asyncio.get_running_loop().time()
            timeout = min(max(cap_seconds, 0.05), max(remaining, 0.0))
            if timeout < 0.05:
                deferred_stages.append(stage)
                return None, False
            try:
                result = await asyncio.wait_for(operation(), timeout=timeout)
            except TimeoutError:
                deferred_stages.append(stage)
                logger.info(
                    "OKX settlement fact stage deferred after pull budget timeout",
                    mode=self.mode,
                    stage=stage,
                    timeout_seconds=round(timeout, 3),
                )
                return None, False
            except Exception as exc:
                if _is_retryable_okx_private_error(exc):
                    deferred_stages.append(stage)
                    logger.info(
                        "OKX settlement fact stage deferred after transient private API response",
                        mode=self.mode,
                        stage=stage,
                        error=safe_error_text(exc, limit=220),
                    )
                    return None, False
                error = safe_error_text(exc, limit=220)
                stage_errors.append(f"{stage}: {error}")
                logger.warning(
                    "OKX settlement fact stage failed",
                    mode=self.mode,
                    stage=stage,
                    error=error,
                )
                return None, False
            completed_stages.append(stage)
            return result, True

        try:
            _, initialized = await run_stage(
                "initialize",
                executor.initialize,
                cap_seconds=min(2.0, self.timeout_seconds),
            )
            if initialized:
                native_facts = OkxNativeFactsClient(executor)
                history_result, bill_result = await asyncio.gather(
                    run_stage(
                        "position_history",
                        lambda: native_facts.fetch_position_history_rows(
                            inst_ids=None,
                            pos_ids=None,
                            since=since,
                            limit=self.limit,
                            max_pages=self.max_pages,
                            strict=True,
                        ),
                        cap_seconds=5.0,
                    ),
                    run_stage(
                        "account_bills",
                        lambda: native_facts.fetch_account_bills(
                            since=since,
                            limit=self.limit,
                            max_pages=self.max_pages,
                            funding_only=True,
                            strict=True,
                        ),
                        cap_seconds=3.0,
                    ),
                )
                history_rows = list(history_result[0] or [])
                account_bills = list(bill_result[0] or [])
                inst_ids = {
                    str(row.get("instId") or "").strip().upper()
                    for row in history_rows
                    if str(row.get("instId") or "").strip()
                }
                if not inst_ids:
                    inst_ids = await self._stored_inst_ids()
                if inst_ids:
                    specs, _ = await run_stage(
                        "contract_specs",
                        lambda: native_facts.fetch_contract_specs(inst_ids=inst_ids),
                        cap_seconds=1.5,
                    )
                    contract_specs = dict(specs or {})
        finally:
            try:
                await asyncio.wait_for(executor.shutdown(), timeout=0.5)
            except Exception as exc:
                logger.debug(
                    "OKX settlement fact executor shutdown failed",
                    error=safe_error_text(exc, limit=120),
                )

        if contract_specs:
            await self._persist_contract_specs(contract_specs, synced_at=started_at)
        for row in history_rows:
            inst_id = str(row.get("instId") or "").strip().upper()
            if spec := contract_specs.get(inst_id):
                row["_bb_contract_spec"] = dict(spec)
                row["_bb_contract_spec_source"] = "okx_public_instruments"

        history_stats, history_samples = await self._persist_history_rows(
            history_rows,
            checked_at=started_at,
        )
        bill_stats, bill_samples = await self._persist_account_bills(
            account_bills,
            checked_at=started_at,
            since=since,
        )
        if not initialized:
            status = "degraded"
            error = "OKX executor initialization did not finish inside the pull budget"
        elif stage_errors:
            status = "warning"
            error = None
        elif deferred_stages:
            status = "deferred"
            error = None
        else:
            status = "ok"
            error = None
        latest_row = _latest_row(history_rows)
        return OkxSettlementFactSyncSummary(
            status=status,
            mode=self.mode,
            source="okx_settlement_fact_mirror",
            checked_at=started_at,
            since=since,
            okx_pull_available=initialized,
            position_history_count=len(history_rows),
            position_history_inserted_count=history_stats["inserted"],
            position_history_updated_count=history_stats["updated"],
            position_history_unchanged_count=history_stats["unchanged"],
            account_bill_count=len(account_bills),
            account_bill_inserted_count=bill_stats["inserted"],
            account_bill_updated_count=bill_stats["updated"],
            account_bill_unchanged_count=bill_stats["unchanged"],
            completed_stages=tuple(completed_stages),
            deferred_stages=tuple(dict.fromkeys(deferred_stages)),
            stage_errors=tuple(stage_errors),
            latest_u_time_ms=_row_u_time_ms(latest_row),
            latest_inst_id=str((latest_row or {}).get("instId") or ""),
            latest_pos_id=str((latest_row or {}).get("posId") or ""),
            error=error,
            samples=tuple([*history_samples, *bill_samples][:10]),
        ).as_dict()

    async def _stored_inst_ids(self) -> set[str]:
        async with self.session_context_factory() as session:
            result = await session.execute(
                select(OkxPositionHistory.inst_id).where(OkxPositionHistory.mode == self.mode)
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
                if (
                    raw.get("_bb_contract_spec") == spec
                    and raw.get("_bb_contract_spec_source") == "okx_public_instruments"
                ):
                    continue
                raw["_bb_contract_spec"] = dict(spec)
                raw["_bb_contract_spec_source"] = "okx_public_instruments"
                raw.pop("_bb_contract_size_evidence", None)
                raw.pop("_bb_contract_spec_error", None)
                record.raw_row = raw
                record.synced_at = synced_at
                updated += 1
            return updated

    async def _persist_history_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        checked_at: datetime,
    ) -> tuple[dict[str, int], list[dict[str, Any]]]:
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
        stats = {"inserted": 0, "updated": 0, "unchanged": 0}
        samples: list[dict[str, Any]] = []
        async with self.session_context_factory() as session:
            for row in rows:
                identity = okx_position_history_row_identity(row, mode=self.mode)
                if not identity.strip("|"):
                    continue
                record = await upsert_okx_position_history_row(
                    session,
                    row,
                    mode=self.mode,
                    source="okx_settlement_fact_mirror",
                    match_status="okx_account_position_history",
                    synced_at=checked_at,
                )
                if record is None:
                    continue
                previous = existing_fingerprints.get(identity)
                current = _history_record_fact_fingerprint(record)
                if previous is None:
                    operation = "inserted"
                elif previous != current:
                    operation = "updated"
                else:
                    operation = "unchanged"
                stats[operation] += 1
                existing_fingerprints[identity] = current
                if len(samples) < 6:
                    samples.append(_history_sample(row, operation=operation))
        return stats, samples

    async def _persist_account_bills(
        self,
        bills: list[OkxNativeAccountBill],
        *,
        checked_at: datetime,
        since: datetime,
    ) -> tuple[dict[str, int], list[dict[str, Any]]]:
        stats = {"inserted": 0, "updated": 0, "unchanged": 0}
        samples: list[dict[str, Any]] = []
        async with self.session_context_factory() as session:
            for bill in bills:
                bill_id = str(bill.bill_id or "").strip()
                bill_time = _aware_utc(bill.timestamp)
                if not bill_id or bill_time is None or bill_time < since:
                    continue
                result = await session.execute(
                    select(OkxAccountBill)
                    .where(
                        OkxAccountBill.mode == self.mode,
                        OkxAccountBill.bill_id == bill_id,
                    )
                    .limit(1)
                )
                existing = result.scalar_one_or_none()
                payload = {
                    "mode": self.mode,
                    "bill_id": bill_id,
                    "inst_id": bill.inst_id or None,
                    "pos_side": bill.pos_side or None,
                    "ccy": bill.ccy or "USDT",
                    "bill_type": bill.bill_type or None,
                    "bill_sub_type": bill.bill_sub_type or None,
                    "bill_ts": bill_time,
                    "balance_change": bill.balance_change,
                    "pnl": bill.pnl,
                    "fee": bill.fee,
                    "funding_fee": bill.funding_fee,
                    "source": "okx_settlement_fact_mirror",
                    "raw_bill": dict(bill.raw),
                }
                if existing is None:
                    session.add(OkxAccountBill(**payload))
                    operation = "inserted"
                else:
                    changed = False
                    for key, value in payload.items():
                        if key in {"mode", "bill_id"}:
                            continue
                        if getattr(existing, key) != value:
                            setattr(existing, key, value)
                            changed = True
                    if changed:
                        existing.updated_at = checked_at
                    operation = "updated" if changed else "unchanged"
                stats[operation] += 1
                if len(samples) < 4:
                    samples.append(
                        {
                            "kind": "okx_account_bill",
                            "bill_id": bill_id,
                            "inst_id": bill.inst_id,
                            "funding_fee": bill.funding_fee,
                            "operation": operation,
                        }
                    )
        return stats, samples


def _latest_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(rows, key=_row_u_time_ms) if rows else None


def _row_u_time_ms(row: dict[str, Any] | None) -> float:
    if not row:
        return 0.0
    try:
        return float(str(row.get("uTime") or row.get("cTime") or 0).strip() or 0)
    except (TypeError, ValueError):
        return 0.0


def _history_record_fact_fingerprint(record: OkxPositionHistory) -> tuple[Any, ...]:
    return (
        str(record.inst_id or ""),
        str(record.pos_id or ""),
        str(record.pos_side or ""),
        str(record.side or ""),
        str(record.close_type or ""),
        str(record.close_status or ""),
        _datetime_fingerprint(record.opened_at),
        _datetime_fingerprint(record.updated_at_okx),
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


def _datetime_fingerprint(value: datetime | None) -> float | None:
    aware = _aware_utc(value)
    return round(aware.timestamp(), 6) if aware is not None else None


def _aware_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _history_sample(row: dict[str, Any], *, operation: str) -> dict[str, Any]:
    return {
        "kind": "okx_position_history",
        "inst_id": str(row.get("instId") or ""),
        "pos_id": str(row.get("posId") or ""),
        "u_time": str(row.get("uTime") or ""),
        "realized_pnl": str(row.get("realizedPnl") or ""),
        "operation": operation,
    }


def _is_retryable_okx_private_error(exc: BaseException) -> bool:
    text = safe_error_text(exc, limit=240).lower()
    return any(
        marker in text
        for marker in (
            "[50011]",
            "[50026]",
            "rate limit",
            "system busy",
            "temporarily unavailable",
        )
    )
