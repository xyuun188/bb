"""Finalize closed-position settlement from OKX official position history.

The trading flow stores a closed position immediately so the local state is
safe after a fill.  This service turns that provisional local close into a
final history/training fact only after OKX positions-history confirms the
official realized PnL and funding fee.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import or_, select

from core.symbols import (
    normalize_trading_symbol,
    okx_inst_id_from_symbol,
    symbol_from_okx_inst_id,
)
from db.session import get_session_ctx
from models.account import OkxAccountBill
from models.trade import OkxPositionHistory, Position
from services.entry_decision_settlement import (
    backfill_settled_entry_decision_outcomes,
    sync_settled_entry_decision_outcome,
)
from services.okx_position_history_store import (
    load_okx_position_history_records,
    okx_position_history_records_to_rows,
)
from services.position_settlement import (
    SETTLEMENT_FORMULA,
    SETTLEMENT_STATUS_EXCEPTION,
    apply_position_settlement_snapshot,
    build_position_settlement_snapshot,
    final_settlement_status_values,
    is_final_settlement_status,
)

logger = structlog.get_logger(__name__)

DEFAULT_SETTLEMENT_LOOKBACK_HOURS = 72
DEFAULT_SETTLEMENT_LIMIT = 20
DEFAULT_SETTLEMENT_RETRY_SECONDS = 10.0
POSITION_HISTORY_CLOSE_MATCH_WINDOW_SECONDS = 45 * 60
POSITION_HISTORY_OPEN_MATCH_WINDOW_SECONDS = 24 * 60 * 60
POSITION_HISTORY_MATCH_MAX_ATTEMPTS = 30
POSITION_HISTORY_MATCH_MAX_AGE_HOURS = 6.0
SUPERSEDED_POSITION_STATUS = "superseded_position_residual"
SUPERSEDED_POSITION_SOURCE = "okx_current_position_deduplication"
SUPERSEDED_POSITION_REASON = "duplicate_local_open_position_for_same_okx_pos_id"
DUPLICATE_CLOSED_POSITION_REASON = "duplicate_local_closed_position_for_same_okx_lifecycle"
SETTLEMENT_STATUS_QUARANTINED = "settlement_quarantined"
SETTLEMENT_QUARANTINE_SOURCE = "okx_position_history_identity_quarantine"
NON_RETRYABLE_SETTLEMENT_STATUSES = frozenset(
    {SUPERSEDED_POSITION_STATUS, SETTLEMENT_STATUS_QUARANTINED}
)

SessionContextFactory = Callable[[], AbstractAsyncContextManager[Any]]


@dataclass(frozen=True, slots=True)
class SettlementCandidate:
    position_id: int
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    leverage: float
    entry_fee: float
    close_fee: float
    okx_inst_id: str
    okx_pos_id: str
    entry_exchange_order_id: str
    close_exchange_order_id: str
    created_at: datetime | None
    closed_at: datetime | None
    settlement_status: str
    settlement_raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SettlementFailure:
    code: str
    message: str
    context: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SettlementSuccess:
    row: dict[str, Any]
    snapshot: Any
    match_reason: str
    fee_source: str
    funding_fee_source: str


@dataclass(frozen=True, slots=True)
class OkxPositionSettlementSyncSummary:
    status: str
    mode: str
    checked_count: int
    reconciled_count: int
    decision_outcome_count: int
    exception_count: int
    skipped_count: int
    samples: tuple[dict[str, Any], ...]
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.mode,
            "checked_count": self.checked_count,
            "reconciled_count": self.reconciled_count,
            "decision_outcome_count": self.decision_outcome_count,
            "exception_count": self.exception_count,
            "skipped_count": self.skipped_count,
            "samples": list(self.samples),
            "error": self.error,
        }


class OkxPositionSettlementSyncService:
    """Finalize local closed positions with OKX official settlement facts."""

    def __init__(
        self,
        *,
        mode: str = "paper",
        lookback_hours: int = DEFAULT_SETTLEMENT_LOOKBACK_HOURS,
        limit: int = DEFAULT_SETTLEMENT_LIMIT,
        retry_seconds: float = DEFAULT_SETTLEMENT_RETRY_SECONDS,
        session_context_factory: SessionContextFactory = get_session_ctx,
    ) -> None:
        self.mode = "live" if str(mode or "").lower() == "live" else "paper"
        self.lookback_hours = max(
            1, min(int(lookback_hours or DEFAULT_SETTLEMENT_LOOKBACK_HOURS), 24 * 14)
        )
        self.limit = max(1, min(int(limit or DEFAULT_SETTLEMENT_LIMIT), 100))
        self.retry_seconds = max(1.0, float(retry_seconds or DEFAULT_SETTLEMENT_RETRY_SECONDS))
        self.session_context_factory = session_context_factory

    async def sync_once(self) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        decision_outcome_changes = await self._backfill_decision_outcomes(started_at)
        candidates = await self._load_candidates(started_at)
        if not candidates:
            return OkxPositionSettlementSyncSummary(
                status="ok",
                mode=self.mode,
                checked_count=0,
                reconciled_count=0,
                decision_outcome_count=len(decision_outcome_changes),
                exception_count=0,
                skipped_count=0,
                samples=tuple(decision_outcome_changes[-10:]),
            ).as_dict()

        samples: list[dict[str, Any]] = []
        checked = 0
        reconciled = 0
        decision_outcome_count = len(decision_outcome_changes)
        exceptions = 0
        skipped = 0
        samples.extend(decision_outcome_changes[-10:])
        for candidate in candidates:
            checked += 1
            result = await self._settle_candidate(candidate, started_at)
            if isinstance(result, SettlementSuccess):
                changed, outcome_change = await self._apply_success(
                    candidate,
                    result,
                    started_at,
                )
                if outcome_change:
                    decision_outcome_count += 1
                    samples.append(outcome_change)
                if changed:
                    reconciled += 1
                    samples.append(
                        {
                            "kind": "okx_position_settlement_reconciled",
                            "position_id": candidate.position_id,
                            "symbol": candidate.symbol,
                            "side": candidate.side,
                            "okx_pos_id": _position_history_pos_id(result.row),
                            "realized_pnl": result.snapshot.realized_pnl,
                            "funding_fee": result.snapshot.funding_fee,
                            "funding_fee_source": result.funding_fee_source,
                            "match_reason": result.match_reason,
                        }
                    )
                else:
                    skipped += 1
                continue
            quarantined = await self._apply_failure(candidate, result, started_at)
            exceptions += 1
            sample = {
                "kind": (
                    "okx_position_settlement_quarantined"
                    if quarantined
                    else "okx_position_settlement_exception"
                ),
                "position_id": candidate.position_id,
                "symbol": candidate.symbol,
                "side": candidate.side,
                "error_code": result.code,
                "error_message": result.message,
            }
            if not quarantined:
                sample["next_retry_seconds"] = self.retry_seconds
            samples.append(sample)

        status = "warning" if exceptions else "ok"
        return OkxPositionSettlementSyncSummary(
            status=status,
            mode=self.mode,
            checked_count=checked,
            reconciled_count=reconciled,
            decision_outcome_count=decision_outcome_count,
            exception_count=exceptions,
            skipped_count=skipped,
            samples=tuple(samples[-10:]),
            error=None,
        ).as_dict()

    async def _backfill_decision_outcomes(self, now: datetime) -> list[dict[str, Any]]:
        async with self.session_context_factory() as session:
            return await backfill_settled_entry_decision_outcomes(
                session,
                mode=self.mode,
                now=now,
                lookback_hours=self.lookback_hours,
            )

    async def _load_candidates(self, now: datetime) -> list[SettlementCandidate]:
        since = now - timedelta(hours=self.lookback_hours)
        final_statuses = final_settlement_status_values()
        async with self.session_context_factory() as session:
            result = await session.execute(
                select(Position)
                .where(
                    Position.execution_mode == self.mode,
                    Position.is_open.is_(False),
                    Position.closed_at.is_not(None),
                    Position.closed_at >= _db_naive(since),
                    or_(
                        Position.settlement_status.is_(None),
                        Position.settlement_status.not_in(
                            tuple(sorted(NON_RETRYABLE_SETTLEMENT_STATUSES))
                        ),
                    ),
                    or_(
                        Position.settlement_status.is_(None),
                        Position.settlement_status == "",
                        Position.settlement_status.not_in(final_statuses),
                    ),
                )
                .order_by(Position.closed_at.desc(), Position.id.desc())
                .limit(self.limit * 5)
            )
            rows = list(result.scalars().all())
            rows, duplicate_rows = _deduplicate_closed_lifecycle_rows(rows, now=now)
            candidates: list[SettlementCandidate] = []
            restored_superseded = bool(duplicate_rows)
            for row in rows:
                raw = getattr(row, "settlement_raw", None)
                raw = raw if isinstance(raw, dict) else {}
                if _has_superseded_position_metadata(row, raw):
                    _restore_superseded_position_status(row, raw, now=now)
                    restored_superseded = True
                    continue
                if _retry_after(raw, now):
                    continue
                if len(candidates) < self.limit:
                    candidates.append(_candidate_from_position(row, raw))
            if restored_superseded:
                await session.flush()
        return candidates

    async def _settle_candidate(
        self,
        candidate: SettlementCandidate,
        now: datetime,
    ) -> SettlementSuccess | SettlementFailure:
        inst_id = candidate.okx_inst_id or okx_inst_id_from_symbol(candidate.symbol)
        if not inst_id:
            return SettlementFailure(
                code="missing_okx_inst_id",
                message="Position has no OKX instId and symbol cannot be converted to one.",
                context={"symbol": candidate.symbol, "position_id": candidate.position_id},
            )
        closed_at = _aware_utc(candidate.closed_at) or now
        created_at = _aware_utc(candidate.created_at) or closed_at
        since = min(created_at, closed_at) - timedelta(hours=1)
        async with self.session_context_factory() as session:
            records = await load_okx_position_history_records(
                session,
                mode=self.mode,
                limit=5000,
            )
        rows = okx_position_history_records_to_rows(records)
        if not rows:
            return SettlementFailure(
                code="position_history_mirror_no_rows",
                message="The local OKX settlement-fact mirror has no position-history rows yet.",
                context={
                    "position_id": candidate.position_id,
                    "inst_id": inst_id,
                    "okx_pos_id": candidate.okx_pos_id,
                    "since": since.isoformat(),
                },
            )
        match = _match_position_history_row(candidate, rows, inst_id=inst_id)
        if isinstance(match, SettlementFailure):
            return match
        row, match_reason = match
        return await self._success_from_position_history_row(
            candidate,
            row,
            now=now,
            match_reason=match_reason,
            inst_id=inst_id,
            created_at=created_at,
            closed_at=closed_at,
        )

    async def _success_from_position_history_row(
        self,
        candidate: SettlementCandidate,
        row: dict[str, Any],
        *,
        now: datetime,
        match_reason: str,
        inst_id: str,
        created_at: datetime,
        closed_at: datetime,
    ) -> SettlementSuccess | SettlementFailure:
        realized_value, realized_key = _first_present_float(
            row,
            ("realizedPnl", "realized_pnl", "realizedPnlInUsd", "realizedPnlUsd"),
        )
        if realized_key is None:
            return SettlementFailure(
                code="official_row_missing_realized_pnl",
                message="OKX positions-history row has no realizedPnl field.",
                context={"position_id": candidate.position_id, "row_keys": sorted(row.keys())},
            )
        funding_value, funding_key = _first_present_float(row, ("fundingFee", "funding_fee"))
        funding_source = f"okx_positions_history.{funding_key}" if funding_key else ""
        if funding_key is None:
            funding_result = await self._funding_fee_from_account_bills(
                candidate,
                inst_id=inst_id,
                created_at=created_at,
                closed_at=closed_at,
            )
            if isinstance(funding_result, SettlementFailure):
                return funding_result
            funding_value, funding_source = funding_result
        fee_value, fee_key = _first_present_float(row, ("fee", "fees", "totalFee", "total_fee"))
        fee_source = (
            f"okx_positions_history.{fee_key}" if fee_key else "local_position_fee_snapshot"
        )
        total_fee_abs = (
            abs(fee_value) if fee_key else abs(candidate.entry_fee) + abs(candidate.close_fee)
        )
        entry_fee, close_fee = _allocate_total_fee(
            total_fee_abs,
            candidate_entry_fee=candidate.entry_fee,
            candidate_close_fee=candidate.close_fee,
        )
        gross_value, gross_key = _first_present_float(row, ("pnl", "closePnl", "close_pnl"))
        gross_source = (
            f"okx_positions_history.{gross_key}" if gross_key else "derived_from_realized_pnl"
        )
        if gross_key is None:
            gross_value = realized_value - funding_value + entry_fee + close_fee
        computed = gross_value + funding_value - entry_fee - close_fee
        formula_delta = computed - realized_value
        adjusted_to_official = abs(formula_delta) > max(abs(realized_value) * 1e-7, 1e-7)
        if adjusted_to_official:
            gross_value = realized_value - funding_value + entry_fee + close_fee
            gross_source = f"{gross_source}:adjusted_to_official_realized_pnl"
        snapshot = build_position_settlement_snapshot(
            close_fill_pnl=gross_value,
            entry_fee=entry_fee,
            close_fee=close_fee,
            funding_fee=funding_value,
            status="reconciled",
            source="okx_position_history_settlement",
            synced_at=now,
            raw={
                "formula": SETTLEMENT_FORMULA,
                "official_formula": "OKX positions-history realizedPnl is authoritative",
                "official_realized_pnl": realized_value,
                "official_realized_pnl_key": realized_key,
                "gross_pnl_source": gross_source,
                "fee_source": fee_source,
                "funding_fee_source": funding_source,
                "match_reason": match_reason,
                "okx_pos_id": _position_history_pos_id(row),
                "okx_inst_id": _position_history_inst_id(row),
                "position_history_closed_at": _iso(_position_history_closed_at(row)),
                "position_history_opened_at": _iso(_position_history_opened_at(row)),
                "formula_delta_before_adjustment": formula_delta,
                "gross_adjusted_to_official_realized_pnl": adjusted_to_official,
                "close_exchange_order_id": candidate.close_exchange_order_id,
                "entry_exchange_order_id": candidate.entry_exchange_order_id,
                "okx_position_history_row": dict(row),
            },
        )
        return SettlementSuccess(
            row=row,
            snapshot=snapshot,
            match_reason=match_reason,
            fee_source=fee_source,
            funding_fee_source=funding_source,
        )

    async def _funding_fee_from_account_bills(
        self,
        candidate: SettlementCandidate,
        *,
        inst_id: str,
        created_at: datetime,
        closed_at: datetime,
    ) -> tuple[float, str] | SettlementFailure:
        window_start = created_at - timedelta(minutes=10)
        window_end = closed_at + timedelta(minutes=10)
        async with self.session_context_factory() as session:
            result = await session.execute(
                select(OkxAccountBill).where(
                    OkxAccountBill.mode == self.mode,
                    OkxAccountBill.inst_id == inst_id,
                    OkxAccountBill.bill_ts >= _db_naive(window_start),
                    OkxAccountBill.bill_ts <= _db_naive(window_end),
                )
            )
            bills = list(result.scalars().all())
        funding_fee = _sum_matching_funding_bills(
            bills,
            inst_id=inst_id,
            side=candidate.side,
            opened_at=created_at,
            closed_at=closed_at,
        )
        return funding_fee, "okx_settlement_fact_mirror.account_bills"

    async def _apply_success(
        self,
        candidate: SettlementCandidate,
        success: SettlementSuccess,
        now: datetime,
    ) -> tuple[bool, dict[str, Any] | None]:
        async with self.session_context_factory() as session:
            position = await session.get(Position, candidate.position_id)
            if position is None or bool(position.is_open):
                return False, None
            raw = getattr(position, "settlement_raw", None)
            raw = raw if isinstance(raw, dict) else {}
            if _has_superseded_position_metadata(position, raw):
                _restore_superseded_position_status(position, raw, now=now)
                await session.flush()
                return False, None
            if is_final_settlement_status(getattr(position, "settlement_status", None)):
                return False, None
            if _is_non_retryable_settlement_status(position):
                return False, None
            apply_position_settlement_snapshot(position, success.snapshot)
            row_inst_id = _position_history_inst_id(success.row)
            row_pos_id = _position_history_pos_id(success.row)
            if row_inst_id:
                position.okx_inst_id = row_inst_id
                position.symbol = symbol_from_okx_inst_id(row_inst_id) or position.symbol
            if row_pos_id:
                position.okx_pos_id = row_pos_id
            row_side = _position_history_side(success.row)
            if row_side in {"long", "short"}:
                position.side = row_side
            history_record_id = _safe_int(
                success.row.get("_dashboard_history_record_id"),
                0,
            )
            history = (
                await session.get(OkxPositionHistory, history_record_id)
                if history_record_id > 0
                else None
            )
            outcome_change = await sync_settled_entry_decision_outcome(
                session,
                position=position,
                history=history,
                now=now,
            )
            position.updated_at = now
            await session.flush()
            return (
                True,
                outcome_change if outcome_change.get("changed") is True else None,
            )

    async def _apply_failure(
        self,
        candidate: SettlementCandidate,
        failure: SettlementFailure,
        now: datetime,
    ) -> bool:
        next_retry_at = now + timedelta(seconds=self.retry_seconds)
        async with self.session_context_factory() as session:
            position = await session.get(Position, candidate.position_id)
            if position is None or bool(position.is_open):
                return False
            raw = getattr(position, "settlement_raw", None)
            raw = raw if isinstance(raw, dict) else {}
            if _has_superseded_position_metadata(position, raw):
                _restore_superseded_position_status(position, raw, now=now)
                await session.flush()
                return False
            if is_final_settlement_status(getattr(position, "settlement_status", None)):
                return False
            if _is_non_retryable_settlement_status(position):
                return False
            attempts = _safe_int(raw.get("settlement_attempt_count"), 0) + 1
            closed_at = _aware_utc(candidate.closed_at)
            closed_age_hours = (
                max((now - closed_at).total_seconds() / 3600.0, 0.0)
                if closed_at is not None
                else 0.0
            )
            quarantine_triggers: list[str] = []
            if attempts >= POSITION_HISTORY_MATCH_MAX_ATTEMPTS:
                quarantine_triggers.append("attempt_limit")
            if closed_age_hours >= POSITION_HISTORY_MATCH_MAX_AGE_HOURS:
                quarantine_triggers.append("closed_age_limit")
            quarantined = bool(
                failure.code == "positions_history_no_matching_row" and quarantine_triggers
            )
            status = (
                SETTLEMENT_STATUS_QUARANTINED
                if quarantined
                else SETTLEMENT_STATUS_EXCEPTION
            )
            source = (
                SETTLEMENT_QUARANTINE_SOURCE
                if quarantined
                else "okx_position_history_settlement"
            )
            position.settlement_status = status
            position.settlement_source = source
            position.settlement_synced_at = now
            updated_raw = {
                **raw,
                "status": status,
                "source": source,
                "formula": SETTLEMENT_FORMULA,
                "funding_fee_status": "unknown_until_official_settlement",
                "last_error_code": failure.code,
                "last_error_message": failure.message,
                "last_error_context": failure.context,
                "last_settlement_attempt_at": now.isoformat(),
                "settlement_attempt_count": attempts,
            }
            if quarantined:
                updated_raw.pop("next_settlement_retry_at", None)
                updated_raw.update(
                    {
                        "quarantine_reason": "official_position_history_identity_unresolved",
                        "quarantined_at": now.isoformat(),
                        "quarantine_evidence": {
                            "triggers": quarantine_triggers,
                            "attempt_count": attempts,
                            "max_attempts": POSITION_HISTORY_MATCH_MAX_ATTEMPTS,
                            "closed_at": _iso(closed_at),
                            "closed_age_hours": closed_age_hours,
                            "max_age_hours": POSITION_HISTORY_MATCH_MAX_AGE_HOURS,
                        },
                        "retry_policy": "permanent_no_retry",
                    }
                )
            else:
                updated_raw.update(
                    {
                        "next_settlement_retry_at": next_retry_at.isoformat(),
                        "retry_policy": (
                            f"retry every {self.retry_seconds:g}s until OKX official "
                            "settlement is available"
                        ),
                    }
                )
            position.settlement_raw = updated_raw
            position.updated_at = now
            await session.flush()
            return quarantined


def _deduplicate_closed_lifecycle_rows(
    rows: list[Position],
    *,
    now: datetime,
) -> tuple[list[Position], list[Position]]:
    grouped: dict[tuple[Any, ...], list[Position]] = {}
    retained_without_identity: list[Position] = []
    for position in rows:
        identity = _closed_lifecycle_identity(position)
        if identity is None:
            retained_without_identity.append(position)
            continue
        grouped.setdefault(identity, []).append(position)
    retained = list(retained_without_identity)
    duplicates: list[Position] = []
    for candidates in grouped.values():
        canonical = min(candidates, key=lambda item: int(getattr(item, "id", 0) or 0))
        retained.append(canonical)
        for duplicate in candidates:
            if duplicate is canonical:
                continue
            raw = _safe_dict(getattr(duplicate, "settlement_raw", None))
            duplicate.settlement_status = SUPERSEDED_POSITION_STATUS
            duplicate.settlement_source = SUPERSEDED_POSITION_SOURCE
            duplicate.settlement_synced_at = now
            duplicate.settlement_raw = {
                **raw,
                "status": SUPERSEDED_POSITION_STATUS,
                "source": SUPERSEDED_POSITION_SOURCE,
                "reason": DUPLICATE_CLOSED_POSITION_REASON,
                "canonical_position_id": int(getattr(canonical, "id", 0) or 0),
                "duplicate_closed_lifecycle_retired_at": now.isoformat(),
            }
            duplicate.updated_at = now
            duplicates.append(duplicate)
    retained.sort(
        key=lambda item: (
            _aware_utc(getattr(item, "closed_at", None))
            or datetime.min.replace(tzinfo=UTC),
            int(getattr(item, "id", 0) or 0),
        ),
        reverse=True,
    )
    return retained, duplicates


def _closed_lifecycle_identity(position: Position) -> tuple[Any, ...] | None:
    pos_id = str(getattr(position, "okx_pos_id", "") or "").strip()
    entry_ids = tuple(
        sorted(_split_exchange_order_ids(getattr(position, "entry_exchange_order_id", None)))
    )
    close_ids = tuple(
        sorted(_split_exchange_order_ids(getattr(position, "close_exchange_order_id", None)))
    )
    created_at = _aware_utc(getattr(position, "created_at", None))
    closed_at = _aware_utc(getattr(position, "closed_at", None))
    quantity = abs(_safe_float(getattr(position, "quantity", None), 0.0))
    if not pos_id or not entry_ids or not close_ids or created_at is None or closed_at is None:
        return None
    return (
        str(getattr(position, "execution_mode", "") or "").lower(),
        pos_id,
        entry_ids,
        close_ids,
        round(quantity, 12),
        created_at.isoformat(),
        closed_at.isoformat(),
    )


def _candidate_from_position(position: Position, raw: dict[str, Any]) -> SettlementCandidate:
    symbol = normalize_trading_symbol(str(getattr(position, "symbol", "") or ""))
    return SettlementCandidate(
        position_id=int(getattr(position, "id", 0) or 0),
        symbol=symbol,
        side=str(getattr(position, "side", "") or "").lower().strip(),
        quantity=_safe_float(getattr(position, "quantity", None), 0.0),
        entry_price=_safe_float(getattr(position, "entry_price", None), 0.0),
        current_price=_safe_float(getattr(position, "current_price", None), 0.0),
        leverage=max(_safe_float(getattr(position, "leverage", None), 1.0), 1.0),
        entry_fee=abs(_safe_float(getattr(position, "entry_fee", None), 0.0)),
        close_fee=abs(_safe_float(getattr(position, "close_fee", None), 0.0)),
        okx_inst_id=str(getattr(position, "okx_inst_id", "") or "").strip().upper(),
        okx_pos_id=str(getattr(position, "okx_pos_id", "") or "").strip(),
        entry_exchange_order_id=str(getattr(position, "entry_exchange_order_id", "") or "").strip(),
        close_exchange_order_id=str(getattr(position, "close_exchange_order_id", "") or "").strip(),
        created_at=_aware_utc(getattr(position, "created_at", None)),
        closed_at=_aware_utc(getattr(position, "closed_at", None)),
        settlement_status=str(getattr(position, "settlement_status", "") or "").strip(),
        settlement_raw=dict(raw),
    )


def _match_position_history_row(
    candidate: SettlementCandidate,
    rows: list[dict[str, Any]],
    *,
    inst_id: str,
) -> tuple[dict[str, Any], str] | SettlementFailure:
    scored: list[tuple[int, float, dict[str, Any], str]] = []
    close_order_ids = _split_exchange_order_ids(candidate.close_exchange_order_id)
    entry_order_ids = _split_exchange_order_ids(candidate.entry_exchange_order_id)
    for row in rows:
        row_inst_id = _position_history_inst_id(row)
        if row_inst_id and row_inst_id != inst_id:
            continue
        row_pos_id = _position_history_pos_id(row)
        row_side = _position_history_side(row)
        if candidate.okx_pos_id and row_pos_id and row_pos_id != candidate.okx_pos_id:
            continue
        if candidate.side and row_side in {"long", "short"} and row_side != candidate.side:
            continue
        score = 0
        reasons: list[str] = []
        if candidate.okx_pos_id and row_pos_id == candidate.okx_pos_id:
            score += 100
            reasons.append("pos_id_exact")
        if row_inst_id == inst_id:
            score += 20
            reasons.append("inst_id")
        if candidate.side and row_side == candidate.side:
            score += 15
            reasons.append("side")
        closed_delta = _time_delta_seconds(candidate.closed_at, _position_history_closed_at(row))
        if closed_delta is not None:
            if closed_delta <= POSITION_HISTORY_CLOSE_MATCH_WINDOW_SECONDS:
                score += max(0, 40 - int(closed_delta // 60))
                reasons.append(f"closed_at_delta={int(closed_delta)}s")
            elif not candidate.okx_pos_id:
                continue
        opened_delta = _time_delta_seconds(candidate.created_at, _position_history_opened_at(row))
        if opened_delta is not None and opened_delta <= POSITION_HISTORY_OPEN_MATCH_WINDOW_SECONDS:
            score += 5
            reasons.append(f"opened_at_delta={int(opened_delta)}s")
        if close_order_ids and _row_contains_any_token(row, close_order_ids):
            score += 50
            reasons.append("close_order_id")
        if entry_order_ids and _row_contains_any_token(row, entry_order_ids):
            score += 20
            reasons.append("entry_order_id")
        if score <= 0:
            continue
        scored.append(
            (score, closed_delta if closed_delta is not None else 1e12, row, ",".join(reasons))
        )
    if not scored:
        return SettlementFailure(
            code="positions_history_no_matching_row",
            message="OKX positions-history rows were returned, but none matched local position identity.",
            context={
                "position_id": candidate.position_id,
                "symbol": candidate.symbol,
                "side": candidate.side,
                "okx_pos_id": candidate.okx_pos_id,
                "row_count": len(rows),
            },
        )
    scored.sort(key=lambda item: (-item[0], item[1]))
    best = scored[0]
    if len(scored) > 1 and scored[1][0] == best[0] and abs(scored[1][1] - best[1]) < 1.0:
        return SettlementFailure(
            code="positions_history_ambiguous_match",
            message="Multiple OKX positions-history rows matched with equal confidence.",
            context={
                "position_id": candidate.position_id,
                "symbol": candidate.symbol,
                "side": candidate.side,
                "okx_pos_id": candidate.okx_pos_id,
                "top_score": best[0],
                "row_count": len(scored),
            },
        )
    return best[2], best[3]


def _sum_matching_funding_bills(
    bills: list[Any],
    *,
    inst_id: str,
    side: str,
    opened_at: datetime,
    closed_at: datetime,
) -> float:
    window_start = opened_at - timedelta(minutes=10)
    window_end = closed_at + timedelta(minutes=10)
    total = 0.0
    for bill in bills:
        bill_inst_id = str(getattr(bill, "inst_id", "") or "").strip().upper()
        if bill_inst_id and bill_inst_id != inst_id:
            continue
        bill_side = str(getattr(bill, "pos_side", "") or "").lower().strip()
        if bill_side in {"long", "short"} and side and bill_side != side:
            continue
        bill_time = _aware_utc(
            getattr(bill, "bill_ts", None) or getattr(bill, "timestamp", None)
        )
        if bill_time is None or bill_time < window_start or bill_time > window_end:
            continue
        total += _safe_float(getattr(bill, "funding_fee", None), 0.0)
    return total


def _allocate_total_fee(
    total_fee_abs: float,
    *,
    candidate_entry_fee: float,
    candidate_close_fee: float,
) -> tuple[float, float]:
    total_fee_abs = abs(_safe_float(total_fee_abs, 0.0))
    existing_entry = abs(_safe_float(candidate_entry_fee, 0.0))
    existing_close = abs(_safe_float(candidate_close_fee, 0.0))
    existing_total = existing_entry + existing_close
    if total_fee_abs <= 0:
        return 0.0, 0.0
    if existing_total > 0:
        entry_fee = total_fee_abs * existing_entry / existing_total
        return entry_fee, total_fee_abs - entry_fee
    return 0.0, total_fee_abs


def _retry_after(raw: dict[str, Any], now: datetime) -> bool:
    next_retry = _parse_datetime(raw.get("next_settlement_retry_at"))
    return next_retry is not None and next_retry > now


def _is_non_retryable_settlement_status(position: Position) -> bool:
    return (
        str(getattr(position, "settlement_status", "") or "").strip()
        in NON_RETRYABLE_SETTLEMENT_STATUSES
    )


def _has_superseded_position_metadata(position: Position, raw: dict[str, Any]) -> bool:
    if str(getattr(position, "settlement_status", "") or "") == SUPERSEDED_POSITION_STATUS:
        return True
    return bool(
        str(raw.get("reason") or "") == SUPERSEDED_POSITION_REASON
        and _safe_int(raw.get("canonical_position_id"), 0) > 0
    )


def _restore_superseded_position_status(
    position: Position,
    raw: dict[str, Any],
    *,
    now: datetime,
) -> None:
    previous_status = str(getattr(position, "settlement_status", "") or "")
    previous_source = str(getattr(position, "settlement_source", "") or "")
    position.settlement_status = SUPERSEDED_POSITION_STATUS
    position.settlement_source = SUPERSEDED_POSITION_SOURCE
    position.settlement_synced_at = now
    position.settlement_raw = {
        **raw,
        "status": SUPERSEDED_POSITION_STATUS,
        "source": SUPERSEDED_POSITION_SOURCE,
        "reason": str(raw.get("reason") or SUPERSEDED_POSITION_REASON),
        "restored_from_status": previous_status,
        "restored_from_source": previous_source,
        "superseded_status_restored_at": now.isoformat(),
    }
    position.updated_at = now


def _first_present_float(row: dict[str, Any], keys: tuple[str, ...]) -> tuple[float, str | None]:
    for key in keys:
        if key in row and row.get(key) is not None:
            return _safe_float(row.get(key), 0.0), key
    return 0.0, None


def _position_history_inst_id(row: dict[str, Any]) -> str:
    return str(row.get("instId") or "").strip().upper()


def _position_history_pos_id(row: dict[str, Any]) -> str:
    return str(row.get("posId") or "").strip()


def _position_history_side(row: dict[str, Any]) -> str:
    for key in ("posSide", "positionSide", "side"):
        value = str(row.get(key) or "").lower().strip()
        if value in {"long", "short"}:
            return value
    direction = str(row.get("direction") or "").lower().strip()
    if direction in {"long", "short"}:
        return direction
    return ""


def _position_history_closed_at(row: dict[str, Any]) -> datetime | None:
    return _ms_datetime(row.get("uTime") or row.get("closedAt") or row.get("closeTime"))


def _position_history_opened_at(row: dict[str, Any]) -> datetime | None:
    return _ms_datetime(row.get("cTime") or row.get("openedAt") or row.get("openTime"))


def _row_contains_any_token(row: dict[str, Any], tokens: set[str]) -> bool:
    if not tokens:
        return False
    stack: list[Any] = [row]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
            continue
        if isinstance(item, list):
            stack.extend(item)
            continue
        text = str(item or "").strip()
        if text in tokens:
            return True
    return False


def _split_exchange_order_ids(value: Any) -> set[str]:
    tokens = {str(value or "").strip()}
    if not next(iter(tokens), ""):
        return set()
    for separator in (",", ";", "|", "\n", "\t", " "):
        pieces: set[str] = set()
        for token in tokens:
            pieces.update(part.strip() for part in token.split(separator) if part.strip())
        tokens = pieces
    return {token for token in tokens if token}


def _time_delta_seconds(left: datetime | None, right: datetime | None) -> float | None:
    left = _aware_utc(left)
    right = _aware_utc(right)
    if left is None or right is None:
        return None
    return abs((left - right).total_seconds())


def _ms_datetime(value: Any) -> datetime | None:
    number = _safe_float(value, 0.0)
    if number <= 0:
        return None
    try:
        return datetime.fromtimestamp(number / 1000.0, UTC)
    except (OSError, OverflowError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _aware_utc(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return _aware_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        return None


def _aware_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _safe_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _db_naive(value: datetime) -> datetime:
    value = _aware_utc(value) or datetime.now(UTC)
    return value.replace(tzinfo=None)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _iso(value: datetime | None) -> str | None:
    value = _aware_utc(value)
    return value.isoformat() if value is not None else None
