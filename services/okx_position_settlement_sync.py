"""Finalize closed-position settlement from OKX official position history.

The trading flow stores a closed position immediately so the local state is
safe after a fill.  This service turns that provisional local close into a
final history/training fact only after OKX positions-history confirms the
official realized PnL and funding fee.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import select

from core.symbols import (
    normalize_trading_symbol,
    okx_inst_id_from_symbol,
    symbol_from_okx_inst_id,
)
from db.session import get_session_ctx
from models.account import OkxAccountBill
from models.trade import OkxPositionHistory, Order, Position
from services.entry_decision_settlement import (
    backfill_settled_entry_decision_outcomes,
    sync_settled_entry_decision_outcome,
)
from services.okx_position_history_store import (
    load_okx_position_history_records,
    okx_position_history_records_to_rows,
    okx_position_history_row_identity,
)
from services.position_settlement import (
    SETTLEMENT_FORMULA,
    SETTLEMENT_STATUS_EXCEPTION,
    apply_position_settlement_snapshot,
    build_position_settlement_snapshot,
    is_final_settlement_status,
)

logger = structlog.get_logger(__name__)

DEFAULT_SETTLEMENT_LOOKBACK_HOURS = 72
DEFAULT_SETTLEMENT_LIMIT = 20
DEFAULT_SETTLEMENT_RETRY_SECONDS = 10.0
MINIMUM_CLOSED_LIFECYCLE_SCAN_ROWS = 500
POSITION_HISTORY_CLOSE_MATCH_WINDOW_SECONDS = 45 * 60
POSITION_HISTORY_CLOSE_EARLY_TOLERANCE_SECONDS = 2 * 60
POSITION_HISTORY_OPEN_MATCH_WINDOW_SECONDS = 24 * 60 * 60
POSITION_HISTORY_MATCH_MAX_ATTEMPTS = 30
POSITION_HISTORY_MATCH_MAX_AGE_HOURS = 6.0
POSITION_HISTORY_QUARANTINE_RETRY_SECONDS = 15 * 60.0
SUPERSEDED_POSITION_STATUS = "superseded_position_residual"
SUPERSEDED_POSITION_SOURCE = "okx_current_position_deduplication"
SUPERSEDED_POSITION_REASON = "duplicate_local_open_position_for_same_okx_pos_id"
DUPLICATE_CLOSED_POSITION_REASON = "duplicate_local_closed_position_for_same_okx_lifecycle"
DISTINCT_CLOSED_FRAGMENT_REACTIVATED_REASON = "distinct_partial_close_fragment_reactivated"
SUPERSEDED_POSITION_REASONS = frozenset(
    {SUPERSEDED_POSITION_REASON, DUPLICATE_CLOSED_POSITION_REASON}
)
SETTLEMENT_STATUS_QUARANTINED = "settlement_quarantined"
SETTLEMENT_QUARANTINE_SOURCE = "okx_position_history_identity_quarantine"
SETTLEMENT_LIFECYCLE_OPEN_SOURCE = "okx_position_lifecycle_still_open"
SETTLEMENT_LIFECYCLE_OPEN_REASON = "position_lifecycle_still_open"
NON_RETRYABLE_SETTLEMENT_STATUSES = frozenset(
    {SUPERSEDED_POSITION_STATUS}
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
    entry_contracts: float = 0.0
    close_contracts: float = 0.0
    lifecycle_group_size: int = 1
    lifecycle_group_position_ids: tuple[int, ...] = ()
    allocation_ratio: float = 1.0
    allocation_basis: str = "single_fragment"
    allocation_complete: bool = True
    entry_fee_authoritative: float | None = None
    close_fee_authoritative: float | None = None
    close_fill_pnl_authoritative: float | None = None
    history_row_identity: str = ""
    history_record_id: int = 0
    allocation_error: str = ""


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

        position_history_rows = await self._load_position_history_rows()

        samples: list[dict[str, Any]] = []
        checked = 0
        reconciled = 0
        decision_outcome_count = len(decision_outcome_changes)
        exceptions = 0
        skipped = 0
        failures: list[tuple[SettlementCandidate, SettlementFailure]] = []
        samples.extend(decision_outcome_changes[-10:])
        # Match and allocate each official row at lifecycle scope.  An OKX
        # positions-history row is one lifecycle total; local partial-close
        # fragments may therefore share it, but may never each receive 100% of
        # its PnL or funding.
        lifecycle_groups = _group_candidates_by_lifecycle(candidates)
        allocation_failures: dict[int, SettlementFailure] = {}
        for group in lifecycle_groups.values():
            _prepare_lifecycle_allocations(group, position_history_rows)
            for member in group:
                if not member.allocation_complete:
                    allocation_failures[member.position_id] = SettlementFailure(
                        code="lifecycle_fragment_allocation_incomplete",
                        message=(
                            "Official lifecycle exists, but local partial-close "
                            "contract quantities cannot be proven to conserve the "
                            "official close quantity."
                        ),
                        context={
                            "position_id": member.position_id,
                            "okx_pos_id": member.okx_pos_id,
                            "group_position_ids": list(member.lifecycle_group_position_ids),
                            "close_contracts": member.close_contracts,
                            "allocation_error": member.allocation_error,
                        },
                    )
        for candidate in candidates:
            checked += 1
            if candidate.position_id in allocation_failures:
                failures.append((candidate, allocation_failures[candidate.position_id]))
                continue
            result = await self._settle_candidate(
                candidate,
                started_at,
                position_history_rows=position_history_rows,
            )
            if isinstance(result, SettlementSuccess):
                result = _scale_settlement_success(result, candidate)
                _claim_history_row_for_position(result.row, candidate.position_id)
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
            failures.append((candidate, result))

        failure_results = await self._apply_failures(failures, started_at)
        for candidate, result in failures:
            quarantined = failure_results.get(candidate.position_id, False)
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
            sample["next_retry_seconds"] = self._failure_retry_seconds(
                quarantined=quarantined
            )
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

    async def _load_position_history_rows(self) -> list[dict[str, Any]]:
        """Load the shared OKX history mirror once for an entire sync batch."""

        async with self.session_context_factory() as session:
            records = await load_okx_position_history_records(
                session,
                mode=self.mode,
                limit=5000,
            )
        return okx_position_history_records_to_rows(records)

    def _failure_retry_seconds(self, *, quarantined: bool) -> float:
        if quarantined:
            return max(self.retry_seconds, POSITION_HISTORY_QUARANTINE_RETRY_SECONDS)
        return self.retry_seconds

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
        async with self.session_context_factory() as session:
            open_pos_ids_result = await session.execute(
                select(Position.okx_pos_id).where(
                    Position.execution_mode == self.mode,
                    Position.is_open.is_(True),
                    Position.okx_pos_id.is_not(None),
                )
            )
            open_pos_ids = {
                str(value or "").strip()
                for value in open_pos_ids_result.scalars().all()
                if str(value or "").strip()
            }
            result = await session.execute(
                select(Position)
                .where(
                    Position.execution_mode == self.mode,
                    Position.is_open.is_(False),
                    Position.closed_at.is_not(None),
                    Position.closed_at >= _db_naive(since),
                )
                .order_by(Position.closed_at.desc(), Position.id.desc())
                .limit(max(self.limit * 25, MINIMUM_CLOSED_LIFECYCLE_SCAN_ROWS))
            )
            rows = list(result.scalars().all())
            rows, duplicate_rows = _deduplicate_closed_lifecycle_rows(rows, now=now)
            # Fill-level contract facts are the only acceptable basis for
            # splitting a shared official lifecycle.  Keep them on the
            # candidate instead of inferring them from base quantity later.
            order_ids = {
                order_id
                for position in rows
                for order_id in (
                    _split_exchange_order_ids(getattr(position, "entry_exchange_order_id", None))
                    | _split_exchange_order_ids(getattr(position, "close_exchange_order_id", None))
                )
            }
            order_contracts: dict[str, float] = {}
            order_facts: dict[str, dict[str, float | None]] = {}
            if order_ids:
                order_result = await session.execute(
                    select(Order).where(
                        Order.execution_mode == self.mode,
                        Order.exchange_order_id.in_(sorted(order_ids)),
                    )
                )
                for order in order_result.scalars().all():
                    raw_order = getattr(order, "okx_raw_fills", None)
                    raw_order = raw_order if isinstance(raw_order, dict) else {}
                    contracts = _safe_float(
                        getattr(order, "okx_fill_contracts", None)
                        or raw_order.get("contracts")
                        or raw_order.get("filled_contracts"),
                        0.0,
                    )
                    if contracts > 0 and getattr(order, "exchange_order_id", None):
                        exchange_id = str(order.exchange_order_id)
                        order_contracts[exchange_id] = contracts
                        order_facts[exchange_id] = {
                            "fee": abs(_safe_float(getattr(order, "fee", None), 0.0)),
                            "fill_pnl": (
                                _safe_float(getattr(order, "okx_fill_pnl", None), 0.0)
                                if getattr(order, "okx_fill_pnl", None) is not None
                                else None
                            ),
                        }
            candidates: list[SettlementCandidate] = []
            restored_superseded = bool(duplicate_rows)
            selected_lifecycle_keys: set[tuple[str, ...]] = set()
            pending_lifecycle_keys = {
                _position_lifecycle_group_key(position)
                for position in rows
                if not is_final_settlement_status(getattr(position, "settlement_status", None))
                and not _is_non_retryable_settlement_status(position)
            }
            pending_lifecycle_keys.update(
                _position_lifecycle_group_key(position)
                for position in rows
                if is_final_settlement_status(getattr(position, "settlement_status", None))
                and _final_fragment_requires_settlement_repair(position)
            )
            for row in rows:
                raw = getattr(row, "settlement_raw", None)
                raw = raw if isinstance(raw, dict) else {}
                if _has_superseded_position_metadata(row, raw):
                    if _reactivate_distinct_superseded_fragment(row, rows, raw=raw, now=now):
                        raw = _safe_dict(getattr(row, "settlement_raw", None))
                        restored_superseded = True
                    else:
                        _restore_superseded_position_status(row, raw, now=now)
                        restored_superseded = True
                        continue
                lifecycle_key = _position_lifecycle_group_key(row)
                if (
                    is_final_settlement_status(getattr(row, "settlement_status", None))
                    and lifecycle_key not in pending_lifecycle_keys
                ):
                    continue
                if _is_non_retryable_settlement_status(row):
                    continue
                okx_pos_id = str(getattr(row, "okx_pos_id", "") or "").strip()
                if okx_pos_id and okx_pos_id in open_pos_ids:
                    _mark_lifecycle_still_open(row, raw, now=now)
                    restored_superseded = True
                    continue
                if _retry_after(raw, now):
                    continue
                if lifecycle_key not in selected_lifecycle_keys and len(selected_lifecycle_keys) >= self.limit:
                    continue
                selected_lifecycle_keys.add(lifecycle_key)
                if lifecycle_key:
                    candidate = _candidate_from_position(row, raw)
                    entry_contracts = sum(
                        order_contracts.get(order_id, 0.0)
                        for order_id in _split_exchange_order_ids(candidate.entry_exchange_order_id)
                    )
                    close_contracts = sum(
                        order_contracts.get(order_id, 0.0)
                        for order_id in _split_exchange_order_ids(candidate.close_exchange_order_id)
                    )
                    entry_ids = _split_exchange_order_ids(candidate.entry_exchange_order_id)
                    close_ids = _split_exchange_order_ids(candidate.close_exchange_order_id)
                    entry_fact_fees = [
                        order_facts[order_id]["fee"]
                        for order_id in entry_ids
                        if order_id in order_facts and order_facts[order_id]["fee"] is not None
                    ]
                    close_fact_fees = [
                        order_facts[order_id]["fee"]
                        for order_id in close_ids
                        if order_id in order_facts and order_facts[order_id]["fee"] is not None
                    ]
                    close_fill_pnls = [
                        order_facts[order_id]["fill_pnl"]
                        for order_id in close_ids
                        if order_id in order_facts and order_facts[order_id]["fill_pnl"] is not None
                    ]
                    candidates.append(
                        replace(
                            candidate,
                            entry_contracts=entry_contracts,
                            close_contracts=close_contracts,
                            entry_fee_authoritative=(
                                sum(float(value) for value in entry_fact_fees)
                                if entry_fact_fees
                                else None
                            ),
                            close_fee_authoritative=(
                                sum(float(value) for value in close_fact_fees)
                                if close_fact_fees
                                else None
                            ),
                            close_fill_pnl_authoritative=(
                                sum(float(value) for value in close_fill_pnls)
                                if close_fill_pnls
                                else None
                            ),
                        )
                    )
            if restored_superseded:
                await session.flush()
        return candidates

    async def _settle_candidate(
        self,
        candidate: SettlementCandidate,
        now: datetime,
        *,
        position_history_rows: list[dict[str, Any]] | None = None,
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
        rows = (
            position_history_rows
            if position_history_rows is not None
            else await self._load_position_history_rows()
        )
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
        match = _match_position_history_row(
            candidate,
            rows,
            inst_id=inst_id,
            allowed_position_ids=set(candidate.lifecycle_group_position_ids),
            allow_shared_lifecycle=candidate.lifecycle_group_size > 1,
            preferred_row_identity=candidate.history_row_identity,
            preferred_record_id=candidate.history_record_id,
        )
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
                allocation = _safe_dict(raw.get("lifecycle_allocation"))
                allocation_ids = tuple(
                    sorted(
                        _safe_int(value, 0)
                        for value in allocation.get("group_position_ids", [])
                        if _safe_int(value, 0) > 0
                    )
                )
                already_current = bool(
                    candidate.lifecycle_group_size <= 1
                    or (
                        allocation.get("basis") == candidate.allocation_basis
                        and allocation_ids == candidate.lifecycle_group_position_ids
                        and abs(
                            _safe_float(allocation.get("ratio"), -1.0)
                            - candidate.allocation_ratio
                        )
                        <= 1e-12
                    )
                )
                if already_current and not _final_fragment_requires_settlement_repair(position):
                    return False, None
            if _is_non_retryable_settlement_status(position):
                return False, None
            apply_position_settlement_snapshot(position, success.snapshot)
            authoritative_quantity = _authoritative_fragment_quantity(candidate, success.row)
            if authoritative_quantity > 0 and not _quantities_close(
                _safe_float(getattr(position, "quantity", None), 0.0),
                authoritative_quantity,
            ):
                position.quantity = authoritative_quantity
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
            if history is not None:
                entry_order_ids = _split_exchange_order_ids(
                    candidate.entry_exchange_order_id
                )
                close_order_ids = _split_exchange_order_ids(
                    candidate.close_exchange_order_id
                )
                history.position_ids = _merge_history_links(
                    history.position_ids,
                    {str(candidate.position_id)},
                )
                history.entry_order_ids = _merge_history_links(
                    history.entry_order_ids,
                    entry_order_ids,
                )
                history.close_order_ids = _merge_history_links(
                    history.close_order_ids,
                    close_order_ids,
                )
                history.linked_order_ids = _merge_history_links(
                    history.linked_order_ids,
                    entry_order_ids | close_order_ids,
                )
                history.match_status = "okx_position_settlement_linked"
                history.synced_at = now
            # A shared official lifecycle has one aggregate decision outcome;
            # syncing it once per local fragment would write the aggregate PnL
            # repeatedly into multiple decisions.  Fragment-level snapshots
            # remain available for display/training evidence, while the
            # lifecycle aggregate is handled by the history-level trainer.
            outcome_change = None
            if candidate.lifecycle_group_size <= 1:
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
                outcome_change if outcome_change and outcome_change.get("changed") is True else None,
            )

    async def _apply_failure(
        self,
        candidate: SettlementCandidate,
        failure: SettlementFailure,
        now: datetime,
    ) -> bool:
        results = await self._apply_failures([(candidate, failure)], now)
        return results.get(candidate.position_id, False)

    async def _apply_failures(
        self,
        failures: list[tuple[SettlementCandidate, SettlementFailure]],
        now: datetime,
    ) -> dict[int, bool]:
        if not failures:
            return {}
        candidate_ids = {candidate.position_id for candidate, _failure in failures}
        async with self.session_context_factory() as session:
            result = await session.execute(
                select(Position)
                .where(Position.id.in_(candidate_ids))
                .with_for_update(skip_locked=True)
            )
            positions = {int(position.id): position for position in result.scalars().all()}
            outcomes: dict[int, bool] = {}
            for candidate, failure in failures:
                outcomes[candidate.position_id] = self._apply_failure_to_position(
                    positions.get(candidate.position_id),
                    candidate=candidate,
                    failure=failure,
                    now=now,
                )
            await session.flush()
            return outcomes

    def _apply_failure_to_position(
        self,
        position: Position | None,
        *,
        candidate: SettlementCandidate,
        failure: SettlementFailure,
        now: datetime,
    ) -> bool:
        if position is None or bool(position.is_open):
            return False
        raw = getattr(position, "settlement_raw", None)
        raw = raw if isinstance(raw, dict) else {}
        if _has_superseded_position_metadata(position, raw):
            _restore_superseded_position_status(position, raw, now=now)
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
            failure.code
            in {
                "positions_history_no_matching_row",
                "lifecycle_fragment_allocation_incomplete",
            }
            and quarantine_triggers
        )
        retry_seconds = self._failure_retry_seconds(quarantined=quarantined)
        next_retry_at = now + timedelta(seconds=retry_seconds)
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
            updated_raw.update(
                {
                    "next_settlement_retry_at": next_retry_at.isoformat(),
                    "quarantine_reason": (
                        "lifecycle_fragment_contract_conservation_unresolved"
                        if failure.code == "lifecycle_fragment_allocation_incomplete"
                        else "official_position_history_identity_unresolved"
                    ),
                    "quarantined_at": now.isoformat(),
                    "quarantine_evidence": {
                        "triggers": quarantine_triggers,
                        "attempt_count": attempts,
                        "max_attempts": POSITION_HISTORY_MATCH_MAX_ATTEMPTS,
                        "closed_at": _iso(closed_at),
                        "closed_age_hours": closed_age_hours,
                        "max_age_hours": POSITION_HISTORY_MATCH_MAX_AGE_HOURS,
                    },
                    "retry_policy": (
                        f"quarantined from authority; retry every {retry_seconds:g}s "
                        "until OKX official settlement identity is available"
                    ),
                }
            )
        else:
            updated_raw.update(
                {
                    "next_settlement_retry_at": next_retry_at.isoformat(),
                    "retry_policy": (
                        f"retry every {retry_seconds:g}s until OKX official "
                        "settlement is available"
                    ),
                }
            )
        position.settlement_raw = updated_raw
        position.updated_at = now
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
        # Distinct partial-close fills under one posId are valid fragments.  A
        # row is only retired when it projects the same close order set (or a
        # strict overlapping subset) as another row; disjoint close orders must
        # remain separate lifecycle fragments for allocation.
        partitions: list[list[Position]] = []
        for candidate in candidates:
            close_ids = _split_exchange_order_ids(
                getattr(candidate, "close_exchange_order_id", None)
            )
            matched_partition = next(
                (
                    partition
                    for partition in partitions
                    if close_ids
                    and any(
                        close_ids.intersection(
                            _split_exchange_order_ids(
                                getattr(item, "close_exchange_order_id", None)
                            )
                        )
                        for item in partition
                    )
                ),
                None,
            )
            if matched_partition is None:
                partitions.append([candidate])
            else:
                matched_partition.append(candidate)
        if len(partitions) > 1:
            for partition in partitions:
                if len(partition) == 1:
                    retained.append(partition[0])
                else:
                    partition_retained, partition_duplicates = _deduplicate_closed_lifecycle_rows(
                        partition,
                        now=now,
                    )
                    retained.extend(partition_retained)
                    duplicates.extend(partition_duplicates)
            continue
        # A stale projection can contain only a subset of the close orders while
        # still carrying the same OKX lifecycle identity. Keep the row with the
        # richest exchange evidence before falling back to its stable local ID.
        canonical = min(
            candidates,
            key=lambda item: (
                -len(_split_exchange_order_ids(getattr(item, "close_exchange_order_id", None))),
                -int(
                    str(getattr(item, "settlement_status", "") or "").lower()
                    in {"reconciled", "okx_position_history"}
                ),
                int(getattr(item, "id", 0) or 0),
            ),
        )
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
    quantity = abs(_safe_float(getattr(position, "quantity", None), 0.0))
    if not pos_id or not entry_ids or quantity <= 0:
        return None
    return (
        str(getattr(position, "execution_mode", "") or "").lower(),
        pos_id,
        normalize_trading_symbol(str(getattr(position, "symbol", "") or "")),
        str(getattr(position, "side", "") or "").lower(),
        entry_ids,
        round(quantity, 12),
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


def _lifecycle_key(candidate: SettlementCandidate) -> tuple[str, ...]:
    """Return the official lifecycle identity, excluding local fragment size."""

    return (
        candidate.okx_inst_id or okx_inst_id_from_symbol(candidate.symbol),
        candidate.okx_pos_id,
        candidate.side,
        tuple(sorted(_split_exchange_order_ids(candidate.entry_exchange_order_id))),
    )


def _position_lifecycle_group_key(position: Position) -> tuple[str, ...]:
    key = (
        str(getattr(position, "execution_mode", "") or "").lower(),
        str(getattr(position, "okx_inst_id", "") or "").upper(),
        str(getattr(position, "okx_pos_id", "") or "").strip(),
        str(getattr(position, "side", "") or "").lower(),
        tuple(
            sorted(
                _split_exchange_order_ids(
                    getattr(position, "entry_exchange_order_id", None)
                )
            )
        ),
    )
    if not key[2]:
        return (*key, str(getattr(position, "id", "") or ""))
    return key


def _group_candidates_by_lifecycle(
    candidates: list[SettlementCandidate],
) -> dict[tuple[str, ...], list[SettlementCandidate]]:
    grouped: dict[tuple[str, ...], list[SettlementCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(_lifecycle_key(candidate), []).append(candidate)
    for members in grouped.values():
        members.sort(key=lambda item: (item.closed_at or datetime.min.replace(tzinfo=UTC), item.position_id))
    return grouped


def _official_lifecycle_contracts(row: dict[str, Any]) -> float:
    value, key = _first_present_float(row, ("closeTotalPos", "closed_quantity", "contracts"))
    return value if key and value > 0 else 0.0


def _prepare_lifecycle_allocations(
    members: list[SettlementCandidate],
    history_rows: list[dict[str, Any]],
) -> None:
    """Attach a proven allocation to each local fragment in-place.

    Multi-fragment allocation is enabled only when every fragment has an
    authoritative close-fill contract count and the counts conserve the
    official row.  Otherwise the entire group remains quarantined.
    """

    ids = tuple(sorted(int(item.position_id) for item in members))
    for member in members:
        object.__setattr__(member, "lifecycle_group_size", len(members))
        object.__setattr__(member, "lifecycle_group_position_ids", ids)
    if len(members) <= 1:
        return
    first = members[-1]
    matched = _match_position_history_row(
        first,
        history_rows,
        inst_id=first.okx_inst_id or okx_inst_id_from_symbol(first.symbol),
        allowed_position_ids=set(ids),
        allow_shared_lifecycle=True,
    )
    official_contracts = (
        _official_lifecycle_contracts(matched[0])
        if not isinstance(matched, SettlementFailure)
        else 0.0
    )
    if not isinstance(matched, SettlementFailure):
        matched_row = matched[0]
        identity = _history_row_identity(matched_row)
        record_id = _safe_int(matched_row.get("_dashboard_history_record_id"), 0)
        for member in members:
            object.__setattr__(member, "history_row_identity", identity)
            object.__setattr__(member, "history_record_id", record_id)
    close_contracts = [max(float(item.close_contracts or 0.0), 0.0) for item in members]
    total_close_contracts = sum(close_contracts)
    contract_allocation_complete = bool(
        not isinstance(matched, SettlementFailure)
        and official_contracts > 0
        and total_close_contracts > 0
        and all(value > 0 for value in close_contracts)
        and abs(total_close_contracts - official_contracts)
        <= max(official_contracts * 1e-6, 1e-9)
    )
    entry_facts_complete = all(item.entry_fee_authoritative is not None for item in members)
    if contract_allocation_complete and entry_facts_complete:
        # Entry orders are often shared by all local partial-close fragments.
        # Allocate their one authoritative fee once across the same lifecycle
        # contract ratios; never copy it into every fragment.
        entry_fee_total = max(
            (abs(float(item.entry_fee_authoritative)) for item in members if item.entry_fee_authoritative is not None),
            default=0.0,
        )
        for member, allocated in zip(members, close_contracts, strict=False):
            object.__setattr__(
                member,
                "entry_fee_authoritative",
                entry_fee_total * allocated / official_contracts,
            )
    close_facts_complete = all(
        item.close_fee_authoritative is not None
        and item.close_fill_pnl_authoritative is not None
        for item in members
    )
    economics_conserved = False
    if (
        contract_allocation_complete
        and entry_facts_complete
        and close_facts_complete
        and not isinstance(matched, SettlementFailure)
    ):
        official_realized, realized_key = _first_present_float(
            matched[0],
            ("realizedPnl", "realized_pnl", "realizedPnlInUsd", "realizedPnlUsd"),
        )
        official_funding, funding_key = _first_present_float(
            matched[0],
            ("fundingFee", "funding_fee"),
        )
        component_realized = (
            sum(float(item.close_fill_pnl_authoritative or 0.0) for item in members)
            + official_funding
            - sum(abs(float(item.entry_fee_authoritative or 0.0)) for item in members)
            - sum(abs(float(item.close_fee_authoritative or 0.0)) for item in members)
        )
        economics_conserved = bool(
            realized_key is not None
            and abs(component_realized - official_realized)
            <= max(abs(official_realized) * 1e-6, 1e-7)
        )
    complete = contract_allocation_complete and economics_conserved
    allocation_error = ""
    if not contract_allocation_complete:
        allocation_error = "close_fill_contracts_do_not_conserve_official_quantity"
    elif not entry_facts_complete or not close_facts_complete:
        allocation_error = "authoritative_order_economics_incomplete"
    elif not economics_conserved:
        allocation_error = "authoritative_order_economics_do_not_conserve_realized_pnl"
    for member, allocated in zip(members, close_contracts, strict=False):
        ratio = (
            allocated / official_contracts
            if contract_allocation_complete and official_contracts > 0
            else 1.0
        )
        object.__setattr__(member, "allocation_ratio", ratio)
        object.__setattr__(
            member,
            "allocation_basis",
            "okx_close_fill_contracts_and_order_economics",
        )
        object.__setattr__(member, "allocation_complete", complete)
        object.__setattr__(member, "allocation_error", allocation_error)


def _scale_settlement_success(
    success: SettlementSuccess,
    candidate: SettlementCandidate,
) -> SettlementSuccess:
    ratio = min(max(float(candidate.allocation_ratio or 1.0), 0.0), 1.0)
    has_authoritative_economics = any(
        value is not None
        for value in (
            candidate.entry_fee_authoritative,
            candidate.close_fee_authoritative,
            candidate.close_fill_pnl_authoritative,
        )
    )
    if (
        not has_authoritative_economics
        and (candidate.lifecycle_group_size <= 1 or abs(ratio - 1.0) <= 1e-12)
    ):
        return success
    snapshot = success.snapshot
    lifecycle_total_fee = abs(snapshot.entry_fee) + abs(snapshot.close_fee)
    local_total_fee = abs(candidate.entry_fee) + abs(candidate.close_fee)
    entry_share = (
        abs(candidate.entry_fee_authoritative)
        if candidate.entry_fee_authoritative is not None
        else lifecycle_total_fee * abs(candidate.entry_fee) / local_total_fee
        if local_total_fee > 0
        else 0.0
    )
    close_share = (
        abs(candidate.close_fee_authoritative)
        if candidate.close_fee_authoritative is not None
        else lifecycle_total_fee - entry_share
    )
    # A lifecycle's official realized PnL is the conservation anchor.  When
    # OKX native close fills provide per-order fillPnl, keep that exact value;
    # otherwise fall back to the proven contract allocation.
    close_fill_pnl = (
        float(candidate.close_fill_pnl_authoritative)
        if candidate.close_fill_pnl_authoritative is not None
        else snapshot.close_fill_pnl * ratio
    )
    entry_fee_allocated = (
        entry_share
        if candidate.entry_fee_authoritative is not None
        else entry_share * ratio
    )
    close_fee_allocated = (
        close_share
        if candidate.close_fee_authoritative is not None
        else close_share * ratio
    )
    scaled = build_position_settlement_snapshot(
        close_fill_pnl=close_fill_pnl,
        entry_fee=entry_fee_allocated,
        close_fee=close_fee_allocated,
        funding_fee=snapshot.funding_fee * ratio,
        status=snapshot.status,
        source=snapshot.source,
        synced_at=snapshot.synced_at,
        raw={
            **dict(snapshot.raw or {}),
            "lifecycle_allocation": {
                "ratio": ratio,
                "basis": candidate.allocation_basis,
                "allocated_contracts": candidate.close_contracts,
                "group_position_ids": list(candidate.lifecycle_group_position_ids),
                "entry_fee_authoritative": candidate.entry_fee_authoritative,
                "close_fee_authoritative": candidate.close_fee_authoritative,
                "close_fill_pnl_authoritative": candidate.close_fill_pnl_authoritative,
            },
        },
    )
    return replace(success, snapshot=scaled)


def _contract_size_from_history_row(row: dict[str, Any]) -> float:
    spec = row.get("_bb_contract_spec")
    if isinstance(spec, dict):
        value = _safe_float(spec.get("ctVal") or spec.get("contract_size"), 0.0)
        if value > 0:
            return value
    return _safe_float(row.get("contract_size") or row.get("ctVal"), 0.0)


def _authoritative_fragment_quantity(
    candidate: SettlementCandidate,
    row: dict[str, Any],
) -> float:
    contracts = max(float(candidate.close_contracts or 0.0), 0.0)
    contract_size = _contract_size_from_history_row(row)
    if contracts > 0 and contract_size > 0:
        return contracts * contract_size
    return 0.0


def _quantities_close(left: float, right: float) -> bool:
    return abs(left - right) <= max(abs(left) * 1e-6, abs(right) * 1e-6, 1e-8)


def _final_fragment_requires_quantity_repair(position: Position) -> bool:
    raw = _safe_dict(getattr(position, "settlement_raw", None))
    allocation = _safe_dict(raw.get("lifecycle_allocation"))
    allocated_contracts = _safe_float(allocation.get("allocated_contracts"), 0.0)
    history_row = _safe_dict(raw.get("okx_position_history_row"))
    contract_size = _contract_size_from_history_row(history_row)
    expected = allocated_contracts * contract_size
    current = _safe_float(getattr(position, "quantity", None), 0.0)
    return allocated_contracts > 0 and contract_size > 0 and not _quantities_close(current, expected)


def _final_fragment_requires_settlement_repair(position: Position) -> bool:
    if _final_fragment_requires_quantity_repair(position):
        return True
    raw = _safe_dict(getattr(position, "settlement_raw", None))
    allocation = _safe_dict(raw.get("lifecycle_allocation"))
    return str(allocation.get("basis") or "").strip() not in {
        "",
        "okx_close_fill_contracts_and_order_economics",
        "single_fragment",
    }


def _match_position_history_row(
    candidate: SettlementCandidate,
    rows: list[dict[str, Any]],
    *,
    inst_id: str,
    allowed_position_ids: set[int] | None = None,
    allow_shared_lifecycle: bool = False,
    preferred_row_identity: str = "",
    preferred_record_id: int = 0,
) -> tuple[dict[str, Any], str] | SettlementFailure:
    scored: list[tuple[int, float, dict[str, Any], str]] = []
    close_order_ids = _split_exchange_order_ids(candidate.close_exchange_order_id)
    entry_order_ids = _split_exchange_order_ids(candidate.entry_exchange_order_id)
    for row in rows:
        row_identity = _history_row_identity(row)
        row_record_id = _safe_int(row.get("_dashboard_history_record_id"), 0)
        if preferred_row_identity and row_identity != preferred_row_identity:
            continue
        if preferred_record_id > 0 and row_record_id != preferred_record_id:
            continue
        row_inst_id = _position_history_inst_id(row)
        if row_inst_id and row_inst_id != inst_id:
            continue
        row_pos_id = _position_history_pos_id(row)
        linked_position_ids = _history_row_position_ids(row)
        if linked_position_ids and not allow_shared_lifecycle:
            if str(candidate.position_id) not in linked_position_ids:
                continue
        if (
            allow_shared_lifecycle
            and allowed_position_ids
            and linked_position_ids
            and not linked_position_ids.intersection({str(value) for value in allowed_position_ids})
        ):
            continue
        row_side = _position_history_side(row)
        if candidate.okx_pos_id and row_pos_id and row_pos_id != candidate.okx_pos_id:
            continue
        if candidate.side and row_side in {"long", "short"} and row_side != candidate.side:
            continue
        row_closed_at = _position_history_closed_at(row)
        signed_close_delta = _signed_time_delta_seconds(
            candidate.closed_at,
            row_closed_at,
        )
        preferred_identity_match = bool(
            (preferred_row_identity and row_identity == preferred_row_identity)
            or (preferred_record_id > 0 and row_record_id == preferred_record_id)
        )
        if not preferred_identity_match and signed_close_delta is not None and (
            signed_close_delta < -POSITION_HISTORY_CLOSE_EARLY_TOLERANCE_SECONDS
            or signed_close_delta > POSITION_HISTORY_CLOSE_MATCH_WINDOW_SECONDS
        ):
            continue
        score = 0
        reasons: list[str] = []
        if preferred_identity_match:
            score += 1000
            reasons.append("lifecycle_history_identity")
        if candidate.okx_pos_id and row_pos_id == candidate.okx_pos_id:
            score += 100
            reasons.append("pos_id_exact")
        if row_inst_id == inst_id:
            score += 20
            reasons.append("inst_id")
        if candidate.side and row_side == candidate.side:
            score += 15
            reasons.append("side")
        closed_delta = _time_delta_seconds(candidate.closed_at, row_closed_at)
        if closed_delta is not None:
            score += max(0, 40 - int(closed_delta // 60))
            reasons.append(f"closed_at_delta={int(closed_delta)}s")
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
        str(raw.get("reason") or "") in SUPERSEDED_POSITION_REASONS
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


def _reactivate_distinct_superseded_fragment(
    position: Position,
    rows: list[Position],
    *,
    raw: dict[str, Any],
    now: datetime,
) -> bool:
    """Recover a legitimate partial close retired by the legacy deduplicator."""

    if str(raw.get("reason") or "") != DUPLICATE_CLOSED_POSITION_REASON:
        return False
    canonical_id = _safe_int(raw.get("canonical_position_id"), 0)
    canonical = next(
        (item for item in rows if int(getattr(item, "id", 0) or 0) == canonical_id),
        None,
    )
    if canonical is None:
        return False
    fragment_close_ids = _split_exchange_order_ids(
        getattr(position, "close_exchange_order_id", None)
    )
    canonical_close_ids = _split_exchange_order_ids(
        getattr(canonical, "close_exchange_order_id", None)
    )
    if (
        not fragment_close_ids
        or not canonical_close_ids
        or fragment_close_ids.intersection(canonical_close_ids)
    ):
        return False
    position.settlement_status = "settling"
    position.settlement_source = "okx_lifecycle_fragment_recovery"
    position.settlement_synced_at = now
    position.settlement_raw = {
        **raw,
        "status": "settling",
        "source": "okx_lifecycle_fragment_recovery",
        "reason": DISTINCT_CLOSED_FRAGMENT_REACTIVATED_REASON,
        "legacy_superseded_reason": DUPLICATE_CLOSED_POSITION_REASON,
        "legacy_canonical_position_id": canonical_id,
        "reactivated_at": now.isoformat(),
        "next_settlement_retry_at": now.isoformat(),
    }
    position.updated_at = now
    return True


def _mark_lifecycle_still_open(
    position: Position,
    raw: dict[str, Any],
    *,
    now: datetime,
) -> None:
    """Keep a partial local close pending while its OKX posId remains open.

    OKX uses one posId for a position lifecycle, including partial closes.  A
    closed local child therefore cannot be finalized or quarantined while the
    exchange still reports that lifecycle as open.
    """

    previous_status = str(getattr(position, "settlement_status", "") or "")
    previous_source = str(getattr(position, "settlement_source", "") or "")
    original_status = str(
        raw.get("lifecycle_open_original_settlement_status") or previous_status
    )
    original_source = str(
        raw.get("lifecycle_open_original_settlement_source") or previous_source
    )
    previous_attempt_count = max(
        _safe_int(raw.get("lifecycle_open_previous_attempt_count"), 0),
        _safe_int(raw.get("settlement_attempt_count"), 0),
    )
    next_retry_at = now + timedelta(seconds=POSITION_HISTORY_QUARANTINE_RETRY_SECONDS)
    position.settlement_status = "settling"
    position.settlement_source = SETTLEMENT_LIFECYCLE_OPEN_SOURCE
    position.settlement_synced_at = now
    position.settlement_raw = {
        **raw,
        "status": "settling",
        "source": SETTLEMENT_LIFECYCLE_OPEN_SOURCE,
        "reason": SETTLEMENT_LIFECYCLE_OPEN_REASON,
        "previous_settlement_status": previous_status,
        "previous_settlement_source": previous_source,
        "lifecycle_open_original_settlement_status": original_status,
        "lifecycle_open_original_settlement_source": original_source,
        "lifecycle_open_previous_attempt_count": previous_attempt_count,
        "settlement_attempt_count": 0,
        "lifecycle_open_checked_at": now.isoformat(),
        "next_settlement_retry_at": next_retry_at.isoformat(),
        "retry_policy": (
            "wait for OKX official lifecycle settlement while the same posId "
            f"remains open; retry every {POSITION_HISTORY_QUARANTINE_RETRY_SECONDS:g}s"
        ),
    }
    position.updated_at = now


def _claim_history_row_for_position(row: dict[str, Any], position_id: int) -> None:
    """Attach one local fragment to its shared official lifecycle row."""

    if not isinstance(row, dict) or int(position_id or 0) <= 0:
        return
    linked = _history_row_position_ids(row)
    linked.add(str(int(position_id)))
    row["_dashboard_position_ids"] = sorted(linked)


def _history_row_position_ids(row: dict[str, Any]) -> set[str]:
    values = row.get("_dashboard_position_ids")
    if values is None:
        values = row.get("position_ids")
    if isinstance(values, (list, tuple, set)):
        return {str(value).strip() for value in values if str(value).strip()}
    return _split_exchange_order_ids(values)


def _history_row_identity(row: dict[str, Any]) -> str:
    value = str(row.get("_dashboard_history_row_identity") or "").strip()
    if value:
        return value
    return okx_position_history_row_identity(row, mode="paper")


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


def _merge_history_links(existing: Any, incoming: set[str]) -> list[str]:
    values = {
        str(value or "").strip()
        for value in (existing if isinstance(existing, list) else [])
        if str(value or "").strip()
    }
    values.update(str(value).strip() for value in incoming if str(value).strip())
    return sorted(values)


def _time_delta_seconds(left: datetime | None, right: datetime | None) -> float | None:
    left = _aware_utc(left)
    right = _aware_utc(right)
    if left is None or right is None:
        return None
    return abs((left - right).total_seconds())


def _signed_time_delta_seconds(
    left: datetime | None,
    right: datetime | None,
) -> float | None:
    left = _aware_utc(left)
    right = _aware_utc(right)
    if left is None or right is None:
        return None
    return (right - left).total_seconds()


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
