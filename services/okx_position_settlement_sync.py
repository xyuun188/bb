"""Finalize closed-position settlement from OKX official position history.

The trading flow stores a closed position immediately so the local state is
safe after a fill.  This service turns that provisional local close into a
final history/training fact only after OKX positions-history confirms the
official realized PnL and funding fee.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import or_, select

from core.safe_output import safe_error_text
from core.symbols import (
    normalize_trading_symbol,
    okx_inst_id_from_symbol,
    symbol_from_okx_inst_id,
)
from db.session import get_session_ctx
from executor.okx_executor import OKXExecutor
from models.decision import AIDecision
from models.trade import Order, Position
from services.entry_decision_settlement import (
    backfill_settled_entry_decision_outcomes,
    sync_settled_entry_decision_outcome,
)
from services.okx_native_facts import OkxNativeAccountBill, OkxNativeFactsClient
from services.okx_order_fact_sync import authoritative_order_fee_fact_source
from services.okx_position_history_store import upsert_okx_position_history_row
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
DEFAULT_SETTLEMENT_TIMEOUT_SECONDS = 6.0
POSITION_HISTORY_CLOSE_MATCH_WINDOW_SECONDS = 45 * 60
POSITION_HISTORY_OPEN_MATCH_WINDOW_SECONDS = 24 * 60 * 60
SUPERSEDED_POSITION_STATUS = "superseded_position_residual"
SUPERSEDED_POSITION_SOURCE = "okx_current_position_deduplication"
SUPERSEDED_POSITION_REASON = "duplicate_local_open_position_for_same_okx_pos_id"
DUPLICATE_CLOSED_POSITION_REASON = "duplicate_local_closed_position_for_same_okx_lifecycle"
VERIFIED_EXECUTION_PAIR_HISTORY_SOURCE = "okx_verified_execution_pair_settlement"

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
    history_source: str = "okx_position_settlement_sync"


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
        timeout_seconds: float = DEFAULT_SETTLEMENT_TIMEOUT_SECONDS,
        executor_factory: Any | None = None,
        session_context_factory: SessionContextFactory = get_session_ctx,
    ) -> None:
        self.mode = "live" if str(mode or "").lower() == "live" else "paper"
        self.lookback_hours = max(
            1, min(int(lookback_hours or DEFAULT_SETTLEMENT_LOOKBACK_HOURS), 24 * 14)
        )
        self.limit = max(1, min(int(limit or DEFAULT_SETTLEMENT_LIMIT), 100))
        self.retry_seconds = max(1.0, float(retry_seconds or DEFAULT_SETTLEMENT_RETRY_SECONDS))
        self.timeout_seconds = max(
            1.0, float(timeout_seconds or DEFAULT_SETTLEMENT_TIMEOUT_SECONDS)
        )
        self.executor_factory = executor_factory or OKXExecutor
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

        executor = self.executor_factory(mode=self.mode, load_markets_on_initialize=False)
        samples: list[dict[str, Any]] = []
        checked = 0
        reconciled = 0
        decision_outcome_count = len(decision_outcome_changes)
        exceptions = 0
        skipped = 0
        samples.extend(decision_outcome_changes[-10:])
        fatal_error: str | None = None
        try:
            await asyncio.wait_for(executor.initialize(), timeout=min(self.timeout_seconds, 3.0))
            native_facts = OkxNativeFactsClient(executor)
            for candidate in candidates:
                checked += 1
                result = await self._settle_candidate(candidate, native_facts, started_at)
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
                await self._apply_failure(candidate, result, started_at)
                exceptions += 1
                samples.append(
                    {
                        "kind": "okx_position_settlement_exception",
                        "position_id": candidate.position_id,
                        "symbol": candidate.symbol,
                        "side": candidate.side,
                        "error_code": result.code,
                        "error_message": result.message,
                        "next_retry_seconds": self.retry_seconds,
                    }
                )
        except Exception as exc:
            fatal_error = safe_error_text(exc, limit=220)
            logger.warning(
                "OKX position settlement sync failed",
                mode=self.mode,
                error=fatal_error,
            )
        finally:
            try:
                await executor.shutdown()
            except Exception as exc:
                logger.debug(
                    "OKX position settlement executor shutdown failed",
                    error=safe_error_text(exc, limit=120),
                )

        status = "ok"
        if fatal_error:
            status = "degraded"
        elif exceptions:
            status = "warning"
        return OkxPositionSettlementSyncSummary(
            status=status,
            mode=self.mode,
            checked_count=checked,
            reconciled_count=reconciled,
            decision_outcome_count=decision_outcome_count,
            exception_count=exceptions,
            skipped_count=skipped,
            samples=tuple(samples[-10:]),
            error=fatal_error,
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
                        Position.settlement_status != "superseded_position_residual",
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
        native_facts: OkxNativeFactsClient,
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
        try:
            rows = await asyncio.wait_for(
                native_facts.fetch_position_history_rows(
                    inst_ids=[inst_id],
                    pos_ids=[candidate.okx_pos_id] if candidate.okx_pos_id else None,
                    since=since,
                    limit=100,
                    max_pages=2,
                    strict=True,
                ),
                timeout=max(1.0, min(self.timeout_seconds * 0.65, 4.0)),
            )
        except Exception as exc:
            return SettlementFailure(
                code="positions_history_api_error",
                message=safe_error_text(exc, limit=240),
                context={
                    "position_id": candidate.position_id,
                    "inst_id": inst_id,
                    "okx_pos_id": candidate.okx_pos_id,
                    "api": "privateGetAccountPositionsHistory",
                },
            )
        if not rows:
            execution_pair = await self._success_from_verified_execution_pair(
                candidate,
                native_facts,
                now=now,
                inst_id=inst_id,
                created_at=created_at,
                closed_at=closed_at,
            )
            if isinstance(execution_pair, SettlementSuccess):
                return execution_pair
            if execution_pair.code in {
                "current_positions_api_error",
                "funding_fee_api_error",
            }:
                return execution_pair
            return SettlementFailure(
                code="positions_history_no_rows",
                message="OKX positions-history returned no rows for the position identity yet.",
                context={
                    "position_id": candidate.position_id,
                    "inst_id": inst_id,
                    "okx_pos_id": candidate.okx_pos_id,
                    "since": since.isoformat(),
                    "execution_pair_fallback_code": execution_pair.code,
                    "execution_pair_fallback_message": execution_pair.message,
                },
            )
        match = _match_position_history_row(candidate, rows, inst_id=inst_id)
        if isinstance(match, SettlementFailure):
            return match
        row, match_reason = match
        return await self._success_from_position_history_row(
            candidate,
            row,
            native_facts,
            now=now,
            match_reason=match_reason,
            inst_id=inst_id,
            created_at=created_at,
            closed_at=closed_at,
        )

    async def _success_from_verified_execution_pair(
        self,
        candidate: SettlementCandidate,
        native_facts: OkxNativeFactsClient,
        *,
        now: datetime,
        inst_id: str,
        created_at: datetime,
        closed_at: datetime,
    ) -> SettlementSuccess | SettlementFailure:
        entry_ids = sorted(_split_exchange_order_ids(candidate.entry_exchange_order_id))
        close_ids = sorted(_split_exchange_order_ids(candidate.close_exchange_order_id))
        if not entry_ids or not close_ids:
            return SettlementFailure(
                code="execution_pair_lineage_incomplete",
                message="Verified execution-pair settlement requires entry and close order IDs.",
                context={"position_id": candidate.position_id},
            )
        if not candidate.okx_pos_id:
            return SettlementFailure(
                code="execution_pair_position_identity_incomplete",
                message="Verified execution-pair settlement requires the OKX position ID.",
                context={"position_id": candidate.position_id, "inst_id": inst_id},
            )
        all_ids = sorted({*entry_ids, *close_ids})
        async with self.session_context_factory() as session:
            order_result = await session.execute(
                select(Order).where(
                    Order.execution_mode == self.mode,
                    Order.exchange_order_id.in_(all_ids),
                )
            )
            orders = list(order_result.scalars().all())
            decision_ids = {
                int(order.decision_id or 0)
                for order in orders
                if int(order.decision_id or 0) > 0
            }
            decision_result = (
                await session.execute(
                    select(AIDecision).where(AIDecision.id.in_(sorted(decision_ids)))
                )
                if decision_ids
                else None
            )
            decisions = (
                list(decision_result.scalars().all())
                if decision_result is not None
                else []
            )
        orders_by_id = {
            str(order.exchange_order_id or "").strip(): order
            for order in orders
            if str(order.exchange_order_id or "").strip()
        }
        if any(order_id not in orders_by_id for order_id in all_ids):
            return SettlementFailure(
                code="execution_pair_order_missing",
                message="One or more linked execution-pair orders are missing.",
                context={"position_id": candidate.position_id, "order_ids": all_ids},
            )
        entry_orders = [orders_by_id[order_id] for order_id in entry_ids]
        close_orders = [orders_by_id[order_id] for order_id in close_ids]
        order_fee_sources: dict[str, str] = {}
        for order_id in all_ids:
            source = authoritative_order_fee_fact_source(
                orders_by_id[order_id],
                order_id=order_id,
            )
            if source is None:
                return SettlementFailure(
                    code="execution_pair_order_fact_incomplete",
                    message="A linked order lacks verified OKX quantity, price, or fee facts.",
                    context={"position_id": candidate.position_id, "order_id": order_id},
                )
            order_fee_sources[order_id] = source

        entry_fill_times = [
            _aware_utc(getattr(order, "filled_at", None)) for order in entry_orders
        ]
        close_fill_times = [
            _aware_utc(getattr(order, "filled_at", None)) for order in close_orders
        ]
        if any(value is None for value in (*entry_fill_times, *close_fill_times)):
            return SettlementFailure(
                code="execution_pair_order_timestamp_incomplete",
                message="A linked order lacks its OKX-confirmed fill timestamp.",
                context={"position_id": candidate.position_id, "order_ids": all_ids},
            )
        settlement_opened_at = min(value for value in entry_fill_times if value is not None)
        settlement_closed_at = max(value for value in close_fill_times if value is not None)
        if settlement_closed_at < settlement_opened_at:
            return SettlementFailure(
                code="execution_pair_timestamp_order_invalid",
                message="Verified close fill time precedes the entry fill time.",
                context={
                    "position_id": candidate.position_id,
                    "opened_at": settlement_opened_at.isoformat(),
                    "closed_at": settlement_closed_at.isoformat(),
                },
            )

        try:
            current_positions = await asyncio.wait_for(
                native_facts.fetch_positions(inst_ids=[inst_id]),
                timeout=max(1.0, min(self.timeout_seconds * 0.25, 2.0)),
            )
        except Exception as exc:
            return SettlementFailure(
                code="current_positions_api_error",
                message=safe_error_text(exc, limit=240),
                context={
                    "position_id": candidate.position_id,
                    "inst_id": inst_id,
                    "okx_pos_id": candidate.okx_pos_id,
                    "api": "privateGetAccountPositions",
                },
            )
        if _has_matching_current_position(
            current_positions,
            inst_id=inst_id,
            okx_pos_id=candidate.okx_pos_id,
        ):
            return SettlementFailure(
                code="execution_pair_lifecycle_still_open",
                message="The linked OKX position lifecycle is still open.",
                context={
                    "position_id": candidate.position_id,
                    "inst_id": inst_id,
                    "okx_pos_id": candidate.okx_pos_id,
                },
            )

        decisions_by_id = {int(decision.id): decision for decision in decisions}
        entry_decision = next(
            (
                decisions_by_id.get(int(order.decision_id or 0))
                for order in entry_orders
                if int(order.decision_id or 0) > 0
            ),
            None,
        )
        contract_spec = _verified_entry_contract_spec(entry_decision, inst_id=inst_id)
        if not contract_spec:
            return SettlementFailure(
                code="execution_pair_contract_spec_incomplete",
                message="Entry decision lacks the verified OKX public contract specification.",
                context={"position_id": candidate.position_id, "inst_id": inst_id},
            )

        entry_quantity = sum(_order_base_quantity(order) for order in entry_orders)
        close_quantity = sum(_order_base_quantity(order) for order in close_orders)
        quantity_tolerance = max(abs(candidate.quantity) * 0.001, 1e-9)
        if (
            candidate.quantity <= 0
            or entry_quantity + quantity_tolerance < candidate.quantity
            or abs(close_quantity - candidate.quantity) > quantity_tolerance
        ):
            return SettlementFailure(
                code="execution_pair_quantity_mismatch",
                message="Verified entry/close quantities do not cover the closed position.",
                context={
                    "position_id": candidate.position_id,
                    "position_quantity": candidate.quantity,
                    "entry_quantity": entry_quantity,
                    "close_quantity": close_quantity,
                },
            )
        close_gross_pnl = 0.0
        close_notional = 0.0
        close_fee = 0.0
        for order in close_orders:
            raw = _safe_dict(getattr(order, "okx_raw_fills", None))
            if raw.get("fill_pnl") is None and getattr(order, "okx_fill_pnl", None) is None:
                return SettlementFailure(
                    code="execution_pair_close_pnl_missing",
                    message="Verified close execution lacks exchange fill PnL.",
                    context={
                        "position_id": candidate.position_id,
                        "order_id": getattr(order, "exchange_order_id", None),
                    },
                )
            quantity = _order_base_quantity(order)
            price = _safe_float(getattr(order, "price", None), 0.0)
            close_notional += quantity * price
            close_gross_pnl += _safe_float(
                raw.get("fill_pnl")
                if raw.get("fill_pnl") is not None
                else getattr(order, "okx_fill_pnl", None),
                0.0,
            )
            close_fee += abs(_safe_float(raw.get("fee_abs"), 0.0))
        total_entry_fee = sum(
            abs(_safe_float(_safe_dict(order.okx_raw_fills).get("fee_abs"), 0.0))
            for order in entry_orders
        )
        entry_allocation_ratio = min(candidate.quantity / entry_quantity, 1.0)
        entry_fee = total_entry_fee * entry_allocation_ratio
        funding_result = await self._funding_fee_from_account_bills(
            candidate,
            native_facts,
            inst_id=inst_id,
            created_at=settlement_opened_at,
            closed_at=settlement_closed_at,
        )
        if isinstance(funding_result, SettlementFailure):
            return funding_result
        funding_fee, funding_source = funding_result
        snapshot = build_position_settlement_snapshot(
            close_fill_pnl=close_gross_pnl,
            entry_fee=entry_fee,
            close_fee=close_fee,
            funding_fee=funding_fee,
            status="reconciled",
            source=VERIFIED_EXECUTION_PAIR_HISTORY_SOURCE,
            synced_at=now,
            raw={
                "formula": SETTLEMENT_FORMULA,
                "authority": "okx_verified_execution_pair_plus_account_bills",
                "entry_order_ids": entry_ids,
                "close_order_ids": close_ids,
                "order_fee_sources": order_fee_sources,
                "funding_fee_source": funding_source,
                "contract_spec": contract_spec,
            },
        )
        entry_contracts = (
            sum(_order_contracts(order) for order in entry_orders) * entry_allocation_ratio
        )
        close_contracts = sum(_order_contracts(order) for order in close_orders)
        close_price = close_notional / close_quantity if close_quantity > 0 else 0.0
        ct_val = _safe_float(contract_spec.get("ctVal"), 0.0)
        ct_mult = _safe_float(contract_spec.get("ctMult"), 0.0)
        entry_notional = entry_contracts * ct_val * ct_mult * candidate.entry_price
        pnl_ratio = (
            snapshot.realized_pnl / max(entry_notional / max(candidate.leverage, 1.0), 1e-12)
            if entry_notional > 0
            else None
        )
        row = {
            "instId": inst_id,
            "posId": candidate.okx_pos_id,
            "posSide": "net",
            "direction": candidate.side,
            "openSide": "buy" if candidate.side == "long" else "sell",
            "type": "2",
            "cTime": str(int(settlement_opened_at.timestamp() * 1000)),
            "uTime": str(int(settlement_closed_at.timestamp() * 1000)),
            "openAvgPx": str(candidate.entry_price),
            "closeAvgPx": str(close_price),
            "openMaxPos": str(entry_contracts),
            "closeTotalPos": str(close_contracts),
            "lever": str(candidate.leverage),
            "realizedPnl": str(snapshot.realized_pnl),
            "pnl": str(snapshot.close_fill_pnl),
            "fee": str(-(snapshot.entry_fee + snapshot.close_fee)),
            "fundingFee": str(snapshot.funding_fee),
            "pnlRatio": str(pnl_ratio) if pnl_ratio is not None else "",
            "_bb_contract_spec": contract_spec,
            "_bb_source_authority": "okx_verified_execution_pair_plus_account_bills",
            "_bb_pnl_source": VERIFIED_EXECUTION_PAIR_HISTORY_SOURCE,
            "_bb_fee_source": "+".join(sorted(set(order_fee_sources.values()))),
            "_bb_funding_fee_source": funding_source,
        }
        return SettlementSuccess(
            row=row,
            snapshot=snapshot,
            match_reason="verified_okx_execution_pair",
            fee_source=row["_bb_fee_source"],
            funding_fee_source=funding_source,
            history_source=VERIFIED_EXECUTION_PAIR_HISTORY_SOURCE,
        )

    async def _success_from_position_history_row(
        self,
        candidate: SettlementCandidate,
        row: dict[str, Any],
        native_facts: OkxNativeFactsClient,
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
                native_facts,
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
        native_facts: OkxNativeFactsClient,
        *,
        inst_id: str,
        created_at: datetime,
        closed_at: datetime,
    ) -> tuple[float, str] | SettlementFailure:
        since = created_at - timedelta(hours=1)
        try:
            bills = await asyncio.wait_for(
                native_facts.fetch_account_bills(
                    inst_ids=[inst_id],
                    since=since,
                    limit=100,
                    max_pages=3,
                    funding_only=True,
                    strict=True,
                ),
                timeout=max(1.0, min(self.timeout_seconds * 0.35, 3.0)),
            )
        except Exception as exc:
            return SettlementFailure(
                code="funding_fee_api_error",
                message=safe_error_text(exc, limit=240),
                context={
                    "position_id": candidate.position_id,
                    "inst_id": inst_id,
                    "api": "privateGetAccountBills",
                    "since": since.isoformat(),
                },
            )
        funding_fee = _sum_matching_funding_bills(
            bills,
            inst_id=inst_id,
            side=candidate.side,
            opened_at=created_at,
            closed_at=closed_at,
        )
        return funding_fee, "okx_account_bills.funding_only"

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
            history = await upsert_okx_position_history_row(
                session,
                success.row,
                mode=self.mode,
                source=success.history_source,
                entry_order_ids=[candidate.entry_exchange_order_id],
                close_order_ids=[candidate.close_exchange_order_id],
                position_ids=[candidate.position_id],
                match_status=success.match_reason,
                synced_at=now,
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
    ) -> None:
        next_retry_at = now + timedelta(seconds=self.retry_seconds)
        async with self.session_context_factory() as session:
            position = await session.get(Position, candidate.position_id)
            if position is None or bool(position.is_open):
                return
            raw = getattr(position, "settlement_raw", None)
            raw = raw if isinstance(raw, dict) else {}
            if _has_superseded_position_metadata(position, raw):
                _restore_superseded_position_status(position, raw, now=now)
                await session.flush()
                return
            if is_final_settlement_status(getattr(position, "settlement_status", None)):
                return
            attempts = _safe_int(raw.get("settlement_attempt_count"), 0) + 1
            position.settlement_status = SETTLEMENT_STATUS_EXCEPTION
            position.settlement_source = "okx_position_history_settlement"
            position.settlement_synced_at = now
            position.settlement_raw = {
                **raw,
                "status": SETTLEMENT_STATUS_EXCEPTION,
                "source": "okx_position_history_settlement",
                "formula": SETTLEMENT_FORMULA,
                "funding_fee_status": "unknown_until_official_settlement",
                "last_error_code": failure.code,
                "last_error_message": failure.message,
                "last_error_context": failure.context,
                "last_settlement_attempt_at": now.isoformat(),
                "next_settlement_retry_at": next_retry_at.isoformat(),
                "settlement_attempt_count": attempts,
                "retry_policy": f"retry every {self.retry_seconds:g}s until OKX official settlement is available",
            }
            position.updated_at = now
            await session.flush()


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


def _verified_entry_contract_spec(
    decision: AIDecision | None,
    *,
    inst_id: str,
) -> dict[str, Any]:
    raw = _safe_dict(getattr(decision, "raw_llm_response", None))
    pre_order = _safe_dict(raw.get("pre_order_execution_facts"))
    spec = _safe_dict(pre_order.get("contract_spec"))
    if (
        str(spec.get("instId") or "").strip().upper() != inst_id
        or str(spec.get("source") or "").strip() != "okx_public_instruments"
        or _safe_float(spec.get("ctVal"), 0.0) <= 0
        or _safe_float(spec.get("ctMult"), 0.0) <= 0
        or _safe_float(spec.get("lotSz"), 0.0) <= 0
    ):
        return {}
    return dict(spec)


def _order_base_quantity(order: Order) -> float:
    raw = _safe_dict(getattr(order, "okx_raw_fills", None))
    return abs(
        _safe_float(
            raw.get("base_quantity")
            or raw.get("filled_base_quantity")
            or getattr(order, "quantity", None),
            0.0,
        )
    )


def _order_contracts(order: Order) -> float:
    raw = _safe_dict(getattr(order, "okx_raw_fills", None))
    return abs(
        _safe_float(
            raw.get("contracts")
            or raw.get("filled_contracts")
            or getattr(order, "okx_fill_contracts", None),
            0.0,
        )
    )


def _has_matching_current_position(
    rows: list[dict[str, Any]],
    *,
    inst_id: str,
    okx_pos_id: str,
) -> bool:
    target_inst_id = str(inst_id or "").strip().upper()
    target_pos_id = str(okx_pos_id or "").strip()
    for row in rows:
        if not isinstance(row, dict):
            continue
        info = _safe_dict(row.get("info"))
        row_inst_id = str(
            info.get("instId")
            or row.get("instId")
            or okx_inst_id_from_symbol(row.get("symbol"))
            or ""
        ).strip().upper()
        if row_inst_id != target_inst_id:
            continue
        row_pos_id = str(
            info.get("posId") or row.get("posId") or row.get("position_id") or ""
        ).strip()
        if not row_pos_id or row_pos_id == target_pos_id:
            return True
    return False


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
    bills: list[OkxNativeAccountBill],
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
        bill_time = _aware_utc(getattr(bill, "timestamp", None))
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
