"""Writable OKX-native order/fill fact sync.

For OKX-backed paper/live modes, local order rows are only a cache of exchange
facts.  This service starts at the Phase 3 clean-order boundary and updates
local rows from OKX native fills (`instId`, `ordId`, `tradeId`, `fillSz`,
`fillPx`, `fee`, `fillPnl`, `ts`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isclose
from typing import Any

import structlog
from sqlalchemy import and_, or_, select, text

from config.settings import settings
from core.safe_output import safe_error_text
from core.symbols import (
    normalize_trading_symbol,
    okx_inst_id_from_payload,
    okx_inst_id_from_symbol,
    symbol_from_okx_inst_id,
)
from db.session import get_engine, get_session_ctx
from executor.okx_executor import OKXExecutor
from models.decision import AIDecision
from models.trade import Order
from services.exchange_exit_decision_lineage import (
    ExitDecisionLineageAmbiguous,
    recover_exit_decision_lineage_from_order_fact,
)
from services.normal_paper_trade import (
    normal_paper_decision_id_from_client_order_id,
    normal_paper_order_identity_reasons,
)
from services.okx_execution_slippage import (
    OKX_FILL_MARK_SLIPPAGE_VERSION,
    build_okx_fill_mark_slippage,
)
from services.okx_native_facts import (
    OkxNativeFactsClient,
    OkxNativeFillGroup,
    build_okx_protection_execution_lifecycle,
)
from services.okx_protection_position_recovery import (
    confirmed_protection_lifecycle,
    recover_protection_position_lifecycles,
)
from services.paper_training import (
    paper_training_contract_reasons,
    paper_training_decision_id_from_client_order_id,
)
from services.phase3_boundary import PHASE3_CLEAN_START_LOCAL

logger = structlog.get_logger(__name__)

PHASE3_DEFAULT_ORDER_SYNC_START = PHASE3_CLEAN_START_LOCAL
DEFAULT_LOOKBACK_HOURS = 24
DEFAULT_LIMIT = 500
DEFAULT_TIMEOUT_SECONDS = 8.0
ORDER_FACT_SYNC_HARD_DEADLINE_GRACE_SECONDS = 1.0
ORDER_FACT_SYNC_MAX_PERSISTENCE_RESERVE_SECONDS = 1.5
ORDER_FACT_SYNC_DB_LOCK_TIMEOUT_MILLISECONDS = 1500
ORDER_FACT_SYNC_DB_STATEMENT_TIMEOUT_MILLISECONDS = 10000
DEFAULT_TARGET_FILL_ORDER_QUERIES_PER_SYNC = 4
DEFAULT_MAX_ORDER_GAP_QUERIES = 4
ACCOUNT_HISTORY_MAX_PAGES = 5
ACCOUNT_HISTORY_OVERLAP_HOURS = 6
OKX_SYNC_CONFIRMED = "okx_confirmed"
OKX_SYNC_UNVERIFIED = "okx_unverified"
OKX_SYNC_OKX_ONLY = "okx_only_backfilled"
OKX_SYNC_NO_FILL_REJECTED = "okx_no_fill_rejected"
OKX_SYNC_ORDER_ONLY = "okx_order_only"
OKX_SYNC_EXECUTION_RESULT_CONFIRMED = "okx_execution_result_confirmed"
OKX_SYNC_ORDER_DETAIL_CONFIRMED = "okx_order_detail_confirmed"
OKX_SYNC_NATIVE_CLOSE_BACKFILL_PENDING = "okx_native_full_close_pending_backfill"
NATIVE_FULL_CLOSE_BACKFILL_WINDOW_SECONDS = 20 * 60
ORDER_FACT_SYNC_ADVISORY_LOCK_BASE = 0x42424F5244455200
AUTHORITATIVE_FILL_ROW_FIELDS = (
    "ordId",
    "instId",
    "tradeId",
    "billId",
    "clOrdId",
    "side",
    "posSide",
    "fillSz",
    "fillPx",
    "fillMarkPx",
    "fee",
    "feeCcy",
    "fillPnl",
    "ts",
    "fillTime",
)


@dataclass(frozen=True, slots=True)
class OkxOrderFactSyncSummary:
    status: str
    mode: str
    source: str
    phase3_order_sync_start: datetime
    checked_at: datetime
    okx_pull_available: bool
    local_checked: int = 0
    confirmed_count: int = 0
    unverified_count: int = 0
    backfilled_count: int = 0
    order_history_backfilled_count: int = 0
    contract_size_deferred_count: int = 0
    protection_execution_count: int = 0
    position_lifecycle_recovered_count: int = 0
    exit_lineage_recovered_count: int = 0
    protection_execution_error: str | None = None
    completed_stages: tuple[str, ...] = ()
    deferred_stages: tuple[str, ...] = ()
    stage_errors: tuple[str, ...] = ()
    skipped_old_count: int = 0
    error: str | None = None
    samples: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.mode,
            "source": self.source,
            "phase3_order_sync_start": self.phase3_order_sync_start.astimezone(UTC).isoformat(),
            "phase3_order_sync_start_local": self.phase3_order_sync_start.astimezone(
                PHASE3_DEFAULT_ORDER_SYNC_START.tzinfo
            ).isoformat(),
            "checked_at": self.checked_at.astimezone(UTC).isoformat(),
            "okx_pull_available": self.okx_pull_available,
            "local_checked": self.local_checked,
            "confirmed_count": self.confirmed_count,
            "unverified_count": self.unverified_count,
            "backfilled_count": self.backfilled_count,
            "order_history_backfilled_count": self.order_history_backfilled_count,
            "contract_size_deferred_count": self.contract_size_deferred_count,
            "protection_execution_count": self.protection_execution_count,
            "position_lifecycle_recovered_count": self.position_lifecycle_recovered_count,
            "exit_lineage_recovered_count": self.exit_lineage_recovered_count,
            "protection_execution_error": self.protection_execution_error,
            "completed_stages": list(self.completed_stages),
            "deferred_stages": list(self.deferred_stages),
            "stage_errors": list(self.stage_errors),
            "skipped_old_count": self.skipped_old_count,
            "error": self.error,
            "samples": list(self.samples),
        }


def _build_contract_size_catalog(
    *,
    public_sizes: dict[str, float],
) -> dict[str, float]:
    """Keep only direct OKX public-instruments contract specifications."""

    return {
        str(inst_id or "").strip().upper(): float(size)
        for inst_id, size in dict(public_sizes or {}).items()
        if str(inst_id or "").strip() and _safe_float(size, 0.0) > 0
    }


def _authoritative_fill_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in AUTHORITATIVE_FILL_ROW_FIELDS}


def _authoritative_pull_slippage_fact(
    *,
    fill: OkxNativeFillGroup,
    contract_size: float,
) -> dict[str, Any]:
    fact = build_okx_fill_mark_slippage(
        order_id=fill.order_id,
        inst_id=fill.inst_id,
        side=fill.side,
        contracts=fill.contracts,
        average_price=fill.avg_price,
        contract_size=contract_size,
        rows=fill.rows,
    )
    fact["recovery_terminal"] = fact.get("complete") is not True
    fact["recovery_source"] = "okx_fills_history_current_pull"
    return fact


class OkxOrderFactSyncService:
    """Synchronize local orders from OKX native fill facts."""

    def __init__(
        self,
        *,
        mode: str = "paper",
        lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
        limit: int = DEFAULT_LIMIT,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        executor_factory: Any | None = None,
        phase3_order_sync_start: datetime | None = PHASE3_DEFAULT_ORDER_SYNC_START,
        priority_only: bool = False,
        recovery_order_ids: Iterable[Any] | None = None,
    ) -> None:
        self.mode = "live" if str(mode or "").lower() == "live" else "paper"
        self.lookback_hours = max(int(lookback_hours or DEFAULT_LOOKBACK_HOURS), 1)
        self.limit = max(1, min(int(limit or DEFAULT_LIMIT), 2000))
        self.timeout_seconds = max(float(timeout_seconds or DEFAULT_TIMEOUT_SECONDS), 0.5)
        self.executor_factory = executor_factory or OKXExecutor
        self.priority_only = bool(priority_only)
        recovery_values = (
            (recovery_order_ids,)
            if isinstance(recovery_order_ids, (str, bytes))
            else tuple(recovery_order_ids or ())
        )
        self.recovery_order_ids = tuple(
            dict.fromkeys(
                token
                for value in recovery_values
                for token in _split_exchange_order_ids(value)
            )
        )[:DEFAULT_TARGET_FILL_ORDER_QUERIES_PER_SYNC]
        self.phase3_order_sync_start = _aware_utc(
            phase3_order_sync_start or PHASE3_DEFAULT_ORDER_SYNC_START
        )

    async def sync(self) -> dict[str, Any]:
        operation = self._sync_single_writer
        if str(settings.database_url or "").startswith("postgresql"):
            operation = self._sync_with_postgres_writer_lock
        return await self._sync_with_hard_deadline(operation)

    async def _sync_with_postgres_writer_lock(self) -> dict[str, Any]:
        lock_key = ORDER_FACT_SYNC_ADVISORY_LOCK_BASE + (1 if self.mode == "live" else 0)
        engine = await get_engine()
        async with engine.connect() as lock_connection:
            acquired = bool(
                (
                    await lock_connection.execute(
                        text("SELECT pg_try_advisory_lock(:lock_key)"),
                        {"lock_key": lock_key},
                    )
                ).scalar()
            )
            if not acquired:
                return OkxOrderFactSyncSummary(
                    status="deferred",
                    mode=self.mode,
                    source="okx_native_orders_and_fills",
                    phase3_order_sync_start=self.phase3_order_sync_start,
                    checked_at=datetime.now(UTC),
                    okx_pull_available=True,
                    deferred_stages=("single_writer_lock",),
                    samples=(
                        {
                            "kind": "order_fact_sync_writer_busy",
                            "mode": self.mode,
                        },
                    ),
                ).as_dict()
            # Keep this AsyncConnection checked out for the full lock lifetime.
            # Committing an AsyncSession can return its physical connection to
            # the pool, making a later pg_advisory_unlock run on another backend.
            await lock_connection.commit()
            try:
                return await self._sync_single_writer()
            finally:
                try:
                    await lock_connection.execute(
                        text("SELECT pg_advisory_unlock(:lock_key)"),
                        {"lock_key": lock_key},
                    )
                except BaseException:
                    # Closing the physical backend is the only reliable fallback
                    # when cancellation or a broken connection prevents unlock.
                    await lock_connection.invalidate()
                    raise

    async def _sync_with_hard_deadline(
        self,
        operation: Any | None = None,
    ) -> dict[str, Any]:
        hard_deadline_seconds = (
            self.timeout_seconds + ORDER_FACT_SYNC_HARD_DEADLINE_GRACE_SECONDS
        )
        try:
            return await asyncio.wait_for(
                (operation or self._sync_single_writer)(),
                timeout=hard_deadline_seconds,
            )
        except TimeoutError:
            logger.warning(
                "OKX order fact sync hard deadline exceeded",
                mode=self.mode,
                hard_deadline_seconds=round(hard_deadline_seconds, 3),
            )
            return OkxOrderFactSyncSummary(
                status="deferred",
                mode=self.mode,
                source="okx_native_orders_and_fills",
                phase3_order_sync_start=self.phase3_order_sync_start,
                checked_at=datetime.now(UTC),
                okx_pull_available=False,
                deferred_stages=("hard_deadline",),
                error="order_fact_sync_hard_deadline_exceeded",
                samples=(
                    {
                        "kind": "order_fact_sync_hard_deadline",
                        "mode": self.mode,
                        "hard_deadline_seconds": round(hard_deadline_seconds, 3),
                    },
                ),
            ).as_dict()
        except Exception as exc:
            timeout_stage = _database_timeout_stage(exc)
            if timeout_stage is None:
                raise
            error = safe_error_text(exc, limit=180)
            logger.warning(
                "OKX order fact sync database operation deferred",
                mode=self.mode,
                stage=timeout_stage,
                error=error,
            )
            return OkxOrderFactSyncSummary(
                status="deferred",
                mode=self.mode,
                source="okx_native_orders_and_fills",
                phase3_order_sync_start=self.phase3_order_sync_start,
                checked_at=datetime.now(UTC),
                okx_pull_available=True,
                deferred_stages=(timeout_stage,),
                error=f"order_fact_sync_{timeout_stage}",
                samples=(
                    {
                        "kind": "order_fact_sync_database_timeout",
                        "mode": self.mode,
                        "stage": timeout_stage,
                    },
                ),
            ).as_dict()

    async def _sync_single_writer(self) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        since = self._effective_since(started_at)
        since_naive = _db_naive_since(since)
        local_orders = await self._load_local_orders(since_naive)
        submit_recovery_orders = [
            order
            for order in local_orders
            if _order_is_rejected_without_exchange_fill(order)
        ]
        submit_recovery_decisions = await self._load_decisions_for_orders(
            submit_recovery_orders
        )
        submit_recovery_target_order_ids = list(
            dict.fromkeys(
                exchange_order_id
                for order in submit_recovery_orders
                for exchange_order_id in _decision_completed_submit_exchange_order_ids(
                    submit_recovery_decisions.get(
                        int(getattr(order, "decision_id", 0) or 0)
                    )
                )
            )
        )
        external_refresh_orders = [
            order for order in local_orders if _order_needs_okx_pull(order)
        ]
        target_order_ids = {
            token
            for order in external_refresh_orders
            for token in _split_exchange_order_ids(getattr(order, "exchange_order_id", None))
        } | set(submit_recovery_target_order_ids) | set(self.recovery_order_ids)
        priority_target_order_ids = tuple(
            dict.fromkeys(
                [
                    *self.recovery_order_ids,
                    *submit_recovery_target_order_ids,
                    *_prioritized_exchange_order_ids(
                        external_refresh_orders,
                        limit=DEFAULT_TARGET_FILL_ORDER_QUERIES_PER_SYNC,
                    ),
                ]
            )
        )[:DEFAULT_TARGET_FILL_ORDER_QUERIES_PER_SYNC]
        urgent_target_order_ids = {
            token
            for order in external_refresh_orders
            if not _order_has_authoritative_stored_okx_fill_fact(order)
            for token in _split_exchange_order_ids(getattr(order, "exchange_order_id", None))
        } | set(submit_recovery_target_order_ids) | set(self.recovery_order_ids)
        if self.priority_only:
            priority_target_order_ids = tuple(
                order_id
                for order_id in priority_target_order_ids
                if order_id in urgent_target_order_ids
            )
            if not priority_target_order_ids:
                return OkxOrderFactSyncSummary(
                    status="ok",
                    mode=self.mode,
                    source="okx_native_order_priority_sync",
                    phase3_order_sync_start=since,
                    checked_at=datetime.now(UTC),
                    okx_pull_available=True,
                    completed_stages=("priority_queue_idle",),
                ).as_dict()
        order_target_inst_ids = {
            inst_id
            for order in [*external_refresh_orders, *submit_recovery_orders]
            if (inst_id := _order_inst_id(order))
        }
        stored_fact_inst_ids = {
            inst_id
            for order in local_orders
            if _order_has_contract_sized_execution_fact(order)
            if (inst_id := _order_inst_id(order))
        }
        stored_slippage_refresh_count = 0
        stored_slippage_refresh_samples: list[dict[str, Any]] = []
        executor = self.executor_factory(mode=self.mode, load_markets_on_initialize=False)
        persistence_reserve = min(
            ORDER_FACT_SYNC_MAX_PERSISTENCE_RESERVE_SECONDS,
            max(self.timeout_seconds * 0.25, 0.2),
        )
        pull_budget_seconds = max(self.timeout_seconds - persistence_reserve, 0.5)
        deadline = asyncio.get_running_loop().time() + pull_budget_seconds
        completed_stages: list[str] = []
        deferred_stages: list[str] = []
        stage_errors: list[str] = []
        initialized = False
        okx_pull_available = False
        pull_error: str | None = None
        fills: list[OkxNativeFillGroup] = []
        order_rows: list[dict[str, Any]] = []
        protection_algo_rows: list[dict[str, Any]] = []
        protection_execution_error: str | None = None
        contract_sizes: dict[str, float] = {}
        priority_confirmed_count = 0
        priority_unverified_count = 0
        priority_skipped_old_count = 0
        priority_contract_size_deferred_count = 0
        priority_backfilled_count = 0
        priority_local_checked = 0
        priority_samples: list[dict[str, Any]] = []
        exit_lineage_recovered_count = 0
        account_fills_complete = False
        account_orders_complete = False
        target_fill_order_ids: set[str] = set()

        async def run_stage(
            stage: str,
            operation: Any,
            *,
            cap_seconds: float,
        ) -> tuple[Any, bool]:
            timeout = _remaining_stage_timeout(deadline, cap_seconds)
            if timeout < 0.05:
                deferred_stages.append(stage)
                return None, False
            try:
                result = await _bounded(operation(), timeout)
            except TimeoutError:
                deferred_stages.append(stage)
                logger.info(
                    "OKX order fact sync stage deferred after pull budget timeout",
                    mode=self.mode,
                    stage=stage,
                    timeout_seconds=round(timeout, 3),
                )
                return None, False
            except Exception as exc:
                if _is_retryable_okx_private_error(exc):
                    deferred_stages.append(stage)
                    logger.info(
                        "OKX order fact sync stage deferred after transient private API response",
                        mode=self.mode,
                        stage=stage,
                        error=safe_error_text(exc, limit=180),
                    )
                    return None, False
                error = safe_error_text(exc, limit=180)
                stage_errors.append(f"{stage}: {error}")
                logger.warning(
                    "OKX order fact sync stage failed",
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
            if not initialized:
                pull_error = "OKX executor initialization did not finish inside the pull budget"
                return await self._sync_from_stored_facts(
                    since=since,
                    since_naive=since_naive,
                    started_at=started_at,
                    completed_stages=completed_stages,
                    deferred_stages=deferred_stages,
                    stage_errors=stage_errors,
                    pull_error=pull_error,
                    initial_confirmed_count=stored_slippage_refresh_count,
                    initial_samples=stored_slippage_refresh_samples,
                )

            okx_pull_available = True
            native_facts = OkxNativeFactsClient(executor)
            account_since = _account_history_since(
                since,
                local_orders,
                overlap_hours=max(self.lookback_hours, ACCOUNT_HISTORY_OVERLAP_HOURS),
            )
            if priority_target_order_ids:
                target_fills, target_fills_complete = await run_stage(
                    "fills_history_targeted",
                    lambda: native_facts.fetch_fill_groups(
                        order_ids=priority_target_order_ids,
                        since=since,
                        limit=100,
                        max_pages=1,
                        target_orders_only=True,
                        target_order_query_limit=DEFAULT_TARGET_FILL_ORDER_QUERIES_PER_SYNC,
                        include_historical=False,
                        strict=True,
                    ),
                    cap_seconds=2.0,
                )
                fills = list(target_fills or [])
                if target_fills_complete:
                    target_fill_order_ids.update(priority_target_order_ids)
                priority_contract_sizes, _ = await run_stage(
                    "contract_sizes_priority",
                    lambda: native_facts.fetch_contract_sizes(
                        inst_ids=(
                            order_target_inst_ids
                            | {fill.inst_id for fill in fills if fill.inst_id}
                        ),
                    ),
                    cap_seconds=1.0,
                )
                contract_sizes.update(dict(priority_contract_sizes or {}))
                targeted_fill_ids = {fill.order_id for fill in fills if fill.order_id}
                recovery_targeted_fill_ids = (
                    set(self.recovery_order_ids) & targeted_fill_ids
                )
                if recovery_targeted_fill_ids:
                    (
                        priority_backfilled_count,
                        recovery_contract_size_deferred_count,
                        recovery_samples,
                    ) = await self._persist_recovery_fill_facts(
                        since=since,
                        target_order_ids=recovery_targeted_fill_ids,
                        fills=fills,
                        contract_sizes=contract_sizes,
                    )
                    priority_contract_size_deferred_count += (
                        recovery_contract_size_deferred_count
                    )
                    priority_samples.extend(recovery_samples)
                    completed_stages.append("recovery_order_facts_persisted")
                if urgent_target_order_ids & targeted_fill_ids:
                    (
                        priority_local_checked,
                        priority_confirmed_count,
                        priority_unverified_count,
                        priority_skipped_old_count,
                        local_priority_contract_size_deferred_count,
                        priority_exit_lineage_recovered_count,
                        priority_exit_lineage_errors,
                        local_priority_samples,
                    ) = await self._persist_priority_fill_facts(
                        since=since,
                        since_naive=since_naive,
                        target_order_ids=urgent_target_order_ids & targeted_fill_ids,
                        fills=fills,
                        contract_sizes=contract_sizes,
                    )
                    priority_contract_size_deferred_count += (
                        local_priority_contract_size_deferred_count
                    )
                    priority_samples.extend(local_priority_samples)
                    exit_lineage_recovered_count += priority_exit_lineage_recovered_count
                    stage_errors.extend(priority_exit_lineage_errors)
                    completed_stages.append("priority_order_facts_persisted")
                    if priority_exit_lineage_recovered_count:
                        completed_stages.append("exit_decision_lineage_auto_recovered")
                    if (
                        not self.recovery_order_ids
                        and urgent_target_order_ids.issubset(targeted_fill_ids)
                    ):
                        status = (
                            "warning"
                            if priority_unverified_count or priority_exit_lineage_errors
                            else "deferred"
                            if priority_contract_size_deferred_count
                            else "ok"
                        )
                        return OkxOrderFactSyncSummary(
                            status=status,
                            mode=self.mode,
                            source="okx_native_orders_and_fills_priority_fast_lane",
                            phase3_order_sync_start=since,
                            checked_at=datetime.now(UTC),
                            okx_pull_available=True,
                            local_checked=priority_local_checked,
                            confirmed_count=(
                                stored_slippage_refresh_count + priority_confirmed_count
                            ),
                            unverified_count=priority_unverified_count,
                            backfilled_count=priority_backfilled_count,
                            contract_size_deferred_count=(
                                priority_contract_size_deferred_count
                            ),
                            exit_lineage_recovered_count=exit_lineage_recovered_count,
                            completed_stages=tuple(completed_stages),
                            deferred_stages=("account_history_after_priority_fast_lane",),
                            skipped_old_count=priority_skipped_old_count,
                            samples=tuple(
                                [
                                    *stored_slippage_refresh_samples,
                                    *priority_samples,
                                ][:8]
                            ),
                        ).as_dict()

            if self.priority_only:
                missing_priority_ids = sorted(
                    urgent_target_order_ids
                    - {fill.order_id for fill in fills if fill.order_id}
                )
                return OkxOrderFactSyncSummary(
                    status=(
                        "warning"
                        if priority_unverified_count
                        else "deferred"
                    ),
                    mode=self.mode,
                    source="okx_native_order_priority_sync",
                    phase3_order_sync_start=since,
                    checked_at=datetime.now(UTC),
                    okx_pull_available=True,
                    local_checked=priority_local_checked,
                    confirmed_count=priority_confirmed_count,
                    unverified_count=priority_unverified_count,
                    backfilled_count=priority_backfilled_count,
                    contract_size_deferred_count=priority_contract_size_deferred_count,
                    exit_lineage_recovered_count=exit_lineage_recovered_count,
                    completed_stages=tuple(completed_stages),
                    deferred_stages=("priority_fill_not_yet_observed",),
                    stage_errors=tuple(stage_errors),
                    skipped_old_count=priority_skipped_old_count,
                    samples=tuple(
                        [
                            *priority_samples,
                            {
                                "kind": "priority_fill_not_yet_observed",
                                "mode": self.mode,
                                "exchange_order_ids": missing_priority_ids[:4],
                            },
                        ][:8]
                    ),
                ).as_dict()

            (
                stored_slippage_refresh_count,
                stored_slippage_refresh_samples,
            ) = await self._refresh_stored_slippage_from_rows()
            if stored_slippage_refresh_count:
                completed_stages.append("stored_slippage_contract_upgrade")

            account_wide_fills, account_fills_complete = await run_stage(
                "fills_history_account",
                lambda: native_facts.fetch_fill_groups(
                    inst_ids=order_target_inst_ids,
                    since=account_since,
                    limit=100,
                    max_pages=ACCOUNT_HISTORY_MAX_PAGES,
                    account_wide_only=True,
                    account_wide_fallback=False,
                    include_historical=False,
                    strict=True,
                ),
                cap_seconds=3.0,
            )
            fills = _dedupe_fills_by_order_id([*fills, *(account_wide_fills or [])])
            seen_fill_order_ids = {fill.order_id for fill in fills if fill.order_id}
            missing_priority_ids = tuple(
                order_id
                for order_id in priority_target_order_ids
                if order_id not in seen_fill_order_ids
            )
            if missing_priority_ids:
                target_fills, target_fills_complete = await run_stage(
                    "fills_history_targeted",
                    lambda: native_facts.fetch_fill_groups(
                        order_ids=missing_priority_ids,
                        since=since,
                        limit=100,
                        max_pages=1,
                        target_orders_only=True,
                        target_order_query_limit=DEFAULT_TARGET_FILL_ORDER_QUERIES_PER_SYNC,
                        include_historical=False,
                        strict=True,
                    ),
                    cap_seconds=2.0,
                )
                fills = _dedupe_fills_by_order_id([*fills, *(target_fills or [])])
                if target_fills_complete:
                    target_fill_order_ids.update(missing_priority_ids)

            account_order_rows: list[dict[str, Any]] = []
            account_order_rows, account_orders_complete = await run_stage(
                "orders_history_account",
                lambda: native_facts.fetch_order_history_rows(
                    since=account_since,
                    limit=100,
                    max_pages=ACCOUNT_HISTORY_MAX_PAGES,
                    strict=True,
                ),
                cap_seconds=2.0,
            )
            account_order_rows = list(account_order_rows or [])
            account_order_ids = set(_order_rows_by_id(account_order_rows))
            missing_order_ids = sorted(target_order_ids - account_order_ids)
            if fills:
                missing_order_ids = sorted(
                    set(missing_order_ids) - {fill.order_id for fill in fills if fill.order_id}
                )
            target_order_rows: list[dict[str, Any]] = []
            if missing_order_ids:
                target_order_rows, _ = await run_stage(
                    "orders_history_targeted",
                    lambda: native_facts.fetch_order_history_rows(
                        order_ids=missing_order_ids[:DEFAULT_MAX_ORDER_GAP_QUERIES],
                        since=since,
                        limit=100,
                        max_pages=1,
                        strict=True,
                    ),
                    cap_seconds=1.0,
                )
                target_order_rows = list(target_order_rows or [])
            order_rows = _dedupe_order_rows([*account_order_rows, *target_order_rows])
            protection_algo_rows, protection_complete = await run_stage(
                "protection_algo_history",
                lambda: native_facts.fetch_protection_algo_history_rows(
                    algo_ids=_algo_ids_from_order_rows(order_rows),
                    order_ids={fill.order_id for fill in fills if fill.order_id},
                    inst_ids=(
                        order_target_inst_ids
                        | {fill.inst_id for fill in fills if fill.inst_id}
                    ),
                    since=account_since,
                    limit=100,
                    max_pages=2,
                    strict=True,
                ),
                cap_seconds=1.5,
            )
            protection_algo_rows = list(protection_algo_rows or [])
            if not protection_complete:
                protection_execution_error = next(
                    (
                        value.split(": ", 1)[1]
                        for value in stage_errors
                        if value.startswith("protection_algo_history: ")
                    ),
                    None,
                )
            required_contract_inst_ids = (
                {fill.inst_id for fill in fills if fill.inst_id}
                | _order_rows_inst_ids(order_rows)
                | order_target_inst_ids
                | stored_fact_inst_ids
            )
            missing_contract_inst_ids = required_contract_inst_ids - set(contract_sizes)
            if missing_contract_inst_ids:
                additional_contract_sizes, _ = await run_stage(
                    "contract_sizes",
                    lambda: native_facts.fetch_contract_sizes(
                        inst_ids=missing_contract_inst_ids,
                    ),
                    cap_seconds=1.0,
                )
                contract_sizes.update(dict(additional_contract_sizes or {}))
        except Exception as exc:
            okx_pull_available = False
            pull_error = safe_error_text(exc, limit=180)
            logger.warning(
                "OKX order fact sync pull failed",
                mode=self.mode,
                error=pull_error,
            )
        finally:
            try:
                await asyncio.wait_for(executor.shutdown(), timeout=0.5)
            except Exception as exc:
                logger.debug("OKX order fact sync shutdown failed", error=safe_error_text(exc))

        fills_by_order_id = {fill.order_id: fill for fill in fills}
        order_rows_by_id = _order_rows_by_id(order_rows)
        protection_execution_by_order_id = _protection_execution_by_order_id(
            fills=fills,
            order_rows_by_id=order_rows_by_id,
            algo_rows=protection_algo_rows,
        )
        async with get_session_ctx() as session:
            await _configure_order_fact_write_transaction(session)
            writable_orders = await self._load_writable_refresh_orders(
                session,
                since_naive,
                authoritative_fill_order_ids=set(fills_by_order_id),
            )
            stored_repair_orders = (
                writable_orders
                if okx_pull_available
                else [
                    order
                    for order in writable_orders
                    if _order_needs_local_stored_fact_recovery(order)
                ]
            )
            decision_ids = {
                int(decision_id)
                for order in (writable_orders if okx_pull_available else stored_repair_orders)
                if (decision_id := getattr(order, "decision_id", None))
            }
            decision_ids.update(
                decision_id
                for row in order_rows
                if (
                    decision_id := _decision_id_from_client_order_id(
                        _order_row_client_order_id(row)
                    )
                )
            )
            decision_ids.update(
                decision_id
                for fill in fills
                if (
                    decision_id := _decision_id_from_client_order_id(
                        _fill_client_order_id(
                            fill,
                            order_rows_by_id.get(str(fill.order_id or "").strip()),
                        )
                    )
                )
            )
            decisions_by_id: dict[int, AIDecision] = {}
            if decision_ids:
                decision_rows = await session.execute(
                    select(AIDecision).where(AIDecision.id.in_(decision_ids))
                )
                decisions_by_id = {
                    int(decision.id): decision for decision in decision_rows.scalars().all()
                }
            confirmed_count = stored_slippage_refresh_count + priority_confirmed_count
            unverified_count = priority_unverified_count
            skipped_old_count = priority_skipped_old_count
            samples: list[dict[str, Any]] = [
                *stored_slippage_refresh_samples,
                *priority_samples,
            ]
            contract_sizes = _build_contract_size_catalog(
                public_sizes=contract_sizes,
            )
            backfilled_count = priority_backfilled_count
            order_history_backfilled_count = 0
            contract_size_deferred_count = priority_contract_size_deferred_count
            if okx_pull_available:
                (
                    local_confirmed_count,
                    unverified_count,
                    skipped_old_count,
                    local_contract_size_deferred_count,
                    local_samples,
                ) = self._apply_local_order_facts(
                    writable_orders,
                    fills=fills,
                    fills_by_order_id=fills_by_order_id,
                    order_rows_by_id=order_rows_by_id,
                    protection_execution_by_order_id=protection_execution_by_order_id,
                    contract_sizes=contract_sizes,
                    decisions_by_id=decisions_by_id,
                    now=datetime.now(UTC),
                    since=since,
                    authoritative_absence_order_ids=(
                        set(target_fill_order_ids)
                        | {
                            exchange_id
                            for order in writable_orders
                            if account_fills_complete
                            and account_orders_complete
                            and (order_time := _order_time(order)) is not None
                            and order_time >= account_since
                            for exchange_id in _split_exchange_order_ids(
                                getattr(order, "exchange_order_id", None)
                            )
                        }
                    ),
                )
                confirmed_count += local_confirmed_count
                samples.extend(local_samples)
                (
                    newly_backfilled_count,
                    order_history_backfilled_count,
                    backfill_contract_size_deferred_count,
                ) = await self._backfill_okx_only_orders(
                    session,
                    fills=fills,
                    order_rows=order_rows,
                    decisions_by_id=decisions_by_id,
                    protection_execution_by_order_id=protection_execution_by_order_id,
                    contract_sizes=contract_sizes,
                    since=since,
                    now=datetime.now(UTC),
                    samples=samples,
                )
                backfilled_count += newly_backfilled_count
                contract_size_deferred_count = (
                    priority_contract_size_deferred_count
                    + local_contract_size_deferred_count
                    + backfill_contract_size_deferred_count
                )
                if contract_size_deferred_count:
                    deferred_stages.append("order_facts_missing_public_contract_size")
            else:
                confirmed_count += self._recover_local_stored_order_facts(
                    stored_repair_orders,
                    decisions_by_id=decisions_by_id,
                    now=datetime.now(UTC),
                    samples=samples,
                )
            (
                exit_lineage_recovered_count,
                exit_lineage_errors,
            ) = await self._recover_exit_decision_lineages(
                session,
                orders=writable_orders,
                fills=fills,
                samples=samples,
            )
            stage_errors.extend(exit_lineage_errors)
            if exit_lineage_recovered_count:
                completed_stages.append("exit_decision_lineage_auto_recovered")
            position_recovery_samples = await _recover_protection_position_links(
                session,
                mode=self.mode,
                orders=writable_orders,
                target_order_ids=set(protection_execution_by_order_id),
            )
            position_lifecycle_recovered_count = len(position_recovery_samples)
            samples.extend(position_recovery_samples)
            if confirmed_count or position_lifecycle_recovered_count:
                await session.flush()

        if not okx_pull_available:
            status = "degraded"
        elif unverified_count or stage_errors:
            status = "warning"
        elif deferred_stages:
            status = "deferred"
        else:
            status = "ok"
        return OkxOrderFactSyncSummary(
            status=status,
            mode=self.mode,
            source="okx_native_orders_and_fills",
            phase3_order_sync_start=since,
            checked_at=datetime.now(UTC),
            okx_pull_available=okx_pull_available,
            local_checked=len(writable_orders) if okx_pull_available else len(stored_repair_orders),
            confirmed_count=confirmed_count,
            unverified_count=unverified_count,
            backfilled_count=backfilled_count,
            order_history_backfilled_count=order_history_backfilled_count,
            contract_size_deferred_count=contract_size_deferred_count,
            protection_execution_count=len(protection_execution_by_order_id),
            position_lifecycle_recovered_count=position_lifecycle_recovered_count,
            exit_lineage_recovered_count=exit_lineage_recovered_count,
            protection_execution_error=protection_execution_error,
            completed_stages=tuple(completed_stages),
            deferred_stages=tuple(dict.fromkeys(deferred_stages)),
            stage_errors=tuple(stage_errors),
            skipped_old_count=skipped_old_count,
            error=pull_error,
            samples=tuple(samples[:8]),
        ).as_dict()

    async def _sync_from_stored_facts(
        self,
        *,
        since: datetime,
        since_naive: datetime,
        started_at: datetime,
        completed_stages: list[str],
        deferred_stages: list[str],
        stage_errors: list[str],
        pull_error: str,
        initial_confirmed_count: int,
        initial_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        async with get_session_ctx() as session:
            await _configure_order_fact_write_transaction(session)
            writable_orders = await self._load_writable_refresh_orders(
                session,
                since_naive,
                authoritative_fill_order_ids=set(),
            )
            stored_repair_orders = [
                order
                for order in writable_orders
                if _order_needs_local_stored_fact_recovery(order)
            ]
            decision_ids = {
                int(order.decision_id)
                for order in stored_repair_orders
                if getattr(order, "decision_id", None)
            }
            decisions_by_id: dict[int, AIDecision] = {}
            if decision_ids:
                rows = await session.execute(
                    select(AIDecision).where(AIDecision.id.in_(sorted(decision_ids)))
                )
                decisions_by_id = {
                    int(decision.id): decision for decision in rows.scalars().all()
                }
            samples: list[dict[str, Any]] = list(initial_samples)
            confirmed_count = (
                initial_confirmed_count
                + self._recover_local_stored_order_facts(
                    stored_repair_orders,
                    decisions_by_id=decisions_by_id,
                    now=datetime.now(UTC),
                    samples=samples,
                )
            )
            exit_lineage_recovered_count, exit_lineage_errors = (
                await self._recover_exit_decision_lineages(
                    session,
                    orders=writable_orders,
                    fills=(),
                    samples=samples,
                )
            )
            stage_errors.extend(exit_lineage_errors)
            position_recovery_samples = await _recover_protection_position_links(
                session,
                mode=self.mode,
                orders=writable_orders,
            )
            position_lifecycle_recovered_count = len(position_recovery_samples)
            samples.extend(position_recovery_samples)
            if confirmed_count or position_lifecycle_recovered_count:
                await session.flush()
        return OkxOrderFactSyncSummary(
            status="degraded",
            mode=self.mode,
            source="okx_native_orders_and_fills",
            phase3_order_sync_start=since,
            checked_at=datetime.now(UTC),
            okx_pull_available=False,
            local_checked=len(stored_repair_orders),
            confirmed_count=confirmed_count,
            position_lifecycle_recovered_count=position_lifecycle_recovered_count,
            exit_lineage_recovered_count=exit_lineage_recovered_count,
            completed_stages=tuple(completed_stages),
            deferred_stages=tuple(dict.fromkeys(deferred_stages)),
            stage_errors=tuple(stage_errors),
            error=pull_error,
            samples=tuple(samples[:8]),
        ).as_dict()

    async def _refresh_stored_slippage_from_rows(
        self,
    ) -> tuple[int, list[dict[str, Any]]]:
        async with get_session_ctx() as session:
            await _configure_order_fact_write_transaction(session)
            orders = await self._load_stored_slippage_refresh_orders(session)
            samples: list[dict[str, Any]] = []
            refreshed_count = 0
            now = datetime.now(UTC)
            for order in orders:
                if not _rebuild_stored_slippage_fact(order, now=now):
                    continue
                refreshed_count += 1
                samples.append(
                    _sample(order, kind="stored_slippage_contract_upgraded")
                )
            if refreshed_count:
                await session.flush()
        return refreshed_count, samples

    async def _recover_exit_decision_lineages(
        self,
        session: Any,
        *,
        orders: Iterable[Order],
        fills: Iterable[OkxNativeFillGroup],
        samples: list[dict[str, Any]],
    ) -> tuple[int, list[str]]:
        """Relink exact close-order facts after execution/reconciliation races."""

        order_list = list(orders)
        orders_by_exchange_id = {
            exchange_order_id: order
            for order in order_list
            for exchange_order_id in _split_exchange_order_ids(
                getattr(order, "exchange_order_id", None)
            )
        }
        fills_by_exchange_id = {
            str(fill.order_id or "").strip(): fill
            for fill in fills
            if str(fill.order_id or "").strip()
        }
        target_order_ids = set(fills_by_exchange_id) & set(orders_by_exchange_id)
        target_order_ids.update(
            exchange_order_id
            for exchange_order_id, order in orders_by_exchange_id.items()
            if _order_has_confirmed_okx_fill_fact(order)
        )
        recovered_count = 0
        errors: list[str] = []
        for close_order_id in sorted(target_order_ids):
            order = orders_by_exchange_id.get(close_order_id)
            if order is None:
                order_result = await session.execute(
                    select(Order)
                    .where(
                        Order.execution_mode == self.mode,
                        Order.exchange_order_id == close_order_id,
                    )
                    .limit(1)
                )
                order = order_result.scalar_one_or_none()
            if order is None:
                continue
            fill = fills_by_exchange_id.get(close_order_id)
            close_fill = fill.as_dict() if fill is not None else None
            try:
                recovered = await recover_exit_decision_lineage_from_order_fact(
                    session,
                    order=order,
                    close_fill=close_fill,
                    now=datetime.now(UTC),
                )
            except ExitDecisionLineageAmbiguous as exc:
                error = safe_error_text(exc, limit=180)
                errors.append(f"exit_decision_lineage: {error}")
                samples.append(
                    {
                        "kind": "exit_decision_lineage_ambiguous",
                        "mode": self.mode,
                        "exchange_order_id": close_order_id,
                        "error": error,
                    }
                )
                continue
            if recovered is None:
                continue
            recovered_count += 1
            samples.append(
                {
                    "kind": "exit_decision_lineage_auto_recovered",
                    "mode": self.mode,
                    "exchange_order_id": close_order_id,
                    "authoritative_decision_id": recovered.get(
                        "authoritative_decision_id"
                    ),
                    "superseded_decision_ids": list(
                        recovered.get("superseded_decision_ids") or []
                    ),
                }
            )
        if recovered_count:
            await session.flush()
        return recovered_count, errors

    def _effective_since(self, now: datetime) -> datetime:
        """Return the Phase 3 clean-order boundary.

        Order facts are an exchange-backed ledger, not a rolling dashboard query.
        Do not move this boundary forward with lookback hours or reset markers,
        otherwise older Phase 3 OKX fills can silently disappear from local facts.
        """

        return _aware_utc(self.phase3_order_sync_start)

    async def _load_local_orders(
        self,
        since_naive: datetime,
    ) -> list[Order]:
        async with get_session_ctx() as session:
            rows = await session.execute(
                select(Order)
                .where(
                    Order.execution_mode == self.mode,
                    or_(Order.created_at >= since_naive, Order.filled_at >= since_naive),
                )
                .order_by(Order.filled_at.desc().nullslast(), Order.created_at.desc())
                .limit(self.limit)
            )
            recent = list(rows.scalars().all())
            stored_slippage_refresh = await self._load_stored_slippage_refresh_orders(
                session,
                for_update=False,
            )
            return _merge_local_order_rows(recent, stored_slippage_refresh)

    async def _load_decisions_for_orders(
        self,
        orders: list[Order],
    ) -> dict[int, AIDecision]:
        decision_ids = {
            int(decision_id)
            for order in orders
            if (decision_id := getattr(order, "decision_id", None))
        }
        if not decision_ids:
            return {}
        async with get_session_ctx() as session:
            rows = await session.execute(
                select(AIDecision).where(AIDecision.id.in_(sorted(decision_ids)))
            )
            return {
                int(decision.id): decision for decision in rows.scalars().all()
            }

    async def _load_stored_slippage_refresh_orders(
        self,
        session: Any,
        *,
        for_update: bool = True,
    ) -> list[Order]:
        slippage_version = Order.okx_raw_fills["execution_slippage"][
            "version"
        ].as_string()
        slippage_complete = Order.okx_raw_fills["execution_slippage"][
            "complete"
        ].as_boolean()
        slippage_recovery_terminal = Order.okx_raw_fills["execution_slippage"][
            "recovery_terminal"
        ].as_boolean()
        statement = (
            select(Order)
            .where(
                Order.execution_mode == self.mode,
                Order.okx_raw_fills["fills_history_confirmed"]
                .as_boolean()
                .is_(True),
                or_(
                    slippage_version.is_(None),
                    slippage_version != OKX_FILL_MARK_SLIPPAGE_VERSION,
                    and_(
                        slippage_complete.is_not(True),
                        slippage_recovery_terminal.is_not(True),
                    ),
                ),
            )
            .order_by(Order.id.desc())
            .limit(max(self.limit * 4, self.limit))
        )
        if for_update:
            statement = statement.with_for_update(skip_locked=True)
        rows = await session.execute(statement)
        return [
            order
            for order in rows.scalars().all()
            if _order_has_authoritative_stored_okx_fill_fact(order)
        ][: self.limit]

    async def _load_writable_refresh_orders(
        self,
        session: Any,
        since_naive: datetime,
        *,
        authoritative_fill_order_ids: set[str] | None = None,
    ) -> list[Order]:
        fill_order_ids = {
            str(order_id or "").strip()
            for order_id in (authoritative_fill_order_ids or set())
            if str(order_id or "").strip()
        }
        rows = await session.execute(
            select(Order)
            .where(
                Order.execution_mode == self.mode,
                or_(Order.created_at >= since_naive, Order.filled_at >= since_naive),
            )
            .order_by(Order.filled_at.desc().nullslast(), Order.created_at.desc())
            .limit(self.limit)
            .with_for_update(skip_locked=True)
        )
        recent = list(rows.scalars().all())
        stored_slippage_refresh = await self._load_stored_slippage_refresh_orders(
            session
        )
        matched_fills: list[Order] = []
        if fill_order_ids:
            matched_fill_rows = await session.execute(
                select(Order).where(
                    Order.execution_mode == self.mode,
                    Order.exchange_order_id.in_(sorted(fill_order_ids)),
                ).with_for_update(skip_locked=True)
            )
            matched_fills = list(matched_fill_rows.scalars().all())
        return [
            order
            for order in _merge_local_order_rows(
                recent,
                matched_fills,
                stored_slippage_refresh,
            )
            if _order_needs_okx_fact_refresh(order)
            or _stored_slippage_fact_needs_refresh(order)
            or confirmed_protection_lifecycle(order) is not None
            or bool(
                _split_exchange_order_ids(getattr(order, "exchange_order_id", None))
                & fill_order_ids
            )
        ]

    async def _persist_priority_fill_facts(
        self,
        *,
        since: datetime,
        since_naive: datetime,
        target_order_ids: set[str],
        fills: list[OkxNativeFillGroup],
        contract_sizes: dict[str, float],
    ) -> tuple[int, int, int, int, int, int, list[str], list[dict[str, Any]]]:
        """Commit targeted native fills before slower account-wide history stages."""

        fills_by_order_id = {fill.order_id: fill for fill in fills if fill.order_id}
        async with get_session_ctx() as session:
            await _configure_order_fact_write_transaction(session)
            writable_orders = await self._load_writable_refresh_orders(
                session,
                since_naive,
                authoritative_fill_order_ids=set(fills_by_order_id),
            )
            decision_ids = {
                int(decision_id)
                for order in writable_orders
                if (decision_id := getattr(order, "decision_id", None))
            }
            decisions_by_id: dict[int, AIDecision] = {}
            if decision_ids:
                decision_rows = await session.execute(
                    select(AIDecision).where(AIDecision.id.in_(decision_ids))
                )
                decisions_by_id = {
                    int(decision.id): decision for decision in decision_rows.scalars().all()
                }
            priority_orders = [
                order
                for order in writable_orders
                if _order_exchange_ids_with_completed_submit(
                    order,
                    decisions_by_id.get(int(getattr(order, "decision_id", 0) or 0)),
                )
                & target_order_ids
            ]
            (
                confirmed_count,
                unverified_count,
                skipped_old_count,
                contract_size_deferred_count,
                samples,
            ) = self._apply_local_order_facts(
                priority_orders,
                fills=fills,
                fills_by_order_id=fills_by_order_id,
                order_rows_by_id={},
                protection_execution_by_order_id={},
                contract_sizes=_build_contract_size_catalog(public_sizes=contract_sizes),
                decisions_by_id=decisions_by_id,
                now=datetime.now(UTC),
                since=since,
                authoritative_absence_order_ids=set(),
            )
            exit_lineage_recovered_count, exit_lineage_errors = (
                await self._recover_exit_decision_lineages(
                    session,
                    orders=priority_orders,
                    fills=fills,
                    samples=samples,
                )
            )
            return (
                len(priority_orders),
                confirmed_count,
                unverified_count,
                skipped_old_count,
                contract_size_deferred_count,
                exit_lineage_recovered_count,
                exit_lineage_errors,
                samples,
            )

    async def _persist_recovery_fill_facts(
        self,
        *,
        since: datetime,
        target_order_ids: set[str],
        fills: list[OkxNativeFillGroup],
        contract_sizes: dict[str, float],
    ) -> tuple[int, int, list[dict[str, Any]]]:
        """Backfill explicitly audited OKX-only fills before slow history pulls."""

        recovery_fills = [
            fill for fill in fills if fill.order_id and fill.order_id in target_order_ids
        ]
        decision_ids = {
            decision_id
            for fill in recovery_fills
            if (
                decision_id := _decision_id_from_client_order_id(
                    _fill_client_order_id(fill, None)
                )
            )
        }
        async with get_session_ctx() as session:
            await _configure_order_fact_write_transaction(session)
            decisions_by_id: dict[int, AIDecision] = {}
            if decision_ids:
                decision_rows = await session.execute(
                    select(AIDecision).where(AIDecision.id.in_(sorted(decision_ids)))
                )
                decisions_by_id = {
                    int(decision.id): decision for decision in decision_rows.scalars().all()
                }
            samples: list[dict[str, Any]] = []
            (
                backfilled_count,
                _order_history_backfilled_count,
                contract_size_deferred_count,
            ) = await self._backfill_okx_only_orders(
                session,
                fills=recovery_fills,
                order_rows=[],
                decisions_by_id=decisions_by_id,
                protection_execution_by_order_id={},
                contract_sizes=_build_contract_size_catalog(public_sizes=contract_sizes),
                since=since,
                now=datetime.now(UTC),
                samples=samples,
            )
            if backfilled_count:
                await session.flush()
            return backfilled_count, contract_size_deferred_count, samples

    def _apply_local_order_facts(
        self,
        orders: list[Order],
        *,
        fills: list[OkxNativeFillGroup],
        fills_by_order_id: dict[str, OkxNativeFillGroup],
        order_rows_by_id: dict[str, dict[str, Any]],
        protection_execution_by_order_id: dict[str, dict[str, Any]],
        contract_sizes: dict[str, float],
        decisions_by_id: dict[int, AIDecision] | None = None,
        now: datetime,
        since: datetime,
        authoritative_absence_order_ids: set[str],
    ) -> tuple[int, int, int, int, list[dict[str, Any]]]:
        confirmed_count = 0
        unverified_count = 0
        skipped_old_count = 0
        contract_size_deferred_count = 0
        samples: list[dict[str, Any]] = []
        for order in orders:
            order_time = _order_time(order)
            if order_time is not None and order_time < since:
                if _stored_slippage_fact_needs_refresh(
                    order
                ) and _repair_stored_fill_contract_size_from_instruments(
                    order,
                    contract_sizes=contract_sizes,
                    now=now,
                ):
                    confirmed_count += 1
                    samples.append(
                        _sample(order, kind="stored_fill_slippage_fact_refreshed")
                    )
                    continue
                skipped_old_count += 1
                continue
            decision = (decisions_by_id or {}).get(
                int(getattr(order, "decision_id", 0) or 0)
            )
            exchange_ids = _order_exchange_ids_with_completed_submit(order, decision)
            fill = next(
                (fills_by_order_id[exchange_id] for exchange_id in exchange_ids if exchange_id in fills_by_order_id),
                None,
            )
            order_row = next(
                (order_rows_by_id[exchange_id] for exchange_id in exchange_ids if exchange_id in order_rows_by_id),
                None,
            )
            if fill is None:
                pending_fill = _matching_native_full_close_pending_fill(
                    order,
                    fills=fills,
                    contract_sizes=contract_sizes,
                )
                if pending_fill is not None:
                    contract_size, contract_size_source = _contract_size_for_fill_with_source(
                        pending_fill,
                        contract_sizes,
                    )
                    if not _is_verified_public_contract_size(
                        contract_size,
                        contract_size_source,
                    ):
                        contract_size_deferred_count += 1
                        samples.append(
                            _sample(order, kind="native_full_close_waiting_public_contract_size")
                        )
                        continue
                    self._apply_fill_to_order(
                        order,
                        pending_fill,
                        now=now,
                        sync_status=OKX_SYNC_CONFIRMED,
                        contract_size=contract_size,
                        contract_size_source=contract_size_source,
                        order_row=order_rows_by_id.get(str(pending_fill.order_id or "").strip()),
                        protection_execution=protection_execution_by_order_id.get(
                            str(pending_fill.order_id or "").strip()
                        ),
                    )
                    confirmed_count += 1
                    samples.append(_sample(order, kind="native_full_close_backfill_confirmed"))
                    continue
                if _order_has_authoritative_stored_okx_fill_fact(order):
                    if _repair_stored_fill_contract_size_from_instruments(
                        order,
                        contract_sizes=contract_sizes,
                        now=now,
                    ):
                        confirmed_count += 1
                        samples.append(_sample(order, kind="local_order_contract_size_repaired"))
                    elif (
                        _safe_float(contract_sizes.get(_order_inst_id(order)), 0.0) > 0
                        and _order_has_verified_public_contract_size(order)
                    ):
                        samples.append(
                            _sample(order, kind="local_order_stored_fill_already_verified")
                        )
                    else:
                        contract_size_deferred_count += 1
                        samples.append(
                            _sample(order, kind="local_order_stored_fill_waiting_contract_size")
                        )
                    continue
                if (
                    _order_has_okx_execution_result_fact(order)
                    or _order_has_okx_order_detail_fact(order)
                ):
                    _repair_execution_contract_size_from_instruments(
                        order,
                        contract_sizes=contract_sizes,
                        now=now,
                    )
                    if _order_has_verified_public_contract_size(order):
                        promoted = _promote_execution_result_to_order_detail(
                            order,
                            now=now,
                        )
                        _apply_execution_result_confirmation_to_order(order, now=now)
                        if promoted:
                            confirmed_count += 1
                        samples.append(
                            _sample(
                                order,
                                kind=(
                                    "local_order_detail_confirmed"
                                    if _order_has_okx_order_detail_fact(order)
                                    else "local_order_execution_result_already_confirmed"
                                ),
                            )
                        )
                    else:
                        contract_size_deferred_count += 1
                        samples.append(
                            _sample(
                                order,
                                kind="local_order_execution_result_waiting_contract_size",
                            )
                        )
                    continue
                decision = (decisions_by_id or {}).get(int(getattr(order, "decision_id", 0) or 0))
                if _recover_okx_execution_result_fact_from_decision(order, decision):
                    _repair_execution_contract_size_from_instruments(
                        order,
                        contract_sizes=contract_sizes,
                        now=now,
                    )
                    if _order_has_verified_public_contract_size(order):
                        _promote_execution_result_to_order_detail(order, now=now)
                        _apply_execution_result_confirmation_to_order(order, now=now)
                        confirmed_count += 1
                        samples.append(
                            _sample(order, kind="local_order_execution_result_recovered")
                        )
                    else:
                        contract_size_deferred_count += 1
                        samples.append(
                            _sample(
                                order,
                                kind="local_order_execution_result_recovery_waiting_contract_size",
                            )
                        )
                    continue
                if _recover_okx_close_fill_fact_from_decision(order, decision):
                    if _repair_stored_fill_contract_size_from_instruments(
                        order,
                        contract_sizes=contract_sizes,
                        now=now,
                    ):
                        _apply_close_fill_confirmation_to_order(order, now=now)
                        confirmed_count += 1
                        samples.append(_sample(order, kind="local_order_close_fill_recovered"))
                    else:
                        contract_size_deferred_count += 1
                        samples.append(
                            _sample(
                                order,
                                kind="local_order_close_fill_recovery_waiting_contract_size",
                            )
                        )
                    continue
                has_authoritative_absence = bool(
                    exchange_ids & authoritative_absence_order_ids
                )
                if (
                    str(getattr(order, "status", "") or "").lower() == "filled"
                    and not has_authoritative_absence
                ):
                    samples.append(_sample(order, kind="local_filled_fact_pull_deferred"))
                elif str(getattr(order, "status", "") or "").lower() == "filled":
                    order.okx_sync_status = OKX_SYNC_UNVERIFIED
                    order.okx_state = _order_row_state(order_row) or "missing_okx_fill"
                    order.okx_synced_at = now
                    _clear_unconfirmed_fill_fact(order)
                    order.okx_last_error = (
                        "OKX orders-history found this order but fills-history did not confirm a fill"
                        if order_row
                        else "OKX orders-history/fills-history did not confirm this local filled order"
                    )
                    if order_row:
                        _apply_order_row_metadata(order, order_row, now=now)
                    unverified_count += 1
                    samples.append(_sample(order, kind="local_filled_unverified"))
                elif _order_is_rejected_without_exchange_fill(order):
                    order.okx_sync_status = OKX_SYNC_NO_FILL_REJECTED
                    order.okx_state = "rejected_no_exchange_fill"
                    order.okx_synced_at = now
                    order.okx_last_error = None
                    if order_row:
                        _apply_order_row_metadata(order, order_row, now=now)
                    samples.append(_sample(order, kind="local_rejected_no_exchange_fill"))
                elif order_row:
                    _apply_order_row_metadata(order, order_row, now=now)
                    order.okx_sync_status = OKX_SYNC_ORDER_ONLY
                    order.okx_last_error = None
                    samples.append(_sample(order, kind="local_order_history_only"))
                continue
            contract_size, contract_size_source = _contract_size_for_fill_with_source(
                fill,
                contract_sizes,
            )
            if not _is_verified_public_contract_size(
                contract_size,
                contract_size_source,
            ):
                contract_size_deferred_count += 1
                samples.append(_sample(order, kind="local_order_fill_waiting_public_contract_size"))
                continue
            if _stored_order_matches_native_fill(
                order,
                fill,
                contract_size=contract_size,
                contract_size_source=contract_size_source,
            ):
                protection_execution = protection_execution_by_order_id.get(fill.order_id)
                needs_slippage_refresh = _stored_slippage_fact_needs_refresh(order)
                needs_protection_refresh = _stored_protection_execution_needs_refresh(
                    order,
                    protection_execution,
                )
                if needs_slippage_refresh or needs_protection_refresh:
                    self._apply_fill_to_order(
                        order,
                        fill,
                        now=now,
                        sync_status=OKX_SYNC_CONFIRMED,
                        contract_size=contract_size,
                        contract_size_source=contract_size_source,
                        order_row=order_row,
                        protection_execution=protection_execution,
                    )
                    confirmed_count += 1
                    samples.append(
                        _sample(
                            order,
                            kind=(
                                "local_order_slippage_fact_refreshed"
                                if needs_slippage_refresh
                                else "local_order_protection_execution_refreshed"
                            ),
                        )
                    )
                continue
            self._apply_fill_to_order(
                order,
                fill,
                now=now,
                sync_status=OKX_SYNC_CONFIRMED,
                contract_size=contract_size,
                contract_size_source=contract_size_source,
                order_row=order_row,
                protection_execution=protection_execution_by_order_id.get(fill.order_id),
            )
            recovered_from_submit_stage = bool(
                decision is not None
                and fill.order_id
                in _decision_completed_submit_exchange_order_ids(decision)
            )
            if recovered_from_submit_stage:
                _apply_completed_submit_fill_recovery_to_decision(
                    decision,
                    fill=fill,
                    contract_size=contract_size,
                    now=now,
                )
            confirmed_count += 1
            samples.append(
                _sample(
                    order,
                    kind=(
                        "local_order_confirmed_from_completed_submit_stage"
                        if recovered_from_submit_stage
                        else "local_order_confirmed"
                    ),
                )
            )
        return (
            confirmed_count,
            unverified_count,
            skipped_old_count,
            contract_size_deferred_count,
            samples,
        )

    def _recover_local_stored_order_facts(
        self,
        orders: list[Order],
        *,
        decisions_by_id: dict[int, AIDecision],
        now: datetime,
        samples: list[dict[str, Any]],
    ) -> int:
        confirmed_count = 0
        for order in orders:
            decision = decisions_by_id.get(int(getattr(order, "decision_id", 0) or 0))
            if _recover_okx_execution_result_fact_from_decision(order, decision):
                _promote_execution_result_to_order_detail(order, now=now)
                _apply_execution_result_confirmation_to_order(order, now=now)
                confirmed_count += 1
                samples.append(_sample(order, kind="local_order_execution_result_recovered"))
                continue
            if _recover_okx_close_fill_fact_from_decision(order, decision):
                _apply_close_fill_confirmation_to_order(order, now=now)
                confirmed_count += 1
                samples.append(_sample(order, kind="local_order_close_fill_recovered"))
                continue
            if (
                _order_has_okx_execution_result_fact(order)
                or _order_has_okx_order_detail_fact(order)
            ):
                _promote_execution_result_to_order_detail(order, now=now)
                _apply_execution_result_confirmation_to_order(order, now=now)
                confirmed_count += 1
                samples.append(_sample(order, kind="local_order_execution_result_confirmed"))
                continue
            if _order_has_authoritative_stored_okx_fill_fact(order):
                _apply_close_fill_confirmation_to_order(order, now=now)
                confirmed_count += 1
                samples.append(_sample(order, kind="local_order_stored_fill_confirmed"))
        return confirmed_count

    async def _backfill_okx_only_orders(
        self,
        session: Any,
        *,
        fills: list[OkxNativeFillGroup],
        order_rows: list[dict[str, Any]],
        decisions_by_id: dict[int, AIDecision],
        protection_execution_by_order_id: dict[str, dict[str, Any]],
        contract_sizes: dict[str, float],
        since: datetime,
        now: datetime,
        samples: list[dict[str, Any]],
    ) -> tuple[int, int, int]:
        backfilled = 0
        order_history_backfilled = 0
        contract_size_deferred_count = 0
        fill_order_ids = {fill.order_id for fill in fills if fill.order_id}
        order_history_ids = {
            order_id
            for row in order_rows
            if (order_id := _order_row_id(row))
        }
        existing_exchange_ids = await _existing_order_ids(
            session,
            self.mode,
            fill_order_ids | order_history_ids,
        )
        order_rows_by_id = _order_rows_by_id(order_rows)
        for fill in fills:
            if fill.order_id in existing_exchange_ids:
                continue
            if fill.timestamp is not None and _aware_utc(fill.timestamp) < since:
                continue
            contract_size, contract_size_source = _contract_size_for_fill_with_source(
                fill,
                contract_sizes,
            )
            if not _is_verified_public_contract_size(
                contract_size,
                contract_size_source,
            ):
                contract_size_deferred_count += 1
                samples.append(_fill_sample(fill, kind="okx_only_fill_waiting_public_contract_size"))
                continue
            order_row = order_rows_by_id.get(fill.order_id)
            decision = _decision_for_order_fact(
                fill=fill,
                order_row=order_row,
                decisions_by_id=decisions_by_id,
            )
            order = Order(
                model_name=(
                    str(getattr(decision, "model_name", "") or "okx_authoritative_sync")
                    if decision is not None
                    else "okx_authoritative_sync"
                ),
                execution_mode=self.mode,
                symbol=fill.symbol,
                side=fill.side,
                order_type=_fill_order_type(fill),
                quantity=_fill_base_quantity(fill, contract_size),
                price=fill.avg_price,
                status="filled",
                fee=fill.fee_abs,
                decision_id=(int(decision.id) if decision is not None else None),
                exchange_order_id=fill.order_id,
                filled_at=fill.timestamp or now,
                created_at=fill.timestamp or now,
            )
            self._apply_fill_to_order(
                order,
                fill,
                now=now,
                sync_status=OKX_SYNC_OKX_ONLY,
                contract_size=contract_size,
                contract_size_source=contract_size_source,
                order_row=order_row,
                protection_execution=protection_execution_by_order_id.get(fill.order_id),
            )
            if decision is not None:
                _apply_exchange_recovery_to_decision(
                    decision,
                    fill=fill,
                    client_order_id=_fill_client_order_id(fill, order_row),
                    now=now,
                )
            session.add(order)
            existing_exchange_ids.add(fill.order_id)
            backfilled += 1
            samples.append(
                _sample(
                    order,
                    kind=(
                        "okx_only_backfilled_with_decision_identity"
                        if decision is not None
                        else "okx_only_backfilled"
                    ),
                )
            )
        for row in order_rows:
            order_id = _order_row_id(row)
            if not order_id or order_id in existing_exchange_ids:
                continue
            order_time = _order_row_time(row)
            if order_time is not None and _aware_utc(order_time) < since:
                continue
            if _order_row_state(row) == "filled":
                continue
            order = _order_from_order_history_row(
                row,
                mode=self.mode,
                now=now,
                contract_size=_contract_size_for_order_row(row, contract_sizes),
            )
            session.add(order)
            existing_exchange_ids.add(order_id)
            order_history_backfilled += 1
            samples.append(_sample(order, kind="okx_order_history_backfilled"))
        return backfilled, order_history_backfilled, contract_size_deferred_count

    @staticmethod
    def _apply_fill_to_order(
        order: Order,
        fill: OkxNativeFillGroup,
        *,
        now: datetime,
        sync_status: str,
        contract_size: float = 0.0,
        contract_size_source: str = "",
        order_row: dict[str, Any] | None = None,
        protection_execution: dict[str, Any] | None = None,
    ) -> None:
        contract_size_source = str(contract_size_source or "").strip()
        if not _is_verified_public_contract_size(contract_size, contract_size_source):
            raise ValueError("OKX fill facts require a positive public-instruments contract size")
        contract_size_verified = True
        order.exchange_order_id = fill.order_id
        order.okx_inst_id = fill.inst_id
        order.symbol = symbol_from_okx_inst_id(fill.inst_id) or fill.symbol
        order.side = fill.side
        order.quantity = _fill_base_quantity(fill, contract_size)
        order.price = fill.avg_price
        order.fee = fill.fee_abs
        order.status = "filled"
        order.filled_at = fill.timestamp or getattr(order, "filled_at", None) or now
        order.okx_trade_ids = ",".join(fill.trade_ids)
        order.okx_fill_contracts = fill.contracts
        order.okx_fill_pnl = fill.fill_pnl
        order.okx_state = "filled"
        order.okx_sync_status = sync_status
        order.okx_synced_at = now
        order.okx_last_error = None
        existing_raw = getattr(order, "okx_raw_fills", None)
        existing_raw = existing_raw if isinstance(existing_raw, dict) else {}
        raw_fact = {
            "fills_history_confirmed": True,
            "order_id": fill.order_id,
            "trade_ids": list(fill.trade_ids),
            "inst_id": fill.inst_id,
            "pos_side": fill.pos_side,
            "contracts": fill.contracts,
            "contract_size": contract_size or None,
            "contract_size_verified": contract_size_verified,
            "contract_size_source": contract_size_source,
            "base_quantity": _fill_base_quantity(fill, contract_size),
            "avg_price": fill.avg_price,
            "fee_abs": fill.fee_abs,
            "fill_pnl": fill.fill_pnl,
            "timestamp": fill.timestamp.isoformat() if fill.timestamp else None,
            "rows": [_authoritative_fill_row(row) for row in fill.rows],
            "order_rows": [dict(order_row)] if isinstance(order_row, dict) and order_row else [],
            "execution_slippage": _authoritative_pull_slippage_fact(
                fill=fill,
                contract_size=contract_size,
            ),
        }
        protection_submission = existing_raw.get("protection_submission")
        if isinstance(protection_submission, dict) and protection_submission:
            raw_fact["protection_submission"] = dict(protection_submission)
        if isinstance(protection_execution, dict) and protection_execution:
            raw_fact["protection_execution"] = dict(protection_execution)
        order.okx_raw_fills = raw_fact


def _order_has_okx_execution_result_fact(order: Order) -> bool:
    raw = getattr(order, "okx_raw_fills", None)
    if not isinstance(raw, dict):
        return False
    if raw.get("execution_result_confirmed") is not True:
        return False
    order_id = str(getattr(order, "exchange_order_id", "") or "").strip()
    raw_order_id = str(raw.get("order_id") or "").strip()
    inst_id = str(raw.get("inst_id") or getattr(order, "okx_inst_id", "") or "").strip().upper()
    contracts = _safe_float(raw.get("contracts") or getattr(order, "okx_fill_contracts", None), 0.0)
    avg_price = _safe_float(raw.get("avg_price") or getattr(order, "price", None), 0.0)
    if not order_id or not raw_order_id or order_id != raw_order_id:
        return False
    if not inst_id or not inst_id.endswith("-SWAP"):
        return False
    return contracts > 0 and avg_price > 0


def _order_has_okx_order_detail_fact(order: Order) -> bool:
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    return bool(
        raw.get("order_detail_confirmed") is True
        and raw.get("fills_history_confirmed") is False
        and raw.get("execution_result_confirmed") is False
        and str(raw.get("source") or "").strip() == "okx_order_detail"
        and _embedded_okx_order_detail_complete(order, raw)
        and _order_has_verified_public_contract_size(order)
    )


def _order_has_contract_sized_execution_fact(order: Order) -> bool:
    return bool(
        _order_has_authoritative_stored_okx_fill_fact(order)
        or _order_has_okx_execution_result_fact(order)
        or _order_has_okx_order_detail_fact(order)
    )


def _recover_okx_execution_result_fact_from_decision(
    order: Order,
    decision: AIDecision | None,
) -> bool:
    if decision is None:
        return False
    raw = getattr(decision, "raw_llm_response", None)
    raw = raw if isinstance(raw, dict) else {}
    execution_result = raw.get("execution_result")
    if not isinstance(execution_result, dict):
        return False
    status = str(execution_result.get("status") or "").lower().strip()
    if status not in {"filled", "partial"}:
        return False
    raw_response = execution_result.get("raw_response")
    raw_response = raw_response if isinstance(raw_response, dict) else {}
    info = raw_response.get("info") if isinstance(raw_response.get("info"), dict) else {}
    order_id = str(getattr(order, "exchange_order_id", "") or "").strip()
    result_order_id = str(
        execution_result.get("exchange_order_id")
        or execution_result.get("order_id")
        or raw_response.get("ordId")
        or raw_response.get("id")
        or info.get("ordId")
        or ""
    ).strip()
    if not order_id or not result_order_id or order_id != result_order_id:
        return False
    inst_id = okx_inst_id_from_payload(raw_response, include_fallback=False)
    if not inst_id:
        inst_id = str(info.get("instId") or "").strip().upper()
    contracts = _safe_float(
        info.get("accFillSz")
        or raw_response.get("filled_contracts")
        or raw_response.get("filled")
        or execution_result.get("filled_contracts")
        or execution_result.get("quantity"),
        0.0,
    )
    avg_price = _safe_float(
        info.get("avgPx")
        or info.get("fillPx")
        or raw_response.get("average")
        or raw_response.get("price")
        or execution_result.get("price"),
        0.0,
    )
    if not inst_id or not inst_id.endswith("-SWAP") or contracts <= 0 or avg_price <= 0:
        return False
    trade_id = str(info.get("tradeId") or raw_response.get("tradeId") or "").strip()
    fee_abs = abs(_safe_float(info.get("fee") or execution_result.get("fee"), 0.0))
    fill_pnl = _safe_float(
        info.get("fillPnl")
        or info.get("pnl")
        or raw_response.get("pnl")
        or execution_result.get("pnl"),
        0.0,
    )
    base_quantity = _safe_float(
        execution_result.get("quantity") or getattr(order, "quantity", None),
        0.0,
    )
    raw_fact = {
        "source": "okx_execution_result",
        "fills_history_confirmed": False,
        "execution_result_confirmed": True,
        "recovered_from_decision": int(getattr(decision, "id", 0) or 0) or None,
        "order_id": order_id,
        "trade_ids": [trade_id] if trade_id else [],
        "inst_id": inst_id,
        "contracts": contracts,
        "base_quantity": base_quantity,
        "avg_price": avg_price,
        "fee_abs": fee_abs,
        "fill_pnl": fill_pnl,
        "timestamp": _execution_result_timestamp(execution_result),
        "rows": [dict(info)] if info else [],
    }
    order.okx_raw_fills = raw_fact
    return _order_has_okx_execution_result_fact(order)


def _recover_okx_close_fill_fact_from_decision(
    order: Order,
    decision: AIDecision | None,
) -> bool:
    if decision is None:
        return False
    raw = getattr(decision, "raw_llm_response", None)
    raw = raw if isinstance(raw, dict) else {}
    close_fill = raw.get("close_fill")
    if not isinstance(close_fill, dict):
        return False
    order_id = str(getattr(order, "exchange_order_id", "") or "").strip()
    fill_order_id = str(close_fill.get("order_id") or close_fill.get("ordId") or "").strip()
    info = close_fill.get("order_info") if isinstance(close_fill.get("order_info"), dict) else {}
    if not fill_order_id:
        fill_order_id = str(info.get("ordId") or "").strip()
    if not order_id or not fill_order_id or order_id != fill_order_id:
        return False
    inst_id = okx_inst_id_from_payload({**close_fill, "info": info}, include_fallback=False)
    if not inst_id:
        inst_id = str(info.get("instId") or "").strip().upper()
    contracts = _safe_float(close_fill.get("contracts") or info.get("fillSz") or info.get("accFillSz"), 0.0)
    avg_price = _safe_float(close_fill.get("price") or info.get("fillPx") or info.get("avgPx"), 0.0)
    if not inst_id or not inst_id.endswith("-SWAP") or contracts <= 0 or avg_price <= 0:
        return False
    trade_id = str(close_fill.get("trade_id") or close_fill.get("tradeId") or info.get("tradeId") or "").strip()
    fee_abs = abs(_safe_float(close_fill.get("fee") or info.get("fee") or getattr(order, "fee", None), 0.0))
    fill_pnl = _safe_float(close_fill.get("pnl") or close_fill.get("fillPnl") or info.get("fillPnl"), 0.0)
    base_quantity = _safe_float(
        close_fill.get("quantity")
        or close_fill.get("base_quantity")
        or getattr(order, "quantity", None),
        0.0,
    )
    timestamp = _close_fill_timestamp(close_fill, info)
    raw_fact = {
        "source": str(close_fill.get("source") or raw.get("source") or "okx_reconcile_close_fill"),
        "fills_history_confirmed": True,
        "execution_result_confirmed": False,
        "recovered_from_decision": int(getattr(decision, "id", 0) or 0) or None,
        "order_id": order_id,
        "trade_ids": [trade_id] if trade_id else [],
        "inst_id": inst_id,
        "pos_side": str(info.get("posSide") or close_fill.get("posSide") or "").lower().strip(),
        "contracts": contracts,
        "contract_size": _safe_float(close_fill.get("contract_size"), 0.0) or None,
        "base_quantity": base_quantity,
        "avg_price": avg_price,
        "fee_abs": fee_abs,
        "fill_pnl": fill_pnl,
        "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else None,
        "rows": [dict(info)] if info else [],
    }
    order.okx_raw_fills = raw_fact
    return _order_has_confirmed_okx_fill_fact(order)


def _embedded_okx_order_detail_complete(order: Order, raw: dict[str, Any]) -> bool:
    exchange_order_id = str(getattr(order, "exchange_order_id", "") or "").strip()
    fact_order_id = str(raw.get("order_id") or "").strip()
    fact_inst_id = str(raw.get("inst_id") or "").strip().upper()
    fact_contracts = _safe_float(raw.get("contracts"), 0.0)
    fact_price = _safe_float(raw.get("avg_price"), 0.0)
    fact_trade_ids = _trade_id_set(raw.get("trade_ids"))
    rows = raw.get("rows") if isinstance(raw.get("rows"), list) else []
    detail = next(
        (
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("ordId") or "").strip() == exchange_order_id
        ),
        None,
    )
    if not isinstance(detail, dict):
        return False
    detail_contracts = _safe_float(detail.get("accFillSz") or detail.get("fillSz"), 0.0)
    detail_price = _safe_float(detail.get("avgPx") or detail.get("fillPx"), 0.0)
    detail_trade_id = str(detail.get("tradeId") or "").strip()
    return bool(
        exchange_order_id
        and fact_order_id == exchange_order_id
        and str(detail.get("state") or "").strip().lower() == "filled"
        and str(detail.get("instId") or "").strip().upper() == fact_inst_id
        and detail_trade_id
        and detail_trade_id in fact_trade_ids
        and fact_contracts > 0
        and _relative_close_enough(detail_contracts, fact_contracts, 0.000001)
        and fact_price > 0
        and _relative_close_enough(detail_price, fact_price, 0.000001)
        and detail.get("fee") is not None
        and _safe_float(detail.get("fillTime") or detail.get("uTime"), 0.0) > 0
    )


def _promote_execution_result_to_order_detail(order: Order, *, now: datetime) -> bool:
    if not _order_has_okx_execution_result_fact(order):
        return False
    raw = dict(getattr(order, "okx_raw_fills", None) or {})
    if not _embedded_okx_order_detail_complete(order, raw):
        return False
    if not _order_has_verified_public_contract_size(order):
        return False
    raw["source"] = "okx_order_detail"
    raw["order_detail_confirmed"] = True
    raw["execution_result_confirmed"] = False
    raw["fills_history_confirmed"] = False
    order.okx_raw_fills = raw
    order.okx_state = "order_detail_confirmed"
    order.okx_sync_status = OKX_SYNC_ORDER_DETAIL_CONFIRMED
    order.okx_synced_at = now
    order.okx_last_error = None
    return True


def _close_fill_timestamp(close_fill: dict[str, Any], info: dict[str, Any]) -> datetime | None:
    for value in (
        close_fill.get("timestamp"),
        close_fill.get("filled_at"),
        info.get("ts"),
        info.get("fillTime"),
        close_fill.get("timestamp_ms"),
    ):
        parsed = _parse_datetime(value) if isinstance(value, str) else None
        if parsed is not None:
            return parsed
        parsed = _datetime_from_ms(value)
        if parsed is not None:
            return parsed
    return None


def _execution_result_timestamp(execution_result: dict[str, Any]) -> str | None:
    for key in ("timestamp", "filled_at", "created_at"):
        value = execution_result.get(key)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _apply_execution_result_confirmation_to_order(order: Order, *, now: datetime) -> None:
    raw = dict(getattr(order, "okx_raw_fills", None) or {})
    inst_id = str(raw.get("inst_id") or getattr(order, "okx_inst_id", "") or "").strip().upper()
    trade_ids = [
        str(item or "").strip()
        for item in (raw.get("trade_ids") or [])
        if str(item or "").strip()
    ]
    contracts = _safe_float(raw.get("contracts") or getattr(order, "okx_fill_contracts", None), 0.0)
    base_quantity = _stored_fill_base_quantity(raw)
    avg_price = _safe_float(raw.get("avg_price") or getattr(order, "price", None), 0.0)
    fee_abs = _safe_float(raw.get("fee_abs") or getattr(order, "fee", None), 0.0)
    fill_pnl = _safe_float(raw.get("fill_pnl") or getattr(order, "okx_fill_pnl", None), 0.0)
    if inst_id:
        order.okx_inst_id = inst_id
        order.symbol = symbol_from_okx_inst_id(inst_id) or order.symbol
    if contracts > 0:
        order.okx_fill_contracts = contracts
    if base_quantity > 0:
        order.quantity = base_quantity
        raw["base_quantity"] = base_quantity
    if avg_price > 0:
        order.price = avg_price
    if fee_abs >= 0:
        order.fee = fee_abs
    order.okx_trade_ids = ",".join(trade_ids) if trade_ids else None
    order.okx_fill_pnl = fill_pnl
    order_detail_confirmed = raw.get("order_detail_confirmed") is True
    order.okx_state = (
        "order_detail_confirmed"
        if order_detail_confirmed
        else "execution_result_confirmed"
    )
    order.okx_sync_status = (
        OKX_SYNC_ORDER_DETAIL_CONFIRMED
        if order_detail_confirmed
        else OKX_SYNC_EXECUTION_RESULT_CONFIRMED
    )
    order.okx_synced_at = now
    order.okx_last_error = None
    raw["execution_result_confirmed"] = not order_detail_confirmed
    raw.setdefault("fills_history_confirmed", False)
    order.okx_raw_fills = raw


def _apply_close_fill_confirmation_to_order(order: Order, *, now: datetime) -> None:
    raw = dict(getattr(order, "okx_raw_fills", None) or {})
    inst_id = str(raw.get("inst_id") or getattr(order, "okx_inst_id", "") or "").strip().upper()
    trade_ids = [
        str(item or "").strip()
        for item in (raw.get("trade_ids") or [])
        if str(item or "").strip()
    ]
    contracts = _safe_float(raw.get("contracts") or getattr(order, "okx_fill_contracts", None), 0.0)
    base_quantity = _stored_fill_base_quantity(raw)
    avg_price = _safe_float(raw.get("avg_price") or getattr(order, "price", None), 0.0)
    fee_abs = _safe_float(raw.get("fee_abs") or getattr(order, "fee", None), 0.0)
    fill_pnl = _safe_float(raw.get("fill_pnl") or getattr(order, "okx_fill_pnl", None), 0.0)
    filled_at = _parse_datetime(raw.get("timestamp"))
    if inst_id:
        order.okx_inst_id = inst_id
        order.symbol = symbol_from_okx_inst_id(inst_id) or order.symbol
    if contracts > 0:
        order.okx_fill_contracts = contracts
    if base_quantity > 0:
        order.quantity = base_quantity
        raw["base_quantity"] = base_quantity
    if avg_price > 0:
        order.price = avg_price
    if fee_abs >= 0:
        order.fee = fee_abs
    if filled_at is not None:
        order.filled_at = filled_at
    order.okx_trade_ids = ",".join(trade_ids) if trade_ids else None
    order.okx_fill_pnl = fill_pnl
    order.okx_state = "filled"
    order.okx_sync_status = OKX_SYNC_CONFIRMED
    order.okx_synced_at = now
    order.okx_last_error = None
    raw["fills_history_confirmed"] = True
    order.okx_raw_fills = raw


def _repair_stored_fill_contract_size_from_instruments(
    order: Order,
    *,
    contract_sizes: dict[str, float],
    now: datetime,
) -> bool:
    if not _order_has_authoritative_stored_okx_fill_fact(order):
        return False
    raw = dict(getattr(order, "okx_raw_fills", None) or {})
    inst_id = str(raw.get("inst_id") or "").strip().upper()
    if not inst_id:
        return False
    contract_size = _safe_float(contract_sizes.get(inst_id), 0.0)
    if contract_size <= 0:
        return False
    contracts = _safe_float(raw.get("contracts"), 0.0)
    if contracts <= 0:
        return False
    base_quantity = contracts * contract_size
    if base_quantity <= 0:
        return False

    existing_contract_size = _safe_float(raw.get("contract_size") or raw.get("contractSize"), 0.0)
    existing_base_quantity = _safe_float(raw.get("base_quantity") or raw.get("filled_base_quantity"), 0.0)
    local_quantity = _safe_float(getattr(order, "quantity", None), 0.0)
    already_verified = (
        raw.get("contract_size_verified") is True
        and _relative_close_enough(existing_contract_size, contract_size, 0.000001)
        and _relative_close_enough(existing_base_quantity, base_quantity, 0.000001)
        and _relative_close_enough(local_quantity, base_quantity, 0.000001)
    )
    contract_size_source = "okx_public_instruments"
    stored_rows = raw.get("rows")
    execution_slippage = (
        build_okx_fill_mark_slippage(
            order_id=raw.get("order_id"),
            inst_id=raw.get("inst_id"),
            side=getattr(order, "side", None),
            contracts=contracts,
            average_price=raw.get("avg_price"),
            contract_size=contract_size,
            rows=stored_rows,
        )
        if _stored_slippage_fact_needs_refresh(order)
        and isinstance(stored_rows, list)
        and stored_rows
        else None
    )
    if execution_slippage is not None:
        execution_slippage["recovery_terminal"] = False
        execution_slippage["recovery_source"] = "stored_okx_fill_rows"
    execution_slippage_changed = bool(
        execution_slippage is not None
        and raw.get("execution_slippage") != execution_slippage
    )
    if already_verified:
        if (
            str(raw.get("contract_size_source") or "").strip()
            != contract_size_source
            or execution_slippage_changed
        ):
            order.okx_inst_id = inst_id
            order.symbol = symbol_from_okx_inst_id(inst_id) or order.symbol
            order.okx_synced_at = now
            order.okx_last_error = None
            raw["contract_size_source"] = contract_size_source
            raw["contract_size_verified"] = True
            raw["fills_history_confirmed"] = True
            if execution_slippage is not None:
                raw["execution_slippage"] = execution_slippage
            order.okx_raw_fills = raw
            return True
        return False

    order.okx_inst_id = inst_id
    order.symbol = symbol_from_okx_inst_id(inst_id) or order.symbol
    order.quantity = base_quantity
    order.okx_fill_contracts = contracts
    order.okx_state = "filled"
    if str(getattr(order, "okx_sync_status", "") or "").strip() not in {
        OKX_SYNC_CONFIRMED,
        OKX_SYNC_OKX_ONLY,
    }:
        order.okx_sync_status = OKX_SYNC_CONFIRMED
    order.okx_synced_at = now
    order.okx_last_error = None
    raw["inst_id"] = inst_id
    raw["contracts"] = contracts
    raw["contract_size"] = contract_size
    raw["contract_size_verified"] = True
    raw["contract_size_source"] = contract_size_source
    raw["base_quantity"] = base_quantity
    raw["fills_history_confirmed"] = True
    if execution_slippage is not None:
        raw["execution_slippage"] = execution_slippage
    order.okx_raw_fills = raw
    return True


def _repair_execution_contract_size_from_instruments(
    order: Order,
    *,
    contract_sizes: dict[str, float],
    now: datetime,
) -> bool:
    if not (
        _order_has_okx_execution_result_fact(order)
        or _order_has_okx_order_detail_fact(order)
    ):
        return False
    raw = dict(getattr(order, "okx_raw_fills", None) or {})
    inst_id = str(raw.get("inst_id") or _order_inst_id(order)).strip().upper()
    if not inst_id:
        return False
    contract_size = _safe_float(contract_sizes.get(inst_id), 0.0)
    if contract_size <= 0:
        return False
    contracts = _safe_float(raw.get("contracts") or getattr(order, "okx_fill_contracts", None), 0.0)
    if contracts <= 0:
        return False
    base_quantity = contracts * contract_size
    if base_quantity <= 0:
        return False

    existing_contract_size = _safe_float(raw.get("contract_size") or raw.get("contractSize"), 0.0)
    existing_base_quantity = _safe_float(raw.get("base_quantity") or raw.get("filled_base_quantity"), 0.0)
    local_quantity = _safe_float(getattr(order, "quantity", None), 0.0)
    already_verified = (
        raw.get("contract_size_verified") is True
        and _relative_close_enough(existing_contract_size, contract_size, 0.000001)
        and _relative_close_enough(existing_base_quantity, base_quantity, 0.000001)
        and _relative_close_enough(local_quantity, base_quantity, 0.000001)
    )
    contract_size_source = "okx_public_instruments"
    if already_verified:
        if str(raw.get("contract_size_source") or "").strip() != contract_size_source:
            order.okx_inst_id = inst_id
            order.symbol = symbol_from_okx_inst_id(inst_id) or order.symbol
            order.okx_synced_at = now
            order.okx_last_error = None
            raw["contract_size_source"] = contract_size_source
            raw["contract_size_verified"] = True
            raw["execution_result_confirmed"] = (
                raw.get("order_detail_confirmed") is not True
            )
            order.okx_raw_fills = raw
            return True
        return False

    order.okx_inst_id = inst_id
    order.symbol = symbol_from_okx_inst_id(inst_id) or order.symbol
    order.quantity = base_quantity
    order.okx_fill_contracts = contracts
    order_detail_confirmed = raw.get("order_detail_confirmed") is True
    order.okx_state = (
        "order_detail_confirmed"
        if order_detail_confirmed
        else "execution_result_confirmed"
    )
    order.okx_sync_status = (
        OKX_SYNC_ORDER_DETAIL_CONFIRMED
        if order_detail_confirmed
        else OKX_SYNC_EXECUTION_RESULT_CONFIRMED
    )
    order.okx_synced_at = now
    order.okx_last_error = None
    raw["inst_id"] = inst_id
    raw["contracts"] = contracts
    raw["contract_size"] = contract_size
    raw["contract_size_verified"] = True
    raw["contract_size_source"] = contract_size_source
    raw["base_quantity"] = base_quantity
    raw["execution_result_confirmed"] = not order_detail_confirmed
    raw.setdefault("fills_history_confirmed", False)
    order.okx_raw_fills = raw
    return True


def _order_inst_id(order: Order) -> str:
    inst_id = str(getattr(order, "okx_inst_id", "") or "").strip().upper()
    if inst_id:
        return inst_id
    return okx_inst_id_from_symbol(getattr(order, "symbol", None)) or ""


def _order_requires_native_full_close_backfill(order: Order) -> bool:
    sync_status = str(getattr(order, "okx_sync_status", "") or "").strip()
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    if sync_status == OKX_SYNC_NATIVE_CLOSE_BACKFILL_PENDING:
        return True
    return bool(
        raw.get("requires_okx_fill_backfill")
        or raw.get("source") == OKX_SYNC_NATIVE_CLOSE_BACKFILL_PENDING
    )


def _matching_native_full_close_pending_fill(
    order: Order,
    *,
    fills: list[OkxNativeFillGroup],
    contract_sizes: dict[str, float],
) -> OkxNativeFillGroup | None:
    if not _order_requires_native_full_close_backfill(order):
        return None
    if _split_exchange_order_ids(getattr(order, "exchange_order_id", None)):
        return None
    inst_id = _order_inst_id(order)
    side = str(getattr(order, "side", "") or "").strip().lower()
    if not inst_id or side not in {"buy", "sell"}:
        return None
    reference_time = _order_time(order)
    if reference_time is None:
        return None
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    target_contracts = _safe_float(
        raw.get("contracts") or raw.get("filled_contracts") or getattr(order, "okx_fill_contracts", None),
        0.0,
    )
    target_base_quantity = _safe_float(
        raw.get("base_quantity") or raw.get("filled_base_quantity") or getattr(order, "quantity", None),
        0.0,
    )
    candidates: list[tuple[float, OkxNativeFillGroup]] = []
    for fill in fills:
        order_id = str(getattr(fill, "order_id", "") or "").strip()
        if not order_id:
            continue
        if str(getattr(fill, "inst_id", "") or "").strip().upper() != inst_id:
            continue
        if str(getattr(fill, "side", "") or "").strip().lower() != side:
            continue
        fill_time = getattr(fill, "timestamp", None)
        if fill_time is None:
            continue
        delta = abs((_aware_utc(fill_time) - _aware_utc(reference_time)).total_seconds())
        if delta > NATIVE_FULL_CLOSE_BACKFILL_WINDOW_SECONDS:
            continue
        contract_size = _contract_size_for_fill(fill, contract_sizes)
        fill_base_quantity = _fill_base_quantity(fill, contract_size)
        fill_contracts = _safe_float(getattr(fill, "contracts", None), 0.0)
        quantity_matches = False
        if target_contracts > 0 and fill_contracts > 0:
            quantity_matches = _relative_close_enough(fill_contracts, target_contracts, 0.02)
        if not quantity_matches and target_base_quantity > 0 and fill_base_quantity > 0:
            quantity_matches = _relative_close_enough(
                fill_base_quantity,
                target_base_quantity,
                0.02,
            )
        if not quantity_matches:
            continue
        candidates.append((delta, fill))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _order_needs_okx_fact_refresh(order: Order) -> bool:
    exchange_ids = _split_exchange_order_ids(getattr(order, "exchange_order_id", None))
    if not exchange_ids:
        if _order_requires_native_full_close_backfill(order):
            return True
        return _order_is_rejected_without_exchange_fill(order)
    status = str(getattr(order, "status", "") or "").lower().strip()
    sync_status = str(getattr(order, "okx_sync_status", "") or "").strip()
    refreshable_status = status in {
        "filled",
        "partial",
        "partially_filled",
        "rejected",
        "failed",
        "error",
        "cancelled",
        "canceled",
    }
    if _order_has_protection_fill_hint_without_execution(order):
        return refreshable_status
    if sync_status == OKX_SYNC_CONFIRMED and _order_has_confirmed_okx_fill_fact(
        order,
        require_verified_contract_size=True,
    ):
        if _stored_fill_contract_size_needs_public_reverification(order):
            return refreshable_status
        return False
    if sync_status == OKX_SYNC_OKX_ONLY and _order_has_confirmed_okx_fill_fact(
        order,
        require_verified_contract_size=True,
    ):
        if _stored_fill_contract_size_needs_public_reverification(order):
            return refreshable_status
        return False
    if (
        sync_status == OKX_SYNC_ORDER_DETAIL_CONFIRMED
        and _order_has_okx_order_detail_fact(order)
    ):
        return False
    if sync_status == OKX_SYNC_NO_FILL_REJECTED and status in {
        "rejected",
        "failed",
        "error",
        "cancelled",
        "canceled",
    }:
        return False
    return refreshable_status


def _decision_completed_submit_exchange_order_ids(
    decision: AIDecision | None,
) -> set[str]:
    if decision is None:
        return set()
    raw = getattr(decision, "raw_llm_response", None)
    raw = raw if isinstance(raw, dict) else {}
    state_machine = raw.get("decision_state_machine")
    state_machine = state_machine if isinstance(state_machine, dict) else {}
    stages = state_machine.get("stages")
    stages = stages if isinstance(stages, list) else []
    result: set[str] = set()
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        data = stage.get("data")
        data = data if isinstance(data, dict) else {}
        if (
            str(stage.get("stage") or "").strip() != "exchange_submit"
            or str(stage.get("status") or "").strip() != "passed"
            or data.get("has_execution_result") is not True
            or str(data.get("status") or "").strip().lower()
            not in {"filled", "partial", "partially_filled"}
        ):
            continue
        exchange_order_id = str(data.get("exchange_order_id") or "").strip()
        if exchange_order_id:
            result.add(exchange_order_id)
    return result


def _order_exchange_ids_with_completed_submit(
    order: Order,
    decision: AIDecision | None,
) -> set[str]:
    return _split_exchange_order_ids(
        getattr(order, "exchange_order_id", None)
    ) | _decision_completed_submit_exchange_order_ids(decision)


def _apply_completed_submit_fill_recovery_to_decision(
    decision: AIDecision,
    *,
    fill: OkxNativeFillGroup,
    contract_size: float,
    now: datetime,
) -> None:
    base_quantity = _fill_base_quantity(fill, contract_size)
    decision.was_executed = True
    decision.executed_at = fill.timestamp or now
    decision.execution_price = fill.avg_price
    decision.execution_reason = (
        "OKX 原生成交已确认；系统已从提交成功阶段恢复被外层超时覆盖的执行结果。"
    )
    raw = getattr(decision, "raw_llm_response", None)
    raw = dict(raw) if isinstance(raw, dict) else {}
    previous_execution_result = raw.get("execution_result")
    recovery = {
        "version": "2026-07-26.completed-submit-fill-recovery.v1",
        "source_authority": "decision_submit_stage_plus_okx_native_fills",
        "exchange_order_id": fill.order_id,
        "contracts": fill.contracts,
        "contract_size": contract_size,
        "base_quantity": base_quantity,
        "average_price": fill.avg_price,
        "fee_usdt": fill.fee_abs,
        "fill_pnl": fill.fill_pnl,
        "recovered_at": now.isoformat(),
        "superseded_execution_result": (
            dict(previous_execution_result)
            if isinstance(previous_execution_result, dict)
            else previous_execution_result
        ),
    }
    raw["completed_submit_fill_recovery"] = recovery
    raw["execution_result"] = {
        "source": "okx_native_fill_recovered_after_outer_cancellation",
        "order_id": fill.order_id,
        "exchange_order_id": fill.order_id,
        "status": "filled",
        "quantity": base_quantity,
        "price": fill.avg_price,
        "fee": fill.fee_abs,
        "pnl": fill.fill_pnl,
        "exchange_confirmed": True,
        "exit_progress": True,
        "okx_exit_position_mismatch_summary": None,
        "raw_response": {
            "recovered_from_completed_submit_stage": True,
            "trade_ids": list(fill.trade_ids),
        },
    }
    recommendation = raw.get("trade_recommendation_contract")
    if isinstance(recommendation, dict):
        recommendation = dict(recommendation)
        recommendation["execution"] = {
            "status": "filled",
            "source": "okx_native_fill_recovered_after_outer_cancellation",
            "exchange_confirmed": True,
            "exit_progress": True,
            "order_id": fill.order_id,
            "exchange_order_id": fill.order_id,
            "filled_quantity": base_quantity,
            "filled_price": fill.avg_price,
            "fee_usdt": fill.fee_abs,
            "realized_pnl_usdt": fill.fill_pnl,
            "recorded_at": now.isoformat(),
        }
        raw["trade_recommendation_contract"] = recommendation
    decision.raw_llm_response = raw


def _order_needs_local_stored_fact_recovery(order: Order) -> bool:
    sync_status = str(getattr(order, "okx_sync_status", "") or "").strip()
    if sync_status in {OKX_SYNC_CONFIRMED, OKX_SYNC_OKX_ONLY} and _order_has_confirmed_okx_fill_fact(order):
        return False
    if sync_status == OKX_SYNC_EXECUTION_RESULT_CONFIRMED and _order_has_okx_execution_result_fact(order):
        return False
    if sync_status == OKX_SYNC_ORDER_DETAIL_CONFIRMED and _order_has_okx_order_detail_fact(order):
        return False
    if sync_status == OKX_SYNC_NO_FILL_REJECTED and _order_is_rejected_without_exchange_fill(order):
        return False
    return _order_needs_okx_fact_refresh(order)


def _order_needs_okx_pull(order: Order) -> bool:
    exchange_ids = _split_exchange_order_ids(getattr(order, "exchange_order_id", None))
    if not exchange_ids:
        return _order_requires_native_full_close_backfill(order)
    status = str(getattr(order, "status", "") or "").lower().strip()
    if status not in {
        "filled",
        "partial",
        "partially_filled",
        "rejected",
        "failed",
        "error",
        "cancelled",
        "canceled",
    }:
        return False
    if _order_has_protection_fill_hint_without_execution(order):
        return True
    if _stored_slippage_fact_needs_refresh(order):
        return True
    if _order_has_authoritative_stored_okx_fill_fact(order):
        return False
    if _order_has_okx_order_detail_fact(order):
        return False
    return True


def _order_has_protection_fill_hint_without_execution(order: Order) -> bool:
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    execution = raw.get("protection_execution")
    if isinstance(execution, dict) and execution.get("lifecycle_complete") is True:
        return False
    rows = [
        row
        for key in ("rows", "order_rows")
        for row in (raw.get(key) if isinstance(raw.get(key), list) else [])
        if isinstance(row, dict)
    ]
    return any(
        str(row.get("algoId") or row.get("algoClOrdId") or "").strip()
        or str(row.get("source") or "").strip() == "7"
        or str(row.get("clOrdId") or "").strip().startswith("O")
        for row in rows
    )


def _order_has_confirmed_okx_fill_fact(
    order: Order,
    *,
    require_verified_contract_size: bool = False,
) -> bool:
    if not _order_has_authoritative_stored_okx_fill_fact(order):
        return False
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    if (
        require_verified_contract_size
        and raw.get("fills_history_confirmed") is True
        and raw.get("contract_size_verified") is not True
    ):
        return False
    return _order_fill_fact_matches_local(order, raw)


def authoritative_order_fee_fact_source(
    order: Order,
    *,
    order_id: str,
) -> str | None:
    """Return the exact OKX source that authorizes an entry-fee fact."""

    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    if raw.get("fee_abs") is None:
        return None
    if str(getattr(order, "exchange_order_id", "") or "").strip() != order_id:
        return None
    if str(raw.get("order_id") or "").strip() != order_id:
        return None
    if _order_has_confirmed_okx_fill_fact(
        order,
        require_verified_contract_size=True,
    ):
        return "okx_fills_history"
    if (
        _order_has_okx_order_detail_fact(order)
        and _order_fill_fact_matches_local(order, raw)
    ):
        return "okx_order_detail"
    return None


def _order_fill_fact_matches_local(order: Order, raw: dict[str, Any]) -> bool:
    expected_quantity = _stored_fill_base_quantity(raw)
    raw_base_quantity = _safe_float(
        raw.get("base_quantity") or raw.get("filled_base_quantity"),
        0.0,
    )
    if expected_quantity > 0 and raw_base_quantity > 0 and not _relative_close_enough(
        raw_base_quantity,
        expected_quantity,
        0.001,
    ):
        return False
    local_quantity = _safe_float(getattr(order, "quantity", None), 0.0)
    if expected_quantity > 0 and not _relative_close_enough(
        local_quantity,
        expected_quantity,
        0.001,
    ):
        return False
    expected_price = _safe_float(raw.get("avg_price") or raw.get("average"), 0.0)
    local_price = _safe_float(getattr(order, "price", None), 0.0)
    if expected_price > 0 and local_price > 0 and not _relative_close_enough(
        local_price,
        expected_price,
        0.001,
    ):
        return False
    expected_fee = _safe_float(raw.get("fee_abs"), -1.0)
    local_fee = _safe_float(getattr(order, "fee", None), -1.0)
    if expected_fee >= 0 and local_fee >= 0:
        fee_matches = (
            isclose(local_fee, expected_fee, rel_tol=1e-9, abs_tol=1e-12)
            if expected_fee == 0 or local_fee == 0
            else _relative_close_enough(local_fee, expected_fee, 0.001)
        )
        if not fee_matches:
            return False
    return True


def _order_has_authoritative_stored_okx_fill_fact(order: Order) -> bool:
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    if raw.get("fills_history_confirmed") is not True:
        return False
    order_id = str(getattr(order, "exchange_order_id", "") or "").strip()
    raw_order_id = str(raw.get("order_id") or "").strip()
    if not order_id or not raw_order_id or order_id != raw_order_id:
        return False
    raw_inst_id = str(raw.get("inst_id") or "").strip().upper()
    order_inst_id = str(getattr(order, "okx_inst_id", "") or "").strip().upper()
    if not raw_inst_id or (order_inst_id and order_inst_id != raw_inst_id):
        return False
    if not _trade_id_set(raw.get("trade_ids")):
        return False
    if _safe_float(raw.get("contracts"), 0.0) <= 0:
        return False
    if _safe_float(raw.get("avg_price"), 0.0) <= 0:
        return False
    return True


def _stored_slippage_fact_needs_refresh(order: Order) -> bool:
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    execution_slippage = raw.get("execution_slippage")
    execution_slippage = (
        execution_slippage if isinstance(execution_slippage, dict) else {}
    )
    return bool(
        raw.get("fills_history_confirmed") is True
        and (
            execution_slippage.get("version") != OKX_FILL_MARK_SLIPPAGE_VERSION
            or (
                execution_slippage.get("complete") is not True
                and execution_slippage.get("recovery_terminal") is not True
            )
        )
    )


def _stored_protection_execution_needs_refresh(
    order: Order,
    protection_execution: dict[str, Any] | None,
) -> bool:
    if (
        not isinstance(protection_execution, dict)
        or protection_execution.get("lifecycle_complete") is not True
    ):
        return False
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    stored = raw.get("protection_execution")
    return not isinstance(stored, dict) or stored != protection_execution


def _rebuild_stored_slippage_fact(order: Order, *, now: datetime) -> bool:
    if (
        not _stored_slippage_fact_needs_refresh(order)
        or not _order_has_authoritative_stored_okx_fill_fact(order)
    ):
        return False
    raw = dict(getattr(order, "okx_raw_fills", None) or {})
    if (
        raw.get("contract_size_verified") is not True
        or str(raw.get("contract_size_source") or "").strip()
        != "okx_public_instruments"
    ):
        return False
    rows = raw.get("rows")
    if not isinstance(rows, list) or not rows:
        return False
    previous = raw.get("execution_slippage")
    previous = previous if isinstance(previous, dict) else {}
    fact = build_okx_fill_mark_slippage(
        order_id=raw.get("order_id"),
        inst_id=raw.get("inst_id"),
        side=getattr(order, "side", None),
        contracts=raw.get("contracts"),
        average_price=raw.get("avg_price"),
        contract_size=raw.get("contract_size"),
        rows=rows,
    )
    fact["recovery_terminal"] = bool(
        fact.get("complete") is not True
        and previous.get("recovery_terminal") is True
    )
    fact["recovery_source"] = "stored_okx_fill_rows_contract_upgrade"
    if fact == previous:
        return False
    raw["execution_slippage"] = fact
    order.okx_raw_fills = raw
    order.okx_synced_at = now
    order.okx_last_error = None
    return True


def _stored_order_matches_native_fill(
    order: Order,
    fill: OkxNativeFillGroup,
    *,
    contract_size: float,
    contract_size_source: str,
) -> bool:
    """Return whether the stored order already reflects the latest cumulative fill."""

    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    expected_quantity = _fill_base_quantity(fill, contract_size)
    if (
        raw.get("fills_history_confirmed") is not True
        or fill.order_id
        not in _split_exchange_order_ids(getattr(order, "exchange_order_id", None))
        or str(raw.get("order_id") or "").strip() != fill.order_id
        or str(raw.get("inst_id") or "").strip().upper()
        != str(fill.inst_id or "").strip().upper()
        or str(getattr(order, "side", "") or "").strip().lower()
        != str(fill.side or "").strip().lower()
        or expected_quantity <= 0
        or not _relative_close_enough(
            _safe_float(getattr(order, "quantity", None), 0.0),
            expected_quantity,
            0.001,
        )
        or not _relative_close_enough(
            _safe_float(getattr(order, "okx_fill_contracts", None), 0.0),
            float(fill.contracts),
            0.001,
        )
        or not _relative_close_enough(
            _safe_float(getattr(order, "price", None), 0.0),
            float(fill.avg_price),
            0.001,
        )
        or not _relative_close_enough(
            _safe_float(getattr(order, "fee", None), 0.0),
            float(fill.fee_abs),
            0.001,
        )
        or not _relative_close_enough(
            _safe_float(getattr(order, "okx_fill_pnl", None), 0.0),
            float(fill.fill_pnl),
            0.001,
        )
        or not _relative_close_enough(
            _safe_float(raw.get("contract_size"), 0.0),
            float(contract_size),
            0.000001,
        )
        or not _relative_close_enough(
            _stored_fill_base_quantity(raw),
            expected_quantity,
            0.001,
        )
        or str(raw.get("contract_size_source") or "").strip()
        != str(contract_size_source or "").strip()
    ):
        return False
    return _trade_id_set(raw.get("trade_ids")) == set(fill.trade_ids)


def _trade_id_set(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(item or "").strip() for item in value if str(item or "").strip()}
    return _split_exchange_order_ids(value)


def _stored_fill_contract_size_needs_public_reverification(order: Order) -> bool:
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    if raw.get("fills_history_confirmed") is not True:
        return False
    return not _order_has_verified_public_contract_size(order)


def _order_has_verified_public_contract_size(order: Order) -> bool:
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    contract_size = _safe_float(raw.get("contract_size") or raw.get("contractSize"), 0.0)
    contracts = _safe_float(
        raw.get("contracts") or getattr(order, "okx_fill_contracts", None),
        0.0,
    )
    expected_quantity = contracts * contract_size
    return bool(
        raw.get("contract_size_verified") is True
        and str(raw.get("contract_size_source") or "").strip()
        == "okx_public_instruments"
        and contract_size > 0
        and contracts > 0
        and expected_quantity > 0
        and _relative_close_enough(
            _safe_float(raw.get("base_quantity"), 0.0),
            expected_quantity,
            0.000001,
        )
        and _relative_close_enough(
            _safe_float(getattr(order, "quantity", None), 0.0),
            expected_quantity,
            0.000001,
        )
    )


def _dedupe_order_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        order_id = _order_row_id(row)
        if not order_id or order_id in seen:
            continue
        seen.add(order_id)
        result.append(row)
    return result


def _order_rows_inst_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {
        inst_id
        for row in rows
        if (inst_id := str(row.get("instId") or "").strip().upper())
    }


def _order_rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        order_id = _order_row_id(row)
        if order_id:
            result.setdefault(order_id, row)
    return result


def _order_row_id(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return ""
    return str(row.get("ordId") or row.get("order") or row.get("id") or "").strip()


def _order_row_client_order_id(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return ""
    return str(row.get("clOrdId") or row.get("clientOrderId") or "").strip()


def _fill_client_order_id(
    fill: OkxNativeFillGroup,
    order_row: dict[str, Any] | None,
) -> str:
    client_order_id = _order_row_client_order_id(order_row)
    if client_order_id:
        return client_order_id
    for row in fill.rows:
        client_order_id = _order_row_client_order_id(row)
        if client_order_id:
            return client_order_id
    return ""


def _decision_id_from_client_order_id(value: Any) -> int | None:
    return normal_paper_decision_id_from_client_order_id(
        value
    ) or paper_training_decision_id_from_client_order_id(value)


def _decision_for_order_fact(
    *,
    fill: OkxNativeFillGroup,
    order_row: dict[str, Any] | None,
    decisions_by_id: dict[int, AIDecision],
) -> AIDecision | None:
    client_order_id = _fill_client_order_id(fill, order_row)
    decision_id = _decision_id_from_client_order_id(client_order_id)
    decision = decisions_by_id.get(int(decision_id or 0))
    if decision is None or getattr(decision, "is_paper", None) is not True:
        return None
    raw = getattr(decision, "raw_llm_response", None)
    raw = raw if isinstance(raw, dict) else {}
    expected_side = {
        "long": "buy",
        "short": "sell",
    }.get(str(getattr(decision, "action", "") or "").strip().lower())
    shared_reasons = [
        reason
        for reason, invalid in (
            (
                "normal_paper_order_identity_symbol_mismatch",
                normalize_trading_symbol(getattr(decision, "symbol", None))
                != normalize_trading_symbol(fill.symbol),
            ),
            (
                "normal_paper_order_identity_side_mismatch",
                expected_side != str(fill.side or "").strip().lower(),
            ),
        )
        if invalid
    ]
    if normal_paper_decision_id_from_client_order_id(client_order_id):
        identity = raw.get("normal_paper_order_identity")
        identity = identity if isinstance(identity, dict) else {}
        identity_reasons = normal_paper_order_identity_reasons(
            identity,
            decision_id=getattr(decision, "id", None),
            contract=raw.get("normal_paper_trade"),
        )
        if (
            identity_reasons
            or shared_reasons
            or str(identity.get("client_order_id") or "").strip() != client_order_id
        ):
            return None
        return decision

    contract = raw.get("paper_training")
    contract = contract if isinstance(contract, dict) else {}
    identity = raw.get("paper_training_order_identity")
    identity = identity if isinstance(identity, dict) else {}
    try:
        identity_decision_id = int(identity.get("decision_id") or 0)
    except (TypeError, ValueError):
        return None
    if (
        paper_training_contract_reasons(contract)
        or identity.get("execution_scope") != "paper_only"
        or identity.get("production_permission") is not False
        or identity_decision_id != int(decision.id or 0)
        or str(identity.get("client_order_id") or "").strip() != client_order_id
        or shared_reasons
    ):
        return None
    return decision


def _apply_exchange_recovery_to_decision(
    decision: AIDecision,
    *,
    fill: OkxNativeFillGroup,
    client_order_id: str,
    now: datetime,
) -> None:
    normal_decision = bool(normal_paper_decision_id_from_client_order_id(client_order_id))
    decision.was_executed = True
    decision.executed_at = fill.timestamp or now
    decision.execution_price = fill.avg_price
    decision.execution_reason = "OKX 已确认模拟盘订单成交；系统已按客户端订单身份恢复精确决策关联。"
    raw = getattr(decision, "raw_llm_response", None)
    raw = dict(raw) if isinstance(raw, dict) else {}
    recovery_key = (
        "normal_paper_exchange_recovery"
        if normal_decision
        else "paper_training_exchange_recovery"
    )
    raw[recovery_key] = {
        "version": (
            "2026-07-29.normal-paper-exchange-recovery.v1"
            if normal_decision
            else "2026-07-22.paper-training-exchange-recovery.v1"
        ),
        "source_authority": "okx_native_fills_and_client_order_identity",
        "execution_scope": "paper_only",
        "production_permission": False,
        "entry_type": "normal_strategy_trade" if normal_decision else "historical_paper_training",
        "client_order_id": client_order_id,
        "exchange_order_id": fill.order_id,
        "decision_id": int(decision.id or 0),
        "contracts": fill.contracts,
        "average_price": fill.avg_price,
        "fee_usdt": fill.fee_abs,
        "recovered_at": now.isoformat(),
    }
    decision.raw_llm_response = raw


def _order_row_state(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return ""
    state = str(row.get("state") or row.get("status") or "").lower().strip()
    return {
        "canceled": "canceled",
        "cancelled": "canceled",
        "partially_filled": "partially_filled",
        "live": "open",
    }.get(state, state)


def _order_row_time(row: dict[str, Any] | None) -> datetime | None:
    if not isinstance(row, dict):
        return None
    return _datetime_from_ms(row.get("uTime") or row.get("cTime"))


def _order_row_created_at(row: dict[str, Any] | None, default: datetime) -> datetime:
    if not isinstance(row, dict):
        return default
    return _datetime_from_ms(row.get("cTime")) or _datetime_from_ms(row.get("uTime")) or default


def _order_row_filled_at(row: dict[str, Any] | None) -> datetime | None:
    if not isinstance(row, dict):
        return None
    state = _order_row_state(row)
    if state not in {"filled", "partially_filled"}:
        return None
    return _datetime_from_ms(row.get("uTime") or row.get("fillTime"))


def _order_row_status(row: dict[str, Any] | None) -> str:
    state = _order_row_state(row)
    if state == "filled":
        return "filled"
    if state == "partially_filled":
        return "partially_filled"
    if state == "canceled":
        return "canceled"
    if state == "open":
        return "open"
    return state or "unknown"


def _order_row_order_type(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return "market"
    return str(row.get("ordType") or row.get("type") or "market").lower().strip() or "market"


def _order_row_price(row: dict[str, Any] | None) -> float | None:
    if not isinstance(row, dict):
        return None
    for key in ("avgPx", "fillPx", "px"):
        value = _safe_float(row.get(key), 0.0)
        if value > 0:
            return value
    return None


def _order_row_contracts(row: dict[str, Any] | None) -> float:
    if not isinstance(row, dict):
        return 0.0
    for key in ("accFillSz", "fillSz", "sz"):
        value = _safe_float(row.get(key), 0.0)
        if value > 0:
            return value
    return 0.0


def _contract_size_for_order_row(
    row: dict[str, Any],
    contract_sizes: dict[str, float],
) -> float:
    inst_id = str(row.get("instId") or "").strip().upper()
    size = _safe_float(contract_sizes.get(inst_id), 0.0)
    return size if size > 0 else 0.0


def _base_quantity_from_order_row(row: dict[str, Any], contract_size: float) -> float:
    contracts = _order_row_contracts(row)
    size = _safe_float(contract_size, 0.0)
    return contracts * size if size > 0 else 0.0


def _apply_order_row_metadata(order: Order, row: dict[str, Any], *, now: datetime) -> None:
    order_id = _order_row_id(row)
    inst_id = str(row.get("instId") or "").strip().upper()
    if order_id:
        order.exchange_order_id = order_id
    if inst_id:
        order.okx_inst_id = inst_id
        order.symbol = symbol_from_okx_inst_id(inst_id) or order.symbol
    side = str(row.get("side") or "").lower().strip()
    if side:
        order.side = side
    price = _order_row_price(row)
    if price is not None:
        order.price = price
    status = _order_row_status(row)
    if status != "unknown":
        order.status = status
    filled_at = _order_row_filled_at(row)
    if filled_at is not None:
        order.filled_at = filled_at
    order.okx_state = _order_row_state(row) or order.okx_state
    order.okx_synced_at = now
    raw = dict(order.okx_raw_fills or {})
    raw["order_id"] = order_id or raw.get("order_id")
    raw["inst_id"] = inst_id or raw.get("inst_id")
    raw["order_rows"] = [dict(row)]
    order.okx_raw_fills = raw


def _clear_unconfirmed_fill_fact(order: Order) -> None:
    """Remove stale execution facts that OKX fills-history does not confirm."""

    order.okx_trade_ids = None
    order.okx_fill_contracts = None
    order.okx_fill_pnl = None
    raw = dict(getattr(order, "okx_raw_fills", None) or {})
    for key in (
        "trade_ids",
        "contracts",
        "filled_contracts",
        "base_quantity",
        "filled_base_quantity",
        "avg_price",
        "fee_abs",
        "fill_pnl",
        "rows",
        "execution_slippage",
    ):
        raw.pop(key, None)
    raw["fills_history_confirmed"] = False
    order.okx_raw_fills = raw


def _order_from_order_history_row(
    row: dict[str, Any],
    *,
    mode: str,
    now: datetime,
    contract_size: float,
) -> Order:
    inst_id = str(row.get("instId") or "").strip().upper()
    order = Order(
        model_name="okx_authoritative_sync",
        execution_mode=mode,
        symbol=symbol_from_okx_inst_id(inst_id) or normalize_trading_symbol(inst_id),
        side=str(row.get("side") or "").lower().strip(),
        order_type=_order_row_order_type(row),
        quantity=_base_quantity_from_order_row(row, contract_size),
        price=_order_row_price(row),
        status=_order_row_status(row),
        fee=0.0,
        decision_id=None,
        exchange_order_id=_order_row_id(row),
        filled_at=_order_row_filled_at(row),
        created_at=_order_row_created_at(row, now),
    )
    order.okx_inst_id = inst_id
    order.okx_state = _order_row_state(row)
    order.okx_sync_status = OKX_SYNC_ORDER_ONLY
    order.okx_synced_at = now
    order.okx_raw_fills = {
        "order_id": _order_row_id(row),
        "inst_id": inst_id,
        "contracts": _order_row_contracts(row),
        "contract_size": contract_size or None,
        "contract_size_verified": contract_size > 0,
        "contract_size_source": (
            "okx_public_instruments"
            if contract_size > 0
            else "okx_public_instruments_missing"
        ),
        "base_quantity": _base_quantity_from_order_row(row, contract_size),
        "avg_price": _order_row_price(row),
        "order_rows": [dict(row)],
        "rows": [],
    }
    return order


def _safe_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _merge_local_order_rows(*groups: Iterable[Order]) -> list[Order]:
    merged: list[Order] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for order in group:
            database_id = str(getattr(order, "id", "") or "")
            exchange_id = str(getattr(order, "exchange_order_id", "") or "")
            key = (
                ("exchange", exchange_id)
                if exchange_id
                else ("database", database_id or str(id(order)))
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(order)
    return merged


def _dedupe_fills_by_order_id(fills: list[OkxNativeFillGroup]) -> list[OkxNativeFillGroup]:
    by_order_id: dict[str, OkxNativeFillGroup] = {}
    for fill in fills:
        order_id = str(fill.order_id or "").strip()
        if not order_id:
            continue
        existing = by_order_id.get(order_id)
        if existing is None or _fill_completeness_key(fill) > _fill_completeness_key(existing):
            by_order_id[order_id] = fill
    return sorted(
        by_order_id.values(),
        key=lambda item: item.timestamp or datetime.min.replace(tzinfo=UTC),
    )


def _fill_completeness_key(fill: OkxNativeFillGroup) -> tuple[float, int, int, float]:
    """Prefer cumulative order facts over a truncated first-page result."""

    return (
        max(float(fill.contracts or 0.0), 0.0),
        len(set(fill.trade_ids)),
        max(int(fill.raw_count or 0), 0),
        max(float(fill.timestamp_ms or 0.0), 0.0),
    )


def _algo_ids_from_order_rows(rows: list[dict[str, Any]]) -> set[str]:
    return {
        algo_id
        for row in rows
        if (algo_id := str(row.get("algoId") or "").strip())
    }


def _protection_execution_by_order_id(
    *,
    fills: list[OkxNativeFillGroup],
    order_rows_by_id: dict[str, dict[str, Any]],
    algo_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    algo_by_order_id = {
        str(row.get("ordId") or "").strip(): row
        for row in algo_rows
        if str(row.get("ordId") or "").strip()
    }
    algo_by_id = {
        str(row.get("algoId") or row.get("algoClOrdId") or "").strip(): row
        for row in algo_rows
        if str(row.get("algoId") or row.get("algoClOrdId") or "").strip()
    }
    result: dict[str, dict[str, Any]] = {}
    for fill in fills:
        order_id = str(fill.order_id or "").strip()
        order_row = order_rows_by_id.get(order_id, {})
        algo_id = str(order_row.get("algoId") or "").strip()
        algo_row = algo_by_order_id.get(order_id) or algo_by_id.get(algo_id)
        if not isinstance(algo_row, dict):
            continue
        lifecycle = build_okx_protection_execution_lifecycle(
            fill,
            order_row=order_row,
            algo_row=algo_row,
        )
        if lifecycle is not None:
            result[order_id] = lifecycle
    return result


def _prioritized_exchange_order_ids(orders: Iterable[Order], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    ordered = sorted(
        list(orders or []),
        key=lambda order: (
            not _order_has_authoritative_stored_okx_fill_fact(order),
            _stored_slippage_fact_needs_refresh(order),
            _order_time(order) or datetime.min.replace(tzinfo=UTC),
        ),
        reverse=True,
    )
    for order in ordered:
        for token in sorted(_split_exchange_order_ids(getattr(order, "exchange_order_id", None))):
            if token in seen:
                continue
            seen.add(token)
            result.append(token)
            if len(result) >= max(1, int(limit or 1)):
                return result
    return result


def _order_time(order: Order) -> datetime | None:
    return _db_datetime_to_utc(
        getattr(order, "filled_at", None) or getattr(order, "created_at", None)
    )


def _order_is_rejected_without_exchange_fill(order: Order) -> bool:
    status = str(getattr(order, "status", "") or "").lower().strip()
    if status not in {"rejected", "failed", "error", "cancelled", "canceled"}:
        return False
    quantity = _safe_float(getattr(order, "quantity", None), 0.0)
    exchange_order_id = str(getattr(order, "exchange_order_id", "") or "").strip()
    if quantity > 0 and exchange_order_id and exchange_order_id not in {"rejected", "hold", "no_position"}:
        return False
    return True


def _sample(order: Order, *, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "local_order_id": getattr(order, "id", None),
        "symbol": getattr(order, "symbol", None),
        "side": getattr(order, "side", None),
        "exchange_order_id": getattr(order, "exchange_order_id", None),
        "okx_sync_status": getattr(order, "okx_sync_status", None),
    }


def _fill_sample(fill: OkxNativeFillGroup, *, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "local_order_id": None,
        "symbol": fill.symbol,
        "side": fill.side,
        "exchange_order_id": fill.order_id,
        "okx_sync_status": None,
    }


def _fill_order_type(fill: OkxNativeFillGroup) -> str:
    row = fill.latest_row
    return str(row.get("ordType") or row.get("type") or "market").lower() or "market"


def _contract_size_for_fill(
    fill: OkxNativeFillGroup,
    contract_sizes: dict[str, float],
) -> float:
    return _contract_size_for_fill_with_source(fill, contract_sizes)[0]


def _contract_size_for_fill_with_source(
    fill: OkxNativeFillGroup,
    contract_sizes: dict[str, float],
) -> tuple[float, str]:
    for key in (fill.inst_id, okx_inst_id_from_symbol(fill.symbol) or ""):
        size = _safe_float(contract_sizes.get(key), 0.0)
        if size > 0:
            return size, "okx_public_instruments"
    return 0.0, "okx_public_instruments_missing"


def _is_verified_public_contract_size(contract_size: float, source: str) -> bool:
    return bool(
        _safe_float(contract_size, 0.0) > 0
        and str(source or "").strip() == "okx_public_instruments"
    )


def _fill_base_quantity(fill: OkxNativeFillGroup, contract_size: float) -> float:
    size = _safe_float(contract_size, 0.0)
    return float(fill.contracts) * size if size > 0 else 0.0


def _split_exchange_order_ids(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    tokens = {text}
    for separator in (",", ";", "|", "\n", "\t", " "):
        pieces: set[str] = set()
        for token in tokens:
            pieces.update(part.strip() for part in token.split(separator) if part.strip())
        tokens = pieces
    return {token for token in tokens if token}


async def _existing_order_ids(session: Any, mode: str, order_ids: set[str]) -> set[str]:
    if not order_ids:
        return set()
    rows = await session.execute(
        select(Order.exchange_order_id).where(
            Order.execution_mode == mode,
            Order.exchange_order_id.in_(sorted(order_ids)),
        )
    )
    existing: set[str] = set()
    for value in rows.scalars().all():
        existing.update(_split_exchange_order_ids(value))
    return existing


def _account_history_since(
    phase3_since: datetime,
    orders: Iterable[Order],
    *,
    overlap_hours: int,
) -> datetime:
    confirmed_times = [
        order_time
        for order in orders
        if (
            order_id := str(getattr(order, "exchange_order_id", "") or "").strip()
        )
        if authoritative_order_fee_fact_source(order, order_id=order_id) is not None
        if (order_time := _order_time(order)) is not None
    ]
    if not confirmed_times:
        return _aware_utc(phase3_since)
    overlap = timedelta(hours=max(int(overlap_hours or 0), 1))
    return max(_aware_utc(phase3_since), max(confirmed_times) - overlap)


def _aware_utc(value: datetime | None) -> datetime:
    if value is None:
        return PHASE3_DEFAULT_ORDER_SYNC_START.astimezone(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=PHASE3_DEFAULT_ORDER_SYNC_START.tzinfo)
    return value.astimezone(UTC)


def _db_datetime_to_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _db_naive_since(value: datetime) -> datetime:
    """Return the DB comparison boundary as a UTC-naive instant.

    SQLAlchemy/PostgreSQL stores these ORM datetimes as timezone-aware
    instants online, while SQLite tests often compare naive values.  The
    exchange boundary is Beijing midnight, but the comparable instant is UTC.
    """

    return _aware_utc(value).replace(tzinfo=None)


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _aware_utc(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _aware_utc(parsed)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _stored_fill_base_quantity(raw: dict[str, Any]) -> float:
    contracts = _safe_float(raw.get("contracts") or raw.get("filled_contracts"), 0.0)
    contract_size = _safe_float(
        raw.get("contract_size") or raw.get("contractSize"),
        0.0,
    )
    if contracts > 0 and contract_size > 0:
        return contracts * contract_size
    base_quantity = _safe_float(
        raw.get("base_quantity") or raw.get("filled_base_quantity"),
        0.0,
    )
    if base_quantity > 0:
        return base_quantity
    if contracts > 0:
        return contracts
    return 0.0


def _relative_close_enough(left: float, right: float, tolerance_ratio: float) -> bool:
    tolerance = max(abs(left), abs(right), 1e-12) * max(tolerance_ratio, 0.0)
    return abs(left - right) <= tolerance


def _datetime_from_ms(value: Any) -> datetime | None:
    timestamp_ms = _safe_float(value, 0.0)
    if timestamp_ms <= 0:
        return None
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000.0, UTC)
    except (OSError, OverflowError, ValueError):
        return None


async def _recover_protection_position_links(
    session: Any,
    *,
    mode: str,
    orders: Iterable[Order],
    target_order_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    order_ids = {
        str(item or "").strip()
        for item in (target_order_ids or set())
        if str(item or "").strip()
    }
    by_exchange_id: dict[str, Order] = {}
    for order in orders:
        exchange_order_id = str(getattr(order, "exchange_order_id", "") or "").strip()
        raw = getattr(order, "okx_raw_fills", None)
        raw = raw if isinstance(raw, dict) else {}
        lifecycle = raw.get("protection_execution")
        if (
            exchange_order_id
            and isinstance(lifecycle, dict)
            and lifecycle.get("lifecycle_complete") is True
        ):
            order_ids.add(exchange_order_id)
            by_exchange_id[exchange_order_id] = order
    if not order_ids:
        return []
    result = await session.execute(
        select(Order).where(
            Order.execution_mode == ("live" if str(mode).lower() == "live" else "paper"),
            Order.exchange_order_id.in_(order_ids),
        )
    )
    for order in result.scalars().all():
        exchange_order_id = str(getattr(order, "exchange_order_id", "") or "").strip()
        if exchange_order_id:
            by_exchange_id[exchange_order_id] = order
    return await recover_protection_position_lifecycles(
        session,
        orders=by_exchange_id.values(),
        mode=mode,
    )


async def _configure_order_fact_write_transaction(session: Any) -> None:
    if not str(settings.database_url or "").startswith("postgresql"):
        return
    await session.execute(
        text("SELECT set_config('lock_timeout', :value, true)"),
        {"value": f"{ORDER_FACT_SYNC_DB_LOCK_TIMEOUT_MILLISECONDS}ms"},
    )
    await session.execute(
        text("SELECT set_config('statement_timeout', :value, true)"),
        {"value": f"{ORDER_FACT_SYNC_DB_STATEMENT_TIMEOUT_MILLISECONDS}ms"},
    )


def _database_timeout_stage(exc: BaseException) -> str | None:
    current: BaseException | None = exc
    messages: list[str] = []
    sqlstates: set[str] = set()
    while current is not None:
        messages.append(str(current).lower())
        for attribute in ("sqlstate", "pgcode"):
            value = getattr(current, attribute, None)
            if value:
                sqlstates.add(str(value))
        next_exc = getattr(current, "orig", None) or current.__cause__ or current.__context__
        current = next_exc if isinstance(next_exc, BaseException) and next_exc is not current else None
    message = " | ".join(messages)
    if "55p03" in sqlstates or "lock timeout" in message or "lock_not_available" in message:
        return "database_lock_timeout"
    if (
        "57014" in sqlstates
        or "statement timeout" in message
        or "query_canceled" in message
        or "canceling statement due to statement timeout" in message
    ):
        return "database_statement_timeout"
    return None


async def _bounded(awaitable: Any, timeout_seconds: float) -> Any:
    return await asyncio.wait_for(awaitable, timeout=max(float(timeout_seconds), 0.05))


def _remaining_stage_timeout(deadline: float, cap_seconds: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        return 0.0
    return min(max(float(cap_seconds), 0.05), remaining)


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
