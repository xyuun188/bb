"""Writable OKX-native order/fill fact sync.

For OKX-backed paper/live modes, local order rows are only a cache of exchange
facts.  This service starts at the Phase 3 clean-order boundary and updates
local rows from OKX native fills (`instId`, `ordId`, `tradeId`, `fillSz`,
`fillPx`, `fee`, `fillPnl`, `ts`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import or_, select

from core.safe_output import safe_error_text
from core.symbols import (
    normalize_trading_symbol,
    okx_inst_id_from_payload,
    okx_inst_id_from_symbol,
    symbol_from_okx_inst_id,
)
from db.session import get_session_ctx
from executor.okx_executor import OKXExecutor
from models.account import OkxAccountBill
from models.decision import AIDecision
from models.learning import StrategyLearningEvent
from models.trade import Order, Position
from services.manual_close_marker import is_manual_close_order
from services.okx_native_facts import (
    OkxNativeAccountBill,
    OkxNativeFactsClient,
    OkxNativeFillGroup,
)
from services.okx_position_confirmation import (
    OkxCurrentPositionEntryConfirmation,
    find_current_position_entry_confirmation,
)
from services.phase3_boundary import PHASE3_CLEAN_START_LOCAL

logger = structlog.get_logger(__name__)

PHASE3_DEFAULT_ORDER_SYNC_START = PHASE3_CLEAN_START_LOCAL
DEFAULT_COLD_START_MARKER_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "phase3_cold_start_reset_marker.json"
)
DEFAULT_LOOKBACK_HOURS = 24
DEFAULT_LIMIT = 500
DEFAULT_TIMEOUT_SECONDS = 8.0
DEFAULT_MAX_ORDER_GAP_QUERIES = 20
CURRENT_POSITION_ENTRY_LINK_WINDOW_SECONDS = 10 * 60
CURRENT_POSITION_ENTRY_PRICE_TOLERANCE_RATIO = 0.002
CURRENT_POSITION_ENTRY_QUANTITY_TOLERANCE_RATIO = 0.02
FILL_PAIR_POSITION_TIME_WINDOW_SECONDS = 24 * 60 * 60
FILL_PAIR_POSITION_PRICE_TOLERANCE_RATIO = 0.01
FILL_PAIR_POSITION_QUANTITY_TOLERANCE_RATIO = 0.02
POSITION_HISTORY_LINK_WINDOW_SECONDS = 30 * 60
OKX_SYNC_CONFIRMED = "okx_confirmed"
OKX_SYNC_UNVERIFIED = "okx_unverified"
OKX_SYNC_OKX_ONLY = "okx_only_backfilled"
OKX_SYNC_NO_FILL_REJECTED = "okx_no_fill_rejected"
OKX_SYNC_ORDER_ONLY = "okx_order_only"
OKX_SYNC_POSITION_CONFIRMED = "okx_position_confirmed"
OKX_SYNC_EXECUTION_RESULT_CONFIRMED = "okx_execution_result_confirmed"
OKX_POSITION_SYNC_SUPPRESSION_EVENT_TYPE = "okx_position_sync_suppression"


@dataclass(frozen=True, slots=True)
class OkxPositionFactSyncSummary:
    checked_count: int = 0
    backfilled_count: int = 0
    updated_count: int = 0
    skipped_count: int = 0
    samples: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "checked_count": self.checked_count,
            "backfilled_count": self.backfilled_count,
            "updated_count": self.updated_count,
            "skipped_count": self.skipped_count,
            "samples": list(self.samples),
        }


@dataclass(frozen=True, slots=True)
class OkxPositionSyncSuppression:
    mode: str
    symbol: str
    side: str
    okx_inst_id: str
    okx_pos_id: str
    entry_order_ids: frozenset[str]
    close_order_ids: frozenset[str]
    created_at: datetime | None = None
    closed_at: datetime | None = None
    reason: str = ""

    def has_strong_identity(self) -> bool:
        return bool(self.okx_pos_id or self.entry_order_ids or self.close_order_ids)


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
    position_confirmed_count: int = 0
    unverified_count: int = 0
    backfilled_count: int = 0
    order_history_backfilled_count: int = 0
    position_history_checked_count: int = 0
    position_history_backfilled_count: int = 0
    position_history_updated_count: int = 0
    position_history_skipped_count: int = 0
    position_history_error: str | None = None
    current_position_checked_count: int = 0
    current_position_backfilled_count: int = 0
    current_position_updated_count: int = 0
    current_position_skipped_count: int = 0
    fill_pair_position_checked_count: int = 0
    fill_pair_position_backfilled_count: int = 0
    fill_pair_position_skipped_count: int = 0
    account_bill_checked_count: int = 0
    account_bill_backfilled_count: int = 0
    account_bill_updated_count: int = 0
    account_bill_skipped_count: int = 0
    account_bill_error: str | None = None
    closed_position_pnl_repair_checked_count: int = 0
    closed_position_pnl_repaired_count: int = 0
    closed_position_pnl_repair_skipped_count: int = 0
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
            "position_confirmed_count": self.position_confirmed_count,
            "unverified_count": self.unverified_count,
            "backfilled_count": self.backfilled_count,
            "order_history_backfilled_count": self.order_history_backfilled_count,
            "position_history_checked_count": self.position_history_checked_count,
            "position_history_backfilled_count": self.position_history_backfilled_count,
            "position_history_updated_count": self.position_history_updated_count,
            "position_history_skipped_count": self.position_history_skipped_count,
            "position_history_error": self.position_history_error,
            "current_position_checked_count": self.current_position_checked_count,
            "current_position_backfilled_count": self.current_position_backfilled_count,
            "current_position_updated_count": self.current_position_updated_count,
            "current_position_skipped_count": self.current_position_skipped_count,
            "fill_pair_position_checked_count": self.fill_pair_position_checked_count,
            "fill_pair_position_backfilled_count": self.fill_pair_position_backfilled_count,
            "fill_pair_position_skipped_count": self.fill_pair_position_skipped_count,
            "account_bill_checked_count": self.account_bill_checked_count,
            "account_bill_backfilled_count": self.account_bill_backfilled_count,
            "account_bill_updated_count": self.account_bill_updated_count,
            "account_bill_skipped_count": self.account_bill_skipped_count,
            "account_bill_error": self.account_bill_error,
            "closed_position_pnl_repair_checked_count": self.closed_position_pnl_repair_checked_count,
            "closed_position_pnl_repaired_count": self.closed_position_pnl_repaired_count,
            "closed_position_pnl_repair_skipped_count": self.closed_position_pnl_repair_skipped_count,
            "skipped_old_count": self.skipped_old_count,
            "error": self.error,
            "samples": list(self.samples),
        }


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
        cold_start_marker_path: str | Path | None = DEFAULT_COLD_START_MARKER_PATH,
        phase3_order_sync_start: datetime | None = PHASE3_DEFAULT_ORDER_SYNC_START,
    ) -> None:
        self.mode = "live" if str(mode or "").lower() == "live" else "paper"
        self.lookback_hours = max(int(lookback_hours or DEFAULT_LOOKBACK_HOURS), 1)
        self.limit = max(1, min(int(limit or DEFAULT_LIMIT), 2000))
        self.timeout_seconds = max(float(timeout_seconds or DEFAULT_TIMEOUT_SECONDS), 0.5)
        self.executor_factory = executor_factory or OKXExecutor
        self.cold_start_marker_path = (
            Path(cold_start_marker_path) if cold_start_marker_path is not None else None
        )
        self.phase3_order_sync_start = _aware_utc(
            phase3_order_sync_start or PHASE3_DEFAULT_ORDER_SYNC_START
        )

    async def sync(self) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        since = self._effective_since(started_at)
        since_naive = _db_naive_since(since)
        local_orders = await self._load_local_orders(since_naive)
        local_positions = await self._load_local_positions(since_naive)
        external_refresh_orders = [
            order for order in local_orders if _order_needs_okx_pull(order)
        ]
        target_order_ids = {
            token
            for order in external_refresh_orders
            for token in _split_exchange_order_ids(getattr(order, "exchange_order_id", None))
        }
        order_target_inst_ids = {
            inst_id
            for order in external_refresh_orders
            if (inst_id := _order_inst_id(order))
        }
        position_target_inst_ids = {
            inst_id
            for position in local_positions
            if (inst_id := _position_inst_id(position))
        }
        target_pos_ids = {
            pos_id
            for position in local_positions
            if (pos_id := str(getattr(position, "okx_pos_id", "") or "").strip())
        }

        executor = self.executor_factory(mode=self.mode, load_markets_on_initialize=False)
        okx_pull_available = True
        pull_error: str | None = None
        fills: list[OkxNativeFillGroup] = []
        order_rows: list[dict[str, Any]] = []
        exchange_positions: list[dict[str, Any]] = []
        contract_sizes: dict[str, float] = {}
        position_history_rows: list[dict[str, Any]] = []
        position_history_error: str | None = None
        account_bills: list[OkxNativeAccountBill] = []
        account_bill_error: str | None = None
        try:
            await _bounded(executor.initialize(), self.timeout_seconds)
            native_facts = OkxNativeFactsClient(executor)
            try:
                # Funding fees are balance-ledger events, not order/fill facts.
                # Pull them account-wide before the slower position/order sync so
                # a timeout in those later calls does not starve historical PnL.
                account_bills = await _bounded(
                    native_facts.fetch_account_bills(
                        since=since,
                        limit=100,
                        max_pages=max(3, min(10, (self.limit // 100) + 2)),
                        funding_only=True,
                        strict=True,
                    ),
                    self.timeout_seconds,
                )
            except Exception as exc:
                account_bill_error = safe_error_text(exc, limit=180)
                logger.warning(
                    "OKX account bill sync degraded; continuing order/fill fact sync",
                    mode=self.mode,
                    error=account_bill_error,
                )
            exchange_positions = await _bounded(
                native_facts.fetch_positions(),
                self.timeout_seconds,
            )
            current_position_target_inst_ids = _current_position_inst_ids(exchange_positions)
            fact_target_inst_ids = (
                order_target_inst_ids
                | position_target_inst_ids
                | current_position_target_inst_ids
            )
            fills = await _bounded(
                native_facts.fetch_fill_groups(
                    inst_ids=fact_target_inst_ids,
                    order_ids=target_order_ids,
                    since=since,
                    limit=100,
                    max_pages=max(3, min(10, (self.limit // 100) + 2)),
                    account_wide_only=not bool(fact_target_inst_ids),
                    strict=True,
                ),
                self.timeout_seconds,
            )
            account_order_rows = await _bounded(
                native_facts.fetch_order_history_rows(
                    inst_ids=fact_target_inst_ids,
                    since=since,
                    limit=100,
                    max_pages=1 if fact_target_inst_ids else max(3, min(10, (self.limit // 100) + 2)),
                    strict=True,
                ),
                self.timeout_seconds,
            )
            if fact_target_inst_ids:
                account_order_rows = _dedupe_order_rows(
                    [
                        *account_order_rows,
                        *await _bounded(
                            native_facts.fetch_order_history_rows(
                                since=since,
                                limit=100,
                                max_pages=1,
                                strict=True,
                            ),
                            self.timeout_seconds,
                        ),
                    ]
                )
            account_order_ids = set(_order_rows_by_id(account_order_rows))
            missing_order_ids = sorted(target_order_ids - account_order_ids)
            if fills:
                missing_order_ids = sorted(
                    set(missing_order_ids) - {fill.order_id for fill in fills if fill.order_id}
                )
            target_order_rows = await _bounded(
                native_facts.fetch_order_history_rows(
                    order_ids=missing_order_ids[: min(20, DEFAULT_MAX_ORDER_GAP_QUERIES)],
                    since=since,
                    limit=100,
                    max_pages=1,
                    strict=True,
                ),
                self.timeout_seconds,
            ) if missing_order_ids else []
            order_rows = _dedupe_order_rows([*account_order_rows, *target_order_rows])
            contract_sizes = await _bounded(
                native_facts.fetch_contract_sizes(
                    inst_ids=(
                        {fill.inst_id for fill in fills if fill.inst_id}
                        | _order_rows_inst_ids(order_rows)
                        | _current_position_inst_ids(exchange_positions)
                        | order_target_inst_ids
                        | position_target_inst_ids
                    ),
                ),
                self.timeout_seconds,
            )
            if target_pos_ids or position_target_inst_ids or fact_target_inst_ids or not local_positions:
                try:
                    position_history_rows = await _bounded(
                        native_facts.fetch_position_history_rows(
                            inst_ids=position_target_inst_ids or fact_target_inst_ids,
                            pos_ids=target_pos_ids,
                            since=since,
                            limit=100,
                            max_pages=2,
                            strict=True,
                        ),
                        self.timeout_seconds,
                    )
                except Exception as exc:
                    position_history_error = safe_error_text(exc, limit=180)
                    logger.warning(
                        "OKX position history sync degraded; continuing order/fill fact sync",
                        mode=self.mode,
                        error=position_history_error,
                    )
        except Exception as exc:
            okx_pull_available = False
            pull_error = safe_error_text(exc, limit=180)
            logger.warning(
                "OKX order fact sync failed to pull native facts; continuing stored fact repair",
                mode=self.mode,
                error=pull_error,
            )
        finally:
            try:
                await executor.shutdown()
            except Exception as exc:
                logger.debug("OKX order fact sync shutdown failed", error=safe_error_text(exc))

        fills_by_order_id = {fill.order_id: fill for fill in fills}
        order_rows_by_id = _order_rows_by_id(order_rows)
        async with get_session_ctx() as session:
            writable_orders = await self._load_writable_refresh_orders(session, since_naive)
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
            confirmed_count = 0
            position_confirmed_count = 0
            unverified_count = 0
            skipped_old_count = 0
            samples: list[dict[str, Any]] = []
            backfilled_count = 0
            order_history_backfilled_count = 0
            position_history_result = OkxPositionFactSyncSummary()
            fill_pair_position_result = OkxPositionFactSyncSummary()
            current_position_result = OkxPositionFactSyncSummary()
            account_bill_result = OkxPositionFactSyncSummary()
            if okx_pull_available:
                (
                    confirmed_count,
                    position_confirmed_count,
                    unverified_count,
                    skipped_old_count,
                    samples,
                ) = self._apply_local_order_facts(
                    writable_orders,
                    local_positions=local_positions,
                    exchange_positions=exchange_positions,
                    fills_by_order_id=fills_by_order_id,
                    order_rows_by_id=order_rows_by_id,
                    contract_sizes=contract_sizes,
                    decisions_by_id=decisions_by_id,
                    now=datetime.now(UTC),
                    since=since,
                )
                backfilled_count, order_history_backfilled_count = await self._backfill_okx_only_orders(
                    session,
                    fills=fills,
                    order_rows=order_rows,
                    contract_sizes=contract_sizes,
                    since=since,
                    now=datetime.now(UTC),
                    samples=samples,
                )
                await session.flush()
                position_history_result = await self._sync_position_history_rows(
                    session,
                    position_history_rows=position_history_rows,
                    fills=fills,
                    contract_sizes=contract_sizes,
                    since=since,
                    now=datetime.now(UTC),
                    samples=samples,
                )
                await session.flush()
                fill_pair_position_result = await self._sync_closed_position_fill_pairs(
                    session,
                    fills=fills,
                    contract_sizes=contract_sizes,
                    since=since,
                    now=datetime.now(UTC),
                    samples=samples,
                )
                account_bill_result = await self._sync_account_bills(
                    session,
                    account_bills=account_bills,
                    since=since,
                    now=datetime.now(UTC),
                    samples=samples,
                )
            else:
                confirmed_count = self._recover_local_stored_order_facts(
                    writable_orders,
                    decisions_by_id=decisions_by_id,
                    now=datetime.now(UTC),
                    samples=samples,
                )
            if not okx_pull_available or account_bill_result.checked_count == 0:
                account_bill_result = await self._sync_account_bills(
                    session,
                    account_bills=account_bills,
                    since=since,
                    now=datetime.now(UTC),
                    samples=samples,
                )
            if confirmed_count:
                await session.flush()
            close_fill_repair_result = await self._repair_closed_position_pnl_from_close_fills(
                session,
                fills=fills,
                contract_sizes=contract_sizes,
                since=since,
                now=datetime.now(UTC),
                samples=samples,
            )
            if okx_pull_available:
                current_position_result = await self._sync_current_position_rows(
                    session,
                    exchange_positions=exchange_positions,
                    contract_sizes=contract_sizes,
                    now=datetime.now(UTC),
                    samples=samples,
                )

        status = (
            "warning"
            if unverified_count
            or position_history_error
            or account_bill_error
            or not okx_pull_available
            else "ok"
        )
        return OkxOrderFactSyncSummary(
            status=status,
            mode=self.mode,
            source="okx_native_orders_and_fills",
            phase3_order_sync_start=since,
            checked_at=datetime.now(UTC),
            okx_pull_available=okx_pull_available,
            local_checked=len(writable_orders) if okx_pull_available else len(local_orders),
            confirmed_count=confirmed_count,
            position_confirmed_count=position_confirmed_count,
            unverified_count=unverified_count,
            backfilled_count=backfilled_count,
            order_history_backfilled_count=order_history_backfilled_count,
                position_history_checked_count=position_history_result.checked_count,
                position_history_backfilled_count=position_history_result.backfilled_count,
                position_history_updated_count=position_history_result.updated_count,
                position_history_skipped_count=position_history_result.skipped_count,
                position_history_error=position_history_error,
            current_position_checked_count=current_position_result.checked_count,
            current_position_backfilled_count=current_position_result.backfilled_count,
            current_position_updated_count=current_position_result.updated_count,
            current_position_skipped_count=current_position_result.skipped_count,
            fill_pair_position_checked_count=fill_pair_position_result.checked_count,
            fill_pair_position_backfilled_count=fill_pair_position_result.backfilled_count,
            fill_pair_position_skipped_count=fill_pair_position_result.skipped_count,
            account_bill_checked_count=account_bill_result.checked_count,
            account_bill_backfilled_count=account_bill_result.backfilled_count,
            account_bill_updated_count=account_bill_result.updated_count,
            account_bill_skipped_count=account_bill_result.skipped_count,
            account_bill_error=account_bill_error,
            closed_position_pnl_repair_checked_count=close_fill_repair_result.checked_count,
            closed_position_pnl_repaired_count=close_fill_repair_result.updated_count,
            closed_position_pnl_repair_skipped_count=close_fill_repair_result.skipped_count,
            skipped_old_count=skipped_old_count,
            error=pull_error,
            samples=tuple(samples[:8]),
        ).as_dict()

    def _effective_since(self, now: datetime) -> datetime:
        """Return the Phase 3 clean-order boundary.

        Order facts are an exchange-backed ledger, not a rolling dashboard query.
        Do not move this boundary forward with lookback hours or reset markers,
        otherwise older Phase 3 OKX fills can silently disappear from local facts.
        """

        return _aware_utc(self.phase3_order_sync_start)

    def _load_cold_start_reset_at(self) -> datetime | None:
        marker_path = self.cold_start_marker_path
        if marker_path is None or not marker_path.exists():
            return None
        try:
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if str(payload.get("mode") or "paper") != self.mode:
            return None
        return _parse_datetime(payload.get("reset_at"))

    async def _load_local_orders(self, since_naive: datetime) -> list[Order]:
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
            return list(rows.scalars().all())

    async def _load_local_positions(self, since_naive: datetime) -> list[Position]:
        async with get_session_ctx() as session:
            rows = await session.execute(
                select(Position)
                .where(
                    Position.execution_mode == self.mode,
                    or_(
                        Position.created_at >= since_naive,
                        Position.closed_at >= since_naive,
                        Position.is_open.is_(True),
                    ),
                )
                .order_by(
                    Position.closed_at.desc().nullslast(),
                    Position.created_at.desc().nullslast(),
                    Position.id.desc(),
                )
                .limit(self.limit)
            )
            return list(rows.scalars().all())

    async def _load_writable_refresh_orders(
        self,
        session: Any,
        since_naive: datetime,
    ) -> list[Order]:
        rows = await session.execute(
            select(Order)
            .where(
                Order.execution_mode == self.mode,
                or_(Order.created_at >= since_naive, Order.filled_at >= since_naive),
            )
            .order_by(Order.filled_at.desc().nullslast(), Order.created_at.desc())
            .limit(self.limit)
        )
        return [
            order for order in rows.scalars().all() if _order_needs_okx_fact_refresh(order)
        ]

    def _apply_local_order_facts(
        self,
        orders: list[Order],
        *,
        local_positions: list[Position],
        exchange_positions: list[dict[str, Any]],
        fills_by_order_id: dict[str, OkxNativeFillGroup],
        order_rows_by_id: dict[str, dict[str, Any]],
        contract_sizes: dict[str, float],
        decisions_by_id: dict[int, AIDecision] | None = None,
        now: datetime,
        since: datetime,
    ) -> tuple[int, int, int, int, list[dict[str, Any]]]:
        confirmed_count = 0
        position_confirmed_count = 0
        unverified_count = 0
        skipped_old_count = 0
        samples: list[dict[str, Any]] = []
        for order in orders:
            order_time = _order_time(order)
            if order_time is not None and order_time < since:
                skipped_old_count += 1
                continue
            exchange_ids = _split_exchange_order_ids(getattr(order, "exchange_order_id", None))
            fill = next(
                (fills_by_order_id[exchange_id] for exchange_id in exchange_ids if exchange_id in fills_by_order_id),
                None,
            )
            order_row = next(
                (order_rows_by_id[exchange_id] for exchange_id in exchange_ids if exchange_id in order_rows_by_id),
                None,
            )
            if fill is None:
                position_confirmation = None
                for exchange_id in exchange_ids:
                    candidate = find_current_position_entry_confirmation(
                        order,
                        exchange_order_id=exchange_id,
                        exchange_positions=exchange_positions,
                        local_positions=local_positions,
                        contract_sizes=contract_sizes,
                    )
                    if candidate is not None:
                        position_confirmation = candidate
                        break
                if (
                    position_confirmation is not None
                    and str(getattr(order, "status", "") or "").lower() == "filled"
                ):
                    _apply_position_confirmation_to_order(
                        order,
                        position_confirmation,
                        now=now,
                    )
                    position_confirmed_count += 1
                    samples.append(_sample(order, kind="local_order_position_confirmed"))
                    continue
                decision = (decisions_by_id or {}).get(int(getattr(order, "decision_id", 0) or 0))
                if _recover_okx_execution_result_fact_from_decision(order, decision):
                    _apply_execution_result_confirmation_to_order(order, now=now)
                    confirmed_count += 1
                    samples.append(_sample(order, kind="local_order_execution_result_recovered"))
                    continue
                if _recover_okx_close_fill_fact_from_decision(order, decision):
                    _apply_close_fill_confirmation_to_order(order, now=now)
                    confirmed_count += 1
                    samples.append(_sample(order, kind="local_order_close_fill_recovered"))
                    continue
                if _order_has_authoritative_stored_okx_fill_fact(order):
                    _apply_close_fill_confirmation_to_order(order, now=now)
                    confirmed_count += 1
                    samples.append(_sample(order, kind="local_order_stored_fill_repaired"))
                    continue
                if _order_has_okx_execution_result_fact(order):
                    _apply_execution_result_confirmation_to_order(order, now=now)
                    confirmed_count += 1
                    samples.append(_sample(order, kind="local_order_execution_result_confirmed"))
                    continue
                if str(getattr(order, "status", "") or "").lower() == "filled":
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
            self._apply_fill_to_order(
                order,
                fill,
                now=now,
                sync_status=OKX_SYNC_CONFIRMED,
                contract_size=_contract_size_for_fill(fill, contract_sizes),
                order_row=order_row,
            )
            confirmed_count += 1
            samples.append(_sample(order, kind="local_order_confirmed"))
        return (
            confirmed_count,
            position_confirmed_count,
            unverified_count,
            skipped_old_count,
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
                _apply_execution_result_confirmation_to_order(order, now=now)
                confirmed_count += 1
                samples.append(_sample(order, kind="local_order_execution_result_recovered"))
                continue
            if _recover_okx_close_fill_fact_from_decision(order, decision):
                _apply_close_fill_confirmation_to_order(order, now=now)
                confirmed_count += 1
                samples.append(_sample(order, kind="local_order_close_fill_recovered"))
                continue
            if _order_has_okx_execution_result_fact(order):
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
        contract_sizes: dict[str, float],
        since: datetime,
        now: datetime,
        samples: list[dict[str, Any]],
    ) -> tuple[int, int]:
        backfilled = 0
        order_history_backfilled = 0
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
            order = Order(
                model_name="okx_authoritative_sync",
                execution_mode=self.mode,
                symbol=fill.symbol,
                side=fill.side,
                order_type=_fill_order_type(fill),
                quantity=_fill_base_quantity(fill, _contract_size_for_fill(fill, contract_sizes)),
                price=fill.avg_price,
                status="filled",
                fee=fill.fee_abs,
                decision_id=None,
                exchange_order_id=fill.order_id,
                filled_at=fill.timestamp or now,
                created_at=fill.timestamp or now,
            )
            self._apply_fill_to_order(
                order,
                fill,
                now=now,
                sync_status=OKX_SYNC_OKX_ONLY,
                contract_size=_contract_size_for_fill(fill, contract_sizes),
                order_row=order_rows_by_id.get(fill.order_id),
            )
            session.add(order)
            existing_exchange_ids.add(fill.order_id)
            backfilled += 1
            samples.append(_sample(order, kind="okx_only_backfilled"))
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
        return backfilled, order_history_backfilled

    async def _sync_position_history_rows(
        self,
        session: Any,
        *,
        position_history_rows: list[dict[str, Any]],
        fills: list[OkxNativeFillGroup],
        contract_sizes: dict[str, float],
        since: datetime,
        now: datetime,
        samples: list[dict[str, Any]],
    ) -> OkxPositionFactSyncSummary:
        checked = 0
        backfilled = 0
        updated = 0
        skipped = 0
        suppressions = await self._load_position_sync_suppressions(session)
        fills_by_inst_side = _fills_by_inst_and_position_side(fills)
        for row in position_history_rows:
            position_time = _position_history_time(row)
            if position_time is not None and _aware_utc(position_time) < since:
                skipped += 1
                continue
            inst_id = _position_history_inst_id(row)
            if not inst_id:
                skipped += 1
                continue
            checked += 1
            pos_id = _position_history_pos_id(row)
            side = _position_history_side(row)
            created_at = _position_history_opened_at(row) or position_time or now
            closed_at = _position_history_closed_at(row) or position_time or now
            close_fills = _matching_position_history_close_fills(
                row,
                fills_by_inst_side=fills_by_inst_side,
            )
            entry_fills = _matching_position_history_entry_fills(
                row,
                fills_by_inst_side=fills_by_inst_side,
            )
            if not close_fills and not entry_fills:
                entry_fills, close_fills = _infer_net_position_history_fills(
                    row,
                    fills_by_inst_side=fills_by_inst_side,
                )
            close_order_ids = _ordered_exchange_ids(fill.order_id for fill in close_fills)
            entry_order_ids = _ordered_exchange_ids(fill.order_id for fill in entry_fills)
            inferred_side = _infer_position_history_side(row, entry_fills=entry_fills, close_fills=close_fills)
            if inferred_side:
                side = inferred_side
                if not close_fills:
                    close_fills = _matching_position_history_close_fills(
                        row,
                        fills_by_inst_side=fills_by_inst_side,
                        side=side,
                    )
                if not entry_fills:
                    entry_fills = _matching_position_history_entry_fills(
                        row,
                        fills_by_inst_side=fills_by_inst_side,
                        side=side,
                    )
                close_order_ids = _ordered_exchange_ids(fill.order_id for fill in close_fills)
                entry_order_ids = _ordered_exchange_ids(fill.order_id for fill in entry_fills)
            if side:
                lifecycle_entry_fills, lifecycle_close_fills = _position_history_lifecycle_fills(
                    row,
                    fills_by_inst_side=fills_by_inst_side,
                    side=side,
                )
                if lifecycle_entry_fills:
                    entry_order_ids = _ordered_exchange_ids(fill.order_id for fill in lifecycle_entry_fills)
                    entry_fills = lifecycle_entry_fills
                if lifecycle_close_fills:
                    close_order_ids = _ordered_exchange_ids(fill.order_id for fill in lifecycle_close_fills)
                    close_fills = lifecycle_close_fills
            suppression = _matching_position_sync_suppression(
                suppressions,
                mode=self.mode,
                symbol=symbol_from_okx_inst_id(inst_id) or normalize_trading_symbol(inst_id),
                side=side,
                okx_inst_id=inst_id,
                okx_pos_id=pos_id,
                entry_order_ids=entry_order_ids,
                close_order_ids=close_order_ids,
                created_at=created_at,
                closed_at=closed_at,
            )
            if suppression is not None:
                skipped += 1
                samples.append(
                    {
                        "kind": "okx_position_history_suppressed",
                        "symbol": symbol_from_okx_inst_id(inst_id) or normalize_trading_symbol(inst_id),
                        "side": side,
                        "okx_pos_id": pos_id,
                        "entry_exchange_order_id": ",".join(entry_order_ids) or None,
                        "close_exchange_order_id": ",".join(close_order_ids) or None,
                        "reason": suppression.reason,
                    }
                )
                continue
            payload = _position_from_history_row(
                row,
                mode=self.mode,
                now=now,
                contract_size=_contract_size_for_inst_id(inst_id, contract_sizes),
                entry_order_ids=entry_order_ids,
                close_order_ids=close_order_ids,
                created_at=created_at,
                closed_at=closed_at,
                side=side,
            )
            existing = await self._find_position_history_cache_row(
                session,
                pos_id=pos_id,
                inst_id=inst_id,
                side=side,
                created_at=created_at,
                closed_at=closed_at,
                entry_order_ids=entry_order_ids,
                close_order_ids=close_order_ids,
            )
            if existing is None:
                session.add(Position(**payload))
                backfilled += 1
                samples.append(
                    {
                        "kind": "okx_position_history_backfilled",
                        "symbol": payload["symbol"],
                        "side": payload["side"],
                        "okx_pos_id": payload.get("okx_pos_id"),
                    }
                )
                continue
            _apply_position_history_payload(existing, payload, now=now)
            updated += 1
            samples.append(
                {
                    "kind": "okx_position_history_updated",
                    "local_position_id": getattr(existing, "id", None),
                    "symbol": getattr(existing, "symbol", None),
                    "side": getattr(existing, "side", None),
                    "okx_pos_id": getattr(existing, "okx_pos_id", None),
                }
            )
        return OkxPositionFactSyncSummary(
            checked_count=checked,
            backfilled_count=backfilled,
            updated_count=updated,
            skipped_count=skipped,
            samples=tuple(samples[-8:]),
        )

    async def _sync_closed_position_fill_pairs(
        self,
        session: Any,
        *,
        fills: list[OkxNativeFillGroup],
        contract_sizes: dict[str, float],
        since: datetime,
        now: datetime,
        samples: list[dict[str, Any]],
    ) -> OkxPositionFactSyncSummary:
        linked_order_ids = await self._load_linked_position_order_ids(session)
        suppressions = await self._load_position_sync_suppressions(session)
        unlinked_fills = [
            fill
            for fill in sorted(fills, key=lambda item: item.timestamp or datetime.min.replace(tzinfo=UTC))
            if fill.order_id
            and fill.order_id not in linked_order_ids
            and fill.timestamp is not None
            and _aware_utc(fill.timestamp) >= since
        ]
        checked = 0
        backfilled = 0
        skipped = 0
        for entry_fill, close_fill, side in _closed_position_fill_pair_candidates(unlinked_fills):
            checked += 1
            if entry_fill.order_id in linked_order_ids or close_fill.order_id in linked_order_ids:
                skipped += 1
                continue
            suppression = _matching_position_sync_suppression(
                suppressions,
                mode=self.mode,
                symbol=symbol_from_okx_inst_id(entry_fill.inst_id) or entry_fill.symbol,
                side=side,
                okx_inst_id=entry_fill.inst_id,
                okx_pos_id="",
                entry_order_ids=[entry_fill.order_id],
                close_order_ids=[close_fill.order_id],
                created_at=entry_fill.timestamp,
                closed_at=close_fill.timestamp,
            )
            if suppression is not None:
                skipped += 1
                linked_order_ids.add(entry_fill.order_id)
                linked_order_ids.add(close_fill.order_id)
                samples.append(
                    {
                        "kind": "okx_fill_pair_position_suppressed",
                        "symbol": symbol_from_okx_inst_id(entry_fill.inst_id) or entry_fill.symbol,
                        "side": side,
                        "entry_exchange_order_id": entry_fill.order_id,
                        "close_exchange_order_id": close_fill.order_id,
                        "reason": suppression.reason,
                    }
                )
                continue
            payload = _position_from_fill_pair(
                entry_fill,
                close_fill,
                side=side,
                mode=self.mode,
                now=now,
                contract_size=_contract_size_for_fill(entry_fill, contract_sizes),
            )
            if await self._closed_position_with_links_exists(
                session,
                entry_order_id=str(payload.get("entry_exchange_order_id") or ""),
                close_order_id=str(payload.get("close_exchange_order_id") or ""),
            ):
                skipped += 1
                linked_order_ids.update(
                    _split_exchange_order_ids(payload.get("entry_exchange_order_id"))
                    | _split_exchange_order_ids(payload.get("close_exchange_order_id"))
                )
                continue
            session.add(Position(**payload))
            linked_order_ids.add(entry_fill.order_id)
            linked_order_ids.add(close_fill.order_id)
            backfilled += 1
            samples.append(
                {
                    "kind": "okx_fill_pair_position_backfilled",
                    "symbol": payload["symbol"],
                    "side": payload["side"],
                    "entry_exchange_order_id": payload.get("entry_exchange_order_id"),
                    "close_exchange_order_id": payload.get("close_exchange_order_id"),
                }
            )
        return OkxPositionFactSyncSummary(
            checked_count=checked,
            backfilled_count=backfilled,
            skipped_count=skipped,
            samples=tuple(samples[-8:]),
        )

    async def _sync_account_bills(
        self,
        session: Any,
        *,
        account_bills: list[OkxNativeAccountBill],
        since: datetime,
        now: datetime,
        samples: list[dict[str, Any]],
    ) -> OkxPositionFactSyncSummary:
        checked = 0
        backfilled = 0
        updated = 0
        skipped = 0
        for bill in account_bills:
            bill_time = _db_datetime_to_utc(bill.timestamp)
            if bill_time is None or bill_time < since:
                skipped += 1
                continue
            bill_id = str(bill.bill_id or "").strip()
            if not bill_id:
                skipped += 1
                continue
            checked += 1
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
                "source": "okx_account_bills",
                "raw_bill": dict(bill.raw),
            }
            if existing is None:
                session.add(OkxAccountBill(**payload))
                backfilled += 1
                samples.append(
                    {
                        "kind": "okx_account_bill_backfilled",
                        "bill_id": bill_id,
                        "inst_id": bill.inst_id,
                        "funding_fee": bill.funding_fee,
                    }
                )
                continue
            changed = False
            for key, value in payload.items():
                if key in {"mode", "bill_id"}:
                    continue
                if getattr(existing, key) != value:
                    setattr(existing, key, value)
                    changed = True
            if changed:
                existing.updated_at = now
                updated += 1
            else:
                skipped += 1
        return OkxPositionFactSyncSummary(
            checked_count=checked,
            backfilled_count=backfilled,
            updated_count=updated,
            skipped_count=skipped,
            samples=tuple(samples[-8:]),
        )

    async def _repair_closed_position_pnl_from_close_fills(
        self,
        session: Any,
        *,
        fills: list[OkxNativeFillGroup],
        contract_sizes: dict[str, float],
        since: datetime,
        now: datetime,
        samples: list[dict[str, Any]],
    ) -> OkxPositionFactSyncSummary:
        fills_by_order_id = {fill.order_id: fill for fill in fills if fill.order_id}
        since_naive = _db_naive_since(since)
        rows = await session.execute(
            select(Position)
            .where(
                Position.execution_mode == self.mode,
                Position.is_open.is_(False),
                Position.close_exchange_order_id.is_not(None),
                Position.close_exchange_order_id != "",
                or_(Position.closed_at >= since_naive, Position.created_at >= since_naive),
            )
            .order_by(Position.closed_at.desc().nullslast(), Position.id.desc())
            .limit(self.limit)
        )
        positions = list(rows.scalars().all())
        if not positions:
            return OkxPositionFactSyncSummary()
        order_ids: set[str] = set()
        for position in positions:
            order_ids.update(_split_exchange_order_ids(getattr(position, "entry_exchange_order_id", None)))
            order_ids.update(_split_exchange_order_ids(getattr(position, "close_exchange_order_id", None)))
        orders_by_id: dict[str, Order] = {}
        if order_ids:
            order_rows = await session.execute(
                select(Order).where(
                    Order.execution_mode == self.mode,
                    Order.exchange_order_id.in_(sorted(order_ids)),
                )
            )
            orders_by_id = {
                str(getattr(order, "exchange_order_id", "") or "").strip(): order
                for order in order_rows.scalars().all()
                if str(getattr(order, "exchange_order_id", "") or "").strip()
            }
        if not fills_by_order_id and not orders_by_id:
            return OkxPositionFactSyncSummary()
        checked = 0
        updated = 0
        skipped = 0
        for position in positions:
            if _is_authoritative_position_history_position(position):
                skipped += 1
                continue
            close_ids = _split_exchange_order_ids(getattr(position, "close_exchange_order_id", None))
            if not close_ids:
                skipped += 1
                continue
            checked += 1
            repaired = _closed_position_realized_pnl_from_close_facts(
                position,
                close_ids=close_ids,
                fills_by_order_id=fills_by_order_id,
                contract_sizes=contract_sizes,
                orders_by_id=orders_by_id,
            )
            if repaired is None:
                skipped += 1
                continue
            current = _safe_float(getattr(position, "realized_pnl", None), 0.0)
            if abs(current - repaired) <= 0.000001:
                skipped += 1
                continue
            position.realized_pnl = repaired
            position.updated_at = now
            updated += 1
            samples.append(
                {
                    "kind": "closed_position_pnl_repaired_from_okx_close_fill",
                    "local_position_id": getattr(position, "id", None),
                    "symbol": getattr(position, "symbol", None),
                    "side": getattr(position, "side", None),
                    "old_realized_pnl": current,
                    "realized_pnl": repaired,
                    "close_exchange_order_id": getattr(position, "close_exchange_order_id", None),
                }
            )
        return OkxPositionFactSyncSummary(
            checked_count=checked,
            updated_count=updated,
            skipped_count=skipped,
            samples=tuple(samples[-8:]),
        )

    async def _load_linked_position_order_ids(self, session: Any) -> set[str]:
        rows = await session.execute(
            select(Position.entry_exchange_order_id, Position.close_exchange_order_id).where(
                Position.execution_mode == self.mode
            )
        )
        linked: set[str] = set()
        for entry_ids, close_ids in rows.all():
            linked.update(_split_exchange_order_ids(entry_ids))
            linked.update(_split_exchange_order_ids(close_ids))
        return linked

    async def _closed_position_with_links_exists(
        self,
        session: Any,
        *,
        entry_order_id: str,
        close_order_id: str,
    ) -> bool:
        if not entry_order_id and not close_order_id:
            return False
        conditions = [Position.execution_mode == self.mode, Position.is_open.is_(False)]
        if entry_order_id:
            conditions.append(Position.entry_exchange_order_id == entry_order_id)
        if close_order_id:
            conditions.append(Position.close_exchange_order_id == close_order_id)
        result = await session.execute(select(Position.id).where(*conditions).limit(1))
        return result.scalar_one_or_none() is not None

    async def _load_position_sync_suppressions(
        self,
        session: Any,
    ) -> tuple[OkxPositionSyncSuppression, ...]:
        rows = await session.execute(
            select(StrategyLearningEvent)
            .where(
                StrategyLearningEvent.execution_mode == self.mode,
                StrategyLearningEvent.event_type == OKX_POSITION_SYNC_SUPPRESSION_EVENT_TYPE,
                StrategyLearningEvent.event_status == "active",
            )
            .order_by(StrategyLearningEvent.updated_at.desc().nullslast(), StrategyLearningEvent.id.desc())
            .limit(200)
        )
        suppressions: list[OkxPositionSyncSuppression] = []
        for event in rows.scalars().all():
            suppression = _position_sync_suppression_from_event(event)
            if suppression is not None and suppression.has_strong_identity():
                suppressions.append(suppression)
        return tuple(suppressions)

    async def _sync_current_position_rows(
        self,
        session: Any,
        *,
        exchange_positions: list[dict[str, Any]],
        contract_sizes: dict[str, float],
        now: datetime,
        samples: list[dict[str, Any]],
    ) -> OkxPositionFactSyncSummary:
        checked = 0
        backfilled = 0
        updated = 0
        skipped = 0
        candidate_orders = await self._load_current_position_entry_link_orders(session)
        for row in exchange_positions:
            inst_id = _current_position_inst_id(row)
            side = _current_position_side(row)
            contracts = _current_position_contracts(row)
            if not inst_id or side not in {"long", "short"} or contracts <= 0:
                skipped += 1
                continue
            checked += 1
            payload = _position_from_current_row(
                row,
                mode=self.mode,
                now=now,
                contract_size=_current_position_contract_size(row, contract_sizes),
            )
            entry_order = _matching_current_position_entry_order(payload, candidate_orders)
            if entry_order is not None:
                payload["entry_exchange_order_id"] = str(
                    getattr(entry_order, "exchange_order_id", "") or ""
                ).strip() or None
            existing = await self._find_current_position_cache_row(
                session,
                pos_id=str(payload.get("okx_pos_id") or ""),
                inst_id=inst_id,
                side=side,
            )
            if existing is None:
                session.add(Position(**payload))
                backfilled += 1
                samples.append(
                    {
                        "kind": "okx_current_position_backfilled",
                        "symbol": payload["symbol"],
                        "side": payload["side"],
                        "okx_pos_id": payload.get("okx_pos_id"),
                        "entry_exchange_order_id": payload.get("entry_exchange_order_id"),
                    }
                )
                continue
            _apply_current_position_payload(existing, payload, now=now)
            updated += 1
            samples.append(
                {
                    "kind": "okx_current_position_updated",
                    "local_position_id": getattr(existing, "id", None),
                    "symbol": getattr(existing, "symbol", None),
                    "side": getattr(existing, "side", None),
                    "okx_pos_id": getattr(existing, "okx_pos_id", None),
                    "entry_exchange_order_id": getattr(existing, "entry_exchange_order_id", None),
                }
            )
        return OkxPositionFactSyncSummary(
            checked_count=checked,
            backfilled_count=backfilled,
            updated_count=updated,
            skipped_count=skipped,
            samples=tuple(samples[-8:]),
        )

    async def _load_current_position_entry_link_orders(
        self,
        session: Any,
    ) -> list[Order]:
        window_start = _db_naive_since(PHASE3_DEFAULT_ORDER_SYNC_START) - timedelta(
            seconds=CURRENT_POSITION_ENTRY_LINK_WINDOW_SECONDS
        )
        rows = await session.execute(
            select(Order)
            .where(
                Order.execution_mode == self.mode,
                Order.status == "filled",
                Order.exchange_order_id.is_not(None),
                Order.exchange_order_id != "",
                or_(
                    Order.created_at >= window_start,
                    Order.filled_at >= window_start,
                    Order.okx_synced_at >= window_start,
                ),
            )
            .order_by(Order.filled_at.desc().nullslast(), Order.created_at.desc())
            .limit(self.limit * 3)
        )
        return [order for order in rows.scalars().all() if not is_manual_close_order(order)]

    async def _find_current_position_cache_row(
        self,
        session: Any,
        *,
        pos_id: str,
        inst_id: str,
        side: str,
    ) -> Position | None:
        if pos_id:
            result = await session.execute(
                select(Position)
                .where(
                    Position.execution_mode == self.mode,
                    Position.okx_pos_id == pos_id,
                    Position.is_open.is_(True),
                )
                .order_by(Position.updated_at.desc().nullslast(), Position.id.desc())
                .limit(100)
            )
            existing = _best_current_position_cache_row(result.scalars().all())
            if existing is not None:
                return existing
        result = await session.execute(
            select(Position)
            .where(
                Position.execution_mode == self.mode,
                Position.okx_inst_id == inst_id,
                Position.side == side,
                Position.is_open.is_(True),
            )
            .order_by(Position.updated_at.desc().nullslast(), Position.id.desc())
            .limit(100)
        )
        return _best_current_position_cache_row(result.scalars().all())

    async def _find_position_history_cache_row(
        self,
        session: Any,
        *,
        pos_id: str,
        inst_id: str,
        side: str,
        created_at: datetime,
        closed_at: datetime,
        entry_order_ids: list[str],
        close_order_ids: list[str],
    ) -> Position | None:
        if pos_id:
            result = await session.execute(
                select(Position)
                .where(
                    Position.execution_mode == self.mode,
                    Position.okx_pos_id == pos_id,
                    Position.okx_inst_id == inst_id,
                    Position.side == side,
                    Position.is_open.is_(False),
                )
                .order_by(Position.updated_at.desc().nullslast(), Position.id.desc())
                .limit(100)
            )
            existing = _best_position_history_lifecycle_match(
                result.scalars().all(),
                created_at=created_at,
                closed_at=closed_at,
                entry_order_ids=entry_order_ids,
                close_order_ids=close_order_ids,
            )
            if existing is not None:
                return existing
        window_start = closed_at - timedelta(seconds=3)
        window_end = closed_at + timedelta(seconds=3)
        result = await session.execute(
            select(Position)
            .where(
                Position.execution_mode == self.mode,
                Position.okx_inst_id == inst_id,
                Position.side == side,
                Position.is_open.is_(False),
                Position.closed_at >= window_start,
                Position.closed_at <= window_end,
            )
            .order_by(Position.updated_at.desc().nullslast(), Position.id.desc())
            .limit(1)
        )
        return _best_position_history_lifecycle_match(
            result.scalars().all(),
            created_at=created_at,
            closed_at=closed_at,
            entry_order_ids=entry_order_ids,
            close_order_ids=close_order_ids,
        )

    @staticmethod
    def _apply_fill_to_order(
        order: Order,
        fill: OkxNativeFillGroup,
        *,
        now: datetime,
        sync_status: str,
        contract_size: float = 0.0,
        order_row: dict[str, Any] | None = None,
    ) -> None:
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
        order.okx_raw_fills = {
            "fills_history_confirmed": True,
            "order_id": fill.order_id,
            "trade_ids": list(fill.trade_ids),
            "inst_id": fill.inst_id,
            "pos_side": fill.pos_side,
            "contracts": fill.contracts,
            "contract_size": contract_size or None,
            "base_quantity": _fill_base_quantity(fill, contract_size),
            "avg_price": fill.avg_price,
            "fee_abs": fill.fee_abs,
            "fill_pnl": fill.fill_pnl,
            "timestamp": fill.timestamp.isoformat() if fill.timestamp else None,
            "rows": list(fill.rows[:20]),
            "order_rows": [dict(order_row)] if isinstance(order_row, dict) and order_row else [],
        }


def _apply_position_confirmation_to_order(
    order: Order,
    confirmation: OkxCurrentPositionEntryConfirmation,
    *,
    now: datetime,
) -> None:
    _clear_unconfirmed_fill_fact(order)
    order.exchange_order_id = confirmation.exchange_order_id
    order.okx_inst_id = confirmation.inst_id
    order.symbol = symbol_from_okx_inst_id(confirmation.inst_id) or confirmation.symbol
    if confirmation.side:
        order.side = confirmation.side
    if confirmation.base_quantity > 0:
        order.quantity = confirmation.base_quantity
    if confirmation.avg_price > 0:
        order.price = confirmation.avg_price
    if confirmation.fee_abs is not None:
        order.fee = confirmation.fee_abs
    order.status = "filled"
    order.filled_at = getattr(order, "filled_at", None) or confirmation.timestamp or now
    order.okx_trade_ids = None
    order.okx_fill_contracts = None
    order.okx_fill_pnl = None
    order.okx_state = "open_position_confirmed"
    order.okx_sync_status = OKX_SYNC_POSITION_CONFIRMED
    order.okx_synced_at = now
    order.okx_last_error = None
    order.okx_raw_fills = confirmation.as_raw_payload()


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
    if avg_price > 0:
        order.price = avg_price
    if fee_abs >= 0:
        order.fee = fee_abs
    order.okx_trade_ids = ",".join(trade_ids) if trade_ids else None
    order.okx_fill_pnl = fill_pnl
    order.okx_state = "execution_result_confirmed"
    order.okx_sync_status = OKX_SYNC_EXECUTION_RESULT_CONFIRMED
    order.okx_synced_at = now
    order.okx_last_error = None
    raw["execution_result_confirmed"] = True
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


def _order_inst_id(order: Order) -> str:
    inst_id = str(getattr(order, "okx_inst_id", "") or "").strip().upper()
    if inst_id:
        return inst_id
    return okx_inst_id_from_symbol(getattr(order, "symbol", None)) or ""


def _position_inst_id(position: Position) -> str:
    inst_id = str(getattr(position, "okx_inst_id", "") or "").strip().upper()
    if inst_id:
        return inst_id
    return okx_inst_id_from_symbol(getattr(position, "symbol", None)) or ""


def _is_authoritative_position_history_position(position: Position) -> bool:
    return (
        str(getattr(position, "model_name", "") or "").strip() == "okx_authoritative_sync"
        and bool(str(getattr(position, "okx_pos_id", "") or "").strip())
        and not bool(getattr(position, "is_open", False))
    )


def _order_needs_okx_fact_refresh(order: Order) -> bool:
    exchange_ids = _split_exchange_order_ids(getattr(order, "exchange_order_id", None))
    if not exchange_ids:
        return _order_is_rejected_without_exchange_fill(order)
    status = str(getattr(order, "status", "") or "").lower().strip()
    sync_status = str(getattr(order, "okx_sync_status", "") or "").strip()
    if sync_status == OKX_SYNC_CONFIRMED and _order_has_confirmed_okx_fill_fact(order):
        return False
    if sync_status == OKX_SYNC_OKX_ONLY and _order_has_confirmed_okx_fill_fact(order):
        return False
    if sync_status == OKX_SYNC_NO_FILL_REJECTED and status in {
        "rejected",
        "failed",
        "error",
        "cancelled",
        "canceled",
    }:
        return False
    return status in {
        "filled",
        "partial",
        "rejected",
        "failed",
        "error",
        "cancelled",
        "canceled",
    }


def _order_needs_okx_pull(order: Order) -> bool:
    exchange_ids = _split_exchange_order_ids(getattr(order, "exchange_order_id", None))
    if not exchange_ids:
        return False
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
    if _order_has_fills_history_confirmed(order) and _order_has_confirmed_okx_fill_fact(order):
        return False
    return True


def _order_has_fills_history_confirmed(order: Order) -> bool:
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    return bool(raw.get("fills_history_confirmed"))


def _order_has_confirmed_okx_fill_fact(order: Order) -> bool:
    if not _order_has_authoritative_stored_okx_fill_fact(order):
        return False
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    expected_quantity = _stored_fill_base_quantity(raw)
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
    if expected_fee >= 0 and local_fee >= 0 and not _relative_close_enough(
        local_fee,
        expected_fee,
        0.001,
    ):
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
    if not str(getattr(order, "okx_inst_id", "") or raw.get("inst_id") or "").strip():
        return False
    if _safe_float(getattr(order, "okx_fill_contracts", None) or raw.get("contracts"), 0.0) <= 0:
        return False
    if _safe_float(raw.get("avg_price") or raw.get("average") or getattr(order, "price", None), 0.0) <= 0:
        return False
    return True


def sync_status_is_confirmed(value: Any) -> bool:
    return str(value or "").strip() in {
        OKX_SYNC_CONFIRMED,
        OKX_SYNC_OKX_ONLY,
        OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
    }


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
    if size > 0:
        return size
    size = _safe_float(row.get("ctVal") or row.get("contractSize"), 0.0)
    return size if size > 0 else 1.0


def _base_quantity_from_order_row(row: dict[str, Any], contract_size: float) -> float:
    contracts = _order_row_contracts(row)
    size = _safe_float(contract_size, 0.0)
    return contracts * (size if size > 0 else 1.0)


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
        "base_quantity": _base_quantity_from_order_row(row, contract_size),
        "avg_price": _order_row_price(row),
        "order_rows": [dict(row)],
        "rows": [],
    }
    return order


def _position_from_history_row(
    row: dict[str, Any],
    *,
    mode: str,
    now: datetime,
    contract_size: float,
    entry_order_ids: list[str],
    close_order_ids: list[str],
    created_at: datetime,
    closed_at: datetime,
    side: str | None = None,
) -> dict[str, Any]:
    inst_id = _position_history_inst_id(row)
    side = side or _position_history_side(row)
    contracts = _position_history_contracts(row)
    quantity = contracts * (contract_size if contract_size > 0 else 1.0)
    entry_price = _safe_float(row.get("openAvgPx") or row.get("avgPx"), 0.0)
    close_price = _safe_float(row.get("closeAvgPx") or row.get("closePx"), 0.0)
    realized_pnl = _safe_float(row.get("realizedPnl"), 0.0)
    return {
        "model_name": "okx_authoritative_sync",
        "execution_mode": mode,
        "symbol": symbol_from_okx_inst_id(inst_id) or normalize_trading_symbol(inst_id),
        "side": side,
        "quantity": quantity,
        "entry_price": entry_price,
        "current_price": close_price or entry_price,
        "leverage": _position_history_leverage(row),
        "unrealized_pnl": 0.0,
        "realized_pnl": realized_pnl,
        "is_open": False,
        "closed_at": closed_at,
        "created_at": created_at,
        "updated_at": now,
        "okx_inst_id": inst_id,
        "okx_pos_id": _position_history_pos_id(row),
        "entry_exchange_order_id": ",".join(entry_order_ids) or None,
        "close_exchange_order_id": ",".join(close_order_ids) or None,
    }


def _position_from_fill_pair(
    entry_fill: OkxNativeFillGroup,
    close_fill: OkxNativeFillGroup,
    *,
    side: str,
    mode: str,
    now: datetime,
    contract_size: float,
) -> dict[str, Any]:
    quantity = min(
        _fill_base_quantity(entry_fill, contract_size),
        _fill_base_quantity(close_fill, contract_size),
    )
    return {
        "model_name": "okx_authoritative_sync",
        "execution_mode": mode,
        "symbol": symbol_from_okx_inst_id(entry_fill.inst_id) or entry_fill.symbol,
        "side": side,
        "quantity": quantity,
        "entry_price": entry_fill.avg_price,
        "current_price": close_fill.avg_price,
        "leverage": 1.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": (
            _safe_float(entry_fill.fill_pnl, 0.0)
            + _safe_float(close_fill.fill_pnl, 0.0)
            - _safe_float(entry_fill.fee_abs, 0.0)
            - _safe_float(close_fill.fee_abs, 0.0)
        ),
        "is_open": False,
        "closed_at": close_fill.timestamp or now,
        "created_at": entry_fill.timestamp or now,
        "updated_at": now,
        "okx_inst_id": entry_fill.inst_id,
        "okx_pos_id": None,
        "entry_exchange_order_id": entry_fill.order_id,
        "close_exchange_order_id": close_fill.order_id,
    }


def _closed_position_realized_pnl_from_close_facts(
    position: Position,
    *,
    close_ids: set[str],
    fills_by_order_id: dict[str, OkxNativeFillGroup],
    contract_sizes: dict[str, float],
    orders_by_id: dict[str, Order],
) -> float | None:
    quantity = abs(_safe_float(getattr(position, "quantity", None), 0.0))
    if quantity <= 0 or not close_ids:
        return None
    close_gross_pnl = 0.0
    close_fee = 0.0
    close_quantity = 0.0
    for close_id in sorted(close_ids):
        fill = fills_by_order_id.get(close_id)
        if fill is not None:
            contract_size = _contract_size_for_fill(fill, contract_sizes)
            fill_quantity = _fill_base_quantity(fill, contract_size)
            fill_pnl = _safe_float(fill.fill_pnl, 0.0)
            fee_abs = _safe_float(fill.fee_abs, 0.0)
        else:
            order = orders_by_id.get(close_id)
            if order is None or not _order_has_confirmed_okx_fill_fact(order):
                return None
            raw = getattr(order, "okx_raw_fills", None)
            raw = raw if isinstance(raw, dict) else {}
            raw_has_fill_pnl = raw.get("fill_pnl") is not None or getattr(order, "okx_fill_pnl", None) is not None
            if not raw_has_fill_pnl:
                return None
            fill_quantity = _safe_float(raw.get("base_quantity") or getattr(order, "quantity", None), 0.0)
            fill_pnl = _safe_float(raw.get("fill_pnl") or getattr(order, "okx_fill_pnl", None), 0.0)
            fee_abs = _safe_float(raw.get("fee_abs") or getattr(order, "fee", None), 0.0)
        if fill_quantity <= 0:
            return None
        close_quantity += fill_quantity
        close_fee += fee_abs
        close_gross_pnl += fill_pnl
    if close_quantity <= 0:
        return None
    if close_quantity > 0 and abs(close_quantity - quantity) > max(quantity * 0.02, 1e-9):
        # Multi-slice rows need a complete close-fill set; otherwise wait for
        # OKX position-history realizedPnl instead of partially overwriting PnL.
        return None
    entry_fee = _entry_fee_from_linked_orders(position, orders_by_id, close_quantity=quantity)
    return close_gross_pnl - entry_fee - close_fee


def _entry_fee_from_linked_orders(
    position: Position,
    orders_by_id: dict[str, Order],
    *,
    close_quantity: float,
) -> float:
    entry_ids = _split_exchange_order_ids(getattr(position, "entry_exchange_order_id", None))
    if not entry_ids:
        return 0.0
    entry_orders = [
        orders_by_id[order_id] for order_id in entry_ids if order_id in orders_by_id
    ]
    if not entry_orders:
        return 0.0
    total_entry_quantity = sum(
        abs(_safe_float(getattr(order, "quantity", None), 0.0)) for order in entry_orders
    )
    total_entry_fee = sum(
        abs(_order_fee_abs(order)) for order in entry_orders
    )
    if total_entry_fee <= 0:
        return 0.0
    if total_entry_quantity <= 0:
        return total_entry_fee
    return total_entry_fee * min(max(close_quantity, 0.0) / total_entry_quantity, 1.0)


def _order_fee_abs(order: Order) -> float:
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    return abs(_safe_float(raw.get("fee_abs") or getattr(order, "fee", None), 0.0))


def _closed_position_fill_pair_candidates(
    fills: list[OkxNativeFillGroup],
) -> list[tuple[OkxNativeFillGroup, OkxNativeFillGroup, str]]:
    result: list[tuple[OkxNativeFillGroup, OkxNativeFillGroup, str]] = []
    used_order_ids: set[str] = set()
    by_inst: dict[str, list[OkxNativeFillGroup]] = {}
    for fill in fills:
        inst_id = str(fill.inst_id or "").strip().upper()
        if not inst_id or fill.side not in {"buy", "sell"} or fill.contracts <= 0:
            continue
        by_inst.setdefault(inst_id, []).append(fill)
    for inst_fills in by_inst.values():
        ordered = sorted(inst_fills, key=lambda item: item.timestamp or datetime.min.replace(tzinfo=UTC))
        for close_fill in ordered:
            if close_fill.order_id in used_order_ids:
                continue
            if abs(_safe_float(close_fill.fill_pnl, 0.0)) == 0:
                continue
            expected_entry_side = "sell" if close_fill.side == "buy" else "buy"
            side = "short" if expected_entry_side == "sell" else "long"
            candidates: list[tuple[float, OkxNativeFillGroup]] = []
            close_time = close_fill.timestamp
            if close_time is None:
                continue
            for entry_fill in ordered:
                if entry_fill.order_id in used_order_ids or entry_fill.order_id == close_fill.order_id:
                    continue
                if entry_fill.side != expected_entry_side:
                    continue
                if entry_fill.timestamp is None or entry_fill.timestamp > close_time:
                    continue
                delta = (close_time - entry_fill.timestamp).total_seconds()
                if delta > FILL_PAIR_POSITION_TIME_WINDOW_SECONDS:
                    continue
                if not _quantity_covers(
                    _safe_float(entry_fill.contracts, 0.0),
                    _safe_float(close_fill.contracts, 0.0),
                    FILL_PAIR_POSITION_QUANTITY_TOLERANCE_RATIO,
                ):
                    continue
                candidates.append((delta, entry_fill))
            if not candidates:
                continue
            entry_fill = sorted(candidates, key=lambda item: item[0])[0][1]
            result.append((entry_fill, close_fill, side))
            used_order_ids.add(entry_fill.order_id)
            used_order_ids.add(close_fill.order_id)
    return result


def _apply_position_history_payload(position: Position, payload: dict[str, Any], *, now: datetime) -> None:
    existing_entry_exchange_order_id = getattr(position, "entry_exchange_order_id", None)
    existing_close_exchange_order_id = getattr(position, "close_exchange_order_id", None)
    position.model_name = str(payload["model_name"])
    position.execution_mode = str(payload["execution_mode"])
    position.symbol = str(payload["symbol"])
    position.side = str(payload["side"])
    position.quantity = float(payload["quantity"])
    position.entry_price = float(payload["entry_price"])
    position.current_price = float(payload["current_price"])
    position.leverage = float(payload["leverage"])
    position.unrealized_pnl = 0.0
    position.realized_pnl = float(payload["realized_pnl"])
    position.is_open = False
    position.closed_at = payload["closed_at"]
    position.created_at = payload["created_at"]
    position.okx_inst_id = payload.get("okx_inst_id")
    position.okx_pos_id = payload.get("okx_pos_id")
    payload_entry_exchange_order_id = payload.get("entry_exchange_order_id")
    payload_close_exchange_order_id = payload.get("close_exchange_order_id")
    position.entry_exchange_order_id = (
        str(payload_entry_exchange_order_id)
        if str(payload_entry_exchange_order_id or "").strip()
        else existing_entry_exchange_order_id
    )
    position.close_exchange_order_id = (
        str(payload_close_exchange_order_id)
        if str(payload_close_exchange_order_id or "").strip()
        else existing_close_exchange_order_id
    )
    position.updated_at = now


def _best_position_history_lifecycle_match(
    candidates: list[Position],
    *,
    created_at: datetime,
    closed_at: datetime,
    entry_order_ids: list[str],
    close_order_ids: list[str],
) -> Position | None:
    """Match one OKX position-history lifecycle without treating posId as unique.

    OKX one-way/net positions can reuse the same posId across multiple ACT/USDT
    open-close lifecycles.  Local cache updates therefore need lifecycle evidence
    (linked order ids or open/close timestamps), not posId alone.
    """

    if not candidates:
        return None
    incoming_entry_ids = set(entry_order_ids)
    incoming_close_ids = set(close_order_ids)
    scored: list[tuple[int, float, int, Position]] = []
    for position in candidates:
        existing_entry_ids = _split_exchange_order_ids(getattr(position, "entry_exchange_order_id", None))
        existing_close_ids = _split_exchange_order_ids(getattr(position, "close_exchange_order_id", None))
        entry_match = _lifecycle_order_ids_match(existing_entry_ids, incoming_entry_ids)
        close_match = _lifecycle_order_ids_match(existing_close_ids, incoming_close_ids)
        if close_match:
            scored.append((0, 0.0, -int(getattr(position, "id", 0) or 0), position))
            continue
        if entry_match:
            scored.append((1, 0.0, -int(getattr(position, "id", 0) or 0), position))
            continue

        existing_opened_at = _db_datetime_to_utc(getattr(position, "created_at", None))
        existing_closed_at = _db_datetime_to_utc(getattr(position, "closed_at", None))
        if existing_closed_at is None:
            continue
        opened_delta = (
            abs((existing_opened_at - _aware_utc(created_at)).total_seconds())
            if existing_opened_at is not None
            else 999999.0
        )
        closed_delta = abs((existing_closed_at - _aware_utc(closed_at)).total_seconds())
        if closed_delta <= 3.0 and opened_delta <= 3.0:
            scored.append((2, opened_delta + closed_delta, -int(getattr(position, "id", 0) or 0), position))
        elif (
            closed_delta <= 3.0
            and _lifecycle_order_ids_are_empty_or_polluted(existing_entry_ids, incoming_entry_ids)
            and _lifecycle_order_ids_are_empty_or_polluted(existing_close_ids, incoming_close_ids)
        ):
            scored.append((3, closed_delta, -int(getattr(position, "id", 0) or 0), position))
    if not scored:
        return None
    return sorted(scored, key=lambda item: (item[0], item[1], item[2]))[0][3]


def _lifecycle_order_ids_match(existing_ids: set[str], incoming_ids: set[str]) -> bool:
    if not existing_ids or not incoming_ids:
        return False
    if not existing_ids & incoming_ids:
        return False
    return existing_ids.issubset(incoming_ids)


def _lifecycle_order_ids_are_empty_or_polluted(existing_ids: set[str], incoming_ids: set[str]) -> bool:
    if not existing_ids:
        return True
    return bool(incoming_ids and existing_ids & incoming_ids and not existing_ids.issubset(incoming_ids))


def _position_from_current_row(
    row: dict[str, Any],
    *,
    mode: str,
    now: datetime,
    contract_size: float,
) -> dict[str, Any]:
    inst_id = _current_position_inst_id(row)
    contracts = _current_position_contracts(row)
    quantity = contracts * (contract_size if contract_size > 0 else 1.0)
    entry_price = _current_position_entry_price(row)
    mark_price = _current_position_mark_price(row)
    timestamp = _current_position_time(row) or now
    return {
        "model_name": "okx_authoritative_sync",
        "execution_mode": mode,
        "symbol": symbol_from_okx_inst_id(inst_id) or normalize_trading_symbol(inst_id),
        "side": _current_position_side(row),
        "quantity": quantity,
        "entry_price": entry_price,
        "current_price": mark_price or entry_price,
        "leverage": _current_position_leverage(row),
        "unrealized_pnl": _safe_float(
            _current_position_info(row).get("upl") or _current_position_info(row).get("unrealizedPnl"),
            0.0,
        ),
        "realized_pnl": 0.0,
        "is_open": True,
        "closed_at": None,
        "created_at": timestamp,
        "updated_at": now,
        "okx_inst_id": inst_id,
        "okx_pos_id": _current_position_pos_id(row),
        "entry_exchange_order_id": None,
        "close_exchange_order_id": None,
    }


def _best_current_position_cache_row(candidates: list[Position]) -> Position | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda position: (
            bool(str(getattr(position, "entry_exchange_order_id", "") or "").strip()),
            abs(_safe_float(getattr(position, "quantity", None), 0.0)),
            _db_datetime_to_utc(getattr(position, "updated_at", None)) or datetime.min.replace(tzinfo=UTC),
            int(getattr(position, "id", 0) or 0),
        ),
        reverse=True,
    )[0]


def _apply_current_position_payload(position: Position, payload: dict[str, Any], *, now: datetime) -> None:
    existing_entry_exchange_order_id = getattr(position, "entry_exchange_order_id", None)
    position.model_name = str(payload["model_name"])
    position.execution_mode = str(payload["execution_mode"])
    position.symbol = str(payload["symbol"])
    position.side = str(payload["side"])
    position.quantity = float(payload["quantity"])
    position.entry_price = float(payload["entry_price"])
    position.current_price = float(payload["current_price"])
    position.leverage = float(payload["leverage"])
    position.unrealized_pnl = float(payload["unrealized_pnl"])
    position.realized_pnl = float(payload["realized_pnl"])
    position.is_open = True
    position.closed_at = None
    position.created_at = payload["created_at"]
    position.okx_inst_id = payload.get("okx_inst_id")
    position.okx_pos_id = payload.get("okx_pos_id")
    position.entry_exchange_order_id = _merge_exchange_order_ids(
        existing_entry_exchange_order_id,
        payload.get("entry_exchange_order_id"),
    ) or None
    position.close_exchange_order_id = None
    position.updated_at = now


def _matching_current_position_entry_order(
    payload: dict[str, Any],
    orders: list[Order],
) -> Order | None:
    side = str(payload.get("side") or "").lower().strip()
    expected_side = "buy" if side == "long" else "sell" if side == "short" else ""
    if not expected_side:
        return None
    inst_id = str(payload.get("okx_inst_id") or "").strip().upper()
    symbol = str(payload.get("symbol") or "").strip()
    reference_time = _db_datetime_to_utc(payload.get("created_at"))
    if reference_time is None:
        return None
    quantity = _safe_float(payload.get("quantity"), 0.0)
    entry_price = _safe_float(payload.get("entry_price"), 0.0)
    candidates: list[tuple[float, Order]] = []
    for order in orders:
        if str(getattr(order, "side", "") or "").lower().strip() != expected_side:
            continue
        if not _order_matches_current_position_symbol(order, inst_id=inst_id, symbol=symbol):
            continue
        order_time = _db_datetime_to_utc(
            getattr(order, "filled_at", None) or getattr(order, "created_at", None)
        )
        if order_time is None:
            continue
        delta = abs((order_time - reference_time).total_seconds())
        if delta > CURRENT_POSITION_ENTRY_LINK_WINDOW_SECONDS:
            continue
        if not _quantity_covers(
            _safe_float(getattr(order, "quantity", None), 0.0),
            quantity,
            CURRENT_POSITION_ENTRY_QUANTITY_TOLERANCE_RATIO,
        ):
            continue
        if not _relative_close_enough(
            _safe_float(getattr(order, "price", None), 0.0),
            entry_price,
            CURRENT_POSITION_ENTRY_PRICE_TOLERANCE_RATIO,
        ):
            continue
        candidates.append((delta, order))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _order_matches_current_position_symbol(order: Order, *, inst_id: str, symbol: str) -> bool:
    order_inst_id = str(getattr(order, "okx_inst_id", "") or "").strip().upper()
    if inst_id and order_inst_id == inst_id:
        return True
    return normalize_trading_symbol(getattr(order, "symbol", None)) == normalize_trading_symbol(symbol)


def _quantity_covers(order_quantity: float, position_quantity: float, tolerance_ratio: float) -> bool:
    if order_quantity <= 0 or position_quantity <= 0:
        return False
    tolerance = max(abs(order_quantity), abs(position_quantity), 1e-9) * tolerance_ratio
    return order_quantity + tolerance >= position_quantity


def _relative_close_enough(left: float, right: float, tolerance_ratio: float) -> bool:
    if left <= 0 or right <= 0:
        return False
    tolerance = max(abs(left), abs(right), 1e-9) * tolerance_ratio
    return abs(left - right) <= tolerance


def _current_position_info(row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    info = row.get("info")
    return info if isinstance(info, dict) else row


def _current_position_inst_id(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return ""
    info = _current_position_info(row)
    return str(info.get("instId") or row.get("instId") or row.get("symbol") or "").strip().upper()


def _current_position_pos_id(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return ""
    info = _current_position_info(row)
    return str(info.get("posId") or row.get("id") or row.get("posId") or "").strip()


def _current_position_contracts(row: dict[str, Any] | None) -> float:
    if not isinstance(row, dict):
        return 0.0
    info = _current_position_info(row)
    for key in ("pos", "qty", "contracts"):
        value = abs(_safe_float(info.get(key) if key in info else row.get(key), 0.0))
        if value > 0:
            return value
    return 0.0


def _current_position_side(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return ""
    info = _current_position_info(row)
    side = str(info.get("posSide") or row.get("side") or row.get("posSide") or "").lower().strip()
    if side in {"long", "short"}:
        return side
    pos = _safe_float(info.get("pos") or info.get("qty") or row.get("contracts"), 0.0)
    if pos < 0:
        return "short"
    if pos > 0:
        return "long"
    return ""


def _current_position_contract_size(
    row: dict[str, Any] | None,
    contract_sizes: dict[str, float],
) -> float:
    inst_id = _current_position_inst_id(row)
    size = _contract_size_for_inst_id(inst_id, contract_sizes)
    if size > 0 and size != 1.0:
        return size
    info = _current_position_info(row)
    payload_size = _safe_float(info.get("ctVal") or row.get("contractSize") if isinstance(row, dict) else None, 0.0)
    if payload_size > 0:
        return payload_size
    return size if size > 0 else 1.0


def _current_position_entry_price(row: dict[str, Any] | None) -> float:
    if not isinstance(row, dict):
        return 0.0
    info = _current_position_info(row)
    for key in ("avgPx", "entryPrice"):
        value = _safe_float(info.get(key) if key in info else row.get(key), 0.0)
        if value > 0:
            return value
    return 0.0


def _current_position_mark_price(row: dict[str, Any] | None) -> float:
    if not isinstance(row, dict):
        return 0.0
    info = _current_position_info(row)
    for key in ("markPx", "last", "markPrice"):
        value = _safe_float(info.get(key) if key in info else row.get(key), 0.0)
        if value > 0:
            return value
    return 0.0


def _current_position_leverage(row: dict[str, Any] | None) -> float:
    if not isinstance(row, dict):
        return 1.0
    info = _current_position_info(row)
    leverage = _safe_float(info.get("lever") or info.get("leverage") or row.get("leverage"), 0.0)
    return leverage if leverage > 0 else 1.0


def _current_position_time(row: dict[str, Any] | None) -> datetime | None:
    if not isinstance(row, dict):
        return None
    info = _current_position_info(row)
    return _datetime_from_ms(info.get("cTime") or info.get("uTime") or row.get("timestamp"))


def _current_position_inst_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {
        inst_id
        for row in rows
        if (inst_id := _current_position_inst_id(row))
    }


def _position_history_inst_id(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return ""
    return str(row.get("instId") or "").strip().upper()


def _position_history_pos_id(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return ""
    return str(row.get("posId") or "").strip()


def _position_history_side(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return ""
    side = str(row.get("posSide") or "").lower().strip()
    if side in {"long", "short"}:
        return side
    direction = str(row.get("direction") or row.get("side") or "").lower().strip()
    if direction in {"long", "short"}:
        return direction
    open_side = str(row.get("openSide") or row.get("openOrdSide") or "").lower().strip()
    if open_side == "sell":
        return "short"
    if open_side == "buy":
        return "long"
    pos = _safe_float(row.get("pos"), 0.0)
    if pos < 0:
        return "short"
    if pos > 0:
        return "long"
    return ""


def _position_history_contracts(row: dict[str, Any] | None) -> float:
    if not isinstance(row, dict):
        return 0.0
    for key in ("closeTotalPos", "openMaxPos", "pos", "sz"):
        value = abs(_safe_float(row.get(key), 0.0))
        if value > 0:
            return value
    return 0.0


def _position_history_leverage(row: dict[str, Any] | None) -> float:
    if not isinstance(row, dict):
        return 1.0
    leverage = _safe_float(row.get("lever") or row.get("leverage"), 0.0)
    return leverage if leverage > 0 else 1.0


def _position_history_time(row: dict[str, Any] | None) -> datetime | None:
    if not isinstance(row, dict):
        return None
    return _datetime_from_ms(row.get("uTime") or row.get("cTime"))


def _position_history_opened_at(row: dict[str, Any] | None) -> datetime | None:
    if not isinstance(row, dict):
        return None
    return _datetime_from_ms(row.get("cTime") or row.get("openTime"))


def _position_history_closed_at(row: dict[str, Any] | None) -> datetime | None:
    if not isinstance(row, dict):
        return None
    return _datetime_from_ms(row.get("uTime") or row.get("closeTime"))


def _fills_by_inst_and_position_side(
    fills: list[OkxNativeFillGroup],
) -> dict[tuple[str, str], list[OkxNativeFillGroup]]:
    result: dict[tuple[str, str], list[OkxNativeFillGroup]] = {}
    for fill in fills:
        inst_id = str(fill.inst_id or "").strip().upper()
        position_sides = _position_sides_from_fill(fill)
        if not inst_id or not position_sides:
            continue
        for position_side in position_sides:
            result.setdefault((inst_id, position_side), []).append(fill)
    for rows in result.values():
        rows.sort(key=lambda item: item.timestamp or datetime.min.replace(tzinfo=UTC))
    return result


def _position_sides_from_fill(fill: OkxNativeFillGroup) -> tuple[str, ...]:
    pos_side = str(fill.pos_side or "").lower().strip()
    if pos_side in {"long", "short"}:
        return (pos_side,)
    # In OKX one-way/net position mode the fill side alone cannot identify
    # whether a trade opened or closed a long/short lifecycle.  Keep both
    # buckets and let the expected entry/close side plus time window decide.
    return ("long", "short")


def _close_side_for_position_side(side: str) -> str:
    return "buy" if side == "short" else "sell" if side == "long" else ""


def _entry_side_for_position_side(side: str) -> str:
    return "sell" if side == "short" else "buy" if side == "long" else ""


def _infer_position_history_side(
    row: dict[str, Any],
    *,
    entry_fills: list[OkxNativeFillGroup],
    close_fills: list[OkxNativeFillGroup],
) -> str:
    side = _position_history_side(row)
    if side in {"long", "short"}:
        return side
    if entry_fills:
        entry_side = str(entry_fills[0].side or "").lower().strip()
        if entry_side == "sell":
            return "short"
        if entry_side == "buy":
            return "long"
    if close_fills:
        close_side = str(close_fills[0].side or "").lower().strip()
        if close_side == "buy":
            return "short"
        if close_side == "sell":
            return "long"
    return ""


def _infer_net_position_history_fills(
    row: dict[str, Any],
    *,
    fills_by_inst_side: dict[tuple[str, str], list[OkxNativeFillGroup]],
) -> tuple[list[OkxNativeFillGroup], list[OkxNativeFillGroup]]:
    inst_id = _position_history_inst_id(row)
    opened_at = _position_history_opened_at(row)
    closed_at = _position_history_closed_at(row) or _position_history_time(row)
    target_contracts = _position_history_contracts(row)
    best: tuple[float, list[OkxNativeFillGroup], list[OkxNativeFillGroup]] | None = None
    for candidate_side in ("short", "long"):
        entry_fills = _matching_position_history_fills(
            fills_by_inst_side.get((inst_id, candidate_side), []),
            expected_side=_entry_side_for_position_side(candidate_side),
            reference_time=opened_at,
            target_contracts=target_contracts,
            max_rows=12,
        )
        close_fills = _matching_position_history_fills(
            fills_by_inst_side.get((inst_id, candidate_side), []),
            expected_side=_close_side_for_position_side(candidate_side),
            reference_time=closed_at,
            target_contracts=target_contracts,
            max_rows=12,
        )
        if not entry_fills or not close_fills:
            continue
        score = _fill_time_distance_seconds(entry_fills, opened_at) + _fill_time_distance_seconds(
            close_fills,
            closed_at,
        )
        if best is None or score < best[0]:
            best = (score, entry_fills, close_fills)
    if best is None:
        return [], []
    return best[1], best[2]


def _fill_time_distance_seconds(fills: list[OkxNativeFillGroup], reference_time: datetime | None) -> float:
    if reference_time is None:
        return 999999.0
    ref = _aware_utc(reference_time)
    distances = [
        abs((_aware_utc(fill.timestamp) - ref).total_seconds())
        for fill in fills
        if fill.timestamp is not None
    ]
    return min(distances) if distances else 999999.0


def _matching_position_history_close_fills(
    row: dict[str, Any],
    *,
    fills_by_inst_side: dict[tuple[str, str], list[OkxNativeFillGroup]],
    side: str | None = None,
) -> list[OkxNativeFillGroup]:
    inst_id = _position_history_inst_id(row)
    side = side or _position_history_side(row)
    close_side = _close_side_for_position_side(side)
    closed_at = _position_history_closed_at(row) or _position_history_time(row)
    return _matching_position_history_fills(
        fills_by_inst_side.get((inst_id, side), []),
        expected_side=close_side,
        reference_time=closed_at,
        target_contracts=_position_history_contracts(row),
        max_rows=12,
    )


def _matching_position_history_entry_fills(
    row: dict[str, Any],
    *,
    fills_by_inst_side: dict[tuple[str, str], list[OkxNativeFillGroup]],
    side: str | None = None,
) -> list[OkxNativeFillGroup]:
    inst_id = _position_history_inst_id(row)
    side = side or _position_history_side(row)
    entry_side = _entry_side_for_position_side(side)
    opened_at = _position_history_opened_at(row)
    return _matching_position_history_fills(
        fills_by_inst_side.get((inst_id, side), []),
        expected_side=entry_side,
        reference_time=opened_at,
        target_contracts=_position_history_contracts(row),
        max_rows=12,
    )


def _position_history_lifecycle_fills(
    row: dict[str, Any],
    *,
    fills_by_inst_side: dict[tuple[str, str], list[OkxNativeFillGroup]],
    side: str,
) -> tuple[list[OkxNativeFillGroup], list[OkxNativeFillGroup]]:
    inst_id = _position_history_inst_id(row)
    opened_at = _position_history_opened_at(row)
    closed_at = _position_history_closed_at(row) or _position_history_time(row)
    if opened_at is None or closed_at is None:
        return [], []
    start = _aware_utc(opened_at) - timedelta(seconds=3)
    end = _aware_utc(closed_at) + timedelta(seconds=3)
    entry_side = _entry_side_for_position_side(side)
    close_side = _close_side_for_position_side(side)
    entry_fills: list[OkxNativeFillGroup] = []
    close_fills: list[OkxNativeFillGroup] = []
    for fill in fills_by_inst_side.get((inst_id, side), []):
        if fill.timestamp is None or not fill.order_id:
            continue
        timestamp = _aware_utc(fill.timestamp)
        if timestamp < start or timestamp > end:
            continue
        if fill.side == entry_side:
            entry_fills.append(fill)
        elif fill.side == close_side:
            close_fills.append(fill)
    return (
        _dedupe_fills_by_order_id(entry_fills),
        _dedupe_fills_by_order_id(close_fills),
    )


def _matching_position_history_fills(
    fills: list[OkxNativeFillGroup],
    *,
    expected_side: str,
    reference_time: datetime | None,
    target_contracts: float = 0.0,
    max_rows: int,
) -> list[OkxNativeFillGroup]:
    if not expected_side:
        return []
    if reference_time is None:
        return [fill for fill in fills if fill.side == expected_side][-max_rows:]
    ref = _aware_utc(reference_time)
    window_seconds = POSITION_HISTORY_LINK_WINDOW_SECONDS
    candidates: list[tuple[float, OkxNativeFillGroup]] = []
    for fill in fills:
        if fill.side != expected_side or fill.timestamp is None:
            continue
        delta = abs((_aware_utc(fill.timestamp) - ref).total_seconds())
        if delta <= window_seconds:
            candidates.append((delta, fill))
    candidates = sorted(candidates, key=lambda item: item[0])
    target = _safe_float(target_contracts, 0.0)
    if target <= 0:
        return [fill for _delta, fill in candidates[:max_rows]]
    selected: list[OkxNativeFillGroup] = []
    selected_contracts = 0.0
    tolerance = max(target * FILL_PAIR_POSITION_QUANTITY_TOLERANCE_RATIO, 1e-9)
    for _delta, fill in candidates:
        selected.append(fill)
        selected_contracts += _safe_float(fill.contracts, 0.0)
        if selected_contracts + tolerance >= target or len(selected) >= max_rows:
            break
    return selected


def _dedupe_fills_by_order_id(fills: list[OkxNativeFillGroup]) -> list[OkxNativeFillGroup]:
    result: list[OkxNativeFillGroup] = []
    seen: set[str] = set()
    for fill in sorted(fills, key=lambda item: item.timestamp or datetime.min.replace(tzinfo=UTC)):
        order_id = str(fill.order_id or "").strip()
        if not order_id or order_id in seen:
            continue
        seen.add(order_id)
        result.append(fill)
    return result


def _ordered_exchange_ids(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip()
        if token and token not in seen:
            result.append(token)
            seen.add(token)
    return result


def _contract_size_for_inst_id(inst_id: str, contract_sizes: dict[str, float]) -> float:
    size = _safe_float(contract_sizes.get(str(inst_id or "").strip().upper()), 0.0)
    return size if size > 0 else 1.0


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


def _fill_order_type(fill: OkxNativeFillGroup) -> str:
    row = fill.latest_row
    return str(row.get("ordType") or row.get("type") or "market").lower() or "market"


def _contract_size_for_fill(
    fill: OkxNativeFillGroup,
    contract_sizes: dict[str, float],
) -> float:
    for key in (fill.inst_id, okx_inst_id_from_symbol(fill.symbol) or ""):
        size = _safe_float(contract_sizes.get(key), 0.0)
        if size > 0:
            return size
    for row in fill.rows:
        size = _safe_float(row.get("ctVal") or row.get("contractSize"), 0.0)
        if size > 0:
            return size
    return 1.0


def _fill_base_quantity(fill: OkxNativeFillGroup, contract_size: float) -> float:
    size = _safe_float(contract_size, 0.0)
    return float(fill.contracts) * (size if size > 0 else 1.0)


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


def _position_sync_suppression_from_event(
    event: StrategyLearningEvent,
) -> OkxPositionSyncSuppression | None:
    data = _dict_payload(getattr(event, "attribution", None))
    inst_id = _normalize_okx_inst_id(
        data.get("okx_inst_id")
        or data.get("instId")
        or data.get("inst_id")
    )
    symbol = normalize_trading_symbol(
        getattr(event, "symbol", None)
        or data.get("symbol")
        or symbol_from_okx_inst_id(inst_id)
        or ""
    )
    side = _normalize_position_side(getattr(event, "side", None) or data.get("side"))
    entry_order_ids = frozenset(
        _suppression_order_ids(
            data,
            "entry_exchange_order_ids",
            "entry_exchange_order_id",
            "entry_order_ids",
            "entry_order_id",
        )
    )
    close_order_ids = frozenset(
        _suppression_order_ids(
            data,
            "close_exchange_order_ids",
            "close_exchange_order_id",
            "close_order_ids",
            "close_order_id",
        )
    )
    pos_id = str(data.get("okx_pos_id") or data.get("posId") or data.get("pos_id") or "").strip()
    if not (symbol or inst_id) or not (pos_id or entry_order_ids or close_order_ids):
        return None
    return OkxPositionSyncSuppression(
        mode="live" if str(getattr(event, "execution_mode", "") or "").lower() == "live" else "paper",
        symbol=symbol,
        side=side,
        okx_inst_id=inst_id,
        okx_pos_id=pos_id,
        entry_order_ids=entry_order_ids,
        close_order_ids=close_order_ids,
        created_at=_parse_datetime(
            data.get("created_at")
            or data.get("opened_at")
            or data.get("open_time")
        ),
        closed_at=_parse_datetime(
            data.get("closed_at")
            or data.get("close_time")
            or data.get("uTime")
        ),
        reason=str(getattr(event, "reason", "") or data.get("reason") or "").strip(),
    )


def _matching_position_sync_suppression(
    suppressions: tuple[OkxPositionSyncSuppression, ...],
    *,
    mode: str,
    symbol: str | None,
    side: str | None,
    okx_inst_id: str | None,
    okx_pos_id: str | None,
    entry_order_ids: list[str],
    close_order_ids: list[str],
    created_at: datetime | None,
    closed_at: datetime | None,
) -> OkxPositionSyncSuppression | None:
    normalized_mode = "live" if str(mode or "").lower() == "live" else "paper"
    normalized_symbol = normalize_trading_symbol(symbol or symbol_from_okx_inst_id(okx_inst_id or "") or "")
    normalized_inst_id = _normalize_okx_inst_id(okx_inst_id)
    normalized_side = _normalize_position_side(side)
    normalized_pos_id = str(okx_pos_id or "").strip()
    entry_ids = set(_ordered_exchange_ids(entry_order_ids))
    close_ids = set(_ordered_exchange_ids(close_order_ids))

    for suppression in suppressions:
        if suppression.mode != normalized_mode:
            continue
        if suppression.okx_inst_id:
            if normalized_inst_id != suppression.okx_inst_id:
                continue
        elif suppression.symbol and normalized_symbol != suppression.symbol:
            continue
        if suppression.side and normalized_side and suppression.side != normalized_side:
            continue
        if suppression.okx_pos_id and normalized_pos_id and suppression.okx_pos_id != normalized_pos_id:
            continue
        if suppression.entry_order_ids and not (suppression.entry_order_ids & entry_ids):
            continue
        if suppression.close_order_ids and not (suppression.close_order_ids & close_ids):
            continue
        if not suppression.entry_order_ids and not suppression.close_order_ids:
            if not suppression.okx_pos_id:
                continue
            if not _suppression_time_matches(suppression.created_at, created_at):
                continue
            if not _suppression_time_matches(suppression.closed_at, closed_at):
                continue
        elif suppression.created_at and not _suppression_time_matches(suppression.created_at, created_at):
            continue
        elif suppression.closed_at and not _suppression_time_matches(suppression.closed_at, closed_at):
            continue
        return suppression
    return None


def _suppression_order_ids(data: dict[str, Any], *keys: str) -> set[str]:
    result: set[str] = set()
    for key in keys:
        result.update(_coerce_order_id_set(data.get(key)))
    return result


def _coerce_order_id_set(value: Any) -> set[str]:
    if isinstance(value, dict):
        result: set[str] = set()
        for item in value.values():
            result.update(_coerce_order_id_set(item))
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        result: set[str] = set()
        for item in value:
            result.update(_coerce_order_id_set(item))
        return result
    return _split_exchange_order_ids(value)


def _dict_payload(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_okx_inst_id(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_position_side(value: Any) -> str:
    text = str(value or "").lower().strip()
    if text in {"long", "short"}:
        return text
    if text in {"buy", "sell"}:
        return "long" if text == "buy" else "short"
    return ""


def _suppression_time_matches(
    expected: datetime | None,
    actual: datetime | None,
    *,
    tolerance_seconds: float = 5 * 60,
) -> bool:
    if expected is None or actual is None:
        return True
    return abs((_aware_utc(actual) - _aware_utc(expected)).total_seconds()) <= tolerance_seconds


def _merge_exchange_order_ids(*values: Any, max_length: int = 100) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in sorted(_split_exchange_order_ids(value)):
            if token in seen:
                continue
            candidate = ",".join([*ordered, token]) if ordered else token
            if len(candidate) > max_length:
                continue
            seen.add(token)
            ordered.append(token)
    return ",".join(ordered)


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
    base_quantity = _safe_float(
        raw.get("base_quantity") or raw.get("filled_base_quantity"),
        0.0,
    )
    if base_quantity > 0:
        return base_quantity
    contracts = _safe_float(raw.get("contracts") or raw.get("filled_contracts"), 0.0)
    contract_size = _safe_float(
        raw.get("contract_size") or raw.get("contractSize"),
        0.0,
    )
    if contracts > 0:
        return contracts * (contract_size if contract_size > 0 else 1.0)
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


async def _bounded(awaitable: Any, timeout_seconds: float) -> Any:
    import asyncio

    return await asyncio.wait_for(awaitable, timeout=max(float(timeout_seconds), 0.5))
