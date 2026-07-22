"""Read-only OKX authoritative fact sync.

This module pulls exchange facts directly from OKX and compares them with local
orders/positions.  It intentionally does not mutate the database; repair apply
must stay in explicit allowlisted scripts with backup.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import or_, select

from core.safe_output import safe_error_text
from core.symbols import (
    normalize_trading_symbol,
    okx_inst_id_from_payload,
    okx_inst_id_from_symbol,
    symbol_from_okx_inst_id,
    trading_symbol_variants,
)
from db.session import get_read_session_ctx
from executor.okx_executor import OKXExecutor
from models.decision import AIDecision
from models.trade import Order, Position
from services.exchange_position_state import parse_exchange_position_snapshot
from services.okx_native_facts import (
    OkxNativeFactsClient,
    build_okx_protection_execution_lifecycle,
)
from services.okx_position_confirmation import (
    find_current_position_entry_confirmation,
    order_has_current_position_snapshot_confirmation,
)

DEFAULT_LOOKBACK_HOURS = 24
DEFAULT_LIMIT = 200
DEFAULT_TIMEOUT_SECONDS = 6.0
DEFAULT_MAX_PULL_ATTEMPTS = 2
MAX_AUTHORITATIVE_FILL_PAGES = 10
LOCAL_ORDER_SYNC_GRACE_SECONDS = 120.0
QUANTITY_TOLERANCE_RATIO = 0.02
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COLD_START_MARKER_PATH = ROOT / "data" / "phase3_cold_start_reset_marker.json"


@dataclass(frozen=True, slots=True)
class OkxFillGroup:
    order_id: str
    trade_ids: tuple[str, ...]
    inst_id: str
    symbol: str
    side: str
    pos_side: str
    contracts: float
    avg_price: float
    fee_abs: float
    fill_pnl: float
    timestamp_ms: float
    timestamp: datetime | None
    raw_count: int
    rows: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "trade_ids": list(self.trade_ids),
            "inst_id": self.inst_id,
            "symbol": self.symbol,
            "side": self.side,
            "pos_side": self.pos_side,
            "contracts": _round(self.contracts),
            "avg_price": _round(self.avg_price),
            "fee_abs": _round(self.fee_abs),
            "fill_pnl": _round(self.fill_pnl),
            "timestamp_ms": _round(self.timestamp_ms),
            "timestamp": _iso(self.timestamp),
            "raw_count": self.raw_count,
        }


@dataclass(frozen=True, slots=True)
class OkxAuthoritativeIssue:
    kind: str
    classification: str
    severity: str
    reason: str
    symbol: str = ""
    side: str = ""
    local_order_id: int | None = None
    local_position_id: int | None = None
    exchange_order_id: str = ""
    local_quantity: float | None = None
    okx_contracts: float | None = None
    okx_contract_size: float | None = None
    expected_base_quantity: float | None = None
    local_price: float | None = None
    okx_price: float | None = None
    okx_timestamp: datetime | None = None
    linked_local_order_id: int | None = None
    linked_exchange_order_id: str = ""
    okx_algo_id: str = ""
    okx_source: str = ""
    repair_entrypoint: str = ""
    protection_execution: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "classification": self.classification,
            "severity": self.severity,
            "reason": self.reason,
            "symbol": self.symbol,
            "side": self.side,
            "local_order_id": self.local_order_id,
            "local_position_id": self.local_position_id,
            "exchange_order_id": self.exchange_order_id,
            "local_quantity": _round_optional(self.local_quantity),
            "okx_contracts": _round_optional(self.okx_contracts),
            "okx_contract_size": _round_optional(self.okx_contract_size),
            "expected_base_quantity": _round_optional(self.expected_base_quantity),
            "local_price": _round_optional(self.local_price),
            "okx_price": _round_optional(self.okx_price),
            "okx_timestamp": _iso(self.okx_timestamp),
            "linked_local_order_id": self.linked_local_order_id,
            "linked_exchange_order_id": self.linked_exchange_order_id,
            "okx_algo_id": self.okx_algo_id,
            "okx_source": self.okx_source,
            "repair_entrypoint": self.repair_entrypoint,
            "protection_execution": (
                dict(self.protection_execution)
                if isinstance(self.protection_execution, dict)
                else None
            ),
        }


class OkxAuthoritativeSyncService:
    """Pull OKX facts and produce a read-only reconciliation plan."""

    def __init__(
        self,
        *,
        mode: str = "paper",
        lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
        limit: int = DEFAULT_LIMIT,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_pull_attempts: int = DEFAULT_MAX_PULL_ATTEMPTS,
        executor_factory: Any | None = None,
        cold_start_marker_path: str | Path | None = DEFAULT_COLD_START_MARKER_PATH,
    ) -> None:
        self.mode = str(mode or "paper")
        self.lookback_hours = max(int(lookback_hours or DEFAULT_LOOKBACK_HOURS), 1)
        self.limit = max(1, min(int(limit or DEFAULT_LIMIT), 1000))
        self.timeout_seconds = max(float(timeout_seconds or DEFAULT_TIMEOUT_SECONDS), 0.5)
        self.max_pull_attempts = max(1, min(int(max_pull_attempts or 1), 3))
        self.executor_factory = executor_factory or OKXExecutor
        self.cold_start_marker_path = (
            Path(cold_start_marker_path) if cold_start_marker_path is not None else None
        )

    async def collect(self) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        nominal_since = started_at - timedelta(hours=self.lookback_hours)
        watermark = self._load_cold_start_watermark()
        since = max(
            item
            for item in (nominal_since, watermark.get("reset_at"))
            if isinstance(item, datetime)
        )
        since_naive = since.replace(tzinfo=None)
        local_orders, local_positions, local_decisions = await self._load_local_facts(since_naive)
        symbols = {
            normalize_trading_symbol(item)
            for item in [
                *(_local_order_symbol(order, local_decisions.get(int(order.decision_id or 0))) for order in local_orders),
                *(_local_position_symbol(position) for position in local_positions),
            ]
        }
        symbols.discard("")
        target_inst_ids = {
            item
            for item in [
                *(okx_inst_id_from_symbol(symbol) for symbol in symbols),
                *(
                    _local_order_okx_inst_id(
                        order,
                        local_decisions.get(int(order.decision_id or 0)),
                    )
                    for order in local_orders
                ),
                *(
                    str(getattr(position, "okx_inst_id", "") or "").strip().upper()
                    for position in local_positions
                ),
            ]
            if item
        }
        target_order_ids = {
            token
            for order in local_orders
            for token in _split_exchange_order_ids(order.exchange_order_id)
        }
        linked_position_order_ids = _linked_position_order_ids(local_positions)
        priority_context_order_ids = target_order_ids - linked_position_order_ids

        exchange_positions: list[dict[str, Any]] = []
        exchange_fills: list[OkxFillGroup] = []
        exchange_order_contexts: dict[str, tuple[dict[str, Any], ...]] = {}
        protection_algo_rows: list[dict[str, Any]] = []
        instrument_contract_sizes: dict[str, float] = {}
        fetch_errors: list[dict[str, Any]] = []
        pull_report = await self._pull_exchange_facts(
            symbols=symbols,
            since=since,
            target_inst_ids=target_inst_ids,
            target_order_ids=target_order_ids,
            priority_order_ids=priority_context_order_ids,
        )
        exchange_positions = pull_report["exchange_positions"]
        exchange_fills = pull_report["exchange_fills"]
        exchange_order_contexts = pull_report["exchange_order_contexts"]
        protection_algo_rows = pull_report["protection_algo_rows"]
        instrument_contract_sizes = pull_report["instrument_contract_sizes"]
        fetch_errors = pull_report["fetch_errors"]
        pull_attempts = pull_report["pull_attempts"]
        pull_success_attempt = pull_report["pull_success_attempt"]
        pull_stages = pull_report["pull_stages"]
        exchange_fill_order_ids = {
            str(fill.order_id or "").strip()
            for fill in exchange_fills
            if str(fill.order_id or "").strip()
        }
        supplemental_orders, supplemental_decisions = (
            await self._load_filled_local_orders_by_exchange_ids(
                exchange_fill_order_ids,
                known_exchange_order_ids=target_order_ids,
            )
        )
        if supplemental_orders:
            local_orders = [*local_orders, *supplemental_orders]
            local_decisions.update(supplemental_decisions)
            target_order_ids.update(
                token
                for order in supplemental_orders
                for token in _split_exchange_order_ids(order.exchange_order_id)
            )
        supplemental_positions = await self._load_local_positions_by_exchange_ids(
            exchange_fill_order_ids,
            known_position_order_ids=_linked_position_order_ids(local_positions),
        )
        if supplemental_positions:
            local_positions = [*local_positions, *supplemental_positions]
        context_orders, context_decisions = await self._load_context_local_orders(
            _order_ids_from_order_history_contexts(exchange_order_contexts),
            known_exchange_order_ids=target_order_ids,
        )

        findings = (
            []
            if fetch_errors
            else self._diff_facts(
                local_orders=local_orders,
                local_positions=local_positions,
                local_decisions=local_decisions,
                exchange_positions=exchange_positions,
                exchange_fills=exchange_fills,
                exchange_order_contexts=exchange_order_contexts,
                protection_algo_rows=protection_algo_rows,
                instrument_contract_sizes=instrument_contract_sizes,
                context_local_orders=context_orders,
                context_local_decisions=context_decisions,
                observed_at=started_at,
            )
        )
        observations = [
            item for item in findings if item.classification == "observation"
        ]
        issues = [item for item in findings if item.classification != "observation"]
        classification_counts = Counter(issue.classification for issue in issues)
        severity_counts = Counter(issue.severity for issue in issues)
        status = "warning" if issues or fetch_errors else "ok"
        if severity_counts.get("critical"):
            status = "critical"
        if fetch_errors and not exchange_positions and not exchange_fills:
            status = "warning"

        return {
            "status": status,
            "read_only": True,
            "audit_only": True,
            "source": "okx_private_api",
            "mode": self.mode,
            "lookback_hours": self.lookback_hours,
            "limit": self.limit,
            "checked_at": datetime.now(UTC).isoformat(),
            "nominal_okx_fill_window_start": nominal_since.isoformat(),
            "okx_fill_window_start": since.isoformat(),
            "effective_fill_window_start": since.isoformat(),
            "cold_start_watermark_applied": bool(watermark.get("applied")),
            "cold_start_reset_at": _iso(watermark.get("reset_at")),
            "cold_start_marker_path": watermark.get("path"),
            "cold_start_marker_error": watermark.get("error"),
            "duration_seconds": round((datetime.now(UTC) - started_at).total_seconds(), 6),
            "local_order_count": len(local_orders),
            "local_position_count": len(local_positions),
            "okx_position_count": len(exchange_positions),
            "okx_fill_order_count": len(exchange_fills),
            "okx_fill_max_pages": MAX_AUTHORITATIVE_FILL_PAGES,
            "okx_order_history_context_count": len(exchange_order_contexts),
            "okx_protection_algo_history_count": len(protection_algo_rows),
            "supplemental_local_order_count": len(supplemental_orders),
            "supplemental_local_position_count": len(supplemental_positions),
            "context_local_order_count": len(context_orders),
            "okx_pull_available": not fetch_errors,
            "pull_attempts": pull_attempts,
            "pull_success_attempt": pull_success_attempt,
            "pull_stages": pull_stages,
            "fetch_errors": fetch_errors,
            "issue_count": len(issues),
            "observation_count": len(observations),
            "pending_local_order_sync_count": sum(
                1
                for item in observations
                if item.kind == "okx_fill_pending_local_order_sync"
            ),
            "classification_counts": dict(classification_counts),
            "severity_counts": dict(severity_counts),
            "repairable_count": int(classification_counts.get("repairable", 0)),
            "manual_review_count": int(classification_counts.get("manual_review", 0)),
            "skipped_count": int(classification_counts.get("skipped", 0)),
            "issues": [issue.as_dict() for issue in issues[:50]],
            "observations": [item.as_dict() for item in observations[:50]],
            "okx_position_samples": [
                _safe_position_sample(row) for row in exchange_positions[:10]
            ],
            "okx_fill_samples": [fill.as_dict() for fill in exchange_fills[:10]],
            "apply_policy": {
                "can_write_database": False,
                "requires_allowlisted_apply": True,
                "requires_backup": True,
                "repair_entrypoint": "scripts/repair_okx_history_position_reconciliation.py",
                "linked_protection_repair_entrypoint": (
                    "scripts/repair_missing_position_links_from_okx_fills.py "
                    "--create-linked-protection-fill-orders"
                ),
                "training_policy": "only_okx_backed_clean_trade_facts",
            },
        }

    async def _pull_exchange_facts(
        self,
        *,
        symbols: set[str],
        since: datetime,
        target_inst_ids: set[str],
        target_order_ids: set[str],
        priority_order_ids: set[str],
    ) -> dict[str, Any]:
        last_errors: list[dict[str, Any]] = []
        all_stages: list[dict[str, Any]] = []
        for attempt in range(1, self.max_pull_attempts + 1):
            executor = self.executor_factory(
                mode=self.mode,
                load_markets_on_initialize=False,
            )
            shutdown_errors: list[dict[str, Any]] = []
            attempt_stages: list[dict[str, Any]] = []
            try:
                (
                    exchange_positions,
                    exchange_fills,
                    exchange_order_contexts,
                    protection_algo_rows,
                    contract_sizes,
                ) = await self._pull_once(
                    executor,
                    symbols=symbols,
                    since=since,
                    target_inst_ids=target_inst_ids,
                    target_order_ids=target_order_ids,
                    priority_order_ids=priority_order_ids,
                    attempt=attempt,
                    stages=attempt_stages,
                )
                all_stages.extend(attempt_stages)
                return {
                    "exchange_positions": exchange_positions,
                    "exchange_fills": exchange_fills,
                    "exchange_order_contexts": exchange_order_contexts,
                    "protection_algo_rows": protection_algo_rows,
                    "instrument_contract_sizes": contract_sizes,
                    "fetch_errors": shutdown_errors,
                    "pull_attempts": attempt,
                    "pull_success_attempt": attempt,
                    "pull_stages": all_stages,
                }
            except Exception as exc:
                all_stages.extend(attempt_stages)
                last_errors = [
                    {
                        "stage": _last_failed_stage(attempt_stages),
                        "attempt": str(attempt),
                        "error": safe_error_text(exc, limit=180),
                    }
                ]
                if attempt < self.max_pull_attempts:
                    await asyncio.sleep(min(0.25 * attempt, 0.75))
            finally:
                try:
                    await executor.shutdown()
                except Exception as exc:
                    shutdown_errors.append(
                        {
                            "stage": "okx_shutdown",
                            "attempt": str(attempt),
                            "error": safe_error_text(exc, limit=120),
                        }
                    )
                    all_stages.append(
                        {
                            "stage": "okx_shutdown",
                            "attempt": attempt,
                            "status": "error",
                            "error": safe_error_text(exc, limit=120),
                        }
                    )
        return {
            "exchange_positions": [],
            "exchange_fills": [],
            "exchange_order_contexts": {},
            "protection_algo_rows": [],
            "instrument_contract_sizes": {},
            "fetch_errors": last_errors,
            "pull_attempts": self.max_pull_attempts,
            "pull_success_attempt": None,
            "pull_stages": all_stages,
        }

    async def _pull_once(
        self,
        executor: Any,
        *,
        symbols: set[str],
        since: datetime,
        target_inst_ids: set[str],
        target_order_ids: set[str],
        priority_order_ids: set[str],
        attempt: int,
        stages: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        list[OkxFillGroup],
        dict[str, tuple[dict[str, Any], ...]],
        list[dict[str, Any]],
        dict[str, float],
    ]:
        await _timed_stage(
            stages,
            attempt=attempt,
            stage="okx_initialize",
            timeout_seconds=self.timeout_seconds,
            operation=executor.initialize(),
        )
        exchange_positions = await _timed_stage(
            stages,
            attempt=attempt,
            stage="okx_positions",
            timeout_seconds=self.timeout_seconds,
            operation=self._fetch_positions(executor),
        )
        exchange_fills = await _timed_stage(
            stages,
            attempt=attempt,
            stage="okx_fills",
            timeout_seconds=self.timeout_seconds,
            operation=self._fetch_fills(
                executor,
                symbols=symbols,
                since=since,
                target_order_ids=target_order_ids,
            ),
        )
        optional_timeout = min(self.timeout_seconds, 3.0)
        exchange_order_contexts = await _optional_timed_stage(
            stages,
            attempt=attempt,
            stage="okx_order_history_contexts",
            timeout_seconds=optional_timeout,
            operation=self._fetch_order_history_contexts(
                executor,
                exchange_fills=exchange_fills,
                local_exchange_order_ids=target_order_ids,
                priority_order_ids=priority_order_ids,
            ),
            default={},
        )
        algo_ids = {
            str(row.get("algoId") or "").strip()
            for rows in exchange_order_contexts.values()
            for row in rows
            if str(row.get("algoId") or "").strip()
        }
        protection_algo_rows = await _optional_timed_stage(
            stages,
            attempt=attempt,
            stage="okx_protection_algo_history",
            timeout_seconds=optional_timeout,
            operation=OkxNativeFactsClient(executor).fetch_protection_algo_history_rows(
                algo_ids=algo_ids,
                order_ids={fill.order_id for fill in exchange_fills if fill.order_id},
                inst_ids={fill.inst_id for fill in exchange_fills if fill.inst_id},
                since=since,
                limit=100,
                max_pages=MAX_AUTHORITATIVE_FILL_PAGES,
                strict=True,
            ),
            default=[],
        )
        instrument_contract_sizes: dict[str, float] = {}
        if symbols or target_inst_ids:
            instrument_contract_sizes = await _optional_timed_stage(
                stages,
                attempt=attempt,
                stage="okx_contract_sizes",
                timeout_seconds=optional_timeout,
                operation=self._fetch_contract_sizes(
                    executor,
                    symbols=symbols,
                    inst_ids=target_inst_ids,
                ),
                default={},
            )
        return (
            exchange_positions,
            exchange_fills,
            exchange_order_contexts,
            protection_algo_rows,
            instrument_contract_sizes,
        )

    def _load_cold_start_watermark(self) -> dict[str, Any]:
        if self.mode != "paper" or self.cold_start_marker_path is None:
            return {"applied": False, "reset_at": None}
        marker_path = self.cold_start_marker_path
        if not marker_path.exists():
            return {"applied": False, "reset_at": None, "path": str(marker_path)}
        try:
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
            if str(payload.get("mode") or "paper") != "paper":
                return {"applied": False, "reset_at": None, "path": str(marker_path)}
            reset_at = _parse_datetime(payload.get("reset_at"))
            if reset_at is None:
                return {
                    "applied": False,
                    "reset_at": None,
                    "path": str(marker_path),
                    "error": "missing_or_invalid_reset_at",
                }
            return {"applied": True, "reset_at": reset_at, "path": str(marker_path)}
        except Exception as exc:
            return {
                "applied": False,
                "reset_at": None,
                "path": str(marker_path),
                "error": safe_error_text(exc, limit=120),
            }

    async def _load_local_facts(
        self,
        since_naive: datetime,
    ) -> tuple[list[Order], list[Position], dict[int, AIDecision]]:
        async with get_read_session_ctx() as session:
            order_rows = await session.execute(
                select(Order)
                .where(
                    Order.execution_mode == self.mode,
                    Order.status == "filled",
                    or_(Order.created_at >= since_naive, Order.filled_at >= since_naive),
                )
                .order_by(Order.filled_at.desc(), Order.created_at.desc())
                .limit(self.limit)
            )
            local_orders = list(order_rows.scalars().all())
            decision_ids = {
                int(order.decision_id)
                for order in local_orders
                if getattr(order, "decision_id", None)
            }
            local_decisions: dict[int, AIDecision] = {}
            if decision_ids:
                decision_rows = await session.execute(
                    select(AIDecision).where(AIDecision.id.in_(decision_ids))
                )
                local_decisions = {
                    int(decision.id): decision for decision in decision_rows.scalars().all()
                }
            position_rows = await session.execute(
                select(Position)
                .where(
                    Position.execution_mode == self.mode,
                    or_(
                        Position.created_at >= since_naive,
                        Position.closed_at >= since_naive,
                        Position.is_open.is_(True),
                    ),
                )
                .order_by(Position.created_at.desc())
                .limit(self.limit)
            )
            local_positions = list(position_rows.scalars().all())
            open_symbol_variants: set[str] = set()
            for order in local_orders:
                if str(getattr(order, "side", "") or "").lower() not in {"buy", "sell"}:
                    continue
                symbol = _local_order_symbol(
                    order,
                    local_decisions.get(int(getattr(order, "decision_id", 0) or 0)),
                )
                variants = trading_symbol_variants(symbol) or {symbol}
                open_symbol_variants.update(item for item in variants if item)
            if open_symbol_variants:
                linked_open_rows = await session.execute(
                    select(Position)
                    .where(
                        Position.execution_mode == self.mode,
                        Position.is_open.is_(True),
                        Position.symbol.in_(open_symbol_variants),
                    )
                    .order_by(Position.created_at.desc())
                    .limit(max(self.limit + len(open_symbol_variants) * 20, self.limit * 10))
                )
                by_id = {int(position.id): position for position in local_positions}
                for position in linked_open_rows.scalars().all():
                    by_id.setdefault(int(position.id), position)
                local_positions = list(by_id.values())
            return local_orders, local_positions, local_decisions

    async def _load_context_local_orders(
        self,
        exchange_order_ids: set[str],
        *,
        known_exchange_order_ids: set[str],
    ) -> tuple[list[Order], dict[int, AIDecision]]:
        target_ids = {
            str(item or "").strip()
            for item in exchange_order_ids
            if str(item or "").strip()
            and str(item or "").strip() not in known_exchange_order_ids
        }
        if not target_ids:
            return [], {}
        clauses = [Order.exchange_order_id.contains(order_id) for order_id in sorted(target_ids)]
        async with get_read_session_ctx() as session:
            order_rows = await session.execute(
                select(Order)
                .where(Order.execution_mode == self.mode, or_(*clauses))
                .order_by(Order.filled_at.desc(), Order.created_at.desc())
                .limit(min(max(len(target_ids) * 2, 1), 200))
            )
            context_orders = list(order_rows.scalars().all())
            decision_ids = {
                int(order.decision_id)
                for order in context_orders
                if getattr(order, "decision_id", None)
            }
            context_decisions: dict[int, AIDecision] = {}
            if decision_ids:
                decision_rows = await session.execute(
                    select(AIDecision).where(AIDecision.id.in_(decision_ids))
                )
                context_decisions = {
                    int(decision.id): decision for decision in decision_rows.scalars().all()
                }
        return context_orders, context_decisions

    async def _load_filled_local_orders_by_exchange_ids(
        self,
        exchange_order_ids: set[str],
        *,
        known_exchange_order_ids: set[str],
    ) -> tuple[list[Order], dict[int, AIDecision]]:
        target_ids = {
            str(item or "").strip()
            for item in exchange_order_ids
            if str(item or "").strip()
            and str(item or "").strip() not in known_exchange_order_ids
        }
        if not target_ids:
            return [], {}

        orders_by_id: dict[int, Order] = {}
        ordered_target_ids = sorted(target_ids)
        async with get_read_session_ctx() as session:
            for offset in range(0, len(ordered_target_ids), 100):
                batch = ordered_target_ids[offset : offset + 100]
                clauses = [Order.exchange_order_id.contains(order_id) for order_id in batch]
                order_rows = await session.execute(
                    select(Order).where(
                        Order.execution_mode == self.mode,
                        Order.status == "filled",
                        or_(*clauses),
                    )
                )
                for order in order_rows.scalars().all():
                    if _split_exchange_order_ids(order.exchange_order_id) & target_ids:
                        orders_by_id[int(order.id)] = order

            context_orders = list(orders_by_id.values())
            decision_ids = {
                int(order.decision_id)
                for order in context_orders
                if getattr(order, "decision_id", None)
            }
            context_decisions: dict[int, AIDecision] = {}
            if decision_ids:
                decision_rows = await session.execute(
                    select(AIDecision).where(AIDecision.id.in_(decision_ids))
                )
                context_decisions = {
                    int(decision.id): decision for decision in decision_rows.scalars().all()
                }
        return context_orders, context_decisions

    async def _load_local_positions_by_exchange_ids(
        self,
        exchange_order_ids: set[str],
        *,
        known_position_order_ids: set[str],
    ) -> list[Position]:
        target_ids = {
            str(item or "").strip()
            for item in exchange_order_ids
            if str(item or "").strip()
            and str(item or "").strip() not in known_position_order_ids
        }
        if not target_ids:
            return []

        positions_by_id: dict[int, Position] = {}
        ordered_target_ids = sorted(target_ids)
        async with get_read_session_ctx() as session:
            for offset in range(0, len(ordered_target_ids), 100):
                batch = ordered_target_ids[offset : offset + 100]
                clauses = [
                    column.contains(order_id)
                    for order_id in batch
                    for column in (
                        Position.entry_exchange_order_id,
                        Position.close_exchange_order_id,
                    )
                ]
                position_rows = await session.execute(
                    select(Position).where(
                        Position.execution_mode == self.mode,
                        or_(*clauses),
                    )
                )
                for position in position_rows.scalars().all():
                    linked_ids = _split_exchange_order_ids(
                        position.entry_exchange_order_id
                    ) | _split_exchange_order_ids(position.close_exchange_order_id)
                    if linked_ids & target_ids:
                        positions_by_id[int(position.id)] = position
        return list(positions_by_id.values())

    async def _fetch_positions(self, executor: Any) -> list[dict[str, Any]]:
        fetch = getattr(executor, "get_positions_strict", None)
        if not callable(fetch):
            raise RuntimeError("OKX authoritative sync requires get_positions_strict")
        rows = await fetch()
        return [row for row in rows or [] if isinstance(row, dict)]

    async def _fetch_fills(
        self,
        executor: Any,
        *,
        symbols: set[str],
        since: datetime,
        target_order_ids: set[str] | None = None,
    ) -> list[OkxFillGroup]:
        groups = await OkxNativeFactsClient(executor).fetch_fill_groups(
            symbols=symbols,
            order_ids=target_order_ids,
            since=since,
            limit=100,
            max_pages=MAX_AUTHORITATIVE_FILL_PAGES,
            account_wide_only=True,
            strict=True,
        )
        return [
            OkxFillGroup(
                order_id=group.order_id,
                trade_ids=group.trade_ids,
                inst_id=group.inst_id,
                symbol=group.symbol,
                side=group.side,
                pos_side=group.pos_side,
                contracts=group.contracts,
                avg_price=group.avg_price,
                fee_abs=group.fee_abs,
                fill_pnl=group.fill_pnl,
                timestamp_ms=group.timestamp_ms,
                timestamp=group.timestamp,
                raw_count=group.raw_count,
                rows=group.rows,
            )
            for group in groups
        ]

    async def _fetch_contract_sizes(
        self,
        executor: Any,
        *,
        symbols: set[str],
        inst_ids: set[str],
    ) -> dict[str, float]:
        return await OkxNativeFactsClient(executor).fetch_contract_sizes(
            symbols=symbols,
            inst_ids=inst_ids,
        )

    async def _fetch_order_history_contexts(
        self,
        executor: Any,
        *,
        exchange_fills: list[OkxFillGroup],
        local_exchange_order_ids: set[str],
        priority_order_ids: set[str] | None = None,
    ) -> dict[str, tuple[dict[str, Any], ...]]:
        fills = [
            fill for fill in exchange_fills if str(getattr(fill, "order_id", "") or "").strip()
        ]
        if not fills:
            return {}
        priority_ids = {str(item or "").strip() for item in priority_order_ids or set() if str(item or "").strip()}
        local_ids = {str(item or "").strip() for item in local_exchange_order_ids if str(item or "").strip()}
        ordered_order_ids: list[str] = []
        seen_order_ids: set[str] = set()
        for fill in fills:
            order_id = str(getattr(fill, "order_id", "") or "").strip()
            if order_id in priority_ids and order_id not in seen_order_ids:
                ordered_order_ids.append(order_id)
                seen_order_ids.add(order_id)
        for order_id in sorted(priority_ids | local_ids):
            if order_id and order_id not in seen_order_ids:
                ordered_order_ids.append(order_id)
                seen_order_ids.add(order_id)
        return await OkxNativeFactsClient(executor).fetch_order_history_contexts(
            fills=fills,
            order_ids=ordered_order_ids,
            limit=5,
            strict=False,
        )

    def _diff_facts(
        self,
        *,
        local_orders: list[Order],
        local_positions: list[Position],
        local_decisions: dict[int, AIDecision],
        exchange_positions: list[dict[str, Any]],
        exchange_fills: list[OkxFillGroup],
        exchange_order_contexts: dict[str, tuple[dict[str, Any], ...]] | None = None,
        protection_algo_rows: list[dict[str, Any]] | None = None,
        instrument_contract_sizes: dict[str, float] | None = None,
        context_local_orders: list[Order] | None = None,
        context_local_decisions: dict[int, AIDecision] | None = None,
        observed_at: datetime | None = None,
    ) -> list[OkxAuthoritativeIssue]:
        issues: list[OkxAuthoritativeIssue] = []
        fills_by_order_id = {fill.order_id: fill for fill in exchange_fills}
        exchange_order_contexts = exchange_order_contexts or {}
        protection_algo_rows = protection_algo_rows or []
        contract_sizes_by_symbol = _contract_sizes_from_exchange_positions(exchange_positions)
        instrument_contract_sizes = instrument_contract_sizes or {}
        context_local_orders = context_local_orders or []
        all_context_orders = [*local_orders, *context_local_orders]
        all_context_decisions = {
            **(context_local_decisions or {}),
            **local_decisions,
        }
        local_orders_by_exchange_id = _local_orders_by_exchange_id(all_context_orders)
        local_exchange_order_ids = {
            token
            for order in local_orders
            for token in _split_exchange_order_ids(order.exchange_order_id)
        }
        linked_position_order_ids = _linked_position_order_ids(local_positions)
        observed_at = observed_at or datetime.now(UTC)

        open_position_keys: set[tuple[str, str]] = set()
        for position in local_positions:
            if bool(getattr(position, "is_open", False)):
                open_position_keys.add(_local_position_key(position))

        exchange_position_keys: set[tuple[str, str]] = set()
        for row in exchange_positions:
            snapshot = parse_exchange_position_snapshot(
                row,
                symbol_normalizer=normalize_trading_symbol,
            )
            if not snapshot:
                continue
            key = (str(snapshot.get("symbol") or ""), str(snapshot.get("side") or ""))
            exchange_position_keys.add(key)
            if key not in open_position_keys:
                issues.append(
                    OkxAuthoritativeIssue(
                        kind="okx_open_position_missing_locally",
                        classification="manual_review",
                        severity="critical",
                        reason=(
                            "OKX reports an open position that has no matching local open "
                            "position in the current audit window."
                        ),
                        symbol=key[0],
                        side=key[1],
                        okx_contracts=_safe_float(snapshot.get("contracts")),
                        okx_contract_size=_safe_float(snapshot.get("contract_size")),
                        expected_base_quantity=_safe_float(snapshot.get("quantity")),
                        okx_price=_safe_float(snapshot.get("mark_price"))
                        or _safe_float(snapshot.get("last_price")),
                    )
                )

        for position in local_positions:
            if not bool(getattr(position, "is_open", False)):
                continue
            key = _local_position_key(position)
            if key not in exchange_position_keys:
                issues.append(
                    OkxAuthoritativeIssue(
                        kind="local_open_position_missing_on_okx",
                        classification="manual_review",
                        severity="critical",
                        reason=(
                            "Local DB has an open position that OKX does not report as open."
                        ),
                        symbol=key[0],
                        side=key[1],
                        local_position_id=int(getattr(position, "id", 0) or 0) or None,
                        local_quantity=_safe_float(getattr(position, "quantity", None)),
                        local_price=_safe_float(getattr(position, "current_price", None)),
                    )
                )

        for order in local_orders:
            order_symbol = _local_order_symbol(
                order,
                local_decisions.get(int(order.decision_id or 0)),
            )
            exchange_ids = _split_exchange_order_ids(order.exchange_order_id)
            if not exchange_ids:
                issues.append(
                    OkxAuthoritativeIssue(
                        kind="local_filled_order_missing_exchange_id",
                        classification="manual_review",
                        severity="warning",
                        reason="Filled local order cannot be matched to OKX without exchange_order_id.",
                        symbol=order_symbol,
                        side=str(order.side or "").lower(),
                        local_order_id=int(order.id),
                        local_quantity=_safe_float(order.quantity),
                        local_price=_safe_float(order.price),
                    )
                )
                continue
            for exchange_order_id in exchange_ids:
                fill = fills_by_order_id.get(exchange_order_id)
                if fill is None:
                    if order_has_current_position_snapshot_confirmation(
                        order,
                        exchange_positions=exchange_positions,
                    ):
                        continue
                    current_position_confirmation = find_current_position_entry_confirmation(
                        order,
                        exchange_order_id=exchange_order_id,
                        exchange_positions=exchange_positions,
                        local_positions=local_positions,
                        contract_sizes=instrument_contract_sizes,
                    )
                    if current_position_confirmation is not None:
                        continue
                    issues.append(
                        OkxAuthoritativeIssue(
                            kind="local_order_not_found_in_recent_okx_fills",
                            classification="skipped",
                            severity="warning",
                            reason=(
                                "Local order id is not present in the bounded OKX fill pull; "
                                "run a wider/order-id targeted pull before repair."
                            ),
                            symbol=order_symbol,
                            side=str(order.side or "").lower(),
                            local_order_id=int(order.id),
                            exchange_order_id=exchange_order_id,
                            local_quantity=_safe_float(order.quantity),
                            local_price=_safe_float(order.price),
                        )
                    )
                    continue
                decision = local_decisions.get(int(order.decision_id or 0))
                contract_size = _local_order_verified_okx_raw_contract_size(order)
                if contract_size <= 0:
                    contract_size = instrument_contract_sizes.get(fill.inst_id, 0.0)
                if contract_size <= 0:
                    fill_inst_id = okx_inst_id_from_symbol(fill.symbol)
                    contract_size = instrument_contract_sizes.get(fill_inst_id, 0.0)
                if contract_size <= 0:
                    contract_size = contract_sizes_by_symbol.get(fill.symbol, 0.0)
                if contract_size <= 0:
                    contract_size = _local_order_okx_raw_contract_size(order)
                if contract_size <= 0:
                    contract_size = _local_order_contract_size(order, decision)
                if not contract_size:
                    continue
                expected_quantity = fill.contracts * contract_size
                local_quantity = _safe_float(order.quantity)
                if (
                    local_quantity > 0
                    and expected_quantity > 0
                    and not _relative_close_enough(
                        local_quantity,
                        expected_quantity,
                        QUANTITY_TOLERANCE_RATIO,
                    )
                ):
                    issues.append(
                        OkxAuthoritativeIssue(
                            kind="local_order_quantity_differs_from_okx_fill",
                            classification="repairable",
                            severity="warning",
                            reason=(
                                "OKX fill contracts converted by ctVal differ from local "
                                "order quantity; this is repairable only after exact order-id review."
                            ),
                            symbol=order_symbol,
                            side=str(order.side or "").lower(),
                            local_order_id=int(order.id),
                            exchange_order_id=exchange_order_id,
                            local_quantity=local_quantity,
                            okx_contracts=fill.contracts,
                            okx_contract_size=contract_size,
                            expected_base_quantity=expected_quantity,
                            local_price=_safe_float(order.price),
                            okx_price=fill.avg_price,
                            okx_timestamp=fill.timestamp,
                        )
                    )

        for fill in exchange_fills:
            if fill.order_id in local_exchange_order_ids:
                continue
            linked_protection = _linked_protection_fill_context(
                fill,
                order_contexts=exchange_order_contexts,
                local_orders_by_exchange_id=local_orders_by_exchange_id,
                local_orders=all_context_orders,
                local_decisions=all_context_decisions,
                protection_algo_rows=protection_algo_rows,
            )
            if linked_protection is not None:
                issues.append(
                    OkxAuthoritativeIssue(
                        kind="okx_linked_protection_fill_missing_local_order",
                        classification="repairable",
                        severity="warning",
                        reason=(
                            "OKX triggered a reduce-only protection order that is linked "
                            "to a local entry order, but the generated close ordId is "
                            "missing from local filled orders."
                        ),
                        symbol=fill.symbol,
                        side=fill.side,
                        local_order_id=linked_protection.get("local_order_id"),
                        exchange_order_id=fill.order_id,
                        okx_contracts=fill.contracts,
                        okx_price=fill.avg_price,
                        okx_timestamp=fill.timestamp,
                        linked_local_order_id=linked_protection.get("local_order_id"),
                        linked_exchange_order_id=str(
                            linked_protection.get("linked_exchange_order_id") or ""
                        ),
                        okx_algo_id=str(linked_protection.get("okx_algo_id") or ""),
                        okx_source=str(linked_protection.get("okx_source") or ""),
                        repair_entrypoint=(
                            "scripts/repair_missing_position_links_from_okx_fills.py "
                            "--create-linked-protection-fill-orders"
                        ),
                        protection_execution=linked_protection.get(
                            "protection_execution"
                        ),
                    )
                )
                continue
            if _is_pending_local_order_sync(fill, observed_at=observed_at):
                issues.append(
                    OkxAuthoritativeIssue(
                        kind="okx_fill_pending_local_order_sync",
                        classification="observation",
                        severity="info",
                        reason=(
                            "OKX fill is within the local order-fact synchronization window. "
                            "It is observed while the background sync persists the local order; "
                            "it becomes an integrity issue only after that window expires."
                        ),
                        symbol=fill.symbol,
                        side=fill.side,
                        exchange_order_id=fill.order_id,
                        okx_contracts=fill.contracts,
                        okx_price=fill.avg_price,
                        okx_timestamp=fill.timestamp,
                    )
                )
                continue
            issues.append(
                OkxAuthoritativeIssue(
                    kind="okx_fill_missing_local_order",
                    classification="manual_review",
                    severity="critical",
                    reason=(
                        "OKX has a recent fill that is not represented by a local filled order."
                    ),
                    symbol=fill.symbol,
                    side=fill.side,
                    exchange_order_id=fill.order_id,
                    okx_contracts=fill.contracts,
                    okx_price=fill.avg_price,
                    okx_timestamp=fill.timestamp,
                )
            )
        for fill in exchange_fills:
            if fill.order_id in linked_position_order_ids:
                continue
            if fill.order_id in local_exchange_order_ids:
                if _fill_covered_by_local_position_lifecycle(fill, local_positions):
                    continue
                linked_protection = _linked_protection_fill_context(
                    fill,
                    order_contexts=exchange_order_contexts,
                    local_orders_by_exchange_id=local_orders_by_exchange_id,
                    local_orders=all_context_orders,
                    local_decisions=all_context_decisions,
                    protection_algo_rows=protection_algo_rows,
                )
                if linked_protection is not None:
                    continue
                issues.append(
                    OkxAuthoritativeIssue(
                        kind="okx_fill_not_linked_to_position",
                        classification="manual_review",
                        severity="warning",
                        reason=(
                            "OKX-backed local order exists, but no position entry/close link "
                            "references this exchange order id."
                        ),
                        symbol=fill.symbol,
                        side=fill.side,
                        exchange_order_id=fill.order_id,
                        okx_contracts=fill.contracts,
                        okx_price=fill.avg_price,
                        okx_timestamp=fill.timestamp,
                    )
                )
        return issues


def _is_pending_local_order_sync(
    fill: OkxFillGroup,
    *,
    observed_at: datetime,
) -> bool:
    timestamp = fill.timestamp
    if timestamp is None:
        return False
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=UTC)
    age_seconds = (observed_at - timestamp).total_seconds()
    return -5.0 <= age_seconds <= LOCAL_ORDER_SYNC_GRACE_SECONDS


def _local_orders_by_exchange_id(local_orders: list[Order]) -> dict[str, Order]:
    result: dict[str, Order] = {}
    for order in local_orders:
        for exchange_order_id in _split_exchange_order_ids(order.exchange_order_id):
            result.setdefault(exchange_order_id, order)
    return result


def _order_ids_from_order_history_contexts(
    order_contexts: dict[str, tuple[dict[str, Any], ...]],
) -> set[str]:
    result: set[str] = set()
    for rows in order_contexts.values():
        for row in rows:
            if not isinstance(row, dict):
                continue
            order_id = str(row.get("ordId") or "").strip()
            if order_id:
                result.add(order_id)
    return result


def _linked_position_order_ids(local_positions: list[Position]) -> set[str]:
    return {
        token
        for position in local_positions
        for token in (
            _split_exchange_order_ids(getattr(position, "entry_exchange_order_id", None))
            | _split_exchange_order_ids(getattr(position, "close_exchange_order_id", None))
        )
    }


def _fill_covered_by_local_position_lifecycle(
    fill: OkxFillGroup,
    local_positions: list[Position],
) -> bool:
    order_id = str(getattr(fill, "order_id", "") or "").strip()
    inst_id = str(getattr(fill, "inst_id", "") or "").strip().upper()
    if not order_id or not inst_id or fill.timestamp is None:
        return False
    fill_time = _parse_datetime(fill.timestamp)
    if fill_time is None:
        return False
    fill_side = str(getattr(fill, "side", "") or "").lower().strip()
    symbol = symbol_from_okx_inst_id(inst_id) or normalize_trading_symbol(inst_id)
    for position in local_positions:
        if bool(getattr(position, "is_open", False)):
            continue
        position_inst_id = (
            str(getattr(position, "okx_inst_id", "") or "").strip().upper()
            or okx_inst_id_from_symbol(str(getattr(position, "symbol", "") or ""))
            or ""
        )
        if position_inst_id != inst_id:
            continue
        position_symbol = normalize_trading_symbol(getattr(position, "symbol", None))
        if position_symbol and symbol and position_symbol != symbol:
            continue
        opened_at = _parse_datetime(getattr(position, "created_at", None))
        closed_at = _parse_datetime(getattr(position, "closed_at", None))
        if opened_at is None or closed_at is None:
            continue
        window_start = opened_at - timedelta(seconds=180)
        window_end = closed_at + timedelta(seconds=180)
        if fill_time < window_start or fill_time > window_end:
            continue
        side = str(getattr(position, "side", "") or "").lower().strip()
        if side == "short" and fill_side in {"sell", "buy"}:
            return True
        if side == "long" and fill_side in {"buy", "sell"}:
            return True
    return False


def _linked_protection_fill_context(
    fill: OkxFillGroup,
    *,
    order_contexts: dict[str, tuple[dict[str, Any], ...]],
    local_orders_by_exchange_id: dict[str, Order],
    local_orders: list[Order],
    local_decisions: dict[int, AIDecision],
    protection_algo_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    rows = list(order_contexts.get(str(fill.order_id)) or ())
    if not rows:
        return None
    target = _order_history_row_for_order_id(rows, fill.order_id)
    if target is None:
        return None
    if not _native_bool(target.get("reduceOnly")):
        return None

    target_inst_id = str(target.get("instId") or fill.inst_id or "").strip().upper()
    target_side = str(target.get("side") or fill.side or "").lower().strip()
    okx_algo_id = str(target.get("algoId") or "").strip()
    okx_source = str(target.get("source") or "").strip()
    looks_like_triggered_protection = bool(
        okx_algo_id
        or okx_source == "7"
        or str(target.get("clOrdId") or "").startswith("O")
    )
    if not looks_like_triggered_protection:
        return None
    algo_rows = protection_algo_rows or []
    algo_row = next(
        (
            row
            for row in algo_rows
            if str(row.get("ordId") or "").strip() == str(fill.order_id)
        ),
        None,
    )
    if algo_row is None and okx_algo_id:
        algo_row = next(
            (
                row
                for row in algo_rows
                if str(row.get("algoId") or row.get("algoClOrdId") or "").strip()
                == okx_algo_id
            ),
            None,
        )
    protection_execution = (
        build_okx_protection_execution_lifecycle(
            fill,
            order_row=target,
            algo_row=algo_row,
        )
        if isinstance(algo_row, dict)
        else None
    )

    for row in rows:
        source_order_id = str(row.get("ordId") or "").strip()
        if not source_order_id or source_order_id == str(fill.order_id):
            continue
        source_order = local_orders_by_exchange_id.get(source_order_id)
        if source_order is None:
            continue
        if not _row_has_attach_algo(row):
            continue
        if not _same_okx_inst_id(target_inst_id, str(row.get("instId") or "")):
            continue
        if not _is_reduce_only_close_side(
            close_side=target_side,
            entry_side=str(getattr(source_order, "side", "") or ""),
        ):
            continue
        return {
            "local_order_id": int(source_order.id),
            "linked_exchange_order_id": source_order_id,
            "okx_algo_id": okx_algo_id,
            "okx_source": okx_source,
            "match_source": "okx_order_history_context",
            "protection_execution": protection_execution,
        }

    if okx_algo_id:
        for order in local_orders:
            decision = local_decisions.get(int(getattr(order, "decision_id", 0) or 0))
            if not _decision_has_attach_algo_id(decision, okx_algo_id):
                continue
            if not _is_reduce_only_close_side(
                close_side=target_side,
                entry_side=str(getattr(order, "side", "") or ""),
            ):
                continue
            linked_exchange_ids = _split_exchange_order_ids(getattr(order, "exchange_order_id", None))
            return {
                "local_order_id": int(order.id),
                "linked_exchange_order_id": next(iter(linked_exchange_ids), ""),
                "okx_algo_id": okx_algo_id,
                "okx_source": okx_source,
                "match_source": "local_decision_attach_algo_id",
                "protection_execution": protection_execution,
            }
    return None


def _order_history_row_for_order_id(
    rows: list[dict[str, Any]],
    order_id: str,
) -> dict[str, Any] | None:
    expected = str(order_id or "").strip()
    for row in rows:
        if str(row.get("ordId") or "").strip() == expected:
            return row
    return None


def _row_has_attach_algo(row: dict[str, Any]) -> bool:
    attached = row.get("attachAlgoOrds")
    if isinstance(attached, list):
        for item in attached:
            if not isinstance(item, dict):
                continue
            if str(item.get("attachAlgoId") or "").strip():
                return True
            if str(item.get("tpTriggerPx") or item.get("slTriggerPx") or "").strip():
                return True
    return False


def _decision_has_attach_algo_id(decision: AIDecision | None, algo_id: str) -> bool:
    expected = str(algo_id or "").strip()
    if not expected or decision is None:
        return False
    raw = getattr(decision, "raw_llm_response", None)
    return _payload_has_attach_algo_id(raw, expected)


def _payload_has_attach_algo_id(value: Any, expected: str, *, depth: int = 0) -> bool:
    if depth > 8:
        return False
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) == "attachAlgoId" and str(item or "").strip() == expected:
                return True
            if _payload_has_attach_algo_id(item, expected, depth=depth + 1):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_payload_has_attach_algo_id(item, expected, depth=depth + 1) for item in value)
    return False


def _same_okx_inst_id(left: str, right: str) -> bool:
    left_text = str(left or "").strip().upper()
    right_text = str(right or "").strip().upper()
    return bool(left_text and right_text and left_text == right_text)


def _is_reduce_only_close_side(*, close_side: str, entry_side: str) -> bool:
    close = str(close_side or "").lower().strip()
    entry = str(entry_side or "").lower().strip()
    return (entry == "buy" and close == "sell") or (entry == "sell" and close == "buy")


def _native_bool(value: Any) -> bool:
    text = str(value or "").lower().strip()
    return text in {"true", "1", "yes"}


def _safe_position_sample(row: dict[str, Any]) -> dict[str, Any]:
    snapshot = parse_exchange_position_snapshot(
        row,
        symbol_normalizer=normalize_trading_symbol,
    )
    if not snapshot:
        return {"raw_symbol": str(row.get("symbol") or _safe_dict(row.get("info")).get("instId"))}
    return {
        "symbol": snapshot.get("symbol"),
        "side": snapshot.get("side"),
        "quantity": _round_optional(snapshot.get("quantity")),
        "contracts": _round_optional(snapshot.get("contracts")),
        "contract_size": _round_optional(snapshot.get("contract_size")),
        "mark_price": _round_optional(snapshot.get("mark_price")),
        "entry_price": _round_optional(snapshot.get("entry_price")),
        "upl": _round_optional(snapshot.get("upl")),
        "raw_symbol": snapshot.get("raw_symbol"),
    }


async def _timed_stage(
    stages: list[dict[str, Any]],
    *,
    attempt: int,
    stage: str,
    timeout_seconds: float,
    operation: Any,
) -> Any:
    started_at = datetime.now(UTC)
    record: dict[str, Any] = {
        "stage": stage,
        "attempt": attempt,
        "timeout_seconds": round(float(timeout_seconds), 3),
        "started_at": started_at.isoformat(),
    }
    try:
        result = await asyncio.wait_for(operation, timeout=timeout_seconds)
    except Exception as exc:
        record.update(
            {
                "status": "error",
                "duration_seconds": round((datetime.now(UTC) - started_at).total_seconds(), 6),
                "error": safe_error_text(exc, limit=180),
                "error_type": type(exc).__name__,
            }
        )
        stages.append(record)
        raise
    record.update(
        {
            "status": "ok",
            "duration_seconds": round((datetime.now(UTC) - started_at).total_seconds(), 6),
            "result_count": _result_count(result),
        }
    )
    stages.append(record)
    return result


async def _optional_timed_stage(
    stages: list[dict[str, Any]],
    *,
    attempt: int,
    stage: str,
    timeout_seconds: float,
    operation: Any,
    default: Any,
) -> Any:
    started_at = datetime.now(UTC)
    record: dict[str, Any] = {
        "stage": stage,
        "attempt": attempt,
        "timeout_seconds": round(float(timeout_seconds), 3),
        "started_at": started_at.isoformat(),
        "optional": True,
    }
    try:
        result = await asyncio.wait_for(operation, timeout=timeout_seconds)
    except Exception as exc:
        record.update(
            {
                "status": "warning",
                "duration_seconds": round((datetime.now(UTC) - started_at).total_seconds(), 6),
                "error": safe_error_text(exc, limit=180),
                "error_type": type(exc).__name__,
                "default_applied": True,
            }
        )
        stages.append(record)
        return default
    record.update(
        {
            "status": "ok",
            "duration_seconds": round((datetime.now(UTC) - started_at).total_seconds(), 6),
            "result_count": _result_count(result),
        }
    )
    stages.append(record)
    return result


def _last_failed_stage(stages: list[dict[str, Any]]) -> str:
    for item in reversed(stages):
        if str(item.get("status") or "") == "error":
            stage = str(item.get("stage") or "").strip()
            if stage:
                return stage
    return "okx_authoritative_pull"


def _result_count(value: Any) -> int | None:
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return None


def _local_position_symbol(position: Position) -> str:
    okx_inst_id = str(getattr(position, "okx_inst_id", "") or "").strip().upper()
    if okx_inst_id:
        symbol = symbol_from_okx_inst_id(okx_inst_id)
        if symbol:
            return symbol
    return normalize_trading_symbol(getattr(position, "symbol", ""))


def _local_order_symbol(order: Order, decision: AIDecision | None = None) -> str:
    okx_inst_id = _local_order_okx_inst_id(order, decision)
    if okx_inst_id:
        symbol = symbol_from_okx_inst_id(okx_inst_id)
        if symbol:
            return symbol
    return normalize_trading_symbol(getattr(order, "symbol", ""))


def _local_order_okx_inst_id(order: Order, decision: AIDecision | None = None) -> str:
    raw = getattr(decision, "raw_llm_response", None) if decision is not None else None
    payloads: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        payloads.append(raw)
        execution_result = raw.get("execution_result")
        if isinstance(execution_result, dict):
            payloads.append(execution_result)
            raw_response = execution_result.get("raw_response")
            if isinstance(raw_response, dict):
                payloads.append(raw_response)
    for payload in payloads:
        inst_id = okx_inst_id_from_payload(payload, include_fallback=False)
        if inst_id:
            return inst_id
    return ""


def _local_position_key(position: Position) -> tuple[str, str]:
    return (
        _local_position_symbol(position),
        str(getattr(position, "side", "") or "").lower(),
    )


def _contract_sizes_from_exchange_positions(
    exchange_positions: list[dict[str, Any]],
) -> dict[str, float]:
    sizes: dict[str, float] = {}
    for row in exchange_positions:
        snapshot = parse_exchange_position_snapshot(
            row,
            symbol_normalizer=normalize_trading_symbol,
        )
        if not snapshot:
            continue
        symbol = str(snapshot.get("symbol") or "")
        contract_size = _safe_float(snapshot.get("contract_size"))
        if symbol and contract_size > 0:
            sizes[symbol] = contract_size
    return sizes


def _local_order_contract_size(order: Order, decision: AIDecision | None) -> float:
    payloads = _local_order_execution_payloads(decision)
    for payload in payloads:
        contract_size = _first_positive(
            payload.get("contract_size"),
            payload.get("contractSize"),
            _nested(payload, "info", "ctVal"),
            _nested(payload, "info", "contractSize"),
            default=0.0,
        )
        if contract_size > 0:
            return contract_size

    local_quantity = _safe_float(getattr(order, "quantity", None))
    for payload in payloads:
        contracts = _first_positive(
            payload.get("filled_contracts"),
            payload.get("order_contracts"),
            payload.get("filled"),
            payload.get("amount"),
            _nested(payload, "info", "accFillSz"),
            _nested(payload, "info", "fillSz"),
            _nested(payload, "info", "sz"),
            default=0.0,
        )
        base_quantity = _first_positive(
            payload.get("base_quantity"),
            payload.get("planned_base_quantity"),
            payload.get("quantity"),
            default=0.0,
        )
        if base_quantity <= 0:
            base_quantity = local_quantity
        if contracts > 0 and base_quantity > 0:
            return base_quantity / contracts
    return 0.0


def _local_order_okx_raw_contract_size(order: Order) -> float:
    raw = _safe_dict(getattr(order, "okx_raw_fills", None))
    if not raw:
        return 0.0
    raw_order_id = str(raw.get("order_id") or "").strip()
    exchange_ids = _split_exchange_order_ids(getattr(order, "exchange_order_id", None))
    if raw_order_id and exchange_ids and raw_order_id not in exchange_ids:
        return 0.0
    contract_size = _first_positive(
        raw.get("contract_size"),
        raw.get("contractSize"),
        _nested(raw, "info", "ctVal"),
        _nested(raw, "info", "contractSize"),
        default=0.0,
    )
    if contract_size > 0:
        return contract_size
    contracts = _first_positive(
        raw.get("contracts"),
        raw.get("filled_contracts"),
        raw.get("fillSz"),
        raw.get("sz"),
        default=0.0,
    )
    if contracts <= 0:
        contracts = _safe_float(getattr(order, "okx_fill_contracts", None))
    base_quantity = _first_positive(
        raw.get("base_quantity"),
        raw.get("filled_base_quantity"),
        raw.get("quantity"),
        default=0.0,
    )
    if contracts > 0 and base_quantity > 0:
        return base_quantity / contracts
    return 0.0


def _local_order_verified_okx_raw_contract_size(order: Order) -> float:
    """Return the order-specific contract size only when sync verified it."""

    raw = _safe_dict(getattr(order, "okx_raw_fills", None))
    if raw.get("contract_size_verified") is not True:
        return 0.0
    if raw.get("fills_history_confirmed") is not True:
        return 0.0
    return _local_order_okx_raw_contract_size(order)


def _local_order_execution_payloads(decision: AIDecision | None) -> list[dict[str, Any]]:
    raw = getattr(decision, "raw_llm_response", None) if decision is not None else None
    payloads: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        payloads.append(raw)
        execution_result = raw.get("execution_result")
        if isinstance(execution_result, dict):
            payloads.append(execution_result)
            raw_response = execution_result.get("raw_response")
            if isinstance(raw_response, dict):
                payloads.append(raw_response)
    return payloads


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_positive(*values: Any, default: float = 0.0) -> float:
    for value in values:
        number = _safe_float(value, 0.0)
        if number > 0:
            return number
    return default


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


def _relative_close_enough(left: float, right: float, tolerance: float) -> bool:
    denominator = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / denominator <= tolerance


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            result = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _round(value: float) -> float:
    return round(float(value), 8)


def _round_optional(value: Any) -> float | None:
    if value is None:
        return None
    return _round(_safe_float(value))


def _iso(value: Any) -> str | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()
