#!/usr/bin/env python3
"""Backfill missing position OKX links from authoritative OKX fills history."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_env_bootstrap import (  # noqa: E402
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

load_runtime_env_files(project_root=ROOT)
drop_privileges_to_runtime_user_if_needed(project_root=ROOT)

from sqlalchemy import or_, select  # noqa: E402

from config.settings import settings  # noqa: E402
from core.symbols import normalize_trading_symbol, symbol_query_variants  # noqa: E402
from db.session import get_session_ctx  # noqa: E402
from executor.okx_executor import OKXExecutor  # noqa: E402
from models.decision import AIDecision  # noqa: E402
from models.learning import TradeReflection  # noqa: E402
from models.trade import Order, Position  # noqa: E402
from scripts.repair_okx_native_full_close_fills import (  # noqa: E402
    FillGroup,
    _fetch_okx_fill_groups,
)
from services.okx_authoritative_sync import (  # noqa: E402
    _linked_protection_fill_context,
    _local_orders_by_exchange_id,
)
from services.okx_native_facts import OkxNativeFactsClient  # noqa: E402

DEFAULT_DAYS = 30
DEFAULT_WINDOW_SECONDS = 180
DEFAULT_DECISION_WINDOW_SECONDS = 600
DEFAULT_LIMIT = 100
QUANTITY_TOLERANCE_RATIO = 0.05
BACKUP_DIR = Path("data/codex_backups/missing-position-links-from-okx-fills")
REPAIR_REFLECTION_SOURCE = "okx_position_link_repair"
ORDER_DECISION_LINEAGE_REPAIR_SOURCE = "okx_order_decision_lineage_repair"
ORPHAN_QUARANTINE_REFLECTION_SOURCE = "okx_orphan_position_quarantine"
ORPHAN_QUARANTINE_CLOSE_PREFIX = "okx_orphan_quarantine:"
TRUSTED_CLOSE_ORDER_SYNC_STATUSES = {"okx_confirmed", "okx_only_backfilled"}


@dataclass(frozen=True, slots=True)
class FillLinkPlan:
    position_id: int
    link_kind: str
    symbol: str
    side: str
    quantity: float
    okx_order_id: str
    old_entry_exchange_order_id: str | None
    old_close_exchange_order_id: str | None
    old_okx_inst_id: str | None
    fill_timestamp: datetime | None
    position_reference_time: datetime | None
    time_delta_seconds: float | None
    fill_quantity: float
    fill_contracts: float
    fill_price: float
    source: str
    okx_inst_id: str = ""
    fill_fee: float = 0.0
    fill_pnl: float = 0.0
    contract_size: float = 0.0
    contract_size_source: str = ""


@dataclass(frozen=True, slots=True)
class MissingOrderRowPlan:
    position_id: int
    link_kind: str
    symbol: str
    side: str
    model_name: str
    execution_mode: str
    exchange_order_id: str
    quantity: float
    price: float
    fee: float
    filled_at: datetime | None
    source: str
    okx_inst_id: str = ""
    decision_id: int | None = None
    decision_match_source: str = ""


@dataclass(frozen=True, slots=True)
class ExistingOrderDecisionLinkPlan:
    position_id: int
    order_id: int
    exchange_order_id: str
    symbol: str
    side: str
    model_name: str
    execution_mode: str
    decision_id: int
    decision_symbol: str
    decision_action: str
    order_filled_at: datetime | None
    decision_executed_at: datetime | None
    position_created_at: datetime | None
    order_decision_delta_seconds: float | None
    position_order_delta_seconds: float | None
    source: str = ORDER_DECISION_LINEAGE_REPAIR_SOURCE


@dataclass(frozen=True, slots=True)
class OpenPositionClosePlan:
    position_id: int
    symbol: str
    side: str
    close_side: str
    model_name: str
    execution_mode: str
    okx_order_id: str
    old_is_open: bool
    old_close_exchange_order_id: str | None
    old_okx_inst_id: str | None
    quantity: float
    fill_quantity: float
    fill_contracts: float
    contract_size: float
    contract_size_source: str
    entry_price: float
    exit_price: float
    close_fee: float
    fill_pnl: float
    computed_realized_pnl: float
    old_realized_pnl: float
    old_current_price: float | None
    fill_timestamp: datetime | None
    position_reference_time: datetime | None
    time_delta_seconds: float | None
    source: str
    okx_inst_id: str


@dataclass(frozen=True, slots=True)
class CloseLinkReassignmentPlan:
    position_id: int
    symbol: str
    side: str
    close_side: str
    model_name: str
    execution_mode: str
    old_okx_order_id: str
    new_okx_order_id: str
    old_fill_quantity: float
    new_fill_quantity: float
    old_fill_contracts: float
    new_fill_contracts: float
    contract_size: float
    contract_size_source: str
    target_quantity: float
    entry_price: float
    exit_price: float
    close_fee: float
    fill_pnl: float
    computed_realized_pnl: float
    old_realized_pnl: float
    fill_timestamp: datetime | None
    position_reference_time: datetime | None
    time_delta_seconds: float | None
    source: str
    okx_inst_id: str


@dataclass(frozen=True, slots=True)
class NativeFullCloseSharedPlan:
    position_ids: tuple[int, ...]
    symbol: str
    side: str
    close_side: str
    model_name: str
    execution_mode: str
    close_order_id: int | None
    old_exchange_order_id: str | None
    okx_order_id: str
    total_quantity: float
    fill_quantity: float
    fill_contracts: float
    contract_size: float
    entry_price_weighted: float
    exit_price: float
    close_fee: float
    fill_pnl: float
    fill_timestamp: datetime | None
    source: str
    okx_inst_id: str


@dataclass(frozen=True, slots=True)
class LinkedProtectionFillOrderPlan:
    local_entry_order_id: int
    linked_exchange_order_id: str
    symbol: str
    side: str
    model_name: str
    execution_mode: str
    exchange_order_id: str
    quantity: float
    price: float
    fee: float
    filled_at: datetime | None
    source: str
    okx_inst_id: str
    okx_algo_id: str = ""
    okx_source: str = ""
    decision_id: int | None = None


@dataclass(frozen=True, slots=True)
class OrphanOpenPositionQuarantinePlan:
    position_id: int
    symbol: str
    side: str
    model_name: str
    execution_mode: str
    quantity: float
    entry_price: float
    old_current_price: float | None
    old_unrealized_pnl: float
    old_realized_pnl: float
    old_close_exchange_order_id: str | None
    old_okx_inst_id: str | None
    source: str
    okx_inst_id: str
    reason: str


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _close_side(position: Position) -> str:
    return "buy" if str(position.side or "").lower() == "short" else "sell"


def _entry_side(position: Position) -> str:
    return "sell" if str(position.side or "").lower() == "short" else "buy"


def _position_okx_inst_id(position: Position) -> str:
    return str(getattr(position, "okx_inst_id", "") or "").strip().upper()


def _position_fill_symbol(position: Position) -> str:
    return normalize_trading_symbol(_position_okx_inst_id(position) or position.symbol)


def _split_exchange_order_ids(value: Any) -> list[str]:
    tokens = {str(value or "").strip()}
    if not next(iter(tokens), ""):
        return []
    for separator in (",", ";", "|", "\n", "\t", " "):
        pieces: set[str] = set()
        for token in tokens:
            pieces.update(part.strip() for part in token.split(separator) if part.strip())
        tokens = pieces
    return [token for token in tokens if token]


def _position_entry_action(position: Position) -> str:
    return "short" if str(getattr(position, "side", "") or "").lower() == "short" else "long"


async def _find_unique_entry_decision_for_position(
    position: Position,
    *,
    order_time: datetime | None,
    window_seconds: int,
) -> AIDecision | None:
    reference_time = _aware(order_time or getattr(position, "created_at", None))
    if reference_time is None:
        return None
    window = max(int(window_seconds or DEFAULT_DECISION_WINDOW_SECONDS), 1)
    symbol_variants = symbol_query_variants(
        {
            getattr(position, "symbol", None),
            _position_fill_symbol(position),
            _position_okx_inst_id(position),
        }
    )
    if not symbol_variants:
        return None
    action = _position_entry_action(position)
    start = reference_time - timedelta(seconds=window)
    end = reference_time + timedelta(seconds=window)
    async with get_session_ctx() as session:
        rows = await session.execute(
            select(AIDecision)
            .where(
                AIDecision.is_paper.is_(
                    str(getattr(position, "execution_mode", "") or "paper").lower()
                    != "live"
                ),
                AIDecision.symbol.in_(symbol_variants),
                AIDecision.action == action,
                or_(
                    AIDecision.was_executed.is_(True),
                    AIDecision.executed_at.is_not(None),
                ),
                or_(
                    AIDecision.executed_at.between(
                        start.replace(tzinfo=None),
                        end.replace(tzinfo=None),
                    ),
                    AIDecision.created_at.between(
                        start.replace(tzinfo=None),
                        end.replace(tzinfo=None),
                    ),
                ),
            )
            .order_by(AIDecision.executed_at.asc(), AIDecision.created_at.asc(), AIDecision.id.asc())
            .limit(10)
        )
        candidates = list(rows.scalars().all())
    filtered: list[tuple[float, int, AIDecision]] = []
    for decision in candidates:
        decision_time = _aware(
            getattr(decision, "executed_at", None) or getattr(decision, "created_at", None)
        )
        if decision_time is None:
            continue
        delta = abs((decision_time - reference_time).total_seconds())
        if delta <= window:
            filtered.append((delta, int(decision.id), decision))
    if len(filtered) != 1:
        return None
    return sorted(filtered, key=lambda item: (item[0], item[1]))[0][2]


async def collect_plans(
    *,
    days: int = DEFAULT_DAYS,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    limit: int = DEFAULT_LIMIT,
    position_ids: tuple[int, ...] = (),
    exchange_order_ids: tuple[str, ...] = (),
) -> list[FillLinkPlan]:
    positions = await _candidate_positions(days=days, limit=limit, position_ids=position_ids)
    symbols = {_position_fill_symbol(position) for position in positions}
    fills_by_symbol = await _fetch_okx_fill_groups(symbols)
    contract_sizes = await _fetch_okx_contract_sizes(positions)
    requested_exchange_ids = {
        str(item or "").strip() for item in exchange_order_ids if str(item or "").strip()
    }
    plans: list[FillLinkPlan] = []
    for position in positions:
        symbol = _position_fill_symbol(position)
        fills = fills_by_symbol.get(symbol, [])
        if not str(getattr(position, "entry_exchange_order_id", "") or "").strip():
            plan = _match_fill_plan(
                position,
                fills,
                link_kind="entry",
                expected_side=_entry_side(position),
                reference_time=_aware(position.created_at),
                window_seconds=window_seconds,
                contract_sizes=contract_sizes,
            )
            if plan is not None:
                if not requested_exchange_ids or plan.okx_order_id in requested_exchange_ids:
                    plans.append(plan)
        if (
            not bool(position.is_open)
            and _safe_float(position.realized_pnl) != 0.0
            and not str(getattr(position, "close_exchange_order_id", "") or "").strip()
        ):
            plan = _match_fill_plan(
                position,
                fills,
                link_kind="close",
                expected_side=_close_side(position),
                reference_time=_aware(position.closed_at),
                window_seconds=window_seconds,
                contract_sizes=contract_sizes,
            )
            if plan is not None:
                if not requested_exchange_ids or plan.okx_order_id in requested_exchange_ids:
                    plans.append(plan)
    return plans


async def collect_missing_order_row_plans(
    *,
    days: int = DEFAULT_DAYS,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    decision_window_seconds: int = DEFAULT_DECISION_WINDOW_SECONDS,
    limit: int = DEFAULT_LIMIT,
    position_ids: tuple[int, ...] = (),
    exchange_order_ids: tuple[str, ...] = (),
) -> list[MissingOrderRowPlan]:
    positions = await _candidate_positions(days=days, limit=limit, position_ids=position_ids)
    symbols = {_position_fill_symbol(position) for position in positions}
    fills_by_symbol = await _fetch_okx_fill_groups(symbols)
    existing_exchange_ids = await _existing_local_order_ids()
    requested_exchange_ids = {
        str(item or "").strip() for item in exchange_order_ids if str(item or "").strip()
    }
    contract_sizes = await _fetch_okx_contract_sizes(positions)
    plans: list[MissingOrderRowPlan] = []
    for position in positions:
        symbol = _position_fill_symbol(position)
        fills = fills_by_symbol.get(symbol, [])
        for link_kind, exchange_ids, expected_side, reference_time in (
            (
                "entry",
                _split_exchange_order_ids(getattr(position, "entry_exchange_order_id", None)),
                _entry_side(position),
                _aware(position.created_at),
            ),
            (
                "close",
                _split_exchange_order_ids(getattr(position, "close_exchange_order_id", None)),
                _close_side(position),
                _aware(position.closed_at),
            ),
        ):
            for exchange_id in exchange_ids:
                if requested_exchange_ids and exchange_id not in requested_exchange_ids:
                    continue
                if not exchange_id or exchange_id in existing_exchange_ids:
                    continue
                fill = _matching_fill_by_order_id(
                    fills,
                    exchange_id,
                    expected_side=expected_side,
                    reference_time=reference_time,
                    window_seconds=window_seconds,
                )
                if fill is None:
                    continue
                contract_size, _contract_size_source = _contract_size_for_open_close(
                    position,
                    fill,
                    contract_sizes,
                    abs(_safe_float(position.quantity)),
                )
                fill_quantity = (
                    _fill_quantity_with_contract_size(fill, contract_size)
                    if contract_size > 0
                    else _fill_quantity(fill, abs(_safe_float(position.quantity)))
                )
                matched_decision = (
                    await _find_unique_entry_decision_for_position(
                        position,
                        order_time=fill.timestamp or reference_time,
                        window_seconds=decision_window_seconds,
                    )
                    if link_kind == "entry"
                    else None
                )
                plans.append(
                    MissingOrderRowPlan(
                        position_id=int(position.id),
                        link_kind=link_kind,
                        symbol=normalize_trading_symbol(fill.inst_id),
                        side=expected_side,
                        model_name=str(position.model_name or "ensemble_trader"),
                        execution_mode=str(position.execution_mode or "paper"),
                        exchange_order_id=exchange_id,
                        quantity=fill_quantity,
                        price=fill.avg_price,
                        fee=fill.fee_abs,
                        filled_at=fill.timestamp,
                        source="okx_fills_history_missing_local_order",
                        okx_inst_id=fill.inst_id,
                        decision_id=(
                            int(matched_decision.id) if matched_decision is not None else None
                        ),
                        decision_match_source=(
                            "unique_entry_decision_time_symbol_side"
                            if matched_decision is not None
                            else ""
                        ),
                    )
                )
    return plans


async def collect_existing_order_decision_link_plans(
    *,
    days: int = DEFAULT_DAYS,
    decision_window_seconds: int = DEFAULT_DECISION_WINDOW_SECONDS,
    limit: int = DEFAULT_LIMIT,
    position_ids: tuple[int, ...] = (),
    exchange_order_ids: tuple[str, ...] = (),
) -> list[ExistingOrderDecisionLinkPlan]:
    """Find OKX-confirmed entry orders that can safely inherit one entry decision."""

    positions = await _candidate_closed_positions(
        days=days,
        limit=limit,
        position_ids=position_ids,
    )
    if not positions:
        return []
    requested_exchange_ids = {
        str(item or "").strip() for item in exchange_order_ids if str(item or "").strip()
    }
    entry_order_ids = sorted(
        {
            order_id
            for position in positions
            for order_id in _split_exchange_order_ids(
                getattr(position, "entry_exchange_order_id", None)
            )
        }
    )
    if requested_exchange_ids:
        entry_order_ids = [order_id for order_id in entry_order_ids if order_id in requested_exchange_ids]
    if not entry_order_ids:
        return []
    async with get_session_ctx() as session:
        rows = await session.execute(
            select(Order)
            .where(
                Order.exchange_order_id.in_(entry_order_ids),
                Order.status == "filled",
            )
            .order_by(Order.filled_at.asc(), Order.created_at.asc(), Order.id.asc())
        )
        orders = list(rows.scalars().all())
    order_by_exchange_id: dict[str, Order] = {}
    for order in orders:
        if getattr(order, "decision_id", None) is not None:
            continue
        if str(getattr(order, "status", "") or "").lower() != "filled":
            continue
        for exchange_id in _split_exchange_order_ids(getattr(order, "exchange_order_id", None)):
            order_by_exchange_id.setdefault(exchange_id, order)

    plans: list[ExistingOrderDecisionLinkPlan] = []
    seen_exchange_ids: set[str] = set()
    for position in positions:
        expected_entry_side = _entry_side(position)
        for exchange_id in _split_exchange_order_ids(
            getattr(position, "entry_exchange_order_id", None)
        ):
            if exchange_id in seen_exchange_ids:
                continue
            order = order_by_exchange_id.get(exchange_id)
            if order is None:
                continue
            if str(getattr(order, "side", "") or "").lower() != expected_entry_side:
                continue
            order_time = _aware(getattr(order, "filled_at", None) or getattr(order, "created_at", None))
            decision = await _find_unique_entry_decision_for_position(
                position,
                order_time=order_time,
                window_seconds=decision_window_seconds,
            )
            if decision is None:
                continue
            decision_time = _aware(
                getattr(decision, "executed_at", None) or getattr(decision, "created_at", None)
            )
            position_created = _aware(getattr(position, "created_at", None))
            plans.append(
                ExistingOrderDecisionLinkPlan(
                    position_id=int(position.id),
                    order_id=int(order.id),
                    exchange_order_id=exchange_id,
                    symbol=normalize_trading_symbol(
                        getattr(order, "okx_inst_id", None)
                        or getattr(order, "symbol", None)
                        or _position_fill_symbol(position)
                    ),
                    side=expected_entry_side,
                    model_name=str(getattr(order, "model_name", None) or position.model_name or ""),
                    execution_mode=str(
                        getattr(order, "execution_mode", None) or position.execution_mode or "paper"
                    ),
                    decision_id=int(decision.id),
                    decision_symbol=str(getattr(decision, "symbol", "") or ""),
                    decision_action=str(getattr(decision, "action", "") or ""),
                    order_filled_at=order_time,
                    decision_executed_at=decision_time,
                    position_created_at=position_created,
                    order_decision_delta_seconds=(
                        abs((order_time - decision_time).total_seconds())
                        if order_time is not None and decision_time is not None
                        else None
                    ),
                    position_order_delta_seconds=(
                        abs((order_time - position_created).total_seconds())
                        if order_time is not None and position_created is not None
                        else None
                    ),
                )
            )
            seen_exchange_ids.add(exchange_id)
    return plans


async def collect_open_position_close_plans(
    *,
    days: int = DEFAULT_DAYS,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    limit: int = DEFAULT_LIMIT,
    position_ids: tuple[int, ...] = (),
) -> list[OpenPositionClosePlan]:
    positions = await _candidate_open_positions(
        days=days,
        limit=limit,
        position_ids=position_ids,
    )
    symbols = {_position_fill_symbol(position) for position in positions}
    fills_by_symbol = await _fetch_okx_fill_groups(symbols)
    existing_close_orders = await _candidate_existing_close_orders_for_open_positions(
        positions,
        days=days,
        limit=limit,
    )
    existing_exchange_ids = await _existing_order_ids(positions)
    contract_sizes = await _fetch_okx_contract_sizes(positions)
    used_exchange_ids: set[str] = set()
    plans: list[OpenPositionClosePlan] = []
    for position in positions:
        if not _position_okx_inst_id(position):
            continue
        symbol = _position_fill_symbol(position)
        plan = _match_existing_close_order_open_position_plan(
            position,
            existing_close_orders,
            contract_sizes=contract_sizes,
        )
        if plan is None:
            plan = _match_open_position_close_plan(
                position,
                fills_by_symbol.get(symbol, []),
                existing_exchange_ids=existing_exchange_ids | used_exchange_ids,
                contract_sizes=contract_sizes,
                window_seconds=window_seconds,
            )
        if plan is not None and plan.okx_order_id not in used_exchange_ids:
            plans.append(plan)
            used_exchange_ids.add(plan.okx_order_id)
    return plans


async def collect_orphan_open_position_quarantine_plans(
    *,
    days: int = DEFAULT_DAYS,
    limit: int = DEFAULT_LIMIT,
    position_ids: tuple[int, ...] = (),
) -> list[OrphanOpenPositionQuarantinePlan]:
    """Plan quarantine for local open rows denied by OKX current positions.

    This does not invent a close fill or PnL. It only moves stale local rows out
    of the current-position truth set when OKX current state has no matching
    open position and no close-fill repair plan exists.
    """

    positions = await _candidate_open_positions(
        days=days,
        limit=limit,
        position_ids=position_ids,
    )
    if not positions:
        return []
    close_plans = await collect_open_position_close_plans(
        days=days,
        limit=limit,
        position_ids=position_ids,
    )
    close_plan_position_ids = {plan.position_id for plan in close_plans}
    exchange_keys = await _fetch_current_okx_position_keys(positions)
    plans: list[OrphanOpenPositionQuarantinePlan] = []
    for position in positions:
        position_id = int(getattr(position, "id", 0) or 0)
        if position_id in close_plan_position_ids:
            continue
        inst_id = _position_okx_inst_id(position)
        if not inst_id:
            continue
        key = (inst_id, str(getattr(position, "side", "") or "").lower().strip())
        if key in exchange_keys:
            continue
        plans.append(
            OrphanOpenPositionQuarantinePlan(
                position_id=position_id,
                symbol=normalize_trading_symbol(inst_id),
                side=str(getattr(position, "side", "") or ""),
                model_name=str(getattr(position, "model_name", "") or "ensemble_trader"),
                execution_mode=str(getattr(position, "execution_mode", "") or "paper"),
                quantity=abs(_safe_float(getattr(position, "quantity", None))),
                entry_price=_safe_float(getattr(position, "entry_price", None)),
                old_current_price=(
                    _safe_float(getattr(position, "current_price", None))
                    if getattr(position, "current_price", None) is not None
                    else None
                ),
                old_unrealized_pnl=_safe_float(getattr(position, "unrealized_pnl", None)),
                old_realized_pnl=_safe_float(getattr(position, "realized_pnl", None)),
                old_close_exchange_order_id=getattr(position, "close_exchange_order_id", None),
                old_okx_inst_id=getattr(position, "okx_inst_id", None),
                source="okx_current_position_absent_no_close_fill",
                okx_inst_id=inst_id,
                reason=(
                    "OKX current positions do not contain this local open row, "
                    "and no OKX close-fill repair plan was found."
                ),
            )
        )
    return plans


async def _fetch_current_okx_position_keys(positions: list[Position]) -> set[tuple[str, str]]:
    inst_ids = {
        _position_okx_inst_id(position)
        for position in positions
        if bool(getattr(position, "is_open", False)) and _position_okx_inst_id(position)
    }
    if not inst_ids:
        return set()
    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    try:
        await executor.initialize()
        rows = await OkxNativeFactsClient(executor).fetch_positions(inst_ids=inst_ids)
    finally:
        await executor.shutdown()
    keys: set[tuple[str, str]] = set()
    for row in rows:
        info = row.get("info") if isinstance(row, dict) else {}
        inst_id = str((info or {}).get("instId") or row.get("symbol") or "").strip().upper()
        side = str(row.get("side") or "").lower().strip()
        if inst_id and side:
            keys.add((inst_id, side))
    return keys


def _match_existing_close_order_open_position_plan(
    position: Position,
    orders: list[Order],
    *,
    contract_sizes: dict[str, float],
) -> OpenPositionClosePlan | None:
    if not bool(getattr(position, "is_open", False)):
        return None
    if str(getattr(position, "close_exchange_order_id", "") or "").strip():
        return None
    position_inst_id = _position_okx_inst_id(position)
    if not position_inst_id:
        return None
    target_quantity = abs(_safe_float(getattr(position, "quantity", None)))
    if target_quantity <= 0:
        return None
    opened_at = _aware(getattr(position, "created_at", None))
    if opened_at is None:
        return None
    expected_side = _close_side(position)
    entry_exchange_id = str(getattr(position, "entry_exchange_order_id", "") or "").strip()
    candidates: list[tuple[float, float, Order, float, float, str]] = []
    for order in orders:
        exchange_order_id = str(getattr(order, "exchange_order_id", "") or "").strip()
        if not exchange_order_id or exchange_order_id == entry_exchange_id:
            continue
        if str(getattr(order, "status", "") or "").lower() != "filled":
            continue
        sync_status = str(getattr(order, "okx_sync_status", "") or "").strip()
        if sync_status not in TRUSTED_CLOSE_ORDER_SYNC_STATUSES:
            continue
        if str(getattr(order, "side", "") or "").lower() != expected_side:
            continue
        order_inst_id = str(getattr(order, "okx_inst_id", "") or "").strip().upper()
        if order_inst_id != position_inst_id:
            continue
        order_time = _aware(
            getattr(order, "filled_at", None) or getattr(order, "created_at", None)
        )
        if order_time is None:
            continue
        delta = (order_time - opened_at).total_seconds()
        if delta < 0:
            continue
        fill_quantity, contract_size, contract_size_source = _order_fill_quantity(
            order,
            contract_sizes,
        )
        if not _quantity_close_enough(fill_quantity, target_quantity):
            continue
        candidates.append(
            (
                delta,
                abs(fill_quantity - target_quantity),
                order,
                fill_quantity,
                contract_size,
                contract_size_source,
            )
        )
    if not candidates:
        return None

    delta, _quantity_delta, order, fill_quantity, contract_size, contract_size_source = sorted(
        candidates,
        key=lambda item: (item[0], item[1]),
    )[0]
    entry_price = _safe_float(position.entry_price)
    exit_price = _safe_float(order.price)
    close_fee = _safe_float(order.fee)
    if str(position.side or "").lower() == "short":
        gross_pnl = (entry_price - exit_price) * fill_quantity
    else:
        gross_pnl = (exit_price - entry_price) * fill_quantity
    computed_realized_pnl = gross_pnl - close_fee
    return OpenPositionClosePlan(
        position_id=int(position.id),
        symbol=normalize_trading_symbol(position_inst_id),
        side=str(position.side or ""),
        close_side=expected_side,
        model_name=str(position.model_name or "ensemble_trader"),
        execution_mode=str(position.execution_mode or "paper"),
        okx_order_id=str(order.exchange_order_id or ""),
        old_is_open=bool(position.is_open),
        old_close_exchange_order_id=getattr(position, "close_exchange_order_id", None),
        old_okx_inst_id=getattr(position, "okx_inst_id", None),
        quantity=target_quantity,
        fill_quantity=fill_quantity,
        fill_contracts=_safe_float(getattr(order, "okx_fill_contracts", None)),
        contract_size=contract_size,
        contract_size_source=contract_size_source,
        entry_price=entry_price,
        exit_price=exit_price,
        close_fee=close_fee,
        fill_pnl=_safe_float(getattr(order, "okx_fill_pnl", None)),
        computed_realized_pnl=computed_realized_pnl,
        old_realized_pnl=_safe_float(position.realized_pnl),
        old_current_price=(
            _safe_float(position.current_price) if position.current_price is not None else None
        ),
        fill_timestamp=_aware(
            getattr(order, "filled_at", None) or getattr(order, "created_at", None)
        ),
        position_reference_time=opened_at,
        time_delta_seconds=round(delta, 6),
        source="okx_confirmed_existing_close_order",
        okx_inst_id=position_inst_id,
    )


def _order_fill_quantity(
    order: Order,
    contract_sizes: dict[str, float],
) -> tuple[float, float, str]:
    inst_id = str(getattr(order, "okx_inst_id", "") or "").strip().upper()
    contract_size = _safe_float(contract_sizes.get(inst_id), 0.0)
    contracts = _safe_float(getattr(order, "okx_fill_contracts", None), 0.0)
    if contract_size > 0 and contracts > 0:
        return contracts * contract_size, contract_size, "okx_order_fill_contracts_ctVal"
    return _safe_float(getattr(order, "quantity", None)), 0.0, "local_order_base_quantity"


async def collect_exchange_fill_open_position_close_plans(
    *,
    days: int = DEFAULT_DAYS,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    limit: int = DEFAULT_LIMIT,
    position_ids: tuple[int, ...] = (),
) -> list[OpenPositionClosePlan]:
    positions = await _candidate_open_positions(
        days=days,
        limit=limit,
        position_ids=position_ids,
    )
    symbols = {_position_fill_symbol(position) for position in positions}
    fills_by_symbol = await _fetch_okx_fill_groups(symbols)
    existing_exchange_ids = await _existing_order_ids(positions)
    contract_sizes = await _fetch_okx_contract_sizes(positions)
    used_exchange_ids: set[str] = set()
    plans: list[OpenPositionClosePlan] = []
    for position in positions:
        if not _position_okx_inst_id(position):
            continue
        symbol = _position_fill_symbol(position)
        plan = _match_open_position_close_plan(
            position,
            fills_by_symbol.get(symbol, []),
            existing_exchange_ids=existing_exchange_ids | used_exchange_ids,
            contract_sizes=contract_sizes,
            window_seconds=window_seconds,
        )
        if plan is not None and plan.okx_order_id not in used_exchange_ids:
            plans.append(plan)
            used_exchange_ids.add(plan.okx_order_id)
    return plans


async def collect_close_link_reassignment_plans(
    *,
    days: int = DEFAULT_DAYS,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    limit: int = DEFAULT_LIMIT,
    position_ids: tuple[int, ...] = (),
) -> list[CloseLinkReassignmentPlan]:
    positions = await _candidate_closed_positions(
        days=days,
        limit=limit,
        position_ids=position_ids,
    )
    symbols = {_position_fill_symbol(position) for position in positions}
    fills_by_symbol = await _fetch_okx_fill_groups(symbols)
    existing_exchange_ids = await _existing_order_ids(positions)
    contract_sizes = await _fetch_okx_contract_sizes(positions)
    plans: list[CloseLinkReassignmentPlan] = []
    for position in positions:
        symbol = _position_fill_symbol(position)
        plan = _match_close_link_reassignment_plan(
            position,
            fills_by_symbol.get(symbol, []),
            existing_exchange_ids=existing_exchange_ids,
            contract_sizes=contract_sizes,
            window_seconds=window_seconds,
        )
        if plan is not None:
            plans.append(plan)
    return plans


async def collect_native_full_close_shared_plans(
    *,
    days: int = DEFAULT_DAYS,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    limit: int = DEFAULT_LIMIT,
    position_ids: tuple[int, ...] = (),
) -> list[NativeFullCloseSharedPlan]:
    positions = await _candidate_native_full_close_positions(
        days=days,
        limit=limit,
        position_ids=position_ids,
    )
    if not positions:
        return []
    orders = await _candidate_native_full_close_orders(days=days, symbols={p.symbol for p in positions})
    symbols = {_position_fill_symbol(position) for position in positions}
    fills_by_symbol = await _fetch_okx_fill_groups(symbols)
    contract_sizes = await _fetch_okx_contract_sizes(positions)
    plans: list[NativeFullCloseSharedPlan] = []
    for group in _group_native_full_close_positions(positions):
        symbol = _position_fill_symbol(group[0])
        plan = _match_native_full_close_shared_plan(
            group,
            orders,
            fills_by_symbol.get(symbol, []),
            contract_sizes=contract_sizes,
            window_seconds=window_seconds,
        )
        if plan is not None:
            plans.append(plan)
    return plans


async def collect_linked_protection_fill_order_plans(
    *,
    days: int = DEFAULT_DAYS,
    limit: int = DEFAULT_LIMIT,
    exchange_order_ids: tuple[str, ...] = (),
) -> list[LinkedProtectionFillOrderPlan]:
    local_orders, local_decisions = await _candidate_linked_protection_orders(
        days=days,
        limit=limit,
    )
    if not local_orders:
        return []
    symbols = {
        normalize_trading_symbol(order.symbol)
        for order in local_orders
        if normalize_trading_symbol(order.symbol)
    }
    fills_by_symbol = await _fetch_okx_fill_groups(symbols)
    fills = [fill for group in fills_by_symbol.values() for fill in group]
    if not fills:
        return []
    existing_exchange_ids = await _existing_local_order_ids()
    requested_exchange_ids = {str(item or "").strip() for item in exchange_order_ids if item}
    if requested_exchange_ids:
        fills = [fill for fill in fills if str(fill.order_id) in requested_exchange_ids]
    fills = [fill for fill in fills if str(fill.order_id) not in existing_exchange_ids]
    if not fills:
        return []
    order_contexts = await _fetch_okx_order_history_contexts(fills)
    contract_sizes = await _fetch_okx_contract_sizes_for_inst_ids(
        {str(fill.inst_id or "").strip().upper() for fill in fills if fill.inst_id}
    )
    local_orders_by_exchange_id = _local_orders_by_exchange_id(local_orders)
    plans: list[LinkedProtectionFillOrderPlan] = []
    seen: set[str] = set()
    for fill in fills:
        if str(fill.order_id) in seen:
            continue
        linked = _linked_protection_fill_context(
            fill,
            order_contexts=order_contexts,
            local_orders_by_exchange_id=local_orders_by_exchange_id,
            local_orders=local_orders,
            local_decisions=local_decisions,
        )
        if linked is None:
            continue
        source_order = local_orders_by_exchange_id.get(
            str(linked.get("linked_exchange_order_id") or "")
        )
        if source_order is None:
            continue
        contract_size = _safe_float(contract_sizes.get(str(fill.inst_id or "").strip().upper()))
        if contract_size <= 0:
            contract_size = _safe_float(source_order.quantity) / max(_safe_float(fill.contracts), 1e-12)
        quantity = (
            _fill_quantity_with_contract_size(fill, contract_size)
            if contract_size > 0
            else _safe_float(fill.contracts)
        )
        plans.append(
            LinkedProtectionFillOrderPlan(
                local_entry_order_id=int(source_order.id),
                linked_exchange_order_id=str(linked.get("linked_exchange_order_id") or ""),
                symbol=normalize_trading_symbol(fill.inst_id),
                side=str(fill.side or "").lower(),
                model_name=str(source_order.model_name or "ensemble_trader"),
                execution_mode=str(source_order.execution_mode or "paper"),
                exchange_order_id=str(fill.order_id),
                quantity=quantity,
                price=_safe_float(fill.avg_price),
                fee=_safe_float(fill.fee_abs),
                filled_at=fill.timestamp,
                source="okx_linked_protection_fill_missing_local_order",
                okx_inst_id=str(fill.inst_id or "").strip().upper(),
                okx_algo_id=str(linked.get("okx_algo_id") or ""),
                okx_source=str(linked.get("okx_source") or ""),
                decision_id=(
                    int(source_order.decision_id)
                    if getattr(source_order, "decision_id", None) is not None
                    else None
                ),
            )
        )
        seen.add(str(fill.order_id))
    return plans


async def _candidate_positions(
    *,
    days: int,
    limit: int,
    position_ids: tuple[int, ...],
) -> list[Position]:
    since = datetime.now(UTC) - timedelta(days=max(int(days or DEFAULT_DAYS), 1))
    capped_limit = max(1, min(int(limit or DEFAULT_LIMIT), 1000))
    async with get_session_ctx() as session:
        conditions = [
            or_(
                Position.created_at >= since.replace(tzinfo=None),
                Position.closed_at >= since.replace(tzinfo=None),
            )
        ]
        if not position_ids:
            conditions.append(
                or_(
                    Position.entry_exchange_order_id.is_(None),
                    Position.entry_exchange_order_id == "",
                    Position.close_exchange_order_id.is_(None),
                    Position.close_exchange_order_id == "",
                )
            )
        stmt = select(Position).where(
            *conditions,
        )
        if position_ids:
            stmt = stmt.where(Position.id.in_(position_ids))
        rows = await session.execute(
            stmt.order_by(Position.created_at.desc(), Position.id.desc()).limit(capped_limit)
        )
        return list(rows.scalars().all())


async def _candidate_linked_protection_orders(
    *,
    days: int,
    limit: int,
) -> tuple[list[Order], dict[int, AIDecision]]:
    since = datetime.now(UTC) - timedelta(days=max(int(days or DEFAULT_DAYS), 1))
    capped_limit = max(1, min(int(limit or DEFAULT_LIMIT), 1000))
    async with get_session_ctx() as session:
        rows = await session.execute(
            select(Order)
            .where(
                Order.execution_mode == "paper",
                Order.status == "filled",
                Order.exchange_order_id.is_not(None),
                Order.exchange_order_id != "",
                or_(Order.created_at >= since.replace(tzinfo=None), Order.filled_at >= since.replace(tzinfo=None)),
            )
            .order_by(Order.filled_at.desc(), Order.created_at.desc())
            .limit(capped_limit)
        )
        orders = list(rows.scalars().all())
        decision_ids = {
            int(order.decision_id)
            for order in orders
            if getattr(order, "decision_id", None) is not None
        }
        decisions: dict[int, AIDecision] = {}
        if decision_ids:
            decision_rows = await session.execute(
                select(AIDecision).where(AIDecision.id.in_(decision_ids))
            )
            decisions = {
                int(decision.id): decision for decision in decision_rows.scalars().all()
            }
    return orders, decisions


async def _candidate_native_full_close_positions(
    *,
    days: int,
    limit: int,
    position_ids: tuple[int, ...],
) -> list[Position]:
    positions = await _candidate_closed_positions(
        days=days,
        limit=limit,
        position_ids=position_ids,
    )
    return [
        position
        for position in positions
        if _is_native_full_close_placeholder(getattr(position, "close_exchange_order_id", None))
    ]


async def _candidate_native_full_close_orders(
    *,
    days: int,
    symbols: set[str],
) -> list[Order]:
    since = datetime.now(UTC) - timedelta(days=max(int(days or DEFAULT_DAYS), 1))
    if not symbols:
        return []
    async with get_session_ctx() as session:
        rows = await session.execute(
            select(Order).where(
                Order.symbol.in_({normalize_trading_symbol(symbol) for symbol in symbols}),
                Order.filled_at >= since.replace(tzinfo=None),
                Order.status == "filled",
            )
        )
        orders = list(rows.scalars().all())
    return [
        order
        for order in orders
        if _is_native_full_close_placeholder(getattr(order, "exchange_order_id", None))
    ]


async def _candidate_open_positions(
    *,
    days: int,
    limit: int,
    position_ids: tuple[int, ...],
) -> list[Position]:
    since = datetime.now(UTC) - timedelta(days=max(int(days or DEFAULT_DAYS), 1))
    capped_limit = max(1, min(int(limit or DEFAULT_LIMIT), 1000))
    async with get_session_ctx() as session:
        stmt = select(Position).where(
            Position.is_open.is_(True),
            Position.created_at >= since.replace(tzinfo=None),
        )
        if position_ids:
            stmt = stmt.where(Position.id.in_(position_ids))
        rows = await session.execute(
            stmt.order_by(Position.created_at.desc(), Position.id.desc()).limit(capped_limit)
        )
        return list(rows.scalars().all())


async def _candidate_existing_close_orders_for_open_positions(
    positions: list[Position],
    *,
    days: int,
    limit: int,
) -> list[Order]:
    inst_ids = {
        _position_okx_inst_id(position)
        for position in positions
        if bool(getattr(position, "is_open", False)) and _position_okx_inst_id(position)
    }
    if not inst_ids:
        return []
    close_sides = {
        _close_side(position)
        for position in positions
        if bool(getattr(position, "is_open", False)) and _position_okx_inst_id(position)
    }
    since = datetime.now(UTC) - timedelta(days=max(int(days or DEFAULT_DAYS), 1))
    capped_limit = max(100, min(max(int(limit or DEFAULT_LIMIT), 1) * 10, 5000))
    async with get_session_ctx() as session:
        rows = await session.execute(
            select(Order)
            .where(
                Order.status == "filled",
                Order.exchange_order_id.is_not(None),
                Order.exchange_order_id != "",
                Order.okx_inst_id.in_(inst_ids),
                Order.side.in_(close_sides),
                Order.okx_sync_status.in_(TRUSTED_CLOSE_ORDER_SYNC_STATUSES),
                or_(
                    Order.filled_at >= since.replace(tzinfo=None),
                    Order.created_at >= since.replace(tzinfo=None),
                ),
            )
            .order_by(Order.filled_at.asc(), Order.created_at.asc(), Order.id.asc())
            .limit(capped_limit)
        )
        return list(rows.scalars().all())


async def _candidate_closed_positions(
    *,
    days: int,
    limit: int,
    position_ids: tuple[int, ...],
) -> list[Position]:
    since = datetime.now(UTC) - timedelta(days=max(int(days or DEFAULT_DAYS), 1))
    capped_limit = max(1, min(int(limit or DEFAULT_LIMIT), 1000))
    async with get_session_ctx() as session:
        stmt = select(Position).where(
            Position.is_open.is_(False),
            Position.closed_at >= since.replace(tzinfo=None),
            Position.close_exchange_order_id.is_not(None),
            Position.close_exchange_order_id != "",
        )
        if position_ids:
            stmt = stmt.where(Position.id.in_(position_ids))
        rows = await session.execute(
            stmt.order_by(Position.closed_at.desc(), Position.id.desc()).limit(capped_limit)
        )
        return list(rows.scalars().all())


async def _existing_order_ids(positions: list[Position]) -> set[str]:
    linked_ids = {
        item.strip()
        for position in positions
        for value in (
            getattr(position, "entry_exchange_order_id", None),
            getattr(position, "close_exchange_order_id", None),
        )
        for item in str(value or "").split(",")
        if item.strip()
    }
    return linked_ids | await _existing_local_order_ids()


async def _existing_local_order_ids() -> set[str]:
    async with get_session_ctx() as session:
        rows = await session.execute(
            select(Order.exchange_order_id).where(
                Order.exchange_order_id.is_not(None),
                Order.exchange_order_id != "",
            )
        )
        order_ids = {
            str(item or "").strip() for item in rows.scalars().all() if str(item or "").strip()
        }
    return order_ids


async def _fetch_okx_contract_sizes(positions: list[Position]) -> dict[str, float]:
    inst_ids = {
        _position_okx_inst_id(position)
        for position in positions
        if _position_okx_inst_id(position)
    }
    if not inst_ids:
        return {}
    executor = OKXExecutor("paper")
    try:
        return await OkxNativeFactsClient(executor).fetch_contract_sizes(inst_ids=inst_ids)
    finally:
        shutdown = getattr(executor, "shutdown", None)
        if callable(shutdown):
            result = shutdown()
            if hasattr(result, "__await__"):
                await result


async def _fetch_okx_contract_sizes_for_inst_ids(inst_ids: set[str]) -> dict[str, float]:
    cleaned = {str(item or "").strip().upper() for item in inst_ids if str(item or "").strip()}
    if not cleaned:
        return {}
    executor = OKXExecutor("paper")
    try:
        return await OkxNativeFactsClient(executor).fetch_contract_sizes(inst_ids=cleaned)
    finally:
        shutdown = getattr(executor, "shutdown", None)
        if callable(shutdown):
            result = shutdown()
            if hasattr(result, "__await__"):
                await result


async def _fetch_okx_order_history_contexts(
    fills: list[FillGroup],
) -> dict[str, tuple[dict[str, Any], ...]]:
    if not fills:
        return {}
    executor = OKXExecutor("paper")
    try:
        return await OkxNativeFactsClient(executor).fetch_order_history_contexts(
            fills=fills,
            limit=5,
            strict=False,
        )
    finally:
        shutdown = getattr(executor, "shutdown", None)
        if callable(shutdown):
            result = shutdown()
            if hasattr(result, "__await__"):
                await result


def _match_fill_plan(
    position: Position,
    fills: list[FillGroup],
    *,
    link_kind: str,
    expected_side: str,
    reference_time: datetime | None,
    window_seconds: int,
    contract_sizes: dict[str, float] | None = None,
) -> FillLinkPlan | None:
    if reference_time is None:
        return None
    target_quantity = abs(_safe_float(position.quantity))
    position_inst_id = _position_okx_inst_id(position)
    contract_sizes = contract_sizes or {}
    candidates: list[tuple[float, float, float, str, FillGroup]] = []
    for fill in fills:
        fill_inst_id = str(getattr(fill, "inst_id", "") or "").strip().upper()
        if not fill_inst_id:
            continue
        if position_inst_id and position_inst_id != fill_inst_id:
            continue
        if fill.side != expected_side or fill.timestamp is None:
            continue
        delta = abs((fill.timestamp - reference_time).total_seconds())
        if delta > max(int(window_seconds), 1):
            continue
        contract_size, contract_size_source = _contract_size_for_open_close(
            position,
            fill,
            contract_sizes,
            target_quantity,
        )
        quantity = (
            _fill_quantity_with_contract_size(fill, contract_size)
            if contract_size > 0
            else _fill_quantity(fill, target_quantity)
        )
        quantity_delta = abs(quantity - target_quantity) if target_quantity > 0 else 0.0
        candidates.append((delta, quantity_delta, contract_size, contract_size_source, fill))
    if not candidates:
        return None
    delta, _quantity_delta, contract_size, contract_size_source, fill = sorted(
        candidates,
        key=lambda item: (item[0], item[1]),
    )[0]
    symbol = normalize_trading_symbol(fill.inst_id)
    fill_quantity = (
        _fill_quantity_with_contract_size(fill, contract_size)
        if contract_size > 0
        else _fill_quantity(fill, target_quantity)
    )
    return FillLinkPlan(
        position_id=int(position.id),
        link_kind=link_kind,
        symbol=symbol,
        side=str(position.side or ""),
        quantity=target_quantity,
        okx_order_id=fill.order_id,
        old_entry_exchange_order_id=getattr(position, "entry_exchange_order_id", None),
        old_close_exchange_order_id=getattr(position, "close_exchange_order_id", None),
        old_okx_inst_id=getattr(position, "okx_inst_id", None),
        fill_timestamp=fill.timestamp,
        position_reference_time=reference_time,
        time_delta_seconds=round(delta, 6),
        fill_quantity=fill_quantity,
        fill_contracts=fill.contracts,
        fill_price=fill.avg_price,
        source="okx_fills_history",
        okx_inst_id=fill.inst_id,
        fill_fee=fill.fee_abs,
        fill_pnl=fill.fill_pnl,
        contract_size=contract_size,
        contract_size_source=contract_size_source,
    )


def _match_open_position_close_plan(
    position: Position,
    fills: list[FillGroup],
    *,
    existing_exchange_ids: set[str],
    contract_sizes: dict[str, float],
    window_seconds: int,
) -> OpenPositionClosePlan | None:
    reference_time = _aware(getattr(position, "created_at", None))
    if reference_time is None:
        return None
    if not bool(getattr(position, "is_open", False)):
        return None
    if str(getattr(position, "close_exchange_order_id", "") or "").strip():
        return None

    position_inst_id = _position_okx_inst_id(position)
    if not position_inst_id:
        return None
    target_quantity = abs(_safe_float(getattr(position, "quantity", None)))
    if target_quantity <= 0:
        return None

    expected_side = _close_side(position)
    candidates: list[tuple[float, float, float, str, FillGroup]] = []
    for fill in fills:
        fill_inst_id = str(getattr(fill, "inst_id", "") or "").strip().upper()
        if fill_inst_id != position_inst_id:
            continue
        if str(getattr(fill, "side", "") or "").lower() != expected_side:
            continue
        order_id = str(getattr(fill, "order_id", "") or "").strip()
        if not order_id or order_id in existing_exchange_ids:
            continue
        if fill.timestamp is None:
            continue
        delta = (fill.timestamp - reference_time).total_seconds()
        if delta < 0 or delta > max(int(window_seconds), 1):
            continue
        contract_size, contract_size_source = _contract_size_for_open_close(
            position,
            fill,
            contract_sizes,
            target_quantity,
        )
        if contract_size <= 0:
            continue
        fill_quantity = _fill_quantity_with_contract_size(fill, contract_size)
        if not _quantity_close_enough(fill_quantity, target_quantity):
            continue
        candidates.append(
            (
                delta,
                abs(fill_quantity - target_quantity),
                contract_size,
                contract_size_source,
                fill,
            )
        )

    if not candidates:
        return None

    delta, _quantity_delta, contract_size, contract_size_source, fill = sorted(
        candidates,
        key=lambda item: (item[0], item[1]),
    )[0]
    fill_quantity = _fill_quantity_with_contract_size(fill, contract_size)
    entry_price = _safe_float(position.entry_price)
    exit_price = _safe_float(fill.avg_price)
    close_fee = _safe_float(fill.fee_abs)
    if str(position.side or "").lower() == "short":
        gross_pnl = (entry_price - exit_price) * fill_quantity
    else:
        gross_pnl = (exit_price - entry_price) * fill_quantity
    computed_realized_pnl = gross_pnl - close_fee

    return OpenPositionClosePlan(
        position_id=int(position.id),
        symbol=normalize_trading_symbol(fill.inst_id),
        side=str(position.side or ""),
        close_side=expected_side,
        model_name=str(position.model_name or "ensemble_trader"),
        execution_mode=str(position.execution_mode or "paper"),
        okx_order_id=str(fill.order_id),
        old_is_open=bool(position.is_open),
        old_close_exchange_order_id=getattr(position, "close_exchange_order_id", None),
        old_okx_inst_id=getattr(position, "okx_inst_id", None),
        quantity=target_quantity,
        fill_quantity=fill_quantity,
        fill_contracts=_safe_float(fill.contracts),
        contract_size=contract_size,
        contract_size_source=contract_size_source,
        entry_price=entry_price,
        exit_price=exit_price,
        close_fee=close_fee,
        fill_pnl=_safe_float(fill.fill_pnl),
        computed_realized_pnl=computed_realized_pnl,
        old_realized_pnl=_safe_float(position.realized_pnl),
        old_current_price=(
            _safe_float(position.current_price) if position.current_price is not None else None
        ),
        fill_timestamp=fill.timestamp,
        position_reference_time=reference_time,
        time_delta_seconds=round(delta, 6),
        source="okx_fills_history_open_position_close",
        okx_inst_id=fill.inst_id,
    )


def _match_close_link_reassignment_plan(
    position: Position,
    fills: list[FillGroup],
    *,
    existing_exchange_ids: set[str],
    contract_sizes: dict[str, float],
    window_seconds: int,
) -> CloseLinkReassignmentPlan | None:
    close_order_id = str(getattr(position, "close_exchange_order_id", "") or "").strip()
    if not close_order_id:
        return None
    reference_time = _aware(getattr(position, "closed_at", None))
    if reference_time is None:
        return None
    target_quantity = abs(_safe_float(getattr(position, "quantity", None)))
    if target_quantity <= 0:
        return None
    expected_side = _close_side(position)

    current_fill = _matching_fill_by_order_id(
        fills,
        close_order_id,
        expected_side=expected_side,
        reference_time=reference_time,
        window_seconds=window_seconds,
    )
    if current_fill is None:
        return None
    contract_size, contract_size_source = _contract_size_for_open_close(
        position,
        current_fill,
        contract_sizes,
        target_quantity,
    )
    if contract_size <= 0:
        return None
    current_quantity = _fill_quantity_with_contract_size(current_fill, contract_size)
    if _quantity_close_enough(current_quantity, target_quantity):
        return None

    blocked_ids = set(existing_exchange_ids)
    blocked_ids.discard(close_order_id)
    candidates: list[tuple[float, float, FillGroup]] = []
    for fill in fills:
        order_id = str(getattr(fill, "order_id", "") or "").strip()
        if not order_id or order_id == close_order_id or order_id in blocked_ids:
            continue
        if str(getattr(fill, "side", "") or "").lower() != expected_side:
            continue
        if fill.timestamp is None:
            continue
        fill_inst_id = str(getattr(fill, "inst_id", "") or "").strip().upper()
        position_inst_id = _position_okx_inst_id(position)
        if position_inst_id and fill_inst_id != position_inst_id:
            continue
        delta = abs((fill.timestamp - reference_time).total_seconds())
        if delta > max(int(window_seconds), 1):
            continue
        candidate_contract_size, _candidate_source = _contract_size_for_open_close(
            position,
            fill,
            contract_sizes,
            target_quantity,
        )
        if candidate_contract_size <= 0:
            continue
        fill_quantity = _fill_quantity_with_contract_size(fill, candidate_contract_size)
        if not _quantity_close_enough(fill_quantity, target_quantity):
            continue
        candidates.append((delta, abs(fill_quantity - target_quantity), fill))

    if not candidates:
        return None
    delta, _quantity_delta, fill = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
    new_contract_size, new_contract_size_source = _contract_size_for_open_close(
        position,
        fill,
        contract_sizes,
        target_quantity,
    )
    new_quantity = _fill_quantity_with_contract_size(fill, new_contract_size)
    entry_price = _safe_float(position.entry_price)
    exit_price = _safe_float(fill.avg_price)
    close_fee = _safe_float(fill.fee_abs)
    if str(position.side or "").lower() == "short":
        gross_pnl = (entry_price - exit_price) * new_quantity
    else:
        gross_pnl = (exit_price - entry_price) * new_quantity
    computed_realized_pnl = gross_pnl - close_fee
    return CloseLinkReassignmentPlan(
        position_id=int(position.id),
        symbol=normalize_trading_symbol(fill.inst_id),
        side=str(position.side or ""),
        close_side=expected_side,
        model_name=str(position.model_name or "ensemble_trader"),
        execution_mode=str(position.execution_mode or "paper"),
        old_okx_order_id=close_order_id,
        new_okx_order_id=str(fill.order_id),
        old_fill_quantity=current_quantity,
        new_fill_quantity=new_quantity,
        old_fill_contracts=_safe_float(current_fill.contracts),
        new_fill_contracts=_safe_float(fill.contracts),
        contract_size=new_contract_size,
        contract_size_source=new_contract_size_source,
        target_quantity=target_quantity,
        entry_price=entry_price,
        exit_price=exit_price,
        close_fee=close_fee,
        fill_pnl=_safe_float(fill.fill_pnl),
        computed_realized_pnl=computed_realized_pnl,
        old_realized_pnl=_safe_float(position.realized_pnl),
        fill_timestamp=fill.timestamp,
        position_reference_time=reference_time,
        time_delta_seconds=round(delta, 6),
        source="okx_fills_history_close_link_reassignment",
        okx_inst_id=fill.inst_id,
    )


def _match_native_full_close_shared_plan(
    positions: list[Position],
    orders: list[Order],
    fills: list[FillGroup],
    *,
    contract_sizes: dict[str, float],
    window_seconds: int,
) -> NativeFullCloseSharedPlan | None:
    if not positions:
        return None
    first = positions[0]
    reference_time = _aware(getattr(first, "closed_at", None))
    if reference_time is None:
        return None
    total_quantity = sum(abs(_safe_float(position.quantity)) for position in positions)
    if total_quantity <= 0:
        return None
    close_side = _close_side(first)
    inst_id = _position_okx_inst_id(first)
    candidates: list[tuple[float, float, float, FillGroup]] = []
    for fill in fills:
        fill_inst_id = str(getattr(fill, "inst_id", "") or "").strip().upper()
        if inst_id and fill_inst_id != inst_id:
            continue
        if str(getattr(fill, "side", "") or "").lower() != close_side:
            continue
        if fill.timestamp is None:
            continue
        delta = abs((fill.timestamp - reference_time).total_seconds())
        if delta > max(int(window_seconds), 1):
            continue
        contract_size, _source = _contract_size_for_open_close(
            first,
            fill,
            contract_sizes,
            total_quantity,
        )
        if contract_size <= 0:
            continue
        fill_quantity = _fill_quantity_with_contract_size(fill, contract_size)
        if not _quantity_close_enough(fill_quantity, total_quantity):
            continue
        candidates.append((delta, abs(fill_quantity - total_quantity), contract_size, fill))
    if not candidates:
        return None

    _delta, _quantity_delta, contract_size, fill = sorted(
        candidates,
        key=lambda item: (item[0], item[1]),
    )[0]
    close_order = _matching_native_full_close_order(
        positions,
        orders,
        close_side=close_side,
        total_quantity=total_quantity,
        reference_time=reference_time,
        window_seconds=window_seconds,
    )
    weighted_entry = (
        sum(_safe_float(position.entry_price) * abs(_safe_float(position.quantity)) for position in positions)
        / total_quantity
    )
    return NativeFullCloseSharedPlan(
        position_ids=tuple(int(position.id) for position in positions),
        symbol=normalize_trading_symbol(fill.inst_id),
        side=str(first.side or ""),
        close_side=close_side,
        model_name=str(first.model_name or "ensemble_trader"),
        execution_mode=str(first.execution_mode or "paper"),
        close_order_id=int(close_order.id) if close_order is not None else None,
        old_exchange_order_id=(
            str(close_order.exchange_order_id).strip()
            if close_order is not None and close_order.exchange_order_id is not None
            else None
        ),
        okx_order_id=str(fill.order_id),
        total_quantity=total_quantity,
        fill_quantity=_fill_quantity_with_contract_size(fill, contract_size),
        fill_contracts=_safe_float(fill.contracts),
        contract_size=contract_size,
        entry_price_weighted=weighted_entry,
        exit_price=_safe_float(fill.avg_price),
        close_fee=_safe_float(fill.fee_abs),
        fill_pnl=_safe_float(fill.fill_pnl),
        fill_timestamp=fill.timestamp,
        source="okx_fills_history_native_full_close_shared",
        okx_inst_id=fill.inst_id,
    )


def _matching_fill_by_order_id(
    fills: list[FillGroup],
    exchange_order_id: str,
    *,
    expected_side: str,
    reference_time: datetime | None,
    window_seconds: int,
) -> FillGroup | None:
    candidates: list[tuple[float, FillGroup]] = []
    for fill in fills:
        if not str(getattr(fill, "inst_id", "") or "").strip():
            continue
        if fill.order_id != exchange_order_id or fill.side != expected_side:
            continue
        if reference_time is not None and fill.timestamp is not None:
            delta = abs((fill.timestamp - reference_time).total_seconds())
            if delta > max(int(window_seconds), 1):
                continue
        else:
            delta = 0.0
        candidates.append((delta, fill))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _matching_native_full_close_order(
    positions: list[Position],
    orders: list[Order],
    *,
    close_side: str,
    total_quantity: float,
    reference_time: datetime,
    window_seconds: int,
) -> Order | None:
    if not positions:
        return None
    symbol = normalize_trading_symbol(positions[0].symbol)
    candidates: list[tuple[float, float, Order]] = []
    for order in orders:
        if normalize_trading_symbol(order.symbol) != symbol:
            continue
        if str(order.side or "").lower() != close_side:
            continue
        if not _is_native_full_close_placeholder(getattr(order, "exchange_order_id", None)):
            continue
        order_time = _aware(order.filled_at or order.created_at)
        if order_time is None:
            continue
        delta = abs((order_time - reference_time).total_seconds())
        if delta > max(int(window_seconds), 1):
            continue
        quantity_delta = abs(abs(_safe_float(order.quantity)) - total_quantity)
        candidates.append((delta, quantity_delta, order))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1]))[0][2]


def _fill_quantity(fill: FillGroup, target_quantity: float) -> float:
    contracts = _safe_float(fill.contracts)
    if target_quantity > 0 and contracts > target_quantity * 1.2:
        return target_quantity
    return contracts


def _contract_size_for_open_close(
    position: Position,
    fill: FillGroup,
    contract_sizes: dict[str, float],
    target_quantity: float,
) -> tuple[float, str]:
    inst_id = str(getattr(fill, "inst_id", "") or _position_okx_inst_id(position)).strip().upper()
    contract_size = _safe_float(contract_sizes.get(inst_id), 0.0)
    if contract_size > 0:
        return contract_size, "okx_public_instruments_ctVal"

    contracts = _safe_float(getattr(fill, "contracts", None), 0.0)
    if target_quantity > 0 and contracts > 0:
        inferred = target_quantity / contracts
        if inferred > 0:
            return inferred, "inferred_from_local_quantity_and_fill_contracts"
    return 0.0, "missing"


def _fill_quantity_with_contract_size(fill: FillGroup, contract_size: float) -> float:
    return _safe_float(getattr(fill, "contracts", None)) * max(_safe_float(contract_size), 0.0)


def _quantity_close_enough(actual: float, expected: float) -> bool:
    if actual <= 0 or expected <= 0:
        return False
    tolerance = max(expected * QUANTITY_TOLERANCE_RATIO, actual * QUANTITY_TOLERANCE_RATIO, 1e-8)
    return abs(actual - expected) <= tolerance


def _is_native_full_close_placeholder(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return not text or text == "none" or "okx_native_full_close" in text


def _group_native_full_close_positions(positions: list[Position]) -> list[list[Position]]:
    groups: dict[tuple[Any, ...], list[Position]] = {}
    for position in positions:
        closed_at = _aware(getattr(position, "closed_at", None))
        closed_bucket = closed_at.replace(microsecond=0).isoformat() if closed_at else ""
        key = (
            str(position.model_name or ""),
            str(position.execution_mode or ""),
            normalize_trading_symbol(position.symbol),
            str(position.side or "").lower(),
            closed_bucket,
        )
        groups.setdefault(key, []).append(position)
    return [
        sorted(group, key=lambda item: int(item.id))
        for group in groups.values()
        if len(group) > 1
    ]


async def apply_plans(plans: list[FillLinkPlan]) -> dict[str, Any]:
    if not plans:
        return {"applied": 0}
    backup_path = await _backup(plans)
    applied = 0
    created_order_rows = 0
    async with get_session_ctx() as session:
        for plan in plans:
            position = await session.get(Position, plan.position_id)
            if position is None:
                continue
            okx_inst_id = str(getattr(plan, "okx_inst_id", "") or "").strip().upper()
            if not okx_inst_id:
                continue
            existing_inst_id = _position_okx_inst_id(position)
            if existing_inst_id and existing_inst_id != okx_inst_id:
                continue
            if not str(getattr(position, "okx_inst_id", "") or "").strip():
                position.okx_inst_id = okx_inst_id
            if plan.link_kind == "entry":
                if str(getattr(position, "entry_exchange_order_id", "") or "").strip():
                    continue
                position.entry_exchange_order_id = plan.okx_order_id
            elif plan.link_kind == "close":
                if str(getattr(position, "close_exchange_order_id", "") or "").strip():
                    continue
                position.close_exchange_order_id = plan.okx_order_id
            else:
                continue
            if await _ensure_missing_order_row_for_fill_link(
                session,
                position=position,
                plan=plan,
            ):
                created_order_rows += 1
            _add_repair_reflection_marker(
                session,
                position=position,
                source=REPAIR_REFLECTION_SOURCE,
                plan=asdict(plan),
            )
            applied += 1
        await session.flush()
    return {
        "applied": applied,
        "created_order_rows": created_order_rows,
        "backup_path": str(backup_path),
    }


async def _ensure_missing_order_row_for_fill_link(
    session: Any,
    *,
    position: Position,
    plan: FillLinkPlan,
) -> bool:
    exchange_order_id = str(getattr(plan, "okx_order_id", "") or "").strip()
    okx_inst_id = str(getattr(plan, "okx_inst_id", "") or "").strip().upper()
    if not exchange_order_id or not okx_inst_id:
        return False
    exists = (
        await session.execute(
            select(Order.id)
            .where(Order.exchange_order_id == exchange_order_id)
            .limit(1)
        )
    ).scalar_one_or_none()
    if exists is not None:
        return False

    if plan.link_kind == "entry":
        order_side = _entry_side(position)
        filled_at = plan.fill_timestamp or getattr(position, "created_at", None)
    elif plan.link_kind == "close":
        order_side = _close_side(position)
        filled_at = plan.fill_timestamp or getattr(position, "closed_at", None)
    else:
        return False
    session.add(
        Order(
            model_name=str(position.model_name or "ensemble_trader"),
            execution_mode=str(position.execution_mode or "paper"),
            symbol=normalize_trading_symbol(okx_inst_id),
            side=order_side,
            order_type="market",
            quantity=_safe_float(plan.fill_quantity, abs(_safe_float(position.quantity))),
            price=_safe_float(plan.fill_price),
            status="filled",
            fee=_safe_float(plan.fill_fee),
            decision_id=None,
            exchange_order_id=exchange_order_id,
            filled_at=filled_at,
            created_at=filled_at,
        )
    )
    return True


async def apply_missing_order_row_plans(plans: list[MissingOrderRowPlan]) -> dict[str, Any]:
    if not plans:
        return {"applied": 0}
    backup_path = await _backup_order_rows(plans)
    applied = 0
    async with get_session_ctx() as session:
        for plan in plans:
            okx_inst_id = str(getattr(plan, "okx_inst_id", "") or "").strip().upper()
            if not okx_inst_id:
                continue
            exists = (
                await session.execute(
                    select(Order.id).where(Order.exchange_order_id == plan.exchange_order_id)
                )
            ).scalar_one_or_none()
            if exists is not None:
                continue
            session.add(
                Order(
                    model_name=plan.model_name,
                    execution_mode=plan.execution_mode,
                    symbol=normalize_trading_symbol(okx_inst_id),
                    side=plan.side,
                    order_type="market",
                    quantity=plan.quantity,
                    price=plan.price,
                    status="filled",
                    fee=plan.fee,
                    decision_id=plan.decision_id,
                    exchange_order_id=plan.exchange_order_id,
                    filled_at=plan.filled_at,
                    created_at=plan.filled_at,
                )
            )
            position = await session.get(Position, plan.position_id)
            if position is not None:
                _add_repair_reflection_marker(
                    session,
                    position=position,
                    source=REPAIR_REFLECTION_SOURCE,
                    plan=asdict(plan),
                )
            applied += 1
        await session.flush()
    return {"applied": applied, "backup_path": str(backup_path)}


async def apply_existing_order_decision_link_plans(
    plans: list[ExistingOrderDecisionLinkPlan],
) -> dict[str, Any]:
    if not plans:
        return {"applied": 0}
    backup_path = await _backup_existing_order_decision_links(plans)
    applied = 0
    skipped = 0
    async with get_session_ctx() as session:
        for plan in plans:
            order = await session.get(Order, int(plan.order_id))
            if order is None:
                skipped += 1
                continue
            if getattr(order, "decision_id", None) is not None:
                skipped += 1
                continue
            if str(getattr(order, "exchange_order_id", "") or "").strip() != plan.exchange_order_id:
                skipped += 1
                continue
            if str(getattr(order, "status", "") or "").lower() != "filled":
                skipped += 1
                continue
            if str(getattr(order, "side", "") or "").lower() != plan.side:
                skipped += 1
                continue
            decision = await session.get(AIDecision, int(plan.decision_id))
            if decision is None:
                skipped += 1
                continue
            if str(getattr(decision, "action", "") or "").lower() != plan.decision_action.lower():
                skipped += 1
                continue
            order.decision_id = int(plan.decision_id)
            applied += 1
        await session.flush()
    return {"applied": applied, "skipped": skipped, "backup_path": str(backup_path)}


async def apply_linked_protection_fill_order_plans(
    plans: list[LinkedProtectionFillOrderPlan],
) -> dict[str, Any]:
    if not plans:
        return {"applied": 0}
    backup_path = await _backup_linked_protection_fill_orders(plans)
    applied = 0
    async with get_session_ctx() as session:
        for plan in plans:
            okx_inst_id = str(getattr(plan, "okx_inst_id", "") or "").strip().upper()
            exchange_order_id = str(getattr(plan, "exchange_order_id", "") or "").strip()
            if not okx_inst_id or not exchange_order_id:
                continue
            exists = (
                await session.execute(
                    select(Order.id).where(Order.exchange_order_id == exchange_order_id).limit(1)
                )
            ).scalar_one_or_none()
            if exists is not None:
                continue
            filled_at = plan.filled_at or datetime.now(UTC)
            session.add(
                Order(
                    model_name=plan.model_name,
                    execution_mode=plan.execution_mode,
                    symbol=normalize_trading_symbol(okx_inst_id),
                    side=plan.side,
                    order_type="market",
                    quantity=plan.quantity,
                    price=plan.price,
                    status="filled",
                    fee=plan.fee,
                    decision_id=plan.decision_id,
                    exchange_order_id=exchange_order_id,
                    filled_at=filled_at,
                    created_at=filled_at,
                )
            )
            positions = (
                (
                    await session.execute(
                        select(Position).where(
                            Position.execution_mode == plan.execution_mode,
                            Position.symbol == normalize_trading_symbol(okx_inst_id),
                            Position.entry_exchange_order_id == plan.linked_exchange_order_id,
                        )
                    )
                )
                .scalars()
                .all()
            )
            for position in positions:
                _add_repair_reflection_marker(
                    session,
                    position=position,
                    source=REPAIR_REFLECTION_SOURCE,
                    plan=asdict(plan),
                )
            applied += 1
        await session.flush()
    return {"applied": applied, "backup_path": str(backup_path)}


async def apply_open_position_close_plans(plans: list[OpenPositionClosePlan]) -> dict[str, Any]:
    if not plans:
        return {"applied": 0}
    backup_path = await _backup_open_position_closes(plans)
    applied = 0
    async with get_session_ctx() as session:
        for plan in plans:
            okx_inst_id = str(getattr(plan, "okx_inst_id", "") or "").strip().upper()
            if not okx_inst_id:
                continue
            position = await session.get(Position, plan.position_id)
            if position is None:
                continue
            if not bool(getattr(position, "is_open", False)):
                continue
            if str(getattr(position, "close_exchange_order_id", "") or "").strip():
                continue
            existing_inst_id = _position_okx_inst_id(position)
            if existing_inst_id and existing_inst_id != okx_inst_id:
                continue
            existing_order = (
                await session.execute(
                    select(Order)
                    .where(Order.exchange_order_id == plan.okx_order_id)
                    .limit(1)
                )
            ).scalar_one_or_none()
            if existing_order is not None and not _existing_close_order_matches_plan(
                existing_order,
                plan,
            ):
                continue

            position.is_open = False
            position.current_price = plan.exit_price
            position.unrealized_pnl = 0.0
            position.realized_pnl = plan.fill_pnl or plan.computed_realized_pnl
            position.closed_at = plan.fill_timestamp or datetime.now(UTC)
            position.close_exchange_order_id = plan.okx_order_id
            if not str(getattr(position, "okx_inst_id", "") or "").strip():
                position.okx_inst_id = okx_inst_id

            if existing_order is None:
                session.add(
                    Order(
                        model_name=plan.model_name,
                        execution_mode=plan.execution_mode,
                        symbol=normalize_trading_symbol(okx_inst_id),
                        side=plan.close_side,
                        order_type="market",
                        quantity=plan.fill_quantity,
                        price=plan.exit_price,
                        status="filled",
                        fee=plan.close_fee,
                        decision_id=None,
                        exchange_order_id=plan.okx_order_id,
                        filled_at=plan.fill_timestamp,
                        created_at=plan.fill_timestamp,
                    )
                )
            _add_repair_reflection_marker(
                session,
                position=position,
                source=REPAIR_REFLECTION_SOURCE,
                plan=asdict(plan),
            )
            applied += 1
        await session.flush()
    return {"applied": applied, "backup_path": str(backup_path)}


async def apply_orphan_open_position_quarantine_plans(
    plans: list[OrphanOpenPositionQuarantinePlan],
) -> dict[str, Any]:
    if not plans:
        return {"applied": 0}
    backup_path = await _backup_orphan_open_position_quarantines(plans)
    applied = 0
    async with get_session_ctx() as session:
        for plan in plans:
            okx_inst_id = str(getattr(plan, "okx_inst_id", "") or "").strip().upper()
            if not okx_inst_id:
                continue
            position = await session.get(Position, int(plan.position_id))
            if position is None:
                continue
            if not bool(getattr(position, "is_open", False)):
                continue
            existing_inst_id = _position_okx_inst_id(position)
            if existing_inst_id != okx_inst_id:
                continue
            if str(getattr(position, "close_exchange_order_id", "") or "").strip():
                continue
            now = datetime.now(UTC)
            position.is_open = False
            position.current_price = plan.old_current_price or plan.entry_price
            position.unrealized_pnl = 0.0
            position.realized_pnl = 0.0
            position.closed_at = now
            position.close_exchange_order_id = f"{ORPHAN_QUARANTINE_CLOSE_PREFIX}{plan.position_id}"
            _add_repair_reflection_marker(
                session,
                position=position,
                source=ORPHAN_QUARANTINE_REFLECTION_SOURCE,
                plan=asdict(plan),
            )
            applied += 1
        await session.flush()
    return {"applied": applied, "backup_path": str(backup_path)}


def _existing_close_order_matches_plan(order: Order, plan: OpenPositionClosePlan) -> bool:
    if str(getattr(order, "status", "") or "").lower() != "filled":
        return False
    if str(getattr(order, "side", "") or "").lower() != plan.close_side:
        return False
    if str(getattr(order, "okx_inst_id", "") or "").strip().upper() != plan.okx_inst_id:
        return False
    if str(getattr(order, "okx_sync_status", "") or "").strip() not in TRUSTED_CLOSE_ORDER_SYNC_STATUSES:
        return False
    return _quantity_close_enough(
        _safe_float(getattr(order, "quantity", None)),
        _safe_float(plan.fill_quantity),
    )


async def apply_close_link_reassignment_plans(
    plans: list[CloseLinkReassignmentPlan],
) -> dict[str, Any]:
    if not plans:
        return {"applied": 0}
    backup_path = await _backup_close_link_reassignments(plans)
    applied = 0
    async with get_session_ctx() as session:
        for plan in plans:
            okx_inst_id = str(getattr(plan, "okx_inst_id", "") or "").strip().upper()
            if not okx_inst_id:
                continue
            position = await session.get(Position, plan.position_id)
            if position is None or bool(getattr(position, "is_open", False)):
                continue
            if str(getattr(position, "close_exchange_order_id", "") or "").strip() != plan.old_okx_order_id:
                continue
            existing_inst_id = _position_okx_inst_id(position)
            if existing_inst_id and existing_inst_id != okx_inst_id:
                continue
            new_id_owner = (
                await session.execute(
                    select(Order.id)
                    .where(Order.exchange_order_id == plan.new_okx_order_id)
                    .limit(1)
                )
            ).scalar_one_or_none()
            if new_id_owner is not None:
                continue

            position.close_exchange_order_id = plan.new_okx_order_id
            position.current_price = plan.exit_price
            position.realized_pnl = plan.fill_pnl or plan.computed_realized_pnl
            position.unrealized_pnl = 0.0
            position.closed_at = plan.fill_timestamp or position.closed_at
            if not str(getattr(position, "okx_inst_id", "") or "").strip():
                position.okx_inst_id = okx_inst_id

            order = (
                await session.execute(
                    select(Order)
                    .where(
                        Order.execution_mode == plan.execution_mode,
                        Order.exchange_order_id == plan.old_okx_order_id,
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            if order is not None:
                order.exchange_order_id = plan.new_okx_order_id
                order.symbol = normalize_trading_symbol(okx_inst_id)
                order.side = plan.close_side
                order.quantity = plan.new_fill_quantity
                order.price = plan.exit_price
                order.fee = plan.close_fee
                order.status = "filled"
                order.filled_at = plan.fill_timestamp or order.filled_at

            reflection_rows = (
                (
                    await session.execute(
                        select(TradeReflection).where(TradeReflection.position_id == plan.position_id)
                    )
                )
                .scalars()
                .all()
            )
            for reflection in reflection_rows:
                reflection.exit_price = plan.exit_price
                reflection.quantity = plan.new_fill_quantity
                reflection.realized_pnl = position.realized_pnl
                reflection.fee_estimate = plan.close_fee
                reflection.closed_at = position.closed_at
                reflection.outcome = (
                    "profit"
                    if _safe_float(position.realized_pnl) > 0
                    else "loss" if _safe_float(position.realized_pnl) < 0 else "flat"
                )

            _add_repair_reflection_marker(
                session,
                position=position,
                source=REPAIR_REFLECTION_SOURCE,
                plan=asdict(plan),
            )
            applied += 1
        await session.flush()
    return {"applied": applied, "backup_path": str(backup_path)}


async def apply_native_full_close_shared_plans(
    plans: list[NativeFullCloseSharedPlan],
) -> dict[str, Any]:
    if not plans:
        return {"applied": 0}
    backup_path = await _backup_native_full_close_shared(plans)
    applied = 0
    async with get_session_ctx() as session:
        for plan in plans:
            okx_inst_id = str(getattr(plan, "okx_inst_id", "") or "").strip().upper()
            if not okx_inst_id or not plan.position_ids:
                continue
            existing_order = (
                await session.execute(
                    select(Order.id)
                    .where(
                        Order.exchange_order_id == plan.okx_order_id,
                        Order.id != plan.close_order_id if plan.close_order_id is not None else True,
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            if existing_order is not None:
                continue
            positions: list[Position] = []
            for position_id in plan.position_ids:
                position = await session.get(Position, int(position_id))
                if position is None:
                    continue
                if bool(getattr(position, "is_open", False)):
                    continue
                if not _is_native_full_close_placeholder(
                    getattr(position, "close_exchange_order_id", None)
                ):
                    continue
                existing_inst_id = _position_okx_inst_id(position)
                if existing_inst_id and existing_inst_id != okx_inst_id:
                    continue
                positions.append(position)
            if len(positions) != len(plan.position_ids):
                continue

            for position in positions:
                ratio = abs(_safe_float(position.quantity)) / plan.total_quantity
                position.close_exchange_order_id = plan.okx_order_id
                position.current_price = plan.exit_price
                position.unrealized_pnl = 0.0
                position.realized_pnl = plan.fill_pnl * ratio
                position.closed_at = plan.fill_timestamp or position.closed_at
                if not str(getattr(position, "okx_inst_id", "") or "").strip():
                    position.okx_inst_id = okx_inst_id

                reflection_rows = (
                    (
                        await session.execute(
                            select(TradeReflection).where(TradeReflection.position_id == position.id)
                        )
                    )
                    .scalars()
                    .all()
                )
                for reflection in reflection_rows:
                    reflection.exit_price = plan.exit_price
                    reflection.quantity = abs(_safe_float(position.quantity))
                    reflection.realized_pnl = position.realized_pnl
                    reflection.fee_estimate = plan.close_fee * ratio
                    reflection.closed_at = position.closed_at
                    reflection.outcome = (
                        "profit"
                        if _safe_float(position.realized_pnl) > 0
                        else "loss" if _safe_float(position.realized_pnl) < 0 else "flat"
                    )
                _add_repair_reflection_marker(
                    session,
                    position=position,
                    source=REPAIR_REFLECTION_SOURCE,
                    plan=asdict(plan),
                )

            if plan.close_order_id is not None:
                order = await session.get(Order, int(plan.close_order_id))
                if order is not None and _is_native_full_close_placeholder(order.exchange_order_id):
                    order.exchange_order_id = plan.okx_order_id
                    order.symbol = normalize_trading_symbol(okx_inst_id)
                    order.side = plan.close_side
                    order.quantity = plan.fill_quantity
                    order.price = plan.exit_price
                    order.fee = plan.close_fee
                    order.status = "filled"
                    order.filled_at = plan.fill_timestamp or order.filled_at
            applied += 1
        await session.flush()
    return {"applied": applied, "backup_path": str(backup_path)}


def _add_repair_reflection_marker(
    session: Any,
    *,
    position: Position,
    source: str,
    plan: dict[str, Any],
) -> None:
    """Mark repaired history so clean training views quarantine the position."""

    session.add(
        TradeReflection(
            position_id=int(position.id),
            model_name=str(position.model_name or "ensemble_trader"),
            execution_mode=str(position.execution_mode or "paper"),
            symbol=str(position.symbol or ""),
            side=str(position.side or ""),
            entry_price=_safe_float(position.entry_price),
            exit_price=_safe_float(position.current_price),
            quantity=abs(_safe_float(position.quantity)),
            realized_pnl=_safe_float(position.realized_pnl),
            fee_estimate=0.0,
            hold_minutes=0.0,
            closed_at=position.closed_at,
            outcome=(
                "profit"
                if _safe_float(position.realized_pnl) > 0
                else "loss" if _safe_float(position.realized_pnl) < 0 else "flat"
            ),
            mistake_summary="historical OKX position link repaired; quarantine from training",
            improvement_summary="review OKX-backed repair before trusting this fact for training",
            expert_lessons={
                "source": source,
                "training_policy": "exclude_until_manual_trust",
                "repair_plan": _json_safe(plan),
            },
            source=source,
        )
    )


async def _backup(plans: list[FillLinkPlan]) -> Path:
    await asyncio.to_thread(BACKUP_DIR.mkdir, parents=True, exist_ok=True)
    path = BACKUP_DIR / f"missing_position_links_before_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    ids = sorted({plan.position_id for plan in plans})
    async with get_session_ctx() as session:
        positions = (
            (await session.execute(select(Position).where(Position.id.in_(ids)))).scalars().all()
            if ids
            else []
        )
    payload = {
        "plans": [_json_safe(asdict(plan)) for plan in plans],
        "positions": [_model_payload(position) for position in positions],
    }
    await asyncio.to_thread(
        path.write_text,
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


async def _backup_order_rows(plans: list[MissingOrderRowPlan]) -> Path:
    await asyncio.to_thread(BACKUP_DIR.mkdir, parents=True, exist_ok=True)
    path = BACKUP_DIR / f"missing_order_rows_before_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    payload = {"plans": [_json_safe(asdict(plan)) for plan in plans]}
    await asyncio.to_thread(
        path.write_text,
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


async def _backup_existing_order_decision_links(
    plans: list[ExistingOrderDecisionLinkPlan],
) -> Path:
    await asyncio.to_thread(BACKUP_DIR.mkdir, parents=True, exist_ok=True)
    path = BACKUP_DIR / f"existing_order_decision_links_before_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    order_ids = sorted({int(plan.order_id) for plan in plans if int(plan.order_id) > 0})
    decision_ids = sorted({int(plan.decision_id) for plan in plans if int(plan.decision_id) > 0})
    position_ids = sorted({int(plan.position_id) for plan in plans if int(plan.position_id) > 0})
    async with get_session_ctx() as session:
        orders = (
            (await session.execute(select(Order).where(Order.id.in_(order_ids)))).scalars().all()
            if order_ids
            else []
        )
        decisions = (
            (
                await session.execute(
                    select(AIDecision).where(AIDecision.id.in_(decision_ids))
                )
            )
            .scalars()
            .all()
            if decision_ids
            else []
        )
        positions = (
            (
                await session.execute(
                    select(Position).where(Position.id.in_(position_ids))
                )
            )
            .scalars()
            .all()
            if position_ids
            else []
        )
    payload = {
        "plans": [_json_safe(asdict(plan)) for plan in plans],
        "orders": [_model_payload(order) for order in orders],
        "decisions": [_model_payload(decision) for decision in decisions],
        "positions": [_model_payload(position) for position in positions],
        "policy": {
            "mutates_exchange": False,
            "only_updates_existing_filled_entry_orders_missing_decision_id": True,
            "requires_unique_time_symbol_side_decision": True,
        },
    }
    await asyncio.to_thread(
        path.write_text,
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


async def _backup_linked_protection_fill_orders(
    plans: list[LinkedProtectionFillOrderPlan],
) -> Path:
    await asyncio.to_thread(BACKUP_DIR.mkdir, parents=True, exist_ok=True)
    path = BACKUP_DIR / f"linked_protection_fill_orders_before_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    exchange_ids = sorted({plan.exchange_order_id for plan in plans if plan.exchange_order_id})
    linked_entry_ids = sorted(
        {plan.linked_exchange_order_id for plan in plans if plan.linked_exchange_order_id}
    )
    async with get_session_ctx() as session:
        existing_orders = (
            (
                await session.execute(
                    select(Order).where(Order.exchange_order_id.in_(exchange_ids))
                )
            )
            .scalars()
            .all()
            if exchange_ids
            else []
        )
        linked_orders = (
            (
                await session.execute(
                    select(Order).where(Order.exchange_order_id.in_(linked_entry_ids))
                )
            )
            .scalars()
            .all()
            if linked_entry_ids
            else []
        )
        positions = (
            (
                await session.execute(
                    select(Position).where(
                        Position.entry_exchange_order_id.in_(linked_entry_ids)
                    )
                )
            )
            .scalars()
            .all()
            if linked_entry_ids
            else []
        )
    payload = {
        "plans": [_json_safe(asdict(plan)) for plan in plans],
        "existing_orders": [_model_payload(order) for order in existing_orders],
        "linked_entry_orders": [_model_payload(order) for order in linked_orders],
        "positions": [_model_payload(position) for position in positions],
    }
    await asyncio.to_thread(
        path.write_text,
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


async def _backup_open_position_closes(plans: list[OpenPositionClosePlan]) -> Path:
    await asyncio.to_thread(BACKUP_DIR.mkdir, parents=True, exist_ok=True)
    path = BACKUP_DIR / f"open_position_closes_before_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    ids = sorted({plan.position_id for plan in plans})
    async with get_session_ctx() as session:
        positions = (
            (await session.execute(select(Position).where(Position.id.in_(ids)))).scalars().all()
            if ids
            else []
        )
        exchange_ids = [plan.okx_order_id for plan in plans if plan.okx_order_id]
        orders = (
            (
                await session.execute(
                    select(Order).where(Order.exchange_order_id.in_(exchange_ids))
                )
            )
            .scalars()
            .all()
            if exchange_ids
            else []
        )
    payload = {
        "plans": [_json_safe(asdict(plan)) for plan in plans],
        "positions": [_model_payload(position) for position in positions],
        "orders": [_model_payload(order) for order in orders],
    }
    await asyncio.to_thread(
        path.write_text,
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


async def _backup_orphan_open_position_quarantines(
    plans: list[OrphanOpenPositionQuarantinePlan],
) -> Path:
    await asyncio.to_thread(BACKUP_DIR.mkdir, parents=True, exist_ok=True)
    path = BACKUP_DIR / f"orphan_open_position_quarantines_before_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    ids = sorted({plan.position_id for plan in plans})
    async with get_session_ctx() as session:
        positions = (
            (await session.execute(select(Position).where(Position.id.in_(ids)))).scalars().all()
            if ids
            else []
        )
    payload = {
        "plans": [_json_safe(asdict(plan)) for plan in plans],
        "positions": [_model_payload(position) for position in positions],
        "policy": {
            "mutates_exchange": False,
            "creates_synthetic_close_order": False,
            "realized_pnl_after_quarantine": 0.0,
            "training_policy": "exclude_until_manual_trust",
        },
    }
    await asyncio.to_thread(
        path.write_text,
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


async def _backup_close_link_reassignments(plans: list[CloseLinkReassignmentPlan]) -> Path:
    await asyncio.to_thread(BACKUP_DIR.mkdir, parents=True, exist_ok=True)
    path = BACKUP_DIR / f"close_link_reassignments_before_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    ids = sorted({plan.position_id for plan in plans})
    exchange_ids = sorted(
        {
            exchange_id
            for plan in plans
            for exchange_id in (plan.old_okx_order_id, plan.new_okx_order_id)
            if exchange_id
        }
    )
    async with get_session_ctx() as session:
        positions = (
            (await session.execute(select(Position).where(Position.id.in_(ids)))).scalars().all()
            if ids
            else []
        )
        orders = (
            (
                await session.execute(
                    select(Order).where(Order.exchange_order_id.in_(exchange_ids))
                )
            )
            .scalars()
            .all()
            if exchange_ids
            else []
        )
        reflections = (
            (
                await session.execute(
                    select(TradeReflection).where(TradeReflection.position_id.in_(ids))
                )
            )
            .scalars()
            .all()
            if ids
            else []
        )
    payload = {
        "plans": [_json_safe(asdict(plan)) for plan in plans],
        "positions": [_model_payload(position) for position in positions],
        "orders": [_model_payload(order) for order in orders],
        "trade_reflections": [_model_payload(reflection) for reflection in reflections],
    }
    await asyncio.to_thread(
        path.write_text,
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


async def _backup_native_full_close_shared(plans: list[NativeFullCloseSharedPlan]) -> Path:
    await asyncio.to_thread(BACKUP_DIR.mkdir, parents=True, exist_ok=True)
    path = BACKUP_DIR / f"native_full_close_shared_before_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    ids = sorted({position_id for plan in plans for position_id in plan.position_ids})
    order_ids = sorted(
        {int(plan.close_order_id) for plan in plans if plan.close_order_id is not None}
    )
    exchange_ids = sorted({plan.okx_order_id for plan in plans if plan.okx_order_id})
    async with get_session_ctx() as session:
        positions = (
            (await session.execute(select(Position).where(Position.id.in_(ids)))).scalars().all()
            if ids
            else []
        )
        orders_by_id = (
            (await session.execute(select(Order).where(Order.id.in_(order_ids)))).scalars().all()
            if order_ids
            else []
        )
        orders_by_exchange_id = (
            (
                await session.execute(
                    select(Order).where(Order.exchange_order_id.in_(exchange_ids))
                )
            )
            .scalars()
            .all()
            if exchange_ids
            else []
        )
        reflections = (
            (
                await session.execute(
                    select(TradeReflection).where(TradeReflection.position_id.in_(ids))
                )
            )
            .scalars()
            .all()
            if ids
            else []
        )
    payload = {
        "plans": [_json_safe(asdict(plan)) for plan in plans],
        "positions": [_model_payload(position) for position in positions],
        "orders": [
            _model_payload(order)
            for order in [*orders_by_id, *orders_by_exchange_id]
        ],
        "trade_reflections": [_model_payload(reflection) for reflection in reflections],
    }
    await asyncio.to_thread(
        path.write_text,
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _model_payload(row: Any) -> dict[str, Any]:
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--decision-window-seconds", type=int, default=DEFAULT_DECISION_WINDOW_SECONDS)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--position-id", action="append", type=int, default=[])
    parser.add_argument("--create-missing-order-rows", action="store_true")
    parser.add_argument("--link-existing-order-decisions", action="store_true")
    parser.add_argument("--create-linked-protection-fill-orders", action="store_true")
    parser.add_argument("--close-missing-exchange-open-position", action="store_true")
    parser.add_argument("--quarantine-missing-exchange-open-position", action="store_true")
    parser.add_argument("--reassign-mismatched-close-links", action="store_true")
    parser.add_argument("--repair-native-full-close-shared", action="store_true")
    parser.add_argument("--exchange-order-id", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply and not args.position_id and not args.exchange_order_id:
        parser.error("--apply requires --position-id or --exchange-order-id after dry-run audit")
    return args


async def main() -> int:
    args = _parse_args()
    creds = settings.get_okx_credentials("paper")
    if not str(creds.get("api_key") or "").strip():
        print(
            json.dumps(
                {
                    "plans": [],
                    "okx_credentials_available": False,
                    "error": "OKX paper/demo credentials are not configured in this environment.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    with redirect_stdout(sys.stderr):
        plans = await collect_plans(
            days=args.days,
            window_seconds=args.window_seconds,
            limit=args.limit,
            position_ids=tuple(args.position_id or ()),
            exchange_order_ids=tuple(args.exchange_order_id or ()),
        )
        order_row_plans = (
            await collect_missing_order_row_plans(
                days=args.days,
                window_seconds=args.window_seconds,
                decision_window_seconds=args.decision_window_seconds,
                limit=args.limit,
                position_ids=tuple(args.position_id or ()),
                exchange_order_ids=tuple(args.exchange_order_id or ()),
            )
            if args.create_missing_order_rows
            else []
        )
        existing_order_decision_link_plans = (
            await collect_existing_order_decision_link_plans(
                days=args.days,
                decision_window_seconds=args.decision_window_seconds,
                limit=args.limit,
                position_ids=tuple(args.position_id or ()),
                exchange_order_ids=tuple(args.exchange_order_id or ()),
            )
            if args.link_existing_order_decisions
            else []
        )
        open_position_close_plans = (
            await collect_open_position_close_plans(
                days=args.days,
                window_seconds=args.window_seconds,
                limit=args.limit,
                position_ids=tuple(args.position_id or ()),
            )
            if args.close_missing_exchange_open_position
            else []
        )
        close_link_reassignment_plans = (
            await collect_close_link_reassignment_plans(
                days=args.days,
                window_seconds=args.window_seconds,
                limit=args.limit,
                position_ids=tuple(args.position_id or ()),
            )
            if args.reassign_mismatched_close_links
            else []
        )
        orphan_open_position_quarantine_plans = (
            await collect_orphan_open_position_quarantine_plans(
                days=args.days,
                limit=args.limit,
                position_ids=tuple(args.position_id or ()),
            )
            if args.quarantine_missing_exchange_open_position
            else []
        )
        native_full_close_shared_plans = (
            await collect_native_full_close_shared_plans(
                days=args.days,
                window_seconds=args.window_seconds,
                limit=args.limit,
                position_ids=tuple(args.position_id or ()),
            )
            if args.repair_native_full_close_shared
            else []
        )
        linked_protection_fill_order_plans = (
            await collect_linked_protection_fill_order_plans(
                days=args.days,
                limit=args.limit,
                exchange_order_ids=tuple(args.exchange_order_id or ()),
            )
            if args.create_linked_protection_fill_orders
            else []
        )
    result: dict[str, Any] = {
        "plans": [_json_safe(asdict(plan)) for plan in plans],
        "missing_order_row_plans": [_json_safe(asdict(plan)) for plan in order_row_plans],
        "existing_order_decision_link_plans": [
            _json_safe(asdict(plan)) for plan in existing_order_decision_link_plans
        ],
        "linked_protection_fill_order_plans": [
            _json_safe(asdict(plan)) for plan in linked_protection_fill_order_plans
        ],
        "open_position_close_plans": [
            _json_safe(asdict(plan)) for plan in open_position_close_plans
        ],
        "close_link_reassignment_plans": [
            _json_safe(asdict(plan)) for plan in close_link_reassignment_plans
        ],
        "orphan_open_position_quarantine_plans": [
            _json_safe(asdict(plan)) for plan in orphan_open_position_quarantine_plans
        ],
        "native_full_close_shared_plans": [
            _json_safe(asdict(plan)) for plan in native_full_close_shared_plans
        ],
        "apply": bool(args.apply),
        "position_ids": [int(item) for item in args.position_id or [] if int(item) > 0],
        "exchange_order_ids": [
            str(item or "").strip() for item in args.exchange_order_id or [] if str(item or "").strip()
        ],
        "apply_policy": "apply_requires_position_id_or_exchange_order_id",
    }
    if args.apply:
        with redirect_stdout(sys.stderr):
            result["apply_result"] = await apply_plans(plans)
            if args.create_missing_order_rows:
                result["apply_missing_order_rows_result"] = await apply_missing_order_row_plans(
                    order_row_plans
                )
            if args.link_existing_order_decisions:
                result["apply_existing_order_decision_link_result"] = (
                    await apply_existing_order_decision_link_plans(
                        existing_order_decision_link_plans
                    )
                )
            if args.create_linked_protection_fill_orders:
                result["apply_linked_protection_fill_order_result"] = (
                    await apply_linked_protection_fill_order_plans(
                        linked_protection_fill_order_plans
                    )
                )
            if args.close_missing_exchange_open_position:
                result["apply_open_position_close_result"] = await apply_open_position_close_plans(
                    open_position_close_plans
                )
            if args.reassign_mismatched_close_links:
                result["apply_close_link_reassignment_result"] = (
                    await apply_close_link_reassignment_plans(close_link_reassignment_plans)
                )
            if args.quarantine_missing_exchange_open_position:
                result["apply_orphan_open_position_quarantine_result"] = (
                    await apply_orphan_open_position_quarantine_plans(
                        orphan_open_position_quarantine_plans
                    )
                )
            if args.repair_native_full_close_shared:
                result["apply_native_full_close_shared_result"] = (
                    await apply_native_full_close_shared_plans(native_full_close_shared_plans)
                )
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
