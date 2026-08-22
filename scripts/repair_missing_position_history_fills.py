#!/usr/bin/env python3
# ruff: noqa: E402
"""Backfill OKX fills missing from historical-position evidence links.

The regular order-fact sync starts from local order ids.  A historical OKX
position can be fully closed by a later order that was never stored locally,
so that order is invisible to the dashboard even though OKX positions-history
reports the complete quantity.  This repair reads OKX fills directly and
links only exact lifecycle quantities; it never changes exchange PnL values.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_env_bootstrap import (  # noqa: E402
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

load_runtime_env_files(project_root=ROOT)
drop_privileges_to_runtime_user_if_needed(project_root=ROOT)

from sqlalchemy import select

from core.symbols import normalize_trading_symbol
from db.session import get_session_ctx
from executor.okx_executor import OKXExecutor
from models.trade import OkxPositionHistory, Order, Position
from services.okx_execution_slippage import build_okx_fill_mark_slippage
from services.okx_lifecycle_order_allocations import (
    LIFECYCLE_ORDER_ALLOCATIONS_KEY,
    build_lifecycle_order_allocation,
    build_lifecycle_order_allocation_document,
)
from services.okx_native_facts import (
    OKX_ACCOUNT_BILLS_TRADE_SOURCE,
    OkxNativeFactsClient,
    OkxNativeFillGroup,
)
from services.okx_order_fact_sync import (
    OKX_SYNC_OKX_ONLY,
    authoritative_orders_by_exchange_id,
)
from services.okx_position_history_store import (
    load_okx_position_history_records,
    publish_okx_position_history_watermark,
)

TOLERANCE_RATIO = 0.02
BACKUP_DIR = Path("data/codex_backups/missing-position-history-fills")
FILL_QUERY_BATCH_SIZE = 20
REVERSAL_BOUNDARY_TOLERANCE_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class FillPlan:
    history_id: int
    position_ids: tuple[int, ...]
    inst_id: str
    symbol: str
    link_kind: str
    target_contracts: float
    order_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RebuildPlan:
    history_id: int
    inst_id: str
    symbol: str
    entry_target_contracts: float
    close_target_contracts: float
    entry_order_ids: tuple[str, ...]
    close_order_ids: tuple[str, ...]
    old_entry_order_ids: tuple[str, ...]
    old_close_order_ids: tuple[str, ...]
    entry_matched: bool
    close_matched: bool
    entry_allocations: tuple[dict[str, Any], ...]
    close_allocations: tuple[dict[str, Any], ...]
    old_allocation_document: dict[str, Any]

    @property
    def allocation_document(self) -> dict[str, Any]:
        return build_lifecycle_order_allocation_document(
            entry=self.entry_allocations,
            close=self.close_allocations,
        )

    @property
    def changed(self) -> bool:
        return (
            self.entry_order_ids != self.old_entry_order_ids
            or self.close_order_ids != self.old_close_order_ids
            or self.allocation_document != self.old_allocation_document
        )

    @property
    def evidence_gaps(self) -> tuple[str, ...]:
        gaps: list[str] = []
        if self.entry_target_contracts > 0 and not self.entry_order_ids:
            gaps.append("missing_position_history_entry_orders")
        if self.close_target_contracts > 0 and not self.close_order_ids:
            gaps.append("missing_position_history_close_orders")
        if self.entry_target_contracts > 0 and not self.entry_matched:
            gaps.append("position_history_entry_quantity_not_matched_to_orders")
        if self.close_target_contracts > 0 and not self.close_matched:
            gaps.append("position_history_close_quantity_not_matched_to_orders")
        return tuple(gaps)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _ms(value: Any) -> datetime | None:
    stamp = _safe_float(value)
    if stamp <= 0:
        return None
    if stamp < 10_000_000_000:
        stamp *= 1000
    try:
        return datetime.fromtimestamp(stamp / 1000, tz=UTC)
    except (OSError, OverflowError, ValueError):
        return None


def _tokens(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = str(value or "").replace(";", ",").replace("|", ",").split(",")
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _chunks(values: Iterable[str], size: int) -> Iterable[set[str]]:
    """Yield bounded instrument batches so the native adapter cannot truncate them."""

    batch: set[str] = set()
    for value in sorted({str(item).strip().upper() for item in values if str(item).strip()}):
        batch.add(value)
        if len(batch) >= max(1, int(size)):
            yield batch
            batch = set()
    if batch:
        yield batch


def _matches(left: float, right: float) -> bool:
    if left <= 0 or right <= 0:
        return False
    return abs(left - right) <= max(left, right, 1.0) * TOLERANCE_RATIO


def _subset(groups: list[OkxNativeFillGroup], target: float) -> list[OkxNativeFillGroup]:
    if target <= 0 or not groups:
        return []
    ordered = sorted(groups, key=lambda item: item.timestamp or datetime.max.replace(tzinfo=UTC))
    if _matches(sum(item.contracts for item in ordered), target):
        return ordered
    best: list[OkxNativeFillGroup] = []
    best_delta = float("inf")

    def visit(index: int, current: list[OkxNativeFillGroup], total: float) -> None:
        nonlocal best, best_delta
        if total > target * (1 + TOLERANCE_RATIO):
            return
        if index >= len(ordered) or _matches(total, target):
            delta = abs(total - target)
            if delta < best_delta:
                best = list(current)
                best_delta = delta
            return
        if len(ordered) <= 24:
            visit(index + 1, [*current, ordered[index]], total + ordered[index].contracts)
            visit(index + 1, current, total)

    visit(0, [], 0.0)
    if not _matches(sum(item.contracts for item in best), target):
        return []
    return sorted(best, key=lambda item: item.timestamp or datetime.max.replace(tzinfo=UTC))


def _row_side(row: dict[str, Any]) -> str:
    value = str(row.get("direction") or row.get("side") or row.get("posSide") or "").lower()
    return value if value in {"long", "short"} else ""


def _entry_target_contracts(history: OkxPositionHistory, raw: dict[str, Any]) -> float:
    max_position = _safe_float(raw.get("openMaxPos") or history.open_max_pos, 0.0)
    close_total = _safe_float(raw.get("closeTotalPos") or history.close_total_pos, 0.0)
    close_type = str(raw.get("type") or raw.get("closeType") or history.close_status or "")
    if close_total > 0 and (close_type == "2" or history.close_status == "full"):
        return close_total
    return max_position


def _plan_for_row(
    history: OkxPositionHistory,
    fills: list[OkxNativeFillGroup],
    *,
    link_kind: str,
    existing_ids: set[str],
    linked_contracts: float = 0.0,
) -> FillPlan | None:
    raw = dict(history.raw_row or {})
    inst_id = str(history.inst_id or raw.get("instId") or "").strip().upper()
    side = str(history.side or _row_side(raw) or "").lower()
    if not inst_id or side not in {"long", "short"}:
        return None
    opened = _utc(history.opened_at) or _ms(raw.get("cTime"))
    closed = _utc(history.updated_at_okx) or _ms(raw.get("uTime"))
    if opened is None or closed is None:
        return None
    target = (
        _entry_target_contracts(history, raw)
        if link_kind == "entry"
        else _safe_float(raw.get("closeTotalPos") or history.close_total_pos, 0.0)
    )
    target = max(target - max(linked_contracts, 0.0), 0.0)
    if target <= 0:
        return None
    expected = "sell" if side == "short" else "buy"
    if link_kind == "close":
        expected = "buy" if side == "short" else "sell"
    start = opened - timedelta(hours=24 if link_kind == "entry" else 1)
    end = closed + timedelta(hours=1)
    candidates = [
        fill
        for fill in fills
        if str(fill.inst_id or "").upper() == inst_id
        and str(fill.side or "").lower() == expected
        and fill.timestamp is not None
        and start <= fill.timestamp <= end
        and fill.order_id not in existing_ids
    ]
    selected = _subset(candidates, target)
    if not selected:
        return None
    return FillPlan(
        history_id=int(history.id),
        position_ids=tuple(int(item) for item in _tokens(history.position_ids)),
        inst_id=inst_id,
        symbol=normalize_trading_symbol(inst_id),
        link_kind=link_kind,
        target_contracts=target,
        order_ids=tuple(item.order_id for item in selected),
    )


def _raw_fill_fact(fill: OkxNativeFillGroup, contract_size: float) -> dict[str, Any]:
    source = (
        OKX_ACCOUNT_BILLS_TRADE_SOURCE
        if fill.rows
        and all(
            str(row.get("_bb_fill_fact_source") or "").strip() == OKX_ACCOUNT_BILLS_TRADE_SOURCE
            for row in fill.rows
        )
        else "okx_fills_history"
    )
    execution_slippage = build_okx_fill_mark_slippage(
        order_id=fill.order_id,
        inst_id=fill.inst_id,
        side=fill.side,
        contracts=fill.contracts,
        average_price=fill.avg_price,
        contract_size=contract_size,
        rows=fill.rows,
    )
    execution_slippage["recovery_terminal"] = execution_slippage.get("complete") is not True
    execution_slippage["recovery_source"] = source
    return {
        "source": source,
        "rows": [dict(row) for row in fill.rows],
        "fee_abs": fill.fee_abs,
        "inst_id": fill.inst_id,
        "fill_pnl": fill.fill_pnl,
        "order_id": fill.order_id,
        "pos_side": fill.pos_side,
        "avg_price": fill.avg_price,
        "contracts": fill.contracts,
        "timestamp": fill.timestamp.isoformat() if fill.timestamp else None,
        "trade_ids": list(fill.trade_ids),
        "base_quantity": fill.contracts * contract_size,
        "contract_size": contract_size,
        "contract_size_verified": True,
        "contract_size_source": "okx_public_instruments",
        "fills_history_confirmed": source == "okx_fills_history",
        "account_bills_trade_confirmed": source == OKX_ACCOUNT_BILLS_TRADE_SOURCE,
        "execution_slippage": execution_slippage,
    }


def _order_fill_fact_needs_refresh(
    order: Order,
    fill: OkxNativeFillGroup,
    contract_size: float,
) -> bool:
    expected = _raw_fill_fact(fill, contract_size)
    raw = order.okx_raw_fills if isinstance(order.okx_raw_fills, dict) else {}

    # Recovery metadata can legitimately differ between targeted and paged OKX
    # pulls. Refresh only when the authoritative execution contract is missing
    # or inconsistent; otherwise a transient pagination shape would rewrite the
    # entire historical order table on every repair run.
    if raw.get("contract_size_verified") is not True:
        return True
    if str(raw.get("contract_size_source") or "").strip() != "okx_public_instruments":
        return True
    if (
        expected["fills_history_confirmed"] is True
        and raw.get("fills_history_confirmed") is not True
    ):
        return True
    if (
        expected["account_bills_trade_confirmed"] is True
        and raw.get("account_bills_trade_confirmed") is not True
    ):
        return True
    if (
        expected["account_bills_trade_confirmed"] is True
        and str(raw.get("source") or "").strip() != OKX_ACCOUNT_BILLS_TRADE_SOURCE
    ):
        return True

    if str(raw.get("order_id") or "").strip() != fill.order_id:
        return True
    if str(getattr(order, "exchange_order_id", "") or "").strip() != fill.order_id:
        return True

    numeric_contract = (
        (raw.get("contracts"), fill.contracts),
        (raw.get("avg_price"), fill.avg_price),
        (raw.get("fee_abs"), fill.fee_abs),
        (raw.get("base_quantity"), fill.contracts * contract_size),
        (raw.get("contract_size"), contract_size),
        (getattr(order, "quantity", None), fill.contracts * contract_size),
        (getattr(order, "okx_fill_contracts", None), fill.contracts),
        (getattr(order, "price", None), fill.avg_price),
        (getattr(order, "fee", None), fill.fee_abs),
    )
    return any(
        not math.isclose(
            _safe_float(actual, float("nan")),
            expected_value,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        for actual, expected_value in numeric_contract
    )


def _plan_needs_history_rebuild(plan: RebuildPlan, history: OkxPositionHistory) -> bool:
    # The canonical fill subset is deterministic. Any complete changed plan
    # must be replayed, including an old subset that happened to have no gap
    # marker; otherwise newly discovered partial fills remain invisible.
    return bool(plan.changed and not plan.evidence_gaps)


async def _load_data(days: int) -> tuple[list[OkxPositionHistory], list[Position], list[Order]]:
    since = datetime.now(UTC) - timedelta(days=max(days, 1))
    async with get_session_ctx() as session:
        histories = [
            history
            for history in await load_okx_position_history_records(
                session,
                mode="paper",
                limit=5000,
            )
            if (_utc(history.updated_at_okx) or datetime.min.replace(tzinfo=UTC)) >= since
        ]
        positions = list(
            (
                await session.execute(select(Position).where(Position.execution_mode == "paper"))
            ).scalars()
        )
        orders = list(
            (await session.execute(select(Order).where(Order.execution_mode == "paper"))).scalars()
        )
    return histories, positions, orders


async def _fetch_native_facts(
    histories: list[OkxPositionHistory],
) -> tuple[dict[str, OkxNativeFillGroup], dict[str, float]]:
    inst_ids = {str(item.inst_id or "").strip().upper() for item in histories if item.inst_id}
    if not inst_ids:
        return {}, {}
    since = min(
        (_utc(item.opened_at) for item in histories if _utc(item.opened_at)),
        default=datetime.now(UTC),
    ) - timedelta(days=1)
    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    try:
        await executor.initialize()
        client = OkxNativeFactsClient(executor)
        fills_by_id: dict[str, OkxNativeFillGroup] = {}
        # OkxNativeFactsClient intentionally bounds one request to 20 contracts.
        # Query every batch explicitly; passing all instruments at once would
        # silently omit the contracts after the first batch (for example STRK).
        for inst_batch in _chunks(inst_ids, FILL_QUERY_BATCH_SIZE):
            batch_fills = await client.fetch_fill_groups(
                inst_ids=inst_batch,
                since=since,
                limit=100,
                max_pages=20,
                include_historical=True,
                strict=True,
            )
            for fill in batch_fills:
                if fill.order_id:
                    fills_by_id[fill.order_id] = fill
        fetched_inst_ids = {
            str(fill.inst_id or "").strip().upper() for fill in fills_by_id.values()
        }
        for inst_id in sorted(inst_ids - fetched_inst_ids):
            inst_histories = [
                item for item in histories if str(item.inst_id or "").strip().upper() == inst_id
            ]
            opened = [_utc(item.opened_at) for item in inst_histories if _utc(item.opened_at)]
            closed = [
                _utc(item.updated_at_okx) for item in inst_histories if _utc(item.updated_at_okx)
            ]
            if not opened or not closed:
                continue
            bill_fills = await client.fetch_trade_bill_fill_groups(
                inst_ids={inst_id},
                since=min(opened) - timedelta(minutes=10),
                until=max(closed) + timedelta(minutes=10),
                limit=100,
                max_pages=20,
                strict=True,
            )
            for fill in bill_fills:
                if fill.order_id:
                    fills_by_id.setdefault(fill.order_id, fill)
        specs = await client.fetch_contract_sizes(inst_ids=inst_ids)
        for history in histories:
            inst_id = str(history.inst_id or "").strip().upper()
            raw = dict(history.raw_row or {})
            stored_spec = raw.get("_bb_contract_spec")
            stored_spec = stored_spec if isinstance(stored_spec, dict) else {}
            if (
                inst_id
                and inst_id not in specs
                and str(stored_spec.get("source") or "").strip() == "okx_public_instruments"
            ):
                contract_size = _safe_float(stored_spec.get("ctVal"), 0.0) * _safe_float(
                    stored_spec.get("ctMult"),
                    1.0,
                )
                if contract_size > 0:
                    specs[inst_id] = contract_size
    finally:
        await executor.shutdown()
    return fills_by_id, specs


async def collect_plans(
    days: int,
) -> tuple[list[FillPlan], dict[str, OkxNativeFillGroup], dict[str, float]]:
    histories, _positions, _orders = await _load_data(days)
    fills_by_id, specs = await _fetch_native_facts(histories)
    fills = list(fills_by_id.values())
    plans: list[FillPlan] = []
    reserved_fill_ids: set[str] = set()
    for history in histories:
        linked = (
            set(_tokens(history.linked_order_ids))
            | set(_tokens(history.entry_order_ids))
            | set(_tokens(history.close_order_ids))
        )
        # Existing but unlinked order facts are valid repair candidates. Only
        # lifecycle-linked and already-reserved fills must be excluded.
        known = linked | reserved_fill_ids
        history_inst_id = str(history.inst_id or "").strip().upper()
        row_fills = [
            fill for fill in fills if str(fill.inst_id or "").strip().upper() == history_inst_id
        ]
        for kind in ("entry", "close"):
            linked_ids = _tokens(
                history.entry_order_ids if kind == "entry" else history.close_order_ids
            )
            linked_contracts = sum(
                fills_by_id[item].contracts for item in linked_ids if item in fills_by_id
            )
            if kind == "entry" and _matches(linked_contracts, _safe_float(history.open_max_pos)):
                continue
            if kind == "close" and _matches(linked_contracts, _safe_float(history.close_total_pos)):
                continue
            plan = _plan_for_row(
                history,
                row_fills,
                link_kind=kind,
                existing_ids=known,
                linked_contracts=linked_contracts,
            )
            if plan is not None:
                plans.append(plan)
                known.update(plan.order_ids)
                reserved_fill_ids.update(plan.order_ids)
    return plans, fills_by_id, specs


def _canonical_fill_subset(
    history: OkxPositionHistory,
    fills: list[OkxNativeFillGroup],
    *,
    link_kind: str,
    used_order_ids: set[str],
) -> tuple[list[OkxNativeFillGroup], float, bool]:
    raw = dict(history.raw_row or {})
    side = str(history.side or _row_side(raw) or "").lower()
    target = (
        _entry_target_contracts(history, raw)
        if link_kind == "entry"
        else _safe_float(raw.get("closeTotalPos") or history.close_total_pos, 0.0)
    )
    if target <= 0:
        return [], target, True
    opened = _utc(history.opened_at) or _ms(raw.get("cTime"))
    closed = _utc(history.updated_at_okx) or _ms(raw.get("uTime"))
    inst_id = str(history.inst_id or raw.get("instId") or "").strip().upper()
    if opened is None or closed is None or not inst_id or side not in {"long", "short"}:
        return [], target, False
    expected_side = "sell" if side == "short" else "buy"
    if link_kind == "close":
        expected_side = "buy" if side == "short" else "sell"
    window = timedelta(seconds=600)
    candidates = [
        fill
        for fill in fills
        if str(fill.inst_id or "").strip().upper() == inst_id
        and str(fill.side or "").lower() == expected_side
        and fill.timestamp is not None
        and opened - window <= fill.timestamp <= closed + window
        and fill.order_id not in used_order_ids
    ]
    selected = _subset(candidates, target)
    matched = bool(selected and _matches(sum(item.contracts for item in selected), target))
    return selected if matched else [], target, matched


def _strict_contracts_match(left: float, right: float) -> bool:
    return bool(left > 0 and right > 0 and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12))


def _apply_reversal_boundary_allocations(
    histories: list[OkxPositionHistory],
    plans: list[RebuildPlan],
    fills_by_inst: dict[str, list[OkxNativeFillGroup]],
) -> list[RebuildPlan]:
    """Split only an exact net-mode reversal order across adjacent lifecycles."""

    plans_by_id = {plan.history_id: plan for plan in plans}
    order_owners: dict[str, set[tuple[int, str]]] = {}
    for plan in plans:
        for role, order_ids in (
            ("entry", plan.entry_order_ids),
            ("close", plan.close_order_ids),
        ):
            for order_id in order_ids:
                order_owners.setdefault(order_id, set()).add((plan.history_id, role))
    histories_by_inst: dict[str, list[OkxPositionHistory]] = {}
    for history in histories:
        inst_id = str(history.inst_id or "").strip().upper()
        if inst_id and int(history.id) in plans_by_id:
            histories_by_inst.setdefault(inst_id, []).append(history)

    for inst_id, inst_histories in histories_by_inst.items():
        ordered = sorted(
            inst_histories,
            key=lambda history: (
                _utc(history.opened_at) or datetime.max.replace(tzinfo=UTC),
                _utc(history.updated_at_okx) or datetime.max.replace(tzinfo=UTC),
                int(history.id),
            ),
        )
        for previous, following in zip(ordered, ordered[1:], strict=False):
            previous_plan = plans_by_id[int(previous.id)]
            following_plan = plans_by_id[int(following.id)]
            previous_side = str(previous.side or _row_side(dict(previous.raw_row or {}))).lower()
            following_side = str(following.side or _row_side(dict(following.raw_row or {}))).lower()
            if (
                previous_side not in {"long", "short"}
                or following_side not in {"long", "short"}
                or previous_side == following_side
            ):
                continue
            boundary_closed = _utc(previous.updated_at_okx) or _ms(
                dict(previous.raw_row or {}).get("uTime")
            )
            boundary_opened = _utc(following.opened_at) or _ms(
                dict(following.raw_row or {}).get("cTime")
            )
            if (
                boundary_closed is None
                or boundary_opened is None
                or abs((boundary_closed - boundary_opened).total_seconds())
                > REVERSAL_BOUNDARY_TOLERANCE_SECONDS
            ):
                continue
            expected_side = "sell" if previous_side == "long" else "buy"
            following_entry_side = "sell" if following_side == "short" else "buy"
            if expected_side != following_entry_side:
                continue
            allowed_owners = {
                (previous_plan.history_id, "close"),
                (following_plan.history_id, "entry"),
            }
            boundary_candidates = [
                fill
                for fill in fills_by_inst.get(inst_id, [])
                if str(fill.side or "").strip().lower() == expected_side
                and fill.timestamp is not None
                and abs((fill.timestamp - boundary_closed).total_seconds())
                <= REVERSAL_BOUNDARY_TOLERANCE_SECONDS
                and not (order_owners.get(fill.order_id, set()) - allowed_owners)
            ]
            matches: list[
                tuple[
                    OkxNativeFillGroup,
                    list[OkxNativeFillGroup],
                    list[OkxNativeFillGroup],
                    float,
                    float,
                ]
            ] = []
            for boundary_fill in boundary_candidates:
                previous_other_fills = [
                    fill
                    for fill in fills_by_inst.get(inst_id, [])
                    if fill.order_id != boundary_fill.order_id
                    and str(fill.side or "").strip().lower() == expected_side
                    and fill.timestamp is not None
                    and (_utc(previous.opened_at) or boundary_closed)
                    <= fill.timestamp
                    < boundary_fill.timestamp
                    and not (
                        order_owners.get(fill.order_id, set())
                        - {(previous_plan.history_id, "close")}
                    )
                ]
                following_other_fills = [
                    fill
                    for fill in fills_by_inst.get(inst_id, [])
                    if fill.order_id != boundary_fill.order_id
                    and str(fill.side or "").strip().lower() == expected_side
                    and fill.timestamp is not None
                    and boundary_fill.timestamp
                    < fill.timestamp
                    <= (_utc(following.updated_at_okx) or boundary_opened)
                    and not (
                        order_owners.get(fill.order_id, set())
                        - {(following_plan.history_id, "entry")}
                    )
                ]
                previous_residual = previous_plan.close_target_contracts - sum(
                    fill.contracts for fill in previous_other_fills
                )
                following_residual = following_plan.entry_target_contracts - sum(
                    fill.contracts for fill in following_other_fills
                )
                if previous_residual <= 0 or following_residual <= 0:
                    continue
                if not _strict_contracts_match(
                    boundary_fill.contracts,
                    previous_residual + following_residual,
                ):
                    continue
                matches.append(
                    (
                        boundary_fill,
                        previous_other_fills,
                        following_other_fills,
                        previous_residual,
                        following_residual,
                    )
                )
            if len(matches) != 1:
                continue
            (
                boundary_fill,
                previous_other_fills,
                following_other_fills,
                previous_residual,
                following_residual,
            ) = matches[0]
            boundary_at = (
                boundary_fill.timestamp.isoformat()
                if boundary_fill.timestamp is not None
                else boundary_closed.isoformat()
            )
            previous_allocation = build_lifecycle_order_allocation(
                order_id=boundary_fill.order_id,
                allocated_contracts=previous_residual,
                order_contracts=boundary_fill.contracts,
                boundary_at=boundary_at,
                peer_history_id=int(following.id),
                peer_role="entry",
            )
            following_allocation = build_lifecycle_order_allocation(
                order_id=boundary_fill.order_id,
                allocated_contracts=following_residual,
                order_contracts=boundary_fill.contracts,
                boundary_at=boundary_at,
                peer_history_id=int(previous.id),
                peer_role="close",
            )
            plans_by_id[int(previous.id)] = replace(
                previous_plan,
                close_order_ids=tuple(
                    fill.order_id
                    for fill in sorted(
                        [*previous_other_fills, boundary_fill],
                        key=lambda fill: fill.timestamp or datetime.max.replace(tzinfo=UTC),
                    )
                ),
                close_matched=True,
                close_allocations=(previous_allocation,),
            )
            plans_by_id[int(following.id)] = replace(
                following_plan,
                entry_order_ids=tuple(
                    fill.order_id
                    for fill in sorted(
                        [boundary_fill, *following_other_fills],
                        key=lambda fill: fill.timestamp or datetime.max.replace(tzinfo=UTC),
                    )
                ),
                entry_matched=True,
                entry_allocations=(following_allocation,),
            )
            for owners in order_owners.values():
                owners.discard((previous_plan.history_id, "close"))
                owners.discard((following_plan.history_id, "entry"))
            for fill in [*previous_other_fills, boundary_fill]:
                order_owners.setdefault(fill.order_id, set()).add(
                    (previous_plan.history_id, "close")
                )
            for fill in [boundary_fill, *following_other_fills]:
                order_owners.setdefault(fill.order_id, set()).add(
                    (following_plan.history_id, "entry")
                )
    return [plans_by_id[plan.history_id] for plan in plans]


async def collect_rebuild_plans(
    days: int,
) -> tuple[list[RebuildPlan], dict[str, OkxNativeFillGroup], dict[str, float]]:
    histories, _positions, _orders = await _load_data(days)
    fills_by_id, specs = await _fetch_native_facts(histories)
    fills_by_inst: dict[str, list[OkxNativeFillGroup]] = {}
    for fill in fills_by_id.values():
        inst_id = str(fill.inst_id or "").strip().upper()
        if inst_id:
            fills_by_inst.setdefault(inst_id, []).append(fill)
    ordered_histories = sorted(
        histories,
        key=lambda history: (
            _utc(history.opened_at) or datetime.max.replace(tzinfo=UTC),
            _utc(history.updated_at_okx) or datetime.max.replace(tzinfo=UTC),
            int(history.id),
        ),
    )
    used_order_ids: set[str] = set()
    plans: list[RebuildPlan] = []
    for history in ordered_histories:
        inst_id = str(history.inst_id or "").strip().upper()
        row_fills = fills_by_inst.get(inst_id, [])
        if not row_fills:
            continue
        entries, entry_target, entry_matched = _canonical_fill_subset(
            history,
            row_fills,
            link_kind="entry",
            used_order_ids=used_order_ids,
        )
        if entry_matched:
            used_order_ids.update(fill.order_id for fill in entries)
        closes, close_target, close_matched = _canonical_fill_subset(
            history,
            row_fills,
            link_kind="close",
            used_order_ids=used_order_ids,
        )
        if close_matched:
            used_order_ids.update(fill.order_id for fill in closes)
        plans.append(
            RebuildPlan(
                history_id=int(history.id),
                inst_id=inst_id,
                symbol=normalize_trading_symbol(inst_id),
                entry_target_contracts=entry_target,
                close_target_contracts=close_target,
                entry_order_ids=tuple(fill.order_id for fill in entries),
                close_order_ids=tuple(fill.order_id for fill in closes),
                old_entry_order_ids=tuple(_tokens(history.entry_order_ids)),
                old_close_order_ids=tuple(_tokens(history.close_order_ids)),
                entry_matched=entry_matched,
                close_matched=close_matched,
                entry_allocations=(),
                close_allocations=(),
                old_allocation_document=(
                    dict(dict(history.raw_row or {})[LIFECYCLE_ORDER_ALLOCATIONS_KEY])
                    if isinstance(
                        dict(history.raw_row or {}).get(LIFECYCLE_ORDER_ALLOCATIONS_KEY),
                        dict,
                    )
                    else {}
                ),
            )
        )
    plans = _apply_reversal_boundary_allocations(
        ordered_histories,
        plans,
        fills_by_inst,
    )
    return plans, fills_by_id, specs


async def apply_plans(
    plans: list[FillPlan], fills_by_id: dict[str, OkxNativeFillGroup], specs: dict[str, float]
) -> dict[str, Any]:
    if not plans:
        publish_okx_position_history_watermark("paper")
        return {"applied": 0, "orders_created": 0}
    await asyncio.to_thread(BACKUP_DIR.mkdir, parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"before-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    async with get_session_ctx() as session:
        history_ids = [plan.history_id for plan in plans]
        histories = {
            int(item.id): item
            for item in (
                await session.execute(
                    select(OkxPositionHistory).where(OkxPositionHistory.id.in_(history_ids))
                )
            ).scalars()
        }
        existing_orders = {
            str(item.exchange_order_id): item
            for item in (
                await session.execute(select(Order).where(Order.execution_mode == "paper"))
            ).scalars()
            if item.exchange_order_id
        }
        backup.write_text(
            json.dumps(
                [
                    {
                        "history_id": plan.history_id,
                        "position_ids": plan.position_ids,
                        "link_kind": plan.link_kind,
                        "order_ids": plan.order_ids,
                    }
                    for plan in plans
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        created = 0
        linked = 0
        for plan in plans:
            history = histories.get(plan.history_id)
            if history is None:
                continue
            contract_size = _safe_float(specs.get(plan.inst_id), 0.0)
            if contract_size <= 0:
                continue
            target_ids = _tokens(
                history.entry_order_ids if plan.link_kind == "entry" else history.close_order_ids
            )
            for exchange_id in plan.order_ids:
                fill = fills_by_id.get(exchange_id)
                if fill is None:
                    continue
                order = existing_orders.get(exchange_id)
                if order is None:
                    order = Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol=plan.symbol,
                        side=fill.side,
                        order_type="market",
                        quantity=fill.contracts * contract_size,
                        price=fill.avg_price,
                        status="filled",
                        fee=fill.fee_abs,
                        exchange_order_id=exchange_id,
                        filled_at=fill.timestamp,
                        created_at=fill.timestamp,
                        okx_inst_id=plan.inst_id,
                        okx_trade_ids=",".join(fill.trade_ids),
                        okx_fill_contracts=fill.contracts,
                        okx_fill_pnl=fill.fill_pnl,
                        okx_sync_status=OKX_SYNC_OKX_ONLY,
                        okx_raw_fills=_raw_fill_fact(fill, contract_size),
                    )
                    session.add(order)
                    existing_orders[exchange_id] = order
                    created += 1
                if exchange_id not in target_ids:
                    target_ids.append(exchange_id)
                linked += 1
            if plan.link_kind == "entry":
                history.entry_order_ids = target_ids
            else:
                history.close_order_ids = target_ids
            history.linked_order_ids = list(
                dict.fromkeys(_tokens(history.entry_order_ids) + _tokens(history.close_order_ids))
            )
            raw = dict(history.raw_row or {})
            raw["_dashboard_entry_order_ids"] = list(history.entry_order_ids or [])
            raw["_dashboard_close_order_ids"] = list(history.close_order_ids or [])
            raw["_dashboard_linked_order_ids"] = list(history.linked_order_ids or [])
            history.raw_row = raw
            history.evidence_gaps = [
                gap
                for gap in _tokens(history.evidence_gaps)
                if gap
                not in {
                    "missing_position_history_entry_orders",
                    "missing_position_history_close_orders",
                }
            ]
        await session.flush()
    # The dashboard may serve a persisted closed-ledger snapshot. Advance the
    # mirror watermark only after the transaction has committed so readers drop
    # stale evidence links immediately.
    publish_okx_position_history_watermark("paper")
    return {"applied": linked, "orders_created": created, "backup": str(backup)}


async def apply_rebuild_plans(
    plans: list[RebuildPlan],
    fills_by_id: dict[str, OkxNativeFillGroup],
    specs: dict[str, float],
) -> dict[str, Any]:
    await asyncio.to_thread(BACKUP_DIR.mkdir, parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"rebuild-before-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    complete_plans = [plan for plan in plans if not plan.evidence_gaps]
    skipped_incomplete = sum(1 for plan in plans if plan.changed and plan.evidence_gaps)
    if not plans:
        publish_okx_position_history_watermark("paper")
        return {
            "applied_histories": 0,
            "orders_created": 0,
            "skipped_histories": 0,
            "skipped_incomplete_histories": skipped_incomplete,
            "backup": str(backup),
        }
    if not complete_plans:
        publish_okx_position_history_watermark("paper")
        return {
            "applied_histories": 0,
            "orders_created": 0,
            "skipped_histories": 0,
            "skipped_incomplete_histories": skipped_incomplete,
            "backup": str(backup),
        }

    async with get_session_ctx() as session:
        histories = {
            int(item.id): item
            for item in (
                await session.execute(
                    select(OkxPositionHistory).where(
                        OkxPositionHistory.id.in_([plan.history_id for plan in complete_plans])
                    )
                )
            ).scalars()
        }
        existing_orders = authoritative_orders_by_exchange_id(
            (
                await session.execute(select(Order).where(Order.execution_mode == "paper"))
            ).scalars()
        )
        created = 0
        applied = 0
        skipped = 0
        refreshed = 0
        applicable_plans: list[RebuildPlan] = []
        rebuild_history_ids: set[int] = set()
        for plan in complete_plans:
            history = histories.get(plan.history_id)
            if history is None:
                continue
            contract_size = _safe_float(specs.get(plan.inst_id), 0.0)
            selected_order_ids = [*plan.entry_order_ids, *plan.close_order_ids]
            fact_refresh_needed = bool(
                contract_size > 0
                and any(
                    (fill := fills_by_id.get(exchange_id)) is not None
                    and (
                        exchange_id not in existing_orders
                        or _order_fill_fact_needs_refresh(
                            existing_orders[exchange_id],
                            fill,
                            contract_size,
                        )
                    )
                    for exchange_id in selected_order_ids
                )
            )
            link_rebuild_needed = _plan_needs_history_rebuild(plan, history)
            if link_rebuild_needed:
                rebuild_history_ids.add(plan.history_id)
            if link_rebuild_needed or fact_refresh_needed:
                applicable_plans.append(plan)
        backup.write_text(
            json.dumps([asdict(plan) for plan in applicable_plans], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for plan in applicable_plans:
            history = histories.get(plan.history_id)
            if history is None:
                skipped += 1
                continue
            selected_order_ids = [*plan.entry_order_ids, *plan.close_order_ids]
            contract_size = _safe_float(specs.get(plan.inst_id), 0.0)
            missing_ids = [
                order_id for order_id in selected_order_ids if order_id not in existing_orders
            ]
            if missing_ids and contract_size <= 0:
                skipped += 1
                continue
            for exchange_id in selected_order_ids:
                fill = fills_by_id.get(exchange_id)
                if fill is None:
                    continue
                if exchange_id in existing_orders:
                    order = existing_orders[exchange_id]
                    fact_changed = _order_fill_fact_needs_refresh(
                        order,
                        fill,
                        contract_size,
                    )
                    order.status = "filled"
                    order.side = fill.side
                    order.price = fill.avg_price
                    order.fee = fill.fee_abs
                    order.filled_at = fill.timestamp
                    order.okx_inst_id = plan.inst_id
                    order.okx_trade_ids = ",".join(fill.trade_ids)
                    order.okx_fill_contracts = fill.contracts
                    order.okx_fill_pnl = fill.fill_pnl
                    if str(order.okx_sync_status or "").strip() not in {
                        "okx_confirmed",
                        "okx_execution_result_confirmed",
                    }:
                        order.okx_sync_status = OKX_SYNC_OKX_ONLY
                    if contract_size > 0:
                        order.quantity = fill.contracts * contract_size
                        current_raw = (
                            dict(order.okx_raw_fills)
                            if isinstance(order.okx_raw_fills, dict)
                            else {}
                        )
                        order.okx_raw_fills = {
                            **current_raw,
                            **_raw_fill_fact(fill, contract_size),
                        }
                    refreshed += int(fact_changed)
                    continue
                order = Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol=plan.symbol,
                    side=fill.side,
                    order_type="market",
                    quantity=fill.contracts * contract_size,
                    price=fill.avg_price,
                    status="filled",
                    fee=fill.fee_abs,
                    exchange_order_id=exchange_id,
                    filled_at=fill.timestamp,
                    created_at=fill.timestamp,
                    okx_inst_id=plan.inst_id,
                    okx_trade_ids=",".join(fill.trade_ids),
                    okx_fill_contracts=fill.contracts,
                    okx_fill_pnl=fill.fill_pnl,
                    okx_sync_status=OKX_SYNC_OKX_ONLY,
                    okx_raw_fills=_raw_fill_fact(fill, contract_size),
                )
                session.add(order)
                existing_orders[exchange_id] = order
                created += 1
                refreshed += 1
            if plan.history_id in rebuild_history_ids:
                history.entry_order_ids = list(plan.entry_order_ids)
                history.close_order_ids = list(plan.close_order_ids)
                history.linked_order_ids = list(
                    dict.fromkeys([*plan.entry_order_ids, *plan.close_order_ids])
                )
                history.evidence_gaps = list(plan.evidence_gaps)
                history.match_status = "okx_fill_lifecycle_rebuild_complete"
                raw = dict(history.raw_row or {})
                raw["_dashboard_entry_order_ids"] = list(plan.entry_order_ids)
                raw["_dashboard_close_order_ids"] = list(plan.close_order_ids)
                raw["_dashboard_linked_order_ids"] = list(history.linked_order_ids)
                if plan.allocation_document:
                    raw[LIFECYCLE_ORDER_ALLOCATIONS_KEY] = plan.allocation_document
                else:
                    raw.pop(LIFECYCLE_ORDER_ALLOCATIONS_KEY, None)
                history.raw_row = raw
            applied += 1
        await session.flush()
    publish_okx_position_history_watermark("paper")
    return {
        "applied_histories": applied,
        "orders_created": created,
        "order_facts_refreshed": refreshed,
        "skipped_histories": skipped,
        "skipped_incomplete_histories": skipped_incomplete,
        "backup": str(backup),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--rebuild-links",
        action="store_true",
        help="Replace latest lifecycle links with one-to-one OKX fill assignments.",
    )
    parser.add_argument(
        "--inspect-symbol",
        default="",
        help="Include fetched OKX fill groups for one symbol in the read-only output.",
    )
    args = parser.parse_args()
    if args.rebuild_links:
        rebuild_plans, fills, specs = await collect_rebuild_plans(args.days)
        gap_counts = Counter(gap for plan in rebuild_plans for gap in plan.evidence_gaps)
        result: dict[str, Any] = {
            "mode": "rebuild_links",
            "histories": len(rebuild_plans),
            "changed_histories": sum(1 for plan in rebuild_plans if plan.changed),
            "applicable_histories": sum(
                1 for plan in rebuild_plans if plan.changed and not plan.evidence_gaps
            ),
            "regression_skipped_histories": sum(
                1 for plan in rebuild_plans if plan.changed and plan.evidence_gaps
            ),
            "complete_histories": sum(1 for plan in rebuild_plans if not plan.evidence_gaps),
            "incomplete_histories": sum(1 for plan in rebuild_plans if plan.evidence_gaps),
            "gap_counts": dict(gap_counts),
            "changed_samples": [asdict(plan) for plan in rebuild_plans if plan.changed][:30],
            "incomplete_samples": [
                {**asdict(plan), "evidence_gaps": plan.evidence_gaps}
                for plan in rebuild_plans
                if plan.evidence_gaps
            ][:30],
            "apply": bool(args.apply),
        }
        if args.apply:
            result["apply_result"] = await apply_rebuild_plans(
                rebuild_plans,
                fills,
                specs,
            )
        print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
        return 0

    plans, fills, specs = await collect_plans(args.days)
    result: dict[str, Any] = {
        "plans": len(plans),
        "by_kind": {
            kind: sum(1 for plan in plans if plan.link_kind == kind)
            for kind in {plan.link_kind for plan in plans}
        },
        "by_symbol": {
            symbol: {
                "plans": sum(1 for plan in plans if plan.symbol == symbol),
                "entry": sum(
                    1 for plan in plans if plan.symbol == symbol and plan.link_kind == "entry"
                ),
                "close": sum(
                    1 for plan in plans if plan.symbol == symbol and plan.link_kind == "close"
                ),
                "target_contracts": sum(
                    plan.target_contracts for plan in plans if plan.symbol == symbol
                ),
                "order_ids": [
                    order_id
                    for plan in plans
                    if plan.symbol == symbol
                    for order_id in plan.order_ids
                ],
            }
            for symbol in sorted({plan.symbol for plan in plans})
        },
        "samples": [asdict(plan) for plan in plans[:30]],
        "apply": bool(args.apply),
    }
    inspect_symbol = normalize_trading_symbol(args.inspect_symbol) if args.inspect_symbol else ""
    if inspect_symbol:
        result["inspect_fills"] = [
            fill.as_dict()
            for fill in fills.values()
            if normalize_trading_symbol(fill.symbol or fill.inst_id) == inspect_symbol
        ]
    if args.apply:
        result["apply_result"] = await apply_plans(plans, fills, specs)
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
