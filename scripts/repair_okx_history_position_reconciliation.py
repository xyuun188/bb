#!/usr/bin/env python3
"""Repair historical closed-position rows from real OKX close orders.

This script fixes legacy rows created when an OKX close order used a CCXT swap
symbol such as `MET/USDT:USDT` while local positions were stored as `MET/USDT`.
Before the global symbol-normalization fix, those rows could be closed later by
an estimated exchange reconciliation pass, producing a wrong close time, close
price, and realized PnL in the dashboard history.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.symbols import normalize_trading_symbol  # noqa: E402
from db.session import get_session_ctx  # noqa: E402
from models.trade import Order, Position  # noqa: E402
from sqlalchemy import select  # noqa: E402

DEFAULT_WINDOW_SECONDS = 240
MIN_QUANTITY_COVERAGE = 0.80
PNL_TOLERANCE = 0.000001
PRICE_TOLERANCE_RATIO = 0.0005
TIME_REPAIR_THRESHOLD_SECONDS = 15


@dataclass(frozen=True)
class RepairItem:
    position: Position
    order: Order
    old_closed_at: datetime | None
    old_price: float | None
    old_realized_pnl: float | None
    new_closed_at: datetime
    new_price: float
    new_realized_pnl: float
    close_fee_allocated: float
    inferred_entry_fee: float


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _close_side_for_position(position: Position) -> str:
    return "buy" if str(position.side or "").lower() == "short" else "sell"


def _position_key(position: Position) -> tuple[str, str, str]:
    return (
        str(position.execution_mode or ""),
        normalize_trading_symbol(position.symbol),
        _close_side_for_position(position),
    )


def _order_key(order: Order) -> tuple[str, str, str]:
    return (
        str(order.execution_mode or ""),
        normalize_trading_symbol(order.symbol),
        str(order.side or "").lower(),
    )


def _gross_pnl(position: Position, exit_price: float) -> float:
    quantity = _safe_float(position.quantity)
    entry_price = _safe_float(position.entry_price)
    if str(position.side or "").lower() == "short":
        return (entry_price - exit_price) * quantity
    return (exit_price - entry_price) * quantity


def _price_changed(old_price: float | None, new_price: float) -> bool:
    previous = _safe_float(old_price)
    tolerance = max(abs(new_price) * PRICE_TOLERANCE_RATIO, 0.0000001)
    return abs(previous - new_price) > tolerance


def _build_repair_item(position: Position, order: Order) -> RepairItem | None:
    order_time = _aware(order.filled_at or order.created_at)
    if order_time is None:
        return None
    order_price = _safe_float(order.price)
    if order_price <= 0:
        return None

    position_quantity = _safe_float(position.quantity)
    order_quantity = _safe_float(order.quantity)
    order_fee = _safe_float(order.fee)
    close_fee = order_fee * position_quantity / order_quantity if order_quantity > 0 else 0.0

    old_gross = _gross_pnl(position, _safe_float(position.current_price, order_price))
    old_realized = _safe_float(position.realized_pnl)
    inferred_entry_fee = max(old_gross - old_realized, 0.0)
    new_realized = _gross_pnl(position, order_price) - inferred_entry_fee - close_fee

    old_time = _aware(position.closed_at)
    time_delta = abs((old_time - order_time).total_seconds()) if old_time else float("inf")
    price_changed = _price_changed(position.current_price, order_price)
    time_changed = time_delta > TIME_REPAIR_THRESHOLD_SECONDS
    needs_update = price_changed or time_changed
    if not needs_update:
        return None

    return RepairItem(
        position=position,
        order=order,
        old_closed_at=position.closed_at,
        old_price=position.current_price,
        old_realized_pnl=position.realized_pnl,
        new_closed_at=order_time,
        new_price=order_price,
        new_realized_pnl=new_realized,
        close_fee_allocated=close_fee,
        inferred_entry_fee=inferred_entry_fee,
    )


def _select_positions_for_order(
    *,
    order: Order,
    candidates: list[Position],
    already_selected: set[int],
    window: timedelta,
) -> list[Position]:
    order_time = _aware(order.filled_at or order.created_at)
    if order_time is None:
        return []
    order_quantity = _safe_float(order.quantity)
    scored: list[tuple[float, float, float, Position]] = []
    for position in candidates:
        if position.id in already_selected:
            continue
        closed_at = _aware(position.closed_at)
        if closed_at is None:
            continue
        delta = abs((closed_at - order_time).total_seconds())
        if delta > window.total_seconds():
            continue
        position_quantity = _safe_float(position.quantity)
        if order_quantity > 0 and position_quantity > order_quantity * 1.05:
            continue
        price_delta = abs(_safe_float(position.current_price) - _safe_float(order.price))
        qty_delta = abs(order_quantity - position_quantity)
        scored.append((delta, price_delta, qty_delta, position))

    if not scored:
        return []

    selected: list[Position] = []
    selected_quantity = 0.0
    for _delta, _price_delta, _qty_delta, position in sorted(
        scored, key=lambda item: (item[0], item[1], item[2])
    ):
        selected.append(position)
        selected_quantity += _safe_float(position.quantity)
        if order_quantity <= 0 or selected_quantity >= order_quantity * 0.98:
            break

    if order_quantity > 0 and selected_quantity < order_quantity * MIN_QUANTITY_COVERAGE:
        return []
    return selected


async def collect_repairs(*, days: int, window_seconds: int) -> list[RepairItem]:
    since = datetime.now(UTC) - timedelta(days=max(int(days), 1))
    window = timedelta(seconds=max(int(window_seconds), 30))
    async with get_session_ctx() as session:
        position_result = await session.execute(
            select(Position).where(
                Position.is_open.is_(False),
                Position.closed_at.is_not(None),
                Position.closed_at >= since,
            )
        )
        positions = list(position_result.scalars().all())
        order_result = await session.execute(
            select(Order).where(
                Order.status == "filled",
                Order.exchange_order_id.is_not(None),
                Order.exchange_order_id != "",
                Order.filled_at >= since - window,
            )
        )
        orders = list(order_result.scalars().all())

    positions_by_key: dict[tuple[str, str, str], list[Position]] = {}
    for position in positions:
        positions_by_key.setdefault(_position_key(position), []).append(position)

    repairs: list[RepairItem] = []
    selected_ids: set[int] = set()
    for order in sorted(orders, key=lambda item: item.filled_at or item.created_at):
        candidates = positions_by_key.get(_order_key(order), [])
        selected = _select_positions_for_order(
            order=order,
            candidates=candidates,
            already_selected=selected_ids,
            window=window,
        )
        if not selected:
            continue
        for position in selected:
            item = _build_repair_item(position, order)
            if item is None:
                continue
            repairs.append(item)
            selected_ids.add(position.id)
    return repairs


async def apply_repairs(repairs: list[RepairItem]) -> None:
    if not repairs:
        return
    async with get_session_ctx() as session:
        for item in repairs:
            position = await session.get(Position, item.position.id)
            if position is None:
                continue
            position.closed_at = item.new_closed_at
            position.current_price = item.new_price
            position.realized_pnl = item.new_realized_pnl
            position.unrealized_pnl = 0.0
        await session.flush()


def _report_item(item: RepairItem) -> dict[str, Any]:
    return {
        "position_id": item.position.id,
        "symbol": normalize_trading_symbol(item.position.symbol),
        "side": item.position.side,
        "quantity": item.position.quantity,
        "order_id": item.order.id,
        "exchange_order_id": item.order.exchange_order_id,
        "old_closed_at": item.old_closed_at.isoformat() if item.old_closed_at else None,
        "new_closed_at": item.new_closed_at.isoformat(),
        "old_price": item.old_price,
        "new_price": item.new_price,
        "old_realized_pnl": item.old_realized_pnl,
        "new_realized_pnl": round(item.new_realized_pnl, 8),
        "close_fee_allocated": round(item.close_fee_allocated, 8),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--apply", action="store_true", help="write repairs to the database")
    args = parser.parse_args()

    repairs = await collect_repairs(days=args.days, window_seconds=args.window_seconds)
    print({"repairs": len(repairs), "apply": bool(args.apply)})
    for item in repairs[:50]:
        print(_report_item(item))
    if len(repairs) > 50:
        print({"truncated": len(repairs) - 50})
    if args.apply:
        await apply_repairs(repairs)
        print({"applied": len(repairs)})
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
