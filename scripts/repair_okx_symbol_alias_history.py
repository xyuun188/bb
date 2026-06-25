#!/usr/bin/env python3
"""Repair order/position symbols when CCXT aliases differ from OKX instId."""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.symbols import (  # noqa: E402
    normalize_trading_symbol,
    symbol_from_okx_payload,
    symbol_query_variants,
)
from db.session import get_session_ctx  # noqa: E402
from models.decision import AIDecision  # noqa: E402
from models.trade import Order, Position  # noqa: E402

DEFAULT_WINDOW_SECONDS = 300


@dataclass(slots=True)
class SymbolRepair:
    old_symbol: str
    new_symbol: str
    order_ids: list[int]
    decision_ids: list[int]
    position_ids: list[int]
    reason: str


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _raw_execution_payload(decision: AIDecision | None) -> dict[str, Any]:
    raw = _safe_dict(getattr(decision, "raw_llm_response", None))
    execution_result = _safe_dict(raw.get("execution_result"))
    raw_response = _safe_dict(execution_result.get("raw_response"))
    if raw_response:
        return raw_response
    close_fill = _safe_dict(raw.get("close_fill"))
    if close_fill.get("instId") or close_fill.get("okx_symbol"):
        return close_fill
    return raw


def _entry_side_for_action(action: str | None) -> str | None:
    if action == "long":
        return "buy"
    if action == "short":
        return "sell"
    return None


def _position_side_for_action(action: str | None) -> str | None:
    if action in {"long", "close_long"}:
        return "long"
    if action in {"short", "close_short"}:
        return "short"
    return None


def _close_side_for_position(side: str | None) -> str:
    return "buy" if str(side or "").lower() == "short" else "sell"


def _entry_side_for_position(side: str | None) -> str:
    return "buy" if str(side or "").lower() == "long" else "sell"


async def _related_positions(
    session: Any,
    *,
    order: Order,
    action: str,
    filled_at: datetime,
    window: timedelta,
) -> list[Position]:
    position_side = _position_side_for_action(action)
    entry_side = _entry_side_for_action(action)
    stmt = select(Position).where(
        Position.model_name == order.model_name,
        Position.execution_mode == order.execution_mode,
        Position.symbol.in_(symbol_query_variants({order.symbol})),
    )
    if position_side:
        stmt = stmt.where(Position.side == position_side)
    rows = list((await session.execute(stmt)).scalars().all())
    positions: list[Position] = []
    for position in rows:
        created_at = _aware(position.created_at)
        closed_at = _aware(position.closed_at)
        if entry_side and order.side == entry_side and created_at:
            if abs((created_at - filled_at).total_seconds()) <= window.total_seconds():
                positions.append(position)
        elif closed_at:
            if abs((closed_at - filled_at).total_seconds()) <= window.total_seconds():
                positions.append(position)
    return positions


async def _related_orders_for_positions(
    session: Any,
    *,
    seed_order: Order,
    positions: list[Position],
    window: timedelta,
) -> tuple[set[int], set[int]]:
    order_ids = {int(seed_order.id)}
    decision_ids = {int(seed_order.decision_id)} if seed_order.decision_id else set()
    for position in positions:
        created_at = _aware(position.created_at)
        if created_at is None:
            continue
        closed_at = _aware(position.closed_at)
        stmt = select(Order).where(
            Order.model_name == seed_order.model_name,
            Order.execution_mode == seed_order.execution_mode,
            Order.symbol.in_(symbol_query_variants({seed_order.symbol})),
            Order.status == "filled",
            Order.filled_at >= created_at - window,
            Order.filled_at <= (closed_at or created_at) + window,
        )
        rows = list((await session.execute(stmt)).scalars().all())
        valid_sides = {
            _entry_side_for_position(position.side),
            _close_side_for_position(position.side),
        }
        for order in rows:
            if order.side not in valid_sides:
                continue
            order_ids.add(int(order.id))
            if order.decision_id:
                decision_ids.add(int(order.decision_id))
    return order_ids, decision_ids


async def collect_repairs(*, hours: int, window_seconds: int) -> list[SymbolRepair]:
    since = datetime.now(UTC) - timedelta(hours=max(int(hours), 1))
    window = timedelta(seconds=max(int(window_seconds), 30))
    repairs_by_key: dict[tuple[str, str], SymbolRepair] = {}
    async with get_session_ctx() as session:
        rows = await session.execute(
            select(Order)
            .where(
                Order.created_at >= since,
                Order.decision_id.is_not(None),
                Order.status == "filled",
            )
            .order_by(Order.created_at.desc())
        )
        orders = list(rows.scalars().all())
        decision_ids = {order.decision_id for order in orders if order.decision_id}
        decisions: dict[int, AIDecision] = {}
        if decision_ids:
            decision_rows = await session.execute(
                select(AIDecision).where(AIDecision.id.in_(decision_ids))
            )
            decisions = {int(row.id): row for row in decision_rows.scalars().all()}

        for order in orders:
            decision = decisions.get(int(order.decision_id or 0))
            payload = _raw_execution_payload(decision)
            new_symbol = symbol_from_okx_payload(payload, fallback=order.symbol)
            old_symbol = normalize_trading_symbol(order.symbol)
            if not new_symbol or new_symbol == old_symbol:
                continue
            filled_at = _aware(order.filled_at or order.created_at)
            if filled_at is None:
                continue
            action = str(getattr(decision, "action", "") or "")
            positions = await _related_positions(
                session,
                order=order,
                action=action,
                filled_at=filled_at,
                window=window,
            )
            order_ids, decision_ids_for_repair = await _related_orders_for_positions(
                session,
                seed_order=order,
                positions=positions,
                window=window,
            )
            key = (old_symbol, new_symbol)
            existing = repairs_by_key.get(key)
            position_ids = {int(position.id) for position in positions}
            if existing is None:
                repairs_by_key[key] = SymbolRepair(
                    old_symbol=old_symbol,
                    new_symbol=new_symbol,
                    order_ids=sorted(order_ids),
                    decision_ids=sorted(decision_ids_for_repair),
                    position_ids=sorted(position_ids),
                    reason="okx_inst_id_alias_mismatch",
                )
            else:
                existing.order_ids = sorted(set(existing.order_ids) | order_ids)
                existing.decision_ids = sorted(set(existing.decision_ids) | decision_ids_for_repair)
                existing.position_ids = sorted(set(existing.position_ids) | position_ids)
    return list(repairs_by_key.values())


async def apply_repairs(repairs: list[SymbolRepair]) -> None:
    async with get_session_ctx() as session:
        for repair in repairs:
            for order_id in repair.order_ids:
                order = await session.get(Order, order_id)
                if order is not None:
                    order.symbol = repair.new_symbol
            for decision_id in repair.decision_ids:
                decision = await session.get(AIDecision, decision_id)
                if decision is not None:
                    decision.symbol = repair.new_symbol
            for position_id in repair.position_ids:
                position = await session.get(Position, position_id)
                if position is not None:
                    position.symbol = repair.new_symbol
        await session.flush()


def render_repairs(repairs: list[SymbolRepair]) -> str:
    if not repairs:
        return "No symbol alias repairs found."
    lines = [f"Found {len(repairs)} symbol alias repairs:"]
    for item in repairs:
        lines.append(
            f"- {item.old_symbol} -> {item.new_symbol} "
            f"orders={item.order_ids} decisions={item.decision_ids} "
            f"positions={item.position_ids} reason={item.reason}"
        )
    return "\n".join(lines)


async def async_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    repairs = await collect_repairs(hours=args.hours, window_seconds=args.window_seconds)
    print(render_repairs(repairs))
    if args.apply and repairs:
        await apply_repairs(repairs)
        print(f"Applied {len(repairs)} symbol alias repairs.")
    elif repairs:
        print("Dry run only. Re-run with --apply to update rows.")
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
