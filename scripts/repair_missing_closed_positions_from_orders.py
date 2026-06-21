#!/usr/bin/env python3
"""Backfill missing closed positions from filled OKX order pairs."""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.symbols import normalize_trading_symbol, trading_symbol_variants  # noqa: E402
from db.session import get_session_ctx  # noqa: E402
from models.learning import TradeReflection  # noqa: E402
from models.trade import Order  # noqa: E402
from services.order_position_reconciliation import (  # noqa: E402
    apply_missing_closed_position_plan,
    plan_missing_closed_position,
)
from sqlalchemy import select  # noqa: E402


@dataclass(frozen=True, slots=True)
class ReconciliationFilters:
    """Bounded filters for safe historical position reconciliation."""

    days: int = 14
    symbols: tuple[str, ...] = ()
    close_order_ids: tuple[int, ...] = ()
    close_exchange_order_ids: tuple[str, ...] = ()
    min_realized_pnl: float | None = None
    max_realized_pnl: float | None = None


async def collect_missing_closed_position_plans(
    *,
    days: int | None = None,
    filters: ReconciliationFilters | None = None,
) -> list[Any]:
    active_filters = filters or ReconciliationFilters(days=int(days or 14))
    since = datetime.now(UTC) - timedelta(days=max(int(active_filters.days), 1))
    plans: list[Any] = []
    async with get_session_ctx() as session:
        stmt = select(Order).where(
            Order.status == "filled",
            Order.exchange_order_id.is_not(None),
            Order.exchange_order_id != "",
            Order.filled_at >= since,
        )
        if active_filters.close_order_ids:
            stmt = stmt.where(Order.id.in_(active_filters.close_order_ids))
        if active_filters.close_exchange_order_ids:
            stmt = stmt.where(Order.exchange_order_id.in_(active_filters.close_exchange_order_ids))
        if active_filters.symbols:
            variants = _symbol_variants(active_filters.symbols)
            stmt = stmt.where(Order.symbol.in_(variants))
        result = await session.execute(stmt.order_by(Order.filled_at.asc(), Order.created_at.asc()))
        orders = list(result.scalars().all())
        for order in orders:
            plan = await plan_missing_closed_position(session, order)
            if plan is not None and _matches_filters(plan, active_filters):
                plans.append(plan)
    return plans


async def apply_plans(
    plans: list[Any],
    *,
    filters: ReconciliationFilters | None = None,
) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    if not plans:
        return applied
    active_filters = filters or ReconciliationFilters()
    async with get_session_ctx() as session:
        for original_plan in plans:
            close_order = await session.get(Order, int(original_plan.close_order_id))
            if close_order is None:
                continue
            plan = await plan_missing_closed_position(session, close_order)
            if plan is None or not _matches_filters(plan, active_filters):
                continue
            position = await apply_missing_closed_position_plan(session, plan)
            session.add(
                TradeReflection(
                    position_id=position.id,
                    model_name=plan.model_name,
                    execution_mode=plan.execution_mode,
                    symbol=plan.symbol,
                    side=plan.side,
                    entry_price=plan.entry_price,
                    exit_price=plan.exit_price,
                    quantity=plan.quantity,
                    realized_pnl=plan.realized_pnl,
                    fee_estimate=abs(plan.entry_fee_allocated) + abs(plan.close_fee_allocated),
                    hold_minutes=max(
                        (plan.closed_at - plan.created_at).total_seconds() / 60.0,
                        0.0,
                    ),
                    closed_at=plan.closed_at,
                    outcome=(
                        "profit"
                        if plan.realized_pnl > 0
                        else "loss" if plan.realized_pnl < 0 else "flat"
                    ),
                    mistake_summary="OKX 成交订单已存在，本地持仓历史曾缺失；已自动补齐用于对账和训练。",
                    improvement_summary="执行链需保持订单与持仓原子对账，避免漏账影响仓位判断。",
                    expert_lessons={"source": "missing_closed_position_repair"},
                    source="okx_order_pair_repair",
                )
            )
            applied.append(_report_plan(plan, position.id))
        await session.flush()
    return applied


def _symbol_variants(symbols: tuple[str, ...]) -> list[str]:
    variants: list[str] = []
    for symbol in symbols:
        normalized = normalize_trading_symbol(symbol)
        for variant in trading_symbol_variants(normalized):
            if variant not in variants:
                variants.append(variant)
    return variants


def _matches_filters(plan: Any, filters: ReconciliationFilters) -> bool:
    if filters.symbols:
        allowed_symbols = {normalize_trading_symbol(symbol) for symbol in filters.symbols}
        if normalize_trading_symbol(plan.symbol) not in allowed_symbols:
            return False
    if filters.close_order_ids and int(plan.close_order_id) not in filters.close_order_ids:
        return False
    if filters.close_exchange_order_ids:
        exchange_id = str(plan.close_exchange_order_id or "")
        if exchange_id not in filters.close_exchange_order_ids:
            return False
    if filters.min_realized_pnl is not None and plan.realized_pnl < filters.min_realized_pnl:
        return False
    if filters.max_realized_pnl is not None and plan.realized_pnl > filters.max_realized_pnl:
        return False
    return True


def _report_plan(plan: Any, position_id: int | None = None) -> dict[str, Any]:
    payload = asdict(plan)
    for key in ("created_at", "closed_at"):
        if payload.get(key) is not None:
            payload[key] = payload[key].isoformat()
    if position_id is not None:
        payload["position_id"] = position_id
    payload["realized_pnl"] = round(float(payload["realized_pnl"]), 8)
    payload["gross_pnl"] = round(float(payload["gross_pnl"]), 8)
    payload["entry_fee_allocated"] = round(float(payload["entry_fee_allocated"]), 8)
    payload["close_fee_allocated"] = round(float(payload["close_fee_allocated"]), 8)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--symbol", action="append", default=[], help="limit to one symbol")
    parser.add_argument(
        "--close-order-id",
        action="append",
        type=int,
        default=[],
        help="limit to local close order id",
    )
    parser.add_argument(
        "--close-exchange-order-id",
        action="append",
        default=[],
        help="limit to OKX close order id",
    )
    parser.add_argument("--min-realized-pnl", type=float, default=None)
    parser.add_argument("--max-realized-pnl", type=float, default=None)
    parser.add_argument("--apply", action="store_true", help="write missing positions")
    parser.add_argument(
        "--allow-bulk-apply",
        action="store_true",
        help="allow --apply without symbol/order filters after manual audit",
    )
    args = parser.parse_args()
    has_apply_filter = bool(args.symbol or args.close_order_id or args.close_exchange_order_id)
    if args.apply and not has_apply_filter and not args.allow_bulk_apply:
        parser.error(
            "--apply requires --symbol, --close-order-id or --close-exchange-order-id; "
            "use --allow-bulk-apply only after auditing dry-run output"
        )
    return args


async def main() -> int:
    args = _parse_args()
    filters = ReconciliationFilters(
        days=args.days,
        symbols=tuple(args.symbol or ()),
        close_order_ids=tuple(args.close_order_id or ()),
        close_exchange_order_ids=tuple(str(item) for item in (args.close_exchange_order_id or ())),
        min_realized_pnl=args.min_realized_pnl,
        max_realized_pnl=args.max_realized_pnl,
    )

    plans = await collect_missing_closed_position_plans(filters=filters)
    print(
        {
            "missing_closed_positions": len(plans),
            "apply": bool(args.apply),
            "filters": asdict(filters),
        }
    )
    for plan in plans[:50]:
        print(_report_plan(plan))
    if len(plans) > 50:
        print({"truncated": len(plans) - 50})
    if args.apply:
        applied = await apply_plans(plans, filters=filters)
        print({"applied": len(applied)})
        for item in applied[:50]:
            print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
