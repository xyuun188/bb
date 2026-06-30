#!/usr/bin/env python3
"""Repair OKX native full-close rows from authoritative fills history.

OKX's close-position endpoint can flatten a position without returning a normal
ordId. Older local rows then used an `okx_native_full_close` placeholder, fee=0,
and a snapshot price. This script matches those rows to OKX fills-history and
updates only rows with a precise time/symbol/side match.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.symbols import normalize_trading_symbol  # noqa: E402
from db.session import get_session_ctx  # noqa: E402
from executor.base_executor import OrderStatus  # noqa: E402
from executor.okx_executor import OKXExecutor  # noqa: E402
from models.account import VirtualAccount  # noqa: E402
from models.decision import AIDecision  # noqa: E402
from models.learning import TradeReflection  # noqa: E402
from models.trade import Order, Position  # noqa: E402
from services.entry_fee_provider import EntryFeeProvider  # noqa: E402

DEFAULT_WINDOW_SECONDS = 180
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FillGroup:
    order_id: str
    symbol: str
    side: str
    avg_price: float
    contracts: float
    fill_pnl: float
    fee_abs: float
    timestamp: datetime | None
    timestamp_ms: float
    rows: list[dict[str, Any]]
    inst_id: str = ""


@dataclass(frozen=True)
class RepairPlan:
    position_id: int
    close_order_id: int | None
    decision_id: int | None
    symbol: str
    side: str
    quantity: float
    old_price: float
    old_realized_pnl: float
    new_price: float
    okx_fill_pnl: float
    entry_fee: float
    close_fee: float
    new_realized_pnl: float
    old_closed_at: datetime | None
    new_closed_at: datetime
    okx_order_id: str
    delta_pnl: float


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


def _datetime_from_ms(value: Any) -> datetime | None:
    timestamp = _safe_float(value, 0.0)
    if timestamp <= 0:
        return None
    return datetime.fromtimestamp(timestamp / 1000.0, UTC)


def _iso(value: datetime | None) -> str | None:
    value = _aware(value)
    return value.isoformat() if value else None


def _close_side_for_position(position: Position) -> str:
    return "buy" if str(position.side or "").lower() == "short" else "sell"


def _is_native_placeholder_order(order: Order | None) -> bool:
    if order is None:
        return False
    exchange_order_id = str(order.exchange_order_id or "").strip().lower()
    return not exchange_order_id or "okx_native_full_close" in exchange_order_id


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


async def _fetch_okx_fill_groups(symbols: set[str]) -> dict[str, list[FillGroup]]:
    if not symbols:
        return {}
    okx = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    try:
        await okx.initialize()
        ccxt = await okx._get_ccxt()
        fetch_fills = getattr(ccxt, "privateGetTradeFillsHistory", None)
        if not callable(fetch_fills):
            return {}
        rows: list[dict[str, Any]] = []
        seen: set[tuple[Any, Any, Any, Any]] = set()
        missing_symbols: set[str] = set()
        for symbol in sorted(symbols):
            params = {
                "instType": "SWAP",
                "instId": f"{symbol.replace('/', '-')}-SWAP",
                "limit": "100",
            }
            try:
                response = await okx._with_retry(fetch_fills, params)
            except Exception as exc:
                LOGGER.warning("failed to fetch OKX fills for %s: %s", symbol, exc)
                missing_symbols.add(symbol)
                continue
            for row in response.get("data", []) if isinstance(response, dict) else []:
                if not isinstance(row, dict):
                    continue
                key = (row.get("ordId"), row.get("tradeId"), row.get("ts"), row.get("instId"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
        if missing_symbols:
            try:
                response = await okx._with_retry(
                    fetch_fills,
                    {
                        "instType": "SWAP",
                        "limit": "100",
                    },
                )
            except Exception as exc:
                LOGGER.warning("failed to fetch account-wide OKX fills: %s", exc)
            else:
                for row in response.get("data", []) if isinstance(response, dict) else []:
                    if not isinstance(row, dict):
                        continue
                    symbol = normalize_trading_symbol(row.get("instId"))
                    if symbol not in missing_symbols:
                        continue
                    key = (row.get("ordId"), row.get("tradeId"), row.get("ts"), row.get("instId"))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(row)
    finally:
        await okx.shutdown()

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        order_id = str(row.get("ordId") or "").strip()
        inst_id = str(row.get("instId") or "").strip().upper()
        symbol = normalize_trading_symbol(inst_id)
        side = str(row.get("side") or "").lower().strip()
        contracts = _safe_float(row.get("fillSz") or row.get("sz"), 0.0)
        price = _safe_float(row.get("fillPx") or row.get("price"), 0.0)
        if not order_id or not inst_id or not symbol or not side or contracts <= 0 or price <= 0:
            continue
        by_order = grouped.setdefault(symbol, {})
        group = by_order.setdefault(
            order_id,
            {
                "order_id": order_id,
                "inst_id": inst_id,
                "symbol": symbol,
                "side": side,
                "contracts": 0.0,
                "price_value": 0.0,
                "fill_pnl": 0.0,
                "fee_abs": 0.0,
                "timestamp_ms": 0.0,
                "rows": [],
            },
        )
        group["contracts"] += contracts
        group["price_value"] += price * contracts
        group["fill_pnl"] += _safe_float(row.get("fillPnl") or row.get("pnl"), 0.0)
        group["fee_abs"] += abs(_safe_float(row.get("fee"), 0.0))
        group["timestamp_ms"] = max(
            _safe_float(group.get("timestamp_ms"), 0.0),
            _safe_float(row.get("ts") or row.get("fillTime"), 0.0),
        )
        group["rows"].append(row)

    result: dict[str, list[FillGroup]] = {}
    for symbol, by_order in grouped.items():
        result[symbol] = []
        for group in by_order.values():
            contracts = _safe_float(group.get("contracts"), 0.0)
            if contracts <= 0:
                continue
            timestamp_ms = _safe_float(group.get("timestamp_ms"), 0.0)
            result[symbol].append(
                FillGroup(
                    order_id=str(group["order_id"]),
                    symbol=str(group["symbol"]),
                    side=str(group["side"]),
                    avg_price=_safe_float(group.get("price_value"), 0.0) / contracts,
                    contracts=contracts,
                    fill_pnl=_safe_float(group.get("fill_pnl"), 0.0),
                    fee_abs=_safe_float(group.get("fee_abs"), 0.0),
                    timestamp=_datetime_from_ms(timestamp_ms),
                    timestamp_ms=timestamp_ms,
                    rows=list(group.get("rows") or []),
                    inst_id=str(group.get("inst_id") or ""),
                )
            )
    return result


async def _collect_candidate_data(days: int) -> tuple[list[Position], list[Order]]:
    since = datetime.now(UTC) - timedelta(days=max(int(days), 1))
    async with get_session_ctx() as session:
        pos_result = await session.execute(
            select(Position)
            .where(
                Position.execution_mode == "paper",
                Position.is_open.is_(False),
                Position.closed_at.is_not(None),
                Position.closed_at >= since,
            )
            .order_by(Position.closed_at.desc())
        )
        positions = list(pos_result.scalars().all())
        order_result = await session.execute(
            select(Order).where(
                Order.execution_mode == "paper",
                Order.status == OrderStatus.FILLED.value,
                Order.filled_at >= since - timedelta(days=1),
            )
        )
        orders = list(order_result.scalars().all())
    return positions, orders


def _match_close_order(
    position: Position, orders: list[Order], window_seconds: int
) -> Order | None:
    symbol = normalize_trading_symbol(position.symbol)
    side = _close_side_for_position(position)
    closed_at = _aware(position.closed_at)
    candidates: list[tuple[float, Order]] = []
    for order in orders:
        if normalize_trading_symbol(order.symbol) != symbol:
            continue
        if str(order.side or "").lower() != side:
            continue
        if not _is_native_placeholder_order(order):
            continue
        order_time = _aware(order.filled_at or order.created_at)
        if not order_time or not closed_at:
            continue
        delta = abs((order_time - closed_at).total_seconds())
        if delta <= max(int(window_seconds), 1):
            candidates.append((delta, order))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _match_fill(
    position: Position, fills: list[FillGroup], window_seconds: int
) -> FillGroup | None:
    closed_at = _aware(position.closed_at)
    if closed_at is None:
        return None
    side = _close_side_for_position(position)
    candidates: list[tuple[float, float, FillGroup]] = []
    for fill in fills:
        if fill.side != side or fill.timestamp is None:
            continue
        delta = abs((fill.timestamp - closed_at).total_seconds())
        if delta > max(int(window_seconds), 1):
            continue
        candidates.append((delta, abs(fill.fill_pnl), fill))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], -item[1]))[0][2]


async def collect_repairs(
    *,
    days: int,
    window_seconds: int,
    position_id: int | None = None,
) -> list[RepairPlan]:
    positions, orders = await _collect_candidate_data(days)
    if position_id is not None:
        positions = [position for position in positions if int(position.id) == int(position_id)]
    candidate_symbols = {
        normalize_trading_symbol(position.symbol)
        for position in positions
        if _match_close_order(position, orders, window_seconds) is not None
    }
    fills_by_symbol = await _fetch_okx_fill_groups(candidate_symbols)
    plans: list[RepairPlan] = []
    entry_fee_provider = EntryFeeProvider()
    async with get_session_ctx() as session:
        for position in positions:
            close_order = _match_close_order(position, orders, window_seconds)
            if close_order is None:
                continue
            symbol = normalize_trading_symbol(position.symbol)
            fill = _match_fill(position, fills_by_symbol.get(symbol, []), window_seconds)
            if fill is None:
                continue
            existing_order = next(
                (
                    order
                    for order in orders
                    if str(order.exchange_order_id or "").strip() == fill.order_id
                ),
                None,
            )
            if existing_order is not None and int(existing_order.id) != int(close_order.id):
                continue
            entry_fee = await entry_fee_provider.entry_fee_for_position(
                session,
                position,
                _safe_float(position.quantity, 0.0),
            )
            new_realized = fill.fill_pnl - entry_fee - fill.fee_abs
            plans.append(
                RepairPlan(
                    position_id=int(position.id),
                    close_order_id=int(close_order.id) if close_order else None,
                    decision_id=(
                        int(close_order.decision_id)
                        if getattr(close_order, "decision_id", None) is not None
                        else None
                    ),
                    symbol=symbol,
                    side=str(position.side or ""),
                    quantity=_safe_float(position.quantity, 0.0),
                    old_price=_safe_float(position.current_price, 0.0),
                    old_realized_pnl=_safe_float(position.realized_pnl, 0.0),
                    new_price=fill.avg_price,
                    okx_fill_pnl=fill.fill_pnl,
                    entry_fee=entry_fee,
                    close_fee=fill.fee_abs,
                    new_realized_pnl=new_realized,
                    old_closed_at=_aware(position.closed_at),
                    new_closed_at=fill.timestamp or _aware(position.closed_at) or datetime.now(UTC),
                    okx_order_id=fill.order_id,
                    delta_pnl=new_realized - _safe_float(position.realized_pnl, 0.0),
                )
            )
    return plans


def _plan_payload(plan: RepairPlan) -> dict[str, Any]:
    return {
        "position_id": plan.position_id,
        "close_order_id": plan.close_order_id,
        "decision_id": plan.decision_id,
        "symbol": plan.symbol,
        "side": plan.side,
        "quantity": plan.quantity,
        "old_price": plan.old_price,
        "new_price": plan.new_price,
        "old_realized_pnl": plan.old_realized_pnl,
        "okx_fill_pnl": plan.okx_fill_pnl,
        "entry_fee": plan.entry_fee,
        "close_fee": plan.close_fee,
        "new_realized_pnl": plan.new_realized_pnl,
        "delta_pnl": plan.delta_pnl,
        "old_closed_at": _iso(plan.old_closed_at),
        "new_closed_at": _iso(plan.new_closed_at),
        "okx_order_id": plan.okx_order_id,
    }


async def _backup_rows(plans: list[RepairPlan], backup_dir: Path) -> Path:
    await asyncio.to_thread(backup_dir.mkdir, parents=True, exist_ok=True)
    path = backup_dir / f"okx_native_full_close_before_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    ids = [plan.position_id for plan in plans]
    order_ids = [plan.close_order_id for plan in plans if plan.close_order_id is not None]
    decision_ids = [plan.decision_id for plan in plans if plan.decision_id is not None]
    async with get_session_ctx() as session:
        positions = (
            (await session.execute(select(Position).where(Position.id.in_(ids)))).scalars().all()
            if ids
            else []
        )
        orders = (
            (await session.execute(select(Order).where(Order.id.in_(order_ids)))).scalars().all()
            if order_ids
            else []
        )
        decisions = (
            (await session.execute(select(AIDecision).where(AIDecision.id.in_(decision_ids))))
            .scalars()
            .all()
            if decision_ids
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
        accounts = (await session.execute(select(VirtualAccount))).scalars().all()
    payload = {
        "plans": [_plan_payload(plan) for plan in plans],
        "positions": [_model_payload(row) for row in positions],
        "orders": [_model_payload(row) for row in orders],
        "decisions": [_model_payload(row) for row in decisions],
        "trade_reflections": [_model_payload(row) for row in reflections],
        "virtual_accounts": [_model_payload(row) for row in accounts],
    }
    await asyncio.to_thread(
        path.write_text,
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _model_payload(row: Any) -> dict[str, Any]:
    columns = row.__table__.columns
    return {column.name: getattr(row, column.name) for column in columns}


async def apply_repairs(plans: list[RepairPlan]) -> dict[str, Any]:
    if not plans:
        return {"applied": 0}
    by_position = {plan.position_id: plan for plan in plans}
    async with get_session_ctx() as session:
        for plan in plans:
            position = await session.get(Position, plan.position_id)
            if position is None:
                continue
            old_realized = _safe_float(position.realized_pnl, 0.0)
            old_win = old_realized > 0
            new_win = plan.new_realized_pnl > 0
            position.current_price = plan.new_price
            position.realized_pnl = plan.new_realized_pnl
            position.unrealized_pnl = 0.0
            position.closed_at = plan.new_closed_at

            if plan.close_order_id is not None:
                order = await session.get(Order, plan.close_order_id)
                if order is not None:
                    order.exchange_order_id = plan.okx_order_id
                    order.price = plan.new_price
                    order.fee = plan.close_fee
                    order.filled_at = plan.new_closed_at

            if plan.decision_id is not None:
                decision = await session.get(AIDecision, plan.decision_id)
                if decision is not None:
                    _update_decision_from_plan(decision, plan)

            reflection_result = await session.execute(
                select(TradeReflection).where(TradeReflection.position_id == plan.position_id)
            )
            for reflection in reflection_result.scalars().all():
                reflection.exit_price = plan.new_price
                reflection.realized_pnl = plan.new_realized_pnl
                reflection.fee_estimate = abs(plan.entry_fee) + abs(plan.close_fee)
                reflection.closed_at = plan.new_closed_at
                reflection.outcome = _outcome(plan.new_realized_pnl)
                reflection.source = "okx_native_full_close_fill_correction"
                lessons = (
                    reflection.expert_lessons if isinstance(reflection.expert_lessons, dict) else {}
                )
                reflection.expert_lessons = {
                    **lessons,
                    "source": "okx_native_full_close_fill_correction",
                    "okx_order_id": plan.okx_order_id,
                }

            account = await _account_for_position(session, position.model_name)
            if account is not None:
                account.current_balance += plan.delta_pnl
                account.realized_pnl += plan.delta_pnl
                if old_win != new_win:
                    account.winning_trades += 1 if new_win else -1
                    account.winning_trades = max(int(account.winning_trades or 0), 0)
        await session.flush()
    return {"applied": len(by_position)}


def _update_decision_from_plan(decision: AIDecision, plan: RepairPlan) -> None:
    raw = decision.raw_llm_response if isinstance(decision.raw_llm_response, dict) else {}
    raw = dict(raw)
    execution_result = (
        raw.get("execution_result") if isinstance(raw.get("execution_result"), dict) else {}
    )
    fill_payload = {
        "source": "okx_fills_history_native_close_repair",
        "order_id": plan.okx_order_id,
        "price": plan.new_price,
        "fee": plan.close_fee,
        "pnl": plan.okx_fill_pnl,
        "entry_fee": plan.entry_fee,
        "realized_pnl": plan.new_realized_pnl,
        "timestamp": _iso(plan.new_closed_at),
    }
    raw["execution_result"] = {
        **execution_result,
        "source": "exchange_confirmed",
        "order_id": plan.okx_order_id,
        "exchange_order_id": plan.okx_order_id,
        "status": "filled",
        "quantity": plan.quantity,
        "price": plan.new_price,
        "fee": plan.close_fee,
        "pnl": plan.new_realized_pnl,
        "native_close_fill": fill_payload,
    }
    raw["native_close_fill"] = fill_payload
    raw["okx_native_full_close_fill_repaired"] = True
    decision.raw_llm_response = raw
    decision.execution_price = plan.new_price
    decision.executed_at = plan.new_closed_at
    decision.outcome = _outcome(plan.new_realized_pnl)
    notional = abs(plan.quantity * plan.new_price)
    decision.outcome_pnl_pct = plan.new_realized_pnl / notional * 100 if notional > 0 else 0.0


async def _account_for_position(session: Any, model_name: str) -> VirtualAccount | None:
    result = await session.execute(
        select(VirtualAccount).where(VirtualAccount.model_name == model_name).limit(1)
    )
    return result.scalar_one_or_none()


def _outcome(value: float) -> str:
    if value > 0:
        return "profit"
    if value < 0:
        return "loss"
    return "flat"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--position-id", type=int, default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("/data/bb/app/data/codex_backups/okx-native-full-close-fills"),
    )
    args = parser.parse_args()

    plans = await collect_repairs(
        days=args.days,
        window_seconds=args.window_seconds,
        position_id=args.position_id,
    )
    print(json.dumps({"repairs": len(plans), "apply": bool(args.apply)}, ensure_ascii=False))
    for plan in plans[:80]:
        print(json.dumps(_plan_payload(plan), ensure_ascii=False, sort_keys=True))
    if len(plans) > 80:
        print(json.dumps({"truncated": len(plans) - 80}, ensure_ascii=False))
    if args.apply and plans:
        backup_path = await _backup_rows(plans, args.backup_dir)
        result = await apply_repairs(plans)
        print(json.dumps({"backup": str(backup_path), **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
