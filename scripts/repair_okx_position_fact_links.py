"""Backfill OKX order links on local position rows.

The repair is intentionally narrow: it only fills missing position link fields
from already-filled local OKX orders.  It does not recalculate PnL, prices, or
quantities.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import or_, select  # noqa: E402

from core.symbols import okx_inst_id_from_symbol, trading_symbol_variants  # noqa: E402
from db.session import close_db, get_session_ctx, init_db  # noqa: E402
from models.trade import Order, Position  # noqa: E402

FILLED_STATUS = "filled"
DEFAULT_DAYS = 14
MATCH_WINDOW_SECONDS = 10 * 60
PRICE_TOLERANCE_RATIO = 0.002
QUANTITY_TOLERANCE_RATIO = 0.02
BACKUP_DIR = Path("/data/bb/app/data/codex_backups/okx-position-fact-links")


@dataclass(frozen=True, slots=True)
class PositionFactLinkPlan:
    position_id: int
    link_kind: str
    symbol: str
    side: str
    order_id: int
    exchange_order_id: str
    okx_inst_id: str | None
    old_entry_exchange_order_id: str | None
    old_close_exchange_order_id: str | None
    old_okx_inst_id: str | None
    order_filled_at: datetime | None
    position_created_at: datetime | None
    position_closed_at: datetime | None


async def collect_plans(
    *,
    days: int = DEFAULT_DAYS,
    position_ids: tuple[int, ...] = (),
) -> list[PositionFactLinkPlan]:
    since = datetime.now(UTC) - timedelta(days=max(int(days or DEFAULT_DAYS), 1))
    since_naive = since.replace(tzinfo=None)
    plans: list[PositionFactLinkPlan] = []
    async with get_session_ctx() as session:
        stmt = select(Position).where(
            or_(
                Position.is_open.is_(True),
                Position.created_at >= since_naive,
                Position.closed_at >= since_naive,
            )
        )
        if position_ids:
            stmt = stmt.where(Position.id.in_(position_ids))
        rows = await session.execute(stmt.order_by(Position.created_at.desc()))
        for position in rows.scalars().all():
            if not str(getattr(position, "entry_exchange_order_id", "") or "").strip():
                order = await _matching_order(session, position, entry=True)
                if order is not None:
                    plans.append(_plan(position, order, "entry"))
            if (
                not bool(position.is_open)
                and _safe_float(position.realized_pnl) != 0.0
                and not str(getattr(position, "close_exchange_order_id", "") or "").strip()
            ):
                order = await _matching_order(session, position, entry=False)
                if order is not None:
                    plans.append(_plan(position, order, "close"))
    return plans


async def apply_plans(plans: list[PositionFactLinkPlan]) -> dict[str, Any]:
    if not plans:
        return {"applied": 0}
    backup_path = await backup_plans(plans)
    async with get_session_ctx() as session:
        for plan in plans:
            position = await session.get(Position, plan.position_id)
            if position is None:
                continue
            if plan.okx_inst_id and not str(getattr(position, "okx_inst_id", "") or "").strip():
                position.okx_inst_id = plan.okx_inst_id
            if plan.link_kind == "entry":
                position.entry_exchange_order_id = plan.exchange_order_id
            elif plan.link_kind == "close":
                position.close_exchange_order_id = plan.exchange_order_id
        await session.flush()
    return {"applied": len(plans), "backup_path": str(backup_path)}


async def backup_plans(plans: list[PositionFactLinkPlan]) -> Path:
    ids = sorted({plan.position_id for plan in plans})
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = BACKUP_DIR / f"position_fact_links_before_{timestamp}.json"
    async with get_session_ctx() as session:
        rows = (
            (await session.execute(select(Position).where(Position.id.in_(ids)))).scalars().all()
            if ids
            else []
        )
        payload = {
            "plans": [_json_safe(asdict(plan)) for plan in plans],
            "positions": [_model_payload(row) for row in rows],
        }
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(
        path.write_text,
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


async def _matching_order(session: Any, position: Position, *, entry: bool) -> Order | None:
    side = str(position.side or "").lower()
    if side not in {"long", "short"}:
        return None
    order_side = "buy" if (entry and side == "long") or (not entry and side == "short") else "sell"
    reference_time = _ensure_aware(position.created_at if entry else position.closed_at)
    if reference_time is None:
        return None
    stmt = select(Order).where(
        Order.model_name == position.model_name,
        Order.execution_mode == position.execution_mode,
        Order.symbol.in_(trading_symbol_variants(position.symbol)),
        Order.side == order_side,
        Order.status == FILLED_STATUS,
        Order.exchange_order_id.is_not(None),
        Order.exchange_order_id != "",
    )
    window_start = reference_time - timedelta(seconds=MATCH_WINDOW_SECONDS)
    window_end = reference_time + timedelta(seconds=MATCH_WINDOW_SECONDS)
    stmt = stmt.where(Order.filled_at >= window_start, Order.filled_at <= window_end)
    rows = await session.execute(stmt.order_by(Order.filled_at.desc(), Order.created_at.desc()))
    candidates = [
        order
        for order in rows.scalars().all()
        if _quantity_covers(_safe_float(order.quantity), _safe_float(position.quantity))
        and _price_close(
            _safe_float(order.price),
            _safe_float(position.entry_price if entry else position.current_price),
        )
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda order: abs((_ensure_aware(order.filled_at) - reference_time).total_seconds())
        if _ensure_aware(order.filled_at)
        else MATCH_WINDOW_SECONDS,
    )[0]


def _plan(position: Position, order: Order, link_kind: str) -> PositionFactLinkPlan:
    return PositionFactLinkPlan(
        position_id=int(position.id),
        link_kind=link_kind,
        symbol=str(position.symbol or ""),
        side=str(position.side or ""),
        order_id=int(order.id),
        exchange_order_id=str(order.exchange_order_id or ""),
        okx_inst_id=okx_inst_id_from_symbol(position.symbol) or None,
        old_entry_exchange_order_id=getattr(position, "entry_exchange_order_id", None),
        old_close_exchange_order_id=getattr(position, "close_exchange_order_id", None),
        old_okx_inst_id=getattr(position, "okx_inst_id", None),
        order_filled_at=order.filled_at,
        position_created_at=position.created_at,
        position_closed_at=position.closed_at,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _quantity_covers(order_quantity: float, position_quantity: float) -> bool:
    if order_quantity <= 0 or position_quantity <= 0:
        return False
    tolerance = max(abs(order_quantity), abs(position_quantity), 1e-9) * QUANTITY_TOLERANCE_RATIO
    return order_quantity + tolerance >= position_quantity


def _price_close(left: float, right: float) -> bool:
    tolerance = max(abs(left), abs(right), 1e-9) * PRICE_TOLERANCE_RATIO
    return abs(left - right) <= tolerance


def _ensure_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


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
    parser.add_argument("--position-id", action="append", type=int, default=[])
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    position_ids = tuple(int(item) for item in args.position_id)
    await init_db()
    try:
        plans = await collect_plans(days=args.days, position_ids=position_ids)
        if args.apply and not position_ids:
            raise SystemExit("--apply requires at least one --position-id")
        result: dict[str, Any] = {"plans": [_json_safe(asdict(plan)) for plan in plans]}
        if args.apply:
            result["apply_result"] = await apply_plans(plans)
        print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(_main())
