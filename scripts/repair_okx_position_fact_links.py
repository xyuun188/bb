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

from core.symbols import (  # noqa: E402
    okx_inst_id_from_symbol,
    symbol_from_okx_inst_id,
    trading_symbol_variants,
)
from db.session import close_db, get_session_ctx, init_db  # noqa: E402
from models.trade import Order, Position  # noqa: E402
from services.manual_close_marker import is_manual_close_order  # noqa: E402

FILLED_STATUS = "filled"
DEFAULT_DAYS = 14
DEFAULT_MAX_POSITIONS = 500
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


@dataclass(frozen=True, slots=True)
class PositionFactLinkScanReport:
    plans: list[PositionFactLinkPlan]
    diagnostics: list[dict[str, Any]]
    lookback_days: int
    candidate_link_count: int
    repairable_count: int
    manual_review_count: int
    classification_counts: dict[str, int]
    scanned_position_count: int
    max_positions: int | None
    truncated: bool


async def collect_plans(
    *,
    days: int = DEFAULT_DAYS,
    position_ids: tuple[int, ...] = (),
    max_positions: int | None = DEFAULT_MAX_POSITIONS,
) -> list[PositionFactLinkPlan]:
    report = await collect_scan_report(
        days=days,
        position_ids=position_ids,
        max_positions=max_positions,
    )
    return report.plans


async def collect_scan_report(
    *,
    days: int = DEFAULT_DAYS,
    position_ids: tuple[int, ...] = (),
    max_positions: int | None = DEFAULT_MAX_POSITIONS,
) -> PositionFactLinkScanReport:
    since = datetime.now(UTC) - timedelta(days=max(int(days or DEFAULT_DAYS), 1))
    since_naive = since.replace(tzinfo=None)
    scan_limit = _normalize_max_positions(max_positions)
    plans: list[PositionFactLinkPlan] = []
    diagnostics: list[dict[str, Any]] = []
    async with get_session_ctx() as session:
        if position_ids:
            rows = await session.execute(
                select(Position)
                .where(Position.id.in_(position_ids))
                .order_by(Position.created_at.desc(), Position.id.desc())
            )
            positions = list(rows.scalars().all())
        else:
            positions = await _candidate_positions(session, since_naive, scan_limit)
        truncated = bool(scan_limit is not None and len(positions) > scan_limit)
        if truncated and scan_limit is not None:
            positions = positions[:scan_limit]
        for position in positions:
            if not str(getattr(position, "entry_exchange_order_id", "") or "").strip():
                order = await _matching_order(session, position, entry=True)
                if order is not None:
                    plan = _plan(position, order, "entry")
                    diagnostic = _classify_plan(plan)
                    diagnostics.append(diagnostic)
                    if diagnostic["status"] == "repairable":
                        plans.append(plan)
                else:
                    diagnostics.append(_manual_review_diagnostic(position, "entry"))
            if (
                not bool(position.is_open)
                and _safe_float(position.realized_pnl) != 0.0
                and not str(getattr(position, "close_exchange_order_id", "") or "").strip()
            ):
                order = await _matching_order(session, position, entry=False)
                if order is not None:
                    plan = _plan(position, order, "close")
                    diagnostic = _classify_plan(plan)
                    diagnostics.append(diagnostic)
                    if diagnostic["status"] == "repairable":
                        plans.append(plan)
                else:
                    diagnostics.append(_manual_review_diagnostic(position, "close"))
    classification_counts = _classification_counts(diagnostics)
    return PositionFactLinkScanReport(
        plans=plans,
        diagnostics=diagnostics,
        lookback_days=max(int(days or DEFAULT_DAYS), 1),
        candidate_link_count=len(diagnostics),
        repairable_count=int(classification_counts.get("repairable", 0)),
        manual_review_count=int(classification_counts.get("manual_review", 0)),
        classification_counts=classification_counts,
        scanned_position_count=len(positions),
        max_positions=scan_limit,
        truncated=truncated,
    )


async def _candidate_positions(
    session: Any,
    since_naive: datetime,
    scan_limit: int | None,
) -> list[Position]:
    limit = scan_limit + 1 if scan_limit is not None else None
    recent_stmt = select(Position).where(
        or_(
            Position.created_at >= since_naive,
            Position.closed_at >= since_naive,
        )
    )
    open_stmt = select(Position).where(Position.is_open.is_(True))
    if limit is not None:
        recent_stmt = recent_stmt.limit(limit)
        open_stmt = open_stmt.limit(limit)
    recent_rows = await session.execute(
        recent_stmt.order_by(Position.created_at.desc(), Position.id.desc())
    )
    open_rows = await session.execute(
        open_stmt.order_by(Position.created_at.desc(), Position.id.desc())
    )
    positions_by_id: dict[int, Position] = {}
    for position in [*recent_rows.scalars().all(), *open_rows.scalars().all()]:
        position_id = int(getattr(position, "id", 0) or 0)
        if position_id and position_id not in positions_by_id:
            positions_by_id[position_id] = position
    return list(positions_by_id.values())


def _normalize_max_positions(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = DEFAULT_MAX_POSITIONS
    if limit <= 0:
        return None
    return max(1, min(limit, 5000))


async def apply_plans(plans: list[PositionFactLinkPlan]) -> dict[str, Any]:
    if not plans:
        return {"applied": 0}
    backup_path = await backup_plans(plans)
    async with get_session_ctx() as session:
        for plan in plans:
            if _classify_plan(plan)["status"] != "repairable":
                continue
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
    position_inst_id = _position_okx_inst_id(position)
    symbol = symbol_from_okx_inst_id(position_inst_id) if position_inst_id else position.symbol
    order_side = "buy" if (entry and side == "long") or (not entry and side == "short") else "sell"
    reference_time = _ensure_aware(position.created_at if entry else position.closed_at)
    if reference_time is None:
        return None
    stmt = select(Order).where(
        Order.model_name == position.model_name,
        Order.execution_mode == position.execution_mode,
        Order.symbol.in_(trading_symbol_variants(symbol)),
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
        if not is_manual_close_order(order)
        and _quantity_covers(_safe_float(order.quantity), _safe_float(position.quantity))
        and _price_close(
            _safe_float(order.price),
            _safe_float(position.entry_price if entry else position.current_price),
        )
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda order: (
            abs((_ensure_aware(order.filled_at) - reference_time).total_seconds())
            if _ensure_aware(order.filled_at)
            else MATCH_WINDOW_SECONDS
        ),
    )[0]


def _plan(position: Position, order: Order, link_kind: str) -> PositionFactLinkPlan:
    okx_inst_id = _position_okx_inst_id(position) or okx_inst_id_from_symbol(order.symbol)
    return PositionFactLinkPlan(
        position_id=int(position.id),
        link_kind=link_kind,
        symbol=str(position.symbol or ""),
        side=str(position.side or ""),
        order_id=int(order.id),
        exchange_order_id=str(order.exchange_order_id or ""),
        okx_inst_id=okx_inst_id or None,
        old_entry_exchange_order_id=getattr(position, "entry_exchange_order_id", None),
        old_close_exchange_order_id=getattr(position, "close_exchange_order_id", None),
        old_okx_inst_id=getattr(position, "okx_inst_id", None),
        order_filled_at=order.filled_at,
        position_created_at=position.created_at,
        position_closed_at=position.closed_at,
    )


def _classify_plan(plan: PositionFactLinkPlan) -> dict[str, Any]:
    reasons: list[str] = []
    if plan.position_id <= 0:
        reasons.append("missing_position_id")
    if plan.order_id <= 0:
        reasons.append("missing_order_id")
    if plan.link_kind not in {"entry", "close"}:
        reasons.append("invalid_link_kind")
    if not str(plan.exchange_order_id or "").strip():
        reasons.append("missing_exchange_order_id")
    if not str(plan.okx_inst_id or "").strip():
        reasons.append("missing_okx_inst_id")
    reference_time = (
        plan.position_created_at if plan.link_kind == "entry" else plan.position_closed_at
    )
    order_time = plan.order_filled_at
    if reference_time is None:
        reasons.append("missing_position_reference_time")
    if order_time is None:
        reasons.append("missing_order_filled_at")
    if reference_time is not None and order_time is not None:
        delta = abs(
            (_ensure_aware(order_time) - _ensure_aware(reference_time)).total_seconds()
        )
        if delta > MATCH_WINDOW_SECONDS:
            reasons.append("order_outside_match_window")
    if plan.link_kind == "entry" and str(plan.old_entry_exchange_order_id or "").strip():
        reasons.append("entry_link_already_present")
    if plan.link_kind == "close" and str(plan.old_close_exchange_order_id or "").strip():
        reasons.append("close_link_already_present")
    status = "manual_review" if reasons else "repairable"
    return {
        "status": status,
        "reason": ";".join(reasons) if reasons else "deterministic_position_order_match",
        "position_id": plan.position_id,
        "link_kind": plan.link_kind,
        "symbol": plan.symbol,
        "side": plan.side,
        "order_id": plan.order_id,
        "exchange_order_id": plan.exchange_order_id,
        "okx_inst_id": plan.okx_inst_id,
    }


def _position_okx_inst_id(position: Position) -> str:
    return str(getattr(position, "okx_inst_id", "") or "").strip().upper()


def _manual_review_diagnostic(position: Position, link_kind: str) -> dict[str, Any]:
    return {
        "status": "manual_review",
        "reason": f"missing_matching_{link_kind}_order",
        "position_id": int(getattr(position, "id", 0) or 0),
        "link_kind": link_kind,
        "symbol": str(getattr(position, "symbol", "") or ""),
        "side": str(getattr(position, "side", "") or ""),
        "order_id": None,
        "exchange_order_id": None,
        "okx_inst_id": getattr(position, "okx_inst_id", None),
    }


def _classification_counts(diagnostics: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"repairable": 0, "manual_review": 0}
    for item in diagnostics:
        status = str(item.get("status") or "manual_review")
        counts[status] = counts.get(status, 0) + 1
    return counts


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
    parser.add_argument(
        "--max-positions",
        type=int,
        default=DEFAULT_MAX_POSITIONS,
        help="maximum candidate positions to scan; <=0 disables the cap",
    )
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    position_ids = tuple(int(item) for item in args.position_id)
    await init_db()
    try:
        report = await collect_scan_report(
            days=args.days,
            position_ids=position_ids,
            max_positions=args.max_positions,
        )
        plans = report.plans
        if args.apply and not position_ids:
            raise SystemExit("--apply requires at least one --position-id")
        result: dict[str, Any] = {
            "plans": [_json_safe(asdict(plan)) for plan in plans],
            "diagnostics": _json_safe(report.diagnostics[:100]),
            "summary": {
                "lookback_days": report.lookback_days,
                "candidate_link_count": report.candidate_link_count,
                "repairable_count": report.repairable_count,
                "manual_review_count": report.manual_review_count,
                "classification_counts": report.classification_counts,
                "scanned_position_count": report.scanned_position_count,
                "max_positions": report.max_positions,
                "truncated": report.truncated,
            },
        }
        if args.apply:
            result["apply_result"] = await apply_plans(plans)
        print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(_main())
