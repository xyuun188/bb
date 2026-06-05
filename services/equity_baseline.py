from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from models.account import ExecutionEquitySnapshot
from models.market_data import Kline

BEIJING_TZ = timezone(timedelta(hours=8))


def beijing_day_bounds(now: datetime | None = None) -> tuple[str, datetime, datetime]:
    """Return Beijing date, Beijing midnight, and the same instant in UTC."""
    now_local = now.astimezone(BEIJING_TZ) if now else datetime.now(BEIJING_TZ)
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_local.date().isoformat(), start_local, start_local.astimezone(timezone.utc)


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def apply_daily_equity_baseline(
    session: AsyncSession,
    *,
    mode: str,
    model_name: str,
    allocated: float,
    positions: Iterable,
    realized_pnl: float,
    unrealized_pnl: float,
    total_pnl: float,
    now: datetime | None = None,
) -> dict:
    """Attach a Beijing-day equity PnL baseline to a PnL summary.

    今日盈亏 uses account equity movement, not "positions closed today". When
    a midnight snapshot does not exist yet, we reconstruct the start-of-day
    equity from closed positions plus the mark/close price around 00:00 for
    positions that were already open.
    """
    snapshot_date, start_local, start_utc = beijing_day_bounds(now)
    selected_mode = "live" if mode == "live" else "paper"
    rows = list(positions or [])

    baseline = await _get_or_create_baseline(
        session,
        mode=selected_mode,
        model_name=model_name,
        snapshot_date=snapshot_date,
        snapshot_at=start_local,
        start_utc=start_utc,
        allocated=allocated,
        current_realized_pnl=realized_pnl,
        current_unrealized_pnl=unrealized_pnl,
        current_total_pnl=total_pnl,
        positions=rows,
    )
    baseline_total_pnl = float(baseline.get("total_pnl") or 0.0)
    today_equity_pnl = total_pnl - baseline_total_pnl
    return {
        "today_equity_pnl": today_equity_pnl,
        "today_equity_baseline": float(baseline.get("equity") or 0.0),
        "today_equity_baseline_total_pnl": baseline_total_pnl,
        "today_equity_baseline_at": baseline.get("snapshot_at"),
        "today_equity_baseline_source": baseline.get("source") or "observed",
        "today_snapshot_date": snapshot_date,
    }


async def _get_or_create_baseline(
    session: AsyncSession,
    *,
    mode: str,
    model_name: str,
    snapshot_date: str,
    snapshot_at: datetime,
    start_utc: datetime,
    allocated: float,
    current_realized_pnl: float,
    current_unrealized_pnl: float,
    current_total_pnl: float,
    positions: list,
) -> dict:
    row = await _select_baseline(session, mode, model_name, snapshot_date)
    if row:
        return _snapshot_to_dict(row)

    estimate = await _estimate_start_of_day_pnl(session, positions, start_utc)
    baseline_total_pnl = estimate["total_pnl"]
    baseline_realized_pnl = estimate["realized_pnl"]
    baseline_unrealized_pnl = estimate["unrealized_pnl"]
    source = estimate["source"]

    snapshot = ExecutionEquitySnapshot(
        mode=mode,
        model_name=model_name,
        snapshot_date=snapshot_date,
        snapshot_at=snapshot_at,
        equity=allocated + baseline_total_pnl,
        total_pnl=baseline_total_pnl,
        realized_pnl=baseline_realized_pnl,
        unrealized_pnl=baseline_unrealized_pnl,
        source=source,
    )
    session.add(snapshot)
    try:
        await session.flush()
        return _snapshot_to_dict(snapshot)
    except IntegrityError:
        await session.rollback()
        row = await _select_baseline(session, mode, model_name, snapshot_date)
        if row:
            return _snapshot_to_dict(row)
        return {
            "snapshot_at": snapshot_at.isoformat(),
            "equity": allocated + current_total_pnl,
            "total_pnl": current_total_pnl,
            "realized_pnl": current_realized_pnl,
            "unrealized_pnl": current_unrealized_pnl,
            "source": "observed",
        }


async def _select_baseline(
    session: AsyncSession,
    mode: str,
    model_name: str,
    snapshot_date: str,
) -> ExecutionEquitySnapshot | None:
    result = await session.execute(
        select(ExecutionEquitySnapshot)
        .where(
            ExecutionEquitySnapshot.mode == mode,
            ExecutionEquitySnapshot.model_name == model_name,
            ExecutionEquitySnapshot.snapshot_date == snapshot_date,
        )
        .order_by(ExecutionEquitySnapshot.id.asc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _estimate_start_of_day_pnl(
    session: AsyncSession,
    positions: list,
    start_utc: datetime,
) -> dict:
    realized = 0.0
    unrealized = 0.0
    missing_price = False

    for pos in positions:
        closed_at = _as_utc(getattr(pos, "closed_at", None))
        created_at = _as_utc(getattr(pos, "created_at", None))
        is_open = bool(getattr(pos, "is_open", False))

        if not is_open and closed_at and closed_at <= start_utc:
            realized += float(getattr(pos, "realized_pnl", 0.0) or 0.0)
            continue

        if not created_at or created_at > start_utc:
            continue
        if closed_at and closed_at <= start_utc:
            continue

        price = await _price_at_or_before(session, str(getattr(pos, "symbol", "")), start_utc)
        if price is None or price <= 0:
            price = float(getattr(pos, "entry_price", 0.0) or 0.0)
            missing_price = True
        quantity = float(getattr(pos, "quantity", 0.0) or 0.0)
        entry_price = float(getattr(pos, "entry_price", 0.0) or 0.0)
        side = str(getattr(pos, "side", "") or "").lower()
        if side == "short":
            unrealized += (entry_price - price) * quantity
        else:
            unrealized += (price - entry_price) * quantity

    return {
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "total_pnl": realized + unrealized,
        "source": "estimated" if missing_price else "reconstructed",
    }


async def _price_at_or_before(
    session: AsyncSession,
    symbol: str,
    start_utc: datetime,
) -> float | None:
    if not symbol:
        return None
    result = await session.execute(
        select(Kline.close)
        .where(
            Kline.symbol == symbol,
            Kline.open_time <= start_utc,
        )
        .order_by(Kline.open_time.desc())
        .limit(1)
    )
    value = result.scalar_one_or_none()
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _snapshot_to_dict(snapshot: ExecutionEquitySnapshot) -> dict:
    snapshot_at = snapshot.snapshot_at
    if isinstance(snapshot_at, datetime):
        snapshot_at_value = snapshot_at.isoformat()
    else:
        snapshot_at_value = None
    return {
        "snapshot_at": snapshot_at_value,
        "equity": float(snapshot.equity or 0.0),
        "total_pnl": float(snapshot.total_pnl or 0.0),
        "realized_pnl": float(snapshot.realized_pnl or 0.0),
        "unrealized_pnl": float(snapshot.unrealized_pnl or 0.0),
        "source": snapshot.source or "observed",
    }
