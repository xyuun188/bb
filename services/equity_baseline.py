from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from models.account import ExecutionEquitySnapshot
from services.phase3_boundary import PHASE3_FIRST_CLEAN_DAY

BEIJING_TZ = timezone(timedelta(hours=8))
OKX_BASELINE_MAX_DRIFT_RATIO = 0.10
OKX_BASELINE_MAX_DRIFT_USDT = 250.0


def beijing_day_bounds(now: datetime | None = None) -> tuple[str, datetime, datetime]:
    """Return Beijing date, Beijing midnight, and the same instant in UTC."""
    now_local = now.astimezone(BEIJING_TZ) if now else datetime.now(BEIJING_TZ)
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_local.date().isoformat(), start_local, start_local.astimezone(UTC)


async def apply_daily_equity_baseline(
    session: AsyncSession,
    *,
    mode: str,
    model_name: str,
    allocated: float,
    positions: object,
    realized_pnl: float,
    unrealized_pnl: float,
    total_pnl: float,
    current_equity: float | None = None,
    now: datetime | None = None,
) -> dict:
    """Attach a Beijing-day OKX equity baseline to a PnL summary.

    Account PnL is exchange equity movement only. If OKX equity is unavailable,
    return an unavailable baseline instead of estimating from local positions,
    fixed allocations, or historical virtual balances.
    """
    snapshot_date, start_local, _start_utc = beijing_day_bounds(now)
    selected_mode = "live" if mode == "live" else "paper"

    baseline = await _get_or_create_baseline(
        session,
        mode=selected_mode,
        model_name=model_name,
        snapshot_date=snapshot_date,
        snapshot_at=start_local,
        current_equity=current_equity,
    )
    baseline_equity = _safe_float(baseline.get("equity"), None)
    okx_equity = _safe_float(current_equity, None)
    if baseline_equity is None or baseline_equity <= 0 or okx_equity is None or okx_equity <= 0:
        return _unavailable_baseline(snapshot_date)
    today_equity_pnl = okx_equity - baseline_equity
    return {
        "today_equity_pnl": today_equity_pnl,
        "today_equity_baseline": baseline_equity,
        "today_equity_baseline_total_pnl": None,
        "today_equity_baseline_at": baseline.get("snapshot_at"),
        "today_equity_baseline_source": baseline.get("source") or "observed",
        "today_snapshot_date": snapshot_date,
    }


async def phase3_equity_change_from_snapshots(
    session: AsyncSession,
    *,
    mode: str,
    model_name: str,
    current_equity: float | None = None,
) -> dict:
    """Return Phase 3 account-equity movement from OKX snapshots only."""

    selected_mode = "live" if mode == "live" else "paper"
    row = await _select_first_phase3_okx_snapshot(session, selected_mode, model_name)
    baseline_equity = _safe_float(getattr(row, "equity", None), None) if row else None
    okx_equity = _safe_float(current_equity, None)
    if baseline_equity is None or baseline_equity <= 0 or okx_equity is None or okx_equity <= 0:
        return {
            "phase3_equity_pnl": None,
            "phase3_equity_pnl_pct": None,
            "phase3_equity_baseline": baseline_equity,
            "phase3_equity_baseline_at": _snapshot_at_iso(row) if row else None,
            "phase3_equity_baseline_source": "okx_snapshot" if row else "okx_unavailable",
            "phase3_equity_start_date": PHASE3_FIRST_CLEAN_DAY,
        }
    pnl = okx_equity - baseline_equity
    return {
        "phase3_equity_pnl": pnl,
        "phase3_equity_pnl_pct": pnl / baseline_equity,
        "phase3_equity_baseline": baseline_equity,
        "phase3_equity_baseline_at": _snapshot_at_iso(row),
        "phase3_equity_baseline_source": "okx_snapshot",
        "phase3_equity_start_date": PHASE3_FIRST_CLEAN_DAY,
    }


async def _get_or_create_baseline(
    session: AsyncSession,
    *,
    mode: str,
    model_name: str,
    snapshot_date: str,
    snapshot_at: datetime,
    current_equity: float | None,
) -> dict:
    row = await _select_baseline(session, mode, model_name, snapshot_date)
    if row:
        okx_equity = _safe_float(current_equity, None)
        if _baseline_must_be_rebuilt(row, okx_equity, snapshot_date):
            if okx_equity is None or okx_equity <= 0:
                return _unavailable_baseline(snapshot_date)
            row.snapshot_at = snapshot_at
            row.equity = okx_equity
            row.total_pnl = 0.0
            row.realized_pnl = 0.0
            row.unrealized_pnl = 0.0
            row.source = "okx_snapshot"
            await session.flush()
        return _snapshot_to_dict(row)

    okx_equity = _safe_float(current_equity, None)
    if okx_equity is None or okx_equity <= 0:
        return _unavailable_baseline(snapshot_date)
    baseline_total_pnl = 0.0
    baseline_realized_pnl = 0.0
    baseline_unrealized_pnl = 0.0
    equity = okx_equity
    source = "okx_snapshot"

    snapshot = ExecutionEquitySnapshot(
        mode=mode,
        model_name=model_name,
        snapshot_date=snapshot_date,
        snapshot_at=snapshot_at,
        equity=equity,
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
            "equity": okx_equity,
            "total_pnl": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "source": "okx_snapshot",
        }


def _baseline_must_be_rebuilt(
    row: ExecutionEquitySnapshot,
    okx_equity: float | None,
    snapshot_date: str,
) -> bool:
    source = str(row.source or "")
    if source != "okx_snapshot":
        return True
    if snapshot_date < PHASE3_FIRST_CLEAN_DAY:
        return False
    row_equity = _safe_float(row.equity, None)
    if row_equity is None or row_equity <= 0:
        return True
    if okx_equity is None or okx_equity <= 0:
        return False
    snapshot_at = row.snapshot_at
    if isinstance(snapshot_at, datetime):
        if snapshot_at.tzinfo is None:
            snapshot_at = snapshot_at.replace(tzinfo=BEIJING_TZ)
        snapshot_local = snapshot_at.astimezone(BEIJING_TZ)
        if snapshot_local.date().isoformat() != snapshot_date:
            return True
    drift = abs(okx_equity - row_equity)
    drift_ratio = drift / max(abs(okx_equity), abs(row_equity), 1e-12)
    if drift > OKX_BASELINE_MAX_DRIFT_USDT and drift_ratio > OKX_BASELINE_MAX_DRIFT_RATIO:
        return True
    return False


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


async def _select_first_phase3_okx_snapshot(
    session: AsyncSession,
    mode: str,
    model_name: str,
) -> ExecutionEquitySnapshot | None:
    result = await session.execute(
        select(ExecutionEquitySnapshot)
        .where(
            ExecutionEquitySnapshot.mode == mode,
            ExecutionEquitySnapshot.model_name == model_name,
            ExecutionEquitySnapshot.source == "okx_snapshot",
            ExecutionEquitySnapshot.snapshot_date >= PHASE3_FIRST_CLEAN_DAY,
        )
        .order_by(
            ExecutionEquitySnapshot.snapshot_date.asc(),
            ExecutionEquitySnapshot.snapshot_at.asc(),
            ExecutionEquitySnapshot.id.asc(),
        )
        .limit(1)
    )
    return result.scalar_one_or_none()


def _snapshot_to_dict(snapshot: ExecutionEquitySnapshot) -> dict:
    return {
        "snapshot_at": _snapshot_at_iso(snapshot),
        "equity": float(snapshot.equity or 0.0),
        "total_pnl": float(snapshot.total_pnl or 0.0),
        "realized_pnl": float(snapshot.realized_pnl or 0.0),
        "unrealized_pnl": float(snapshot.unrealized_pnl or 0.0),
        "source": snapshot.source or "observed",
    }


def _snapshot_at_iso(snapshot: ExecutionEquitySnapshot | None) -> str | None:
    if snapshot is None:
        return None
    snapshot_at = snapshot.snapshot_at
    if isinstance(snapshot_at, datetime):
        return snapshot_at.isoformat()
    return None


def _unavailable_baseline(snapshot_date: str) -> dict:
    return {
        "today_equity_pnl": None,
        "today_equity_baseline": None,
        "today_equity_baseline_total_pnl": None,
        "today_equity_baseline_at": None,
        "today_equity_baseline_source": "okx_unavailable",
        "today_snapshot_date": snapshot_date,
    }


def _safe_float(value, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
