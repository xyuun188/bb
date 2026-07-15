"""Run Phase 3 OKX fact sync and reconciliation checks.

Default mode is read-only: generate the OKX daily reconciliation report without
mutating local data. Use --apply-order-sync to refresh the local OKX order/fill
fact cache from OKX native fills history, then run the report again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime, timedelta, timezone
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

from sqlalchemy import delete, or_, select  # noqa: E402

from config.settings import ENSEMBLE_TRADER_NAME, settings  # noqa: E402
from core.safe_output import safe_error_text  # noqa: E402
from db.session import close_db, get_session_ctx  # noqa: E402
from executor.okx_executor import OKXExecutor  # noqa: E402
from models.account import ExecutionEquitySnapshot  # noqa: E402
from models.trade import Order, Position  # noqa: E402
from scripts.run_okx_daily_reconciliation_report import (  # noqa: E402
    DEFAULT_REPORT_DIR,
    collect_report,
    write_report,
)
from services.okx_order_fact_sync import (  # noqa: E402
    PHASE3_DEFAULT_ORDER_SYNC_START,
    OkxOrderFactSyncService,
    _db_naive_since,
)

PHASE3_CLEAN_SNAPSHOT_DATE = "2026-06-28"
BEIJING_TZ = timezone(timedelta(hours=8))
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("paper", "live"), default="paper")
    parser.add_argument(
        "--apply-order-sync",
        action="store_true",
        help="Write local order/fill fact cache from OKX native fills history.",
    )
    parser.add_argument(
        "--reset-local-cache",
        action="store_true",
        help=(
            "Explicit recovery-only reset before rebuilding local Phase 3 facts. "
            "Normal scheduled sync remains incremental."
        ),
    )
    parser.add_argument(
        "--allow-cache",
        action="store_true",
        help="Allow cached reconciliation cards. Default forces a fresh report.",
    )
    parser.add_argument("--json-indent", type=int, default=2)
    args = parser.parse_args(argv)
    if args.reset_local_cache and not args.apply_order_sync:
        parser.error("--reset-local-cache requires --apply-order-sync")
    return args


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _authoritative_report_output_dir() -> Path:
    return settings.data_dir / DEFAULT_REPORT_DIR


async def run(
    *,
    mode: str,
    apply_order_sync: bool,
    allow_cache: bool,
    reset_local_cache: bool = False,
) -> dict[str, Any]:
    before_report = await collect_report(allow_cache=allow_cache)
    sync_result: dict[str, Any] | None = None
    sync_error: str | None = None
    cleanup_result: dict[str, Any] | None = None
    equity_snapshot_result: dict[str, Any] | None = None
    if apply_order_sync:
        try:
            if reset_local_cache:
                cleanup_result = await _cleanup_phase3_local_okx_cache(mode=mode)
            equity_snapshot_result = await _sync_okx_equity_snapshot(mode=mode)
            sync_result = await OkxOrderFactSyncService(mode=mode).sync()
        except Exception as exc:  # pragma: no cover - defensive operator output
            sync_error = f"{type(exc).__name__}: {safe_error_text(exc, limit=240)}"
    after_report = await collect_report(allow_cache=False)
    return _json_safe(
        {
            "script": "run_phase3_okx_fact_sync",
            "mode": "live" if mode == "live" else "paper",
            "generated_at": datetime.now(UTC).isoformat(),
            "mutated_database": bool(apply_order_sync and not sync_error),
            "order_sync_applied": bool(apply_order_sync),
            "local_cache_reset_requested": bool(reset_local_cache),
            "cleanup_result": cleanup_result,
            "equity_snapshot_result": equity_snapshot_result,
            "order_sync_result": sync_result,
            "order_sync_error": sync_error,
            "before_reconciliation": {
                "status": before_report.get("status"),
                "can_open_new_entries": before_report.get("can_open_new_entries"),
                "can_refresh_training": before_report.get("can_refresh_training"),
                "requires_attention": before_report.get("requires_attention"),
                "issue_summary": (before_report.get("issue_ledger") or {}).get("summary"),
            },
            "after_reconciliation": {
                "status": after_report.get("status"),
                "can_open_new_entries": after_report.get("can_open_new_entries"),
                "can_refresh_training": after_report.get("can_refresh_training"),
                "requires_attention": after_report.get("requires_attention"),
                "issue_summary": (after_report.get("issue_ledger") or {}).get("summary"),
            },
            "after_report": after_report,
        }
    )


async def _cleanup_phase3_local_okx_cache(*, mode: str) -> dict[str, Any]:
    """Remove Phase 3 local exchange-fact cache before rebuilding from OKX.

    Phase 3 paper/live account truth is OKX-only.  The local DB may cache OKX
    facts, but stale locally generated orders, positions, and equity snapshots
    must not survive a resync and keep polluting dashboard PnL or training.
    """

    selected_mode = "live" if mode == "live" else "paper"
    since_naive = _db_naive_since(PHASE3_DEFAULT_ORDER_SYNC_START)
    async with get_session_ctx() as session:
        order_result = await session.execute(
            delete(Order).where(
                Order.execution_mode == selected_mode,
                or_(
                    Order.created_at >= since_naive,
                    Order.filled_at >= since_naive,
                    Order.okx_synced_at >= since_naive,
                ),
            )
        )
        position_result = await session.execute(
            delete(Position).where(
                Position.execution_mode == selected_mode,
                or_(
                    Position.created_at >= since_naive,
                    Position.closed_at >= since_naive,
                    Position.is_open.is_(True),
                ),
            )
        )
        equity_result = await session.execute(
            delete(ExecutionEquitySnapshot).where(
                ExecutionEquitySnapshot.mode == selected_mode,
                or_(
                    ExecutionEquitySnapshot.snapshot_date < PHASE3_CLEAN_SNAPSHOT_DATE,
                    _phase3_boundary_synthetic_snapshot_filter(),
                    ExecutionEquitySnapshot.source != "okx_snapshot",
                    ExecutionEquitySnapshot.source.is_(None),
                    ExecutionEquitySnapshot.model_name != ENSEMBLE_TRADER_NAME,
                    ExecutionEquitySnapshot.model_name.is_(None),
                ),
            )
        )
    return {
        "orders_deleted": int(order_result.rowcount or 0),
        "positions_deleted": int(position_result.rowcount or 0),
        "execution_equity_snapshots_deleted": int(equity_result.rowcount or 0),
        "snapshot_date_from": PHASE3_CLEAN_SNAPSHOT_DATE,
        "phase3_boundary_utc": since_naive.replace(tzinfo=UTC).isoformat(),
        "phase3_boundary_local": PHASE3_DEFAULT_ORDER_SYNC_START.isoformat(),
        "reason": "phase3_okx_authoritative_resync_rebuilds_orders_positions_and_equity_cache",
    }


async def _sync_okx_equity_snapshot(*, mode: str, now: datetime | None = None) -> dict[str, Any]:
    """Persist today's account-equity baseline from OKX only.

    This intentionally records only the first OKX equity observed for the
    Beijing day. Current equity is always read live from OKX by dashboard code;
    the persisted row is the OKX-backed baseline used for daily/Phase-3 deltas.
    """

    selected_mode = "live" if mode == "live" else "paper"
    snapshot_at = now or datetime.now(UTC)
    if snapshot_at.tzinfo is None:
        snapshot_at = snapshot_at.replace(tzinfo=UTC)
    snapshot_day = snapshot_at.astimezone(BEIJING_TZ).date().isoformat()

    executor = OKXExecutor(mode=selected_mode, load_markets_on_initialize=False)
    try:
        await executor.initialize()
        snapshot = await executor.get_balance_snapshot("USDT")
        bill_audit = await _audit_phase3_account_bills(
            executor,
            current_equity=_snapshot_equity(snapshot) if isinstance(snapshot, dict) else 0.0,
        )
    finally:
        try:
            await executor.shutdown()
        except Exception:
            logger.debug("OKX equity snapshot executor shutdown failed", exc_info=True)

    if not isinstance(snapshot, dict) or snapshot.get("error"):
        return {
            "status": "okx_unavailable",
            "mode": selected_mode,
            "snapshot_date": snapshot_day,
            "error": safe_error_text((snapshot or {}).get("error") if isinstance(snapshot, dict) else "empty OKX balance snapshot", limit=180),
            "mutated": False,
        }

    equity = _snapshot_equity(snapshot)
    if equity <= 0:
        return {
            "status": "okx_unavailable",
            "mode": selected_mode,
            "snapshot_date": snapshot_day,
            "error": "OKX balance snapshot has no positive equity",
            "mutated": False,
        }

    async with get_session_ctx() as session:
        legacy_snapshot_cleanup = await _delete_legacy_phase3_equity_snapshots(
            session,
            mode=selected_mode,
            bill_audit=bill_audit,
        )

        result = await session.execute(
            select(ExecutionEquitySnapshot)
            .where(
                ExecutionEquitySnapshot.mode == selected_mode,
                ExecutionEquitySnapshot.model_name == ENSEMBLE_TRADER_NAME,
                ExecutionEquitySnapshot.snapshot_date == snapshot_day,
            )
            .order_by(ExecutionEquitySnapshot.id.asc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row:
            if row.source != "okx_snapshot":
                row.snapshot_at = snapshot_at
                row.equity = equity
                row.total_pnl = 0.0
                row.realized_pnl = 0.0
                row.unrealized_pnl = 0.0
                row.source = "okx_snapshot"
                await session.flush()
                return {
                    "status": "updated",
                    "mode": selected_mode,
                    "snapshot_date": snapshot_day,
                    "snapshot_at": snapshot_at.isoformat(),
                    "equity": equity,
                    "mutated": True,
                }
            return {
                "status": "kept_existing_okx_snapshot",
                "mode": selected_mode,
                "snapshot_date": snapshot_day,
                "snapshot_at": row.snapshot_at.isoformat() if row.snapshot_at else None,
                "equity": float(row.equity or 0.0),
                "account_bill_audit": bill_audit,
                "legacy_phase3_snapshot_cleanup": legacy_snapshot_cleanup,
                "mutated": False,
            }

        session.add(
            ExecutionEquitySnapshot(
                mode=selected_mode,
                model_name=ENSEMBLE_TRADER_NAME,
                snapshot_date=snapshot_day,
                snapshot_at=snapshot_at,
                equity=equity,
                total_pnl=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                source="okx_snapshot",
            )
        )
        await session.flush()
    return {
        "status": "created",
        "mode": selected_mode,
        "snapshot_date": snapshot_day,
        "snapshot_at": snapshot_at.isoformat(),
        "equity": equity,
        "account_bill_audit": bill_audit,
        "legacy_phase3_snapshot_cleanup": legacy_snapshot_cleanup,
        "mutated": True,
    }


def _snapshot_equity(snapshot: dict[str, Any]) -> float:
    for key in ("equity", "total", "cash", "allocatable", "free"):
        try:
            value = float(snapshot.get(key) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value
    return 0.0


async def _audit_phase3_account_bills(
    executor: OKXExecutor,
    *,
    current_equity: float,
) -> dict[str, Any]:
    """Audit OKX account bills without using them as an equity baseline.

    OKX bills are balance-changing transaction records, not account-equity
    snapshots.  They are useful for audit/debugging, but Phase 3 account equity
    PnL must not be rebuilt from bills because current equity also includes
    mark-to-market position value.
    """

    if current_equity <= 0:
        return {"available": False, "reason": "current_okx_equity_unavailable"}
    try:
        ccxt = await executor._get_ccxt()
    except Exception as exc:
        return {
            "available": False,
            "reason": "okx_client_unavailable",
            "error": safe_error_text(exc, limit=180),
        }
    fetch_methods = [
        method
        for method in (
            getattr(ccxt, "privateGetAccountBills", None),
            getattr(ccxt, "privateGetAccountBillsArchive", None),
        )
        if callable(method)
    ]
    if not fetch_methods:
        return {"available": False, "reason": "okx_account_bills_api_unavailable"}

    since_ms = int(PHASE3_DEFAULT_ORDER_SYNC_START.astimezone(UTC).timestamp() * 1000)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for fetch in fetch_methods:
        after = ""
        complete = True
        for _page in range(20):
            params: dict[str, Any] = {
                "ccy": "USDT",
                "begin": str(since_ms),
                "limit": "100",
            }
            if after:
                params["after"] = after
            try:
                response = await executor._with_retry(fetch, params)
            except Exception as exc:
                if fetch is fetch_methods[-1] and not rows:
                    return {
                        "available": False,
                        "reason": "okx_account_bills_read_failed",
                        "error": safe_error_text(exc, limit=180),
                    }
                complete = False
                break
            page_rows = response.get("data", []) if isinstance(response, dict) else []
            if not isinstance(page_rows, list):
                break
            for row in page_rows:
                if not isinstance(row, dict):
                    continue
                ts = str(row.get("ts") or row.get("uTime") or "").strip()
                if _safe_float(ts, 0.0) < since_ms:
                    continue
                if str(row.get("ccy") or "USDT").upper() != "USDT":
                    continue
                key = (
                    ts,
                    str(row.get("billId") or row.get("subType") or ""),
                    str(row.get("balChg") or row.get("pnl") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
            if len(page_rows) < 100:
                break
            cursor = _oldest_bill_cursor(page_rows)
            if not cursor or cursor == after:
                complete = False
                break
            after = cursor

    balance_change = sum(_bill_balance_change(row) for row in rows)
    return {
        "available": True,
        "audit_only": True,
        "usable_for_equity_baseline": False,
        "reason": "okx_account_bills_are_balance_changes_not_account_equity_snapshots",
        "source": "okx_account_bills",
        "bill_count": len(rows),
        "complete": complete,
        "current_equity": current_equity,
        "net_balance_change_since_phase3": balance_change,
        "phase3_start_at": PHASE3_DEFAULT_ORDER_SYNC_START.astimezone(UTC).isoformat(),
    }


async def _delete_legacy_phase3_equity_snapshots(
    session: Any,
    *,
    mode: str,
    bill_audit: dict[str, Any],
) -> dict[str, Any]:
    """Remove synthetic Phase 3 start snapshots that are not OKX balance reads.

    The previous implementation wrote `current_equity - sum(balChg)` as an
    `okx_snapshot`.  That value is not an OKX equity snapshot, so keeping it
    makes Phase 3 cumulative equity PnL drift away from the OKX balance page.
    """

    result = await session.execute(
        select(ExecutionEquitySnapshot)
        .where(
            ExecutionEquitySnapshot.mode == mode,
            ExecutionEquitySnapshot.model_name == ENSEMBLE_TRADER_NAME,
            ExecutionEquitySnapshot.snapshot_date == PHASE3_CLEAN_SNAPSHOT_DATE,
            ExecutionEquitySnapshot.source == "okx_snapshot",
        )
        .order_by(ExecutionEquitySnapshot.id.asc())
    )
    phase3_start = PHASE3_DEFAULT_ORDER_SYNC_START.astimezone(UTC)
    deleted_ids: list[int] = []
    for row in result.scalars().all():
        row_at = _as_utc(getattr(row, "snapshot_at", None))
        is_boundary_synthetic = (
            row_at is not None and abs((row_at - phase3_start).total_seconds()) <= 1
        )
        if not is_boundary_synthetic:
            continue
        deleted_ids.append(int(getattr(row, "id", 0) or 0))
        await session.delete(row)
    if deleted_ids:
        await session.flush()
    return {
        "deleted": len(deleted_ids),
        "deleted_ids": deleted_ids,
        "reason": "removed_legacy_synthetic_phase3_equity_snapshot",
    }


def _phase3_boundary_synthetic_snapshot_filter() -> Any:
    """Match the old synthetic Phase 3 start row created at the boundary."""

    boundary_naive = PHASE3_DEFAULT_ORDER_SYNC_START.astimezone(UTC).replace(tzinfo=None)
    boundary_aware = PHASE3_DEFAULT_ORDER_SYNC_START.astimezone(UTC)
    return (
        (ExecutionEquitySnapshot.model_name == ENSEMBLE_TRADER_NAME)
        & (ExecutionEquitySnapshot.snapshot_date == PHASE3_CLEAN_SNAPSHOT_DATE)
        & (ExecutionEquitySnapshot.source == "okx_snapshot")
        & (
            (ExecutionEquitySnapshot.snapshot_at == boundary_naive)
            | (ExecutionEquitySnapshot.snapshot_at == boundary_aware)
        )
    )


def _as_utc(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _bill_balance_change(row: dict[str, Any]) -> float:
    for key in ("balChg", "balanceChange", "pnl", "fee"):
        value = _safe_float(row.get(key), None)
        if value is not None:
            return value
    return 0.0


def _oldest_bill_cursor(rows: list[dict[str, Any]]) -> str:
    oldest = ""
    oldest_ts = float("inf")
    for row in rows:
        ts = _safe_float(row.get("ts") or row.get("uTime"), 0.0)
        if ts and ts < oldest_ts:
            oldest_ts = ts
            oldest = str(row.get("billId") or row.get("ts") or row.get("uTime") or "")
    return oldest


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


async def async_main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = await run(
            mode=str(args.mode or "paper"),
            apply_order_sync=bool(args.apply_order_sync),
            allow_cache=bool(args.allow_cache),
            reset_local_cache=bool(args.reset_local_cache),
        )
    finally:
        await close_db()
    indent = None if int(args.json_indent or 0) <= 0 else int(args.json_indent)
    artifact_error: dict[str, str] | None = None
    after_report = result.get("after_report") if isinstance(result, dict) else None
    report_output_dir = _authoritative_report_output_dir()
    try:
        if not isinstance(after_report, dict):
            raise TypeError("final reconciliation report is not an object")
        result["reconciliation_report_artifacts"] = write_report(
            after_report,
            report_output_dir,
            indent=indent,
        )
    except Exception as exc:
        artifact_error = {
            "code": "authoritative_reconciliation_report_write_failed",
            "message": safe_error_text(exc, limit=240),
            "output_dir": str(report_output_dir),
        }
        result["reconciliation_report_artifact_error"] = artifact_error
    print(json.dumps(result, ensure_ascii=False, indent=indent, sort_keys=True))
    after = result.get("after_reconciliation") if isinstance(result, dict) else {}
    if result.get("order_sync_error") or artifact_error:
        return 2
    if not isinstance(after, dict):
        return 2
    return 0 if after.get("requires_attention") is False else 1


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
