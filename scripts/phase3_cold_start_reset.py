#!/usr/bin/env python3
"""Guarded Phase 3 cold-start reset for online paper-trading history.

This script is intentionally conservative:

- it never deletes dashboard users, encrypted settings, or secure-setting audit rows;
- it refuses apply unless the trading service is stopped and OKX paper has no
  open positions or open orders;
- it backs up every affected row before deletion/update.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import and_, delete, func, select, text, update

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models  # noqa: F401,E402 - register SQLAlchemy metadata
import db.session as session_module  # noqa: E402
from config.settings import settings  # noqa: E402
from db.session import close_db, get_engine, init_db  # noqa: E402
from executor.okx_executor import OKXExecutor  # noqa: E402
from models.account import ExecutionEquitySnapshot, VirtualAccount  # noqa: E402
from models.decision import AIDecision  # noqa: E402
from models.learning import (  # noqa: E402
    ExpertMemory,
    ShadowBacktest,
    StrategyLearningEvent,
    StrategyProfileSnapshot,
    TradeReflection,
)
from models.market_data import Kline, Ticker  # noqa: E402
from models.news import NewsArticle, SocialPost  # noqa: E402
from models.risk import ModelPerformanceSnapshot, RiskEvent  # noqa: E402
from models.trade import Order, Position  # noqa: E402
from services.exchange_position_state import parse_exchange_position_snapshot  # noqa: E402
from core.symbols import normalize_trading_symbol  # noqa: E402


CONFIRMATION = "PHASE3_COLD_START_RESET"
DEFAULT_BACKUP_DIR = Path("data/codex_backups/phase3-cold-start-reset")
DEFAULT_MARKER_PATH = ROOT / "data" / "phase3_cold_start_reset_marker.json"
TRADING_SERVICE_NAME = "bb-paper-trading.service"


@dataclass(frozen=True, slots=True)
class ResetTarget:
    name: str
    table: Any
    condition: Callable[[Any], Any] | None = None
    backup_mode: str = "full"
    optional: bool = False


def _paper_condition(table: Any) -> Any:
    return table.c.execution_mode == "paper"


def _paper_decision_condition(table: Any) -> Any:
    return table.c.is_paper.is_(True)


def _paper_equity_condition(table: Any) -> Any:
    return table.c.mode == "paper"


def _all_rows(_table: Any) -> Any | None:
    return None


DELETE_TARGETS: tuple[ResetTarget, ...] = (
    ResetTarget("strategy_learning_events", StrategyLearningEvent.__table__, _paper_condition),
    ResetTarget("trade_reflections", TradeReflection.__table__, _paper_condition),
    ResetTarget("shadow_backtests", ShadowBacktest.__table__, _paper_condition, "summary"),
    ResetTarget(
        "strategy_profile_snapshots",
        StrategyProfileSnapshot.__table__,
        _paper_condition,
        "summary",
    ),
    ResetTarget("orders", Order.__table__, _paper_condition),
    ResetTarget("positions", Position.__table__, _paper_condition),
    ResetTarget("ai_decisions", AIDecision.__table__, _paper_decision_condition),
    ResetTarget("execution_equity_snapshots", ExecutionEquitySnapshot.__table__, _paper_equity_condition),
    ResetTarget("expert_memories", ExpertMemory.__table__, _all_rows, "summary"),
    ResetTarget("risk_events", RiskEvent.__table__, _all_rows, "summary"),
    ResetTarget("model_performance_snapshots", ModelPerformanceSnapshot.__table__, _all_rows),
    ResetTarget("market_klines", Kline.__table__, _all_rows, "summary"),
    ResetTarget("market_tickers", Ticker.__table__, _all_rows, "summary"),
    ResetTarget("news_articles", NewsArticle.__table__, _all_rows, "summary"),
    ResetTarget("social_posts", SocialPost.__table__, _all_rows, "summary"),
)

PRESERVED_TABLES = (
    "dashboard_users",
    "secure_settings",
    "secure_setting_audit",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return str(value)


def _row_to_json(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_safe(value) for key, value in row.items()}


def _condition_for(target: ResetTarget) -> Any | None:
    return target.condition(target.table) if target.condition is not None else None


async def _count_target(target: ResetTarget) -> int:
    engine = await get_engine()
    stmt = select(func.count()).select_from(target.table)
    condition = _condition_for(target)
    if condition is not None:
        stmt = stmt.where(condition)
    async with engine.connect() as conn:
        result = await conn.execute(stmt)
        return int(result.scalar_one() or 0)


async def collect_reset_plan() -> dict[str, Any]:
    await init_db()
    target_counts: dict[str, int] = {}
    for target in DELETE_TARGETS:
        target_counts[target.name] = await _count_target(target)
    account_count = await _count_table(VirtualAccount.__table__)
    return {
        "scope": "paper_cold_start",
        "delete_counts": target_counts,
        "virtual_accounts_to_reset": account_count,
        "preserved_tables": list(PRESERVED_TABLES),
        "policy": {
            "dashboard_users_preserved": True,
            "secure_settings_preserved": True,
            "secure_setting_audit_preserved": True,
            "live_execution_rows_preserved": True,
            "large_derived_tables_use_summary_backup": True,
            "requires_okx_empty_gate": True,
            "requires_trading_service_stopped": True,
        },
    }


async def _count_table(table: Any) -> int:
    engine = await get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(select(func.count()).select_from(table))
        return int(result.scalar_one() or 0)


async def _backup_target(
    target: ResetTarget,
    backup_dir: Path,
    *,
    batch_size: int = 1000,
) -> dict[str, Any]:
    backup_dir.mkdir(parents=True, exist_ok=True)
    if target.backup_mode == "summary":
        return await _backup_target_summary(target, backup_dir)

    path = backup_dir / f"{target.name}.jsonl"
    condition = _condition_for(target)
    stmt = select(target.table)
    if condition is not None:
        stmt = stmt.where(condition)
    if "id" in target.table.c:
        stmt = stmt.order_by(target.table.c.id)

    count = 0
    engine = await get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(stmt)
        with path.open("w", encoding="utf-8") as fh:
            while True:
                rows = result.mappings().fetchmany(batch_size)
                if not rows:
                    break
                for row in rows:
                    fh.write(json.dumps(_row_to_json(dict(row)), ensure_ascii=False) + "\n")
                    count += 1
    return {"table": target.name, "rows": count, "path": str(path)}


async def _backup_target_summary(target: ResetTarget, backup_dir: Path) -> dict[str, Any]:
    condition = _condition_for(target)
    table = target.table
    stmt = select(func.count()).select_from(table)
    min_stmt = select(func.min(table.c.id)) if "id" in table.c else None
    max_stmt = select(func.max(table.c.id)) if "id" in table.c else None
    created_min_stmt = select(func.min(table.c.created_at)) if "created_at" in table.c else None
    created_max_stmt = select(func.max(table.c.created_at)) if "created_at" in table.c else None
    if condition is not None:
        stmt = stmt.where(condition)
        if min_stmt is not None:
            min_stmt = min_stmt.where(condition)
        if max_stmt is not None:
            max_stmt = max_stmt.where(condition)
        if created_min_stmt is not None:
            created_min_stmt = created_min_stmt.where(condition)
        if created_max_stmt is not None:
            created_max_stmt = created_max_stmt.where(condition)

    engine = await get_engine()
    async with engine.connect() as conn:
        count = int((await conn.execute(stmt)).scalar_one() or 0)
        min_id = (await conn.execute(min_stmt)).scalar_one_or_none() if min_stmt is not None else None
        max_id = (await conn.execute(max_stmt)).scalar_one_or_none() if max_stmt is not None else None
        min_created = (
            (await conn.execute(created_min_stmt)).scalar_one_or_none()
            if created_min_stmt is not None
            else None
        )
        max_created = (
            (await conn.execute(created_max_stmt)).scalar_one_or_none()
            if created_max_stmt is not None
            else None
        )
        samples = await _sample_rows(conn, target, limit=20)

    digest = hashlib.sha256(
        json.dumps(
            {
                "table": target.name,
                "rows": count,
                "min_id": _json_safe(min_id),
                "max_id": _json_safe(max_id),
                "min_created_at": _json_safe(min_created),
                "max_created_at": _json_safe(max_created),
                "sample_ids": [sample.get("id") for sample in samples],
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "table": target.name,
        "backup_mode": "summary",
        "rows": count,
        "min_id": _json_safe(min_id),
        "max_id": _json_safe(max_id),
        "min_created_at": _json_safe(min_created),
        "max_created_at": _json_safe(max_created),
        "sample_rows": samples,
        "sha256": digest,
        "reason": "Large derived/cache table. Phase 3 cold start intentionally does not retain full dirty training/cache history.",
    }
    path = backup_dir / f"{target.name}.summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"table": target.name, "rows": count, "path": str(path), "backup_mode": "summary"}


async def _sample_rows(conn: Any, target: ResetTarget, *, limit: int) -> list[dict[str, Any]]:
    table = target.table
    stmt = select(table)
    condition = _condition_for(target)
    if condition is not None:
        stmt = stmt.where(condition)
    if "id" in table.c:
        stmt = stmt.order_by(table.c.id.desc())
    stmt = stmt.limit(limit)
    result = await conn.execute(stmt)
    return [_row_to_json(dict(row)) for row in result.mappings().all()]


async def backup_reset_rows(backup_dir: Path) -> dict[str, Any]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = backup_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    plan = await collect_reset_plan()
    table_backups = []
    for target in DELETE_TARGETS:
        table_backups.append(await _backup_target(target, run_dir))
    table_backups.append(await _backup_target(ResetTarget("virtual_accounts", VirtualAccount.__table__), run_dir))

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "policy_id": CONFIRMATION,
        "plan": plan,
        "table_backups": table_backups,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"backup_dir": str(run_dir), "manifest_path": str(manifest_path), "table_backups": table_backups}


async def apply_reset(*, batch_size: int = 10_000) -> dict[str, Any]:
    await init_db()
    engine = await get_engine()
    deleted: dict[str, int] = {}
    for target in DELETE_TARGETS:
        deleted[target.name] = await _delete_target(target, batch_size=batch_size)

    async with engine.begin() as conn:
        account_result = await conn.execute(
            update(VirtualAccount)
            .values(
                current_balance=VirtualAccount.initial_balance,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                total_trades=0,
                winning_trades=0,
            )
        )
        account_reset_count = int(account_result.rowcount or 0)

    await _reset_sequences()
    return {
        "deleted": deleted,
        "virtual_accounts_reset": account_reset_count,
        "preserved_tables": list(PRESERVED_TABLES),
    }


def write_cold_start_marker(
    marker_path: Path,
    *,
    backup: dict[str, Any],
    reset_result: dict[str, Any],
    service_gate: dict[str, Any],
    okx_gate: dict[str, Any],
) -> dict[str, Any]:
    reset_at = datetime.now(UTC).isoformat()
    payload = {
        "reset_at": reset_at,
        "policy_id": CONFIRMATION,
        "scope": "paper_cold_start",
        "mode": "paper",
        "backup_dir": backup.get("backup_dir"),
        "backup_manifest_path": backup.get("manifest_path"),
        "deleted": reset_result.get("deleted", {}),
        "virtual_accounts_reset": reset_result.get("virtual_accounts_reset", 0),
        "preserved_tables": list(PRESERVED_TABLES),
        "service_gate": service_gate,
        "okx_gate": {
            "ok": bool(okx_gate.get("ok")),
            "mode": okx_gate.get("mode"),
            "open_position_count": okx_gate.get("open_position_count"),
            "open_order_count": okx_gate.get("open_order_count"),
        },
        "watermark_policy": {
            "okx_authoritative_sync_ignores_pre_reset_fills": True,
            "okx_authoritative_sync_min_fill_timestamp": reset_at,
        },
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(marker_path), "reset_at": reset_at, "payload": payload}


async def _delete_target(target: ResetTarget, *, batch_size: int) -> int:
    table = target.table
    condition = _condition_for(target)
    if "id" not in table.c:
        return await _delete_target_once(target)
    if batch_size <= 0:
        batch_size = 10_000

    deleted_total = 0
    engine = await get_engine()
    while True:
        async with engine.begin() as conn:
            select_ids = select(table.c.id)
            if condition is not None:
                select_ids = select_ids.where(condition)
            select_ids = select_ids.order_by(table.c.id).limit(batch_size)
            id_count = int(
                (
                    await conn.execute(
                        select(func.count()).select_from(select_ids.subquery())
                    )
                ).scalar_one()
                or 0
            )
            if id_count <= 0:
                return deleted_total
            delete_condition = table.c.id.in_(select_ids.scalar_subquery())
            if condition is not None:
                delete_condition = and_(condition, delete_condition)
            result = await conn.execute(delete(table).where(delete_condition))
            deleted_total += int(result.rowcount or 0)


async def _delete_target_once(target: ResetTarget) -> int:
    engine = await get_engine()
    async with engine.begin() as conn:
        stmt = delete(target.table)
        condition = _condition_for(target)
        if condition is not None:
            stmt = stmt.where(condition)
        result = await conn.execute(stmt)
        return int(result.rowcount or 0)


async def _reset_sequences() -> None:
    if "postgresql" not in settings.database_url:
        return
    engine = await get_engine()
    async with engine.begin() as conn:
        for target in DELETE_TARGETS:
            table = target.table
            if "id" not in table.c:
                continue
            max_result = await conn.execute(select(func.max(table.c.id)))
            max_id = int(max_result.scalar_one_or_none() or 0)
            await conn.execute(
                text(
                    "SELECT setval("
                    "pg_get_serial_sequence(:table_name, 'id'), "
                    ":sequence_value, "
                    ":is_called)"
                ),
                {
                    "table_name": table.name,
                    "sequence_value": max(max_id, 1),
                    "is_called": max_id > 0,
                },
            )


def check_trading_service_stopped(service_name: str = TRADING_SERVICE_NAME) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["systemctl", "is-active", service_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return {"ok": False, "service": service_name, "status": "systemctl_missing"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "service": service_name, "status": "timeout"}
    status = (completed.stdout or completed.stderr or "").strip()
    return {
        "ok": status in {"inactive", "failed", "unknown"},
        "service": service_name,
        "status": status or f"exit_{completed.returncode}",
    }


async def check_okx_paper_empty() -> dict[str, Any]:
    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    try:
        await executor.initialize()
        positions = await executor.get_positions_strict()
        open_positions = []
        for row in positions or []:
            snapshot = parse_exchange_position_snapshot(
                row,
                symbol_normalizer=normalize_trading_symbol,
            )
            if not snapshot:
                continue
            exposure = max(
                abs(float(snapshot.get("contracts") or 0.0)),
                abs(float(snapshot.get("quantity") or 0.0)),
            )
            if exposure > 0:
                open_positions.append(snapshot)
        open_orders = await executor.get_open_orders_strict()
        return {
            "ok": not open_positions and not open_orders,
            "mode": "paper",
            "open_position_count": len(open_positions),
            "open_order_count": len(open_orders or []),
            "open_positions": open_positions[:20],
            "open_orders": [
                {
                    "id": row.get("id") or (row.get("info") or {}).get("ordId"),
                    "symbol": row.get("symbol") or (row.get("info") or {}).get("instId"),
                    "side": row.get("side") or (row.get("info") or {}).get("side"),
                    "status": row.get("status") or (row.get("info") or {}).get("state"),
                }
                for row in (open_orders or [])[:20]
                if isinstance(row, dict)
            ],
        }
    except Exception as exc:
        return {"ok": False, "mode": "paper", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        await executor.shutdown()


async def run(
    *,
    apply: bool,
    confirm: str,
    backup_dir: Path,
    marker_path: Path = DEFAULT_MARKER_PATH,
    batch_size: int = 10_000,
    skip_service_gate: bool = False,
    skip_okx_gate: bool = False,
) -> dict[str, Any]:
    if apply and confirm != CONFIRMATION:
        raise SystemExit(f"--apply requires --confirm {CONFIRMATION}")
    if apply and skip_okx_gate:
        raise SystemExit("--skip-okx-gate is not allowed with --apply")

    service_gate = (
        {"ok": True, "skipped": True}
        if skip_service_gate
        else check_trading_service_stopped(TRADING_SERVICE_NAME)
    )
    okx_gate = (
        {"ok": True, "skipped": True}
        if skip_okx_gate
        else await check_okx_paper_empty()
    )
    plan = await collect_reset_plan()
    result: dict[str, Any] = {
        "apply": apply,
        "confirmation": confirm,
        "service_gate": service_gate,
        "okx_gate": okx_gate,
        "plan": plan,
    }
    if not apply:
        return result
    if not service_gate.get("ok"):
        raise SystemExit(f"Trading service gate failed: {service_gate}")
    if not okx_gate.get("ok"):
        raise SystemExit(f"OKX empty gate failed: {okx_gate}")

    backup = await backup_reset_rows(backup_dir)
    reset_result = await apply_reset(batch_size=batch_size)
    marker = write_cold_start_marker(
        marker_path,
        backup=backup,
        reset_result=reset_result,
        service_gate=service_gate,
        okx_gate=okx_gate,
    )
    result["backup"] = backup
    result["cold_start_marker"] = marker
    result["result"] = reset_result
    result["post_plan"] = await collect_reset_plan()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply the reset after all gates pass.")
    parser.add_argument("--confirm", default="", help=f"Required confirmation token: {CONFIRMATION}")
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--marker-path", type=Path, default=DEFAULT_MARKER_PATH)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument(
        "--skip-service-gate",
        action="store_true",
        help="Skip systemd service check. Intended for local dry-run/tests only.",
    )
    parser.add_argument(
        "--skip-okx-gate",
        action="store_true",
        help="Skip OKX paper empty check. Allowed only without --apply.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    try:
        result = await run(
            apply=bool(args.apply),
            confirm=str(args.confirm or ""),
            backup_dir=args.backup_dir,
            marker_path=args.marker_path,
            batch_size=int(args.batch_size or 10_000),
            skip_service_gate=bool(args.skip_service_gate),
            skip_okx_gate=bool(args.skip_okx_gate),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    finally:
        await close_db()
        session_module._engine = None
        session_module._sessionmaker = None


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
