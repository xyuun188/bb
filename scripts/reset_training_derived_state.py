#!/usr/bin/env python3
"""Remove the current training-derived state and start a clean training epoch.

The command deliberately does not touch exchange-backed trade facts or audit
events.  It removes only derived samples, model artifacts, cursors and caches;
the next training run rebuilds those outputs from the preserved fact ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import db.session as session_module  # noqa: E402
import models  # noqa: F401,E402 - register all SQLAlchemy tables
from config.settings import settings  # noqa: E402
from db.session import close_db, get_engine, init_db  # noqa: E402
from models.account import ExecutionEquitySnapshot, OkxAccountBill, VirtualAccount  # noqa: E402
from models.dashboard_auth import DashboardUser  # noqa: E402
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
from models.risk import RiskEvent  # noqa: E402
from models.secure_config import SecureSetting, SecureSettingAudit  # noqa: E402
from models.trade import OkxPositionHistory, Order, Position  # noqa: E402
from services.training_epoch import write_training_epoch  # noqa: E402

CONFIRMATION = "RESET_TRAINING_DERIVED_STATE"
RESET_SERVICES = (
    "bb-paper-trading.service",
    "bb-model-tunnels.service",
    "bb-dashboard.service",
)
MANIFEST_FILENAME = "training_reset_manifest.json"

DERIVED_TABLES = (
    ("shadow_backtests", ShadowBacktest.__table__),
    ("trade_reflections", TradeReflection.__table__),
    ("expert_memories", ExpertMemory.__table__),
    ("strategy_profile_snapshots", StrategyProfileSnapshot.__table__),
    ("execution_equity_snapshots", ExecutionEquitySnapshot.__table__),
)

DERIVED_DIRECTORIES = (
    "ml_signal",
    "local_ai_tools",
    "model_artifacts",
    "models",
    "training_cache",
    "model_cache",
    "vector_memory",
    "model_training_scheduler_state.locks",
)
DERIVED_FILES = (
    "model_training_scheduler_state.json",
    "local_ai_tools_training.lock",
    "strategy_learning_state.json",
    "system_audit_latest.json",
)

PRESERVED_TABLES = (
    ("orders", Order.__table__),
    ("positions", Position.__table__),
    ("okx_position_history", OkxPositionHistory.__table__),
    ("okx_account_bills", OkxAccountBill.__table__),
    ("ai_decisions", AIDecision.__table__),
    ("strategy_learning_events", StrategyLearningEvent.__table__),
    ("risk_events", RiskEvent.__table__),
    ("market_klines", Kline.__table__),
    ("market_tickers", Ticker.__table__),
    ("news_articles", NewsArticle.__table__),
    ("social_posts", SocialPost.__table__),
    ("virtual_accounts", VirtualAccount.__table__),
    ("dashboard_users", DashboardUser.__table__),
    ("secure_settings", SecureSetting.__table__),
    ("secure_setting_audit", SecureSettingAudit.__table__),
)


@dataclass(frozen=True, slots=True)
class FileCandidate:
    path: str
    kind: str
    size_bytes: int
    deletion_blocker: str | None


def _resolved_data_dir(data_dir: Path) -> Path:
    resolved = data_dir.resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise NotADirectoryError(f"data directory does not exist: {resolved}")
    return resolved


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _file_candidates(data_dir: Path) -> list[FileCandidate]:
    candidates: list[FileCandidate] = []
    for relative in DERIVED_DIRECTORIES:
        path = data_dir / relative
        if path.is_symlink() or not path.exists() or not _inside(path, data_dir):
            continue
        if path.is_dir():
            size = sum(
                child.stat().st_size
                for child in path.rglob("*")
                if child.is_file() and not child.is_symlink()
            )
            candidates.append(
                FileCandidate(
                    str(path),
                    "directory",
                    size,
                    _deletion_blocker(path),
                )
            )
    for relative in DERIVED_FILES:
        path = data_dir / relative
        if path.is_symlink() or not path.is_file() or not _inside(path, data_dir):
            continue
        candidates.append(
            FileCandidate(
                str(path),
                "file",
                path.stat().st_size,
                _deletion_blocker(path),
            )
        )
    return candidates


def _deletion_blocker(path: Path) -> str | None:
    parent = path.parent
    if not os.access(parent, os.W_OK | os.X_OK):
        return f"parent_not_writable:{parent}"
    if path.is_file():
        return None
    for current, directories, files in os.walk(path):
        current_path = Path(current)
        if not os.access(current_path, os.W_OK | os.X_OK):
            return f"directory_not_writable:{current_path}"
        for name in (*directories, *files):
            child = current_path / name
            if child.is_symlink():
                return f"symlink_not_allowed:{child}"
    return None


async def _table_counts(tables: tuple[tuple[str, Any], ...]) -> dict[str, int]:
    engine = await get_engine()
    counts: dict[str, int] = {}
    async with engine.connect() as conn:
        for name, table in tables:
            counts[name] = int(
                (await conn.execute(select(func.count()).select_from(table))).scalar_one()
                or 0
            )
    return counts


def check_services_stopped(services: tuple[str, ...] = RESET_SERVICES) -> dict[str, Any]:
    statuses: dict[str, str] = {}
    for service in services:
        try:
            completed = subprocess.run(
                ["systemctl", "is-active", service],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError:
            return {"ok": False, "reason": "systemctl_missing", "services": statuses}
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": "systemctl_timeout", "services": statuses}
        statuses[service] = (completed.stdout or completed.stderr or "").strip()
    return {
        "ok": all(status in {"inactive", "failed", "unknown"} for status in statuses.values()),
        "services": statuses,
    }


async def collect_plan(data_dir: Path | None = None) -> dict[str, Any]:
    resolved = _resolved_data_dir(data_dir or settings.data_dir)
    await init_db()
    return {
        "scope": "training_derived_state_only",
        "delete_table_counts": await _table_counts(DERIVED_TABLES),
        "delete_file_candidates": [asdict(candidate) for candidate in _file_candidates(resolved)],
        "preserved_table_counts": await _table_counts(PRESERVED_TABLES),
        "policy": {
            "raw_exchange_facts_preserved": True,
            "audit_events_preserved": True,
            "model_artifacts_rebuilt_from_current_epoch": True,
            "requires_trading_and_model_services_stopped": True,
            "no_backup_of_deleted_derived_payloads": True,
        },
    }


async def _delete_tables() -> dict[str, int]:
    engine = await get_engine()
    deleted: dict[str, int] = {}
    async with engine.begin() as conn:
        for name, table in DERIVED_TABLES:
            result = await conn.execute(delete(table))
            deleted[name] = int(result.rowcount or 0)
    return deleted


def _delete_files(data_dir: Path) -> list[str]:
    deleted: list[str] = []
    for candidate in _file_candidates(data_dir):
        path = Path(candidate.path)
        if not _inside(path, data_dir) or path.is_symlink():
            raise RuntimeError(f"refusing to delete unsafe derived path: {path}")
        if candidate.kind == "directory":
            shutil.rmtree(path)
        else:
            path.unlink()
        deleted.append(str(path))
    return deleted


def _write_manifest(data_dir: Path, payload: dict[str, Any]) -> str:
    path = data_dir / MANIFEST_FILENAME
    temporary = data_dir / f".{MANIFEST_FILENAME}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return str(path)


async def run(
    *,
    apply: bool,
    confirm: str,
    data_dir: Path | None = None,
    skip_service_gate: bool = False,
) -> dict[str, Any]:
    if apply and confirm != CONFIRMATION:
        raise SystemExit(f"--apply requires --confirm {CONFIRMATION}")
    if apply and skip_service_gate:
        raise SystemExit("--skip-service-gate is allowed only for dry-run validation")
    resolved = _resolved_data_dir(data_dir or settings.data_dir)
    service_gate = (
        {"ok": True, "skipped": True}
        if skip_service_gate
        else check_services_stopped()
    )
    plan = await collect_plan(resolved)
    result: dict[str, Any] = {
        "apply": apply,
        "confirmation": confirm,
        "service_gate": service_gate,
        "plan": plan,
    }
    if not apply:
        return result
    if not service_gate.get("ok"):
        raise SystemExit(f"service gate failed: {service_gate}")
    deletion_blockers = [
        candidate
        for candidate in plan["delete_file_candidates"]
        if candidate.get("deletion_blocker")
    ]
    if deletion_blockers:
        raise RuntimeError(
            "derived file permission preflight failed before database deletion: "
            + json.dumps(deletion_blockers, ensure_ascii=False)
        )

    deleted_tables = await _delete_tables()
    deleted_files = _delete_files(resolved)
    reset_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    epoch_payload = write_training_epoch(
        resolved / "training_epoch.json",
        reset_id=reset_id,
    )
    post_plan = await collect_plan(resolved)
    preserved_counts_match = (
        post_plan["preserved_table_counts"] == plan["preserved_table_counts"]
    )
    manifest_payload = {
        "reset_id": reset_id,
        "reset_at": epoch_payload["epoch_started_at"],
        "policy": CONFIRMATION,
        "deleted_tables": deleted_tables,
        "deleted_files": deleted_files,
        "preserved_table_counts_before": plan["preserved_table_counts"],
        "preserved_table_counts_after": post_plan["preserved_table_counts"],
        "preserved_table_counts_match": preserved_counts_match,
        "training_epoch": epoch_payload,
    }
    manifest_path = _write_manifest(resolved, manifest_payload)
    result["result"] = {
        "deleted_tables": deleted_tables,
        "deleted_files": deleted_files,
        "training_epoch": epoch_payload,
        "preserved_table_counts_match": preserved_counts_match,
        "manifest_path": manifest_path,
    }
    result["post_plan"] = post_plan
    if not preserved_counts_match:
        raise RuntimeError(
            f"preserved table counts changed during derived reset; manifest: {manifest_path}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--skip-service-gate", action="store_true")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    try:
        result = await run(
            apply=bool(args.apply),
            confirm=str(args.confirm or ""),
            data_dir=args.data_dir,
            skip_service_gate=bool(args.skip_service_gate),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    finally:
        await close_db()
        session_module._engine = None
        session_module._sessionmaker = None


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
