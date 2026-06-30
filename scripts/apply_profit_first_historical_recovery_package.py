"""Dry-run or explicitly apply Profit-First historical recovery raw patches."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_env_bootstrap import (  # noqa: E402
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

load_runtime_env_files(project_root=ROOT)
drop_privileges_to_runtime_user_if_needed(project_root=ROOT)

from config.settings import settings  # noqa: E402
from core.safe_output import safe_error_text  # noqa: E402
from db.session import get_session_ctx  # noqa: E402
from models.decision import AIDecision  # noqa: E402
from scripts.plan_profit_first_historical_recovery_package import (  # noqa: E402
    collect_historical_recovery_package,
)
from services.profit_first_historical_recovery_apply import (  # noqa: E402
    APPROVAL_TOKEN,
    build_historical_recovery_apply_plan,
    merge_raw_patch,
    validate_apply_request,
)

DEFAULT_BACKUP_DIR = "codex_backups/profit-first-historical-recovery"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _backup_dir(value: Path | None) -> Path:
    if value is not None:
        return value
    return settings.data_dir / DEFAULT_BACKUP_DIR


async def collect_apply_preview(
    *,
    entry_decision_ids: list[int],
    exit_decision_ids: list[int],
    order_ids: list[int],
    exchange_order_ids: list[str],
    use_current_blockers: bool,
    apply: bool,
    approval_token: str,
    backup_dir: Path | None = None,
) -> dict[str, Any]:
    package = await collect_historical_recovery_package(
        entry_decision_ids=entry_decision_ids,
        exit_decision_ids=exit_decision_ids,
        order_ids=order_ids,
        exchange_order_ids=exchange_order_ids,
        use_current_blockers=use_current_blockers,
    )
    allowed_decision_ids = _dedupe_ints([*entry_decision_ids, *exit_decision_ids])
    apply_plan = build_historical_recovery_apply_plan(
        package,
        allowed_decision_ids=allowed_decision_ids,
    )
    can_apply, apply_blockers = validate_apply_request(
        apply=bool(apply),
        approval_token=approval_token,
        allowed_decision_ids=allowed_decision_ids,
        applicable_count=int(apply_plan["summary"]["applicable_count"]),
    )
    result = {
        "report_type": "profit_first_historical_recovery_apply",
        "generated_at": _now_iso(),
        "status": "apply_ready" if can_apply else "dry_run",
        "dry_run": not bool(apply),
        "read_only": not bool(apply),
        "audit_only": not bool(apply),
        "mutates_database": False,
        "starts_trading_service": False,
        "submits_orders": False,
        "changes_model_routing": False,
        "changes_live_sizing": False,
        "live_mutation": False,
        "resume_allowed_by_this_apply": False,
        "approval_token_required": APPROVAL_TOKEN,
        "apply_requested": bool(apply),
        "can_apply": can_apply,
        "apply_blockers": apply_blockers,
        "package_summary": package.get("summary") or {},
        "targets": package.get("targets") or {},
        "apply_plan": apply_plan,
        "apply_policy": {
            "requires_backup": True,
            "requires_explicit_decision_id_allowlist": True,
            "requires_approval_token": True,
            "applies_only_ai_decision_raw_patches": True,
            "does_not_touch_orders_positions_ranking_or_okx": True,
            "post_apply_go_no_go_required": True,
        },
    }
    if can_apply:
        result["apply_result"] = await _apply_decision_raw_patches(
            apply_plan["applicable_items"],
            backup_dir=_backup_dir(backup_dir),
        )
        result["mutates_database"] = True
        result["read_only"] = False
        result["audit_only"] = False
        result["status"] = "applied"
    return result


async def _apply_decision_raw_patches(
    items: list[dict[str, Any]],
    *,
    backup_dir: Path,
) -> dict[str, Any]:
    decision_ids = _dedupe_ints([int(item["decision_id"]) for item in items])
    if not decision_ids:
        return {"applied": 0, "backup_path": None}
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"ai_decisions_before_profit_first_recovery_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    async with get_session_ctx() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(AIDecision).where(AIDecision.id.in_(decision_ids)).order_by(AIDecision.id.asc())
        )
        rows = list(result.scalars().all())
        by_id = {int(row.id): row for row in rows}
        backup_payload = {
            "generated_at": _now_iso(),
            "policy": {
                "source": "apply_profit_first_historical_recovery_package",
                "training_policy": "exclude_until_manual_trust",
            },
            "rows": [
                {
                    "id": int(row.id),
                    "symbol": row.symbol,
                    "action": row.action,
                    "analysis_type": row.analysis_type,
                    "raw_llm_response": row.raw_llm_response,
                }
                for row in rows
            ],
        }
        backup_path.write_text(
            json.dumps(_json_safe(backup_payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        applied: list[int] = []
        missing: list[int] = []
        for item in items:
            decision_id = int(item["decision_id"])
            row = by_id.get(decision_id)
            if row is None:
                missing.append(decision_id)
                continue
            raw = row.raw_llm_response if isinstance(row.raw_llm_response, dict) else {}
            row.raw_llm_response = merge_raw_patch(raw, item["proposed_raw_patch"])
            applied.append(decision_id)
        await session.flush()
    return {
        "applied": len(applied),
        "applied_decision_ids": applied,
        "missing_decision_ids": missing,
        "backup_path": str(backup_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-indent", type=int, default=2)
    parser.add_argument("--skip-current-blockers", action="store_true")
    parser.add_argument("--entry-decision-id", action="append", type=int, default=[])
    parser.add_argument("--exit-decision-id", action="append", type=int, default=[])
    parser.add_argument("--order-id", action="append", type=int, default=[])
    parser.add_argument("--exchange-order-id", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--approval-token", default="")
    parser.add_argument("--backup-dir", type=Path, default=None)
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    indent = None if int(args.json_indent or 0) <= 0 else int(args.json_indent)
    with redirect_stdout(sys.stderr):
        try:
            report = await collect_apply_preview(
                entry_decision_ids=[int(item) for item in args.entry_decision_id or []],
                exit_decision_ids=[int(item) for item in args.exit_decision_id or []],
                order_ids=[int(item) for item in args.order_id or []],
                exchange_order_ids=[
                    str(item).strip() for item in args.exchange_order_id or [] if str(item).strip()
                ],
                use_current_blockers=not bool(args.skip_current_blockers),
                apply=bool(args.apply),
                approval_token=str(args.approval_token or ""),
                backup_dir=args.backup_dir,
            )
        except Exception as exc:
            report = {
                "report_type": "profit_first_historical_recovery_apply",
                "status": "unavailable",
                "generated_at": _now_iso(),
                "dry_run": True,
                "read_only": True,
                "audit_only": True,
                "mutates_database": False,
                "starts_trading_service": False,
                "submits_orders": False,
                "changes_model_routing": False,
                "changes_live_sizing": False,
                "live_mutation": False,
                "resume_allowed_by_this_apply": False,
                "error": safe_error_text(exc, limit=240),
            }
    print(json.dumps(_json_safe(report), ensure_ascii=False, indent=indent, sort_keys=True))
    return 0


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


def _dedupe_ints(values: list[Any]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number <= 0 or number in seen:
            continue
        seen.add(number)
        result.append(number)
    return result


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
