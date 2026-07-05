"""Build a read-only Profit-First recovery repair/quarantine plan."""

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
from services.profit_first_recovery_repair_plan import (  # noqa: E402
    build_profit_first_recovery_repair_plan,
)
from web_dashboard.api.system_audit import collect_system_audit_status  # noqa: E402

DEFAULT_REPORT_DIR = "profit_first_recovery_repair_plans"
DEFAULT_LATEST_MAX_AGE_SECONDS = 60 * 60


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_report_name(timestamp: str) -> str:
    return timestamp.replace(":", "").replace("-", "").replace("+", "Z").replace(".", "_")


def _report_output_dir(value: Path | None) -> Path:
    if value is not None:
        return value
    return settings.data_dir / DEFAULT_REPORT_DIR


def _parse_report_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_latest_report(output_dir: Path, *, max_age_seconds: int) -> dict[str, Any] | None:
    latest_path = output_dir / "latest.json"
    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    generated_at = _parse_report_datetime(payload.get("generated_at") or payload.get("checked_at"))
    age_seconds = None
    if generated_at is not None:
        age_seconds = max((datetime.now(UTC) - generated_at).total_seconds(), 0.0)
        if age_seconds > max(60, int(max_age_seconds or DEFAULT_LATEST_MAX_AGE_SECONDS)):
            return None
    result = dict(payload)
    result["report_source"] = "latest_persisted"
    result["latest_report_path"] = str(latest_path)
    result["latest_report_age_seconds"] = (
        round(age_seconds, 3) if age_seconds is not None else None
    )
    return result


def write_report(report: dict[str, Any], output_dir: Path, *, indent: int | None) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = str(report.get("generated_at") or _now_iso())
    report_path = output_dir / f"profit-first-recovery-repair-plan-{_safe_report_name(timestamp)}.json"
    latest_path = output_dir / "latest.json"
    artifacts = {"report_path": str(report_path), "latest_path": str(latest_path)}
    report["report_artifacts"] = artifacts
    text = json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True)
    report_path.write_text(text + "\n", encoding="utf-8")
    latest_path.write_text(text + "\n", encoding="utf-8")
    return artifacts


async def collect_profit_first_recovery_repair_plan() -> dict[str, Any]:
    audit = await collect_system_audit_status(
        record_history=False,
        source="profit_first_recovery_repair_plan",
    )
    cards = [card for card in audit.get("cards") or [] if isinstance(card, dict)]
    recovery_card = next(
        (card for card in cards if str(card.get("key") or "") == "profit_first_recovery_blockers"),
        {},
    )
    recovery_details = (
        recovery_card.get("details") if isinstance(recovery_card.get("details"), dict) else {}
    )
    report = build_profit_first_recovery_repair_plan(recovery_details)
    report["overall_audit_status"] = audit.get("status")
    report["overall_summary"] = audit.get("summary") or {}
    report["recovery_blockers_card_status"] = recovery_card.get("status") or "missing"
    report["safety_note"] = (
        "Read-only Profit-First recovery repair plan; it does not write database history, "
        "start services, submit orders, change routing, or change sizing."
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-indent", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stdout-only", action="store_true")
    parser.add_argument("--prefer-latest", action="store_true")
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument(
        "--max-latest-age-seconds",
        type=int,
        default=DEFAULT_LATEST_MAX_AGE_SECONDS,
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return exit code 2 when the dry-run repair plan still blocks resume.",
    )
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    indent = None if int(args.json_indent or 0) <= 0 else int(args.json_indent)
    output_dir = _report_output_dir(args.output_dir)
    with redirect_stdout(sys.stderr):
        try:
            report = None
            if bool(getattr(args, "prefer_latest", False) or getattr(args, "latest_only", False)):
                report = _load_latest_report(
                    output_dir,
                    max_age_seconds=int(
                        getattr(args, "max_latest_age_seconds", DEFAULT_LATEST_MAX_AGE_SECONDS)
                        or DEFAULT_LATEST_MAX_AGE_SECONDS
                    ),
                )
            if report is None and bool(getattr(args, "latest_only", False)):
                report = {
                    "report_type": "profit_first_recovery_repair_plan",
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
                    "resume_allowed_by_this_plan": False,
                    "report_source": "latest_persisted_unavailable",
                }
            if report is None:
                report = await collect_profit_first_recovery_repair_plan()
        except Exception as exc:
            report = {
                "report_type": "profit_first_recovery_repair_plan",
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
                "resume_allowed_by_this_plan": False,
                "error": safe_error_text(exc, limit=240),
            }
        if not args.stdout_only:
            try:
                write_report(report, output_dir, indent=indent)
            except Exception as exc:
                report["status"] = "unavailable"
                report["report_artifact_error"] = {
                    "code": "profit_first_recovery_repair_plan_write_failed",
                    "message": safe_error_text(exc, limit=240),
                }
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    if args.fail_on_blocked and report.get("status") in {"blocked", "unavailable"}:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
