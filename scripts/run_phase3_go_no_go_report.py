"""Build a read-only Phase 3 go/no-go report from system audit cards."""

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
sys.path.insert(0, str(ROOT))

from scripts.runtime_env_bootstrap import (  # noqa: E402
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

load_runtime_env_files(project_root=ROOT)
drop_privileges_to_runtime_user_if_needed(project_root=ROOT)

from config.settings import settings  # noqa: E402
from core.safe_output import safe_error_text  # noqa: E402
from web_dashboard.api.system_audit import collect_system_audit_status  # noqa: E402

DEFAULT_REPORT_DIR = "phase3_go_no_go_reports"
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
    checked_at = _parse_report_datetime(payload.get("checked_at") or payload.get("generated_at"))
    age_seconds = None
    if checked_at is not None:
        age_seconds = max((datetime.now(UTC) - checked_at).total_seconds(), 0.0)
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
    timestamp = str(report.get("checked_at") or _now_iso())
    report_path = output_dir / f"phase3-go-no-go-{_safe_report_name(timestamp)}.json"
    latest_path = output_dir / "latest.json"
    artifacts = {"report_path": str(report_path), "latest_path": str(latest_path)}
    report["report_artifacts"] = artifacts
    text = json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True)
    report_path.write_text(text + "\n", encoding="utf-8")
    latest_path.write_text(text + "\n", encoding="utf-8")
    return artifacts


async def collect_phase3_go_no_go_report() -> dict[str, Any]:
    audit = await collect_system_audit_status(record_history=False, source="phase3_go_no_go_report")
    cards = [card for card in audit.get("cards") or [] if isinstance(card, dict)]
    gate_card = next((card for card in cards if card.get("key") == "phase3_go_no_go"), {})
    gate_details = gate_card.get("details") if isinstance(gate_card.get("details"), dict) else {}
    return {
        "status": gate_details.get("status") or "missing",
        "checked_at": audit.get("checked_at") or _now_iso(),
        "read_only": True,
        "audit_only": True,
        "starts_trading_service": False,
        "submits_orders": False,
        "changes_model_routing": False,
        "overall_audit_status": audit.get("status"),
        "overall_summary": audit.get("summary") or {},
        "go_no_go_card_status": gate_card.get("status"),
        "go_no_go": gate_details,
        "root_causes": audit.get("root_causes") or [],
        "issue_ledger_summary": (audit.get("issue_ledger") or {}).get("summary") or {},
        "safety_note": (
            "Read-only Phase 3 go/no-go report; it does not start paper/live trading, "
            "submit orders, train artifacts, or change model routing."
        ),
    }


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
        help="Return exit code 2 when the total gate is blocked.",
    )
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    indent = None if int(args.json_indent or 0) <= 0 else int(args.json_indent)
    output_dir = _report_output_dir(args.output_dir)
    with redirect_stdout(sys.stderr):
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
                "status": "unavailable",
                "checked_at": _now_iso(),
                "read_only": True,
                "audit_only": True,
                "starts_trading_service": False,
                "submits_orders": False,
                "changes_model_routing": False,
                "report_source": "latest_persisted_unavailable",
            }
        if report is None:
            report = await collect_phase3_go_no_go_report()
        if not args.stdout_only:
            try:
                write_report(report, output_dir, indent=indent)
            except Exception as exc:
                report["status"] = "blocked"
                report["report_artifact_error"] = {
                    "code": "go_no_go_report_write_failed",
                    "message": safe_error_text(exc, limit=240),
                }
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    if args.fail_on_blocked and report.get("status") == "blocked":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
