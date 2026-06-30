"""Build a read-only Profit-First 24h governance report."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stdout
import json
import sys
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
from services.profit_first_governance_report import (  # noqa: E402
    DEFAULT_GOVERNANCE_HOURS,
    DEFAULT_GOVERNANCE_LIMIT,
    ProfitFirstGovernanceReportService,
)

DEFAULT_REPORT_DIR = "profit_first_governance_reports"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_report_name(timestamp: str) -> str:
    return timestamp.replace(":", "").replace("-", "").replace("+", "Z").replace(".", "_")


def _report_output_dir(value: Path | None) -> Path:
    if value is not None:
        return value
    return settings.data_dir / DEFAULT_REPORT_DIR


def write_report(report: dict[str, Any], output_dir: Path, *, indent: int | None) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = str(report.get("generated_at") or _now_iso())
    report_path = output_dir / f"profit-first-governance-{_safe_report_name(timestamp)}.json"
    latest_path = output_dir / "latest.json"
    artifacts = {"report_path": str(report_path), "latest_path": str(latest_path)}
    report["report_artifacts"] = artifacts
    text = json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True)
    report_path.write_text(text + "\n", encoding="utf-8")
    latest_path.write_text(text + "\n", encoding="utf-8")
    return artifacts


async def collect_profit_first_governance_report(
    *,
    hours: int = DEFAULT_GOVERNANCE_HOURS,
    limit: int = DEFAULT_GOVERNANCE_LIMIT,
) -> dict[str, Any]:
    return await ProfitFirstGovernanceReportService().report(hours=hours, limit=limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-indent", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stdout-only", action="store_true")
    parser.add_argument("--hours", type=int, default=DEFAULT_GOVERNANCE_HOURS)
    parser.add_argument("--limit", type=int, default=DEFAULT_GOVERNANCE_LIMIT)
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="Return exit code 2 if the governance report is unavailable or incomplete.",
    )
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    indent = None if int(args.json_indent or 0) <= 0 else int(args.json_indent)
    with redirect_stdout(sys.stderr):
        try:
            report = await collect_profit_first_governance_report(
                hours=int(args.hours or DEFAULT_GOVERNANCE_HOURS),
                limit=int(args.limit or DEFAULT_GOVERNANCE_LIMIT),
            )
        except Exception as exc:
            report = {
                "report_type": "profit_first_governance",
                "status": "unavailable",
                "generated_at": _now_iso(),
                "read_only": True,
                "audit_only": True,
                "live_mutation": False,
                "can_submit_orders": False,
                "can_start_trading_service": False,
                "error": safe_error_text(exc, limit=240),
            }
        if not args.stdout_only:
            try:
                write_report(report, _report_output_dir(args.output_dir), indent=indent)
            except Exception as exc:
                report["status"] = "unavailable"
                report["report_artifact_error"] = {
                    "code": "profit_first_governance_report_write_failed",
                    "message": safe_error_text(exc, limit=240),
                }
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    if args.fail_on_incomplete and report.get("status") in {"unavailable", "incomplete"}:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
