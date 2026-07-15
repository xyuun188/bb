"""Build a read-only hard-gate report before Phase 3 paper trading resumes."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
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

# These imports must follow runtime environment loading because settings are read at import time.
from config.settings import settings  # noqa: E402
from core.safe_output import safe_error_text  # noqa: E402
from services.phase3_paper_resume_preflight import (  # noqa: E402
    Phase3PaperResumePreflightService,
)

DEFAULT_REPORT_DIR = "phase3_paper_resume_preflight_reports"


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
    timestamp = str(report.get("checked_at") or _now_iso())
    report_path = output_dir / f"phase3-paper-resume-preflight-{_safe_report_name(timestamp)}.json"
    latest_path = output_dir / "latest.json"
    artifacts = {"report_path": str(report_path), "latest_path": str(latest_path)}
    report["report_artifacts"] = artifacts
    text = json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True)
    report_path.write_text(text + "\n", encoding="utf-8")
    latest_path.write_text(text + "\n", encoding="utf-8")
    return artifacts


async def collect_phase3_paper_resume_preflight(
    *,
    okx_lookback_hours: int = 24,
    okx_limit: int = 120,
    okx_timeout_seconds: float = 5.0,
    model_server_timeout_seconds: int = 24,
    specialist_report_max_age_seconds: int = 7200,
) -> dict[str, Any]:
    return await Phase3PaperResumePreflightService(
        okx_lookback_hours=okx_lookback_hours,
        okx_limit=okx_limit,
        okx_timeout_seconds=okx_timeout_seconds,
        model_server_timeout_seconds=model_server_timeout_seconds,
        specialist_report_max_age_seconds=specialist_report_max_age_seconds,
    ).report()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--okx-lookback-hours", type=int, default=24)
    parser.add_argument("--okx-limit", type=int, default=120)
    parser.add_argument("--okx-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--model-server-timeout-seconds", type=int, default=24)
    parser.add_argument("--specialist-report-max-age-seconds", type=int, default=7200)
    parser.add_argument("--json-indent", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stdout-only", action="store_true")
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return exit code 2 when can_resume_paper=false.",
    )
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    drop_privileges_to_runtime_user_if_needed(project_root=ROOT)
    report = await collect_phase3_paper_resume_preflight(
        okx_lookback_hours=max(int(args.okx_lookback_hours or 1), 1),
        okx_limit=max(int(args.okx_limit or 1), 1),
        okx_timeout_seconds=max(float(args.okx_timeout_seconds or 0.5), 0.5),
        model_server_timeout_seconds=max(int(args.model_server_timeout_seconds or 1), 1),
        specialist_report_max_age_seconds=max(
            int(args.specialist_report_max_age_seconds or 60),
            60,
        ),
    )
    indent = None if int(args.json_indent or 0) <= 0 else int(args.json_indent)
    if not args.stdout_only:
        try:
            write_report(report, _report_output_dir(args.output_dir), indent=indent)
        except Exception as exc:
            report["status"] = "blocked"
            report["can_resume_paper"] = False
            report["report_artifact_error"] = {
                "code": "report_artifact_write_failed",
                "message": safe_error_text(exc, limit=240),
                "output_dir": str(_report_output_dir(args.output_dir)),
            }
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    if args.fail_on_blocked and not bool(report.get("can_resume_paper")):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
