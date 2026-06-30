"""Build a read-only Phase 3 post-resume observation report."""

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

from scripts.runtime_env_bootstrap import (
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

load_runtime_env_files(project_root=ROOT)

from config.settings import settings
from core.safe_output import safe_error_text
from services.phase3_paper_resume_observation import Phase3PaperResumeObservationService

DEFAULT_REPORT_DIR = "phase3_paper_resume_observation_reports"


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
    report_path = output_dir / f"phase3-paper-resume-observation-{_safe_report_name(timestamp)}.json"
    latest_path = output_dir / "latest.json"
    artifacts = {"report_path": str(report_path), "latest_path": str(latest_path)}
    report["report_artifacts"] = artifacts
    text = json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True)
    report_path.write_text(text + "\n", encoding="utf-8")
    latest_path.write_text(text + "\n", encoding="utf-8")
    return artifacts


async def collect_phase3_paper_resume_observation(
    *,
    observation_hours: int = 2,
    min_created_shadow_samples: int = 5,
    min_completed_shadow_samples: int = 1,
    report_max_age_seconds: int = 7200,
) -> dict[str, Any]:
    return await Phase3PaperResumeObservationService(
        observation_hours=observation_hours,
        min_created_shadow_samples=min_created_shadow_samples,
        min_completed_shadow_samples=min_completed_shadow_samples,
        report_max_age_seconds=report_max_age_seconds,
    ).report()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-hours", type=int, default=2)
    parser.add_argument("--min-created-shadow-samples", type=int, default=5)
    parser.add_argument("--min-completed-shadow-samples", type=int, default=1)
    parser.add_argument("--report-max-age-seconds", type=int, default=7200)
    parser.add_argument("--json-indent", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stdout-only", action="store_true")
    parser.add_argument("--fail-on-critical", action="store_true")
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    drop_privileges_to_runtime_user_if_needed(project_root=ROOT)
    report = await collect_phase3_paper_resume_observation(
        observation_hours=max(int(args.observation_hours or 1), 1),
        min_created_shadow_samples=max(int(args.min_created_shadow_samples or 0), 0),
        min_completed_shadow_samples=max(int(args.min_completed_shadow_samples or 0), 0),
        report_max_age_seconds=max(int(args.report_max_age_seconds or 60), 60),
    )
    indent = None if int(args.json_indent or 0) <= 0 else int(args.json_indent)
    if not args.stdout_only:
        try:
            write_report(report, _report_output_dir(args.output_dir), indent=indent)
        except Exception as exc:
            report["status"] = "critical"
            report.setdefault("blockers", []).append(
                {
                    "code": "observation_report_write_failed",
                    "severity": "blocking",
                    "message": safe_error_text(exc, limit=240),
                }
            )
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    if args.fail_on_critical and report.get("status") == "critical":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
