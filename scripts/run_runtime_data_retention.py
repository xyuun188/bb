#!/usr/bin/env python3
"""Report or compact old redundant runtime payloads without deleting facts."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
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

from config.settings import settings  # noqa: E402
from core.runtime_data_retention_contract import (  # noqa: E402
    RUNTIME_DATA_RETENTION_APPLY_CONFIRMATION,
)
from core.safe_output import safe_error_text  # noqa: E402
from db.session import close_db  # noqa: E402
from services.runtime_data_retention import (  # noqa: E402
    RuntimeDataRetentionPolicy,
    RuntimeDataRetentionService,
)

DEFAULT_REPORT_DIR = "runtime_data_retention_reports"
APPLY_CONFIRMATION = RUNTIME_DATA_RETENTION_APPLY_CONFIRMATION


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--confirm",
        default="",
        help=f"Required with --apply; must equal {APPLY_CONFIRMATION!r}.",
    )
    parser.add_argument("--decision-raw-days", type=int, default=14)
    parser.add_argument("--shadow-grace-days", type=int, default=2)
    parser.add_argument("--strategy-event-days", type=int, default=14)
    parser.add_argument("--keep-trainable-shadow-rows", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--max-rows-per-table", type=int, default=5_000)
    parser.add_argument("--batch-pause-seconds", type=float, default=0.05)
    parser.add_argument(
        "--skip-byte-estimate",
        action="store_true",
        help="Skip full TOAST payload-size scans during routine bounded apply runs.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stdout-only", action="store_true")
    parser.add_argument("--json-indent", type=int, default=2)
    args = parser.parse_args(argv)
    if args.apply and args.confirm != APPLY_CONFIRMATION:
        parser.error(f"--apply requires --confirm {APPLY_CONFIRMATION}")
    if not args.apply and args.confirm:
        parser.error("--confirm is only valid together with --apply")
    return args


def _safe_report_name(timestamp: str) -> str:
    return timestamp.replace(":", "").replace("-", "").replace("+", "Z").replace(".", "_")


def write_report(
    report: dict[str, Any],
    output_dir: Path,
    *,
    indent: int | None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = str(report.get("generated_at") or datetime.now(UTC).isoformat())
    report_path = output_dir / f"runtime-data-retention-{_safe_report_name(timestamp)}.json"
    latest_path = output_dir / "latest.json"
    artifacts = {"report_path": str(report_path), "latest_path": str(latest_path)}
    report["report_artifacts"] = artifacts
    payload = json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True) + "\n"
    report_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    return artifacts


async def run(args: argparse.Namespace) -> dict[str, Any]:
    policy = RuntimeDataRetentionPolicy(
        decision_raw_days=args.decision_raw_days,
        shadow_grace_days=args.shadow_grace_days,
        strategy_event_days=args.strategy_event_days,
        keep_trainable_shadow_rows=args.keep_trainable_shadow_rows,
        batch_size=args.batch_size,
        max_rows_per_table=args.max_rows_per_table,
        batch_pause_seconds=args.batch_pause_seconds,
    )
    return await RuntimeDataRetentionService(policy).run(
        apply=bool(args.apply),
        measure_reclaimable_bytes=not bool(args.skip_byte_estimate),
    )


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    indent = None if int(args.json_indent or 0) <= 0 else int(args.json_indent)
    try:
        report = await run(args)
    finally:
        await close_db()

    artifact_error: dict[str, str] | None = None
    if not args.stdout_only:
        output_dir = args.output_dir or settings.data_dir / DEFAULT_REPORT_DIR
        try:
            write_report(report, output_dir, indent=indent)
        except Exception as exc:
            artifact_error = {
                "code": "runtime_data_retention_report_write_failed",
                "message": safe_error_text(exc, limit=240),
                "output_dir": str(output_dir),
            }
            report["report_artifact_error"] = artifact_error
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    return 2 if artifact_error else 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
