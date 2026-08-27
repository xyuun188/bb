"""Generate one cached training-effectiveness report without starting training."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

# Allow direct ``python scripts/...py`` execution to resolve repository modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from services.training_effectiveness_report import (
    TrainingEffectivenessReportService,
    build_input_fingerprint,
    load_cached_training_effectiveness_report,
    report_directory,
)

LOCK_NAME = "training_effectiveness_report.lock"
GENERATION_TIMEOUT_SECONDS = 60


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="baseline")
    parser.add_argument("--from", dest="start", required=False)
    parser.add_argument("--to", dest="end", required=False)
    parser.add_argument("--mode", default="all", choices=("paper", "live", "all"))
    parser.add_argument("--run-id")
    return parser.parse_args(argv)


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _acquire_lock(path: Path) -> int | None:
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None


def _release_lock(path: Path, descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        os.close(descriptor)
    finally:
        path.unlink(missing_ok=True)


def _freshness_inputs(args: argparse.Namespace) -> dict:
    return {
        "report_version": "2026-08-25.v1",
        "stage": args.stage,
        "from": args.start,
        "to": args.end,
        "mode": args.mode,
    }


async def _build(args: argparse.Namespace, fingerprint: str) -> dict:
    service = TrainingEffectivenessReportService()
    filters = {
        "mode": args.mode,
        "from": args.start,
        "to": args.end,
    }
    return await asyncio.wait_for(
        service.build(filters=filters, run_id=args.run_id, input_fingerprint=fingerprint),
        timeout=GENERATION_TIMEOUT_SECONDS,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = report_directory(settings.data_dir)
    lock_path = Path(settings.data_dir) / LOCK_NAME
    fingerprint = build_input_fingerprint(_freshness_inputs(args))
    latest = load_cached_training_effectiveness_report(data_dir=settings.data_dir)
    if latest.get("status") == "complete" and latest.get("input_fingerprint") == fingerprint:
        print(json.dumps({"status": "cached", "report_id": latest.get("report_id"), "input_fingerprint": fingerprint}))
        return 0
    descriptor = _acquire_lock(lock_path)
    if descriptor is None:
        print(json.dumps({"status": "already_running", "input_fingerprint": fingerprint}))
        return 0
    started = time.monotonic()
    try:
        try:
            report = asyncio.run(_build(args, fingerprint))
        except asyncio.TimeoutError:
            now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            report = {
                "report_version": "2026-08-25.v1",
                "report_id": f"te-{fingerprint[7:19]}",
                "generated_at": now,
                "data_cutoff_at": args.end or now,
                "status": "partial",
                "input_fingerprint": fingerprint,
                "run": {"run_id": args.run_id or fingerprint[7:19], "stage": args.stage},
                "versions": {}, "filters": _freshness_inputs(args), "metrics": {},
                "cost_attribution": {"gross_pnl": 0, "fee": 0, "slippage": 0, "funding_fee": 0, "fee_after_net_pnl": 0},
                "expert_contributions": [], "execution_funnel": {}, "sample_quality": {},
                "conclusion": {"promotion_eligible": False, "blocking_reasons": ["generation_timeout"]},
                "freshness": {"state": "timeout", "is_stale": True},
            }
        report.setdefault("run", {})["stage"] = args.stage
        report.setdefault("run", {})["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report_path = root / f"{report['report_id']}.json"
        _atomic_write(report_path, report)
        _atomic_write(root / "latest.json", report)
        print(json.dumps({"status": "written", "report_id": report.get("report_id"), "path": str(report_path), "input_fingerprint": fingerprint}))
        return 0
    finally:
        _release_lock(lock_path, descriptor)


if __name__ == "__main__":
    sys.exit(main())
