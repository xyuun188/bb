"""Generate one cached training-effectiveness report without starting training."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

# Allow direct ``python scripts/...py`` execution to resolve repository modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from services.training_effectiveness_report import (  # noqa: E402
    TRAINING_EFFECTIVENESS_REPORT_VERSION,
    TrainingEffectivenessReportService,
    build_generation_failed_report,
    build_input_fingerprint,
    generation_failure_report_path,
    load_cached_training_effectiveness_report,
    report_directory,
)  # noqa: E402

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
        "report_version": TRAINING_EFFECTIVENESS_REPORT_VERSION,
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
        generation_failed = False
        try:
            report = asyncio.run(_build(args, fingerprint))
        except TimeoutError:
            generation_failed = True
            report = build_generation_failed_report(
                filters=_freshness_inputs(args),
                run_id=args.run_id,
                input_fingerprint=fingerprint,
                error_code="generation_timeout",
                stage=args.stage,
            )
        report.setdefault("run", {})["stage"] = args.stage
        report.setdefault("run", {})["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report_path = root / f"{report['report_id']}.json"
        _atomic_write(report_path, report)
        if generation_failed:
            failure_path = generation_failure_report_path(
                data_dir=settings.data_dir,
                mode=args.mode,
            )
            _atomic_write(failure_path, report)
            print(
                json.dumps(
                    {
                        "status": "generation_failed",
                        "report_id": report.get("report_id"),
                        "path": str(report_path),
                        "failure_path": str(failure_path),
                        "latest_preserved": True,
                        "input_fingerprint": fingerprint,
                    }
                )
            )
        else:
            _atomic_write(root / "latest.json", report)
            generation_failure_report_path(
                data_dir=settings.data_dir,
                mode=args.mode,
            ).unlink(missing_ok=True)
            print(json.dumps({"status": "written", "report_id": report.get("report_id"), "path": str(report_path), "input_fingerprint": fingerprint}))
        return 0
    finally:
        _release_lock(lock_path, descriptor)


if __name__ == "__main__":
    sys.exit(main())
