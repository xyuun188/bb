"""Requalify verified historical shadow labels previously blocked by quote quality."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.shadow_training_quarantine import (  # noqa: E402
    requalify_verified_historical_shadow_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--after-id", type=int)
    parser.add_argument("--through-id", type=int)
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use the configured SSH runtime identity instead of the local database.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write eligible rows back to completed status; default is read-only.",
    )
    args = parser.parse_args()
    kwargs = {
        "batch_size": max(1, min(args.batch_size, 1000)),
        "max_batches": max(1, min(args.max_batches, 300)),
        "dry_run": not args.apply,
        "only_newer_than_id": args.after_id,
        "only_up_to_id": args.through_id,
    }
    if args.online:
        from core.remote_ssh import connect_remote_ssh
        from scripts import audit_online_market_hold_economics as transport

        transport.REMOTE_SCRIPT = (
            "import asyncio,json\n"
            "from services.shadow_training_quarantine import "
            "requalify_verified_historical_shadow_samples\n"
            f"kwargs=json.loads({json.dumps(kwargs)!r})\n"
            "result=asyncio.run(requalify_verified_historical_shadow_samples(**kwargs))\n"
            "print(json.dumps(result,default=str))\n"
        )
        ssh = connect_remote_ssh(Path(__file__).resolve().parents[1], timeout=25)
        try:
            if not args.apply:
                report = transport._read_remote_report(ssh, 1, 1)
            else:
                cursor_id = args.after_id
                total = {
                    "dry_run": False,
                    "batch_size": kwargs["batch_size"],
                    "max_batches": kwargs["max_batches"],
                    "only_newer_than_id": args.after_id,
                    "only_up_to_id": args.through_id,
                    "scanned": 0,
                    "coverage_complete": False,
                    "eligible": 0,
                    "requalified": 0,
                    "last_scanned_id": cursor_id,
                    "candidate_ids_sample": [],
                    "skipped_reasons": {},
                    "chunks": 0,
                }
                batches_per_chunk = min(5, int(kwargs["max_batches"]))
                for _ in range(int(kwargs["max_batches"])):
                    chunk_kwargs = {
                        **kwargs,
                        "max_batches": batches_per_chunk,
                        "only_newer_than_id": cursor_id,
                    }
                    transport.REMOTE_SCRIPT = (
                        "import asyncio,json\n"
                        "from services.shadow_training_quarantine import "
                        "requalify_verified_historical_shadow_samples\n"
                        f"kwargs=json.loads({json.dumps(chunk_kwargs)!r})\n"
                        "result=asyncio.run("
                        "requalify_verified_historical_shadow_samples(**kwargs))\n"
                        "print(json.dumps(result,default=str))\n"
                    )
                    chunk = transport._read_remote_report(ssh, 1, 1)
                    total["chunks"] += 1
                    total["scanned"] += int(chunk.get("scanned") or 0)
                    total["eligible"] += int(chunk.get("eligible") or 0)
                    total["requalified"] += int(chunk.get("requalified") or 0)
                    total["last_scanned_id"] = chunk.get("last_scanned_id")
                    total["candidate_ids_sample"] = (
                        total["candidate_ids_sample"]
                        + list(chunk.get("candidate_ids_sample") or [])
                    )[:50]
                    for reason, count in (chunk.get("skipped_reasons") or {}).items():
                        total["skipped_reasons"][reason] = (
                            total["skipped_reasons"].get(reason, 0) + int(count)
                        )
                    cursor_id = chunk.get("last_scanned_id")
                    if chunk.get("coverage_complete") is True:
                        total["coverage_complete"] = True
                        break
                    if not chunk.get("scanned"):
                        total["coverage_complete"] = True
                        break
                report = total
        finally:
            ssh.close()
    else:
        report = asyncio.run(requalify_verified_historical_shadow_samples(**kwargs))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
