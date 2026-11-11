"""Requalify only shadow labels blocked by the fixed legacy market-fact fingerprint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = (ROOT / "data" / "profitability_remediation").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.shadow_training_quarantine import (  # noqa: E402
    requalify_legacy_market_fact_fingerprint_shadow_samples,
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
        help="Write only currently eligible rows back to completed status.",
    )
    parser.add_argument(
        "--expected-eligible-count",
        type=int,
        help="Required with --apply; must match a complete read-only requalification.",
    )
    parser.add_argument(
        "--expected-eligible-ids-sha256",
        help="Required with --apply; must match the complete read-only ID fingerprint.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON evidence path under data/profitability_remediation.",
    )
    args = parser.parse_args()
    if args.apply and (
        args.after_id is None
        or args.through_id is None
        or args.expected_eligible_count is None
        or args.expected_eligible_ids_sha256 is None
    ):
        parser.error(
            "--apply requires the ID range, eligible count, and eligible-ID SHA-256"
        )
    if args.after_id is not None and args.through_id is not None:
        if args.after_id >= args.through_id:
            parser.error("--after-id must be less than --through-id")
    kwargs = {
        "batch_size": max(1, min(args.batch_size, 1000)),
        "max_batches": max(1, min(args.max_batches, 300)),
        "dry_run": not args.apply,
        "only_newer_than_id": args.after_id,
        "only_up_to_id": args.through_id,
        "expected_eligible_count": args.expected_eligible_count,
        "expected_eligible_ids_sha256": args.expected_eligible_ids_sha256,
    }
    if args.online:
        from core.remote_ssh import connect_remote_ssh
        from scripts import audit_online_market_hold_economics as transport

        transport.REMOTE_SCRIPT = (
            "import asyncio,json\n"
            "from services.shadow_training_quarantine import "
            "requalify_legacy_market_fact_fingerprint_shadow_samples\n"
            f"kwargs=json.loads({json.dumps(kwargs)!r})\n"
            "result=asyncio.run("
            "requalify_legacy_market_fact_fingerprint_shadow_samples(**kwargs))\n"
            "print(json.dumps(result,default=str))\n"
        )
        ssh = connect_remote_ssh(ROOT, timeout=25)
        try:
            report = transport._read_remote_report(ssh, 1, 1)
        finally:
            ssh.close()
    else:
        report = asyncio.run(
            requalify_legacy_market_fact_fingerprint_shadow_samples(**kwargs)
        )
    report["checked_at"] = datetime.now(UTC).isoformat()
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path = output_path.resolve()
        try:
            output_path.relative_to(EVIDENCE_ROOT)
        except ValueError:
            parser.error("--output must stay under data/profitability_remediation")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report["saved_report"] = str(output_path)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
