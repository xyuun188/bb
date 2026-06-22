"""CLI wrapper for automatic dirty shadow-sample quarantine."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.shadow_training_quarantine import (  # noqa: E402
    QUARANTINE_STATUS as QUARANTINE_STATUS,
    note_with_quarantine_reason,
    quarantine_dirty_shadow_samples,
    shadow_quality_sample,
)

_quality_sample = shadow_quality_sample
_note_with_quarantine_reason = note_with_quarantine_reason


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quarantine dirty completed shadow samples from ML training."
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--oldest-first", action="store_true")
    parser.add_argument("--only-newer-than-id", type=int, default=None)
    args = parser.parse_args()
    result = asyncio.run(
        quarantine_dirty_shadow_samples(
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            dry_run=args.dry_run,
            newest_first=not args.oldest_first,
            only_newer_than_id=args.only_newer_than_id,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
