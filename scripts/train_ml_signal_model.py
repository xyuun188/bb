"""Train the local ML win-rate model from completed shadow backtests."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.ml_signal_service import (
    MIN_TRAINING_SAMPLES,
    TRAINING_SHADOW_SAMPLE_LIMIT,
    build_training_frame,
    count_shadow_training_rows,
    load_shadow_training_rows,
    shadow_training_quality_report,
    train_from_frame,
)
from services.shadow_training_quarantine import quarantine_dirty_shadow_samples


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Train local ML signal model")
    parser.add_argument(
        "--limit",
        type=int,
        default=TRAINING_SHADOW_SAMPLE_LIMIT,
        help="Max completed shadow samples to load",
    )
    parser.add_argument("--min-samples", type=int, default=MIN_TRAINING_SAMPLES)
    parser.add_argument("--skip-quarantine", action="store_true")
    args = parser.parse_args()

    quarantine_result = {
        "skipped": True,
        "reason": "skip_quarantine flag enabled",
    }
    if not args.skip_quarantine:
        quarantine_result = await quarantine_dirty_shadow_samples(
            batch_size=min(args.limit, 1000),
            max_batches=max((int(args.limit) + 999) // 1000, 1),
        )

    rows = await load_shadow_training_rows(limit=args.limit)
    quality_state = shadow_training_quality_report(rows)
    frame = build_training_frame(rows)
    completed_count = await count_shadow_training_rows()
    metadata = train_from_frame(
        frame,
        min_samples=args.min_samples,
        completed_sample_count=completed_count,
        training_quality_report=quality_state["quality_report"],
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({"training_quarantine": quarantine_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
