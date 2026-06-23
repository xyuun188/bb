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


async def run_training(
    *,
    limit: int,
    min_samples: int,
    skip_quarantine: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    quarantine_result: dict[str, object] = {
        "skipped": True,
        "reason": "skip_quarantine flag enabled",
    }
    if dry_run:
        quarantine_result = {
            "skipped": True,
            "reason": "dry_run_no_quarantine_writes",
        }
    elif not skip_quarantine:
        quarantine_result = await quarantine_dirty_shadow_samples(
            batch_size=min(limit, 1000),
            max_batches=max((int(limit) + 999) // 1000, 1),
        )

    rows = await load_shadow_training_rows(limit=limit)
    quality_state = shadow_training_quality_report(rows)
    frame = build_training_frame(rows)
    completed_count = await count_shadow_training_rows()
    metadata = train_from_frame(
        frame,
        min_samples=min_samples,
        completed_sample_count=completed_count,
        training_quality_report=quality_state["quality_report"],
        persist_artifact=not dry_run,
    )
    return {
        "metadata": metadata,
        "training_quarantine": quarantine_result,
        "dry_run": dry_run,
        "frame_sample_count": int(len(frame)),
        "loaded_row_count": int(len(rows)),
        "completed_shadow_sample_count": int(completed_count),
    }


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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate training metrics without quarantining rows or writing model artifacts.",
    )
    args = parser.parse_args()

    result = await run_training(
        limit=args.limit,
        min_samples=args.min_samples,
        skip_quarantine=bool(args.skip_quarantine),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "training_quarantine": result["training_quarantine"],
                "dry_run": result["dry_run"],
                "frame_sample_count": result["frame_sample_count"],
                "loaded_row_count": result["loaded_row_count"],
                "completed_shadow_sample_count": result["completed_shadow_sample_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(_main())
