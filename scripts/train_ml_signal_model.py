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
    build_training_frame,
    count_shadow_training_rows,
    load_shadow_training_rows,
    train_from_frame,
)


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Train local ML signal model")
    parser.add_argument(
        "--limit", type=int, default=20000, help="Max completed shadow samples to load"
    )
    parser.add_argument("--min-samples", type=int, default=MIN_TRAINING_SAMPLES)
    args = parser.parse_args()

    rows = await load_shadow_training_rows(limit=args.limit)
    frame = build_training_frame(rows)
    completed_count = await count_shadow_training_rows()
    metadata = train_from_frame(
        frame,
        min_samples=args.min_samples,
        completed_sample_count=completed_count,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
