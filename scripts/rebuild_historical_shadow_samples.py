#!/usr/bin/env python3
"""Rebuild matured shadow labels from preserved decisions and 1m K-line facts."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_env_bootstrap import (  # noqa: E402
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

load_runtime_env_files(project_root=ROOT)
drop_privileges_to_runtime_user_if_needed(project_root=ROOT)

from db.session import close_db, init_db  # noqa: E402
from services.historical_shadow_rebuild import (  # noqa: E402
    DEFAULT_HISTORICAL_HORIZONS_MINUTES,
    rebuild_historical_shadow_samples,
)
from services.training_epoch import load_training_data_start  # noqa: E402


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="")
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument(
        "--horizons",
        default=",".join(str(item) for item in DEFAULT_HISTORICAL_HORIZONS_MINUTES),
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    await init_db()
    try:
        since = _timestamp(args.since) if args.since else load_training_data_start()
        horizons = tuple(
            int(item.strip())
            for item in str(args.horizons).split(",")
            if item.strip()
        )
        result = await rebuild_historical_shadow_samples(
            since=since,
            horizons_minutes=horizons,
            max_decisions=args.max_decisions,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        await close_db()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
