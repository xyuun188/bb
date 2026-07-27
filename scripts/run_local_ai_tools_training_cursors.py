#!/usr/bin/env python3
"""Read Local AI training cursors in an isolated database process."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.safe_output import safe_error_text  # noqa: E402
from db.session import close_db  # noqa: E402
from scripts.train_local_ai_tools_models import (  # noqa: E402
    _completed_shadow_sample_count,
    _load_shadow_samples,
    _load_trade_samples,
)
from services.local_ai_training_contract import local_ai_training_cursor  # noqa: E402
from services.training_data_quality import annotate_training_payload  # noqa: E402

CursorCounter = Callable[[], Awaitable[int]]
SampleLoader = Callable[[], Awaitable[list[dict[str, Any]]]]


async def run_once(
    *,
    shadow_counter: CursorCounter = _completed_shadow_sample_count,
    shadow_loader: SampleLoader = _load_shadow_samples,
    trade_loader: SampleLoader = _load_trade_samples,
) -> dict[str, Any]:
    try:
        shadow_count, shadow_samples, trade_samples = await asyncio.gather(
            shadow_counter(),
            shadow_loader(),
            trade_loader(),
        )
        payload = annotate_training_payload(
            shadow_samples=shadow_samples,
            trade_samples=trade_samples,
            sequence_samples=[],
            text_sentiment_samples=[],
        )
        cursor = local_ai_training_cursor(
            shadow_samples=payload["shadow_samples"],
            trade_samples=payload["trade_samples"],
        )
        return {
            "trained": False,
            "reason": "cursor_probe_complete",
            "completed_shadow_sample_count": int(shadow_count),
            "completed_trade_sample_count": len(payload["trade_samples"]),
            **cursor,
            "training_process_isolated": True,
            "cursor_policy": "canonical_clean_independent_decision_group_view",
        }
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return {
            "trained": False,
            "reason": "error",
            "error": safe_error_text(exc, limit=500),
            "training_process_isolated": True,
        }
    finally:
        await close_db()


async def _main() -> int:
    result = await run_once()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
