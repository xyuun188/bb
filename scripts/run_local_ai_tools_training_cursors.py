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
    _completed_trade_sample_count,
)

CursorCounter = Callable[[], Awaitable[int]]


async def run_once(
    *,
    shadow_counter: CursorCounter = _completed_shadow_sample_count,
    trade_counter: CursorCounter = _completed_trade_sample_count,
) -> dict[str, Any]:
    try:
        shadow_count = await shadow_counter()
        trade_count = await trade_counter()
        return {
            "trained": False,
            "reason": "cursor_probe_complete",
            "completed_shadow_sample_count": int(shadow_count),
            "completed_trade_sample_count": int(trade_count),
            "training_process_isolated": True,
            "cursor_policy": "canonical_clean_training_view",
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
