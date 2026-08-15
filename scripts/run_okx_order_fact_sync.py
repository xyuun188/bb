#!/usr/bin/env python3
"""Run one account-level OKX order-fact sync in an isolated process."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_env_bootstrap import (  # noqa: E402
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

load_runtime_env_files(project_root=ROOT)
drop_privileges_to_runtime_user_if_needed(project_root=ROOT)

from core.safe_output import safe_error_text  # noqa: E402
from db.session import close_db  # noqa: E402
from services.okx_order_fact_sync import (  # noqa: E402
    OKX_ORDER_FACT_SYNC_RESULT_PREFIX,
    OkxOrderFactSyncService,
)


async def run_once(
    *,
    mode: str,
    lookback_hours: int,
    limit: int,
    timeout_seconds: float,
    service_factory: Callable[..., Any] = OkxOrderFactSyncService,
) -> dict[str, Any]:
    try:
        result = await service_factory(
            mode=mode,
            lookback_hours=lookback_hours,
            limit=limit,
            timeout_seconds=timeout_seconds,
        ).sync()
        if isinstance(result, dict):
            return dict(result)
        return {
            "status": "deferred",
            "okx_pull_available": False,
            "error": "invalid OKX order fact sync response",
            "deferred_stages": ["isolated_process"],
        }
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return {
            "status": "deferred",
            "okx_pull_available": False,
            "error": safe_error_text(exc, limit=500),
            "deferred_stages": ["isolated_process"],
        }
    finally:
        await close_db()


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("paper", "live"), default="paper")
    parser.add_argument("--lookback-hours", type=int, default=24)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    result = await run_once(
        mode=str(args.mode),
        lookback_hours=max(int(args.lookback_hours), 1),
        limit=max(int(args.limit), 1),
        timeout_seconds=max(float(args.timeout_seconds), 0.5),
    )
    print(
        OKX_ORDER_FACT_SYNC_RESULT_PREFIX + json.dumps(result, ensure_ascii=False, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
