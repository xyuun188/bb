#!/usr/bin/env python3
"""Run one local-ML auto-training check in an isolated database process."""

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

from core.safe_output import safe_error_text  # noqa: E402
from db.session import close_db  # noqa: E402
from services.ml_signal_service import MLSignalService  # noqa: E402


async def run_once(
    *,
    force: bool = False,
    service_factory: Callable[[], Any] = MLSignalService,
) -> dict[str, Any]:
    try:
        result = await service_factory().maybe_auto_train(force=force)
        return dict(result) if isinstance(result, dict) else {
            "trained": False,
            "reason": "invalid_training_response",
        }
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return {
            "trained": False,
            "reason": "error",
            "error": safe_error_text(exc, limit=500),
        }
    finally:
        await close_db()


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = await run_once(force=bool(args.force))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
