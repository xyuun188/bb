#!/usr/bin/env python3
"""Collect one complete system-audit snapshot in an isolated process."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
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
from web_dashboard.api.system_audit import (  # noqa: E402
    SYSTEM_AUDIT_RUNNER_RESULT_PREFIX,
    collect_system_audit_status,
)


async def run_once(
    *,
    record_history: bool,
    source: str,
    collector: Callable[..., Awaitable[dict[str, Any]]] = collect_system_audit_status,
) -> dict[str, Any]:
    try:
        payload = await collector(record_history=record_history, source=source)
        return {
            "ok": True,
            "checked_at": payload.get("checked_at"),
            "status": payload.get("status"),
        }
    finally:
        await close_db()


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="subprocess")
    parser.add_argument("--no-record-history", action="store_true")
    args = parser.parse_args()
    try:
        result = await run_once(
            record_history=not bool(args.no_record_history),
            source=str(args.source or "subprocess")[:80],
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        result = {
            "ok": False,
            "error": safe_error_text(exc, limit=500),
        }
        exit_code = 1
    else:
        exit_code = 0
    print(
        SYSTEM_AUDIT_RUNNER_RESULT_PREFIX + json.dumps(result, ensure_ascii=False, sort_keys=True)
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
