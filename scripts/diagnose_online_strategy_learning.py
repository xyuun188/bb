from __future__ import annotations

import asyncio
import json
import shlex
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text


REMOTE_CODE = r'''
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

APP_ROOT = Path("/data/bb/app")
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

pid_text = subprocess.check_output(
    ["systemctl", "show", "--property=MainPID", "--value", "bb-dashboard.service"],
    text=True,
).strip()
pid = int(pid_text or "0")
for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
    key, separator, value = item.partition(b"=")
    if separator and key:
        os.environ[key.decode("utf-8", errors="surrogateescape")] = value.decode(
            "utf-8", errors="surrogateescape"
        )

from services.strategy_learning import StrategyLearningService
from services.authoritative_trade_outcome import load_authoritative_trade_outcomes
from web_dashboard.api.dashboard import (
    _strategy_learning_watermark_for_request,
    get_strategy_learning,
)


async def main():
    since = datetime.now(UTC) - timedelta(hours=168)
    started = time.monotonic()
    try:
        watermark = await asyncio.wait_for(
            _strategy_learning_watermark_for_request(selected_mode="paper", since=since),
            timeout=30,
        )
        watermark_result = {
            "elapsed": round(time.monotonic() - started, 3),
            "length": len(watermark),
            "tail": [str(value) for value in watermark[-4:]],
        }
    except Exception as exc:
        watermark_result = {
            "elapsed": round(time.monotonic() - started, 3),
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }
    started = time.monotonic()
    try:
        outcomes = await load_authoritative_trade_outcomes(
            mode="paper",
            since=since,
            limit=500,
            compact=True,
        )
        outcome_result = {
            "elapsed": round(time.monotonic() - started, 3),
            "count": len(outcomes),
            "complete": sum(bool(item.get("outcome_complete")) for item in outcomes),
            "trusted": sum(bool(item.get("trade_fact_trusted")) for item in outcomes),
            "modes": sorted({str(item.get("execution_mode") or "") for item in outcomes}),
            "sample_keys": sorted(
                set().union(*(item.keys() for item in outcomes[:1]))
            )[:30]
            if outcomes
            else [],
        }
    except Exception as exc:
        outcome_result = {
            "elapsed": round(time.monotonic() - started, 3),
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }
    started = time.monotonic()
    try:
        payload = await get_strategy_learning(mode="paper", detail="summary")
        strategy_result = {
            "elapsed": round(time.monotonic() - started, 3),
            "status": payload.get("status"),
            "stale": payload.get("stale"),
            "stale_reason": payload.get("stale_reason"),
            "snapshot_age_seconds": payload.get("snapshot_age_seconds"),
            "snapshot_saved_at": payload.get("snapshot_saved_at"),
            "checked_at": payload.get("checked_at"),
            "schedule_mode": (payload.get("schedule") or {}).get("scheduler_mode"),
            "schedule_reason": (payload.get("schedule") or {}).get("reason"),
            "feedback_generated_at": (payload.get("feedback") or {}).get("generated_at"),
            "authoritative_source": (
                ((payload.get("schedule") or {}).get("current_production_strategy") or {})
                .get("data_sources", {})
                .get("authoritative_trade_outcome")
            ),
            "trade_fact_quarantine": (payload.get("feedback") or {}).get(
                "trade_fact_quarantine"
            ),
            "problems": (payload.get("feedback") or {}).get("problems"),
        }
    except Exception as exc:
        strategy_result = {
            "elapsed": round(time.monotonic() - started, 3),
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }
    print(
        json.dumps(
            {
                "watermark": watermark_result,
                "authoritative_outcomes": outcome_result,
                "strategy": strategy_result,
            },
            ensure_ascii=False,
        )
    )


asyncio.run(main())
'''


def main() -> int:
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        command = (
            "cd /data/bb/app && "
            "runuser -u bb -- /bin/bash -lc "
            + shlex.quote(
                "export DATABASE_URL='postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql'; "
                + ".venv/bin/python -c "
                + shlex.quote(REMOTE_CODE)
            )
        )
        output = run_remote_text(ssh, command, timeout=120, check=False, max_output_chars=100000)
    finally:
        ssh.close()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
