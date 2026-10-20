"""Run a bounded same-window model diagnostic without changing model pointers."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402

REMOTE_CODE = """
import asyncio
import json
from db.session import close_db
from services.model_forward_comparison import run_model_forward_comparison
from services.strategy_replay_governance import write_replay_governance
from config.settings import settings

async def main():
    try:
        report = await asyncio.wait_for(run_model_forward_comparison(), timeout=300)
        write_replay_governance(report, path=settings.data_dir / "strategy_replay/model_comparison.json")
        print(json.dumps(report, allow_nan=False))
    finally:
        await close_db()

asyncio.run(main())
"""


def main() -> int:
    command = [
        "systemd-run",
        "--quiet",
        "--wait",
        "--pipe",
        "--collect",
        "--property=WorkingDirectory=/data/bb/app",
        "--property=User=bb",
        "--property=EnvironmentFile=-/data/bb/app/.env",
        "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
        "--property=CPUQuota=50%",
        "--property=MemoryMax=2G",
        "--property=RuntimeMaxSec=330",
        "--property=KillMode=control-group",
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1",
        "--setenv=DATABASE_POOL_SIZE=2",
        "--setenv=DATABASE_MAX_OVERFLOW=0",
        "/data/bb/app/.venv/bin/python",
        "-c",
        REMOTE_CODE,
    ]
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        output = run_remote_text(ssh, shlex.join(command), timeout=360, max_output_chars=20000)
    finally:
        ssh.close()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
