#!/usr/bin/env python3
"""Run one online read/write settlement reconciliation pass.

This command only applies the existing fail-closed settlement rules to local
paper-mode projections using OKX-authoritative facts. It never places orders,
changes model routing, or enables live trading.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

REMOTE_APP_DIR = "/data/bb/app"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("paper", "live"), default="paper")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--lookback-hours", type=int, default=72)
    args = parser.parse_args()
    script = f"""
import asyncio, json
from pathlib import Path
from scripts.runtime_env_bootstrap import load_runtime_env_files, drop_privileges_to_runtime_user_if_needed
root = Path({REMOTE_APP_DIR!r})
load_runtime_env_files(project_root=root)
drop_privileges_to_runtime_user_if_needed(project_root=root)
from services.okx_position_settlement_sync import OkxPositionSettlementSyncService
async def main():
    result = await OkxPositionSettlementSyncService(
        mode={args.mode!r},
        limit={max(int(args.limit or 1), 1)!r},
        lookback_hours={max(int(args.lookback_hours or 1), 1)!r},
    ).sync_once()
    print(json.dumps(result, ensure_ascii=False, default=str))
asyncio.run(main())
"""
    command = (
        f"cd {shlex.quote(REMOTE_APP_DIR)} && runuser -u bb -- /bin/bash -lc "
        + shlex.quote(".venv/bin/python - <<'PY'\n" + script + "\nPY")
    )
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        safe_print(run_remote_text(ssh, command, timeout=240, max_output_chars=100_000))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
