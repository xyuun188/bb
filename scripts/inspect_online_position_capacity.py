"""Read-only online position-capacity audit detail."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402


REMOTE_CODE = r"""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

APP_ROOT = Path("/data/bb/app")
sys.path.insert(0, str(APP_ROOT))
pid = int(
    subprocess.check_output(
        ["systemctl", "show", "--property=MainPID", "--value", "bb-dashboard.service"],
        text=True,
    ).strip()
    or "0"
)
for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
    key, separator, value = item.partition(b"=")
    if separator and key:
        os.environ[key.decode("utf-8", errors="surrogateescape")] = value.decode(
            "utf-8", errors="surrogateescape"
        )

from db.session import close_db
from services.position_capacity_release_audit import PositionCapacityReleaseAuditService


async def main():
    try:
        report = await PositionCapacityReleaseAuditService(
            lookback_hours=48,
            limit=5000,
            exhaustive=True,
        ).report(exhaustive=True)
        print(
            json.dumps(
                {
                    "checked_at": report.get("checked_at"),
                    "open_position_count": report.get("open_position_count"),
                    "open_position_group_count": report.get("open_position_group_count"),
                    "position_economics_incomplete_count": report.get(
                        "position_economics_incomplete_count"
                    ),
                    "position_economics_incomplete": report.get(
                        "position_economics_incomplete"
                    ),
                    "position_economics_pending_count": report.get(
                        "position_economics_pending_count"
                    ),
                    "capacity": report.get("capacity"),
                    "executed_dynamic_exit_contract_gap_count": report.get(
                        "executed_dynamic_exit_contract_gap_count"
                    ),
                    "coverage": report.get("coverage"),
                },
                ensure_ascii=False,
                default=str,
            )
        )
    finally:
        await close_db()


asyncio.run(main())
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    command = (
        "cd /data/bb/app && runuser -u bb -- /bin/bash -lc "
        + shlex.quote(".venv/bin/python -c " + shlex.quote(REMOTE_CODE))
    )
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        output = run_remote_text(
            ssh,
            command,
            timeout=180,
            check=False,
            max_output_chars=100_000,
        )
    finally:
        ssh.close()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
