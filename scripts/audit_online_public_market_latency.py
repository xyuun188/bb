"""Read bounded, redacted public-market latency aggregates from online logs."""

from __future__ import annotations

import argparse
import base64
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

REMOTE_SCRIPT = r'''
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from services.public_market_latency_audit import (
    MAX_BYTES_PER_LOG,
    MAX_LOG_FILES,
    summarize_public_market_latency_logs,
)

MINUTES = __MINUTES__
checked_at = datetime.now(UTC)
since = checked_at - timedelta(minutes=MINUTES)
tails = []
journal_error = None
try:
    journal = subprocess.run(
        [
            "journalctl",
            "-u",
            "bb-paper-trading.service",
            "--since",
            since.isoformat(),
            "--until",
            checked_at.isoformat(),
            "-n",
            "5000",
            "-o",
            "json",
            "--no-pager",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    tails.append(("journald", journal.stdout))
except Exception as exc:
    journal_error = f"{type(exc).__name__}: {str(exc)[:160]}"

for index in range(MAX_LOG_FILES):
    path = Path("/data/bb/app/data/trading.log")
    if index:
        path = Path(f"/data/bb/app/data/trading.log.{index}")
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(size - MAX_BYTES_PER_LOG, 0))
            content = handle.read(MAX_BYTES_PER_LOG)
    except FileNotFoundError:
        continue
    tails.append((str(path), content))

report = summarize_public_market_latency_logs(
    tails,
    since=since,
    until=checked_at,
)
report["journal_error"] = journal_error
report["checked_at"] = checked_at.isoformat()
print(json.dumps(report, ensure_ascii=True, allow_nan=False))
'''


def _build_remote_command(minutes: int, unit: str) -> str:
    rendered = REMOTE_SCRIPT.replace("__MINUTES__", str(minutes))
    payload = base64.b64encode(rendered.encode("utf-8")).decode("ascii")
    runner = f"import base64;exec(base64.b64decode({payload!r}))"
    return " ".join(
        (
            "systemd-run --quiet --wait --pipe --collect",
            f"--unit={shlex.quote(unit)}",
            "--property=WorkingDirectory=/data/bb/app",
            "--property=User=root",
            "--property=CPUQuota=25%",
            "--property=MemoryMax=128M",
            "--property=RuntimeMaxSec=30s",
            "--property=KillMode=control-group",
            "--property=Nice=10",
            "--property=EnvironmentFile=-/data/bb/app/.env",
            "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
            "/data/bb/app/.venv/bin/python -c",
            shlex.quote(runner),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=int, default=45)
    args = parser.parse_args()
    minutes = max(1, min(int(args.minutes), 240))
    checked_at = datetime.now(UTC)
    unit = f"bb-market-latency-audit-{checked_at.strftime('%Y%m%dT%H%M%SZ')}"

    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        state = run_remote_text(
            ssh,
            f"systemctl show {shlex.quote(unit)} -p LoadState,ActiveState --value "
            "2>/dev/null || true",
            timeout=20,
            max_output_chars=500,
        ).strip().splitlines()
        if any(
            value in {"active", "activating", "reloading", "deactivating"}
            for value in state
        ):
            raise RuntimeError(f"matching audit unit already exists: {unit}")
        output = run_remote_text(
            ssh,
            _build_remote_command(minutes, unit),
            timeout=45,
            max_output_chars=8000,
        )
    finally:
        ssh.close()

    safe_print(output.strip())


if __name__ == "__main__":
    main()
