#!/usr/bin/env python3
"""Run the online OKX order fact sync once and print a compact report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402


def _remote_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-app-dir", default="/data/bb/app")
    parser.add_argument("--mode", default="paper", choices=("paper", "live"))
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remote_script = f"""
import asyncio
import json
from pathlib import Path

from scripts.runtime_env_bootstrap import load_runtime_env_files, drop_privileges_to_runtime_user_if_needed

root = Path({args.remote_app_dir!r})
load_runtime_env_files(project_root=root)
drop_privileges_to_runtime_user_if_needed(project_root=root)

from services.okx_order_fact_sync import OkxOrderFactSyncService

async def main():
    report = await OkxOrderFactSyncService(mode={args.mode!r}).sync()
    status_path = root / "data" / "trading_runtime_status.json"
    status = {{}}
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            status = {{"status_read_error": str(exc)}}
    payload = {{
        "sync_report": {{
            "status": report.get("status"),
            "okx_pull_available": report.get("okx_pull_available"),
            "local_checked": report.get("local_checked"),
            "confirmed_count": report.get("confirmed_count"),
            "position_confirmed_count": report.get("position_confirmed_count"),
            "unverified_count": report.get("unverified_count"),
            "samples": report.get("samples", [])[:4],
        }},
        "runtime_okx_authoritative_sync": status.get("okx_authoritative_sync", {{}}),
    }}
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))

asyncio.run(main())
"""
    command = (
        f"cd {_remote_quote(args.remote_app_dir)} && "
        "PYBIN=python3; "
        "if [ -x .venv/bin/python ]; then PYBIN=.venv/bin/python; "
        "elif [ -x venv/bin/python ]; then PYBIN=venv/bin/python; fi; "
        "$PYBIN - <<'PY'\n"
        f"{remote_script}\nPY"
    )
    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        safe_print(run_remote_text(ssh, command, timeout=args.timeout, check=True))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
