#!/usr/bin/env python3
"""Run the Phase 3 go/no-go audit inside the online platform runtime.

This wrapper is read-only. It executes the repository's canonical audit as the
``bb`` runtime user so local Windows credentials, proxies, and SQLite defaults
cannot be mistaken for the online PostgreSQL/platform state.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

REMOTE_APP_DIR = "/data/bb/app"


def _quote(value: str) -> str:
    return shlex.quote(value)


def _decode_json(output: str) -> dict:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        payload = None
        for line in reversed(str(output or "").splitlines()):
            candidate = line.strip()
            if not candidate.startswith("{"):
                continue
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                payload = parsed
                break
        if payload is None:
            raise
    if not isinstance(payload, dict):
        raise ValueError("online Phase 3 audit output must be a JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--json-indent", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inner = (
        f"cd {_quote(REMOTE_APP_DIR)} && "
        "export DATABASE_URL='postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql' && "
        "exec .venv/bin/python scripts/run_phase3_go_no_go_report.py "
        f"--stdout-only --json-indent {max(int(args.json_indent or 0), 0)}"
    )
    command = "runuser -u bb -- /bin/bash -lc " + _quote(inner)
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        output = run_remote_text(
            ssh,
            command,
            timeout=max(int(args.timeout or 1), 60),
            max_output_chars=200_000,
            check=True,
        )
    finally:
        ssh.close()
    try:
        payload = _decode_json(output)
    except (json.JSONDecodeError, ValueError) as exc:
        safe_print(output)
        raise SystemExit("online Phase 3 audit did not return JSON") from exc
    safe_print(json.dumps(payload, ensure_ascii=False, indent=None if args.json_indent <= 0 else args.json_indent, sort_keys=True))
    return 0 if payload.get("status") != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
