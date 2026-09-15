#!/usr/bin/env python3
"""Run the paper-resume preflight inside the online platform runtime.

The wrapper is read-only. It pins the audit to the online ``bb`` user,
PostgreSQL database, runtime environment, and loopback model tunnels so local
Windows state cannot be mistaken for deploy evidence.
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
        raise ValueError("online paper-resume preflight output must be a JSON object")
    return payload


def build_remote_command(*, json_indent: int) -> str:
    inner = (
        f"cd {_quote(REMOTE_APP_DIR)} && "
        "export DATABASE_URL='postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql' && "
        "exec .venv/bin/python scripts/run_phase3_paper_resume_preflight.py "
        f"--stdout-only --json-indent {max(int(json_indent or 0), 0)}"
    )
    return "runuser -u bb -- /bin/bash -lc " + _quote(inner)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--json-indent", type=int, default=0)
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        output = run_remote_text(
            ssh,
            build_remote_command(json_indent=args.json_indent),
            timeout=max(int(args.timeout or 1), 60),
            max_output_chars=300_000,
            check=True,
        )
    finally:
        ssh.close()
    try:
        payload = _decode_json(output)
    except (json.JSONDecodeError, ValueError) as exc:
        safe_print(output)
        raise SystemExit("online paper-resume preflight did not return JSON") from exc
    safe_print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=None if args.json_indent <= 0 else args.json_indent,
            sort_keys=True,
        )
    )
    if args.fail_on_blocked and payload.get("can_resume_paper") is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
