#!/usr/bin/env python3
"""Run the controlled paper-resume entrypoint on the online platform."""

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
from scripts.start_phase3_paper_with_preflight import CONFIRMATION_PHRASE  # noqa: E402

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
        raise ValueError("online paper-resume output must be a JSON object")
    return payload


def build_remote_command(
    *,
    start_service: bool,
    confirmation: str,
    json_indent: int,
) -> str:
    argv = [
        ".venv/bin/python",
        "scripts/start_phase3_paper_with_preflight.py",
        "--stdout-only",
        "--json-indent",
        str(max(int(json_indent or 0), 0)),
    ]
    if start_service:
        argv.extend(
            [
                "--start-service",
                "--confirm-resume-paper",
                str(confirmation or ""),
            ]
        )
    return f"cd {_quote(REMOTE_APP_DIR)} && exec " + " ".join(_quote(item) for item in argv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-service", action="store_true")
    parser.add_argument(
        "--confirm-resume-paper",
        default="",
        help=f"Required token when starting: {CONFIRMATION_PHRASE}",
    )
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--json-indent", type=int, default=0)
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = build_remote_command(
        start_service=bool(args.start_service),
        confirmation=str(args.confirm_resume_paper or ""),
        json_indent=args.json_indent,
    )
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        output = run_remote_text(
            ssh,
            command,
            timeout=max(int(args.timeout or 1), 60),
            max_output_chars=400_000,
            check=True,
        )
    finally:
        ssh.close()
    try:
        payload = _decode_json(output)
    except (json.JSONDecodeError, ValueError) as exc:
        safe_print(output)
        raise SystemExit("online paper-resume command did not return JSON") from exc
    safe_print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=None if args.json_indent <= 0 else args.json_indent,
            sort_keys=True,
        )
    )
    if args.fail_on_blocked and payload.get("status") == "blocked":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
