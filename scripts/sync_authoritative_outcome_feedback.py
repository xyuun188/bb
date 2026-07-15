"""Idempotently bind canonical outcomes to reflections and expert memories."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.remote_ssh import connect_remote_ssh, run_remote_text
from core.safe_output import safe_print
from services.authoritative_trade_outcome import load_authoritative_trade_outcomes
from services.expert_memory_service import ExpertMemoryService


async def sync_feedback(*, mode: str, apply: bool) -> dict[str, Any]:
    outcomes = await load_authoritative_trade_outcomes(mode=mode)
    trusted = [item for item in outcomes if item.get("settlement_fact_trusted") is True]
    if not apply:
        return {
            "status": "dry_run",
            "mode": mode,
            "outcome_count": len(outcomes),
            "trusted_outcome_count": len(trusted),
            "would_write": bool(trusted),
        }
    result = await ExpertMemoryService().backfill_trade_reflections(mode)
    return {
        "status": "completed" if result.get("status") == "completed" else "blocked",
        "mode": mode,
        "outcome_count": len(outcomes),
        "trusted_outcome_count": len(trusted),
        "backfill": result,
    }


def _online_report(*, mode: str, apply: bool) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    remote_args = [
        ".venv/bin/python",
        "scripts/sync_authoritative_outcome_feedback.py",
        "--mode",
        mode,
    ]
    if apply:
        remote_args.append("--apply")
    app_script = "\n".join(
        (
            "cd /data/bb/app",
            "export DATABASE_URL='postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql'",
            "exec " + " ".join(shlex.quote(value) for value in remote_args),
        )
    )
    ssh = connect_remote_ssh(root, timeout=20)
    try:
        output = run_remote_text(
            ssh,
            "runuser -u bb -- /bin/bash -lc " + shlex.quote(app_script),
            timeout=240,
            check=False,
        )
    finally:
        ssh.close()
    safe_print(output)
    try:
        return json.loads(output[output.find("{") :])
    except json.JSONDecodeError as exc:
        raise SystemExit("online outcome feedback sync did not return JSON") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("paper", "live"), default="paper")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    report = (
        _online_report(mode=args.mode, apply=args.apply)
        if args.online
        else asyncio.run(sync_feedback(mode=args.mode, apply=args.apply))
    )
    if not args.online:
        safe_print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if report.get("status") == "blocked":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
