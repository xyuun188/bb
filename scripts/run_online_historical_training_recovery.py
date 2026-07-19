#!/usr/bin/env python3
"""Audit or apply deterministic historical OKX entry-decision links online."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402


def _remote_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-app-dir", default="/data/bb/app")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--decision-window-seconds", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-fingerprint", default="")
    args = parser.parse_args()
    if args.apply and not args.expected_fingerprint:
        parser.error("--apply requires --expected-fingerprint from a fresh dry-run")
    return args


def main() -> None:
    args = parse_args()
    remote_script = f"""
import asyncio
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from scripts.runtime_env_bootstrap import load_runtime_env_files, drop_privileges_to_runtime_user_if_needed

root = Path({args.remote_app_dir!r})
load_runtime_env_files(project_root=root)
drop_privileges_to_runtime_user_if_needed(project_root=root)

from scripts.repair_missing_position_links_from_okx_fills import (
    apply_existing_order_decision_link_plans,
    collect_existing_order_decision_link_plans,
)

async def main():
    plans = await collect_existing_order_decision_link_plans(
        days={max(int(args.days or 1), 1)!r},
        decision_window_seconds={max(int(args.decision_window_seconds or 1), 1)!r},
        limit={max(int(args.limit or 1), 1)!r},
    )
    canonical = [
        {{
            "position_id": plan.position_id,
            "order_id": plan.order_id,
            "exchange_order_id": plan.exchange_order_id,
            "decision_id": plan.decision_id,
            "order_decision_delta_seconds": plan.order_decision_delta_seconds,
        }}
        for plan in sorted(plans, key=lambda item: (item.exchange_order_id, item.decision_id))
    ]
    fingerprint = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected = {args.expected_fingerprint!r}
    apply_requested = {bool(args.apply)!r}
    if apply_requested and fingerprint != expected:
        raise SystemExit(
            json.dumps(
                {{
                    "status": "blocked",
                    "reason": "recovery_plan_fingerprint_changed",
                    "expected_fingerprint": expected,
                    "actual_fingerprint": fingerprint,
                    "plan_count": len(plans),
                }},
                ensure_ascii=False,
            )
        )
    result = await apply_existing_order_decision_link_plans(plans) if apply_requested else None
    print(
        json.dumps(
            {{
                "status": "applied" if apply_requested else "dry_run",
                "plan_count": len(plans),
                "plan_fingerprint": fingerprint,
                "max_order_decision_delta_seconds": max(
                    (float(plan.order_decision_delta_seconds or 0.0) for plan in plans),
                    default=0.0,
                ),
                "sample_plans": [asdict(plan) for plan in plans[:12]],
                "apply_result": result,
            }},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

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
