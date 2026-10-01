#!/usr/bin/env python3
"""Run the authoritative missing-position-link repair inside the online runtime."""

from __future__ import annotations

import argparse
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
    return shlex.quote(str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange-order-id", action="append", required=True)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--window-seconds", type=int, default=180)
    parser.add_argument("--decision-window-seconds", type=int, default=600)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--create-missing-order-rows", action="store_true")
    parser.add_argument("--link-existing-order-decisions", action="store_true")
    parser.add_argument("--link-additional-entry-orders", action="store_true")
    parser.add_argument("--create-linked-protection-fill-orders", action="store_true")
    parser.add_argument("--close-missing-exchange-open-position", action="store_true")
    parser.add_argument("--close-shared-open-position-fragments", action="store_true")
    parser.add_argument("--quarantine-missing-exchange-open-position", action="store_true")
    parser.add_argument("--reassign-mismatched-close-links", action="store_true")
    parser.add_argument("--repair-native-full-close-shared", action="store_true")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    order_ids = tuple(
        str(value).strip() for value in args.exchange_order_id if str(value).strip()
    )
    remote_args = [
        ".venv/bin/python",
        "scripts/repair_missing_position_links_from_okx_fills.py",
        "--days",
        str(max(int(args.days or 1), 1)),
        "--window-seconds",
        str(max(int(args.window_seconds or 1), 1)),
        "--decision-window-seconds",
        str(max(int(args.decision_window_seconds or 1), 1)),
        "--limit",
        str(max(int(args.limit or 1), 1)),
    ]
    for order_id in order_ids:
        remote_args.extend(["--exchange-order-id", order_id])
    for flag in (
        "create_missing_order_rows",
        "link_existing_order_decisions",
        "link_additional_entry_orders",
        "create_linked_protection_fill_orders",
        "close_missing_exchange_open_position",
        "close_shared_open_position_fragments",
        "quarantine_missing_exchange_open_position",
        "reassign_mismatched_close_links",
        "repair_native_full_close_shared",
    ):
        if getattr(args, flag):
            remote_args.append("--" + flag.replace("_", "-"))
    if args.apply:
        remote_args.append("--apply")
    command = (
        f"cd {_quote(REMOTE_APP_DIR)} && "
        "export DATABASE_URL='postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql' && "
        "runuser -u bb -- /bin/bash -lc "
        + _quote("exec " + " ".join(_quote(value) for value in remote_args))
    )
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        output = run_remote_text(
            ssh,
            command,
            timeout=240,
            max_output_chars=120_000,
            check=False,
        )
    finally:
        ssh.close()
    safe_print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
