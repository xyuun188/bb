#!/usr/bin/env python3
"""Audit and repair OKX protection quantity coverage without changing prices."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from core.remote_ssh import connect_remote_ssh, run_remote_text
from core.safe_output import safe_print
from executor.okx_executor import OKXExecutor
from services.position_protection_rebalance import apply_protection_repair_actions
from services.protection_order_integrity import audit_protection_order_integrity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("paper", "live"), default="paper")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-fingerprint", default="")
    parser.add_argument("--online", action="store_true")
    return parser


async def _snapshot(executor: OKXExecutor) -> dict[str, Any]:
    positions = await executor.get_positions_strict(None)
    protection_orders = await executor.get_position_protection_orders(None)
    pending_orders = await executor.get_open_orders_strict(None)
    symbols = [str(position.get("symbol") or "") for position in positions]
    specs = await executor.get_contract_specs_strict(symbols)
    report = audit_protection_order_integrity(
        positions,
        protection_orders,
        pending_orders,
        specs,
        pending_snapshot_complete=True,
    )
    return {
        "report": report,
        "positions": positions,
        "protection_orders": protection_orders,
        "pending_orders": pending_orders,
        "contract_specs": specs,
    }


def _backup(payload: dict[str, Any]) -> Path:
    directory = settings.data_dir / "protection_order_repairs"
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"okx-protection-before-{timestamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


async def _apply_actions(
    executor: OKXExecutor,
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return await apply_protection_repair_actions(executor, actions)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    executor = OKXExecutor(mode=args.mode, load_markets_on_initialize=False)
    try:
        before = await _snapshot(executor)
        report = before["report"]
        output: dict[str, Any] = {
            "mode": args.mode,
            "apply": bool(args.apply),
            "before": report,
            "submits_orders": False,
            "changes_positions": False,
            "changes_protection_only": bool(args.apply),
        }
        if not args.apply:
            return output
        if not args.expected_fingerprint:
            raise RuntimeError("--apply requires --expected-fingerprint from a fresh dry-run")
        if args.expected_fingerprint != report.get("input_fingerprint"):
            raise RuntimeError("Protection inventory changed after dry-run; refusing stale repair plan")
        if report.get("repair_ready") is not True:
            raise RuntimeError(
                "Protection repair plan is blocked: "
                + ",".join(report.get("repair_blockers") or [])
            )
        backup_path = _backup(before)
        output["backup_path"] = str(backup_path)
        output["applied_actions"] = await _apply_actions(
            executor,
            list(report.get("repair_actions") or []),
        )
        after = await _snapshot(executor)
        output["after"] = after["report"]
        positions_unchanged = bool(
            report.get("position_inventory_fingerprint")
            == after["report"].get("position_inventory_fingerprint")
        )
        output["positions_unchanged"] = positions_unchanged
        output["verified"] = bool(
            positions_unchanged
            and
            not after["report"].get("missing_keys")
            and not after["report"].get("orphan_keys")
            and not after["report"].get("coverage_mismatches")
            and not after["report"].get("invalid_orders")
        )
        output["verification_errors"] = [
            reason
            for reason, failed in (
                ("positions_changed_during_protection_repair", not positions_unchanged),
                ("missing_protection_remaining", bool(after["report"].get("missing_keys"))),
                ("orphan_protection_remaining", bool(after["report"].get("orphan_keys"))),
                ("coverage_mismatch_remaining", bool(after["report"].get("coverage_mismatches"))),
                ("invalid_protection_remaining", bool(after["report"].get("invalid_orders"))),
            )
            if failed
        ]
        return output
    finally:
        await executor.shutdown()


def _run_online(args: argparse.Namespace) -> dict[str, Any]:
    remote_args = [
        ".venv/bin/python",
        "scripts/repair_okx_protection_integrity.py",
        "--mode",
        args.mode,
    ]
    if args.apply:
        remote_args.append("--apply")
        remote_args.extend(("--expected-fingerprint", args.expected_fingerprint))
    app_script = "\n".join(
        (
            "cd /data/bb/app",
            "export DATABASE_URL='postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql'",
            "exec " + " ".join(shlex.quote(value) for value in remote_args),
        )
    )
    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        output = run_remote_text(
            ssh,
            "runuser -u bb -- /bin/bash -lc " + shlex.quote(app_script),
            timeout=180,
            check=True,
        )
    finally:
        ssh.close()
    safe_print(output)
    json_lines = [line for line in output.splitlines() if line.lstrip().startswith("{")]
    if not json_lines:
        raise RuntimeError("Online protection repair did not return a JSON report")
    return json.loads(json_lines[-1])


def main() -> None:
    args = _parser().parse_args()
    result = _run_online(args) if args.online else asyncio.run(_run(args))
    if not args.online:
        safe_print(json.dumps(result, ensure_ascii=False, default=str))
    if args.apply and result.get("verified") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
