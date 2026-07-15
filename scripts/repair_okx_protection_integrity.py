#!/usr/bin/env python3
"""Audit and repair OKX protection quantity coverage without changing prices."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from executor.okx_executor import OKXExecutor
from services.protection_order_integrity import audit_protection_order_integrity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("paper", "live"), default="paper")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-fingerprint", default="")
    return parser


async def _snapshot(executor: OKXExecutor) -> dict[str, Any]:
    positions = await executor.get_positions_strict(None)
    protection_orders = await executor.get_position_protection_orders(None)
    pending_orders = await executor.get_open_orders_strict(None)
    symbols = [str(position.get("symbol") or "") for position in positions]
    specs = await executor._native_facts_client().fetch_contract_specs(symbols=symbols)
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
    results: list[dict[str, Any]] = []
    for action in actions:
        action_name = str(action.get("action") or "")
        if action_name == "amend_size":
            response = await executor.amend_position_protection_size(
                inst_id=str(action.get("inst_id") or ""),
                algo_id=str(action.get("algo_id") or ""),
                contracts=float(action.get("new_contracts") or 0.0),
            )
        elif action_name == "cancel":
            response = await executor.cancel_position_protection_order(
                inst_id=str(action.get("inst_id") or ""),
                algo_id=str(action.get("algo_id") or ""),
            )
        else:
            raise RuntimeError(f"Unsupported protection repair action: {action_name}")
        results.append(
            {
                "action": action,
                "okx_code": response.get("code") if isinstance(response, dict) else None,
                "applied": True,
            }
        )
    return results


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


def main() -> None:
    args = _parser().parse_args()
    result = asyncio.run(_run(args))
    print(json.dumps(result, ensure_ascii=False, default=str))
    if args.apply and result.get("verified") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
