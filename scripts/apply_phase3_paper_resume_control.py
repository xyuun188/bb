"""Apply the Phase 3 paper pause state with an explicit operator token.

This process is intentionally separate from the root-owned systemd launcher.
On the online host it drops to the ``bb`` runtime user before writing the
process-shared control file, so later dashboard and trading-worker writes keep
the correct ownership.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runtime_env_bootstrap import (  # noqa: E402
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

load_runtime_env_files(project_root=ROOT)

from core.trading_mode import TradingMode, TradingModeManager, mode_manager  # noqa: E402

CONFIRMATION_PHRASE = "CONFIRM_PHASE3_PAPER_RESUME"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def apply_phase3_paper_control_action(
    *,
    action: str,
    confirmation: str,
    manager: TradingModeManager = mode_manager,
) -> dict[str, Any]:
    selected_action = str(action or "").strip().lower()
    confirmed = str(confirmation or "").strip() == CONFIRMATION_PHRASE
    before = manager.get_state()
    blockers: list[dict[str, Any]] = []

    if selected_action not in {"resume", "pause"}:
        blockers.append(
            {
                "code": "invalid_control_action",
                "severity": "blocking",
                "message": "Paper control action must be resume or pause.",
            }
        )
    if not confirmed:
        blockers.append(
            {
                "code": "resume_confirmation_missing",
                "severity": "blocking",
                "message": f"Paper control changes require {CONFIRMATION_PHRASE}.",
            }
        )
    if before.get("mode") != TradingMode.PAPER.value:
        blockers.append(
            {
                "code": "execution_mode_not_paper",
                "severity": "blocking",
                "message": "This entrypoint refuses to change control state outside paper mode.",
                "evidence": {"mode": before.get("mode")},
            }
        )

    changed = False
    if not blockers:
        expected_paused = selected_action == "pause"
        if bool(before.get("paused")) != expected_paused:
            if expected_paused:
                await manager.pause()
            else:
                await manager.resume()
            changed = True

    after = manager.get_state()
    if not blockers and bool(after.get("paused")) != (selected_action == "pause"):
        blockers.append(
            {
                "code": "control_state_verification_failed",
                "severity": "blocking",
                "message": "Persisted paper control state did not match the requested action.",
                "evidence": {"after": after},
            }
        )

    return {
        "status": "blocked" if blockers else "ok",
        "checked_at": _now_iso(),
        "action": selected_action,
        "confirmation_present": confirmed,
        "changed": changed,
        "before": before,
        "after": after,
        "submits_orders": False,
        "changes_model_routing": False,
        "live_routing_enabled": False,
        "blockers": blockers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("resume", "pause"), required=True)
    parser.add_argument("--confirm-resume-paper", default="")
    parser.add_argument("--json-indent", type=int, default=2)
    parser.add_argument("--stdout-only", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    drop_privileges_to_runtime_user_if_needed(project_root=ROOT)
    report = await apply_phase3_paper_control_action(
        action=str(args.action),
        confirmation=str(args.confirm_resume_paper or ""),
    )
    indent = None if int(args.json_indent or 0) <= 0 else int(args.json_indent)
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    if args.fail_on_blocked and report.get("status") == "blocked":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
