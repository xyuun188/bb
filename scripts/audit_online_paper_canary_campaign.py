#!/usr/bin/env python3
"""Audit the accelerated OKX Demo canary sampling campaign online."""

# ruff: noqa: S608 - the remote template uses SQLAlchemy expressions, not raw SQL.

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
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remote_script = f"""
import asyncio
import json
from datetime import UTC
from pathlib import Path

from scripts.runtime_env_bootstrap import load_runtime_env_files, drop_privileges_to_runtime_user_if_needed

root = Path({args.remote_app_dir!r})
load_runtime_env_files(project_root=root)
drop_privileges_to_runtime_user_if_needed(project_root=root)

from sqlalchemy import select
from db.session import get_read_session_ctx
from models.decision import AIDecision
from services.paper_bootstrap_canary import (
    PAPER_BOOTSTRAP_AUTHORITATIVE_BASELINE_SAMPLES,
    PAPER_BOOTSTRAP_CAMPAIGN_START,
    PAPER_BOOTSTRAP_CAMPAIGN_VERSION,
    PAPER_BOOTSTRAP_CANARY_VERSION,
    PAPER_BOOTSTRAP_DAILY_LOSS_EQUITY_RISK,
    PAPER_BOOTSTRAP_EXPECTED_COMPLETION_RATE,
    PAPER_BOOTSTRAP_MAX_DAILY_ENTRIES,
    PAPER_BOOTSTRAP_MAX_OPEN_POSITIONS,
    PAPER_BOOTSTRAP_PORTFOLIO_EQUITY_RISK,
    PAPER_BOOTSTRAP_SINGLE_TRADE_EQUITY_RISK,
    PAPER_BOOTSTRAP_TARGET_AUTHORITATIVE_SAMPLES,
)

def obj(value):
    return value if isinstance(value, dict) else {{}}

async def run():
    async with get_read_session_ctx() as session:
        result = await session.execute(
            select(AIDecision)
            .where(
                AIDecision.is_paper.is_(True),
                AIDecision.created_at >= PAPER_BOOTSTRAP_CAMPAIGN_START.replace(tzinfo=None),
            )
            .order_by(AIDecision.created_at.desc(), AIDecision.id.desc())
            .limit({max(int(args.limit or 1), 1)!r})
        )
        decisions = list(result.scalars().all())
    rows = []
    for decision in decisions:
        raw = obj(decision.raw_llm_response)
        canary = obj(raw.get("paper_bootstrap_canary"))
        if canary.get("version") != PAPER_BOOTSTRAP_CANARY_VERSION:
            continue
        guard = obj(canary.get("runtime_guard"))
        sizing = obj(raw.get("profit_risk_sizing"))
        selected_observation = obj(canary.get("selected_observation"))
        rows.append({{
            "decision_id": int(decision.id),
            "created_at": decision.created_at,
            "symbol": decision.symbol,
            "persisted_action": decision.action,
            "candidate_action": canary.get("candidate_action") or canary.get("selected_side"),
            "authorized": canary.get("authorized") is True,
            "runtime_authorized": canary.get("runtime_preflight_authorized") is True,
            "was_executed": bool(decision.was_executed),
            "executed_at": decision.executed_at,
            "execution_price": decision.execution_price,
            "execution_reason": decision.execution_reason,
            "outcome": decision.outcome,
            "horizon_minutes": selected_observation.get("horizon_minutes"),
            "blocking_reasons": guard.get("blocking_reasons") or canary.get("runtime_preflight_blocking_reasons") or [],
            "daily_entry_count": guard.get("daily_entry_count"),
            "max_daily_entries": guard.get("max_daily_entries"),
            "open_position_count": guard.get("open_position_count"),
            "max_open_positions": guard.get("max_open_positions"),
            "completed_authoritative_sample_count": guard.get("completed_authoritative_sample_count"),
            "remaining_authoritative_sample_count": guard.get("remaining_authoritative_sample_count"),
            "campaign_completed_sample_count": guard.get("campaign_completed_sample_count"),
            "sizing_version": sizing.get("contract_version"),
            "sizing_eligible": sizing.get("production_eligible") is True,
            "sizing_reasons": sizing.get("reasons") or [],
        }})
    completed = sum(bool(row.get("outcome")) for row in rows)
    current_count = PAPER_BOOTSTRAP_AUTHORITATIVE_BASELINE_SAMPLES + completed
    payload = {{
        "status": "ok",
        "read_only": True,
        "campaign_version": PAPER_BOOTSTRAP_CAMPAIGN_VERSION,
        "canary_version": PAPER_BOOTSTRAP_CANARY_VERSION,
        "campaign_start": PAPER_BOOTSTRAP_CAMPAIGN_START.astimezone(UTC).isoformat(),
        "baseline_authoritative_sample_count": PAPER_BOOTSTRAP_AUTHORITATIVE_BASELINE_SAMPLES,
        "target_authoritative_sample_count": PAPER_BOOTSTRAP_TARGET_AUTHORITATIVE_SAMPLES,
        "campaign_candidate_count": len(rows),
        "campaign_runtime_authorized_count": sum(row["runtime_authorized"] for row in rows),
        "campaign_executed_count": sum(row["was_executed"] for row in rows),
        "campaign_completed_count": completed,
        "current_authoritative_sample_count": current_count,
        "remaining_authoritative_sample_count": max(PAPER_BOOTSTRAP_TARGET_AUTHORITATIVE_SAMPLES - current_count, 0),
        "sampling_policy": {{
            "expected_completion_rate": PAPER_BOOTSTRAP_EXPECTED_COMPLETION_RATE,
            "max_daily_entries": PAPER_BOOTSTRAP_MAX_DAILY_ENTRIES,
            "max_open_positions": PAPER_BOOTSTRAP_MAX_OPEN_POSITIONS,
            "single_trade_equity_risk": PAPER_BOOTSTRAP_SINGLE_TRADE_EQUITY_RISK,
            "portfolio_equity_risk": PAPER_BOOTSTRAP_PORTFOLIO_EQUITY_RISK,
            "daily_loss_equity_risk": PAPER_BOOTSTRAP_DAILY_LOSS_EQUITY_RISK,
        }},
        "latest_candidates": rows[:20],
    }}
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))

asyncio.run(run())
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
        safe_print(
            run_remote_text(
                ssh,
                command,
                timeout=max(int(args.timeout or 1), 1),
                check=True,
            )
        )
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
