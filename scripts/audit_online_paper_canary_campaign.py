#!/usr/bin/env python3
"""Audit current normal paper sampling and trading activity online."""

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


def _remote_script(*, remote_app_dir: str, hours: int, limit: int) -> str:
    return f"""
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.runtime_env_bootstrap import (
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

root = Path({remote_app_dir!r})
load_runtime_env_files(project_root=root)
drop_privileges_to_runtime_user_if_needed(project_root=root)

from sqlalchemy import select
from db.session import get_read_session_ctx
from models.decision import AIDecision
from services.ml_signal_service import load_authoritative_trade_training_samples
from services.normal_paper_trade import (
    NORMAL_PAPER_TRADE_LEVERAGE_POLICY,
    NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION,
    NORMAL_PAPER_TRADE_VERSION,
    normal_paper_trade_contract_reasons,
)

WINDOW_HOURS = {max(int(hours), 1)!r}
LIMIT = {max(int(limit), 1)!r}


def obj(value):
    return value if isinstance(value, dict) else {{}}


async def run():
    now = datetime.now(UTC)
    window_start = now - timedelta(hours=WINDOW_HOURS)
    async with get_read_session_ctx() as session:
        result = await session.execute(
            select(AIDecision)
            .where(
                AIDecision.is_paper.is_(True),
                AIDecision.created_at >= window_start.replace(tzinfo=None),
                AIDecision.raw_llm_response.is_not(None),
            )
            .order_by(AIDecision.created_at.desc(), AIDecision.id.desc())
        )
        decisions = list(result.scalars().all())

    rows = []
    for decision in decisions:
        raw = obj(decision.raw_llm_response)
        contract = obj(raw.get("normal_paper_trade"))
        if not contract:
            continue
        reasons = normal_paper_trade_contract_reasons(contract)
        rows.append({{
            "decision_id": int(decision.id),
            "created_at": decision.created_at,
            "symbol": decision.symbol,
            "persisted_action": decision.action,
            "contract_version": contract.get("version"),
            "contract_valid": not reasons,
            "contract_reasons": reasons,
            "authorized": contract.get("authorized") is True,
            "selected_side": contract.get("side"),
            "selection_reason": contract.get("selection_reason"),
            "expected_net_return_pct": contract.get("expected_net_return_pct"),
            "objective_net_return_pct": contract.get("objective_net_return_pct"),
            "loss_probability": contract.get("loss_probability"),
            "was_executed": bool(decision.was_executed),
            "executed_at": decision.executed_at,
            "execution_price": decision.execution_price,
            "execution_reason": decision.execution_reason,
            "outcome": decision.outcome,
            "outcome_pnl_pct": decision.outcome_pnl_pct,
        }})

    authoritative_samples = await load_authoritative_trade_training_samples()
    payload = {{
        "status": "ok",
        "read_only": True,
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "window_start": window_start.isoformat(),
        "contract_version": NORMAL_PAPER_TRADE_VERSION,
        "loaded_decision_count": len(decisions),
        "candidate_count": len(rows),
        "contract_valid_count": sum(row["contract_valid"] for row in rows),
        "authorized_count": sum(row["authorized"] for row in rows),
        "executed_count": sum(row["was_executed"] for row in rows),
        "completed_outcome_count": sum(bool(row.get("outcome")) for row in rows),
        "current_authoritative_sample_count": len(authoritative_samples),
        "sampling_policy": {{
            "single_trade_risk_fraction_cap": NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION,
            "leverage_policy": NORMAL_PAPER_TRADE_LEVERAGE_POLICY,
            "production_permission": False,
        }},
        "latest_candidates": rows[:LIMIT],
    }}
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


asyncio.run(run())
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-app-dir", default="/data/bb/app")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remote_script = _remote_script(
        remote_app_dir=args.remote_app_dir,
        hours=args.hours,
        limit=args.limit,
    )
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
