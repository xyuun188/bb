"""Replay the local paper-entry policy on bounded online evidence; never order."""

from __future__ import annotations

import argparse
import base64
import json
import shlex
import sys
import zlib
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402
from services.normal_paper_trade import (  # noqa: E402
    build_normal_paper_trade_contract,
    normal_paper_entry_contract_reasons,
    select_normal_paper_trade_side,
)

REMOTE_CODE = """
import asyncio, base64, json, zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from scripts.runtime_env_bootstrap import load_runtime_env_files, drop_privileges_to_runtime_user_if_needed
root = Path('/data/bb/app')
load_runtime_env_files(project_root=root)
drop_privileges_to_runtime_user_if_needed(project_root=root)
from sqlalchemy import select
from db.session import get_read_session_ctx, close_db
from models.decision import AIDecision

async def main():
    try:
        since = datetime.now(UTC) - timedelta(minutes=__MINUTES__)
        ids = select(AIDecision.id).where(
            AIDecision.is_paper.is_(True), AIDecision.analysis_type == 'market',
            AIDecision.created_at >= since,
        ).order_by(AIDecision.id.desc()).limit(__LIMIT__).scalar_subquery()
        async with get_read_session_ctx(statement_timeout_ms=20000) as session:
            rows = (await session.execute(select(
                AIDecision.id, AIDecision.created_at, AIDecision.symbol,
                AIDecision.was_executed,
                AIDecision.raw_llm_response['paper_trade_selection'].label('selection'),
            ).where(AIDecision.id.in_(ids)).order_by(AIDecision.id.desc()))).all()
        payload = json.dumps([dict(row._mapping) for row in rows], default=str).encode()
        print('BB_POLICY_DATA=' + base64.b64encode(zlib.compress(payload)).decode())
    finally:
        await close_db()
asyncio.run(main())
"""


def replay_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    blockers: Counter[str] = Counter()
    sides: Counter[str] = Counter()
    examples = []
    for row in rows:
        selection = row.get("selection") or {}
        if not isinstance(selection, dict) or not selection.get("by_side"):
            counts["missing_support"] += 1
            continue
        by_side = deepcopy(selection["by_side"])
        # This reason was added by the old selector, after direction support.
        # Re-run selection from that stage without erasing any upstream guard.
        for support in by_side.values():
            support["blocking_reasons"] = [
                reason for reason in support.get("blocking_reasons") or []
                if reason != "direction_support_risk_adjusted_return_not_positive"
            ]
        replay = select_normal_paper_trade_side(by_side)
        counts["replayed"] += 1
        counts["originally_selected"] += selection.get("selected") is True
        if not replay["selected"]:
            counts["still_blocked"] += 1
            blockers.update(set(replay["blocking_reasons"]))
            continue
        side = replay["selected_side"]
        contract = build_normal_paper_trade_contract(
            symbol=row["symbol"], side=side,
            selection_reason=replay["selection_reason"],
            direction_support=replay["selected_support"],
        )
        reasons = normal_paper_entry_contract_reasons(contract)
        if reasons:
            counts["contract_rejected"] += 1
            blockers.update(reasons)
            continue
        counts["selection_contract_ready"] += 1
        sides[side] += 1
        if selection.get("selected") is not True:
            counts["newly_selection_contract_ready"] += 1
            if len(examples) < 5:
                examples.append({
                    "id": row["id"], "created_at": row.get("created_at"),
                    "symbol": row["symbol"], "side": side,
                    "selection_reason": replay["selection_reason"],
                    **{key: contract[key] for key in (
                        "expected_net_return_pct", "objective_net_return_pct",
                        "loss_probability", "single_trade_risk_fraction_cap",
                        "portfolio_risk_fraction_cap", "production_permission",
                    ) if key in contract},
                })
    return {
        "read_only": True, "submits_orders": False,
        "scope": "selection_and_contract_only_not_execution_or_profitability",
        "row_count": len(rows), "counts": dict(counts),
        "ready_sides": dict(sides), "blocking_reasons": dict(blockers),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=int, default=420)
    parser.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()
    code = REMOTE_CODE.replace("__MINUTES__", str(max(1, min(args.minutes, 1440))))
    code = code.replace("__LIMIT__", str(max(1, min(args.limit, 256))))
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        output = run_remote_text(
            ssh, "cd /data/bb/app && .venv/bin/python -c " + shlex.quote(code),
            timeout=60, max_output_chars=200_000,
        )
    finally:
        ssh.close()
    payloads = [line.removeprefix("BB_POLICY_DATA=") for line in output.splitlines()
                if line.startswith("BB_POLICY_DATA=")]
    if len(payloads) != 1:
        raise ValueError("online policy evidence missing or ambiguous")
    rows = json.loads(zlib.decompress(base64.b64decode(payloads[0], validate=True)))
    safe_print(json.dumps(replay_policy(rows), ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
