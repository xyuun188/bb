"""Read-only audit of online position-exit reviews and budget deferrals."""

from __future__ import annotations

import argparse
import base64
import json
import secrets
import shlex
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

REMOTE_SCRIPT = r'''
import asyncio
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from db.session import get_read_session_ctx
from models.decision import AIDecision
from models.trade import Order

HOURS = __HOURS__
LIMIT = __LIMIT__


def dictionary(value):
    return value if isinstance(value, dict) else {}


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compact_decision(row, order_facts):
    raw = dictionary(row.raw_llm_response)
    dynamic = dictionary(raw.get("dynamic_exit_policy"))
    fast_scan = dictionary(raw.get("position_fast_scan"))
    contract = dictionary(raw.get("dynamic_exit_execution_contract"))
    quality = dictionary(raw.get("analysis_quality_contract"))
    cross = dictionary(quality.get("cross_validation"))
    stage_machine = dictionary(raw.get("decision_state_machine"))
    stage_summary = dictionary(stage_machine.get("summary"))
    stages = stage_machine.get("stages")
    stages = stages if isinstance(stages, list) else []
    final_reason = str(stage_summary.get("final_reason") or "")
    if not final_reason and stages:
        final_reason = str(dictionary(stages[-1]).get("reason") or "")
    action = str(row.action or "").lower()
    is_close = action in {"close_long", "close_short"}
    if fast_scan.get("skipped_llm") is True:
        category = "budget_deferred"
    elif is_close and row.was_executed is True:
        category = "close_executed"
    elif is_close:
        category = "close_not_executed"
    elif action == "hold":
        category = "model_hold"
    else:
        category = "other"
    execution = dictionary(raw.get("execution_result"))
    return {
        "id": row.id,
        "created_at": row.created_at,
        "model_name": row.model_name,
        "symbol": row.symbol,
        "action": action,
        "was_executed": bool(row.was_executed),
        "category": category,
        "fast_risk_trigger": str(raw.get("fast_risk_trigger") or ""),
        "execution_reason": str(row.execution_reason or ""),
        "fast_scan": {
            "skipped_llm": fast_scan.get("skipped_llm") is True,
            "eligible": fast_scan.get("dynamic_exit_eligible"),
            "close_fraction": number(fast_scan.get("close_fraction")),
            "reason": str(fast_scan.get("reason") or ""),
        },
        "dynamic_exit": {
            "eligible": dynamic.get("eligible"),
            "reason": str(dynamic.get("reason") or ""),
            "close_fraction": number(dynamic.get("close_fraction")),
            "fee_after_unrealized_pnl_usdt": number(
                dynamic.get("fee_after_unrealized_pnl_usdt")
            ),
            "lifecycle_net_pnl_usdt": number(dynamic.get("lifecycle_net_pnl_usdt")),
            "profit_lock_pressure": number(dynamic.get("profit_lock_pressure")),
            "profit_retrace_ratio": number(dynamic.get("profit_retrace_ratio")),
            "stop_risk_usage": number(dynamic.get("stop_risk_usage")),
            "position_age_minutes": number(dynamic.get("position_age_minutes")),
            "exit_reassessment_ready": dynamic.get("exit_reassessment_ready"),
            "hard_risk": dynamic.get("hard_risk"),
            "current_management_contract_complete": dynamic.get(
                "current_management_contract_complete"
            ),
            "execution_cost_complete": dynamic.get("execution_cost_complete"),
            "policy_fallback_reason": str(
                dictionary(dynamic.get("policy_provenance")).get("fallback_reason") or ""
            ),
        },
        "execution_contract": {
            "allowed": contract.get("allowed"),
            "block_reasons": contract.get("block_reasons") or [],
        },
        "analysis_quality": {
            "analysis_complete": quality.get("analysis_complete"),
            "unresolved_major_conflict_count": cross.get(
                "unresolved_major_conflict_count"
            ),
            "returned_expert_count": quality.get("returned_expert_count"),
            "expected_expert_count": quality.get("expected_expert_count"),
        },
        "decision_state": {
            "final_stage": stage_summary.get("final_stage"),
            "final_status": stage_summary.get("final_status"),
            "final_reason": final_reason,
        },
        "final_reason": final_reason,
        "dynamic_exit_block_reasons": contract.get("block_reasons") or [],
        "orders": order_facts.get(row.id, []),
        "execution_result": {
            "status": execution.get("status"),
            "exchange_order_id": execution.get("exchange_order_id"),
            "quantity": number(execution.get("quantity")),
            "price": number(execution.get("price")),
        },
    }


async def main():
    now = datetime.now(UTC)
    since = now - timedelta(hours=HOURS)
    window = (
        AIDecision.is_paper.is_(True),
        AIDecision.analysis_type == "position",
        AIDecision.created_at >= since,
        AIDecision.created_at <= now,
    )
    async with get_read_session_ctx(statement_timeout_ms=30000) as session:
        total_matching_count = await session.scalar(
            select(func.count(AIDecision.id)).where(*window)
        )
        decisions = (
            await session.execute(
                select(AIDecision)
                .where(*window)
                .order_by(AIDecision.id.desc())
                .limit(LIMIT)
            )
        ).scalars().all()
        decision_ids = [row.id for row in decisions]
        order_rows = []
        if decision_ids:
            order_rows = (
                await session.execute(
                    select(
                        Order.decision_id,
                        Order.status,
                        Order.side,
                        Order.quantity,
                        Order.exchange_order_id,
                    )
                    .where(
                        Order.execution_mode == "paper",
                        Order.decision_id.in_(decision_ids),
                    )
                    .order_by(Order.id.desc())
                )
            ).all()

    order_facts = defaultdict(list)
    for item in order_rows:
        order_facts[item.decision_id].append({
            "status": item.status,
            "side": item.side,
            "quantity": number(item.quantity),
            "exchange_order_id": item.exchange_order_id,
        })
    records = [compact_decision(row, order_facts) for row in decisions]
    action_counts = Counter(
        f"{item['action']}:{'executed' if item['was_executed'] else 'not_executed'}"
        for item in records
    )
    category_counts = Counter(item["category"] for item in records)
    deferred_eligible = [
        item for item in records
        if item["category"] == "budget_deferred"
        and item["fast_scan"].get("eligible") is True
    ]
    close_not_executed = [
        item for item in records
        if item["category"] == "close_not_executed"
    ]
    eligible_hold = [
        item for item in records
        if item["category"] == "model_hold"
        and item["dynamic_exit"].get("eligible") is True
    ]

    def reason_counts(items, field):
        reasons = Counter()
        for item in items:
            value = item.get(field)
            if isinstance(value, list):
                reasons.update(str(entry) for entry in value if str(entry))
            elif str(value or "").strip():
                reasons[str(value)] += 1
        return dict(reasons)

    streaks = {}
    by_group = defaultdict(list)
    for item in reversed(records):
        by_group[(item["model_name"], item["symbol"])].append(item)
    for (model_name, symbol), items in by_group.items():
        current = 0
        maximum = 0
        for item in items:
            deferred = (
                item["category"] == "budget_deferred"
                and item["fast_scan"].get("eligible") is True
            )
            if deferred:
                current += 1
                maximum = max(maximum, current)
            else:
                current = 0
        if maximum:
            streaks[f"{model_name}:{symbol}"] = {
                "current_consecutive_deferred_eligible_reviews": current,
                "maximum_consecutive_deferred_eligible_reviews": maximum,
            }

    print(json.dumps({
        "generated_at": now.isoformat(),
        "since": since.isoformat(),
        "hours": HOURS,
        "requested_limit": LIMIT,
        "total_matching_count": int(total_matching_count or 0),
        "record_count": len(records),
        "truncated": int(total_matching_count or 0) > len(records),
        "read_only": True,
        "action_counts": dict(action_counts),
        "category_counts": dict(category_counts),
        "eligible_budget_deferred_count": len(deferred_eligible),
        "eligible_model_hold_count": len(eligible_hold),
        "close_not_executed_count": len(close_not_executed),
        "okx_price_limit_rejection_count": sum(
            "OKX API error [51006]" in item["execution_reason"]
            for item in close_not_executed
        ),
        "fast_exit_counts": dict(Counter(
            f"{item['fast_risk_trigger']}:{item['category']}" for item in records
            if item["fast_risk_trigger"]
        )),
        "close_not_executed_reason_counts": reason_counts(
            close_not_executed, "execution_reason"
        ),
        "close_not_executed_final_reason_counts": reason_counts(
            close_not_executed, "final_reason"
        ),
        "dynamic_exit_block_reason_counts": reason_counts(
            close_not_executed, "dynamic_exit_block_reasons"
        ),
        "deferred_eligible_groups": streaks,
        "close_not_executed": close_not_executed[:50],
        "eligible_model_holds": eligible_hold[:50],
        "eligible_budget_deferred": deferred_eligible[:50],
    }, ensure_ascii=False, default=str))


asyncio.run(main())
'''


def _render_remote_script(hours: int, limit: int) -> str:
    return REMOTE_SCRIPT.replace("__HOURS__", str(hours)).replace(
        "__LIMIT__", str(limit)
    )


def _build_remote_command(hours: int, limit: int) -> str:
    remote_script = _render_remote_script(hours, limit)
    payload = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
    runner = f"import base64;exec(base64.b64decode({payload!r}))"
    return " ".join(
        (
            "systemd-run --quiet --wait --pipe --collect",
            "--property=WorkingDirectory=/data/bb/app",
            "--property=User=bb",
            "--property=EnvironmentFile=-/data/bb/app/.env",
            "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
            "/data/bb/app/.venv/bin/python -c",
            shlex.quote(runner),
        )
    )


def _read_remote_report(ssh, hours: int, limit: int) -> dict[str, Any]:
    directory = "/data/bb/app/tmp/codex-position-exit-queue-audit"
    result_path = f"{directory}/result_{secrets.token_hex(12)}.json"
    sftp = ssh.open_sftp()
    try:
        command = (
            "set -e\numask 077\n"
            f"install -d -m 0700 {shlex.quote(directory)}\n"
            f"{_build_remote_command(hours, limit)} > {shlex.quote(result_path)}"
        )
        run_remote_text(ssh, command, timeout=180, max_output_chars=4000)
        with sftp.file(result_path, "r") as remote_file:
            payload = json.loads(remote_file.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("online position-exit audit output must be a JSON object")
        return payload
    finally:
        with suppress(OSError):
            sftp.remove(result_path)
        sftp.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=12)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    hours = max(1, min(int(args.hours or 12), 168))
    limit = max(1, min(int(args.limit or 500), 10000))
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        report = _read_remote_report(ssh, hours, limit)
    finally:
        ssh.close()
    if args.summary:
        for key in ("close_not_executed", "eligible_model_holds", "eligible_budget_deferred",
                    "deferred_eligible_groups", "close_not_executed_reason_counts",
                    "close_not_executed_final_reason_counts"):
            report.pop(key, None)
    safe_print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
