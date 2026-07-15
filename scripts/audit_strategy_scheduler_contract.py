"""Read-only audit for production strategy and historical-prior ownership."""

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
from services.strategy_learning import (
    PRODUCTION_STRATEGY_ID,
    PRODUCTION_STRATEGY_VERSION,
    StrategyLearningService,
)
from services.trading_params import DEFAULT_TRADING_PARAMS

_FORBIDDEN_RUNTIME_KEYS = {
    "active_profile",
    "strategy_profile_id",
    "strategy_profile_version",
}


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk(nested)


async def audit(*, mode: str) -> dict[str, Any]:
    params = DEFAULT_TRADING_PARAMS.strategy_learning
    payload = await StrategyLearningService().dashboard_payload(
        mode=mode,
        hours=params.default_lookback_hours,
        limit=params.dashboard_summary_limit,
        detail="summary",
    )
    schedule = payload.get("schedule") if isinstance(payload.get("schedule"), dict) else {}
    production = (
        payload.get("current_production_strategy")
        if isinstance(payload.get("current_production_strategy"), dict)
        else {}
    )
    runtime = schedule.get("runtime") if isinstance(schedule.get("runtime"), dict) else {}
    candidates = schedule.get("candidates") if isinstance(schedule.get("candidates"), list) else []
    usage = (
        payload.get("feedback", {}).get("runtime_prior_usage", {})
        if isinstance(payload.get("feedback"), dict)
        else {}
    )
    decision_records = (
        usage.get("decision_records") if isinstance(usage.get("decision_records"), list) else []
    )
    violations: list[str] = []
    if production.get("id") != PRODUCTION_STRATEGY_ID:
        violations.append("current_production_strategy_missing_or_wrong_owner")
    if production.get("version") != PRODUCTION_STRATEGY_VERSION:
        violations.append("current_production_strategy_version_mismatch")
    if production.get("enabled") is not True:
        violations.append("current_production_strategy_not_enabled")
    if production.get("historical_prior_can_authorize_entry") is not False:
        violations.append("historical_prior_can_authorize_entry")
    if runtime.get("can_authorize_entry") is not False:
        violations.append("scheduler_runtime_can_authorize_entry")
    if runtime.get("can_change_size_or_leverage") is not False:
        violations.append("scheduler_runtime_can_change_size_or_leverage")
    for node in _walk(payload):
        for key in _FORBIDDEN_RUNTIME_KEYS.intersection(node):
            violations.append(f"forbidden_runtime_key:{key}")
    if any(int(candidate.get("version") or 0) <= 0 for candidate in candidates):
        violations.append("historical_prior_version_missing")
    if any(
        candidate.get("promotion", {}).get("can_authorize_entry") is not False
        for candidate in candidates
    ):
        violations.append("candidate_can_authorize_entry")
    for record in decision_records:
        if record.get("final_reason") in (None, ""):
            violations.append("decision_final_reason_missing")
        for side_record in record.get("side_evaluations") or []:
            if side_record.get("evaluation_status") not in {
                "matched_historical_prior",
                "not_matched",
                "not_evaluated",
            }:
                violations.append("decision_prior_evaluation_status_missing")
            if side_record.get("can_authorize_entry") is not False:
                violations.append("decision_prior_can_authorize_entry")
    return {
        "status": "ok" if not violations else "blocked",
        "mode": mode,
        "violations": sorted(set(violations)),
        "current_production_strategy": production,
        "scheduler": {
            "mode": schedule.get("scheduler_mode"),
            "candidate_count": len(candidates),
            "governed_candidate_count": schedule.get("governed_candidate_count"),
            "rejected_candidate_count": schedule.get("rejected_candidate_count"),
            "can_authorize_entry": runtime.get("can_authorize_entry"),
            "can_change_size_or_leverage": runtime.get("can_change_size_or_leverage"),
        },
        "candidate_versions": [
            {"id": item.get("id"), "version": item.get("version")} for item in candidates
        ],
        "prior_usage": {
            "inspected_decision_count": usage.get("inspected_decision_count"),
            "matched_decision_count": usage.get("matched_decision_count"),
            "matched_profile_count": usage.get("matched_profile_count"),
            "decision_record_count": len(decision_records),
            "latest_matches": usage.get("latest_matches") or [],
        },
    }


def _online_report(*, mode: str) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    app_script = "\n".join(
        (
            "cd /data/bb/app",
            "export DATABASE_URL='postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql'",
            "exec .venv/bin/python scripts/audit_strategy_scheduler_contract.py --mode "
            + shlex.quote(mode),
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
        raise SystemExit("online strategy scheduler audit did not return JSON") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("paper", "live"), default="paper")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    report = (
        _online_report(mode=args.mode)
        if args.online
        else asyncio.run(audit(mode=args.mode))
    )
    if not args.online:
        safe_print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if report.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
