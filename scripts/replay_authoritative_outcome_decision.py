"""Replay a historical entry feature snapshot through the current ML artifact."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.remote_ssh import connect_remote_ssh, run_remote_text
from core.safe_output import safe_print
from db.session import get_read_session_ctx
from models.decision import AIDecision
from services.authoritative_trade_outcome import load_authoritative_trade_outcomes
from services.ml_signal_service import MLSignalService

_FORBIDDEN_INFERENCE_FACT_KEYS = {
    "realized_pnl",
    "outcome",
    "outcome_id",
    "outcome_version",
    "stop_loss_slippage_pct",
}


def _safe_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _compact_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    rows = []
    violations: list[str] = []
    for prediction in payload.get("predictions") or []:
        contracts = _safe_dict(prediction.get("return_distribution_contract"))
        compact_sides: dict[str, Any] = {}
        for side in ("long", "short"):
            contract = _safe_dict(contracts.get(side))
            raw_expected = contract.get("raw_expected_return_pct")
            lower = contract.get("lower_quantile_return_pct")
            dispersion = contract.get("dispersion_pct")
            tail_probability = contract.get("tail_loss_probability")
            tail_scale = contract.get("tail_loss_scale_pct")
            values = (raw_expected, lower, dispersion, tail_probability, tail_scale)
            if all(value is not None and math.isfinite(float(value)) for value in values):
                if float(lower) > float(raw_expected):
                    violations.append(f"{side}:lower_quantile_above_expected")
                if float(dispersion) < 0:
                    violations.append(f"{side}:negative_dispersion")
                if not 0 <= float(tail_probability) <= 1:
                    violations.append(f"{side}:tail_probability_out_of_bounds")
                if float(tail_scale) < 0:
                    violations.append(f"{side}:negative_tail_scale")
                if (
                    int(contract.get("distribution_member_count") or 0) > 1
                    and float(contract.get("upper_quantile_return_pct")) > float(lower)
                    and float(dispersion) <= 0
                ):
                    violations.append(f"{side}:nondegenerate_distribution_has_zero_uncertainty")
            compact_sides[side] = {
                key: contract.get(key)
                for key in (
                    "raw_expected_return_pct",
                    "objective_expected_return_pct",
                    "lower_quantile_return_pct",
                    "upper_quantile_return_pct",
                    "dispersion_pct",
                    "tail_loss_probability",
                    "tail_loss_scale_pct",
                    "uncertainty_penalty_pct",
                    "tail_loss_penalty_pct",
                    "distribution_member_count",
                    "production_eligible",
                    "blockers",
                )
            }
        rows.append(
            {
                "horizon_minutes": prediction.get("horizon_minutes"),
                "best_side": prediction.get("best_side"),
                "risk_score": prediction.get("risk_score"),
                "profit_signal": prediction.get("profit_signal"),
                "actual_trade_calibration_ready": prediction.get(
                    "actual_trade_calibration_ready"
                ),
                "sides": compact_sides,
            }
        )
    return {
        "available": payload.get("available"),
        "route_mode": payload.get("route_mode"),
        "live_ml_ready": payload.get("live_ml_ready"),
        "model_version": payload.get("model_version"),
        "trained_sample_count": payload.get("trained_sample_count"),
        "prediction_quality": payload.get("prediction_quality"),
        "rows": rows,
        "invariant_violations": violations,
    }


async def replay(*, decision_id: int, retrain_shadow: bool) -> dict[str, Any]:
    async with get_read_session_ctx() as session:
        decision = await session.scalar(
            select(AIDecision).where(AIDecision.id == int(decision_id)).limit(1)
        )
    if decision is None:
        return {"status": "blocked", "reason": "decision_not_found", "decision_id": decision_id}
    features = _safe_dict(decision.feature_snapshot)
    forbidden_present = sorted(_FORBIDDEN_INFERENCE_FACT_KEYS.intersection(features))
    outcomes = await load_authoritative_trade_outcomes(mode="paper")
    linked_outcomes = [
        outcome for outcome in outcomes if int(outcome.get("decision_id") or 0) == decision_id
    ]

    before_service = MLSignalService()
    before = _compact_prediction(before_service.predict(features))
    training_result: dict[str, Any] = {"trained": False, "reason": "not_requested"}
    if retrain_shadow:
        training_result = await before_service.maybe_auto_train(force=True)
    after_service = MLSignalService()
    after = _compact_prediction(after_service.predict(features))
    violations = [*after["invariant_violations"]]
    if forbidden_present:
        violations.append("inference_features_contain_realized_outcome_facts")
    if after.get("live_ml_ready") is True:
        violations.append("shadow_replay_unexpectedly_live_ml_ready")
    if not linked_outcomes:
        violations.append("authoritative_outcome_not_linked_to_decision")
    return {
        "status": "ok" if not violations else "blocked",
        "decision": {
            "id": int(decision.id),
            "symbol": decision.symbol,
            "original_action": decision.action,
            "feature_count": len(features),
            "forbidden_realized_fact_keys": forbidden_present,
        },
        "authoritative_outcomes": [
            {
                "outcome_id": item.get("outcome_id"),
                "outcome_version": item.get("outcome_version"),
                "outcome_fingerprint": item.get("outcome_fingerprint"),
                "realized_pnl": item.get("realized_pnl"),
                "stop_loss_slippage_pct": item.get("stop_loss_slippage_pct"),
                "training_complete": item.get("outcome_complete"),
            }
            for item in linked_outcomes
        ],
        "training_result": training_result,
        "before": before,
        "after": after,
        "direction_policy": "feature_distribution_only_no_outcome_direction_override",
        "violations": violations,
    }


def _online_report(*, decision_id: int, retrain_shadow: bool) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    remote_args = [
        ".venv/bin/python",
        "scripts/replay_authoritative_outcome_decision.py",
        "--decision-id",
        str(decision_id),
    ]
    if retrain_shadow:
        remote_args.append("--retrain-shadow")
    app_script = "\n".join(
        (
            "cd /data/bb/app",
            "export DATABASE_URL='postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql'",
            "exec " + " ".join(shlex.quote(value) for value in remote_args),
        )
    )
    ssh = connect_remote_ssh(root, timeout=20)
    try:
        output = run_remote_text(
            ssh,
            "runuser -u bb -- /bin/bash -lc " + shlex.quote(app_script),
            timeout=900,
            check=False,
        )
    finally:
        ssh.close()
    safe_print(output)
    start = output.find("{")
    try:
        return json.loads(output[start:])
    except json.JSONDecodeError as exc:
        raise SystemExit("online outcome replay did not return JSON") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-id", type=int, required=True)
    parser.add_argument("--retrain-shadow", action="store_true")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    report = (
        _online_report(
            decision_id=args.decision_id,
            retrain_shadow=args.retrain_shadow,
        )
        if args.online
        else asyncio.run(
            replay(
                decision_id=args.decision_id,
                retrain_shadow=args.retrain_shadow,
            )
        )
    )
    if not args.online:
        safe_print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if report.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
