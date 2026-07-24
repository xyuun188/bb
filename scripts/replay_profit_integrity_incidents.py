#!/usr/bin/env python3
"""Replay the ROBO, ICP, and DOGE incidents through current profit contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.market_facts import (  # noqa: E402
    build_market_fact,
    build_shadow_market_fact_contract,
    market_fact_contract_reasons,
)
from services.authoritative_trade_outcome import (  # noqa: E402
    build_authoritative_trade_outcome,
)
from services.profit_supervision import PROFIT_SUPERVISION_VERSION  # noqa: E402
from services.profit_training_contract import PROFIT_TRAINING_TARGET  # noqa: E402
from services.return_objective import standardized_return_distribution  # noqa: E402

REPLAY_VERSION = "2026-07-15.profit-integrity-incident-replay.v1"
BASELINE_PATH = (
    ROOT
    / "tests"
    / "fixtures"
    / "profit_integrity"
    / "2026-07-14-root-cause-baseline.json"
)
ROBO_PATH = (
    ROOT
    / "tests"
    / "fixtures"
    / "profit_integrity"
    / "2026-07-14-robo-native-source-regression.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"fixture must contain an object: {path}")
    return payload


def _decision(payload: dict[str, Any], decision_id: int) -> dict[str, Any]:
    return next(
        row
        for row in payload.get("decisions", [])
        if isinstance(row, dict) and int(row.get("id") or 0) == decision_id
    )


def _prediction(decision: dict[str, Any]) -> dict[str, Any]:
    evidence = decision.get("production_evidence") or {}
    signal = evidence.get("ml_signal") or {}
    rows = signal.get("predictions") or []
    return next(row for row in rows if isinstance(row, dict))


def _robo_replay(baseline: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    shadows_by_decision: dict[int, list[dict[str, Any]]] = {}
    for row in baseline.get("shadow_backtests", []):
        if not isinstance(row, dict) or str(row.get("symbol")) != "ROBO/USDT":
            continue
        shadows_by_decision.setdefault(int(row.get("decision_id") or 0), []).append(row)

    rows: list[dict[str, Any]] = []
    for incident in fixture.get("incident_rest_snapshots", []):
        decision_id = int(incident.get("decision_id") or 0)
        fact = build_market_fact(
            fixture["symbol"],
            {
                "symbol": fixture["symbol"],
                "inst_id": fixture["inst_id"],
                "inst_type": "SWAP",
                "source": "rest",
                "source_endpoint": "okx_demo_rest_market_ticker",
                "source_channel": "tickers",
                "timestamp": incident["source_timestamp_ms"],
                "last_price": incident["last"],
                "bid": incident["bid"],
                "ask": incident["ask"],
                "notional_24h_usdt": incident["notional_24h_usdt"],
                "orderbook_bid_depth": 1407.47398,
                "orderbook_ask_depth": 1502.61238,
            },
            contract_spec=fixture["instrument_response"],
        )
        contract = build_shadow_market_fact_contract(fact, None, None)
        contract_reasons = market_fact_contract_reasons(contract)
        old_shadows = shadows_by_decision.get(decision_id, [])
        rows.append(
            {
                "decision_id": decision_id,
                "before": {
                    "entry_price": incident["last"],
                    "maximum_recorded_short_return_pct": max(
                        (float(row.get("short_return_pct") or 0.0) for row in old_shadows),
                        default=None,
                    ),
                    "training_contract_present": False,
                },
                "after": {
                    "market_fact_status": fact["quality"]["status"],
                    "market_fact_reasons": fact["quality"]["reasons"],
                    "shadow_contract_status": contract["status"],
                    "shadow_contract_reasons": contract_reasons,
                    "training_sample_weight": 0.0 if contract_reasons else 1.0,
                    "production_contribution": 0.0 if contract_reasons else 1.0,
                },
            }
        )
    return {
        "symbol": fixture["symbol"],
        "incident_count": len(rows),
        "all_invalid_entries_quarantined": bool(rows)
        and all(row["after"]["training_sample_weight"] == 0.0 for row in rows),
        "replays": rows,
    }


def _icp_replay(baseline: dict[str, Any]) -> dict[str, Any]:
    decision = _decision(baseline, 79318)
    prediction = _prediction(decision)
    evidence = decision["production_evidence"]
    old_expected = float(evidence["ml_signal"]["expected_return_pct"])
    old_lower = float(prediction["short_lower_quantile_return_pct"])
    distribution = standardized_return_distribution(
        side="short",
        horizon_minutes=prediction["horizon_minutes"],
        raw_expected_return_pct=old_expected,
        median_return_pct=old_expected,
        lower_quantile_return_pct=old_lower,
        upper_quantile_return_pct=float(prediction["short_raw_expected_return_pct"]),
        dispersion_pct=float(prediction["short_uncertainty_penalty_pct"]),
        tail_loss_probability=float(prediction["short_tail_loss_probability"]),
        tail_loss_scale_pct=float(prediction["tail_loss_threshold_pct"]),
        distribution_member_count=1,
        return_semantics="gross_market_opportunity_before_execution",
        source_authority="incident_local_ml_prediction",
        profit_supervision_version=PROFIT_SUPERVISION_VERSION,
    )

    position = next(row for row in baseline["positions"] if int(row.get("id") or 0) == 4879)
    reflection = baseline["position_4879_learning"]["reflections"][0]
    entry_price = float(reflection["entry_price"])
    exit_price = float(reflection["exit_price"])
    stop_price = float(position["stop_loss_price"])
    quantity = float(position["quantity"])
    notional = abs(entry_price * quantity)
    authoritative = build_authoritative_trade_outcome(
        {
            "source": "okx_position_history",
            "lifecycle_key": "paper|ICP-USDT-SWAP|3740771018462691328|short|1",
            "position_id": 4879,
            "position_ids": [4879],
            "decision_id": 79318,
            "okx_pos_id": position["okx_pos_id"],
            "execution_mode": "paper",
            "symbol": "ICP/USDT",
            "side": "short",
            "entry_price": entry_price,
            "close_price": exit_price,
            "quantity": quantity,
            "notional": notional,
            "notional_source": "incident_replay_immutable_fixture",
            "realized_pnl": float(position["realized_pnl"]),
            "funding_fee": float(position.get("funding_fee") or 0.0),
            "planned_stop_loss_price": stop_price,
            "stop_loss_fill_confirmed": True,
            "slippage": None,
            "slippage_source": "",
            PROFIT_TRAINING_TARGET: (
                float(position["realized_pnl"]) / notional * 100.0
            ),
            "training_evidence_gaps": ["missing_authoritative_slippage"],
        },
        reflection=SimpleNamespace(
            id=reflection["id"],
            position_id=reflection["position_id"],
            source=reflection["source"],
            outcome=reflection["outcome"],
            mistake_summary=reflection["mistake_summary"],
            improvement_summary=reflection["improvement_summary"],
            created_at=None,
        ),
    )
    slippage = authoritative["attribution"]["execution_slippage"]
    return {
        "symbol": "ICP/USDT",
        "decision_id": 79318,
        "before": {
            "expected_return_pct": old_expected,
            "lower_quantile_return_pct": old_lower,
            "live_ml_ready": bool(evidence["ml_signal"]["live_ml_ready"]),
        },
        "after": {
            "distribution_production_eligible": distribution["production_eligible"],
            "distribution_blockers": distribution["blockers"],
            "production_contribution": 0.0 if distribution["blockers"] else 1.0,
            "authoritative_return_after_all_cost_pct": authoritative[
                PROFIT_TRAINING_TARGET
            ],
            "outcome_complete": authoritative["outcome_complete"],
            "execution_slippage": slippage,
        },
    }


def _doge_replay(baseline: dict[str, Any]) -> dict[str, Any]:
    decision = _decision(baseline, 79568)
    prediction = _prediction(decision)
    evidence = decision["production_evidence"]
    old_opportunity = evidence["opportunity_score"]
    raw = float(prediction["short_raw_expected_return_pct"])
    lower = float(prediction["short_lower_quantile_return_pct"])
    dispersion = float(prediction["short_uncertainty_penalty_pct"])
    tail_probability = float(prediction["short_tail_loss_probability"])
    tail_penalty = float(prediction["short_tail_loss_penalty_pct"])
    tail_scale = tail_penalty / tail_probability if tail_probability > 0 else 0.0
    distribution = standardized_return_distribution(
        side="short",
        horizon_minutes=prediction["horizon_minutes"],
        raw_expected_return_pct=raw,
        median_return_pct=(raw + lower) / 2.0,
        lower_quantile_return_pct=lower,
        upper_quantile_return_pct=raw + dispersion,
        dispersion_pct=dispersion,
        tail_loss_probability=tail_probability,
        tail_loss_scale_pct=tail_scale,
        distribution_member_count=1,
        return_semantics="gross_market_opportunity_before_execution",
        source_authority="incident_local_ml_tree_distribution",
        profit_supervision_version=PROFIT_SUPERVISION_VERSION,
    )
    return {
        "symbol": "DOGE/USDT",
        "decision_id": 79568,
        "before": {
            "source_count": 1,
            "return_uncertainty_pct": old_opportunity["return_uncertainty_pct"],
            "return_lcb_pct": old_opportunity["return_lcb_pct"],
            "expected_net_return_pct": old_opportunity["expected_net_return_pct"],
        },
        "after": {
            "distribution_member_count": distribution["distribution_member_count"],
            "dispersion_pct": distribution["dispersion_pct"],
            "uncertainty_penalty_pct": distribution["uncertainty_penalty_pct"],
            "objective_expected_return_pct": distribution["objective_expected_return_pct"],
            "production_eligible": distribution["production_eligible"],
            "blockers": distribution["blockers"],
        },
    }


def build_replay_report() -> dict[str, Any]:
    baseline = _load_json(BASELINE_PATH)
    report = {
        "replay_version": REPLAY_VERSION,
        "optimization_target": PROFIT_TRAINING_TARGET,
        "win_rate_used_for_acceptance": False,
        "inputs": {
            "baseline_fixture": BASELINE_PATH.relative_to(ROOT).as_posix(),
            "robo_fixture": ROBO_PATH.relative_to(ROOT).as_posix(),
        },
        "incidents": {
            "ROBO": _robo_replay(baseline, _load_json(ROBO_PATH)),
            "ICP": _icp_replay(baseline),
            "DOGE": _doge_replay(baseline),
        },
    }
    incidents = report["incidents"]
    report["status"] = (
        "passed"
        if incidents["ROBO"]["all_invalid_entries_quarantined"]
        and incidents["ICP"]["after"]["production_contribution"] == 0.0
        and incidents["ICP"]["after"]["outcome_complete"] is False
        and incidents["ICP"]["after"]["execution_slippage"]["status"]
        == "unavailable"
        and incidents["DOGE"]["after"]["uncertainty_penalty_pct"] > 0.0
        else "blocked"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_replay_report()
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
