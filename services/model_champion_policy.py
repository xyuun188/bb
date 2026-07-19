"""Non-degrading champion/challenger comparison for the unified ML artifact."""

from __future__ import annotations

import math
from typing import Any

MODEL_CHAMPION_POLICY_VERSION = "2026-07-19.fee-after-champion.v1"

_STAGE_RANK = {
    "shadow": 0,
    "canary": 1,
    "active": 2,
    "live": 2,
}
_PRIMARY_TOLERANCE_PCT = 1e-9
_TAIL_TOLERANCE_PCT = 0.05
_MIN_ACTIVE_WALK_FORWARD_FOLDS = 2


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _side_metrics(metadata: dict[str, Any], side: str) -> dict[str, float | None]:
    evidence = _dict(_dict(metadata.get("oos_return_evaluation")).get(side))
    return {
        "avg_return_pct": _float(evidence.get("avg_return_pct")),
        "return_lcb_pct": _float(evidence.get("return_lcb_pct")),
        "profit_factor": _float(evidence.get("profit_factor")),
        "cvar_10_pct": _float(evidence.get("cvar_10_pct")),
        "max_drawdown_pct": _float(evidence.get("max_drawdown_pct")),
    }


def _average(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def _aggregate(metadata: dict[str, Any]) -> dict[str, float | None]:
    sides = [_side_metrics(metadata, side) for side in ("long", "short")]
    return {
        key: _average([side[key] for side in sides])
        for key in (
            "avg_return_pct",
            "return_lcb_pct",
            "profit_factor",
            "cvar_10_pct",
            "max_drawdown_pct",
        )
    }


def _stable_both_sides(metadata: dict[str, Any]) -> bool:
    walk_report = _dict(metadata.get("walk_forward_report"))
    walk_sides = _dict(walk_report.get("sides"))
    folds = walk_report.get("folds")
    if not isinstance(folds, list) or len(folds) < _MIN_ACTIVE_WALK_FORWARD_FOLDS:
        return False
    loso = _dict(metadata.get("leave_one_symbol_out_report"))
    oos = _dict(metadata.get("oos_return_evaluation"))
    return all(
        _dict(walk_sides.get(side)).get("promotion_math_ready") is True
        and _dict(loso.get(side)).get("stable") is True
        and _dict(oos.get(side)).get("promotion_math_ready") is True
        and _dict(_dict(walk_sides.get(side)).get("market_regime_stability")).get(
            "stable"
        )
        is True
        and all(
            _dict(_dict(_dict(fold).get("sides")).get(side)).get(
                "promotion_math_ready"
            )
            is True
            for fold in folds
            if isinstance(fold, dict)
        )
        for side in ("long", "short")
    )


def compare_candidate_to_champion(
    candidate: dict[str, Any],
    champion: dict[str, Any] | None,
    *,
    candidate_stage: str,
    champion_stage: str | None,
) -> dict[str, Any]:
    """Accept lifecycle progress, but require strict improvement over an active champion."""

    candidate_stage = str(candidate_stage or "shadow").lower()
    champion_stage = str(champion_stage or "").lower()
    report: dict[str, Any] = {
        "version": MODEL_CHAMPION_POLICY_VERSION,
        "candidate_stage": candidate_stage,
        "champion_stage": champion_stage or None,
        "candidate_training_data_sha256": candidate.get("training_data_sha256"),
        "champion_training_data_sha256": (
            champion.get("training_data_sha256") if isinstance(champion, dict) else None
        ),
        "accepted": False,
        "reason": "comparison_incomplete",
        "blocking_reasons": [],
    }
    if candidate_stage not in _STAGE_RANK:
        report["blocking_reasons"] = ["candidate_stage_invalid"]
        return report
    if not champion:
        return {
            **report,
            "accepted": True,
            "reason": "initial_champion",
        }
    if champion_stage not in _STAGE_RANK:
        report["blocking_reasons"] = ["champion_stage_invalid"]
        return report

    candidate_rank = _STAGE_RANK[candidate_stage]
    champion_rank = _STAGE_RANK[champion_stage]
    if candidate_rank < champion_rank:
        report["blocking_reasons"] = ["candidate_lifecycle_regression"]
        return report
    if candidate_rank > champion_rank:
        if candidate_stage == "active" and not _stable_both_sides(candidate):
            report["blocking_reasons"] = ["active_candidate_cross_section_unstable"]
            return report
        return {
            **report,
            "accepted": True,
            "reason": "governed_lifecycle_upgrade",
        }

    candidate_aggregate = _aggregate(candidate)
    champion_aggregate = _aggregate(champion)
    report["candidate_metrics"] = candidate_aggregate
    report["champion_metrics"] = champion_aggregate
    report["metric_deltas"] = {
        key: (
            candidate_aggregate[key] - champion_aggregate[key]
            if candidate_aggregate[key] is not None
            and champion_aggregate[key] is not None
            else None
        )
        for key in candidate_aggregate
    }
    missing = [
        key
        for key in candidate_aggregate
        if candidate_aggregate[key] is None or champion_aggregate[key] is None
    ]
    if missing:
        report["blocking_reasons"] = [
            f"champion_comparison_metric_missing:{key}" for key in missing
        ]
        return report

    assert all(value is not None for value in candidate_aggregate.values())
    assert all(value is not None for value in champion_aggregate.values())
    candidate_values = {key: float(value) for key, value in candidate_aggregate.items()}
    champion_values = {key: float(value) for key, value in champion_aggregate.items()}

    if champion_rank == _STAGE_RANK["active"]:
        blockers: list[str] = []
        if candidate_stage != "active":
            blockers.append("active_champion_requires_active_challenger")
        if not _stable_both_sides(candidate):
            blockers.append("active_candidate_cross_section_unstable")
        for key in ("avg_return_pct", "return_lcb_pct", "profit_factor"):
            if candidate_values[key] <= champion_values[key] + _PRIMARY_TOLERANCE_PCT:
                blockers.append(f"candidate_{key}_not_improved")
        if candidate_values["return_lcb_pct"] <= 0:
            blockers.append("candidate_return_lcb_not_positive")
        if candidate_values["profit_factor"] <= 1:
            blockers.append("candidate_profit_factor_not_above_one")
        if candidate_values["cvar_10_pct"] < champion_values["cvar_10_pct"]:
            blockers.append("candidate_cvar_worsened")
        if candidate_values["max_drawdown_pct"] > champion_values["max_drawdown_pct"]:
            blockers.append("candidate_max_drawdown_worsened")
        report["blocking_reasons"] = blockers
        report["accepted"] = not blockers
        report["reason"] = (
            "strict_fee_after_improvement" if not blockers else "active_champion_retained"
        )
        return report

    primary_improved = any(
        candidate_values[key] > champion_values[key] + _PRIMARY_TOLERANCE_PCT
        for key in ("avg_return_pct", "return_lcb_pct", "profit_factor")
    )
    blockers = []
    if not primary_improved:
        blockers.append("candidate_primary_fee_after_metrics_not_improved")
    if (
        candidate_values["cvar_10_pct"]
        < champion_values["cvar_10_pct"] - _TAIL_TOLERANCE_PCT
    ):
        blockers.append("candidate_cvar_materially_worsened")
    if (
        candidate_values["max_drawdown_pct"]
        > champion_values["max_drawdown_pct"] + _TAIL_TOLERANCE_PCT
    ):
        blockers.append("candidate_max_drawdown_materially_worsened")
    report["blocking_reasons"] = blockers
    report["accepted"] = not blockers
    report["reason"] = (
        "challenger_quality_improved" if not blockers else "champion_retained"
    )
    return report
