"""Non-degrading champion/challenger comparison for the unified ML artifact."""

from __future__ import annotations

import math
from typing import Any

from services.ml_prediction_contract import ENTRY_COHORT_VERSION
from services.ml_training_contract import MIN_TRAINING_SAMPLE_COUNT

MODEL_CHAMPION_POLICY_VERSION = "2026-11-13.positive-entry-portfolio-champion.v5"
PAIRED_COMPARISON_VERSION = "2026-11-13.positive-entry-common-holdout.v2"

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


def _entry_metrics(evidence: dict[str, Any]) -> dict[str, float | None]:
    metrics = {
        "avg_return_pct": _float(evidence.get("avg_return_pct")),
        "return_lcb_pct": _float(evidence.get("return_lcb_pct")),
        "profit_factor": None,
        "cvar_10_pct": _float(evidence.get("cvar_10_pct")),
        "max_drawdown_pct": _float(evidence.get("max_drawdown_pct")),
    }
    if (_float(evidence.get("count")) or 0) <= 0:
        return dict.fromkeys(metrics)
    gain = _float(evidence.get("gross_profit_pct"))
    loss = _float(evidence.get("gross_loss_pct"))
    if gain is not None and gain >= 0 and loss is not None and loss > 1e-12:
        metrics["profit_factor"] = gain / loss
    return metrics


def _comparison_entry_evidence(
    evaluation: dict[str, Any], role: str, sides: tuple[str, ...]
) -> dict[str, Any]:
    report = _dict(evaluation.get(f"{role}_entry_evaluation"))
    side_reports = _dict(report.get("sides"))
    if report.get("version") != ENTRY_COHORT_VERSION or any(
        not isinstance(side_reports.get(side), dict) for side in sides
    ):
        return {}
    # Use the chronological pooled outcomes, including a legitimately empty side.
    return _dict(report.get("portfolio")) if len(sides) == 2 else side_reports[sides[0]]


def _training_data_version(metadata: dict[str, Any] | None) -> str:
    if not isinstance(metadata, dict):
        return ""
    quality = _dict(metadata.get("quality_report"))
    return str(
        metadata.get("training_data_version")
        or quality.get("data_quality_version")
        or ""
    ).strip()


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
    candidate_enabled_sides: list[str] | None = None,
    comparison_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Accept lifecycle progress, but require strict improvement over an active champion."""

    candidate_stage = str(candidate_stage or "shadow").lower()
    champion_stage = str(champion_stage or "").lower()
    if candidate_stage == "live":
        candidate_stage = "active"
    if champion_stage == "live":
        champion_stage = "active"
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
    comparison_sides = ("long", "short")
    if candidate_stage == "canary" and candidate_enabled_sides is not None:
        if not candidate_enabled_sides or any(
            side not in comparison_sides for side in candidate_enabled_sides
        ):
            report["blocking_reasons"] = ["candidate_enabled_sides_invalid"]
            return report
        comparison_sides = tuple(
            side for side in comparison_sides if side in candidate_enabled_sides
        )
    report["comparison_sides"] = list(comparison_sides)
    if not champion:
        return {
            **report,
            "accepted": True,
            "reason": "initial_champion",
        }
    if champion_stage not in _STAGE_RANK:
        report["blocking_reasons"] = ["champion_stage_invalid"]
        return report

    candidate_training_data_version = _training_data_version(candidate)
    champion_training_data_version = _training_data_version(champion)
    report["candidate_training_data_version"] = candidate_training_data_version or None
    report["champion_training_data_version"] = champion_training_data_version or None
    training_contract_refresh = bool(
        candidate_training_data_version
        and champion_training_data_version
        and candidate_training_data_version != champion_training_data_version
    )

    candidate_rank = _STAGE_RANK[candidate_stage]
    champion_rank = _STAGE_RANK[champion_stage]
    if candidate_rank < champion_rank:
        if training_contract_refresh and champion_stage != "active":
            return {
                **report,
                "accepted": True,
                "reason": "training_data_contract_refresh",
            }
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

    if training_contract_refresh and champion_rank < _STAGE_RANK["active"]:
        return {
            **report,
            "accepted": True,
            "reason": "training_data_contract_refresh",
        }

    # Historical walk-forward reports evaluate different fits and market windows.
    # Replacement must compare the frozen artifacts on one common unseen window.
    evaluation = _dict(comparison_evaluation)
    fingerprint = str(evaluation.get("evaluation_data_sha256") or "")
    paired_errors = []
    if (
        evaluation.get("version") != PAIRED_COMPARISON_VERSION
        or evaluation.get("ready") is not True
        or evaluation.get("chronological_label_disjoint") is not True
        or (_float(evaluation.get("sample_count")) or 0) < MIN_TRAINING_SAMPLE_COUNT
        or len(fingerprint) != 64
        or any(char not in "0123456789abcdef" for char in fingerprint)
    ):
        paired_errors.append("common_unfitted_holdout_unavailable")
    for role, metadata in (("candidate", candidate), ("champion", champion)):
        expected = str(metadata.get("training_data_sha256") or "")
        if not expected or evaluation.get(f"{role}_training_data_sha256") != expected:
            paired_errors.append(f"paired_{role}_training_identity_mismatch")
    if paired_errors:
        report["reason"] = "paired_comparison_unavailable"
        report["blocking_reasons"] = paired_errors
        report["comparison_evaluation"] = evaluation
        return report
    report["comparison_evaluation"] = evaluation
    entry_evidence = {
        role: _comparison_entry_evidence(evaluation, role, comparison_sides)
        for role in ("candidate", "champion")
    }
    candidate_aggregate = _entry_metrics(entry_evidence["candidate"])
    champion_aggregate = _entry_metrics(entry_evidence["champion"])
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
            if key != "profit_factor"
        ]
        if "profit_factor" in missing:
            for role, metrics in (("candidate", candidate_aggregate), ("champion", champion_aggregate)):
                if metrics["profit_factor"] is not None:
                    continue
                no_losses = (
                    entry_evidence[role].get("profit_factor_status") == "no_loss_denominator"
                    and (_float(entry_evidence[role].get("count")) or 0) > 0
                )
                report["blocking_reasons"].append(
                    f"{role}_profit_factor_no_loss_denominator" if no_losses
                    else "champion_comparison_metric_missing:profit_factor"
                )
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
