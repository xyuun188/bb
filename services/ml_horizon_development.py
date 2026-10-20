"""Nested horizon research on a frozen development snapshot, without order access."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from services.ml_development_baselines import (
    _baseline_model,
    _development_input_sha256,
    _evidence,
    _net_returns,
)
from services.ml_nested_development import ALPHAS, NET_THRESHOLDS_PCT, _folds, _sides
from services.ml_signal_service import FEATURE_KEYS, MIN_TRAINING_SAMPLE_COUNT

HORIZON_POLICY_VERSION = "2026-10-21.frozen-nested-horizons.v1"
HORIZONS = (5, 15, 60, 240)
_NON_MARKET_KEYS = {
    "decision_confidence", "sentiment_data_available",
    "direct_sentiment_data_available", "news_sentiment_avg",
    "social_sentiment_avg", "social_mention_count", "news_article_count",
    "direct_news_item_count", "market_news_item_count", "news_shock_present",
}
FEATURE_POLICIES = {
    "existing": tuple(FEATURE_KEYS),
    "market_only": tuple(key for key in FEATURE_KEYS if key not in _NON_MARKET_KEYS),
}


def _at_horizon(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return frame[frame["horizon_minutes"] == horizon].copy()


def _predict(
    train: pd.DataFrame, validation: pd.DataFrame, *, alpha: float,
    feature_policy: str,
) -> dict[str, np.ndarray]:
    features = list(FEATURE_POLICIES[feature_policy])
    weights = train.get("sample_weight", pd.Series(1.0, index=train.index))
    train = train[weights > 0].copy()
    weights = weights[weights > 0]
    if len(train) < MIN_TRAINING_SAMPLE_COUNT:
        raise ValueError("insufficient positive-weight training observations")
    result = {}
    for side in ("long", "short"):
        target = _net_returns(train, side)
        model = _baseline_model("ridge_net", target)
        model.set_params(model__alpha=alpha)
        model.fit(train[features], target, model__sample_weight=weights)
        result[side] = np.asarray(model.predict(validation[features]), dtype=float)
        if not np.isfinite(result[side]).all():
            raise ValueError("non-finite horizon research prediction")
    return result


def _serial_sides(frame: pd.DataFrame, sides: np.ndarray) -> np.ndarray:
    """One global position; choose by decision time and ID without reading outcomes."""

    result = np.full(len(frame), "hold", dtype=object)
    decisions = pd.to_datetime(frame["decision_timestamp"], utc=True)
    # Use the known holding duration rather than an outcome-dependent exit.
    ends = pd.concat([
        decisions + pd.to_timedelta(frame["horizon_minutes"], unit="m"),
        pd.to_datetime(frame["label_timestamp"], utc=True),
    ], axis=1).max(axis=1)
    available = None
    order = sorted(
        range(len(frame)),
        key=lambda index: (decisions.iloc[index], str(frame.iloc[index]["decision_group"])),
    )
    for index in order:
        if sides[index] not in ("long", "short"):
            continue
        if available is None or decisions.iloc[index] >= available:
            result[index] = sides[index]
            available = ends.iloc[index]
    return result


def _views(frame: pd.DataFrame, sides: np.ndarray) -> dict[str, Any]:
    return {
        "all_opportunities": _evidence(frame, sides),
        "nonoverlapping": _evidence(frame, _serial_sides(frame, sides)),
    }


def _rejections(views: dict[str, Any], per_fold: list[dict[str, Any]]) -> list[str]:
    reasons = []
    for name, evidence in views.items():
        if evidence["selected_count"] < MIN_TRAINING_SAMPLE_COUNT:
            reasons.append(f"{name}_insufficient_selected_count")
        if (evidence.get("avg_return_pct") or 0.0) <= 0.0:
            reasons.append(f"{name}_nonpositive_net_mean")
        if (evidence.get("return_lcb_pct") or 0.0) <= 0.0:
            reasons.append(f"{name}_nonpositive_naive_lcb")
        if (evidence.get("profit_factor") or 0.0) <= 1.0:
            reasons.append(f"{name}_profit_factor_not_above_one")
    if not all(
            fold["nonoverlapping"]["selected_count"] > 0
            and (fold["nonoverlapping"].get("avg_return_pct") or 0.0) > 0.0
            and (fold["all_opportunities"].get("avg_return_pct") or 0.0) > 0.0
            for fold in per_fold
    ):
        reasons.append("inner_fold_fee_after_instability")
    return reasons


def _select_policy(train: pd.DataFrame) -> dict[str, Any]:
    folds = _folds(train, 2)
    if len(folds) != 2:
        return {
            "policy": None, "reason": "insufficient_inner_folds",
            "inner_folds": [], "candidates": [],
        }
    if any(
        len(_at_horizon(inner_train, horizon).query("sample_weight > 0"))
        < MIN_TRAINING_SAMPLE_COUNT
        for inner_train, _, _ in folds for horizon in HORIZONS
    ):
        return {
            "policy": None, "reason": "insufficient_positive_weight_inner_training",
            "inner_folds": [boundary for _, _, boundary in folds], "candidates": [],
        }
    candidates = []
    for horizon in HORIZONS:
        validations = [_at_horizon(validation, horizon) for _, validation, _ in folds]
        evaluation = pd.concat(validations)
        for feature_policy in FEATURE_POLICIES:
            for alpha in ALPHAS:
                predictions = [
                    _predict(
                        _at_horizon(inner_train, horizon), validation,
                        alpha=alpha, feature_policy=feature_policy,
                    )
                    for (inner_train, _, _), validation in zip(
                        folds, validations, strict=True
                    )
                ]
                for threshold in NET_THRESHOLDS_PCT:
                    sides = [
                        _sides(prediction, threshold) for prediction in predictions
                    ]
                    evidence = _views(evaluation, np.concatenate(sides))
                    per_fold = [
                        _views(validation, selection)
                        for validation, selection in zip(validations, sides, strict=True)
                    ]
                    rejections = _rejections(evidence, per_fold)
                    candidates.append({
                        "horizon_minutes": horizon,
                        "feature_policy": feature_policy,
                        "alpha": alpha,
                        "net_threshold_pct": threshold,
                        "inner_selection_eligible": not rejections,
                        "rejection_reasons": rejections,
                        "evidence": evidence,
                        "per_fold": per_fold,
                    })
    best = max(
        (row for row in candidates if row["inner_selection_eligible"]),
        key=lambda row: (
            row["evidence"]["nonoverlapping"]["average_return_per_opportunity_pct"],
            row["evidence"]["nonoverlapping"]["return_lcb_pct"],
            -row["horizon_minutes"], -row["alpha"], -row["net_threshold_pct"],
        ),
        default=None,
    )
    return {
        "policy": {
            key: best[key]
            for key in ("horizon_minutes", "feature_policy", "alpha", "net_threshold_pct")
        } if best else None,
        "reason": "inner_evidence_qualified" if best else "no_inner_policy_qualified",
        "candidate_rejection_counts": dict(Counter(
            reason for candidate in candidates
            for reason in candidate["rejection_reasons"]
        )),
        "inner_folds": [boundary for _, _, boundary in folds],
        "candidates": candidates,
    }


def build_horizon_development_report(
    development: pd.DataFrame, *, snapshot: dict[str, Any],
) -> dict[str, Any]:
    if _development_input_sha256(development) != snapshot["development_input_sha256"]:
        raise ValueError("research must use the exact frozen development frame")
    if development.duplicated(["decision_group", "horizon_minutes"]).any():
        raise ValueError("one observation per group and horizon required")
    coverage = development.groupby("decision_group")["horizon_minutes"].agg(set)
    complete = set(coverage[coverage.map(lambda values: set(HORIZONS) <= values)].index)
    # Purging always sees all captured horizons, including horizons outside a policy.
    common = development[development["decision_group"].isin(complete)].copy()
    report: dict[str, Any] = {
        "version": HORIZON_POLICY_VERSION,
        "status": "insufficient_development_data",
        "diagnostic_only": True, "artifact_persisted": False,
        "promotion_authorized": False, "submits_orders": False,
        "exam_labels_evaluated": False,
        "snapshot": {key: value for key, value in snapshot.items() if key != "records"},
        "common_horizon_decision_group_count": len(complete),
        "incomplete_horizon_group_count": len(coverage) - len(complete),
        "common_horizon_row_count": len(common),
        "search_space": {
            "model": "standardized_ridge_net", "alphas": list(ALPHAS),
            "net_thresholds_pct": list(NET_THRESHOLDS_PCT),
            "horizons_minutes": list(HORIZONS),
            "feature_policies": {name: list(keys) for name, keys in FEATURE_POLICIES.items()},
        },
        "selection_policy": (
            "two_purged_inner_folds_only; both_all_and_nonoverlapping_views:"
            "minimum_16_selected_positive_lcb_pf_above_one;"
            "positive_means_and_nonzero_selections_in_each_inner_fold_else_hold"
        ),
        "nonoverlap_policy": "one_global_position_ordered_by_decision_time_and_group",
        "metric_limitations": (
            "development counterfactuals, not settled portfolio PnL; "
            "nonoverlapping windows can still be serially correlated; "
            "naive LCB is a research filter, not independent profitability evidence"
        ),
        "folds": [],
    }
    if common.empty:
        return report
    outer_folds = _folds(common, 3)
    if len(outer_folds) != 3:
        return report
    evaluated_frames, evaluated_sides = [], []
    baselines: dict[int, list[pd.DataFrame]] = {horizon: [] for horizon in HORIZONS}
    for train, validation, boundary in outer_folds:
        selection = _select_policy(train)
        policy = selection["policy"]
        horizon = policy["horizon_minutes"] if policy else HORIZONS[0]
        evaluation = _at_horizon(validation, horizon)
        sides = _sides(
            _predict(
                _at_horizon(train, horizon), evaluation,
                alpha=policy["alpha"], feature_policy=policy["feature_policy"],
            ), policy["net_threshold_pct"],
        ) if policy else np.full(len(evaluation), "hold")
        report["folds"].append({
            **boundary, "selection": selection,
            "outer_evidence": _views(evaluation, sides),
        })
        evaluated_frames.append(evaluation)
        evaluated_sides.append(sides)
        for horizon in HORIZONS:
            baselines[horizon].append(_at_horizon(validation, horizon))
    evaluated = pd.concat(evaluated_frames)
    report["aggregate_outer_evidence"] = _views(
        evaluated, np.concatenate(evaluated_sides)
    )
    report["selected_policy_fold_count"] = sum(
        fold["selection"]["policy"] is not None for fold in report["folds"]
    )
    report["fixed_baselines_by_horizon"] = {
        str(horizon): {
            side: _views(
                pd.concat(frames), np.full(sum(map(len, frames)), side),
            ) for side in ("long", "short")
        } for horizon, frames in baselines.items()
    }
    report["status"] = "complete"
    return report
