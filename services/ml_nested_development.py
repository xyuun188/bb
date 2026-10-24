"""Bounded nested development research; no model persistence or order access."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from services.ml_development_baselines import (
    _baseline_model,
    _development_input_sha256,
    _evidence,
    _net_returns,
)
from services.ml_signal_service import (
    FEATURE_KEYS,
    MIN_TRAINING_SAMPLE_COUNT,
    PRIMARY_PREDICTION_HORIZON_MINUTES,
    _decision_group_availability,
    decision_group_partition,
)

NESTED_POLICY_VERSION = "2026-10-21.nested-development-fixed-ridge.v1"
ALPHAS = (1.0, 10.0, 100.0)
NET_THRESHOLDS_PCT = (0.0, 0.05, 0.1)


def _folds(
    frame: pd.DataFrame,
    count: int,
    *,
    purge_horizon_minutes: int | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    # Fixed-horizon research must purge on the labels it actually fits. A
    # long-horizon row in the same decision group is irrelevant to a primary
    # five-minute target and would otherwise erase usable early training data.
    boundary_frame = frame
    if purge_horizon_minutes is not None:
        boundary_frame = frame[
            frame["horizon_minutes"] == int(purge_horizon_minutes)
        ].copy()
    groups, bounds = _decision_group_availability(boundary_frame)
    initial = max(int(np.ceil(len(groups) * 0.5)), 1)
    result = []
    for chunk in np.array_split(np.asarray(groups[initial:]), count):
        validation_groups = chunk.tolist()
        if not validation_groups:
            return []
        start = min(bounds[group]["decision_start"] for group in validation_groups)
        train_groups = {
            group for group in groups if bounds[group]["end"] < start
        }
        train = frame[frame["decision_group"].astype(str).isin(train_groups)].copy()
        validation = frame[
            frame["decision_group"].astype(str).isin(validation_groups)
        ].copy()
        primary_train = train[
            train["horizon_minutes"] == PRIMARY_PREDICTION_HORIZON_MINUTES
        ]
        primary_validation = validation[
            validation["horizon_minutes"] == PRIMARY_PREDICTION_HORIZON_MINUTES
        ]
        if len(primary_train) < MIN_TRAINING_SAMPLE_COUNT or primary_validation.empty:
            return []
        if primary_train["decision_group"].duplicated().any() or (
            primary_validation["decision_group"].duplicated().any()
        ):
            raise ValueError("one primary row per decision group is required")
        label_end = max(bounds[group]["end"] for group in train_groups)
        if label_end >= start or train_groups.intersection(validation_groups):
            raise ValueError("nested development label periods overlap")
        result.append((train, validation, {
            "train_primary_sample_count": len(primary_train),
            "validation_primary_sample_count": len(primary_validation),
            "training_label_end": str(label_end),
            "validation_decision_start": str(start),
            "decision_group_overlap_count": 0,
            "chronological_label_disjoint": True,
        }))
    return result


def _primary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame["horizon_minutes"] == PRIMARY_PREDICTION_HORIZON_MINUTES
    ].copy()


def _predict(
    train: pd.DataFrame, validation: pd.DataFrame, alpha: float,
) -> dict[str, np.ndarray]:
    predictions = {}
    weights = train.get("sample_weight", pd.Series(1.0, index=train.index))
    for side in ("long", "short"):
        target = _net_returns(train, side)
        model = _baseline_model("ridge_net", target)
        model.set_params(model__alpha=alpha)
        model.fit(train[FEATURE_KEYS], target, model__sample_weight=weights)
        predictions[side] = np.asarray(
            model.predict(validation[FEATURE_KEYS]), dtype=float
        )
        if not np.isfinite(predictions[side]).all():
            raise ValueError("non-finite nested development prediction")
    return predictions


def _sides(predictions: dict[str, np.ndarray], threshold: float) -> np.ndarray:
    long = predictions["long"]
    short = predictions["short"]
    return np.where(
        np.maximum(long, short) <= threshold,
        "hold",
        np.where(long >= short, "long", "short"),
    )


def _select_policy(train: pd.DataFrame) -> dict[str, Any]:
    folds = _folds(
        train,
        2,
        purge_horizon_minutes=PRIMARY_PREDICTION_HORIZON_MINUTES,
    )
    if len(folds) != 2:
        return {
            "policy": None, "reason": "insufficient_inner_folds",
            "inner_folds": [], "candidates": [],
        }
    frames = [_primary(validation) for _, validation, _ in folds]
    evaluation = pd.concat(frames)
    candidates = []
    for alpha in ALPHAS:
        predictions = [
            _predict(_primary(inner_train), validation, alpha)
            for (inner_train, _, _), validation in zip(folds, frames, strict=True)
        ]
        for threshold in NET_THRESHOLDS_PCT:
            sides = [_sides(prediction, threshold) for prediction in predictions]
            evidence = _evidence(evaluation, np.concatenate(sides))
            per_fold = [
                _evidence(validation, selection)
                for validation, selection in zip(frames, sides, strict=True)
            ]
            # This chooses a research policy only. Correlated shadow evidence
            # still cannot establish independent profitability or promotion.
            eligible = bool(
                evidence.get("promotion_math_ready") is True
                and evidence["selected_count"] >= MIN_TRAINING_SAMPLE_COUNT
                and all(
                    fold["selected_count"] > 0
                    and fold.get("avg_return_pct", 0.0) > 0.0
                    for fold in per_fold
                )
            )
            candidates.append({
                "alpha": alpha,
                "net_threshold_pct": threshold,
                "inner_selection_eligible": eligible,
                "evidence": evidence,
                "per_fold": per_fold,
            })
    eligible = [
        candidate for candidate in candidates if candidate["inner_selection_eligible"]
    ]
    best = max(
        eligible,
        key=lambda row: (
            row["evidence"]["average_return_per_opportunity_pct"],
            row["evidence"]["return_lcb_pct"],
            -row["alpha"],
            -row["net_threshold_pct"],
        ),
        default=None,
    )
    return {
        "policy": (
            {key: best[key] for key in ("alpha", "net_threshold_pct")} if best else None
        ),
        "reason": "inner_evidence_qualified" if best else "no_inner_policy_qualified",
        "inner_folds": [boundary for _, _, boundary in folds],
        "candidates": candidates,
    }


def build_nested_development_report(frame: pd.DataFrame) -> dict[str, Any]:
    primary_frame = frame[
        frame["horizon_minutes"] == PRIMARY_PREDICTION_HORIZON_MINUTES
    ].copy()
    partition = decision_group_partition(primary_frame)
    report: dict[str, Any] = {
        "version": NESTED_POLICY_VERSION,
        "status": "insufficient_development_data",
        "diagnostic_only": True,
        "artifact_persisted": False,
        "promotion_authorized": False,
        "submits_orders": False,
        "exam_labels_evaluated": False,
        "outer_partition": partition.report,
        "horizon_minutes": PRIMARY_PREDICTION_HORIZON_MINUTES,
        "search_space": {
            "model": "standardized_ridge_net", "alphas": list(ALPHAS),
            "net_thresholds_pct": list(NET_THRESHOLDS_PCT),
            "feature_policy": "existing_feature_keys_no_feature_search",
        },
        "selection_policy": (
            "inner_two_purged_folds_only; positive_fee_after_lcb_pf_above_one;"
            "minimum_16_selections_and_positive_mean_in_both_folds_else_no_trade"
        ),
        "metric_limitations": (
            "development shadow opportunities with overlapping holding windows; "
            "not portfolio PnL, independent profitability or promotion evidence"
        ),
        "folds": [],
    }
    if not partition.report["ready"]:
        return report
    development = partition.train
    report["development_input_sha256"] = _development_input_sha256(development)
    outer_folds = _folds(
        development,
        3,
        purge_horizon_minutes=PRIMARY_PREDICTION_HORIZON_MINUTES,
    )
    if len(outer_folds) != 3:
        return report
    evaluated_frames, evaluated_sides = [], []
    for train, validation, boundary in outer_folds:
        selection = _select_policy(train)
        evaluation = _primary(validation)
        policy = selection["policy"]
        sides = (
            _sides(
                _predict(_primary(train), evaluation, policy["alpha"]),
                policy["net_threshold_pct"],
            ) if policy else np.full(len(evaluation), "hold")
        )
        report["folds"].append({
            **boundary,
            "selection": selection,
            "outer_evidence": _evidence(evaluation, sides),
        })
        evaluated_frames.append(evaluation)
        evaluated_sides.append(sides)
    report["aggregate_outer_evidence"] = _evidence(
        pd.concat(evaluated_frames), np.concatenate(evaluated_sides)
    )
    report["selected_policy_fold_count"] = sum(
        fold["selection"]["policy"] is not None for fold in report["folds"]
    )
    report["status"] = "complete"
    return report
