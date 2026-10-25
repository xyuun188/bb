"""Bounded nested development research for a fixed histogram-gradient model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from services.ml_development_baselines import (
    _development_input_sha256,
    _evidence,
    _net_returns,
)
from services.ml_nested_development import _folds, _primary, _sides
from services.ml_signal_service import (
    FEATURE_KEYS,
    MIN_TRAINING_SAMPLE_COUNT,
    PRIMARY_PREDICTION_HORIZON_MINUTES,
    decision_group_partition,
)

NESTED_HGB_POLICY_VERSION = "2026-10-25.nested-development-fixed-hgb.v1"
HGB_MAX_ITER = 160
HGB_LEARNING_RATE = 0.05
HGB_MAX_LEAF_NODES = 15
HGB_MIN_SAMPLES_LEAF = 24
HGB_L2_REGULARIZATION = 1.0
HGB_RANDOM_STATE = 42
NET_THRESHOLDS_PCT = (0.0, 0.05, 0.1)


def _model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=HGB_MAX_ITER,
                    learning_rate=HGB_LEARNING_RATE,
                    max_leaf_nodes=HGB_MAX_LEAF_NODES,
                    min_samples_leaf=HGB_MIN_SAMPLES_LEAF,
                    l2_regularization=HGB_L2_REGULARIZATION,
                    early_stopping=False,
                    random_state=HGB_RANDOM_STATE,
                ),
            ),
        ]
    )


def _predict(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    weights = train.get("sample_weight", pd.Series(1.0, index=train.index))
    for side in ("long", "short"):
        model = _model()
        model.fit(
            train[FEATURE_KEYS],
            _net_returns(train, side),
            model__sample_weight=weights,
        )
        predicted = np.asarray(model.predict(validation[FEATURE_KEYS]), dtype=float)
        if not np.isfinite(predicted).all():
            raise ValueError("non-finite histogram-gradient development prediction")
        predictions[side] = predicted
    return predictions


def _select_policy(train: pd.DataFrame) -> dict[str, Any]:
    folds = _folds(
        train,
        2,
        purge_horizon_minutes=PRIMARY_PREDICTION_HORIZON_MINUTES,
    )
    if len(folds) != 2:
        return {
            "policy": None,
            "reason": "insufficient_inner_folds",
            "inner_folds": [],
            "candidates": [],
        }
    frames = [_primary(validation) for _, validation, _ in folds]
    evaluation = pd.concat(frames)
    predictions = [
        _predict(_primary(inner_train), validation)
        for (inner_train, _, _), validation in zip(folds, frames, strict=True)
    ]
    candidates = []
    for threshold in NET_THRESHOLDS_PCT:
        sides = [_sides(prediction, threshold) for prediction in predictions]
        evidence = _evidence(evaluation, np.concatenate(sides))
        per_fold = [
            _evidence(validation, selection)
            for validation, selection in zip(frames, sides, strict=True)
        ]
        eligible = bool(
            evidence.get("promotion_math_ready") is True
            and evidence["selected_count"] >= MIN_TRAINING_SAMPLE_COUNT
            and all(
                fold["selected_count"] > 0
                and fold.get("avg_return_pct", 0.0) > 0.0
                for fold in per_fold
            )
        )
        candidates.append(
            {
                "net_threshold_pct": threshold,
                "inner_selection_eligible": eligible,
                "evidence": evidence,
                "per_fold": per_fold,
            }
        )
    eligible = [
        candidate for candidate in candidates
        if candidate["inner_selection_eligible"]
    ]
    best = max(
        eligible,
        key=lambda row: (
            row["evidence"]["average_return_per_opportunity_pct"],
            row["evidence"]["return_lcb_pct"],
            -row["net_threshold_pct"],
        ),
        default=None,
    )
    return {
        "policy": (
            {"net_threshold_pct": best["net_threshold_pct"]}
            if best
            else None
        ),
        "reason": (
            "inner_evidence_qualified"
            if best
            else "no_inner_policy_qualified"
        ),
        "inner_folds": [boundary for _, _, boundary in folds],
        "candidates": candidates,
    }


def build_nested_hgb_development_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Evaluate one fixed HGB family on development folds only."""

    primary_frame = frame[
        frame["horizon_minutes"] == PRIMARY_PREDICTION_HORIZON_MINUTES
    ].copy()
    partition = decision_group_partition(primary_frame)
    report: dict[str, Any] = {
        "version": NESTED_HGB_POLICY_VERSION,
        "status": "insufficient_development_data",
        "diagnostic_only": True,
        "artifact_persisted": False,
        "promotion_authorized": False,
        "submits_orders": False,
        "exam_labels_evaluated": False,
        "outer_partition": partition.report,
        "horizon_minutes": PRIMARY_PREDICTION_HORIZON_MINUTES,
        "search_space": {
            "model": "hist_gradient_boosting_regressor_net",
            "max_iter": HGB_MAX_ITER,
            "learning_rate": HGB_LEARNING_RATE,
            "max_leaf_nodes": HGB_MAX_LEAF_NODES,
            "min_samples_leaf": HGB_MIN_SAMPLES_LEAF,
            "l2_regularization": HGB_L2_REGULARIZATION,
            "early_stopping": False,
            "random_state": HGB_RANDOM_STATE,
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

    evaluated_frames: list[pd.DataFrame] = []
    evaluated_sides: list[np.ndarray] = []
    for train, validation, boundary in outer_folds:
        selection = _select_policy(train)
        evaluation = _primary(validation)
        policy = selection["policy"]
        sides = (
            _sides(_predict(_primary(train), evaluation), policy["net_threshold_pct"])
            if policy
            else np.full(len(evaluation), "hold")
        )
        report["folds"].append(
            {
                **boundary,
                "selection": selection,
                "outer_evidence": _evidence(evaluation, sides),
            }
        )
        evaluated_frames.append(evaluation)
        evaluated_sides.append(sides)

    report["aggregate_outer_evidence"] = _evidence(
        pd.concat(evaluated_frames),
        np.concatenate(evaluated_sides),
    )
    report["selected_policy_fold_count"] = sum(
        fold["selection"]["policy"] is not None for fold in report["folds"]
    )
    report["status"] = "complete"
    return report
