"""Chronological evaluation of a fixed-side development-only policy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from services.ml_development_baselines import (
    _development_input_sha256,
    _evidence,
)
from services.ml_horizon_development import _serial_sides
from services.ml_signal_service import _decision_group_availability

MIN_FOLD_SAMPLE_COUNT = 16
FOLD_COUNT = 3


def build_fixed_direction_development_report(
    frame: pd.DataFrame,
    *,
    horizon_minutes: int,
    side: str,
) -> dict[str, Any]:
    """Score one frozen side/horizon policy on purged chronological folds."""

    horizon = int(horizon_minutes)
    direction = str(side).strip().lower()
    if horizon <= 0:
        raise ValueError("horizon_minutes must be positive")
    if direction not in {"long", "short"}:
        raise ValueError("side must be long or short")

    selected = frame.loc[frame["horizon_minutes"] == horizon].copy()
    if selected.empty:
        return {
            "status": "insufficient_development_data",
            "reason": "no_rows_for_registered_horizon",
            "horizon_minutes": horizon,
            "side": direction,
            "row_count": 0,
            "decision_group_count": 0,
            "folds": [],
        }
    if selected["decision_group"].duplicated().any():
        raise ValueError("one observation per decision group and horizon required")

    groups, bounds = _decision_group_availability(selected)
    initial_count = max(int(np.ceil(len(groups) * 0.5)), 1)
    validation_chunks = np.array_split(np.asarray(groups[initial_count:]), FOLD_COUNT)
    if any(not chunk.size for chunk in validation_chunks):
        return {
            "status": "insufficient_development_data",
            "reason": "insufficient_validation_groups",
            "horizon_minutes": horizon,
            "side": direction,
            "row_count": len(selected),
            "decision_group_count": len(groups),
            "folds": [],
        }

    folds: list[dict[str, Any]] = []
    validation_frames: list[pd.DataFrame] = []
    validation_sides: list[np.ndarray] = []
    for fold_index, chunk in enumerate(validation_chunks):
        validation_groups = {str(value) for value in chunk.tolist()}
        validation_start = min(bounds[group]["decision_start"] for group in validation_groups)
        training_groups = {
            group for group in groups if bounds[group]["end"] < validation_start
        }
        training_end = max(
            (bounds[group]["end"] for group in training_groups),
            default=None,
        )
        validation = selected[
            selected["decision_group"].astype(str).isin(validation_groups)
        ]
        label_disjoint = training_end is not None and training_end < validation_start
        if training_groups.intersection(validation_groups) or not label_disjoint:
            raise ValueError("fixed-side development fold is not chronologically purged")

        sides = np.full(len(validation), direction, dtype=object)
        all_opportunities = _evidence(validation, sides)
        nonoverlapping = _evidence(validation, _serial_sides(validation, sides))
        validation_frames.append(validation)
        validation_sides.append(sides)
        folds.append(
            {
                "fold_index": fold_index,
                "training_group_count": len(training_groups),
                "validation_group_count": len(validation_groups),
                "training_label_end": training_end.isoformat() if training_end else None,
                "validation_decision_start": validation_start.isoformat(),
                "decision_group_overlap_count": 0,
                "chronological_label_disjoint": label_disjoint,
                "all_opportunities": all_opportunities,
                "nonoverlapping": nonoverlapping,
            }
        )

    # Fold order is already chronological, so concatenation preserves the
    # fixed policy's evaluation order and keeps side labels aligned.
    aggregate_validation = pd.concat(validation_frames, ignore_index=True)
    aggregate_sides = np.concatenate(validation_sides)
    aggregate = {
        "all_opportunities": _evidence(aggregate_validation, aggregate_sides),
        "nonoverlapping": _evidence(
            aggregate_validation,
            _serial_sides(aggregate_validation, aggregate_sides),
        ),
    }
    required_views = ("all_opportunities", "nonoverlapping")
    fold_criteria = [
        all(
            int(fold[view].get("selected_count") or 0) >= MIN_FOLD_SAMPLE_COUNT
            and float(fold[view].get("avg_return_pct") or 0.0) > 0.0
            and float(fold[view].get("return_lcb_pct") or 0.0) > 0.0
            and float(fold[view].get("profit_factor") or 0.0) > 1.0
            for view in required_views
        )
        for fold in folds
    ]
    aggregate_criteria = all(
        int(aggregate[view].get("selected_count") or 0) >= MIN_FOLD_SAMPLE_COUNT
        and float(aggregate[view].get("avg_return_pct") or 0.0) > 0.0
        and float(aggregate[view].get("return_lcb_pct") or 0.0) > 0.0
        and float(aggregate[view].get("profit_factor") or 0.0) > 1.0
        for view in required_views
    )
    sufficient = all(
        fold["training_group_count"] >= MIN_FOLD_SAMPLE_COUNT
        and fold["validation_group_count"] >= MIN_FOLD_SAMPLE_COUNT
        for fold in folds
    )
    passed = sufficient and aggregate_criteria and all(fold_criteria)
    return {
        "status": "development_candidate_qualified" if passed else "rejected",
        "reason": (
            "all_registered_fee_after_rules_passed"
            if passed
            else "insufficient_fold_coverage"
            if not sufficient
            else "fee_after_stability_rules_failed"
        ),
        "policy": {
            "type": "fixed_side_horizon",
            "side": direction,
            "horizon_minutes": horizon,
            "selection_or_tuning": False,
            "entry_costs": "recorded_native_fee_after_complete_labels",
            "position_policy": "one_global_position_for_nonoverlapping_view",
        },
        "row_count": len(selected),
        "decision_group_count": len(groups),
        "development_input_sha256": _development_input_sha256(selected),
        "minimum_selected_count_per_view_per_fold": MIN_FOLD_SAMPLE_COUNT,
        "fold_count": FOLD_COUNT,
        "folds": folds,
        "aggregate_outer_evidence": aggregate,
        "metric_limitations": (
            "all rows are development data; the naive LCB is only a research filter, "
            "not an independent confidence bound or profitability guarantee"
        ),
        "exam_labels_evaluated": False,
        "artifact_persisted": False,
        "database_mutated": False,
        "promotion_authorized": False,
        "orders_submitted": False,
        "runtime_policy_changed": False,
    }
