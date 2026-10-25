"""Chronological evaluation of the fixed technical-consensus canary policy."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from services.live_rules_canary_signal import build_live_rules_canary_signal
from services.ml_development_baselines import _development_input_sha256, _evidence
from services.ml_horizon_development import _serial_sides
from services.ml_signal_service import _decision_group_availability

RULES_CANARY_DEVELOPMENT_VERSION = (
    "2026-10-25.fixed-live-rules-canary-development.v1"
)
RULES_CANARY_PRIMARY_HORIZON_MINUTES = 5
MIN_FOLD_SAMPLE_COUNT = 16
FOLD_COUNT = 3


def rules_canary_side(snapshot: dict[str, Any] | None) -> str:
    """Return the immutable rules-canary side for one frozen feature snapshot."""

    signal = build_live_rules_canary_signal(snapshot)
    action = str(signal.get("action") or "hold").lower()
    return action if action in {"long", "short"} else "hold"


def filter_rules_canary_decision_window(
    frame: pd.DataFrame,
    *,
    since: datetime,
    cutoff: datetime,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select observations by frozen decision time, not label update time."""

    if since.tzinfo is None or cutoff.tzinfo is None or since >= cutoff:
        raise ValueError("decision window must have ordered timezone-aware bounds")
    row_count = len(frame)
    if row_count == 0 or "decision_timestamp" not in frame:
        return frame.iloc[0:0].copy(), {
            "decision_timestamp_source": "feature_snapshot.original_decision_timestamp",
            "candidate_row_count": row_count,
            "valid_decision_timestamp_row_count": 0,
            "selected_row_count": 0,
            "missing_decision_timestamp_row_count": row_count,
            "outside_decision_window_row_count": 0,
        }

    timestamps = pd.to_datetime(frame["decision_timestamp"], utc=True, errors="coerce")
    valid = timestamps.notna()
    in_window = valid & (timestamps >= pd.Timestamp(since)) & (
        timestamps < pd.Timestamp(cutoff)
    )
    return frame.loc[in_window].copy(), {
        "decision_timestamp_source": "feature_snapshot.original_decision_timestamp",
        "candidate_row_count": row_count,
        "valid_decision_timestamp_row_count": int(valid.sum()),
        "selected_row_count": int(in_window.sum()),
        "missing_decision_timestamp_row_count": int((~valid).sum()),
        "outside_decision_window_row_count": int((valid & ~in_window).sum()),
    }


def _fold_view(frame: pd.DataFrame, sides: np.ndarray) -> dict[str, Any]:
    return {
        "all_opportunities": _evidence(frame, sides),
        "nonoverlapping": _evidence(frame, _serial_sides(frame, sides)),
    }


def _view_passes(view: dict[str, Any]) -> bool:
    return (
        int(view.get("selected_count") or 0) >= MIN_FOLD_SAMPLE_COUNT
        and float(view.get("avg_return_pct") or 0.0) > 0.0
        and float(view.get("return_lcb_pct") or 0.0) > 0.0
        and float(view.get("profit_factor") or 0.0) > 1.0
    )


def build_rules_canary_development_report(
    frame: pd.DataFrame,
    *,
    snapshots_by_id: dict[int, dict[str, Any]] | None = None,
    horizon_minutes: int = RULES_CANARY_PRIMARY_HORIZON_MINUTES,
) -> dict[str, Any]:
    """Evaluate one fixed rules policy on purged chronological folds."""

    horizon = int(horizon_minutes)
    snapshots = snapshots_by_id or {}
    selected = frame.loc[frame["horizon_minutes"] == horizon].copy()
    base = {
        "version": RULES_CANARY_DEVELOPMENT_VERSION,
        "diagnostic_only": True,
        "policy": {
            "type": "fixed_live_rules_canary_technical_consensus",
            "signal_version": "2026-07-23.live-rules-canary-signal.v1",
            "horizon_minutes": horizon,
            "selection_or_tuning": False,
            "entry_costs": "recorded_native_fee_after_complete_labels",
            "position_policy": "one_global_position_for_nonoverlapping_view",
            "orders_submitted": False,
        },
        "horizon_minutes": horizon,
        "exam_labels_evaluated": False,
        "artifact_persisted": False,
        "database_mutated": False,
        "orders_submitted": False,
        "promotion_authorized": False,
        "runtime_policy_changed": False,
    }
    if selected.empty:
        return {
            **base,
            "status": "insufficient_development_data",
            "reason": "no_rows_for_registered_horizon",
            "row_count": 0,
            "decision_group_count": 0,
            "folds": [],
        }
    if selected["decision_group"].duplicated().any():
        raise ValueError("one observation per decision group and horizon required")

    selected["rules_side"] = [
        rules_canary_side(snapshots.get(int(sample_id or 0)))
        for sample_id in selected["id"]
    ]
    groups, bounds = _decision_group_availability(selected)
    initial_count = max(int(np.ceil(len(groups) * 0.5)), 1)
    validation_chunks = np.array_split(np.asarray(groups[initial_count:]), FOLD_COUNT)
    if any(not chunk.size for chunk in validation_chunks):
        return {
            **base,
            "status": "insufficient_development_data",
            "reason": "insufficient_validation_groups",
            "row_count": len(selected),
            "decision_group_count": len(groups),
            "folds": [],
        }

    folds: list[dict[str, Any]] = []
    validation_frames: list[pd.DataFrame] = []
    for fold_index, chunk in enumerate(validation_chunks):
        validation_groups = {str(value) for value in chunk.tolist()}
        validation_start = min(
            bounds[group]["decision_start"] for group in validation_groups
        )
        training_groups = {
            group for group in groups if bounds[group]["end"] < validation_start
        }
        training_end = max(
            (bounds[group]["end"] for group in training_groups),
            default=None,
        )
        validation = selected[
            selected["decision_group"].astype(str).isin(validation_groups)
        ].copy()
        label_disjoint = training_end is not None and training_end < validation_start
        if training_groups.intersection(validation_groups) or not label_disjoint:
            raise ValueError("rules-canary development fold is not chronologically purged")
        sides = validation["rules_side"].to_numpy(dtype=object)
        views = _fold_view(validation, sides)
        validation_frames.append(validation)
        folds.append(
            {
                "fold_index": fold_index,
                "training_group_count": len(training_groups),
                "validation_group_count": len(validation_groups),
                "training_label_end": training_end.isoformat() if training_end else None,
                "validation_decision_start": validation_start.isoformat(),
                "decision_group_overlap_count": 0,
                "chronological_label_disjoint": label_disjoint,
                "signal_selected_count": int(sum(side in {"long", "short"} for side in sides)),
                **views,
            }
        )

    aggregate_validation = pd.concat(validation_frames, ignore_index=True)
    aggregate_sides = aggregate_validation["rules_side"].to_numpy(dtype=object)
    aggregate = _fold_view(aggregate_validation, aggregate_sides)
    required_views = ("all_opportunities", "nonoverlapping")
    fold_criteria = [
        all(_view_passes(fold[view]) for view in required_views)
        for fold in folds
    ]
    aggregate_criteria = all(
        _view_passes(aggregate[view]) for view in required_views
    )
    sufficient = all(
        fold["training_group_count"] >= MIN_FOLD_SAMPLE_COUNT
        and fold["validation_group_count"] >= MIN_FOLD_SAMPLE_COUNT
        for fold in folds
    )
    passed = sufficient and aggregate_criteria and all(fold_criteria)
    return {
        **base,
        "status": "development_candidate_qualified" if passed else "rejected",
        "reason": (
            "all_registered_fee_after_rules_passed"
            if passed
            else "insufficient_fold_coverage"
            if not sufficient
            else "fee_after_stability_rules_failed"
        ),
        "row_count": len(selected),
        "decision_group_count": len(groups),
        "signal_selected_count": int(
            (selected["rules_side"].isin(["long", "short"])).sum()
        ),
        "signal_selected_share": float(
            (selected["rules_side"].isin(["long", "short"])).mean()
        ),
        "development_input_sha256": _development_input_sha256(selected),
        "minimum_selected_count_per_view_per_fold": MIN_FOLD_SAMPLE_COUNT,
        "fold_count": FOLD_COUNT,
        "folds": folds,
        "aggregate_outer_evidence": aggregate,
        "metric_limitations": (
            "all rows are development data; the research lower bound is a filter, "
            "not an independent confidence bound or profitability guarantee"
        ),
    }
