"""Compare fixed baselines inside the training partition without opening the exam."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from services.ml_signal_service import (
    FEATURE_KEYS,
    MIN_TRAINING_SAMPLE_COUNT,
    PRIMARY_PREDICTION_HORIZON_MINUTES,
    _decision_group_availability,
    _make_regressor,
    _return_evidence,
    decision_group_partition,
)

BASELINE_POLICY_VERSION = "2026-10-21.development-only-fixed-baselines.v1"
FOLD_COUNT = 3
INITIAL_TRAIN_SHARE = 0.5


def _net_returns(frame: pd.DataFrame, side: str) -> pd.Series:
    return frame[f"{side}_return_pct"].astype(float) - frame[
        f"{side}_execution_cost_pct"
    ].astype(float)


def _baseline_model(name: str, target: pd.Series) -> Pipeline:
    if name == "ridge_net":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
    if name == "existing_tree_net":
        return _make_regressor(target)
    raise ValueError(f"unknown development baseline: {name}")


def _evidence(
    frame: pd.DataFrame, sides: np.ndarray,
) -> dict[str, Any]:
    selected = []
    for offset, (_, row) in enumerate(frame.iterrows()):
        side = str(sides[offset])
        if side not in {"long", "short"}:
            continue
        gross_return = float(row[f"{side}_return_pct"])
        execution_cost = float(row[f"{side}_execution_cost_pct"])
        selected.append({
            "decision_group": str(row["decision_group"]),
            "label_timestamp": str(row["label_timestamp"]),
            "gross_return_pct": gross_return,
            "execution_cost_pct": execution_cost,
            "return_pct": gross_return - execution_cost,
        })
    evidence = _return_evidence(selected)
    evidence["tail_loss_policy"].pop("generated_at", None)
    selected_count = len(selected)
    return {
        **evidence,
        "opportunity_count": int(len(frame)),
        "selected_count": selected_count,
        "selected_share": selected_count / len(frame) if len(frame) else 0.0,
        "average_return_per_opportunity_pct": (
            sum(row["return_pct"] for row in selected) / len(frame)
            if len(frame) else 0.0
        ),
        "mean_selected_gross_return_pct": (
            float(np.mean([row["gross_return_pct"] for row in selected]))
            if selected else None
        ),
        "mean_selected_execution_cost_pct": (
            float(np.mean([row["execution_cost_pct"] for row in selected]))
            if selected else None
        ),
        "selected_decision_group_count": len({
            row["decision_group"] for row in selected
        }),
    }


def _rank_only_evidence(
    frame: pd.DataFrame, predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    """A fixed score slice tests ranking even when every prediction is negative."""

    rows = []
    for index, (_, row) in enumerate(frame.iterrows()):
        side = "long" if predictions["long"][index] >= predictions["short"][index] else "short"
        rows.append({
            "decision_group": str(row["decision_group"]),
            "label_timestamp": str(row["label_timestamp"]),
            "return_pct": float(row[f"{side}_return_pct"])
            - float(row[f"{side}_execution_cost_pct"]),
            "side": side,
            "score": float(predictions[side][index]),
        })
    selected_count = max(math.ceil(len(rows) * 0.2), 1) if rows else 0
    selected = sorted(rows, key=lambda row: row["score"], reverse=True)[:selected_count]
    evidence = _return_evidence(selected)
    evidence["tail_loss_policy"].pop("generated_at", None)
    return {
        **evidence,
        "selection_policy": "fixed_top_20_percent_diagnostic_only",
        "opportunity_count": len(rows),
        "selected_count": len(selected),
        "selected_share": len(selected) / len(rows) if rows else 0.0,
        "selected_long_count": sum(row["side"] == "long" for row in selected),
        "selected_short_count": sum(row["side"] == "short" for row in selected),
        "mean_selected_predicted_net_return_pct": (
            float(np.mean([row["score"] for row in selected])) if selected else None
        ),
    }


def _prediction_diagnostics(
    actual: np.ndarray, predicted: np.ndarray,
) -> dict[str, Any]:
    actual_series = pd.Series(actual)
    predicted_series = pd.Series(predicted)
    rank_correlation = None
    if actual_series.nunique() > 1 and predicted_series.nunique() > 1:
        rank_correlation = float(
            actual_series.rank().corr(predicted_series.rank())
        )
    bins = []
    if len(actual):
        frame = pd.DataFrame({"predicted": predicted, "actual": actual})
        frame["bin"] = pd.qcut(
            frame["predicted"].rank(method="first"),
            q=min(5, len(actual)), labels=False, duplicates="drop",
        )
        for bin_index, group in frame.groupby("bin", sort=True):
            bins.append({
                "bin": int(bin_index),
                "sample_count": int(len(group)),
                "mean_predicted_net_return_pct": float(group["predicted"].mean()),
                "mean_actual_net_return_pct": float(group["actual"].mean()),
            })
    return {
        "sample_count": len(actual),
        "mean_actual_net_return_pct": float(np.mean(actual)),
        "mean_predicted_net_return_pct": float(np.mean(predicted)),
        "prediction_bias_pct": float(np.mean(predicted - actual)),
        "mean_absolute_error_pct": float(np.mean(np.abs(predicted - actual))),
        "spearman_rank_correlation": rank_correlation,
        "fixed_rank_calibration_bins": bins,
    }


def _development_input_sha256(development: pd.DataFrame) -> str:
    identity_columns = [
        "id", "decision_group", "decision_timestamp", "label_timestamp",
        "horizon_minutes", *FEATURE_KEYS, "long_return_pct", "short_return_pct",
        "long_execution_cost_pct", "short_execution_cost_pct",
    ]
    if "sample_weight" in development:
        identity_columns.append("sample_weight")
    return hashlib.sha256(
        json.dumps(
            development[list(dict.fromkeys(identity_columns))].to_dict("records"),
            sort_keys=True, default=str, allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def build_development_baseline_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Use purged expanding folds of development only; never fit on exam rows."""

    partition = decision_group_partition(frame)
    report: dict[str, Any] = {
        "version": BASELINE_POLICY_VERSION,
        "status": "insufficient_development_data",
        "diagnostic_only": True,
        "artifact_persisted": False,
        "promotion_authorized": False,
        "submits_orders": False,
        "horizon_minutes": PRIMARY_PREDICTION_HORIZON_MINUTES,
        "outer_partition": partition.report,
        "exam_labels_evaluated": False,
        "selection_policy": "choose_larger_positive_predicted_net_return_else_hold",
        "model_policy": "fixed_ridge_alpha_1_and_existing_tree_defaults_no_search",
        "metric_limitations": (
            "development shadow opportunities with overlapping holding windows; "
            "not executed portfolio PnL or independent profitability validation"
        ),
        "folds": [],
        "baselines": {},
    }
    if not partition.report["ready"]:
        return report
    development = partition.train.copy()
    groups, bounds = _decision_group_availability(development)
    initial_count = max(int(np.ceil(len(groups) * INITIAL_TRAIN_SHARE)), 1)
    validation_chunks = [
        chunk.tolist()
        for chunk in np.array_split(np.asarray(groups[initial_count:]), FOLD_COUNT)
        if len(chunk)
    ]
    primary = development[
        pd.to_numeric(development["horizon_minutes"], errors="coerce")
        == PRIMARY_PREDICTION_HORIZON_MINUTES
    ].copy()
    if primary.empty or len(validation_chunks) != FOLD_COUNT:
        return report
    if primary["decision_group"].astype(str).duplicated().any():
        raise ValueError("one primary-horizon row per decision group is required")
    report["development_input_sha256"] = _development_input_sha256(development)
    evaluated_frames: list[pd.DataFrame] = []
    all_predictions: dict[str, dict[str, list[np.ndarray]]] = {
        name: {side: [] for side in ("long", "short")}
        for name in ("ridge_net", "existing_tree_net")
    }
    for fold_index, validation_groups in enumerate(validation_chunks):
        validation_start = min(
            bounds[group]["decision_start"] for group in validation_groups
        )
        train_groups = {
            group for group in groups
            if bounds[group]["end"] < validation_start
            and bounds[group]["decision_start"] < validation_start
        }
        train = primary[
            primary["decision_group"].astype(str).isin(train_groups)
        ].copy()
        validation = primary[
            primary["decision_group"].astype(str).isin(validation_groups)
        ].copy()
        if len(train) < MIN_TRAINING_SAMPLE_COUNT or validation.empty:
            return report
        if train_groups.intersection(validation_groups):
            raise ValueError("development fold decision groups overlap")
        label_end = max(bounds[group]["end"] for group in train_groups)
        if label_end >= validation_start:
            raise ValueError("development fold label periods overlap")
        fold: dict[str, Any] = {
            "fold_index": fold_index,
            "train_sample_count": int(len(train)),
            "validation_sample_count": int(len(validation)),
            "training_label_end": str(label_end),
            "validation_decision_start": str(validation_start),
            "decision_group_overlap_count": 0,
            "chronological_label_disjoint": True,
            "models": {},
            "rank_only_top_20_percent": {},
        }
        weights = train.get(
            "sample_weight", pd.Series(1.0, index=train.index)
        ).astype(float)
        for name, side_predictions in all_predictions.items():
            predictions: dict[str, np.ndarray] = {}
            for side in ("long", "short"):
                target = _net_returns(train, side)
                model = _baseline_model(name, target)
                model.fit(
                    train[FEATURE_KEYS], target,
                    model__sample_weight=weights,
                )
                predictions[side] = np.asarray(
                    model.predict(validation[FEATURE_KEYS]), dtype=float
                )
                if not np.isfinite(predictions[side]).all():
                    raise ValueError("non-finite development baseline prediction")
                side_predictions[side].append(predictions[side])
            selected_sides = _selected_sides(predictions)
            fold["models"][name] = _evidence(validation, selected_sides)
            fold["rank_only_top_20_percent"][name] = _rank_only_evidence(
                validation, predictions
            )
        evaluated_frames.append(validation)
        report["folds"].append(fold)
    evaluated = pd.concat(evaluated_frames)
    report["baselines"] = {
        "no_trade": {
            "opportunity_count": len(evaluated),
            "selected_count": 0,
            "selected_share": 0.0,
            "average_return_per_opportunity_pct": 0.0,
        },
        **{
            f"always_{side}": _evidence(
                evaluated, np.full(len(evaluated), side)
            ) for side in ("long", "short")
        },
    }
    for name, side_predictions in all_predictions.items():
        predictions = {
            side: np.concatenate(chunks)
            for side, chunks in side_predictions.items()
        }
        evidence = _evidence(evaluated, _selected_sides(predictions))
        report["baselines"][name] = {
            **evidence,
            "rank_only_top_20_percent": _rank_only_evidence(evaluated, predictions),
            "prediction_diagnostics": {
                side: _prediction_diagnostics(
                    _net_returns(evaluated, side).to_numpy(), predictions[side]
                ) for side in ("long", "short")
            },
        }
    report["status"] = "complete"
    report["development_sample_count"] = int(len(development))
    report["evaluated_opportunity_count"] = int(len(evaluated))
    report["evaluated_decision_group_count"] = int(
        evaluated["decision_group"].nunique()
    )
    return report


def _selected_sides(predictions: dict[str, np.ndarray]) -> np.ndarray:
    long = predictions["long"]
    short = predictions["short"]
    return np.where(
        np.maximum(long, short) <= 0.0,
        "hold",
        np.where(long >= short, "long", "short"),
    )
