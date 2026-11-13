"""Prediction-distribution contracts shared by local ML training and inference."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from services.profit_supervision import PROFIT_SUPERVISION_VERSION
from services.return_objective import (
    COST_MODEL_VERSION,
    RETURN_LABEL_VERSION,
    RETURN_OBJECTIVE_VERSION,
    risk_adjusted_expected_return,
    standardized_return_distribution,
)

ENTRY_COHORT_VERSION = "2026-11-13.positive-objective-entry-cohort.v1"
PREDICTION_SCORE_POLICY_VERSION = "2026-11-13.leaf-total-risk-objective.v1"


def positive_objective_entry_rows(
    scored_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Select one positive objective per decision; ranking alone cannot enter."""

    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for side in ("long", "short"):
        indexed[side] = {}
        for row in scored_rows.get(side, []):
            group = str(row.get("decision_group") or "")
            if not group or group in indexed[side]:
                raise ValueError("entry_cohort_requires_unique_decision_groups")
            indexed[side][group] = row
    selected: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}
    for group in indexed["long"]:
        if group not in indexed["short"]:
            continue
        long_row, short_row = (indexed[side][group] for side in ("long", "short"))
        long_score, short_score = float(long_row["score"]), float(short_row["score"])
        if not all(math.isfinite(value) for value in (long_score, short_score)):
            continue
        if max(long_score, short_score) <= 0.0:
            continue
        side = "long" if long_score >= short_score else "short"
        selected[side].append(indexed[side][group])
    return selected


def regression_prediction_distribution(
    model: Pipeline,
    x: pd.DataFrame,
) -> dict[str, Any]:
    """Include outcome variance inside leaves as well as tree disagreement."""

    expected = np.asarray(model.predict(x), dtype=float)
    named_steps = getattr(model, "named_steps", {})
    getter = getattr(named_steps, "get", None)
    estimator = getter("model") if callable(getter) else None
    imputer = getter("imputer") if callable(getter) else None
    trees = list(getattr(estimator, "estimators_", []) or [])
    if not trees or imputer is None:
        return {
            "expected": expected,
            "median": expected.copy(),
            "lower_quantile": expected.copy(),
            "upper_quantile": expected.copy(),
            "std": np.zeros(len(expected), dtype=float),
            "member_count": 0,
            "source_authority": "regressor_point_prediction_without_members",
        }
    transformed = imputer.transform(x)
    tree_predictions = np.asarray([tree.predict(transformed) for tree in trees], dtype=float)
    leaf_variances = []
    for tree in trees:
        structure = getattr(tree, "tree_", None)
        if (
            structure is not None
            and getattr(tree, "criterion", None) in {"squared_error", "friedman_mse"}
        ):
            leaf_variances.append(
                np.maximum(np.asarray(structure.impurity[tree.apply(transformed)]), 0.0)
            )
    between_variance = np.var(tree_predictions, axis=0)
    within_variance = (
        np.mean(leaf_variances, axis=0)
        if len(leaf_variances) == len(trees)
        else np.zeros(len(expected), dtype=float)
    )
    ordered_tree_predictions = np.sort(tree_predictions, axis=0)
    lower_tail_count = max(int(np.sqrt(len(ordered_tree_predictions))), 1)
    return {
        "expected": expected,
        "median": np.median(tree_predictions, axis=0),
        "lower_quantile": np.median(ordered_tree_predictions[:lower_tail_count], axis=0),
        "upper_quantile": np.median(ordered_tree_predictions[-lower_tail_count:], axis=0),
        # Total predictive variance = E[Var(Y | leaf)] + Var(E[Y | leaf]).
        "std": np.sqrt(within_variance + between_variance),
        "between_tree_std": np.sqrt(between_variance),
        "within_leaf_std": np.sqrt(within_variance),
        "dispersion_basis": (
            "tree_leaf_total_variance"
            if len(leaf_variances) == len(trees)
            else "tree_mean_disagreement_only"
        ),
        "member_count": len(trees),
        "source_authority": "random_forest_tree_empirical_distribution",
    }


def standardized_model_return_distribution(
    distribution: dict[str, Any],
    index: int,
    *,
    side: str,
    horizon_minutes: int,
    tail_loss_probability: float | None,
    tail_loss_scale_pct: float,
) -> dict[str, Any]:
    """Map model output onto the authoritative return-distribution contract."""

    return standardized_return_distribution(
        side=side,
        horizon_minutes=horizon_minutes,
        raw_expected_return_pct=distribution["expected"][index],
        median_return_pct=distribution["median"][index],
        lower_quantile_return_pct=distribution["lower_quantile"][index],
        upper_quantile_return_pct=distribution["upper_quantile"][index],
        dispersion_pct=distribution["std"][index],
        tail_loss_probability=tail_loss_probability,
        tail_loss_scale_pct=tail_loss_scale_pct,
        distribution_member_count=distribution.get("member_count"),
        return_semantics="gross_market_opportunity_before_execution",
        source_authority=str(distribution.get("source_authority") or ""),
        objective_version=RETURN_OBJECTIVE_VERSION,
        label_version=RETURN_LABEL_VERSION,
        cost_model_version=COST_MODEL_VERSION,
        profit_supervision_version=PROFIT_SUPERVISION_VERSION,
    )


def risk_adjusted_expected_scores(
    market_distribution: dict[str, np.ndarray],
    cost_distribution: dict[str, np.ndarray],
    tail_loss_scores: np.ndarray,
    *,
    tail_loss_scale_pct: float,
) -> np.ndarray:
    """Score every candidate using fee-after return and lower-tail risk."""

    gross_expected = np.asarray(market_distribution["expected"], dtype=float)
    gross_lower = np.asarray(market_distribution["lower_quantile"], dtype=float)
    gross_std = np.asarray(market_distribution["std"], dtype=float)
    # These labels include signed funding, so income can exceed fees.
    cost_expected = np.asarray(cost_distribution["expected"], dtype=float)
    cost_upper = np.maximum(
        np.asarray(cost_distribution["upper_quantile"], dtype=float),
        cost_expected,
    )
    expected_net = gross_expected - cost_expected
    lower_net = np.minimum(gross_lower - cost_upper, expected_net)
    return np.asarray(
        [
            risk_adjusted_expected_return(
                expected_return_pct=float(expected_net[index]),
                lower_quantile_return_pct=float(lower_net[index]),
                dispersion_pct=float(
                    gross_std[index]
                ),
                tail_loss_probability=float(tail_loss_scores[index]),
                tail_loss_scale_pct=tail_loss_scale_pct,
            )["objective_net_return_pct"]
            for index in range(len(expected_net))
        ],
        dtype=float,
    )


def profit_quality_score(
    objective_return_pct: float,
    lower_quantile_return_pct: float,
    edge_pct: float,
    tail_loss_probability: float,
    tail_loss_scale_pct: float,
) -> float:
    """Score fee-after return quality without win-rate input."""

    expected_component = max(objective_return_pct, 0.0)
    lower_bound_component = max(lower_quantile_return_pct, 0.0)
    edge_component = max(edge_pct, 0.0)
    tail_penalty = min(max(float(tail_loss_probability), 0.0), 1.0) * max(
        tail_loss_scale_pct,
        0.0,
    )
    return expected_component + lower_bound_component + edge_component - tail_penalty
