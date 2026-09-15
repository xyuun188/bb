"""Prediction-distribution contracts shared by local ML training and inference."""

from __future__ import annotations

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


def regression_prediction_distribution(
    model: Pipeline,
    x: pd.DataFrame,
) -> dict[str, Any]:
    """Return a point or empirical tree-member prediction distribution."""

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
    ordered_tree_predictions = np.sort(tree_predictions, axis=0)
    lower_tail_count = max(int(np.sqrt(len(ordered_tree_predictions))), 1)
    return {
        "expected": expected,
        "median": np.median(tree_predictions, axis=0),
        "lower_quantile": np.median(ordered_tree_predictions[:lower_tail_count], axis=0),
        "upper_quantile": np.median(ordered_tree_predictions[-lower_tail_count:], axis=0),
        "std": np.std(tree_predictions, axis=0),
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
    cost_expected = np.maximum(
        np.asarray(cost_distribution["expected"], dtype=float),
        0.0,
    )
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
