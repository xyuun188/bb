"""Shared fee-after-return objective and evaluation helpers.

Win rate and classification metrics are intentionally excluded from this
module.  They may still be reported as diagnostics, but production model
quality, promotion, routing, and sizing must be derived from fee-after return
and downside-risk evidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

RETURN_OBJECTIVE_NAME = "maximize_expected_realized_net_return_after_cost"
RETURN_OBJECTIVE_VERSION = "2026-07-12.v1"
RETURN_LABEL_NAME = "net_return_after_cost_pct"
RETURN_LABEL_VERSION = "2026-07-12.v1"
COST_MODEL_VERSION = "okx_fee_slippage_funding_v1"


def safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def profit_factor(returns: Iterable[float]) -> float | None:
    values = np.asarray(list(returns), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    gross_profit = float(values[values > 0].sum())
    gross_loss = abs(float(values[values < 0].sum()))
    if gross_loss <= 1e-12:
        return None if gross_profit <= 1e-12 else 1_000_000.0
    return gross_profit / gross_loss


def mean_confidence_lower_bound(
    returns: Iterable[float],
    *,
    z_score: float = 1.645,
) -> float | None:
    """One-sided lower confidence bound for the mean fee-after return."""

    values = np.asarray(list(returns), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    mean = float(values.mean())
    if values.size == 1:
        return mean
    standard_error = float(values.std(ddof=1)) / math.sqrt(values.size)
    return mean - max(float(z_score), 0.0) * standard_error


def cvar(returns: Iterable[float], *, tail_quantile: float = 0.10) -> float | None:
    values = np.asarray(list(returns), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    quantile = min(max(float(tail_quantile), 0.01), 0.50)
    cutoff = float(np.quantile(values, quantile))
    tail = values[values <= cutoff]
    return float(tail.mean()) if tail.size else cutoff


def return_distribution_summary(
    returns: Iterable[float],
    *,
    tail_loss_threshold_pct: float,
) -> dict[str, Any]:
    values = np.asarray(list(returns), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "avg_return_pct": None,
            "median_return_pct": None,
            "return_lcb_pct": None,
            "profit_factor": None,
            "tail_loss_rate": None,
            "cvar_10_pct": None,
        }
    pf = profit_factor(values)
    return {
        "count": int(values.size),
        "avg_return_pct": float(values.mean()),
        "median_return_pct": float(np.median(values)),
        "return_lcb_pct": mean_confidence_lower_bound(values),
        "profit_factor": pf,
        "tail_loss_rate": float((values < -abs(float(tail_loss_threshold_pct))).mean()),
        "cvar_10_pct": cvar(values),
    }


def risk_adjusted_expected_return(
    *,
    expected_return_pct: float,
    lower_quantile_return_pct: float,
    tail_loss_probability: float | None,
    tail_loss_scale_pct: float,
) -> dict[str, float]:
    """Apply model-disagreement and left-tail penalties to expected return."""

    expected = float(expected_return_pct)
    lower = min(float(lower_quantile_return_pct), expected)
    uncertainty_penalty = max(expected - lower, 0.0)
    tail_probability = min(max(float(tail_loss_probability or 0.0), 0.0), 1.0)
    tail_penalty = tail_probability * abs(float(tail_loss_scale_pct))
    objective = expected - uncertainty_penalty - tail_penalty
    return {
        "expected_net_return_pct": expected,
        "lower_quantile_net_return_pct": lower,
        "uncertainty_penalty_pct": uncertainty_penalty,
        "tail_loss_penalty_pct": tail_penalty,
        "objective_net_return_pct": objective,
    }
