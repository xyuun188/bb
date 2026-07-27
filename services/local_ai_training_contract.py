"""Shared contracts for Local AI Tools training cadence and data identity."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from statistics import fmean, pstdev
from typing import Any

from services.profit_supervision import (
    COUNTERFACTUAL_EXECUTION_COST_TASK as EXECUTION_COST_TASK,
)
from services.profit_supervision import (
    MARKET_OPPORTUNITY_TASK,
    PROFIT_SUPERVISION_VERSION,
)

RETURN_OBJECTIVE_NAME = "maximize_expected_realized_net_return_after_cost"
RETURN_OBJECTIVE_VERSION = "2026-07-27.separated-source-supervision.v3"
RETURN_LABEL_NAME = "separated_market_cost_and_realized_return_tasks"
RETURN_LABEL_VERSION = "2026-07-27.separated-source-supervision.v3"
COST_MODEL_VERSION = "okx_authoritative_execution_cost_distribution_v3"
TRAINING_COST_POLICY = "shadow_market_opportunity_plus_authoritative_okx_execution_cost"
TRAINING_CURSOR_VERSION = "2026-07-27.independent-decision-groups.v1"
TRAINING_DISTRIBUTION_PROFILE_VERSION = "2026-07-27.training-distribution-profile.v1"
TRAINING_DISTRIBUTION_DRIFT_VERSION = "2026-07-27.standardized-mean-shift.v1"
TRAINING_TRIGGER_POLICY_VERSION = "2026-07-27.decision-group-batch.v1"

_PROFILE_FEATURES = (
    "returns_5",
    "returns_20",
    "volatility_20",
    "spread_pct",
    "orderbook_imbalance",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive_weight(sample: Mapping[str, Any]) -> float | None:
    weight = _finite_float(sample.get("sample_weight"))
    return weight if weight is not None and weight > 0.0 else None


def market_training_identity(sample: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the canonical market-label identity used by training and cursors."""

    if bool(sample.get("exclude_from_training")):
        return None
    features = _mapping(sample.get("features"))
    supervision = _mapping(sample.get("profit_supervision"))
    market_task = _mapping(_mapping(supervision.get("tasks")).get(MARKET_OPPORTUNITY_TASK))
    long_return = _finite_float(market_task.get("long_gross_market_return_pct"))
    short_return = _finite_float(market_task.get("short_gross_market_return_pct"))
    horizon = int(sample.get("horizon_minutes") or features.get("horizon_minutes") or 0)
    weight = _positive_weight(sample)
    correlation_group = str(
        _mapping(sample.get("correlation_weight")).get("correlation_group") or ""
    ).strip()
    decision_identity = sample.get("decision_id") or sample.get("id")
    decision_group = correlation_group or (
        f"shadow_decision:{decision_identity}" if decision_identity else ""
    )
    if (
        not features
        or horizon <= 0
        or supervision.get("version") != PROFIT_SUPERVISION_VERSION
        or market_task.get("eligible") is not True
        or long_return is None
        or short_return is None
        or weight is None
        or not decision_group
    ):
        return None
    return {
        "decision_group": decision_group,
        "features": dict(features),
        "horizon_minutes": horizon,
        "long_return_pct": long_return,
        "short_return_pct": short_return,
        "sample_weight": weight,
    }


def authoritative_cost_training_identity(
    sample: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the canonical OKX lifecycle identity used by cost training."""

    if bool(sample.get("exclude_from_training")):
        return None
    features = _mapping(sample.get("features"))
    supervision = _mapping(sample.get("profit_supervision"))
    cost_task = _mapping(_mapping(supervision.get("tasks")).get(EXECUTION_COST_TASK))
    side = str(sample.get("side") or "").strip().lower()
    total_cost = _finite_float(cost_task.get("total_cost_pct"))
    weight = _positive_weight(sample)
    lifecycle = str(
        sample.get("lifecycle_key") or sample.get("position_id") or sample.get("id") or ""
    ).strip()
    if (
        not features
        or supervision.get("version") != PROFIT_SUPERVISION_VERSION
        or cost_task.get("eligible") is not True
        or cost_task.get("source_authority") != "okx_fills_fees_funding"
        or side not in {"long", "short"}
        or total_cost is None
        or weight is None
        or not lifecycle
    ):
        return None
    return {
        "decision_group": f"okx_lifecycle:{lifecycle}",
        "features": dict(features),
        "side": side,
        "execution_cost_pct": total_cost,
        "sample_weight": weight,
    }


def local_ai_training_cursor(
    *,
    shadow_samples: Sequence[Mapping[str, Any]],
    trade_samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    market_rows = [
        identity
        for sample in shadow_samples
        if (identity := market_training_identity(sample)) is not None
    ]
    cost_rows = [
        identity
        for sample in trade_samples
        if (identity := authoritative_cost_training_identity(sample)) is not None
    ]
    market_groups = {str(row["decision_group"]) for row in market_rows}
    cost_groups = {str(row["decision_group"]) for row in cost_rows}
    values: dict[str, list[float]] = defaultdict(list)
    for row in market_rows:
        features = _mapping(row.get("features"))
        for feature in _PROFILE_FEATURES:
            if (value := _finite_float(features.get(feature))) is not None:
                values[feature].append(value)
        values["long_return_pct"].append(float(row["long_return_pct"]))
        values["short_return_pct"].append(float(row["short_return_pct"]))
    for row in cost_rows:
        values["authoritative_execution_cost_pct"].append(float(row["execution_cost_pct"]))
    profile = {
        key: {"count": len(rows), "mean": fmean(rows), "std": pstdev(rows)}
        for key, rows in values.items()
        if rows
    }
    return {
        "version": TRAINING_CURSOR_VERSION,
        "completed_market_sample_count": len(market_rows),
        "completed_authoritative_cost_sample_count": len(cost_rows),
        "completed_market_decision_group_count": len(market_groups),
        "completed_authoritative_cost_decision_group_count": len(cost_groups),
        "completed_training_decision_group_count": len(market_groups | cost_groups),
        "training_distribution_profile": {
            "version": TRAINING_DISTRIBUTION_PROFILE_VERSION,
            "features": profile,
        },
    }


def training_distribution_drift(
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    *,
    threshold: float,
) -> dict[str, Any]:
    current_features = _mapping(current.get("features"))
    previous_features = _mapping(_mapping(previous).get("features"))
    shifts: dict[str, float] = {}
    for key, current_value in current_features.items():
        current_row = _mapping(current_value)
        previous_row = _mapping(previous_features.get(key))
        if not previous_row:
            continue
        current_mean = _finite_float(current_row.get("mean"))
        previous_mean = _finite_float(previous_row.get("mean"))
        current_std = _finite_float(current_row.get("std")) or 0.0
        previous_std = _finite_float(previous_row.get("std")) or 0.0
        if current_mean is not None and previous_mean is not None:
            shifts[str(key)] = abs(current_mean - previous_mean) / max(
                current_std, previous_std, 1e-9
            )
    maximum_shift = max(shifts.values(), default=0.0)
    return {
        "version": TRAINING_DISTRIBUTION_DRIFT_VERSION,
        "detected": bool(shifts and maximum_shift >= float(threshold)),
        "threshold": float(threshold),
        "maximum_shift": round(maximum_shift, 6),
        "feature_shifts": {key: round(value, 6) for key, value in shifts.items()},
    }


def decision_group_training_trigger(
    *,
    force: bool,
    has_artifact: bool,
    completed_group_count: int,
    previous_group_count: int,
    trained_at: Any,
    now: datetime,
    distribution_drift: Mapping[str, Any],
    batch_threshold: int,
    minimum_increment: int,
    drift_minimum_increment: int,
    maximum_interval_seconds: int,
) -> dict[str, Any]:
    """Evaluate cadence without treating elapsed time as new evidence."""

    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    previous = max(int(previous_group_count), 0)
    completed = max(int(completed_group_count), 0)
    rebased = completed < previous
    new_groups = max(completed - previous, 0)
    parsed_trained_at: datetime | None = None
    if trained_at:
        try:
            parsed_trained_at = datetime.fromisoformat(str(trained_at).replace("Z", "+00:00"))
            if parsed_trained_at.tzinfo is None:
                parsed_trained_at = parsed_trained_at.replace(tzinfo=UTC)
            parsed_trained_at = parsed_trained_at.astimezone(UTC)
        except ValueError:
            parsed_trained_at = None
    seconds_since_training = (
        max((now.astimezone(UTC) - parsed_trained_at).total_seconds(), 0.0)
        if parsed_trained_at is not None
        else None
    )
    batch_due = new_groups >= int(batch_threshold)
    interval_due = bool(
        new_groups >= int(minimum_increment)
        and seconds_since_training is not None
        and seconds_since_training >= int(maximum_interval_seconds)
    )
    drift_due = bool(
        distribution_drift.get("detected") is True and new_groups >= int(drift_minimum_increment)
    )
    reason = (
        "forced"
        if force
        else "initial_artifact"
        if not has_artifact
        else "training_view_rebased"
        if rebased
        else "mature_decision_group_batch"
        if batch_due
        else "daily_minimum_increment"
        if interval_due
        else "distribution_drift_with_new_labels"
        if drift_due
        else "not_due"
    )
    return {
        "version": TRAINING_TRIGGER_POLICY_VERSION,
        "due": reason != "not_due",
        "reason": reason,
        "completed_mature_decision_group_count": completed,
        "last_trained_mature_decision_group_count": previous,
        "new_mature_decision_group_count": new_groups,
        "training_view_rebased": rebased,
        "batch_decision_group_threshold": int(batch_threshold),
        "minimum_decision_group_increment": int(minimum_increment),
        "drift_minimum_decision_group_increment": int(drift_minimum_increment),
        "maximum_training_interval_seconds": int(maximum_interval_seconds),
        "seconds_since_last_successful_training": seconds_since_training,
        "distribution_drift": dict(distribution_drift),
    }
