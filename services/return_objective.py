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
RETURN_OBJECTIVE_VERSION = "2026-07-14.separated-supervision.v2"
RETURN_LABEL_NAME = "separated_market_cost_and_realized_return_tasks"
RETURN_LABEL_VERSION = "2026-07-14.separated-supervision.v2"
COST_MODEL_VERSION = "okx_live_cost_and_authoritative_slippage_distribution_v2"
RETURN_DISTRIBUTION_CONTRACT_VERSION = (
    "2026-07-15.standardized-return-distribution.v1"
)
RETURN_DISTRIBUTION_INPUT_VERSION = "2026-07-15.model-return-distribution-input.v1"


def safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def standardized_return_distribution(
    *,
    side: str,
    horizon_minutes: int | float | None,
    raw_expected_return_pct: Any,
    median_return_pct: Any,
    lower_quantile_return_pct: Any,
    upper_quantile_return_pct: Any,
    dispersion_pct: Any,
    tail_loss_probability: Any,
    tail_loss_scale_pct: Any,
    distribution_member_count: Any,
    return_semantics: str,
    source_authority: str,
    objective_version: str = RETURN_OBJECTIVE_VERSION,
    label_version: str = RETURN_LABEL_VERSION,
    cost_model_version: str = COST_MODEL_VERSION,
    profit_supervision_version: str = "",
) -> dict[str, Any]:
    """Build the only production-facing return-distribution contract."""

    expected = safe_float(raw_expected_return_pct, None)
    median = safe_float(median_return_pct, None)
    lower = safe_float(lower_quantile_return_pct, None)
    upper = safe_float(upper_quantile_return_pct, None)
    dispersion = safe_float(dispersion_pct, None)
    tail_probability = safe_float(tail_loss_probability, None)
    tail_scale = safe_float(tail_loss_scale_pct, None)
    try:
        member_count = int(distribution_member_count or 0)
    except (TypeError, ValueError):
        member_count = 0
    try:
        horizon = int(horizon_minutes or 0)
    except (TypeError, ValueError):
        horizon = 0

    blockers: list[str] = []
    for value, code in (
        (expected, "raw_expected_return_missing"),
        (median, "median_return_missing"),
        (lower, "lower_quantile_return_missing"),
        (upper, "upper_quantile_return_missing"),
        (dispersion, "return_dispersion_missing"),
        (tail_probability, "tail_loss_probability_missing"),
        (tail_scale, "tail_loss_scale_missing"),
    ):
        if value is None:
            blockers.append(code)
    if side not in {"long", "short"}:
        blockers.append("distribution_side_invalid")
    if horizon <= 0:
        blockers.append("distribution_horizon_missing")
    if member_count <= 0:
        blockers.append("distribution_members_missing")
    if expected is not None and lower is not None and lower > expected:
        blockers.append("lower_quantile_above_raw_expected")
    if lower is not None and median is not None and lower > median:
        blockers.append("lower_quantile_above_median")
    if median is not None and upper is not None and median > upper:
        blockers.append("median_above_upper_quantile")
    if dispersion is not None and dispersion < 0:
        blockers.append("return_dispersion_negative")
    if tail_probability is not None and not 0.0 <= tail_probability <= 1.0:
        blockers.append("tail_loss_probability_out_of_bounds")
    if tail_scale is not None and tail_scale < 0:
        blockers.append("tail_loss_scale_negative")
    if not str(return_semantics or "").strip():
        blockers.append("return_semantics_missing")
    if not str(source_authority or "").strip():
        blockers.append("source_authority_missing")

    uncertainty_penalty = None
    tail_penalty = None
    objective_expected = None
    if not blockers:
        assert expected is not None
        assert lower is not None
        assert dispersion is not None
        assert tail_probability is not None
        assert tail_scale is not None
        uncertainty_penalty = max(expected - lower, dispersion, 0.0)
        tail_penalty = tail_probability * tail_scale
        objective_expected = expected - uncertainty_penalty - tail_penalty

    return {
        "version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
        "side": side,
        "horizon_minutes": horizon or None,
        "unit": "pct",
        "return_semantics": return_semantics,
        "source_authority": source_authority,
        "objective_version": objective_version,
        "label_version": label_version,
        "cost_model_version": cost_model_version,
        "profit_supervision_version": profit_supervision_version,
        "raw_expected_return_pct": expected,
        "median_return_pct": median,
        "lower_quantile_return_pct": lower,
        "upper_quantile_return_pct": upper,
        "dispersion_pct": dispersion,
        "tail_loss_probability": tail_probability,
        "tail_loss_scale_pct": tail_scale,
        "uncertainty_penalty_pct": uncertainty_penalty,
        "tail_loss_penalty_pct": tail_penalty,
        "objective_expected_return_pct": objective_expected,
        "distribution_member_count": member_count,
        "production_eligible": not blockers,
        "blockers": blockers,
    }


def validate_return_distribution_contract(
    contract: dict[str, Any] | None,
    *,
    side: str,
    return_semantics: str,
    profit_supervision_version: str,
) -> dict[str, Any]:
    """Validate every field that may grant a return distribution production use."""

    payload = contract if isinstance(contract, dict) else {}
    blockers = list(payload.get("blockers") or [])
    expected_versions = {
        "version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "label_version": RETURN_LABEL_VERSION,
        "cost_model_version": COST_MODEL_VERSION,
        "profit_supervision_version": profit_supervision_version,
    }
    for field, expected in expected_versions.items():
        if str(payload.get(field) or "") != expected:
            blockers.append(f"return_distribution_{field}_mismatch")

    if payload.get("side") != side:
        blockers.append("return_distribution_side_mismatch")
    if payload.get("unit") != "pct":
        blockers.append("return_distribution_unit_mismatch")
    if payload.get("return_semantics") != return_semantics:
        blockers.append("return_distribution_semantics_mismatch")
    if not str(payload.get("source_authority") or "").strip():
        blockers.append("return_distribution_source_authority_missing")

    horizon = safe_float(payload.get("horizon_minutes"), None)
    member_count = safe_float(payload.get("distribution_member_count"), None)
    if horizon is None or horizon <= 0:
        blockers.append("return_distribution_horizon_invalid")
    if member_count is None or member_count <= 0:
        blockers.append("return_distribution_members_invalid")

    numeric_fields = (
        "raw_expected_return_pct",
        "median_return_pct",
        "lower_quantile_return_pct",
        "upper_quantile_return_pct",
        "dispersion_pct",
        "tail_loss_probability",
        "tail_loss_scale_pct",
        "objective_expected_return_pct",
    )
    numeric: dict[str, float] = {}
    for field in numeric_fields:
        value = safe_float(payload.get(field), None)
        if value is None:
            blockers.append(f"return_distribution_{field}_invalid")
        else:
            numeric[field] = value

    expected = numeric.get("raw_expected_return_pct")
    median = numeric.get("median_return_pct")
    lower = numeric.get("lower_quantile_return_pct")
    upper = numeric.get("upper_quantile_return_pct")
    if expected is not None and lower is not None and lower > expected:
        blockers.append("lower_quantile_above_raw_expected")
    if lower is not None and median is not None and lower > median:
        blockers.append("lower_quantile_above_median")
    if median is not None and upper is not None and median > upper:
        blockers.append("median_above_upper_quantile")
    if numeric.get("dispersion_pct", 0.0) < 0:
        blockers.append("return_dispersion_negative")
    tail_probability = numeric.get("tail_loss_probability")
    if tail_probability is not None and not 0.0 <= tail_probability <= 1.0:
        blockers.append("tail_loss_probability_out_of_bounds")
    if numeric.get("tail_loss_scale_pct", 0.0) < 0:
        blockers.append("tail_loss_scale_negative")
    if payload.get("production_eligible") is not True:
        blockers.append("return_distribution_not_production_eligible")

    unique_blockers = list(dict.fromkeys(str(item) for item in blockers if item))
    return {
        "eligible": not unique_blockers,
        "reason": (
            "standardized_return_distribution_verified"
            if not unique_blockers
            else unique_blockers[0]
        ),
        "side": side,
        "horizon_minutes": int(horizon) if horizon is not None and horizon > 0 else None,
        "blockers": unique_blockers,
        "contract": payload,
    }


def combine_production_return_distribution(
    *,
    side: str,
    model_contracts: Iterable[dict[str, Any]],
    live_execution_cost_pct: Any,
    live_slippage_pct: Any,
    counterfactual_cost_distributions: Iterable[dict[str, Any]],
    actual_trade_calibrations: Iterable[dict[str, Any]],
    profit_supervision_version: str,
    source_authority: str,
    input_blockers: Iterable[str] = (),
) -> dict[str, Any]:
    """Build the sole fee-after production distribution and transformation audit."""

    contracts = [item for item in model_contracts if isinstance(item, dict)]
    cost_rows = [item for item in counterfactual_cost_distributions if isinstance(item, dict)]
    calibration_rows = [item for item in actual_trade_calibrations if isinstance(item, dict)]
    blockers = [str(item) for item in input_blockers if item]
    if not contracts:
        blockers.append("governed_market_opportunity_distribution_missing")

    validations = [
        validate_return_distribution_contract(
            contract,
            side=side,
            return_semantics="gross_market_opportunity_before_execution",
            profit_supervision_version=profit_supervision_version,
        )
        for contract in contracts
    ]
    for validation in validations:
        blockers.extend(validation.get("blockers") or [])

    signatures = {
        (
            contract.get("objective_version"),
            contract.get("label_version"),
            contract.get("cost_model_version"),
            contract.get("profit_supervision_version"),
            contract.get("side"),
            contract.get("horizon_minutes"),
        )
        for contract in contracts
    }
    if len(signatures) > 1:
        fields = (
            "objective_version",
            "label_version",
            "cost_model_version",
            "profit_supervision_version",
            "side",
            "horizon_minutes",
        )
        for index, field in enumerate(fields):
            if len({signature[index] for signature in signatures}) > 1:
                blockers.append(f"model_distribution_{field}_mismatch")

    gross_expected_values = [
        float(contract["raw_expected_return_pct"])
        for contract, validation in zip(contracts, validations, strict=False)
        if validation.get("eligible") is True
    ]
    gross_median_values = [
        float(contract["median_return_pct"])
        for contract, validation in zip(contracts, validations, strict=False)
        if validation.get("eligible") is True
    ]
    gross_lower_values = [
        float(contract["lower_quantile_return_pct"])
        for contract, validation in zip(contracts, validations, strict=False)
        if validation.get("eligible") is True
    ]
    gross_upper_values = [
        float(contract["upper_quantile_return_pct"])
        for contract, validation in zip(contracts, validations, strict=False)
        if validation.get("eligible") is True
    ]
    model_dispersion_values = [
        max(
            float(contract["dispersion_pct"]),
            float(contract["raw_expected_return_pct"])
            - float(contract["lower_quantile_return_pct"]),
        )
        for contract, validation in zip(contracts, validations, strict=False)
        if validation.get("eligible") is True
    ]
    tail_probabilities = [
        float(contract["tail_loss_probability"])
        for contract, validation in zip(contracts, validations, strict=False)
        if validation.get("eligible") is True
    ]
    tail_scales = [
        float(contract["tail_loss_scale_pct"])
        for contract, validation in zip(contracts, validations, strict=False)
        if validation.get("eligible") is True
    ]

    gross_expected = (
        sum(gross_expected_values) / len(gross_expected_values)
        if gross_expected_values
        else None
    )
    gross_median = (
        sum(gross_median_values) / len(gross_median_values)
        if gross_median_values
        else None
    )
    between_model_dispersion = None
    if gross_expected_values and gross_expected is not None:
        between_model_dispersion = math.sqrt(
            sum((value - gross_expected) ** 2 for value in gross_expected_values)
            / len(gross_expected_values)
        )
    market_dispersion = (
        max([between_model_dispersion or 0.0, *model_dispersion_values])
        if model_dispersion_values
        else None
    )

    historical_cost_uncertainties: list[float] = []
    for row in cost_rows:
        expected_cost = safe_float(row.get("expected_pct"), None)
        upper_cost = safe_float(row.get("upper_tail_pct"), None)
        uncertainty = safe_float(row.get("uncertainty_pct"), None)
        if (
            expected_cost is None
            or upper_cost is None
            or uncertainty is None
            or expected_cost < 0
            or upper_cost < expected_cost
            or uncertainty < 0
            or row.get("distribution_ready") is not True
            or row.get("source_authority")
            != "shadow_counterfactual_live_microstructure"
        ):
            blockers.append("counterfactual_execution_cost_distribution_invalid")
            continue
        historical_cost_uncertainties.append(
            max(upper_cost - expected_cost, uncertainty)
        )
    if not historical_cost_uncertainties:
        blockers.append("counterfactual_execution_cost_distribution_missing")

    actual_uncertainties: list[float] = []
    slippage_expected_values: list[float] = []
    slippage_upper_values: list[float] = []
    for row in calibration_rows:
        realized = row.get("net_return_after_cost_pct")
        slippage = row.get("slippage_pct")
        realized = realized if isinstance(realized, dict) else {}
        slippage = slippage if isinstance(slippage, dict) else {}
        realized_count = safe_float(realized.get("count"), None)
        slippage_count = safe_float(slippage.get("count"), None)
        actual_expected = safe_float(realized.get("expected"), None)
        actual_lower = safe_float(realized.get("lower_hinge"), None)
        slippage_expected = safe_float(slippage.get("expected"), None)
        slippage_upper = safe_float(slippage.get("upper_hinge"), None)
        if (
            row.get("source_authority") != "okx_position_history"
            or row.get("side") != side
            or realized_count is None
            or realized_count <= 0
            or slippage_count is None
            or slippage_count <= 0
            or actual_expected is None
            or actual_lower is None
            or slippage_expected is None
            or slippage_upper is None
            or slippage_expected < 0
            or slippage_upper < slippage_expected
        ):
            blockers.append("authoritative_trade_calibration_invalid")
            continue
        if actual_lower > actual_expected:
            blockers.append("authoritative_realized_lower_above_expected")
            continue
        actual_uncertainties.append(actual_expected - actual_lower)
        slippage_expected_values.append(slippage_expected)
        slippage_upper_values.append(slippage_upper)
    if not actual_uncertainties:
        blockers.append("authoritative_realized_return_or_slippage_distribution_missing")

    live_cost = safe_float(live_execution_cost_pct, None)
    live_slippage = safe_float(live_slippage_pct, None)
    if live_cost is None or live_cost < 0 or live_slippage is None or live_slippage < 0:
        blockers.append("live_execution_cost_distribution_missing")
    authoritative_slippage_upper = max(slippage_upper_values) if slippage_upper_values else None
    slippage_tail_excess = (
        authoritative_slippage_upper - live_slippage
        if authoritative_slippage_upper is not None
        and live_slippage is not None
        and authoritative_slippage_upper > live_slippage
        else 0.0
        if authoritative_slippage_upper is not None and live_slippage is not None
        else None
    )

    net_expected = (
        gross_expected - live_cost - slippage_tail_excess
        if gross_expected is not None
        and live_cost is not None
        and slippage_tail_excess is not None
        else None
    )
    net_median = (
        gross_median - live_cost - slippage_tail_excess
        if gross_median is not None
        and live_cost is not None
        and slippage_tail_excess is not None
        else None
    )
    total_dispersion = (
        market_dispersion
        + max(historical_cost_uncertainties)
        + max(actual_uncertainties)
        if market_dispersion is not None
        and historical_cost_uncertainties
        and actual_uncertainties
        else None
    )
    net_lower = (
        net_expected - total_dispersion
        if net_expected is not None and total_dispersion is not None
        else None
    )
    net_upper = (
        max(
            sum(gross_upper_values) / len(gross_upper_values)
            - live_cost
            - slippage_tail_excess,
            net_median if net_median is not None else float("-inf"),
            net_expected if net_expected is not None else float("-inf"),
        )
        if gross_upper_values
        and live_cost is not None
        and slippage_tail_excess is not None
        else None
    )
    horizon = (
        contracts[0].get("horizon_minutes")
        if contracts and len(signatures) == 1
        else None
    )
    member_count = sum(
        int(safe_float(contract.get("distribution_member_count"), 0.0) or 0)
        for contract in contracts
    )
    contract = standardized_return_distribution(
        side=side,
        horizon_minutes=horizon,
        raw_expected_return_pct=net_expected,
        median_return_pct=net_median,
        lower_quantile_return_pct=net_lower,
        upper_quantile_return_pct=net_upper,
        dispersion_pct=total_dispersion,
        tail_loss_probability=max(tail_probabilities) if tail_probabilities else None,
        tail_loss_scale_pct=max(tail_scales) if tail_scales else None,
        distribution_member_count=member_count,
        return_semantics="realized_net_return_after_live_cost_and_authoritative_slippage",
        source_authority=source_authority,
        profit_supervision_version=profit_supervision_version,
    )
    merged_blockers = list(
        dict.fromkeys(
            [
                *(str(item) for item in blockers if item),
                *(str(item) for item in contract.get("blockers") or [] if item),
            ]
        )
    )
    contract["blockers"] = merged_blockers
    contract["production_eligible"] = not merged_blockers
    contract["gross_market_distribution"] = {
        "raw_expected_return_pct": gross_expected,
        "median_return_pct": gross_median,
        "lower_quantile_return_pct": min(gross_lower_values) if gross_lower_values else None,
        "upper_quantile_return_pct": max(gross_upper_values) if gross_upper_values else None,
        "dispersion_pct": market_dispersion,
        "model_count": len(gross_expected_values),
    }
    contract["transformations"] = {
        "formula": (
            "gross_market_expected-live_execution_cost-"
            "authoritative_slippage_tail_excess"
        ),
        "live_execution_cost_pct": live_cost,
        "live_slippage_pct": live_slippage,
        "authoritative_slippage_expected_pct": (
            sum(slippage_expected_values) / len(slippage_expected_values)
            if slippage_expected_values
            else None
        ),
        "authoritative_slippage_upper_hinge_pct": authoritative_slippage_upper,
        "authoritative_slippage_tail_excess_pct": slippage_tail_excess,
        "market_dispersion_pct": market_dispersion,
        "counterfactual_cost_uncertainty_pct": (
            max(historical_cost_uncertainties)
            if historical_cost_uncertainties
            else None
        ),
        "actual_trade_calibration_uncertainty_pct": (
            max(actual_uncertainties) if actual_uncertainties else None
        ),
        "cost_deduction_count": (
            1
            if live_cost is not None and slippage_tail_excess is not None
            else 0
        ),
    }
    return contract


def profit_factor(returns: Iterable[float]) -> float | None:
    values = np.asarray(list(returns), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    gross_profit = float(values[values > 0].sum())
    gross_loss = abs(float(values[values < 0].sum()))
    if gross_loss <= 1e-12:
        return None
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
    lower = float(lower_quantile_return_pct)
    if lower > expected:
        raise ValueError("lower_quantile_above_raw_expected")
    uncertainty_penalty = expected - lower
    tail_probability = float(tail_loss_probability or 0.0)
    if not 0.0 <= tail_probability <= 1.0:
        raise ValueError("tail_loss_probability_out_of_bounds")
    tail_scale = float(tail_loss_scale_pct)
    if tail_scale < 0:
        raise ValueError("tail_loss_scale_negative")
    tail_penalty = tail_probability * tail_scale
    objective = expected - uncertainty_penalty - tail_penalty
    return {
        "expected_net_return_pct": expected,
        "lower_quantile_net_return_pct": lower,
        "uncertainty_penalty_pct": uncertainty_penalty,
        "tail_loss_penalty_pct": tail_penalty,
        "objective_net_return_pct": objective,
    }
