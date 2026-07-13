"""Unified production entry adjudication for fee-after return quality."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import isfinite, sqrt
from typing import Any

from ai_brain.base_model import DecisionOutput

RETURN_EXECUTION_POLICY_VERSION = "2026-07-12.return-execution.v1"


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def _complete_provenance(value: Any) -> bool:
    provenance = _safe_dict(value)
    required = (
        "source",
        "observation_window",
        "sample_count",
        "generated_at",
        "strategy_version",
        "fallback_reason",
    )
    if any(key not in provenance for key in required):
        return False
    return bool(
        str(provenance.get("source") or "").strip()
        and str(provenance.get("observation_window") or "").strip()
        and str(provenance.get("generated_at") or "").strip()
        and str(provenance.get("strategy_version") or "").strip()
        and _safe_float(provenance.get("sample_count"), 0.0) > 0
        and not str(provenance.get("fallback_reason") or "").strip()
    )


@dataclass(frozen=True, slots=True)
class ReturnExecutionAssessment:
    eligible: bool
    reason: str
    expected_net_return_pct: float
    return_lcb_pct: float
    uncertainty_pct: float
    expected_loss_pct: float
    execution_cost_pct: float
    production_source_count: int
    position_size_pct: float
    policy_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _production_return_observations(opportunity: dict[str, Any]) -> list[float]:
    breakdown = _safe_dict(opportunity.get("expected_net_breakdown"))
    distribution_mode = str(opportunity.get("return_distribution_mode") or "").strip()
    observations: list[float] = []
    for component in _safe_list(breakdown.get("components")):
        item = _safe_dict(component)
        included = item.get("included_in_return_distribution")
        if included is not True and not (
            included is None and item.get("production_eligible") is True
        ):
            continue
        if distribution_mode == "governed_models" and item.get("production_eligible") is not True:
            continue
        if (
            distribution_mode == "runtime_recovery"
            and item.get("recovery_observation_eligible") is not True
        ):
            continue
        if distribution_mode not in {"", "governed_models", "runtime_recovery"}:
            continue
        value = _safe_float(item.get("raw_return_pct"), float("nan"))
        if isfinite(value):
            observations.append(value)
    return observations


def _return_uncertainty(
    observations: list[float],
    *,
    expected_net: float,
    expected_loss: float,
    execution_cost: float,
) -> float:
    if len(observations) > 1:
        center = sum(observations) / len(observations)
        variance = sum((value - center) ** 2 for value in observations) / (len(observations) - 1)
        sampling_uncertainty = sqrt(max(variance, 0.0) / len(observations))
    elif observations:
        sampling_uncertainty = abs(observations[0] - expected_net)
    else:
        sampling_uncertainty = abs(expected_net)
    return max(sampling_uncertainty, expected_loss, execution_cost)


def assess_production_entry(decision: DecisionOutput) -> ReturnExecutionAssessment:
    raw = _safe_dict(decision.raw_response)
    opportunity = _safe_dict(raw.get("opportunity_score"))
    sizing = _safe_dict(raw.get("profit_risk_sizing"))
    execution_cost = _safe_dict(opportunity.get("execution_cost"))
    expected_net = _safe_float(opportunity.get("expected_net_return_pct"), float("nan"))
    expected_loss = max(_safe_float(opportunity.get("expected_loss_pct"), 0.0), 0.0)
    cost_pct = max(_safe_float(execution_cost.get("total_pct"), 0.0), 0.0)
    observations = _production_return_observations(opportunity)
    uncertainty = _return_uncertainty(
        observations,
        expected_net=expected_net if isfinite(expected_net) else 0.0,
        expected_loss=expected_loss,
        execution_cost=cost_pct,
    )
    return_lcb = expected_net - uncertainty if isfinite(expected_net) else float("-inf")

    leverage = max(_safe_float(decision.suggested_leverage, 1.0), 1.0)
    balance = max(_safe_float(sizing.get("account_balance_usdt"), 0.0), 0.0)
    max_loss = max(_safe_float(sizing.get("max_stop_loss_usdt"), 0.0), 0.0)
    stop_distance = max(_safe_float(sizing.get("stress_stop_loss_pct"), 0.0), 0.0)
    risk_budget_size = (
        max_loss / (balance * leverage * stop_distance)
        if balance > 0 and max_loss > 0 and stop_distance > 0
        else 0.0
    )
    denominator = max(abs(expected_net), uncertainty, 1e-12)
    return_quality = min(max(return_lcb / denominator, 0.0), 1.0)
    position_size = min(max(risk_budget_size * return_quality, 0.0), 1.0)

    opportunity_provenance = _safe_dict(opportunity.get("policy_provenance"))
    generated_at = str(opportunity_provenance.get("generated_at") or "").strip()
    if not generated_at:
        generated_at = datetime.now(UTC).isoformat()
    opportunity_provenance_complete = _complete_provenance(
        opportunity_provenance
    )
    cost_provenance_complete = _complete_provenance(
        execution_cost.get("policy_provenance")
    )
    sizing_provenance_complete = _complete_provenance(sizing.get("policy_provenance"))
    provenance = {
        "source": "selected_runtime_return_distribution_and_account_stop_budget",
        "observation_window": "current_decision_plus_active_model_return_observations",
        "sample_count": len(observations),
        "generated_at": generated_at,
        "strategy_version": RETURN_EXECUTION_POLICY_VERSION,
        "fallback_reason": "",
        "upstream_provenance": {
            "return_distribution_mode": opportunity.get("return_distribution_mode"),
            "opportunity": opportunity_provenance,
            "execution_cost": execution_cost.get("policy_provenance"),
            "sizing": sizing.get("policy_provenance"),
        },
    }

    reasons: list[str] = []
    if not opportunity:
        reasons.append("opportunity_return_distribution_missing")
    if opportunity.get("production_eligible") is not True:
        reasons.append("opportunity_not_production_eligible")
    if not opportunity_provenance_complete:
        reasons.append("opportunity_policy_provenance_incomplete")
    if not observations:
        reasons.append("production_return_observations_missing")
    if (
        not execution_cost
        or execution_cost.get("production_eligible") is not True
        or cost_pct <= 0
    ):
        reasons.append("execution_cost_distribution_missing")
    if str(execution_cost.get("spread_source") or "") == "missing":
        reasons.append("live_spread_observation_missing")
    if not cost_provenance_complete:
        reasons.append("execution_cost_policy_provenance_incomplete")
    if not isfinite(expected_net) or expected_net <= 0:
        reasons.append("fee_after_expected_return_not_positive")
    if not isfinite(return_lcb) or return_lcb <= 0:
        reasons.append("fee_after_return_lcb_not_positive")
    if sizing.get("production_eligible") is not True:
        reasons.append("dynamic_entry_risk_budget_ineligible")
    if not sizing_provenance_complete:
        reasons.append("dynamic_entry_risk_budget_provenance_incomplete")
    if balance <= 0 or max_loss <= 0 or stop_distance <= 0:
        reasons.append("account_stop_risk_budget_incomplete")
    if position_size <= 0:
        reasons.append("dynamic_position_budget_zero")

    eligible = not reasons
    provenance["fallback_reason"] = "" if eligible else ",".join(reasons)
    return ReturnExecutionAssessment(
        eligible=eligible,
        reason="production_return_policy_passed" if eligible else ",".join(reasons),
        expected_net_return_pct=round(expected_net, 8) if isfinite(expected_net) else 0.0,
        return_lcb_pct=round(return_lcb, 8) if isfinite(return_lcb) else 0.0,
        uncertainty_pct=round(uncertainty, 8),
        expected_loss_pct=round(expected_loss, 8),
        execution_cost_pct=round(cost_pct, 8),
        production_source_count=len(observations),
        position_size_pct=round(position_size, 8),
        policy_provenance=provenance,
    )


def apply_production_entry_policy(decision: DecisionOutput) -> ReturnExecutionAssessment:
    assessment = assess_production_entry(decision)
    raw = _safe_dict(decision.raw_response)
    raw["production_return_policy"] = assessment.to_dict()
    decision.raw_response = raw
    decision.position_size_pct = assessment.position_size_pct if assessment.eligible else 0.0
    return assessment
