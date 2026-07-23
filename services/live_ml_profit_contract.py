"""Unified production entry adjudication for fee-after return quality."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import isclose, isfinite
from typing import Any

from ai_brain.base_model import DecisionOutput
from services.profit_supervision import (
    PRODUCTION_RETURN_COMBINATION_VERSION,
    PROFIT_SUPERVISION_VERSION,
)
from services.return_objective import validate_return_distribution_contract

LIVE_ML_PROFIT_CONTRACT_VERSION = "2026-07-23.live-ml-profit-contract.v1"


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
class LiveMLProfitContractAssessment:
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


def _live_ml_profit_observations(opportunity: dict[str, Any]) -> list[float]:
    if (
        opportunity.get("profit_supervision_version") != PROFIT_SUPERVISION_VERSION
        or opportunity.get("return_combination_version")
        != PRODUCTION_RETURN_COMBINATION_VERSION
        or opportunity.get("return_distribution_mode")
        != "governed_market_opportunity"
    ):
        return []
    breakdown = _safe_dict(opportunity.get("expected_net_breakdown"))
    observations: list[float] = []
    for component in _safe_list(breakdown.get("components")):
        item = _safe_dict(component)
        if (
            item.get("included_in_return_distribution") is not True
            or item.get("production_eligible") is not True
            or _safe_float(item.get("production_weight"), 0.0) <= 0
        ):
            continue
        distribution = _safe_dict(item.get("return_distribution_contract"))
        value = _safe_float(
            distribution.get("raw_expected_return_pct"),
            float("nan"),
        )
        if isfinite(value):
            observations.append(value)
    return observations


def assess_live_ml_profit_contract(decision: DecisionOutput) -> LiveMLProfitContractAssessment:
    raw = _safe_dict(decision.raw_response)
    opportunity = _safe_dict(raw.get("opportunity_score"))
    sizing = _safe_dict(raw.get("profit_risk_sizing"))
    pre_order_facts = _safe_dict(raw.get("pre_order_execution_facts"))
    cost_sizing_pass = _safe_dict(raw.get("execution_cost_sizing_pass"))
    execution_cost = _safe_dict(opportunity.get("execution_cost"))
    breakdown = _safe_dict(opportunity.get("expected_net_breakdown"))
    distribution = _safe_dict(opportunity.get("return_distribution_contract"))
    distribution_validation = validate_return_distribution_contract(
        distribution,
        side=str(opportunity.get("side") or ""),
        return_semantics=(
            "realized_net_return_after_live_cost_and_authoritative_slippage"
        ),
        profit_supervision_version=PROFIT_SUPERVISION_VERSION,
    )
    expected_net = _safe_float(
        distribution.get("raw_expected_return_pct"),
        float("nan"),
    )
    expected_loss = _safe_float(
        distribution.get("tail_loss_penalty_pct"),
        float("nan"),
    )
    cost_pct = max(_safe_float(execution_cost.get("total_pct"), 0.0), 0.0)
    combined_cost_pct = _safe_float(
        breakdown.get("live_execution_cost_pct"),
        float("nan"),
    )
    observations = _live_ml_profit_observations(opportunity)
    uncertainty = _safe_float(
        distribution.get("uncertainty_penalty_pct"),
        float("nan"),
    )
    return_lcb = _safe_float(
        distribution.get("objective_expected_return_pct"),
        float("nan"),
    )

    position_size = max(_safe_float(sizing.get("position_size_pct"), 0.0), 0.0)
    risk_budget = max(_safe_float(sizing.get("risk_budget_usdt"), 0.0), 0.0)
    planned_loss = max(
        _safe_float(sizing.get("planned_stressed_loss_usdt"), 0.0),
        0.0,
    )
    target_notional = max(_safe_float(sizing.get("target_notional_usdt"), 0.0), 0.0)
    final_notional = max(_safe_float(sizing.get("final_notional_usdt"), 0.0), 0.0)
    stressed_loss_fraction = max(
        _safe_float(sizing.get("stressed_loss_fraction"), 0.0),
        0.0,
    )

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
        "source": "validated_realized_net_distribution_and_authoritative_risk_sizing",
        "observation_window": (
            "current_governed_market_live_cost_and_authoritative_trade_calibration"
        ),
        "sample_count": len(observations),
        "generated_at": generated_at,
        "strategy_version": LIVE_ML_PROFIT_CONTRACT_VERSION,
        "fallback_reason": "",
        "upstream_provenance": {
            "return_distribution_mode": opportunity.get("return_distribution_mode"),
            "profit_supervision_version": opportunity.get(
                "profit_supervision_version"
            ),
            "return_combination_version": opportunity.get(
                "return_combination_version"
            ),
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
    if distribution_validation.get("eligible") is not True:
        reasons.extend(distribution_validation.get("blockers") or [])
    if opportunity.get("profit_supervision_version") != PROFIT_SUPERVISION_VERSION:
        reasons.append("profit_supervision_version_mismatch")
    if (
        opportunity.get("return_combination_version")
        != PRODUCTION_RETURN_COMBINATION_VERSION
    ):
        reasons.append("production_return_combination_version_mismatch")
    if opportunity.get("return_distribution_mode") != "governed_market_opportunity":
        reasons.append("non_governed_market_distribution_mode")
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
    if execution_cost.get("order_size_complete") is not True:
        reasons.append("order_size_execution_cost_incomplete")
    if _safe_float(execution_cost.get("order_notional_usdt"), 0.0) + 1e-8 < final_notional:
        reasons.append("execution_cost_notional_below_final_order_notional")
    if pre_order_facts.get("production_eligible") is not True:
        reasons.append("pre_order_execution_facts_ineligible")
    if not str(pre_order_facts.get("input_fingerprint") or "").strip():
        reasons.append("pre_order_execution_facts_fingerprint_missing")
    if cost_sizing_pass.get("order_size_complete") is not True:
        reasons.append("order_size_sizing_pass_incomplete")
    if not cost_provenance_complete:
        reasons.append("execution_cost_policy_provenance_incomplete")
    if (
        not isfinite(combined_cost_pct)
        or not isclose(cost_pct, combined_cost_pct, rel_tol=1e-9, abs_tol=1e-8)
    ):
        reasons.append("live_execution_cost_combination_mismatch")
    if int(_safe_float(breakdown.get("counterfactual_cost_distribution_count"), 0.0)) <= 0:
        reasons.append("counterfactual_cost_distribution_missing")
    if int(_safe_float(breakdown.get("authoritative_trade_calibration_count"), 0.0)) <= 0:
        reasons.append("authoritative_trade_calibration_missing")
    if int(_safe_float(breakdown.get("cost_deduction_count"), 0.0)) != 1:
        reasons.append("execution_cost_deduction_count_invalid")
    if not isfinite(expected_net) or expected_net <= 0:
        reasons.append("fee_after_expected_return_not_positive")
    if not isfinite(uncertainty) or uncertainty < 0:
        reasons.append("realized_net_uncertainty_missing")
    if not isfinite(return_lcb) or return_lcb <= 0:
        reasons.append("fee_after_return_lcb_not_positive")
    if (
        isfinite(expected_net)
        and isfinite(uncertainty)
        and isfinite(return_lcb)
        and not isclose(
            expected_net - uncertainty - expected_loss,
            return_lcb,
            rel_tol=1e-9,
            abs_tol=1e-8,
        )
    ):
        reasons.append("standardized_objective_return_algebra_mismatch")
    if not isfinite(expected_loss) or expected_loss < 0:
        reasons.append("calibrated_downside_missing")
    if sizing.get("production_eligible") is not True:
        reasons.append("dynamic_entry_risk_budget_ineligible")
    if not sizing_provenance_complete:
        reasons.append("dynamic_entry_risk_budget_provenance_incomplete")
    if risk_budget <= 0 or stressed_loss_fraction <= 0:
        reasons.append("independent_risk_budget_incomplete")
    if planned_loss <= 0 or planned_loss > risk_budget + 1e-8:
        reasons.append("planned_stressed_loss_exceeds_risk_budget")
    if final_notional <= 0 or final_notional > target_notional + 1e-8:
        reasons.append("final_notional_exceeds_authoritative_target")
    if not isclose(
        planned_loss,
        final_notional * stressed_loss_fraction,
        rel_tol=1e-9,
        abs_tol=1e-8,
    ):
        reasons.append("risk_sizing_algebra_mismatch")
    if not isclose(
        position_size,
        max(_safe_float(decision.position_size_pct, 0.0), 0.0),
        rel_tol=1e-9,
        abs_tol=1e-8,
    ):
        reasons.append("decision_position_size_differs_from_authoritative_sizing")
    if position_size <= 0:
        reasons.append("dynamic_position_budget_zero")

    eligible = not reasons
    provenance["fallback_reason"] = "" if eligible else ",".join(reasons)
    return LiveMLProfitContractAssessment(
        eligible=eligible,
        reason="live_ml_profit_contract_passed" if eligible else ",".join(reasons),
        expected_net_return_pct=round(expected_net, 8) if isfinite(expected_net) else 0.0,
        return_lcb_pct=round(return_lcb, 8) if isfinite(return_lcb) else 0.0,
        uncertainty_pct=round(uncertainty, 8) if isfinite(uncertainty) else 0.0,
        expected_loss_pct=round(expected_loss, 8) if isfinite(expected_loss) else 0.0,
        execution_cost_pct=round(cost_pct, 8),
        production_source_count=len(observations),
        position_size_pct=round(position_size, 8),
        policy_provenance=provenance,
    )


def apply_live_ml_profit_contract(decision: DecisionOutput) -> LiveMLProfitContractAssessment:
    assessment = assess_live_ml_profit_contract(decision)
    raw = _safe_dict(decision.raw_response)
    raw["live_ml_profit_contract"] = assessment.to_dict()
    decision.raw_response = raw
    if not assessment.eligible:
        decision.position_size_pct = 0.0
    return assessment
