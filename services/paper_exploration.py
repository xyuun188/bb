"""Bounded-risk exploration for normal OKX paper trading.

Exploration is an execution risk profile, not a sample quota.  It can only turn a
fully costed, positive-mean but slightly uncertain paper candidate into a real
trade.  Live execution and known negative-return candidates remain forbidden.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isclose, isfinite
from typing import Any

from ai_brain.base_model import Action, DecisionOutput

PAPER_EXPLORATION_VERSION = "2026-07-21.bounded-paper-exploration.v1"
PAPER_EXPLORATION_SIZING_VERSION = "2026-07-21.bounded-paper-risk.v1"
PAPER_EXPLORATION_MAX_SINGLE_TRADE_RISK_FRACTION = 0.0001
PAPER_EXPLORATION_MAX_PORTFOLIO_RISK_FRACTION = 0.0003
PAPER_EXPLORATION_MAX_LCB_GAP_RATIO = 0.75
PAPER_EXPLORATION_MAX_LOSS_PROBABILITY = 0.60
PAPER_EXPLORATION_MAX_TAIL_RISK_SCORE = 0.60
PAPER_EXPLORATION_MIN_RETURN_SOURCE_COUNT = 2


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if isfinite(result) else default


def _int(value: Any) -> int:
    try:
        return max(int(float(value)), 0)
    except (TypeError, ValueError):
        return 0


def _governance_complete(value: Any) -> bool:
    provenance = _dict(value)
    return bool(
        str(provenance.get("source") or "").strip()
        and str(provenance.get("observation_window") or "").strip()
        and _int(provenance.get("sample_count")) > 0
        and str(provenance.get("generated_at") or "").strip()
        and str(provenance.get("strategy_version") or "").strip()
        and not str(provenance.get("fallback_reason") or "").strip()
    )


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contract_fingerprint_payload(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        key: contract.get(key)
        for key in (
            "version",
            "execution_scope",
            "production_permission",
            "trade_kind",
            "symbol",
            "selected_side",
            "expected_net_return_pct",
            "return_lcb_pct",
            "lcb_gap_ratio",
            "loss_probability",
            "tail_risk_score",
            "return_source_count",
            "feature_opportunity_score",
            "information_value_score",
            "single_trade_risk_fraction_cap",
            "portfolio_risk_fraction_cap",
            "leverage_cap",
            "sample_target",
            "daily_sample_quota",
            "policy_provenance",
        )
    }


def evaluate_paper_exploration_side(
    side_evidence: dict[str, Any],
    *,
    feature_opportunity_score: float,
) -> dict[str, Any]:
    """Classify one side without granting execution permission."""

    evidence = _dict(side_evidence)
    expected_net = _float(evidence.get("expected_net_return_pct"), None)
    return_lcb = _float(evidence.get("return_lcb_pct"), None)
    loss_probability = _float(evidence.get("loss_probability"), 1.0) or 0.0
    tail_risk = _float(evidence.get("tail_risk_score"), 1.0) or 0.0
    source_count = _int(evidence.get("production_source_count"))
    feature_score = max(float(feature_opportunity_score), 0.0)
    cost = _dict(evidence.get("execution_cost"))
    reasons: list[str] = []

    if evidence.get("return_distribution_ready") is not True:
        reasons.append("paper_exploration_return_distribution_incomplete")
    if expected_net is None or expected_net <= 0:
        reasons.append("paper_exploration_expected_net_return_not_positive")
    if return_lcb is None:
        reasons.append("paper_exploration_return_lcb_missing")
    elif return_lcb > 0:
        reasons.append("paper_exploration_should_use_normal_profitable_entry")
    lcb_gap = max(-(return_lcb or 0.0), 0.0)
    lcb_gap_ratio = (
        lcb_gap / expected_net
        if expected_net is not None and expected_net > 0
        else float("inf")
    )
    if not isfinite(lcb_gap_ratio) or lcb_gap_ratio > PAPER_EXPLORATION_MAX_LCB_GAP_RATIO:
        reasons.append("paper_exploration_not_close_to_profitable_threshold")
    if loss_probability > PAPER_EXPLORATION_MAX_LOSS_PROBABILITY:
        reasons.append("paper_exploration_loss_probability_too_high")
    if tail_risk > PAPER_EXPLORATION_MAX_TAIL_RISK_SCORE:
        reasons.append("paper_exploration_tail_risk_too_high")
    if source_count < PAPER_EXPLORATION_MIN_RETURN_SOURCE_COUNT:
        reasons.append("paper_exploration_return_sources_incomplete")
    if feature_score <= 0:
        reasons.append("paper_exploration_feature_value_not_positive")
    if cost.get("production_eligible") is not True or (_float(cost.get("total_pct"), 0.0) or 0.0) <= 0:
        reasons.append("paper_exploration_execution_cost_incomplete")
    if not _governance_complete(evidence.get("policy_provenance")):
        reasons.append("paper_exploration_provenance_incomplete")

    closeness = max(1.0 - min(lcb_gap_ratio, 1.0), 0.0) if isfinite(lcb_gap_ratio) else 0.0
    survival = max(1.0 - loss_probability, 0.0) * max(1.0 - tail_risk, 0.0)
    source_strength = min(source_count / 3.0, 1.0)
    feature_strength = feature_score / (feature_score + 10.0)
    information_value = closeness * survival * source_strength * feature_strength
    if information_value <= 0:
        reasons.append("paper_exploration_information_value_zero")

    return {
        "eligible": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "expected_net_return_pct": round(expected_net, 8) if expected_net is not None else None,
        "return_lcb_pct": round(return_lcb, 8) if return_lcb is not None else None,
        "lcb_gap_ratio": round(lcb_gap_ratio, 8) if isfinite(lcb_gap_ratio) else None,
        "loss_probability": round(loss_probability, 8),
        "tail_risk_score": round(tail_risk, 8),
        "return_source_count": source_count,
        "feature_opportunity_score": round(feature_score, 8),
        "information_value_score": round(information_value, 8),
        "policy_provenance": _dict(evidence.get("policy_provenance")),
    }


def select_paper_exploration_side(
    side_evidence: dict[str, dict[str, Any]],
    *,
    feature_opportunity_score: float,
) -> dict[str, Any]:
    """Choose one identifiable near-threshold side by relative information value."""

    rows = []
    by_side: dict[str, dict[str, Any]] = {}
    for side in ("long", "short"):
        assessed = evaluate_paper_exploration_side(
            _dict(side_evidence.get(side)),
            feature_opportunity_score=feature_opportunity_score,
        )
        assessed["side"] = side
        by_side[side] = assessed
        if assessed["eligible"]:
            rows.append(assessed)
    rows.sort(
        key=lambda item: (
            float(item.get("information_value_score") or 0.0),
            float(item.get("expected_net_return_pct") or 0.0),
        ),
        reverse=True,
    )
    selected = rows[0] if rows else None
    if len(rows) > 1 and isclose(
        float(rows[0].get("information_value_score") or 0.0),
        float(rows[1].get("information_value_score") or 0.0),
        abs_tol=1e-12,
    ):
        selected = None
    return {
        "preferred_side": selected.get("side") if selected else "neutral",
        "selected": dict(selected) if selected else {},
        "by_side": by_side,
        "eligible_side_count": len(rows),
        "reason": (
            "bounded_paper_exploration_side_selected"
            if selected
            else "paper_exploration_direction_not_identifiable"
            if rows
            else "no_bounded_paper_exploration_side"
        ),
    }


def build_paper_exploration_contract(
    candidate_evidence: dict[str, Any],
    *,
    symbol: str,
) -> dict[str, Any]:
    exploration = _dict(candidate_evidence.get("paper_exploration"))
    selected = _dict(exploration.get("selected"))
    side = str(exploration.get("preferred_side") or "").lower()
    if side not in {"long", "short"} or selected.get("eligible") is not True:
        return {}
    generated_at = datetime.now(UTC).isoformat()
    contract = {
        "version": PAPER_EXPLORATION_VERSION,
        "authorized": True,
        "execution_scope": "paper_only",
        "production_permission": False,
        "trade_kind": "normal_trade_with_bounded_exploration_risk",
        "trade_is_normal": True,
        "continuous_training_after_settlement": True,
        "purpose": "execute_positive_mean_uncertain_paper_opportunity_and_learn_after_settlement",
        "symbol": str(symbol or ""),
        "selected_side": side,
        "expected_net_return_pct": selected.get("expected_net_return_pct"),
        "return_lcb_pct": selected.get("return_lcb_pct"),
        "lcb_gap_ratio": selected.get("lcb_gap_ratio"),
        "loss_probability": selected.get("loss_probability"),
        "tail_risk_score": selected.get("tail_risk_score"),
        "return_source_count": selected.get("return_source_count"),
        "feature_opportunity_score": selected.get("feature_opportunity_score"),
        "information_value_score": selected.get("information_value_score"),
        "single_trade_risk_fraction_cap": (
            PAPER_EXPLORATION_MAX_SINGLE_TRADE_RISK_FRACTION
        ),
        "portfolio_risk_fraction_cap": PAPER_EXPLORATION_MAX_PORTFOLIO_RISK_FRACTION,
        "leverage_cap": 1,
        "sample_target": None,
        "daily_sample_quota": None,
        "selection_reason": exploration.get("reason"),
        "policy_provenance": {
            "source": "current_cost_complete_return_distribution_and_ranked_market_features",
            "observation_window": "current_pre_order_paper_candidate",
            "sample_count": _int(selected.get("return_source_count")),
            "generated_at": generated_at,
            "strategy_version": PAPER_EXPLORATION_VERSION,
            "fallback_reason": "",
            "upstream_return_provenance": _dict(selected.get("policy_provenance")),
        },
    }
    contract["contract_fingerprint"] = _fingerprint(
        _contract_fingerprint_payload(contract)
    )
    return contract


def is_paper_exploration_decision(decision: DecisionOutput) -> bool:
    contract = _dict(_dict(decision.raw_response).get("paper_exploration"))
    return bool(
        decision.is_entry
        and contract.get("version") == PAPER_EXPLORATION_VERSION
        and contract.get("authorized") is True
    )


def paper_exploration_contract_reasons(contract_value: Any) -> list[str]:
    """Validate immutable selection fields without requiring a runtime decision."""

    contract = _dict(contract_value)
    reasons: list[str] = []
    if contract.get("version") != PAPER_EXPLORATION_VERSION:
        reasons.append("paper_exploration_version_invalid")
    if contract.get("authorized") is not True:
        reasons.append("paper_exploration_not_authorized")
    if contract.get("execution_scope") != "paper_only":
        reasons.append("paper_exploration_scope_invalid")
    if contract.get("production_permission") is not False:
        reasons.append("paper_exploration_production_permission_invalid")
    if contract.get("trade_is_normal") is not True:
        reasons.append("paper_exploration_normal_trade_contract_missing")
    expected_net = _float(contract.get("expected_net_return_pct"), None)
    return_lcb = _float(contract.get("return_lcb_pct"), None)
    lcb_gap_ratio = _float(contract.get("lcb_gap_ratio"), None)
    if expected_net is None or expected_net <= 0:
        reasons.append("paper_exploration_expected_net_return_not_positive")
    if return_lcb is None or return_lcb > 0:
        reasons.append("paper_exploration_return_lcb_not_uncertain")
    if (
        lcb_gap_ratio is None
        or lcb_gap_ratio < 0
        or lcb_gap_ratio > PAPER_EXPLORATION_MAX_LCB_GAP_RATIO
    ):
        reasons.append("paper_exploration_not_close_to_profitable_threshold")
    if (_float(contract.get("loss_probability"), 1.0) or 0.0) > PAPER_EXPLORATION_MAX_LOSS_PROBABILITY:
        reasons.append("paper_exploration_loss_probability_too_high")
    if (_float(contract.get("tail_risk_score"), 1.0) or 0.0) > PAPER_EXPLORATION_MAX_TAIL_RISK_SCORE:
        reasons.append("paper_exploration_tail_risk_too_high")
    if _int(contract.get("return_source_count")) < PAPER_EXPLORATION_MIN_RETURN_SOURCE_COUNT:
        reasons.append("paper_exploration_return_sources_incomplete")
    if (_float(contract.get("feature_opportunity_score"), 0.0) or 0.0) <= 0:
        reasons.append("paper_exploration_feature_value_not_positive")
    if (_float(contract.get("information_value_score"), 0.0) or 0.0) <= 0:
        reasons.append("paper_exploration_information_value_zero")
    if contract.get("sample_target") is not None or contract.get("daily_sample_quota") is not None:
        reasons.append("paper_exploration_sample_quota_forbidden")
    if not _governance_complete(contract.get("policy_provenance")):
        reasons.append("paper_exploration_provenance_incomplete")
    expected_fingerprint = _fingerprint(_contract_fingerprint_payload(contract))
    if contract.get("contract_fingerprint") != expected_fingerprint:
        reasons.append("paper_exploration_contract_fingerprint_mismatch")
    return list(dict.fromkeys(reasons))


def paper_exploration_selection_reasons(
    decision: DecisionOutput,
    model_mode: str,
) -> list[str]:
    raw = _dict(decision.raw_response)
    contract = _dict(raw.get("paper_exploration"))
    side = "long" if decision.action == Action.LONG else "short"
    reasons = paper_exploration_contract_reasons(contract)
    if str(model_mode or "").lower() != "paper":
        reasons.append("paper_exploration_live_execution_forbidden")
    if contract.get("selected_side") != side:
        reasons.append("paper_exploration_side_mismatch")
    if str(contract.get("symbol") or "") != str(decision.symbol or ""):
        reasons.append("paper_exploration_symbol_mismatch")
    evidence = _dict(raw.get("entry_candidate_evidence"))
    selected = _dict(_dict(evidence.get("paper_exploration")).get("selected"))
    if selected.get("eligible") is not True or selected.get("side") != side:
        reasons.append("paper_exploration_selected_evidence_incomplete")
    return list(dict.fromkeys(reasons))


@dataclass(frozen=True, slots=True)
class PaperExplorationAssessment:
    eligible: bool
    reason: str
    blocking_reasons: list[str]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "reason": self.reason,
            "blocking_reasons": list(self.blocking_reasons),
            "details": dict(self.details),
        }


def assess_paper_exploration_entry(
    decision: DecisionOutput,
    model_mode: str,
) -> PaperExplorationAssessment:
    """Validate the final paper-only selection, cost, and risk contract."""

    raw = _dict(decision.raw_response)
    contract = _dict(raw.get("paper_exploration"))
    sizing = _dict(raw.get("profit_risk_sizing"))
    opportunity = _dict(raw.get("opportunity_score"))
    distribution = _dict(opportunity.get("return_distribution_contract"))
    execution_cost = _dict(opportunity.get("execution_cost"))
    pre_order = _dict(raw.get("pre_order_execution_facts"))
    sizing_pass = _dict(raw.get("execution_cost_sizing_pass"))
    reasons = paper_exploration_selection_reasons(decision, model_mode)
    equity = _float(sizing.get("account_equity_usdt"), 0.0) or 0.0
    risk_budget = _float(sizing.get("risk_budget_usdt"), 0.0) or 0.0
    portfolio_budget = _float(sizing.get("portfolio_risk_budget_usdt"), 0.0) or 0.0
    planned_loss = _float(sizing.get("planned_stressed_loss_usdt"), 0.0) or 0.0
    final_notional = _float(sizing.get("final_notional_usdt"), 0.0) or 0.0
    current_expected_net = _float(
        distribution.get("raw_expected_return_pct"),
        _float(opportunity.get("expected_net_return_pct"), None),
    )
    current_return_lcb = _float(
        distribution.get("objective_expected_return_pct"),
        _float(opportunity.get("return_lcb_pct"), None),
    )
    if current_expected_net is None or current_expected_net <= 0:
        reasons.append("paper_exploration_size_aware_expected_return_not_positive")
    if current_return_lcb is None or current_return_lcb > 0:
        reasons.append("paper_exploration_size_aware_lcb_not_uncertain")
    elif (
        current_expected_net is not None
        and current_expected_net > 0
        and max(-current_return_lcb, 0.0) / current_expected_net
        > PAPER_EXPLORATION_MAX_LCB_GAP_RATIO
    ):
        reasons.append("paper_exploration_size_aware_not_close_to_threshold")
    if sizing.get("production_eligible") is not True:
        reasons.append("paper_exploration_risk_contract_ineligible")
    if sizing.get("contract_lifecycle") != "paper_exploration":
        reasons.append("paper_exploration_risk_lifecycle_invalid")
    if sizing.get("execution_scope") != "paper_only" or sizing.get("production_permission") is not False:
        reasons.append("paper_exploration_risk_scope_invalid")
    if sizing.get("contract_version") != PAPER_EXPLORATION_SIZING_VERSION:
        reasons.append("paper_exploration_sizing_version_invalid")
    if equity <= 0 or risk_budget <= 0 or planned_loss <= 0 or final_notional <= 0:
        reasons.append("paper_exploration_risk_budget_incomplete")
    if planned_loss > risk_budget + 1e-8:
        reasons.append("paper_exploration_planned_loss_exceeds_budget")
    if risk_budget > equity * PAPER_EXPLORATION_MAX_SINGLE_TRADE_RISK_FRACTION + 1e-8:
        reasons.append("paper_exploration_single_trade_risk_cap_exceeded")
    if portfolio_budget > equity * PAPER_EXPLORATION_MAX_PORTFOLIO_RISK_FRACTION + 1e-8:
        reasons.append("paper_exploration_portfolio_risk_cap_exceeded")
    if not isclose(float(decision.suggested_leverage), 1.0, abs_tol=1e-8):
        reasons.append("paper_exploration_leverage_must_be_one")
    if (_float(decision.position_size_pct, 0.0) or 0.0) <= 0:
        reasons.append("paper_exploration_position_size_zero")
    if execution_cost.get("production_eligible") is not True:
        reasons.append("paper_exploration_execution_cost_incomplete")
    if execution_cost.get("order_size_complete") is not True:
        reasons.append("paper_exploration_order_size_cost_incomplete")
    if pre_order.get("production_eligible") is not True or not str(
        pre_order.get("input_fingerprint") or ""
    ).strip():
        reasons.append("paper_exploration_pre_order_facts_incomplete")
    if sizing_pass.get("order_size_complete") is not True:
        reasons.append("paper_exploration_size_aware_cost_incomplete")
    reasons = list(dict.fromkeys(reasons))
    return PaperExplorationAssessment(
        eligible=not reasons,
        reason=("paper_exploration_contract_ready" if not reasons else ",".join(reasons)),
        blocking_reasons=reasons,
        details={
            "contract": contract,
            "sizing": sizing,
            "execution_cost": execution_cost,
            "size_aware_expected_net_return_pct": current_expected_net,
            "size_aware_return_lcb_pct": current_return_lcb,
            "pre_order_execution_facts": pre_order,
        },
    )
