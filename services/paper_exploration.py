"""Read-only validation for retired paper-exploration contracts."""

from __future__ import annotations

import hashlib
import json
from math import isclose, isfinite, sqrt
from typing import Any

from services.entry_direction_support import directional_entry_support_reasons

PAPER_EXPLORATION_VERSION = "2026-07-27.independent-paper-exploration.v5"
PAPER_EXPLORATION_SIZING_VERSION = "2026-07-21.bounded-paper-risk.v1"
PAPER_EXPLORATION_MAX_SINGLE_TRADE_RISK_FRACTION = 0.0001
PAPER_EXPLORATION_MAX_PORTFOLIO_RISK_FRACTION = 0.0003
PAPER_EXPLORATION_MAX_LCB_GAP_RATIO = 0.75
PAPER_EXPLORATION_MAX_LOSS_PROBABILITY = 0.60
PAPER_EXPLORATION_MAX_TAIL_RISK_SCORE = 0.60
PAPER_EXPLORATION_MIN_RETURN_SOURCE_COUNT = 2
UNPROMOTED_MODEL_INTERVENTION_SCOPE = "bounded_unpromoted_model_intervention"


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
            "intervention_scope",
            "expected_net_return_pct",
            "objective_net_return_pct",
            "return_lcb_pct",
            "lcb_gap_ratio",
            "loss_probability",
            "tail_risk_score",
            "return_source_count",
            "historical_evidence_count",
            "validated_route_evidence_count",
            "reliable_evidence_count",
            "exploration_maturity_source",
            "exploration_maturity_evidence",
            "exploration_allocation_multiplier",
            "feature_opportunity_score",
            "information_value_score",
            "independent_direction_support",
            "prediction_horizon_minutes",
            "valid_for_seconds",
            "single_trade_risk_fraction_cap",
            "portfolio_risk_fraction_cap",
            "leverage_cap",
            "sample_target",
            "daily_sample_quota",
            "policy_provenance",
        )
    }


def paper_exploration_contract_reasons(contract_value: Any) -> list[str]:
    """Validate an immutable historical exploration contract."""

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
    reasons.extend(
        directional_entry_support_reasons(
            contract.get("independent_direction_support"),
            str(contract.get("selected_side") or ""),
        )
    )
    expected_net = _float(contract.get("expected_net_return_pct"), None)
    return_lcb = _float(contract.get("return_lcb_pct"), None)
    lcb_gap_ratio = _float(contract.get("lcb_gap_ratio"), None)
    if expected_net is None or expected_net <= 0:
        reasons.append("paper_exploration_expected_net_return_not_positive")
    intervention_scope = str(contract.get("intervention_scope") or "")
    if return_lcb is None or (
        return_lcb > 0 and intervention_scope != UNPROMOTED_MODEL_INTERVENTION_SCOPE
    ):
        reasons.append("paper_exploration_return_lcb_not_uncertain")
    if (
        lcb_gap_ratio is None
        or lcb_gap_ratio < 0
        or lcb_gap_ratio > PAPER_EXPLORATION_MAX_LCB_GAP_RATIO
    ):
        reasons.append("paper_exploration_not_close_to_profitable_threshold")
    if (
        _float(contract.get("loss_probability"), 1.0) or 0.0
    ) > PAPER_EXPLORATION_MAX_LOSS_PROBABILITY:
        reasons.append("paper_exploration_loss_probability_too_high")
    if (
        _float(contract.get("tail_risk_score"), 1.0) or 0.0
    ) > PAPER_EXPLORATION_MAX_TAIL_RISK_SCORE:
        reasons.append("paper_exploration_tail_risk_too_high")
    if _int(contract.get("return_source_count")) < PAPER_EXPLORATION_MIN_RETURN_SOURCE_COUNT:
        reasons.append("paper_exploration_return_sources_incomplete")
    horizon_minutes = _float(contract.get("prediction_horizon_minutes"), 0.0) or 0.0
    valid_for_seconds = _float(contract.get("valid_for_seconds"), 0.0) or 0.0
    if horizon_minutes <= 0:
        reasons.append("paper_exploration_prediction_horizon_missing")
    if valid_for_seconds <= 0 or not isclose(
        valid_for_seconds,
        horizon_minutes * 60.0,
        rel_tol=1e-9,
        abs_tol=1e-8,
    ):
        reasons.append("paper_exploration_validity_contract_invalid")
    if (_float(contract.get("feature_opportunity_score"), 0.0) or 0.0) <= 0:
        reasons.append("paper_exploration_feature_value_not_positive")
    if (_float(contract.get("information_value_score"), 0.0) or 0.0) <= 0:
        reasons.append("paper_exploration_information_value_zero")
    allocation = _float(contract.get("exploration_allocation_multiplier"), 0.0) or 0.0
    historical_evidence_count = _int(contract.get("historical_evidence_count"))
    validated_route_evidence_count = _int(contract.get("validated_route_evidence_count"))
    reliable_evidence_count = _int(contract.get("reliable_evidence_count"))
    expected_reliable_evidence_count = max(
        historical_evidence_count,
        validated_route_evidence_count,
    )
    expected_maturity_source = (
        "validated_continuous_strategy_route"
        if validated_route_evidence_count >= historical_evidence_count
        and validated_route_evidence_count > 0
        else "governed_historical_prior"
        if historical_evidence_count > 0
        else "cold_start"
    )
    maturity_evidence = _dict(contract.get("exploration_maturity_evidence"))
    expected_allocation = max(
        0.10,
        1.0 / sqrt(1.0 + reliable_evidence_count / 20.0),
    )
    single_cap = _float(contract.get("single_trade_risk_fraction_cap"), 0.0) or 0.0
    portfolio_cap = _float(contract.get("portfolio_risk_fraction_cap"), 0.0) or 0.0
    if allocation < 0.10 or allocation > 1.0:
        reasons.append("paper_exploration_allocation_multiplier_invalid")
    if reliable_evidence_count != expected_reliable_evidence_count:
        reasons.append("paper_exploration_reliable_evidence_count_mismatch")
    if contract.get("exploration_maturity_source") != expected_maturity_source:
        reasons.append("paper_exploration_maturity_source_mismatch")
    if validated_route_evidence_count > 0 and not (
        maturity_evidence.get("available") is True
        and maturity_evidence.get("source") == "validated_continuous_strategy_route"
        and _int(maturity_evidence.get("evidence_count")) == validated_route_evidence_count
        and maturity_evidence.get("can_authorize_entry") is False
        and maturity_evidence.get("can_change_size_or_leverage") is False
    ):
        reasons.append("paper_exploration_route_maturity_evidence_invalid")
    if not isclose(allocation, expected_allocation, abs_tol=1e-8):
        reasons.append("paper_exploration_maturity_allocation_mismatch")
    if not isclose(
        single_cap,
        PAPER_EXPLORATION_MAX_SINGLE_TRADE_RISK_FRACTION * allocation,
        abs_tol=1e-12,
    ):
        reasons.append("paper_exploration_single_trade_risk_cap_invalid")
    if not isclose(
        portfolio_cap,
        PAPER_EXPLORATION_MAX_PORTFOLIO_RISK_FRACTION * allocation,
        abs_tol=1e-12,
    ):
        reasons.append("paper_exploration_portfolio_risk_cap_invalid")
    if contract.get("sample_target") is not None or contract.get("daily_sample_quota") is not None:
        reasons.append("paper_exploration_sample_quota_forbidden")
    if not _governance_complete(contract.get("policy_provenance")):
        reasons.append("paper_exploration_provenance_incomplete")
    if contract.get("contract_fingerprint") != _fingerprint(
        _contract_fingerprint_payload(contract)
    ):
        reasons.append("paper_exploration_contract_fingerprint_mismatch")
    return list(dict.fromkeys(reasons))
