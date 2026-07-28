"""Single execution contract for every new OKX paper strategy trade."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from math import isclose, isfinite
from typing import Any

from ai_brain.base_model import DecisionOutput

NORMAL_PAPER_TRADE_VERSION = "2026-07-28.normal-paper-strategy-trade.v3"
NORMAL_PAPER_TRADE_SIZING_VERSION = "2026-07-28.normal-paper-dynamic-risk.v3"
LEGACY_NORMAL_PAPER_TRADE_VERSION = "2026-07-27.normal-paper-strategy-trade.v2"
LEGACY_NORMAL_PAPER_TRADE_SIZING_VERSION = "2026-07-27.normal-paper-risk.v2"
HISTORICAL_NORMAL_PAPER_TRADE_VERSION = "2026-07-22.normal-paper-trade.v1"
HISTORICAL_NORMAL_PAPER_TRADE_ROUTES = {
    "evidence_best",
    "evidence_best_canary",
    "bounded_exploration",
    "cold_start_exploration",
}
NORMAL_PAPER_TRADE_SELECTION_REASONS = {
    "policy_exploitation",
    "coverage_sampling",
}
NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION = 0.0005
NORMAL_PAPER_TRADE_MAX_COVERAGE_RISK_FRACTION = 0.0001
NORMAL_PAPER_TRADE_LEVERAGE_POLICY = "dynamic_risk_and_okx_tier"
NORMAL_PAPER_TRADE_MIN_FILL_DRIFT_RESERVE_FRACTION = 0.0025


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


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
            "authorized",
            "trade_mode",
            "execution_scope",
            "entry_type",
            "trade_kind",
            "production_permission",
            "decision_authority",
            "selection_reason",
            "symbol",
            "side",
            "prediction_horizon_minutes",
            "valid_for_seconds",
            "expected_net_return_pct",
            "objective_net_return_pct",
            "loss_probability",
            "quant_evidence_families",
            "strong_expert_opposition",
            "single_trade_risk_fraction_cap",
            "portfolio_risk_fraction_cap",
            "leverage_policy",
            "model_leverage_role",
            "uses_shared_order_pipeline",
            "uses_shared_position_ledger",
            "continuous_training_after_trusted_settlement",
            "separate_sampling_order",
            "risk_override_permission",
            "sample_target",
            "daily_sample_quota",
        )
    }


def select_normal_paper_trade_side(
    support_by_side: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Select one auditable model direction without applying promotion statistics."""

    by_side = {side: dict(_dict(_dict(support_by_side).get(side))) for side in ("long", "short")}
    candidates: list[dict[str, Any]] = []
    for side, support in by_side.items():
        if support.get("eligible") is not True:
            continue
        expected_net = _float(support.get("expected_net_return_pct"), None)
        objective_net = _float(support.get("objective_net_return_pct"), None)
        loss_probability = _float(support.get("loss_probability"), 1.0) or 1.0
        families = sorted(
            {
                str(item).strip()
                for item in support.get("quant_evidence_families") or []
                if str(item).strip()
            }
        )
        candidates.append(
            {
                "side": side,
                "support": support,
                "expected_net_return_pct": expected_net,
                "objective_net_return_pct": objective_net,
                "loss_probability": loss_probability,
                "quant_evidence_families": families,
                "selection_reason": (
                    "policy_exploitation"
                    if expected_net is not None and expected_net > 0.0
                    else "coverage_sampling"
                ),
            }
        )

    candidates.sort(
        key=lambda item: (
            float(item["expected_net_return_pct"])
            if item["expected_net_return_pct"] is not None
            else float("-inf"),
            float(item["objective_net_return_pct"])
            if item["objective_net_return_pct"] is not None
            else float("-inf"),
            len(item["quant_evidence_families"]),
            -float(item["loss_probability"]),
        ),
        reverse=True,
    )
    selected = candidates[0] if candidates else None
    if len(candidates) > 1:
        first = candidates[0]
        second = candidates[1]
        first_expected = first["expected_net_return_pct"]
        second_expected = second["expected_net_return_pct"]
        first_objective = first["objective_net_return_pct"]
        second_objective = second["objective_net_return_pct"]
        if (
            first_expected is not None
            and second_expected is not None
            and first_objective is not None
            and second_objective is not None
            and isclose(first_expected, second_expected, abs_tol=1e-12)
            and isclose(first_objective, second_objective, abs_tol=1e-12)
        ):
            selected = None

    return {
        "version": NORMAL_PAPER_TRADE_VERSION,
        "selected": bool(selected),
        "selected_side": selected["side"] if selected else "neutral",
        "selection_reason": selected["selection_reason"] if selected else "no_direction",
        "selected_support": dict(selected["support"]) if selected else {},
        "eligible_side_count": len(candidates),
        "by_side": by_side,
        "production_permission": False,
    }


def build_normal_paper_trade_contract(
    *,
    symbol: str,
    side: str,
    selection_reason: str,
    direction_support: dict[str, Any],
    decision_authority: str = "ensemble",
) -> dict[str, Any]:
    """Build the only contract that can authorize a new paper strategy entry."""

    normalized_side = str(side or "").lower()
    support = _dict(direction_support)
    horizon = _float(support.get("prediction_horizon_minutes"), 0.0) or 0.0
    if (
        normalized_side not in {"long", "short"}
        or selection_reason not in NORMAL_PAPER_TRADE_SELECTION_REASONS
        or support.get("eligible") is not True
        or support.get("selected_side") != normalized_side
        or horizon <= 0.0
    ):
        return {}

    single_trade_cap = (
        NORMAL_PAPER_TRADE_MAX_COVERAGE_RISK_FRACTION
        if selection_reason == "coverage_sampling"
        else NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION
    )
    contract = {
        "version": NORMAL_PAPER_TRADE_VERSION,
        "authorized": True,
        "trade_mode": "paper",
        "execution_scope": "paper_only",
        "entry_type": "normal_strategy_trade",
        "trade_kind": "normal_strategy_trade",
        "production_permission": False,
        "decision_authority": str(decision_authority or "ensemble"),
        "selection_reason": selection_reason,
        "symbol": str(symbol or ""),
        "side": normalized_side,
        "prediction_horizon_minutes": horizon,
        "valid_for_seconds": horizon * 60.0,
        "expected_net_return_pct": _float(support.get("expected_net_return_pct"), None),
        "objective_net_return_pct": _float(support.get("objective_net_return_pct"), None),
        "loss_probability": _float(support.get("loss_probability"), None),
        "quant_evidence_families": list(support.get("quant_evidence_families") or []),
        "strong_expert_opposition": bool(support.get("strong_expert_opposition") is True),
        "single_trade_risk_fraction_cap": single_trade_cap,
        "leverage_policy": NORMAL_PAPER_TRADE_LEVERAGE_POLICY,
        "model_leverage_role": "upper_bound_when_explicit",
        "uses_shared_order_pipeline": True,
        "uses_shared_position_ledger": True,
        "continuous_training_after_trusted_settlement": True,
        "training_eligibility_source": ("trusted_settlement_and_task_specific_training_contract"),
        "separate_sampling_order": False,
        "risk_override_permission": False,
        "sample_target": None,
        "daily_sample_quota": None,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    contract["contract_fingerprint"] = _fingerprint(_contract_fingerprint_payload(contract))
    return contract


def ensure_normal_paper_trade_contract(
    decision: DecisionOutput,
    model_mode: str,
) -> dict[str, Any]:
    """Attach a pre-authorized normal-paper contract; never infer permission."""

    if str(model_mode or "").lower() != "paper" or not decision.is_entry:
        return {}
    raw = _dict(decision.raw_response)
    existing = _dict(raw.get("normal_paper_trade"))
    if not normal_paper_trade_contract_reasons(existing):
        return existing

    selection = _dict(raw.get("paper_trade_selection"))
    support = _dict(raw.get("independent_direction_support"))
    contract = build_normal_paper_trade_contract(
        symbol=decision.symbol,
        side="long" if str(decision.action.value).lower() == "long" else "short",
        selection_reason=str(selection.get("selection_reason") or ""),
        direction_support=support,
        decision_authority=str(selection.get("decision_authority") or "ensemble"),
    )
    if contract:
        raw["normal_paper_trade"] = contract
        decision.raw_response = raw
    return contract


def is_normal_paper_trade_decision(decision: DecisionOutput) -> bool:
    if not decision.is_entry:
        return False
    contract = _dict(_dict(decision.raw_response).get("normal_paper_trade"))
    return not normal_paper_trade_contract_reasons(contract)


def build_normal_paper_position_lifecycle(decision: Any) -> dict[str, Any]:
    """Snapshot the normal paper entry lineage without creating exit behavior."""

    raw = _dict(getattr(decision, "raw_response", None))
    contract = _dict(raw.get("normal_paper_trade"))
    if normal_paper_trade_contract_reasons(contract):
        return {}
    return {
        "version": NORMAL_PAPER_TRADE_VERSION,
        "kind": "normal_strategy_position",
        "execution_scope": "paper_only",
        "entry_type": "normal_strategy_trade",
        "production_permission": False,
        "decision_authority": contract.get("decision_authority"),
        "selection_reason": contract.get("selection_reason"),
        "prediction_horizon_minutes": contract.get("prediction_horizon_minutes"),
        "entry_decision_id": getattr(decision, "id", None),
        "entry_contract_fingerprint": contract.get("contract_fingerprint"),
        "horizon_is_exit_deadline": False,
    }


def normal_paper_trade_contract_reasons(value: Any) -> list[str]:
    contract = _dict(value)
    reasons: list[str] = []
    if contract.get("version") != NORMAL_PAPER_TRADE_VERSION:
        reasons.append("normal_paper_trade_version_invalid")
    if contract.get("authorized") is not True:
        reasons.append("normal_paper_trade_not_authorized")
    if contract.get("trade_mode") != "paper":
        reasons.append("normal_paper_trade_mode_invalid")
    if contract.get("execution_scope") != "paper_only":
        reasons.append("normal_paper_trade_scope_invalid")
    if contract.get("entry_type") != "normal_strategy_trade":
        reasons.append("normal_paper_trade_entry_type_invalid")
    if contract.get("trade_kind") != "normal_strategy_trade":
        reasons.append("normal_paper_trade_kind_invalid")
    if contract.get("production_permission") is not False:
        reasons.append("normal_paper_trade_production_permission_invalid")
    if contract.get("decision_authority") not in {"model", "ensemble"}:
        reasons.append("normal_paper_trade_decision_authority_invalid")
    if contract.get("selection_reason") not in NORMAL_PAPER_TRADE_SELECTION_REASONS:
        reasons.append("normal_paper_trade_selection_reason_invalid")
    if str(contract.get("side") or "").lower() not in {"long", "short"}:
        reasons.append("normal_paper_trade_side_missing")
    if not str(contract.get("symbol") or "").strip():
        reasons.append("normal_paper_trade_symbol_missing")
    if contract.get("strong_expert_opposition") is True:
        reasons.append("normal_paper_trade_strong_expert_opposition")
    if contract.get("uses_shared_order_pipeline") is not True:
        reasons.append("normal_paper_trade_order_pipeline_split")
    if contract.get("uses_shared_position_ledger") is not True:
        reasons.append("normal_paper_trade_position_ledger_split")
    if contract.get("separate_sampling_order") is not False:
        reasons.append("normal_paper_trade_sampling_order_split")
    if contract.get("continuous_training_after_trusted_settlement") is not True:
        reasons.append("normal_paper_trade_training_disabled")
    if contract.get("risk_override_permission") is not False:
        reasons.append("normal_paper_trade_risk_override_invalid")
    if contract.get("sample_target") is not None or contract.get("daily_sample_quota") is not None:
        reasons.append("normal_paper_trade_sample_quota_forbidden")
    horizon = _float(contract.get("prediction_horizon_minutes"), 0.0) or 0.0
    valid_for = _float(contract.get("valid_for_seconds"), 0.0) or 0.0
    if horizon <= 0.0 or not isclose(valid_for, horizon * 60.0, abs_tol=1e-8):
        reasons.append("normal_paper_trade_horizon_invalid")
    single_cap = _float(contract.get("single_trade_risk_fraction_cap"), 0.0) or 0.0
    expected_single_cap = (
        NORMAL_PAPER_TRADE_MAX_COVERAGE_RISK_FRACTION
        if contract.get("selection_reason") == "coverage_sampling"
        else NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION
    )
    if not isclose(single_cap, expected_single_cap, abs_tol=1e-12):
        reasons.append("normal_paper_trade_single_risk_cap_invalid")
    if contract.get("leverage_policy") != NORMAL_PAPER_TRADE_LEVERAGE_POLICY:
        reasons.append("normal_paper_trade_leverage_policy_invalid")
    if contract.get("model_leverage_role") != "upper_bound_when_explicit":
        reasons.append("normal_paper_trade_model_leverage_role_invalid")
    if contract.get("contract_fingerprint") != _fingerprint(
        _contract_fingerprint_payload(contract)
    ):
        reasons.append("normal_paper_trade_fingerprint_mismatch")
    return list(dict.fromkeys(reasons))


def legacy_normal_paper_v2_trade_contract_reasons(value: Any) -> list[str]:
    """Validate the immutable fixed-1x v2 envelope for historical outcomes only."""

    contract = _dict(value)
    reasons: list[str] = []
    if contract.get("version") != LEGACY_NORMAL_PAPER_TRADE_VERSION:
        reasons.append("legacy_normal_paper_trade_version_invalid")
    if contract.get("authorized") is not True:
        reasons.append("legacy_normal_paper_trade_not_authorized")
    if contract.get("trade_mode") != "paper" or contract.get("execution_scope") != "paper_only":
        reasons.append("legacy_normal_paper_trade_scope_invalid")
    if contract.get("entry_type") != "normal_strategy_trade" or contract.get(
        "trade_kind"
    ) != "normal_strategy_trade":
        reasons.append("legacy_normal_paper_trade_kind_invalid")
    if contract.get("production_permission") is not False:
        reasons.append("legacy_normal_paper_trade_production_permission_invalid")
    if contract.get("selection_reason") not in NORMAL_PAPER_TRADE_SELECTION_REASONS:
        reasons.append("legacy_normal_paper_trade_selection_reason_invalid")
    if str(contract.get("side") or "").lower() not in {"long", "short"}:
        reasons.append("legacy_normal_paper_trade_side_missing")
    if not str(contract.get("symbol") or "").strip():
        reasons.append("legacy_normal_paper_trade_symbol_missing")
    if not isclose(_float(contract.get("leverage_cap"), 0.0) or 0.0, 1.0, abs_tol=1e-12):
        reasons.append("legacy_normal_paper_trade_leverage_cap_invalid")
    legacy_fingerprint_payload = {
        key: contract.get(key)
        for key in (
            "version",
            "authorized",
            "trade_mode",
            "execution_scope",
            "entry_type",
            "trade_kind",
            "production_permission",
            "decision_authority",
            "selection_reason",
            "symbol",
            "side",
            "prediction_horizon_minutes",
            "valid_for_seconds",
            "expected_net_return_pct",
            "objective_net_return_pct",
            "loss_probability",
            "quant_evidence_families",
            "strong_expert_opposition",
            "single_trade_risk_fraction_cap",
            "portfolio_risk_fraction_cap",
            "leverage_cap",
            "uses_shared_order_pipeline",
            "uses_shared_position_ledger",
            "continuous_training_after_trusted_settlement",
            "separate_sampling_order",
            "risk_override_permission",
            "sample_target",
            "daily_sample_quota",
        )
    }
    if contract.get("contract_fingerprint") != _fingerprint(legacy_fingerprint_payload):
        reasons.append("legacy_normal_paper_trade_fingerprint_mismatch")
    return list(dict.fromkeys(reasons))


def historical_normal_paper_trade_contract_reasons(value: Any) -> list[str]:
    """Validate the immutable v1 envelope for settlement and training recovery only."""

    contract = _dict(value)
    reasons: list[str] = []
    if contract.get("version") != HISTORICAL_NORMAL_PAPER_TRADE_VERSION:
        reasons.append("historical_normal_paper_trade_version_invalid")
    if contract.get("authorized") is not True:
        reasons.append("historical_normal_paper_trade_not_authorized")
    if contract.get("execution_scope") != "paper_only":
        reasons.append("historical_normal_paper_trade_scope_invalid")
    if contract.get("live_execution_permission") is not False:
        reasons.append("historical_normal_paper_trade_live_permission_invalid")
    if contract.get("trade_kind") != "normal_paper_trade":
        reasons.append("historical_normal_paper_trade_kind_invalid")
    if contract.get("route_kind") not in HISTORICAL_NORMAL_PAPER_TRADE_ROUTES:
        reasons.append("historical_normal_paper_trade_route_invalid")
    if str(contract.get("side") or "").lower() not in {"long", "short"}:
        reasons.append("historical_normal_paper_trade_side_missing")
    if not str(contract.get("symbol") or "").strip():
        reasons.append("historical_normal_paper_trade_symbol_missing")
    if contract.get("uses_shared_order_pipeline") is not True:
        reasons.append("historical_normal_paper_trade_order_pipeline_split")
    if contract.get("uses_shared_position_ledger") is not True:
        reasons.append("historical_normal_paper_trade_position_ledger_split")
    if contract.get("separate_sampling_order") is not False:
        reasons.append("historical_normal_paper_trade_sampling_order_split")
    if contract.get("continuous_training_after_trusted_settlement") is not True:
        reasons.append("historical_normal_paper_trade_training_disabled")
    if contract.get("order_creation_owner") != "ensemble_trader_unified_decision":
        reasons.append("historical_normal_paper_trade_order_owner_invalid")
    if contract.get("risk_override_permission") is not False:
        reasons.append("historical_normal_paper_trade_risk_override_invalid")
    if contract.get("sample_target") is not None or contract.get("daily_sample_quota") is not None:
        reasons.append("historical_normal_paper_trade_sample_quota_forbidden")
    horizon = _float(contract.get("prediction_horizon_minutes"), 0.0) or 0.0
    valid_for = _float(contract.get("valid_for_seconds"), 0.0) or 0.0
    if horizon <= 0.0 or not isclose(valid_for, horizon * 60.0, abs_tol=1e-8):
        reasons.append("historical_normal_paper_trade_horizon_invalid")
    expected_fingerprint = _fingerprint(
        {
            key: item
            for key, item in contract.items()
            if key not in {"generated_at", "contract_fingerprint"}
        }
    )
    if contract.get("contract_fingerprint") != expected_fingerprint:
        reasons.append("historical_normal_paper_trade_fingerprint_mismatch")
    return list(dict.fromkeys(reasons))
