"""One business contract for every executable simulated trade."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from math import isclose, isfinite
from typing import Any

from ai_brain.base_model import DecisionOutput

NORMAL_PAPER_TRADE_VERSION = "2026-07-22.normal-paper-trade.v1"
NORMAL_PAPER_TRADE_ROUTES = {
    "evidence_best",
    "evidence_best_canary",
    "bounded_exploration",
    "cold_start_exploration",
}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any, default: float = 0.0) -> float:
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


def _route_kind(raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if _dict(raw.get("paper_training")).get("authorized") is True:
        return "cold_start_exploration", _dict(raw.get("paper_training"))
    if _dict(raw.get("paper_exploration")).get("authorized") is True:
        return "bounded_exploration", _dict(raw.get("paper_exploration"))
    if _dict(raw.get("paper_bootstrap_canary")).get("authorized") is True:
        return "evidence_best_canary", _dict(raw.get("paper_bootstrap_canary"))
    return "evidence_best", _dict(raw.get("authoritative_return_candidate"))


def _prediction_horizon(raw: dict[str, Any], source: dict[str, Any]) -> float:
    direct = _float(source.get("prediction_horizon_minutes"), 0.0)
    if direct > 0:
        return direct
    opportunity = _dict(raw.get("opportunity_score"))
    distribution = _dict(opportunity.get("return_distribution_contract"))
    return max(_float(distribution.get("horizon_minutes"), 0.0), 0.0)


def ensure_normal_paper_trade_contract(
    decision: DecisionOutput,
    model_mode: str,
) -> dict[str, Any]:
    """Attach the shared paper order/ledger/training lifecycle to one entry."""

    if str(model_mode or "").lower() != "paper" or not decision.is_entry:
        return {}
    raw = _dict(decision.raw_response)
    existing = _dict(raw.get("normal_paper_trade"))
    if existing.get("version") == NORMAL_PAPER_TRADE_VERSION:
        return existing
    route_kind, source = _route_kind(raw)
    side = "long" if str(decision.action.value).lower() == "long" else "short"
    horizon = _prediction_horizon(raw, source)
    contract = {
        "version": NORMAL_PAPER_TRADE_VERSION,
        "authorized": True,
        "execution_scope": "paper_only",
        "live_execution_permission": False,
        "trade_kind": "normal_paper_trade",
        "route_kind": route_kind,
        "symbol": str(decision.symbol or ""),
        "side": side,
        "prediction_horizon_minutes": horizon,
        "valid_for_seconds": horizon * 60.0,
        "uses_shared_order_pipeline": True,
        "uses_shared_position_ledger": True,
        "separate_sampling_order": False,
        "continuous_training_after_trusted_settlement": True,
        "training_eligibility_source": "trusted_settlement_and_training_quarantine",
        "order_creation_owner": "ensemble_trader_unified_decision",
        "risk_override_permission": False,
        "sample_target": None,
        "daily_sample_quota": None,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    contract["contract_fingerprint"] = _fingerprint(
        {key: value for key, value in contract.items() if key != "generated_at"}
    )
    raw["normal_paper_trade"] = contract
    decision.raw_response = raw
    return contract


def normal_paper_trade_contract_reasons(value: Any) -> list[str]:
    contract = _dict(value)
    reasons: list[str] = []
    if contract.get("version") != NORMAL_PAPER_TRADE_VERSION:
        reasons.append("normal_paper_trade_version_invalid")
    if contract.get("authorized") is not True:
        reasons.append("normal_paper_trade_not_authorized")
    if contract.get("execution_scope") != "paper_only":
        reasons.append("normal_paper_trade_scope_invalid")
    if contract.get("live_execution_permission") is not False:
        reasons.append("normal_paper_trade_live_permission_invalid")
    if contract.get("trade_kind") != "normal_paper_trade":
        reasons.append("normal_paper_trade_kind_invalid")
    if contract.get("route_kind") not in NORMAL_PAPER_TRADE_ROUTES:
        reasons.append("normal_paper_trade_route_invalid")
    if str(contract.get("side") or "").lower() not in {"long", "short"}:
        reasons.append("normal_paper_trade_side_missing")
    if not str(contract.get("symbol") or "").strip():
        reasons.append("normal_paper_trade_symbol_missing")
    if contract.get("uses_shared_order_pipeline") is not True:
        reasons.append("normal_paper_trade_order_pipeline_split")
    if contract.get("uses_shared_position_ledger") is not True:
        reasons.append("normal_paper_trade_position_ledger_split")
    if contract.get("separate_sampling_order") is not False:
        reasons.append("normal_paper_trade_sampling_order_split")
    if contract.get("continuous_training_after_trusted_settlement") is not True:
        reasons.append("normal_paper_trade_training_disabled")
    if contract.get("order_creation_owner") != "ensemble_trader_unified_decision":
        reasons.append("normal_paper_trade_order_owner_invalid")
    if contract.get("risk_override_permission") is not False:
        reasons.append("normal_paper_trade_risk_override_invalid")
    if contract.get("sample_target") is not None or contract.get("daily_sample_quota") is not None:
        reasons.append("normal_paper_trade_sample_quota_forbidden")
    horizon = _float(contract.get("prediction_horizon_minutes"), 0.0)
    valid_for = _float(contract.get("valid_for_seconds"), 0.0)
    if horizon <= 0 or not isclose(valid_for, horizon * 60.0, abs_tol=1e-8):
        reasons.append("normal_paper_trade_horizon_invalid")
    expected_fingerprint = _fingerprint(
        {
            key: item
            for key, item in contract.items()
            if key not in {"generated_at", "contract_fingerprint"}
        }
    )
    if contract.get("contract_fingerprint") != expected_fingerprint:
        reasons.append("normal_paper_trade_fingerprint_mismatch")
    return list(dict.fromkeys(reasons))
