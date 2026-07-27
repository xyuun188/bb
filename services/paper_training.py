"""Read-only support for retired paper-training orders and positions.

This module may parse, validate, settle, and audit historical contracts. It must
not construct or authorize a new order.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from math import isclose, isfinite
from typing import Any

PAPER_TRAINING_VERSION = "2026-07-22.paper-training-bootstrap.v1"
PAPER_TRAINING_SIZING_VERSION = "2026-07-22.paper-training-sizing.v1"
PAPER_TRAINING_POSITION_LIFECYCLE_VERSION = "2026-07-22.paper-training-position-lifecycle.v1"
PAPER_TRAINING_ORDER_IDENTITY_VERSION = "2026-07-22.paper-training-order-identity.v1"
PAPER_TRAINING_CLIENT_ORDER_ID_PREFIX = "BBPT"
PAPER_TRAINING_MAX_SINGLE_TRADE_RISK_FRACTION = 0.0001
PAPER_TRAINING_MAX_PORTFOLIO_RISK_FRACTION = 0.0003
PAPER_TRAINING_MIN_FILL_DRIFT_RESERVE_FRACTION = 0.0025


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
            "execution_scope",
            "production_permission",
            "trade_kind",
            "trade_is_normal",
            "continuous_training_after_settlement",
            "loss_tolerant_for_training",
            "risk_profile",
            "single_trade_risk_fraction_cap",
            "portfolio_risk_fraction_cap",
            "separate_sampling_order",
            "purpose",
            "symbol",
            "selected_side",
            "signal_source",
            "expected_net_return_pct",
            "return_lcb_pct",
            "feature_opportunity_score",
            "prediction_horizon_minutes",
            "valid_for_seconds",
            "sample_target",
            "daily_sample_quota",
            "selection_reason",
            "policy_provenance",
        )
    }


def paper_training_decision_id_from_client_order_id(value: Any) -> int | None:
    """Recover a historical decision identity from an OKX client order id."""

    client_order_id = str(value or "").strip().upper()
    if not client_order_id.startswith(PAPER_TRAINING_CLIENT_ORDER_ID_PREFIX):
        return None
    raw_decision_id = client_order_id[len(PAPER_TRAINING_CLIENT_ORDER_ID_PREFIX) :]
    if not raw_decision_id.isdigit():
        return None
    decision_id = int(raw_decision_id)
    return decision_id if decision_id > 0 else None


def paper_training_contract_reasons(value: Any) -> list[str]:
    """Validate an immutable historical paper-training contract."""

    contract = _dict(value)
    reasons: list[str] = []
    if contract.get("version") != PAPER_TRAINING_VERSION:
        reasons.append("paper_training_version_invalid")
    if contract.get("authorized") is not True:
        reasons.append("paper_training_not_authorized")
    if contract.get("execution_scope") != "paper_only":
        reasons.append("paper_training_scope_invalid")
    if contract.get("production_permission") is not False:
        reasons.append("paper_training_production_permission_invalid")
    if contract.get("trade_is_normal") is not True:
        reasons.append("paper_training_normal_trade_contract_missing")
    if contract.get("trade_kind") != "normal_paper_training_trade":
        reasons.append("paper_training_trade_kind_invalid")
    if contract.get("continuous_training_after_settlement") is not True:
        reasons.append("paper_training_continuous_training_missing")
    if contract.get("loss_tolerant_for_training") is not True:
        reasons.append("paper_training_loss_tolerance_missing")
    if contract.get("risk_profile") != "cold_start_exploration":
        reasons.append("paper_training_risk_profile_invalid")
    if not isclose(
        _float(contract.get("single_trade_risk_fraction_cap"), 0.0) or 0.0,
        PAPER_TRAINING_MAX_SINGLE_TRADE_RISK_FRACTION,
        abs_tol=1e-12,
    ):
        reasons.append("paper_training_single_trade_risk_cap_invalid")
    if not isclose(
        _float(contract.get("portfolio_risk_fraction_cap"), 0.0) or 0.0,
        PAPER_TRAINING_MAX_PORTFOLIO_RISK_FRACTION,
        abs_tol=1e-12,
    ):
        reasons.append("paper_training_portfolio_risk_cap_invalid")
    if contract.get("separate_sampling_order") is not False:
        reasons.append("paper_training_separate_sampling_order_forbidden")
    if str(contract.get("selected_side") or "").lower() not in {"long", "short"}:
        reasons.append("paper_training_side_missing")
    if not str(contract.get("symbol") or "").strip():
        reasons.append("paper_training_symbol_missing")
    if contract.get("sample_target") is not None:
        reasons.append("paper_training_sample_quota_forbidden")
    if contract.get("daily_sample_quota") is not None:
        reasons.append("paper_training_daily_quota_forbidden")
    horizon_minutes = _float(contract.get("prediction_horizon_minutes"), 0.0) or 0.0
    valid_for_seconds = _float(contract.get("valid_for_seconds"), 0.0) or 0.0
    if horizon_minutes <= 0:
        reasons.append("paper_training_prediction_horizon_missing")
    if valid_for_seconds <= 0 or not isclose(
        valid_for_seconds,
        horizon_minutes * 60.0,
        rel_tol=1e-9,
        abs_tol=1e-8,
    ):
        reasons.append("paper_training_validity_contract_invalid")
    provenance = _dict(contract.get("policy_provenance"))
    for key in ("source", "observation_window", "generated_at", "strategy_version"):
        if not str(provenance.get(key) or "").strip():
            reasons.append("paper_training_provenance_incomplete")
            break
    if (_float(provenance.get("sample_count"), 0.0) or 0.0) <= 0:
        reasons.append("paper_training_provenance_sample_count_missing")
    if not isclose(
        _float(provenance.get("valid_for_seconds"), 0.0) or 0.0,
        valid_for_seconds,
        rel_tol=1e-9,
        abs_tol=1e-8,
    ):
        reasons.append("paper_training_provenance_validity_mismatch")
    if contract.get("contract_fingerprint") != _fingerprint(
        _contract_fingerprint_payload(contract)
    ):
        reasons.append("paper_training_contract_fingerprint_mismatch")
    return list(dict.fromkeys(reasons))


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _row_raw(row: Any) -> dict[str, Any]:
    return _dict(
        _row_value(row, "raw_llm_response")
        or _row_value(row, "raw_response")
        or _row_value(row, "decision_learning_snapshot")
    )


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _normalized_symbol(value: Any) -> str:
    return str(value or "").upper().replace("/", "").replace("-", "").replace(":USDT", "")


def build_paper_training_position_lifecycle(decision: Any) -> dict[str, Any]:
    """Recover a position lifecycle from one already-executed historical decision."""

    raw = _row_raw(decision)
    contract = _dict(raw.get("paper_training"))
    identity = _dict(raw.get("paper_training_order_identity"))
    identity_decision_id = paper_training_decision_id_from_client_order_id(
        identity.get("client_order_id")
    )
    decision_id = _row_value(decision, "id")
    if (
        not decision_id
        and identity.get("version") == PAPER_TRAINING_ORDER_IDENTITY_VERSION
        and identity.get("execution_scope") == "paper_only"
        and identity.get("production_permission") is False
        and identity_decision_id == _float(identity.get("decision_id"), None)
    ):
        decision_id = identity_decision_id
    action = str(_row_value(decision, "action") or "").lower()
    executed_at = _as_utc(_row_value(decision, "executed_at"))
    horizon_minutes = _float(contract.get("prediction_horizon_minutes"), 0.0) or 0.0
    if (
        paper_training_contract_reasons(contract)
        or not bool(_row_value(decision, "is_paper"))
        or not bool(_row_value(decision, "was_executed"))
        or action not in {"long", "short"}
        or action != str(contract.get("selected_side") or "").lower()
        or executed_at is None
        or horizon_minutes <= 0
    ):
        return {}
    expires_at = executed_at + timedelta(minutes=horizon_minutes)
    return {
        "version": PAPER_TRAINING_POSITION_LIFECYCLE_VERSION,
        "kind": "normal_paper_training_position",
        "authorized": True,
        "execution_scope": "paper_only",
        "production_permission": False,
        "decision_id": decision_id,
        "symbol": str(_row_value(decision, "symbol") or ""),
        "side": action,
        "executed_at": executed_at.isoformat(),
        "horizon_minutes": horizon_minutes,
        "expires_at": expires_at.isoformat(),
        "source_contract_version": PAPER_TRAINING_VERSION,
        "continuous_training_after_settlement": True,
        "loss_tolerant_for_training": True,
    }


def paper_training_position_lifecycle(position: dict[str, Any]) -> dict[str, Any]:
    direct = _dict(position.get("paper_training_lifecycle"))
    if direct:
        return direct
    management = _dict(position.get("current_management_contract"))
    return _dict(management.get("paper_training_lifecycle"))


def assess_paper_training_position_horizon(
    position: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Observe a historical horizon without granting position-exit authority."""

    lifecycle = paper_training_position_lifecycle(position)
    current = _as_utc(now) or datetime.now(UTC)
    try:
        expires_at = datetime.fromisoformat(
            str(lifecycle.get("expires_at") or "").replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        expires_at = None
    expires_at = _as_utc(expires_at)
    position_side = str(position.get("side") or "").lower()
    lifecycle_side = str(lifecycle.get("side") or "").lower()
    horizon_minutes = _float(lifecycle.get("horizon_minutes"), 0.0) or 0.0
    authorized = bool(
        lifecycle.get("version") == PAPER_TRAINING_POSITION_LIFECYCLE_VERSION
        and lifecycle.get("kind") == "normal_paper_training_position"
        and lifecycle.get("authorized") is True
        and lifecycle.get("execution_scope") == "paper_only"
        and lifecycle.get("production_permission") is False
        and lifecycle.get("continuous_training_after_settlement") is True
        and lifecycle.get("loss_tolerant_for_training") is True
        and str(position.get("execution_mode") or "").lower() == "paper"
        and position_side in {"long", "short"}
        and lifecycle_side == position_side
        and bool(_normalized_symbol(position.get("symbol")))
        and _normalized_symbol(position.get("symbol"))
        == _normalized_symbol(lifecycle.get("symbol"))
        and horizon_minutes > 0
        and expires_at is not None
    )
    return {
        "authorized": authorized,
        "elapsed": bool(authorized and current >= expires_at),
        "horizon_minutes": horizon_minutes,
        "expires_at": expires_at.isoformat() if expires_at is not None else None,
        "decision_id": lifecycle.get("decision_id"),
        "version": lifecycle.get("version"),
    }
