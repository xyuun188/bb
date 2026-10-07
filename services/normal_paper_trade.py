"""Single execution contract for every new OKX paper strategy trade."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from math import isclose, isfinite
from typing import Any

from ai_brain.base_model import DecisionOutput

NORMAL_PAPER_TRADE_VERSION = "2026-09-26.normal-paper-strategy-trade.v14"
NORMAL_PAPER_TRADE_SIZING_VERSION = "2026-10-07.normal-paper-dynamic-risk.v6"
HISTORICAL_NORMAL_PAPER_TRADE_SIZING_VERSIONS = frozenset(
    {"2026-08-25.normal-paper-dynamic-risk.v5"}
)
NORMAL_PAPER_ORDER_IDENTITY_VERSION = "2026-07-29.normal-paper-order-identity.v1"
NORMAL_PAPER_CLIENT_ORDER_ID_PREFIX = "BBNP"
HISTORICAL_NORMAL_PAPER_TRADE_VERSIONS = frozenset(
    {
        "2026-09-26.normal-paper-strategy-trade.v13",
        "2026-09-19.normal-paper-strategy-trade.v12",
        "2026-09-19.normal-paper-strategy-trade.v11",
        "2026-09-18.normal-paper-strategy-trade.v10",
        "2026-09-17.normal-paper-strategy-trade.v9",
        "2026-08-25.normal-paper-strategy-trade.v8",
        "2026-08-21.normal-paper-strategy-trade.v7",
        "2026-08-19.normal-paper-strategy-trade.v6",
        "2026-07-29.normal-paper-strategy-trade.v5",
        "2026-07-28.normal-paper-strategy-trade.v4",
        "2026-07-28.normal-paper-strategy-trade.v3",
        "2026-07-27.normal-paper-strategy-trade.v2",
        "2026-07-22.normal-paper-trade.v1",
    }
)
NORMAL_PAPER_TRADE_SELECTION_REASONS = {
    "strategy_edge_selected",
    "paper_quality_observation",
    "paper_training_entry",
}
# Quality observations are current paper-training entries under the same v14
# contract and risk/order pipeline. They can collect fresh settlement evidence
# when current expected net return is positive, but never grant production/live
# permission and never bypass the current cost, loss-probability, or sizing
# contracts.
NORMAL_PAPER_TRADE_NEW_ENTRY_SELECTION_REASONS = frozenset(
    {"strategy_edge_selected", "paper_quality_observation", "paper_training_entry"}
)
NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION = 0.005
NORMAL_PAPER_TRADE_MAX_QUALITY_OBSERVATION_LOSS_PROBABILITY = 0.60
NORMAL_PAPER_TRADE_DIRECTION_CONCENTRATION_ALERT_THRESHOLD = 0.80
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


def normal_paper_client_order_id(decision_id: Any) -> str:
    """Return the stable OKX client id for one persisted normal-paper decision."""

    try:
        normalized_id = int(decision_id or 0)
    except (TypeError, ValueError):
        return ""
    if normalized_id <= 0:
        return ""
    return f"{NORMAL_PAPER_CLIENT_ORDER_ID_PREFIX}{normalized_id}"


def normal_paper_decision_id_from_client_order_id(value: Any) -> int | None:
    """Recover a normal-paper decision id from an OKX client order id."""

    client_order_id = str(value or "").strip().upper()
    if not client_order_id.startswith(NORMAL_PAPER_CLIENT_ORDER_ID_PREFIX):
        return None
    raw_decision_id = client_order_id[len(NORMAL_PAPER_CLIENT_ORDER_ID_PREFIX) :]
    if not raw_decision_id.isdigit():
        return None
    decision_id = int(raw_decision_id)
    return decision_id if decision_id > 0 else None


def normalize_normal_paper_contract(value: Any) -> dict[str, Any]:
    """Map current or historical envelopes to one analysis protocol.

    The returned object is an analysis/training projection. It is never an
    authorization envelope for a new order. Raw historical fields remain
    available under ``source_*`` keys so settlement and training stay
    auditable while all downstream calculations consume one normalized shape.
    """

    contract = _dict(value)
    version = str(contract.get("version") or "").strip()
    if not version:
        return {}
    if version == NORMAL_PAPER_TRADE_VERSION:
        normalized = dict(contract)
        normalized.update(
            {
                "canonical_protocol_version": NORMAL_PAPER_TRADE_VERSION,
                "protocol_state": "current",
                "source_version": version,
                "production_permission": False,
            }
        )
        return normalized
    if version not in HISTORICAL_NORMAL_PAPER_TRADE_VERSIONS:
        return {}

    expected = _float(
        contract.get("expected_net_return_pct"),
        _float(contract.get("expected_return_pct"), 0.0),
    )
    objective = _float(
        contract.get("objective_net_return_pct"),
        _float(contract.get("return_lcb_pct"), 0.0),
    )
    loss_probability = _float(
        contract.get("loss_probability"),
        _float(contract.get("server_profit_loss_probability"), 1.0),
    )
    route_kind = str(
        contract.get("selection_reason")
        or contract.get("route_kind")
        or "historical_normalized"
    ).strip()
    observation = bool(
        contract.get("paper_quality_observation_only") is True
        or route_kind == "paper_quality_observation"
        or (objective is not None and objective <= 0.0)
    )
    source_selection_reason = str(contract.get("selection_reason") or "").strip()
    selection_reason = (
        source_selection_reason
        if source_selection_reason in NORMAL_PAPER_TRADE_SELECTION_REASONS
        else "paper_quality_observation"
        if observation
        else "strategy_edge_selected"
    )
    normalized = {
        "canonical_protocol_version": NORMAL_PAPER_TRADE_VERSION,
        "protocol_state": "historical_normalized",
        "source_version": version,
        "source_contract": deepcopy(contract),
        "version": NORMAL_PAPER_TRADE_VERSION,
        "authorized": False,
        "trade_mode": "paper",
        "execution_scope": "paper_only",
        "entry_type": "normal_strategy_trade",
        "trade_kind": "normal_strategy_trade",
        "production_permission": False,
        "decision_authority": str(
            contract.get("decision_authority")
            or (
                "ensemble"
                if contract.get("order_creation_owner")
                == "ensemble_trader_unified_decision"
                else "historical"
            )
        ),
        "selection_reason": selection_reason,
        "historical_selection_reason": route_kind,
        "symbol": str(contract.get("symbol") or ""),
        "side": str(contract.get("side") or "").lower(),
        "prediction_horizon_minutes": _float(
            contract.get("prediction_horizon_minutes"),
            0.0,
        ),
        "valid_for_seconds": _float(
            contract.get("valid_for_seconds"),
            0.0,
        ),
        "expected_net_return_pct": expected,
        "objective_net_return_pct": objective,
        "loss_probability": loss_probability,
        "quant_evidence_families": list(
            contract.get("quant_evidence_families") or []
        ),
        "strong_expert_opposition": bool(
            contract.get("strong_expert_opposition") is True
        ),
        # Historical caps are preserved only inside source_contract. The
        # canonical projection must not carry old micro-sizing into new logic.
        "single_trade_risk_fraction_cap": NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION,
        "paper_quality_mode": (
            "quality_observation" if observation else "validated"
        ),
        "paper_quality_observation_only": observation,
        "quality_observation_reasons": list(
            contract.get("quality_observation_reasons")
            or contract.get("paper_quality_observation_reasons")
            or []
        ),
        "risk_override_permission": False,
        "uses_shared_order_pipeline": True,
        "uses_shared_position_ledger": True,
        "continuous_training_after_trusted_settlement": True,
        "migration_status": "normalized_historical",
    }
    return normalized


def historical_normalized_contract_reasons(value: Any) -> list[str]:
    """Validate the normalized shape used by historical analysis/settlement."""

    normalized = normalize_normal_paper_contract(value)
    if not normalized:
        return ["normal_paper_trade_contract_not_normalizable"]
    if normalized.get("protocol_state") != "historical_normalized":
        return ["normal_paper_trade_historical_state_invalid"]
    reasons: list[str] = []
    if not normalized.get("symbol"):
        reasons.append("normal_paper_trade_symbol_missing")
    if normalized.get("side") not in {"long", "short"}:
        reasons.append("normal_paper_trade_side_missing")
    if normalized.get("execution_scope") != "paper_only":
        reasons.append("normal_paper_trade_scope_invalid")
    return list(dict.fromkeys(reasons))


def attach_normal_paper_order_identity(
    decision: DecisionOutput,
    *,
    model_mode: str,
    decision_id: Any,
) -> dict[str, Any]:
    """Attach an exchange-recoverable identity after the decision row exists."""

    if str(model_mode or "").lower() != "paper" or not decision.is_entry:
        return {}
    raw = dict(_dict(decision.raw_response))
    contract = _dict(raw.get("normal_paper_trade"))
    if normal_paper_trade_contract_reasons(contract):
        return {}
    client_order_id = normal_paper_client_order_id(decision_id)
    if not client_order_id:
        return {}
    identity = {
        "version": NORMAL_PAPER_ORDER_IDENTITY_VERSION,
        "decision_id": int(decision_id),
        "client_order_id": client_order_id,
        "execution_scope": "paper_only",
        "entry_type": "normal_strategy_trade",
        "production_permission": False,
        "normal_trade_contract_fingerprint": contract.get("contract_fingerprint"),
    }
    raw["normal_paper_order_identity"] = identity
    decision.raw_response = raw
    return identity


def normal_paper_order_identity_reasons(
    value: Any,
    *,
    decision_id: Any,
    contract: Any,
) -> list[str]:
    """Validate the identity against its exact decision and strategy contract."""

    identity = _dict(value)
    normal_contract = _dict(contract)
    expected_client_id = normal_paper_client_order_id(decision_id)
    reasons: list[str] = []
    if identity.get("version") != NORMAL_PAPER_ORDER_IDENTITY_VERSION:
        reasons.append("normal_paper_order_identity_version_invalid")
    try:
        identity_decision_id = int(identity.get("decision_id") or 0)
        expected_decision_id = int(decision_id or 0)
    except (TypeError, ValueError):
        identity_decision_id = 0
        expected_decision_id = 0
    if expected_decision_id <= 0 or identity_decision_id != expected_decision_id:
        reasons.append("normal_paper_order_identity_decision_mismatch")
    if not expected_client_id or identity.get("client_order_id") != expected_client_id:
        reasons.append("normal_paper_order_identity_client_id_invalid")
    if identity.get("execution_scope") != "paper_only":
        reasons.append("normal_paper_order_identity_scope_invalid")
    if identity.get("entry_type") != "normal_strategy_trade":
        reasons.append("normal_paper_order_identity_entry_type_invalid")
    if identity.get("production_permission") is not False:
        reasons.append("normal_paper_order_identity_production_permission_invalid")
    if normal_paper_settlement_contract_reasons(normal_contract):
        reasons.append("normal_paper_order_identity_trade_contract_invalid")
    if identity.get("normal_trade_contract_fingerprint") != normal_contract.get(
        "contract_fingerprint"
    ):
        reasons.append("normal_paper_order_identity_contract_mismatch")
    return list(dict.fromkeys(reasons))


def _contract_fingerprint_payload(contract: dict[str, Any]) -> dict[str, Any]:
    payload = {
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
            "current_raw_expected_return_pct",
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
            "risk_override_permission",
            "quant_quality_permissions",
            "paper_quality_mode",
            "paper_quality_observation_only",
            "quality_observation_reasons",
            "paper_training_only",
            "paper_training_reasons",
        )
    }
    return payload


def select_normal_paper_trade_side(
    support_by_side: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Select one current paper direction under the canonical v14 contract.

    A validated current edge is preferred. When no validated edge exists, a
    separate paper-training entry may be selected only from complete current
    directional evidence; it is marked shadow-only and cannot authorize live
    execution or production promotion.
    """

    by_side = {
        side: dict(_dict(_dict(support_by_side).get(side)))
        for side in ("long", "short")
    }
    for support in by_side.values():
        reasons = [
            str(reason)
            for reason in support.get("blocking_reasons") or []
            if str(reason).strip()
        ]
        if support.get("eligible") is not True:
            reason = str(support.get("reason") or "").strip()
            if reason:
                reasons.append(reason)
        expected_net = _float(support.get("expected_net_return_pct"), None)
        if expected_net is None or expected_net <= 0.0:
            reasons.append("direction_support_expected_net_not_positive")
        objective_net = _float(support.get("objective_net_return_pct"), None)
        if objective_net is None:
            reasons.append("direction_support_objective_net_missing")
        elif objective_net <= 0.0:
            reasons.append("direction_support_objective_net_not_positive")
        if not support.get("quant_evidence_families"):
            reasons.append("direction_support_quant_evidence_missing")
        if support.get("execution_cost_complete") is False:
            reasons.append("direction_support_execution_cost_incomplete")
        horizon = _float(support.get("prediction_horizon_minutes"), None)
        if horizon is None or horizon <= 0.0:
            reasons.append("direction_support_prediction_horizon_missing")
        if support.get("strong_expert_opposition") is True:
            reasons.append("direction_support_strong_expert_opposition")
        support["blocking_reasons"] = list(dict.fromkeys(reasons))
    candidates: list[dict[str, Any]] = []
    for side, support in by_side.items():
        if support.get("eligible") is not True:
            continue
        expected_net = _float(support.get("expected_net_return_pct"), None)
        objective_net = _float(support.get("objective_net_return_pct"), None)
        loss_probability = _float(support.get("loss_probability"), 1.0) or 1.0
        current_edge = bool(
            support.get("current_edge_validated") is True
            and objective_net is not None
            and objective_net > 0.0
        )
        families = sorted(
            {
                str(item).strip()
                for item in support.get("quant_evidence_families") or []
                if str(item).strip()
            }
        )
        if current_edge and expected_net is not None and expected_net > 0.0:
            candidates.append(
                {
                    "side": side,
                    "support": support,
                    "expected_net_return_pct": expected_net,
                    "objective_net_return_pct": objective_net,
                    "loss_probability": loss_probability,
                    "quant_evidence_families": families,
                    "selection_reason": "strategy_edge_selected",
                }
            )

    if not candidates:
        training_candidates: list[dict[str, Any]] = []
        for side, support in by_side.items():
            if support.get("eligible") is not True:
                continue
            if support.get("paper_training_only") is not True:
                continue
            expected_net = _float(support.get("expected_net_return_pct"), None)
            objective_net = _float(support.get("objective_net_return_pct"), None)
            loss_probability = _float(support.get("loss_probability"), 1.0) or 1.0
            if (
                expected_net is None
                or objective_net is None
                or loss_probability > NORMAL_PAPER_TRADE_MAX_QUALITY_OBSERVATION_LOSS_PROBABILITY
            ):
                continue
            families = sorted(
                {
                    str(item).strip()
                    for item in support.get("quant_evidence_families") or []
                    if str(item).strip()
                }
            )
            if not families:
                continue
            training_candidates.append(
                {
                    "side": side,
                    "support": support,
                    "expected_net_return_pct": expected_net,
                    "objective_net_return_pct": objective_net,
                    "loss_probability": loss_probability,
                    "quant_evidence_families": families,
                    "selection_reason": "paper_training_entry",
                }
            )
        candidates.extend(training_candidates)

    candidates.sort(
        key=lambda item: (
            item["selection_reason"] == "strategy_edge_selected",
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

    blocking_reasons = list(
        dict.fromkeys(
            str(reason)
            for support in by_side.values()
            for reason in support.get("blocking_reasons") or []
            if str(reason).strip()
        )
    )
    # Keep the old selection_reason for persisted-contract compatibility, but
    # expose a single diagnostic vocabulary so callers do not have to infer a
    # profit gate from the ambiguous ``no_direction`` value.
    quality_gate_reasons = {
        "direction_support_expected_net_not_positive",
        "direction_support_objective_net_not_positive",
        "direction_support_objective_net_missing",
        "direction_support_quality_observation_loss_probability_too_high",
        "direction_support_quant_family_conflict",
        "direction_support_strong_expert_opposition",
    }
    evidence_reasons = {
        "direction_support_execution_cost_incomplete",
        "direction_support_quant_evidence_missing",
        "direction_support_prediction_horizon_missing",
    }
    if selected:
        diagnostic_status = "selected"
        blocking_category = "none"
    elif blocking_reasons and set(blocking_reasons).issubset(quality_gate_reasons):
        diagnostic_status = "profit_gate_blocked"
        blocking_category = "profit_gate"
    elif blocking_reasons and set(blocking_reasons) & evidence_reasons:
        diagnostic_status = "evidence_blocked"
        blocking_category = "evidence"
    elif blocking_reasons:
        diagnostic_status = "direction_blocked"
        blocking_category = "direction"
    else:
        diagnostic_status = "no_candidate"
        blocking_category = "unknown"

    return {
        "version": NORMAL_PAPER_TRADE_VERSION,
        "selected": bool(selected),
        "selected_side": selected["side"] if selected else "neutral",
        "selection_reason": selected["selection_reason"] if selected else "no_direction",
        "selected_support": dict(selected["support"]) if selected else {},
        "eligible_side_count": len(candidates),
        "by_side": by_side,
        "blocking_reasons": blocking_reasons,
        "blocking_reasons_by_side": {
            side: list(
                dict.fromkeys(
                    str(reason)
                    for reason in support.get("blocking_reasons") or []
                    if str(reason).strip()
                )
            )
            for side, support in by_side.items()
        },
        "blocking_category": blocking_category,
        "diagnostic_status": diagnostic_status,
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
    expected_net = _float(support.get("expected_net_return_pct"), None)
    objective_net = _float(support.get("objective_net_return_pct"), None)
    quality_permissions = {
        str(source): deepcopy(permission)
        for source, permission in _dict(
            support.get("quant_quality_permissions")
        ).items()
        if str(source).strip() and isinstance(permission, dict)
    }
    current_edge_signal = support.get("current_edge_validated")
    if current_edge_signal is None:
        current_edge_signal = not bool(support.get("paper_quality_observation_only"))
    current_edge_validated = bool(
        current_edge_signal is True
        and expected_net is not None
        and expected_net > 0.0
        and objective_net is not None
        and objective_net > 0.0
        and support.get("strong_expert_opposition") is not True
    )
    quality_observation_only = bool(
        not current_edge_validated
        and (
            support.get("paper_quality_observation_only") is True
            or any(
                permission.get("paper_execution_permission") is not True
                for permission in quality_permissions.values()
            )
        )
    )
    if current_edge_validated:
        quality_observation_only = False
    paper_training_only = selection_reason == "paper_training_entry"
    if paper_training_only:
        # Training entries are explicit current-evidence observations, not the
        # older unauthorized quality-observation envelope.
        quality_observation_only = False
    current_raw_expected = _float(support.get("raw_expected_return_pct"), None)
    paper_training_reasons = sorted(
        {
            str(reason)
            for reason in (support.get("paper_training_reasons") or [])
            if str(reason).strip()
        }
    )
    if paper_training_only and not paper_training_reasons:
        paper_training_reasons = [
            "historical_quality_gate_is_observation_only",
            "current_directional_evidence_is_training_eligible",
        ]
    quality_observation_reasons = sorted(
        {
            str(reason)
            for reason in (support.get("paper_quality_observation_reasons") or [])
            if str(reason).strip()
        }
    )
    loss_probability = _float(support.get("loss_probability"), None)
    if (
        normalized_side not in {"long", "short"}
        or selection_reason not in NORMAL_PAPER_TRADE_SELECTION_REASONS
        or support.get("eligible") is not True
        or support.get("selected_side") != normalized_side
        or horizon <= 0.0
        or expected_net is None
        or (expected_net <= 0.0 and not paper_training_only)
        or (paper_training_only and (current_raw_expected is None or current_raw_expected <= 0.0))
        or objective_net is None
        or (
            objective_net <= 0.0
            and selection_reason not in {"paper_quality_observation", "paper_training_entry"}
        )
        or not quality_permissions
        or (
            selection_reason == "strategy_edge_selected"
            and not current_edge_validated
            and any(
                permission.get("paper_execution_permission") is not True
                for permission in quality_permissions.values()
            )
        )
        or (
            selection_reason == "paper_quality_observation"
            and (
                not quality_observation_only
                or not quality_observation_reasons
                or loss_probability is None
                or loss_probability
                > NORMAL_PAPER_TRADE_MAX_QUALITY_OBSERVATION_LOSS_PROBABILITY
            )
        )
        or (
            paper_training_only
            and (
                not paper_training_reasons
                or loss_probability is None
                or loss_probability > NORMAL_PAPER_TRADE_MAX_QUALITY_OBSERVATION_LOSS_PROBABILITY
                or support.get("paper_training_only") is not True
                or int(support.get("aligned_expert_count") or 0) < 2
                or int(support.get("aligned_expert_count") or 0)
                <= int(support.get("opposition_expert_count") or 0)
            )
        )
    ):
        return {}

    is_new_entry_authorized = bool(
        selection_reason in NORMAL_PAPER_TRADE_NEW_ENTRY_SELECTION_REASONS
    )
    contract = {
        "version": NORMAL_PAPER_TRADE_VERSION,
        # Quality observations are evidence for shadow settlement/training only.
        # They are deliberately persisted as unauthorized so retries cannot
        # turn them into a normal paper order.
        "authorized": is_new_entry_authorized,
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
        "expected_net_return_pct": expected_net,
        "current_raw_expected_return_pct": current_raw_expected,
        "objective_net_return_pct": objective_net,
        "loss_probability": loss_probability,
        "quant_evidence_families": list(support.get("quant_evidence_families") or []),
        "strong_expert_opposition": bool(support.get("strong_expert_opposition") is True),
        "single_trade_risk_fraction_cap": NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION,
        "leverage_policy": NORMAL_PAPER_TRADE_LEVERAGE_POLICY,
        "model_leverage_role": "upper_bound_when_explicit",
        "uses_shared_order_pipeline": True,
        "uses_shared_position_ledger": True,
        "continuous_training_after_trusted_settlement": True,
        "training_eligibility_source": ("trusted_settlement_and_task_specific_training_contract"),
        "risk_override_permission": False,
        "quant_quality_permissions": quality_permissions,
        "current_edge_validated": current_edge_validated,
        "paper_quality_mode": (
            "training" if paper_training_only
            else "quality_observation" if quality_observation_only else "validated"
        ),
        "paper_quality_observation_only": quality_observation_only,
        "quality_observation_reasons": quality_observation_reasons,
        "paper_training_only": paper_training_only,
        "paper_training_reasons": paper_training_reasons,
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

    # A recognized historical envelope may settle an old fill, but it must not
    # survive into the authorization path for a new submission.
    raw.pop("normal_paper_trade", None)
    decision.raw_response = raw

    selection = _dict(raw.get("paper_trade_selection"))
    # Quality observations are shadow/training evidence only. They must not
    # be re-materialized as a normal order by a retry or recovery path.
    if (
        str(selection.get("selection_reason") or "").strip()
        not in NORMAL_PAPER_TRADE_NEW_ENTRY_SELECTION_REASONS
    ):
        raw.pop("normal_paper_trade", None)
        decision.raw_response = raw
        return {}
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


def _normal_strategy_trade_contract_reasons(
    value: Any,
    *,
    require_positive_objective: bool,
    require_quality_permission: bool = True,
    allow_non_positive_objective_observation: bool = False,
    allow_unauthorized_observation: bool = False,
) -> list[str]:
    contract = _dict(value)
    reasons: list[str] = []
    if contract.get("version") != NORMAL_PAPER_TRADE_VERSION:
        reasons.append("normal_paper_trade_version_invalid")
    selection_reason = str(contract.get("selection_reason") or "")
    observation_mode = selection_reason == "paper_quality_observation"
    training_mode = selection_reason == "paper_training_entry"
    expected_net = _float(contract.get("expected_net_return_pct"), None)
    objective_net = _float(contract.get("objective_net_return_pct"), None)
    current_edge_validated = bool(
        contract.get("current_edge_validated") is True
        or (
            selection_reason == "strategy_edge_selected"
            and expected_net is not None
            and expected_net > 0.0
            and objective_net is not None
            and objective_net > 0.0
        )
    )
    observation_only = bool(allow_unauthorized_observation and observation_mode)
    if contract.get("authorized") is not True and not observation_only:
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
    if selection_reason not in NORMAL_PAPER_TRADE_SELECTION_REASONS:
        reasons.append("normal_paper_trade_selection_reason_invalid")
    if (
        require_positive_objective
        and selection_reason not in NORMAL_PAPER_TRADE_NEW_ENTRY_SELECTION_REASONS
    ):
        reasons.append("normal_paper_trade_quality_observation_shadow_only")
    if str(contract.get("side") or "").lower() not in {"long", "short"}:
        reasons.append("normal_paper_trade_side_missing")
    if not str(contract.get("symbol") or "").strip():
        reasons.append("normal_paper_trade_symbol_missing")
    if contract.get("strong_expert_opposition") is True:
        reasons.append("normal_paper_trade_strong_expert_opposition")
    if expected_net is None or (expected_net <= 0.0 and not training_mode):
        reasons.append("normal_paper_trade_expected_net_not_positive")
    raw_expected = _float(contract.get("current_raw_expected_return_pct"), None)
    if training_mode and (raw_expected is None or raw_expected <= 0.0):
        reasons.append("normal_paper_trade_training_raw_return_not_positive")
    if objective_net is None:
        reasons.append("normal_paper_trade_objective_net_missing")
    elif (
        require_positive_objective
        and objective_net <= 0.0
        and not (
            (observation_mode or training_mode)
            and allow_non_positive_objective_observation
        )
    ):
        reasons.append("normal_paper_trade_objective_net_not_positive")
    if contract.get("uses_shared_order_pipeline") is not True:
        reasons.append("normal_paper_trade_order_pipeline_split")
    if contract.get("uses_shared_position_ledger") is not True:
        reasons.append("normal_paper_trade_position_ledger_split")
    if contract.get("continuous_training_after_trusted_settlement") is not True:
        reasons.append("normal_paper_trade_training_disabled")
    if contract.get("risk_override_permission") is not False:
        reasons.append("normal_paper_trade_risk_override_invalid")
    observation_reasons = [
        str(reason)
        for reason in contract.get("quality_observation_reasons") or []
        if str(reason).strip()
    ]
    if contract.get("paper_quality_observation_only") is not observation_mode:
        reasons.append("normal_paper_trade_quality_mode_invalid")
    expected_quality_mode = (
        "training" if training_mode else "quality_observation" if observation_mode else "validated"
    )
    if contract.get("paper_quality_mode") != expected_quality_mode:
        reasons.append("normal_paper_trade_quality_mode_invalid")
    if observation_mode and not observation_reasons:
        reasons.append("normal_paper_trade_quality_observation_reason_missing")
    if contract.get("paper_training_only") is not training_mode:
        reasons.append("normal_paper_trade_training_scope_invalid")
    training_reasons = [str(reason) for reason in contract.get("paper_training_reasons") or [] if str(reason).strip()]
    if training_mode and not training_reasons:
        reasons.append("normal_paper_trade_training_reason_missing")
    loss_probability = _float(contract.get("loss_probability"), None)
    if (
        (observation_mode or training_mode)
        and (
            loss_probability is None
            or loss_probability
            > NORMAL_PAPER_TRADE_MAX_QUALITY_OBSERVATION_LOSS_PROBABILITY
        )
    ):
        reasons.append(
            "normal_paper_trade_quality_observation_loss_probability_too_high"
        )
    if require_quality_permission:
        quality_permissions = _dict(contract.get("quant_quality_permissions"))
        if not quality_permissions:
            reasons.append("normal_paper_trade_quality_permission_missing")
        for source, permission in quality_permissions.items():
            if not str(source).strip() or not isinstance(permission, dict):
                reasons.append("normal_paper_trade_quality_permission_invalid")
                continue
            if (
                not observation_mode and not training_mode
                and not current_edge_validated
                and permission.get("paper_execution_permission") is not True
            ):
                reasons.append("normal_paper_trade_quality_permission_denied")
            evidence = _dict(permission.get("paper_execution_evidence"))
            if (
                not observation_mode and not training_mode
                and not current_edge_validated
                and int(_float(evidence.get("sample_count"), 0.0) or 0) <= 0
            ):
                reasons.append("normal_paper_trade_quality_evidence_missing")
    horizon = _float(contract.get("prediction_horizon_minutes"), 0.0) or 0.0
    valid_for = _float(contract.get("valid_for_seconds"), 0.0) or 0.0
    if horizon <= 0.0 or not isclose(valid_for, horizon * 60.0, abs_tol=1e-8):
        reasons.append("normal_paper_trade_horizon_invalid")
    single_cap = _float(contract.get("single_trade_risk_fraction_cap"), 0.0) or 0.0
    if not isclose(
        single_cap,
        NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION,
        abs_tol=1e-12,
    ):
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


def normal_paper_trade_contract_reasons(value: Any) -> list[str]:
    """Validate the only contract allowed to authorize a new paper entry."""

    return _normal_strategy_trade_contract_reasons(
        value,
        require_positive_objective=True,
        require_quality_permission=True,
        allow_non_positive_objective_observation=True,
    )


def normal_paper_trade_observation_contract_reasons(value: Any) -> list[str]:
    """Validate a current observation envelope for audit and training."""

    return _normal_strategy_trade_contract_reasons(
        value,
        require_positive_objective=False,
        require_quality_permission=True,
        allow_non_positive_objective_observation=True,
        allow_unauthorized_observation=True,
    )


def normal_paper_settlement_contract_reasons(value: Any) -> list[str]:
    """Validate one current or normalized historical contract for recovery.

    Historical envelopes are converted to the canonical analysis shape first.
    They can support settlement/protection of an already-open position, but the
    normalized projection is never accepted by the new-entry authorization
    validator.
    """

    contract = _dict(value)
    version = contract.get("version")
    if version == NORMAL_PAPER_TRADE_VERSION:
        if contract.get("selection_reason") == "paper_quality_observation":
            return normal_paper_trade_observation_contract_reasons(contract)
        return normal_paper_trade_contract_reasons(contract)
    return historical_normalized_contract_reasons(contract)
