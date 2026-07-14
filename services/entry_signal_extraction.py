"""Shared extraction of entry-side model signals from decision payloads."""

from __future__ import annotations

from typing import Any

from services.profit_supervision import PROFIT_SUPERVISION_VERSION
from services.return_objective import (
    RETURN_LABEL_NAME,
    RETURN_LABEL_VERSION,
    RETURN_OBJECTIVE_NAME,
    RETURN_OBJECTIVE_VERSION,
    validate_return_distribution_contract,
)

LEGACY_MOJIBAKE_LONG_LABELS = ("\u934b\u6c2c\ue63f",)
LEGACY_MOJIBAKE_SHORT_LABELS = ("\u934b\u6c31\u2516",)


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


_WRAPPED_TOOL_KEYS = ("data", "result", "prediction", "payload", "output")
_WRAPPER_METADATA_KEYS = (
    "available",
    "enabled",
    "ok",
    "trained",
    "status",
    "model",
    "primary_model",
    "challenger_model",
    "model_version",
    "route_mode",
    "live_mutation",
    "live_influence",
    "influence_enabled",
    "allow_live_position_influence",
    "promotion_ready",
    "readiness",
    "evaluation_policy",
    "model_stage",
    "training_mode",
    "objective",
    "objective_name",
    "objective_version",
    "artifact_objective",
    "artifact_objective_version",
    "artifact_persisted",
    "training_cost_policy",
    "label_name",
    "label_version",
    "profit_supervision_version",
    "return_semantics",
    "return_distribution_contract",
    "return_distribution_contract_version",
    "counterfactual_execution_cost_distribution",
    "actual_trade_calibration",
    "prediction_quality",
    "fallback_reason",
    "feature_coverage",
    "backend",
    "endpoint",
    "path",
    "duration_sec",
    "latency_ms",
)


def unwrap_tool_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    payload = dict(value)
    for key in _WRAPPED_TOOL_KEYS:
        child = payload.get(key)
        if not isinstance(child, dict):
            continue
        inner = unwrap_tool_payload(child)
        if not inner:
            continue
        for meta_key in _WRAPPER_METADATA_KEYS:
            if meta_key in payload and meta_key not in inner:
                inner[meta_key] = payload[meta_key]
        return inner
    return payload


def enrich_signal_payload(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a tool payload into the common side/expected-return contract."""

    if not isinstance(payload, dict) or not payload:
        return {}
    normalized = unwrap_tool_payload(payload) or dict(payload)
    if "available" not in normalized and "ok" in normalized:
        normalized["available"] = bool(normalized.get("ok"))
    if name == "profit_prediction":
        side = payload_side(normalized)
        if side in {"long", "short"}:
            normalized["best_side"] = side
            normalized["side"] = side
    elif name == "time_series_prediction":
        side = payload_side(normalized)
        if side in {"long", "short"}:
            normalized["best_side"] = side
            normalized["side"] = side
    elif name == "sentiment_analysis":
        side = payload_side(normalized)
        if side in {"long", "short"}:
            normalized["best_side"] = side
            normalized["side"] = side
        for legacy_return_field in (
            "expected_return_pct",
            "expected_return_from_sentiment_pct",
            "long_expected_return_pct",
            "short_expected_return_pct",
            "adjusted_expected_return_pct",
        ):
            normalized.pop(legacy_return_field, None)
    return normalized


def first_tool_payload(raw: dict[str, Any], *keys: str) -> dict[str, Any]:
    source_containers = (
        raw.get("local_ai_tools"),
        raw.get("server_quant_tools"),
        raw.get("quant_tools"),
        raw.get("local_tools"),
        raw.get("server_tools"),
        raw,
    )
    containers: list[dict[str, Any]] = []
    for container in source_containers:
        if not isinstance(container, dict):
            continue
        containers.append(container)
        unwrapped = unwrap_tool_payload(container)
        if unwrapped and unwrapped != container:
            containers.append(unwrapped)
    for container in containers:
        for key in keys:
            value = container.get(key)
            payload = unwrap_tool_payload(value)
            if payload:
                return enrich_signal_payload(key, payload)
    return {}


def signal_available(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    if payload.get("error") or payload.get("exception"):
        return False
    status = str(payload.get("status") or "").lower()
    if status in {"unavailable", "error", "disabled", "circuit_open", "failed"}:
        return False
    for key in ("available", "enabled", "ok"):
        if key in payload and payload.get(key) is False:
            return False
    return True


_NON_PRODUCTION_ROUTE_MARKERS = (
    "shadow",
    "candidate",
    "observation",
    "inference_only",
    "learning_only",
    "promotion_blocked",
)
_NON_PRODUCTION_STAGES = {
    "shadow",
    "shadow_evaluating",
    "candidate",
    "inference_only",
    "learning_only",
    "promotion_blocked",
}


def _signal_governance_nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    current = payload
    seen: set[int] = set()
    while isinstance(current, dict) and id(current) not in seen:
        seen.add(id(current))
        nodes.append(current)
        child = next(
            (
                current.get(key)
                for key in _WRAPPED_TOOL_KEYS
                if isinstance(current.get(key), dict)
            ),
            None,
        )
        if not isinstance(child, dict):
            break
        current = child
    return nodes


def signal_return_distribution(
    payload: dict[str, Any] | None,
    side: str,
) -> dict[str, Any]:
    """Return one side's standardized distribution without legacy aliases."""

    if not isinstance(payload, dict) or side not in {"long", "short"}:
        return {}
    for node in _signal_governance_nodes(payload):
        container = safe_dict(node.get("return_distribution_contract"))
        contract = safe_dict(container.get(side))
        if contract:
            return contract
    return {}


def signal_return_distribution_eligibility(
    payload: dict[str, Any] | None,
    side: str,
) -> dict[str, Any]:
    """Validate the full production distribution contract at the app boundary."""

    contract = signal_return_distribution(payload, side)
    if not contract:
        return {
            "eligible": False,
            "reason": "return_distribution_contract_missing",
            "side": side,
        }

    return validate_return_distribution_contract(
        contract,
        side=side,
        return_semantics="gross_market_opportunity_before_execution",
        profit_supervision_version=PROFIT_SUPERVISION_VERSION,
    )


def _signal_contract_side(payload: dict[str, Any]) -> str:
    side = payload_side(payload)
    if side in {"long", "short"}:
        return side
    predictions = safe_list(payload.get("predictions"))
    primary = safe_dict(predictions[0] if predictions else {})
    return payload_side(primary)


def signal_production_eligibility(payload: dict[str, Any]) -> dict[str, Any]:
    """Return auditable live-decision eligibility without hiding shadow observations."""

    if not signal_available(payload):
        return {"eligible": False, "reason": "signal_unavailable"}

    governance_seen = False
    route_live = False
    live_influence_allowed = False
    promotion_ready = False
    quality_approved = False
    objective_approved = False
    objective_version_approved = False
    label_approved = False
    label_version_approved = False
    cost_policy_approved = False
    supervision_approved = False
    return_semantics_approved = False
    for node in _signal_governance_nodes(payload):
        route_mode = str(node.get("route_mode") or "").strip().lower()
        if route_mode:
            governance_seen = True
            if any(marker in route_mode for marker in _NON_PRODUCTION_ROUTE_MARKERS):
                return {
                    "eligible": False,
                    "reason": "non_production_route_mode",
                    "route_mode": route_mode,
                }
            route_live = route_mode == "live"

        stage = str(node.get("model_stage") or node.get("training_mode") or "").strip().lower()
        if stage:
            governance_seen = True
            if stage in _NON_PRODUCTION_STAGES:
                return {
                    "eligible": False,
                    "reason": "non_production_model_stage",
                    "model_stage": stage,
                }

        for key in (
            "live_mutation",
            "live_influence",
            "influence_enabled",
            "allow_live_position_influence",
        ):
            if key in node:
                governance_seen = True
                if node.get(key) is False:
                    return {"eligible": False, "reason": f"{key}_disabled"}
                if node.get(key) is True:
                    live_influence_allowed = True

        if "promotion_ready" in node:
            governance_seen = True
            if node.get("promotion_ready") is False:
                return {"eligible": False, "reason": "promotion_not_ready"}
            promotion_ready = node.get("promotion_ready") is True

        readiness = safe_dict(node.get("readiness"))
        if readiness:
            governance_seen = True
            if readiness.get("allow_live_position_influence") is False:
                return {
                    "eligible": False,
                    "reason": "readiness_blocks_live_influence",
                }
            if readiness.get("allow_live_position_influence") is True:
                live_influence_allowed = True

        evaluation_policy = safe_dict(node.get("evaluation_policy"))
        if evaluation_policy:
            governance_seen = True
            if evaluation_policy.get("live_mutation") is False:
                return {
                    "eligible": False,
                    "reason": "evaluation_policy_blocks_live_mutation",
                }
            if evaluation_policy.get("live_mutation") is True:
                live_influence_allowed = True

        prediction_quality = safe_dict(node.get("prediction_quality"))
        if prediction_quality:
            governance_seen = True
            if prediction_quality.get("production_eligible") is False or prediction_quality.get(
                "anomalous"
            ) is True:
                return {
                    "eligible": False,
                    "reason": str(
                        prediction_quality.get("reason") or "prediction_quality_blocked"
                    ),
                }
            quality_approved = (
                prediction_quality.get("production_eligible") is True
                and prediction_quality.get("anomalous") is not True
            )

        objective_name = str(
            node.get("artifact_objective")
            or node.get("objective_name")
            or node.get("objective")
            or ""
        ).strip()
        objective_version = str(
            node.get("artifact_objective_version") or node.get("objective_version") or ""
        ).strip()
        if objective_name:
            governance_seen = True
            if objective_name != RETURN_OBJECTIVE_NAME:
                return {"eligible": False, "reason": "artifact_objective_mismatch"}
            objective_approved = True
        if objective_version:
            governance_seen = True
            if objective_version != RETURN_OBJECTIVE_VERSION:
                return {"eligible": False, "reason": "artifact_objective_version_mismatch"}
            objective_version_approved = True

        label_name = str(node.get("label_name") or "").strip()
        if label_name:
            governance_seen = True
            if label_name != RETURN_LABEL_NAME:
                return {"eligible": False, "reason": "artifact_return_label_mismatch"}
            label_approved = True

        label_version = str(node.get("label_version") or "").strip()
        if label_version:
            governance_seen = True
            if label_version != RETURN_LABEL_VERSION:
                return {
                    "eligible": False,
                    "reason": "artifact_return_label_version_mismatch",
                }
            label_version_approved = True

        cost_policy = str(node.get("training_cost_policy") or "").strip()
        if cost_policy:
            governance_seen = True
            if cost_policy != "separated_market_opportunity_and_execution_cost_tasks":
                return {"eligible": False, "reason": "artifact_cost_policy_incomplete"}
            cost_policy_approved = True

        supervision_version = str(node.get("profit_supervision_version") or "").strip()
        if supervision_version:
            governance_seen = True
            if supervision_version != PROFIT_SUPERVISION_VERSION:
                return {
                    "eligible": False,
                    "reason": "artifact_profit_supervision_version_mismatch",
                }
            supervision_approved = True

        return_semantics = str(node.get("return_semantics") or "").strip()
        if return_semantics:
            governance_seen = True
            if return_semantics != "gross_market_opportunity_before_execution":
                return {"eligible": False, "reason": "artifact_return_semantics_mismatch"}
            return_semantics_approved = True

    required = {
        "governance_metadata": governance_seen,
        "live_route": route_live,
        "live_influence": live_influence_allowed,
        "promotion_ready": promotion_ready,
        "prediction_quality": quality_approved,
        "return_objective": objective_approved,
        "return_objective_version": objective_version_approved,
        "return_label": label_approved,
        "return_label_version": label_version_approved,
        "separated_cost_policy": cost_policy_approved,
        "profit_supervision": supervision_approved,
        "gross_market_return_semantics": return_semantics_approved,
    }
    missing = [name for name, present in required.items() if not present]
    if missing:
        return {
            "eligible": False,
            "reason": "production_governance_incomplete",
            "missing_governance": missing,
        }
    contract_side = _signal_contract_side(payload)
    distribution_eligibility = signal_return_distribution_eligibility(
        payload,
        contract_side,
    )
    if distribution_eligibility.get("eligible") is not True:
        return distribution_eligibility
    return {
        "eligible": True,
        "reason": "governance_and_return_distribution_allow_live_influence",
        "side": contract_side,
        "return_distribution": distribution_eligibility.get("contract"),
    }


def signal_production_eligible(payload: dict[str, Any]) -> bool:
    return bool(signal_production_eligibility(payload).get("eligible"))


def payload_side(payload: dict[str, Any] | None, side_key: str = "best_side") -> str:
    if not isinstance(payload, dict):
        return ""
    value = str(
        payload.get(side_key)
        or payload.get("side")
        or payload.get("predicted_side")
        or payload.get("prediction_side")
        or payload.get("recommendation_side")
        or payload.get("best_action")
        or payload.get("action")
        or payload.get("side_label")
        or payload.get("action_label")
        or ""
    ).lower()
    if value in {"long", "short"}:
        return value
    if any(
        token in value
        for token in (
            "long",
            "bull",
            "up",
            "做多",
            "开多",
            "平空",
            *LEGACY_MOJIBAKE_LONG_LABELS,
        )
    ):
        return "long"
    if any(
        token in value
        for token in (
            "short",
            "bear",
            "down",
            "做空",
            "开空",
            "平多",
            *LEGACY_MOJIBAKE_SHORT_LABELS,
        )
    ):
        return "short"

    direction = str(payload.get("direction") or payload.get("forecast_direction") or "").lower()
    if direction == "up":
        return "long"
    if direction == "down":
        return "short"

    label = str(payload.get("label") or payload.get("sentiment") or "").lower()
    score = safe_float(payload.get("score", payload.get("sentiment_score", 0.0)), 0.0)
    if label in {"positive", "bullish"} or score > 0:
        return "long"
    if label in {"negative", "bearish"} or score < 0:
        return "short"
    return ""


def has_any_key(payload: dict[str, Any], *keys: str) -> bool:
    return any(key in payload for key in keys)


def has_signal_evidence(payload: dict[str, Any]) -> bool:
    """Return True when a payload has displayable model evidence."""
    if not isinstance(payload, dict) or not payload:
        return False
    if payload_side(payload) in {"long", "short"}:
        return True
    if safe_dict(payload.get("return_distribution_contract")):
        return True
    return has_any_key(
        payload,
        "expected_return_pct",
        "expected_move_pct",
        "forecast_return_pct",
        "return_pct",
        "expected_profit_pct",
        "score",
        "sentiment_score",
    )


def entry_signal_payloads(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ml = safe_dict(raw.get("ml_signal"))
    if not ml:
        ml = first_tool_payload(
            raw,
            "ml_signal",
            "local_ml_signal",
            "ml_prediction",
            "local_ml_prediction",
        )
    predictions = safe_list(ml.get("predictions"))
    primary_prediction = safe_dict(predictions[0]) if predictions else {}
    primary_ml = primary_prediction or ml
    return {
        "ml": ml,
        "primary_ml": primary_ml,
        "server_profit": first_tool_payload(
            raw,
            "profit_prediction",
            "profit_model",
            "server_profit",
            "server_profit_model",
            "profit",
        ),
        "timeseries": first_tool_payload(
            raw,
            "time_series_prediction",
            "timeseries_prediction",
            "sequence_prediction",
            "timeseries",
            "time_series",
        ),
        "sentiment": first_tool_payload(
            raw,
            "sentiment_analysis",
            "sentiment_prediction",
            "sentiment_model",
            "sentiment",
        ),
    }


def extract_entry_signal_sides(
    raw: dict[str, Any],
    *,
    ml_influence_enabled: bool = True,
) -> dict[str, Any]:
    payloads = entry_signal_payloads(raw)
    ml = payloads["ml"]
    primary_ml = payloads["primary_ml"]
    profit = payloads["server_profit"]
    timeseries = payloads["timeseries"]
    sentiment = payloads["sentiment"]
    ml_side = payload_side(primary_ml)
    profit_side = payload_side(profit)
    timeseries_side = payload_side(timeseries)
    sentiment_side = payload_side(sentiment)

    def distribution_signal(
        payload: dict[str, Any],
        side: str,
    ) -> dict[str, Any]:
        distribution = signal_return_distribution(payload, side)
        eligibility = signal_return_distribution_eligibility(payload, side)
        return {
            "available": signal_available(payload) and bool(distribution),
            "side": side,
            "raw_expected_return_pct": distribution.get(
                "raw_expected_return_pct"
            ),
            "objective_expected_return_pct": distribution.get(
                "objective_expected_return_pct"
            ),
            "lower_quantile_return_pct": distribution.get(
                "lower_quantile_return_pct"
            ),
            "dispersion_pct": distribution.get("dispersion_pct"),
            "tail_loss_probability": distribution.get(
                "tail_loss_probability"
            ),
            "tail_loss_scale_pct": distribution.get("tail_loss_scale_pct"),
            "production_eligible": eligibility.get("eligible") is True,
            "eligibility_reason": eligibility.get("reason"),
            "return_distribution_contract": distribution,
        }

    signals: dict[str, dict[str, Any]] = {
        "ml": {
            **distribution_signal(ml, ml_side),
            "influence_enabled": bool(ml_influence_enabled),
        },
        "server_profit": distribution_signal(profit, profit_side),
        "timeseries": distribution_signal(timeseries, timeseries_side),
        "sentiment": {
            "available": signal_available(sentiment),
            "side": sentiment_side,
            "score": safe_float(sentiment.get("score", sentiment.get("sentiment_score", 0.0)), 0.0),
        },
    }
    return signals
