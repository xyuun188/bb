"""Shared extraction of entry-side model signals from decision payloads."""

from __future__ import annotations

from typing import Any

from services.return_objective import RETURN_OBJECTIVE_NAME, RETURN_OBJECTIVE_VERSION

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
        if side not in {"long", "short"}:
            long_expected = safe_float(
                normalized.get(
                    "adjusted_long_return_pct",
                    normalized.get(
                        "long_expected_return_pct",
                        normalized.get("expected_long_return_pct"),
                    ),
                ),
                0.0,
            )
            short_expected = safe_float(
                normalized.get(
                    "adjusted_short_return_pct",
                    normalized.get(
                        "short_expected_return_pct",
                        normalized.get("expected_short_return_pct"),
                    ),
                ),
                0.0,
            )
            if long_expected or short_expected:
                side = "long" if long_expected >= short_expected else "short"
        if side in {"long", "short"}:
            normalized["best_side"] = side
            normalized["side"] = side
        if "expected_return_pct" not in normalized:
            normalized["expected_return_pct"] = expected_return_pct(normalized, side)
        if "profit_edge_pct" not in normalized:
            long_value = expected_return_pct(normalized, "long")
            short_value = expected_return_pct(normalized, "short")
            normalized["profit_edge_pct"] = round(abs(long_value - short_value), 6)
    elif name == "time_series_prediction":
        side = payload_side(normalized)
        if side in {"long", "short"}:
            normalized["best_side"] = side
            normalized["side"] = side
        if "expected_return_pct" not in normalized and "expected_move_pct" in normalized:
            normalized["expected_return_pct"] = normalized.get("expected_move_pct")
    elif name == "sentiment_analysis":
        side = payload_side(normalized)
        if side in {"long", "short"}:
            normalized["best_side"] = side
            normalized["side"] = side
        if (
            "expected_return_pct" not in normalized
            and "expected_return_from_sentiment_pct" in normalized
        ):
            normalized["expected_return_pct"] = normalized.get("expected_return_from_sentiment_pct")
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


def signal_production_eligibility(payload: dict[str, Any]) -> dict[str, Any]:
    """Return auditable live-decision eligibility without hiding shadow observations."""

    if not signal_available(payload):
        return {"eligible": False, "reason": "signal_unavailable"}

    governance_seen = False
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

        if "promotion_ready" in node:
            governance_seen = True
            if node.get("promotion_ready") is False:
                return {"eligible": False, "reason": "promotion_not_ready"}

        readiness = safe_dict(node.get("readiness"))
        if readiness:
            governance_seen = True
            if readiness.get("allow_live_position_influence") is False:
                return {
                    "eligible": False,
                    "reason": "readiness_blocks_live_influence",
                }

        evaluation_policy = safe_dict(node.get("evaluation_policy"))
        if evaluation_policy:
            governance_seen = True
            if evaluation_policy.get("live_mutation") is False:
                return {
                    "eligible": False,
                    "reason": "evaluation_policy_blocks_live_mutation",
                }

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
        if objective_version:
            governance_seen = True
            if objective_version != RETURN_OBJECTIVE_VERSION:
                return {"eligible": False, "reason": "artifact_objective_version_mismatch"}

    return {
        "eligible": True,
        "reason": "governance_allows_live_influence" if governance_seen else "legacy_internal_signal",
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


def opposite_side(side: str) -> str:
    if side == "long":
        return "short"
    if side == "short":
        return "long"
    return ""


def mark_signal_from_opportunity(
    signal: dict[str, Any],
    *,
    side: str = "",
    expected_return_pct: Any = None,
    available: bool = True,
) -> None:
    if not available:
        return
    if side in {"long", "short"} and not signal.get("side"):
        signal["side"] = side
    if expected_return_pct is not None and not signal.get("expected_return_pct"):
        signal["expected_return_pct"] = safe_float(expected_return_pct, 0.0)
    if signal.get("side") or expected_return_pct is not None:
        signal["available"] = True


def has_any_key(payload: dict[str, Any], *keys: str) -> bool:
    return any(key in payload for key in keys)


def has_signal_evidence(payload: dict[str, Any]) -> bool:
    """Return True when a payload has displayable model evidence."""
    if not isinstance(payload, dict) or not payload:
        return False
    if payload_side(payload) in {"long", "short"}:
        return True
    return has_any_key(
        payload,
        "expected_return_pct",
        "best_expected_return_pct",
        "expected_move_pct",
        "forecast_return_pct",
        "return_pct",
        "expected_profit_pct",
        "score",
        "sentiment_score",
    )


def expected_return_pct(payload: dict[str, Any], side: str = "") -> float:
    side = str(side or "").lower()
    keys: list[str] = []
    if side in {"long", "short"}:
        keys.extend(
            [
                f"adjusted_{side}_return_pct",
                f"{side}_expected_return_pct",
                f"expected_{side}_return_pct",
                f"{side}_return_pct",
            ]
        )
    keys.extend(
        [
            "expected_return_pct",
            "expected_net_return_pct",
            "best_expected_return_pct",
            "expected_move_pct",
            "expected_return_from_sentiment_pct",
            "forecast_return_pct",
            "return_pct",
            "expected_profit_pct",
        ]
    )
    for key in keys:
        if key in payload:
            return safe_float(payload.get(key), 0.0)
    return 0.0


def directional_expected_return_pct(payload: dict[str, Any], side: str = "") -> float:
    """Return expected return from the perspective of the requested trade side.

    Directional models often report price movement: a negative move is adverse for
    long entries but favorable for short entries. Profit models already expose
    side-adjusted fields such as ``short_expected_return_pct``; this helper only
    flips generic movement fields for short-side directional signals.
    """

    side = str(side or "").lower()
    if side not in {"long", "short"}:
        return expected_return_pct(payload, side)
    for key in (
        f"adjusted_{side}_return_pct",
        f"{side}_expected_return_pct",
        f"expected_{side}_return_pct",
        f"{side}_return_pct",
    ):
        if key in payload:
            return safe_float(payload.get(key), 0.0)
    move = expected_return_pct(payload, side)
    if side == "short":
        return -move
    return move


def side_has_positive_expected_return(payload: dict[str, Any], side: str) -> bool:
    """True when a side has positive adjusted/side expected return evidence."""
    return expected_return_pct(payload, side) > 0.0


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
    profit_side = payload_side(profit)
    timeseries_side = payload_side(timeseries)
    sentiment_side = payload_side(sentiment)
    signals: dict[str, dict[str, Any]] = {
        "ml": {
            "available": signal_available(primary_ml) and has_signal_evidence(primary_ml),
            "side": payload_side(primary_ml),
            "expected_return_pct": safe_float(
                primary_ml.get("best_expected_return_pct", ml.get("expected_return_pct", 0.0)),
                0.0,
            ),
            "influence_enabled": bool(ml_influence_enabled),
        },
        "server_profit": {
            "available": signal_available(profit),
            "side": profit_side,
            "expected_return_pct": expected_return_pct(profit, profit_side),
        },
        "timeseries": {
            "available": signal_available(timeseries),
            "side": timeseries_side,
            "expected_return_pct": expected_return_pct(timeseries, timeseries_side),
        },
        "sentiment": {
            "available": signal_available(sentiment),
            "side": sentiment_side,
            "expected_return_pct": expected_return_pct(sentiment, sentiment_side),
            "score": safe_float(sentiment.get("score", sentiment.get("sentiment_score", 0.0)), 0.0),
        },
    }
    opportunity = safe_dict(raw.get("opportunity_score"))
    evidence_score = safe_dict(opportunity.get("evidence_score"))
    components = safe_list(evidence_score.get("components"))
    for component in components:
        if not isinstance(component, dict):
            continue
        source = str(component.get("source") or "")
        if source not in signals:
            continue
        current = signals[source]
        component_side = payload_side(component, side_key="side")
        if not current.get("side") and component_side in {"long", "short"}:
            current["side"] = component_side
        if not current.get("available") and component.get("status") != "missing":
            current["available"] = bool(component.get("available", True))
        if not current.get("expected_return_pct"):
            current["expected_return_pct"] = safe_float(component.get("expected_return_pct"), 0.0)

    entry_side = payload_side(opportunity, side_key="side")
    ml_enabled = bool(opportunity.get("ml_influence_enabled", ml_influence_enabled))
    signals["ml"]["influence_enabled"] = ml_enabled
    has_ml_opportunity_data = has_any_key(
        opportunity,
        "expected_return_pct",
        "ml_aligned",
        "ml_profit_quality_score",
        "ml_influence_enabled",
    )
    if entry_side:
        mark_signal_from_opportunity(
            signals["ml"],
            side=entry_side if opportunity.get("ml_aligned", True) else opposite_side(entry_side),
            expected_return_pct=opportunity.get("expected_return_pct"),
            available=has_ml_opportunity_data,
        )

    server_side = payload_side({"best_side": opportunity.get("server_profit_best_side")})
    if not server_side and entry_side and opportunity.get("local_profit_aligned"):
        server_side = entry_side
    mark_signal_from_opportunity(
        signals["server_profit"],
        side=server_side,
        expected_return_pct=opportunity.get("server_profit_expected_return_pct"),
        available=(
            "server_profit_expected_return_pct" in opportunity
            or "server_profit_best_side" in opportunity
            or "server_profit_loss_probability" in opportunity
        ),
    )

    timeseries_side = payload_side({"best_side": opportunity.get("timeseries_best_side")})
    if not timeseries_side and entry_side and "timeseries_aligned" in opportunity:
        timeseries_side = (
            entry_side if opportunity.get("timeseries_aligned") else opposite_side(entry_side)
        )
    mark_signal_from_opportunity(
        signals["timeseries"],
        side=timeseries_side,
        expected_return_pct=opportunity.get("timeseries_expected_return_pct"),
        available="timeseries_expected_return_pct" in opportunity
        or "timeseries_aligned" in opportunity,
    )

    sentiment_side = payload_side({"best_side": opportunity.get("sentiment_best_side")})
    if sentiment_side or "sentiment_expected_return_pct" in opportunity:
        mark_signal_from_opportunity(
            signals["sentiment"],
            side=sentiment_side,
            expected_return_pct=opportunity.get("sentiment_expected_return_pct"),
        )
    return signals
