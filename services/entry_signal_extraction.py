"""Shared extraction of entry-side model signals from decision payloads."""

from __future__ import annotations

from typing import Any

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


def first_tool_payload(raw: dict[str, Any], *keys: str) -> dict[str, Any]:
    containers = (
        raw.get("local_ai_tools"),
        raw.get("server_quant_tools"),
        raw.get("quant_tools"),
        raw.get("local_tools"),
        raw.get("server_tools"),
        raw,
    )
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in keys:
            value = container.get(key)
            if isinstance(value, dict):
                return value
    return {}


def signal_available(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    for key in ("available", "enabled", "ok"):
        if key in payload and payload.get(key) is False:
            return False
    return True


def payload_side(payload: dict[str, Any] | None, side_key: str = "best_side") -> str:
    if not isinstance(payload, dict):
        return ""
    value = str(
        payload.get(side_key)
        or payload.get("side")
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
            "best_expected_return_pct",
            "expected_move_pct",
            "forecast_return_pct",
            "return_pct",
            "expected_profit_pct",
        ]
    )
    for key in keys:
        if key in payload:
            return safe_float(payload.get(key), 0.0)
    return 0.0


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
