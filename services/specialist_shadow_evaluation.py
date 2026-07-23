from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from math import sqrt
from typing import Any

from services.execution_cost_model import execution_cost_estimate
from services.profit_supervision import shadow_fee_after_return_labels
from services.profit_training_contract import PROFIT_TRAINING_TARGET
from services.training_epoch import load_training_epoch_start

DEFAULT_WINDOW_HOURS = 168
MAX_WORST_SAMPLE_COUNT = 8


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        number = float(value)
        return number if number == number and abs(number) != float("inf") else default
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _row_in_window(row: Any, since: datetime) -> bool:
    for key in ("due_at", "created_at", "updated_at"):
        value = _as_utc(_row_get(row, key))
        if value is not None:
            return value >= since
    return True


def _features(row: Any) -> dict[str, Any]:
    value = _row_get(row, "feature_snapshot") or {}
    return value if isinstance(value, dict) else {}


def _local_shadow(row: Any) -> dict[str, Any]:
    value = _features(row).get("local_ai_tools_shadow") or {}
    return value if isinstance(value, dict) else {}


def _cost_complete_shadow_return(
    row: Any,
    *,
    side: str,
    gross_return_pct: float,
) -> tuple[float, dict[str, Any]] | None:
    features = _features(row)
    execution_cost = execution_cost_estimate(features)
    funding_rate = _safe_float(features.get("funding_rate"), None)
    funding_interval_minutes = _safe_float(
        features.get("funding_interval_minutes"),
        None,
    )
    if funding_interval_minutes is None:
        funding_interval_hours = _safe_float(features.get("funding_interval_hours"), None)
        if funding_interval_hours is not None:
            funding_interval_minutes = funding_interval_hours * 60.0
    horizon_minutes = max(
        _safe_float(_row_get(row, "horizon_minutes"), 0.0) or 0.0,
        0.0,
    )
    if (
        not execution_cost.production_eligible
        or funding_rate is None
        or funding_interval_minutes is None
        or funding_interval_minutes <= 0
        or horizon_minutes <= 0
    ):
        return None
    funding_drag = funding_rate * 100.0 * horizon_minutes / funding_interval_minutes
    net_return = (
        gross_return_pct
        - execution_cost.fee_pct
        - execution_cost.slippage_pct
        - funding_drag
        if side == "long"
        else gross_return_pct
        - execution_cost.fee_pct
        - execution_cost.slippage_pct
        + funding_drag
    )
    return net_return, {
        **execution_cost.to_dict(),
        "funding_drag_pct": round(funding_drag, 8),
        "funding_interval_minutes": round(funding_interval_minutes, 8),
        "horizon_minutes": round(horizon_minutes, 8),
    }


def _mean_lcb(values: list[float]) -> float | None:
    if not values:
        return None
    center = sum(values) / len(values)
    if len(values) <= 1:
        return center
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return center - sqrt(max(variance, 0.0) / len(values))


def _market_regime_from_features(features: dict[str, Any]) -> str:
    for key in ("market_regime", "regime", "market_state"):
        value = features.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()[:80]
        if isinstance(value, dict):
            label = _safe_str(
                value.get("regime")
                or value.get("mode")
                or value.get("label")
                or value.get("state")
            ).lower()
            if label:
                return label[:80]
    nested = features.get("market_regime_context")
    if isinstance(nested, dict):
        label = _safe_str(
            nested.get("regime")
            or nested.get("mode")
            or nested.get("label")
            or nested.get("state")
        ).lower()
        if label:
            return label[:80]
    volatility = _safe_float(features.get("volatility"), None)
    if volatility is None:
        return "unknown"
    return "observed_volatility"


def _market_regime(row: Any) -> str:
    return _market_regime_from_features(_features(row))


def _authoritative_local_tools(sample: Any) -> dict[str, Any]:
    raw = _row_get(sample, "raw_llm_response")
    if not isinstance(raw, dict):
        return {}
    tools = raw.get("local_ai_tools")
    if isinstance(tools, dict):
        return tools
    unified = raw.get("unified")
    if isinstance(unified, dict) and isinstance(unified.get("local_ai_tools"), dict):
        return unified["local_ai_tools"]
    return {}


def _authoritative_market_regime(sample: Any) -> str:
    raw = _row_get(sample, "raw_llm_response")
    return _market_regime_from_features(raw if isinstance(raw, dict) else {})


def _actual_best_side(row: Any) -> str:
    best = _safe_str(_row_get(row, "best_action")).lower()
    if best in {"long", "short", "hold"}:
        return best
    fee_after = shadow_fee_after_return_labels(_shadow_quality_payload(row))
    long_return = _safe_float(
        fee_after.get("long_net_return_after_all_cost_pct"), None
    )
    short_return = _safe_float(
        fee_after.get("short_net_return_after_all_cost_pct"), None
    )
    if long_return is None or short_return is None:
        return ""
    if max(long_return, short_return) <= 0.0:
        return "hold"
    return "long" if long_return >= short_return else "short"


def _shadow_quality_payload(row: Any) -> dict[str, Any]:
    return {
        "horizon_minutes": _row_get(row, "horizon_minutes"),
        "long_return_pct": _row_get(row, "long_return_pct"),
        "short_return_pct": _row_get(row, "short_return_pct"),
        "features": _features(row),
    }


def _tool_direction(tool: dict[str, Any]) -> str:
    side = _safe_str(
        tool.get("timesfm_shadow_side")
        or tool.get("best_side")
        or tool.get("side")
    ).lower()
    if side in {"long", "short"}:
        return side
    direction = _safe_str(tool.get("direction")).lower()
    return "long" if direction == "up" else "short" if direction == "down" else ""


def _result_direction(result: dict[str, Any]) -> str:
    side = _safe_str(result.get("best_side") or result.get("side")).lower()
    if side in {"long", "short"}:
        return side
    direction = _safe_str(result.get("direction")).lower()
    return "long" if direction == "up" else "short" if direction == "down" else ""


def _tool_expected_return(tool: dict[str, Any]) -> float | None:
    for key in (
        "timesfm_shadow_expected_return_pct",
        "timesfm_shadow_expected_move_pct",
        "expected_return_pct",
        "expected_move_pct",
    ):
        value = _safe_float(tool.get(key), None)
        if value is not None:
            return value
    professional = tool.get("professional_model_shadow")
    if isinstance(professional, dict):
        result = professional.get("shadow_result")
        if isinstance(result, dict):
            for key in ("expected_return_pct", "expected_move_pct"):
                value = _safe_float(result.get(key), None)
                if value is not None:
                    return value
    return None


def _result_expected_return(result: dict[str, Any]) -> float | None:
    for key in ("expected_return_pct", "expected_move_pct"):
        value = _safe_float(result.get(key), None)
        if value is not None:
            return value
    return None


def _actual_return_for_side(row: Any, side: str) -> float | None:
    if side == "long":
        return _safe_float(_row_get(row, "long_return_pct"), None)
    if side == "short":
        return _safe_float(_row_get(row, "short_return_pct"), None)
    return None


def _actual_inference(tool: dict[str, Any]) -> bool:
    professional = tool.get("professional_model_shadow")
    if isinstance(professional, dict):
        if bool(professional.get("actual_inference")):
            return True
        result = professional.get("shadow_result")
        if isinstance(result, dict) and bool(result.get("actual_inference")):
            return True
    return bool(tool.get("specialist_inference_active"))


def _result_actual_inference(result: dict[str, Any]) -> bool:
    return bool(isinstance(result, dict) and result.get("actual_inference"))


def _baseline_only_shadow(tool: dict[str, Any]) -> bool:
    professional = tool.get("professional_model_shadow")
    if not isinstance(professional, dict):
        return False
    if _actual_inference(tool):
        return False
    return bool(professional.get("baseline_response"))


def _tool_model_name(tool_name: str, tool: dict[str, Any]) -> str:
    if tool_name == "time_series_prediction" and tool.get("timesfm_shadow_expected_return_pct") is not None:
        return "timesfm_shadow_challenger"
    professional = tool.get("professional_model_shadow")
    if isinstance(professional, dict):
        result = professional.get("shadow_result")
        if isinstance(result, dict):
            model = _safe_str(result.get("model"))
            if model:
                return model
    return _safe_str(tool.get("model")) or tool_name


def _timeseries_shadow_candidates(tool: dict[str, Any]) -> list[dict[str, Any]]:
    professional = tool.get("professional_model_shadow")
    if not isinstance(professional, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key, model_key in (
        ("primary_shadow_result", "primary_model"),
        ("challenger_shadow_result", "challenger_model"),
    ):
        result = professional.get(key)
        result = result if isinstance(result, dict) else {}
        model = _safe_str(professional.get(model_key) or result.get("model"))
        if not model:
            continue
        rows.append(
            {
                "tool": "time_series_prediction",
                "model": model,
                "direction": _result_direction(result),
                "expected_return_pct": _result_expected_return(result),
                "actual_inference": _result_actual_inference(result),
                "sequence_length": int(_safe_float(result.get("sequence_length"), 0.0) or 0),
                "legacy_mixed_shadow": False,
                "fallback_reason": (
                    ""
                    if _result_actual_inference(result)
                    else _safe_str(result.get("reason"))
                    or f"{model_key}_inference_unavailable"
                ),
            }
        )
    if rows:
        return rows

    result = professional.get("shadow_result")
    if not isinstance(result, dict) or not _result_actual_inference(result):
        return []
    model = _safe_str(result.get("model"))
    if not model:
        return []
    return [
        {
            "tool": "time_series_prediction",
            "model": model,
            "direction": _result_direction(result) or _tool_direction(tool),
            "expected_return_pct": _result_expected_return(result) or _tool_expected_return(tool),
            "actual_inference": True,
            "sequence_length": int(_safe_float(result.get("sequence_length"), 0.0) or 0),
            "legacy_mixed_shadow": True,
            "fallback_reason": "legacy_mixed_timeseries_result",
        }
    ]


def _sentiment_shadow_candidates(tool: dict[str, Any]) -> list[dict[str, Any]]:
    professional = tool.get("professional_model_shadow")
    if not isinstance(professional, dict):
        return []
    predictions = professional.get("predictions")
    predictions = predictions if isinstance(predictions, dict) else {}
    model_by_slot = {
        "sentiment_primary": _safe_str(professional.get("primary_model"))
        or "ProsusAI/finbert",
        "sentiment_challenger": _safe_str(professional.get("challenger_model"))
        or "yiyanghkust/finbert-tone",
    }
    rows = []
    for slot, model_name in model_by_slot.items():
        prediction = predictions.get(slot)
        prediction = prediction if isinstance(prediction, dict) else {}
        label = _safe_str(prediction.get("label")).lower()
        score = _safe_float(prediction.get("score"), None)
        direction = "long" if label == "positive" else "short" if label == "negative" else ""
        rows.append(
            {
                "tool": "sentiment_analysis",
                "model": model_name,
                "direction": direction,
                "expected_return_pct": None,
                "signal_score": score,
                "actual_inference": bool(prediction.get("available")),
                "sequence_length": 0,
                "legacy_mixed_shadow": False,
                "fallback_reason": (
                    ""
                    if prediction.get("available")
                    else _safe_str(prediction.get("reason")) or f"{slot}_inference_unavailable"
                ),
            }
        )
    return rows


def _tool_shadow_candidates(tool_name: str, tool: dict[str, Any]) -> list[dict[str, Any]]:
    if tool_name == "time_series_prediction":
        rows = _timeseries_shadow_candidates(tool)
        if rows:
            return rows
    if tool_name == "sentiment_analysis":
        rows = _sentiment_shadow_candidates(tool)
        if rows:
            return rows
    return [
        {
            "tool": tool_name,
            "model": _tool_model_name(tool_name, tool),
            "direction": _tool_direction(tool),
            "expected_return_pct": _tool_expected_return(tool),
            "actual_inference": _actual_inference(tool),
            "sequence_length": 0,
            "legacy_mixed_shadow": False,
            "fallback_reason": _safe_str(tool.get("fallback_reason")),
        }
    ]


def _empty_metric(tool_name: str, model_name: str) -> dict[str, Any]:
    return {
        "tool": tool_name,
        "model": model_name,
        "sample_count": 0,
        "actual_inference_count": 0,
        "fallback_count": 0,
        "shadow_fallback_count": 0,
        "fallback_reasons": Counter(),
        "direction_count": 0,
        "direction_hit_count": 0,
        "false_signal_count": 0,
        "realized_return_sum_pct": 0.0,
        "realized_gross_profit_pct": 0.0,
        "realized_gross_loss_pct": 0.0,
        "expected_return_sum_pct": 0.0,
        "expected_return_count": 0,
        "signal_score_sum": 0.0,
        "signal_score_count": 0,
        "worst_realized_return_pct": None,
        "best_realized_return_pct": None,
        "symbols": Counter(),
        "tail_loss_count": 0,
        "tail_loss_symbols": Counter(),
        "worst_samples": [],
        "blockers": [],
        "legacy_mixed_shadow_count": 0,
        "legacy_quarantined_count": 0,
        "shadow_events": [],
        "authoritative_events": [],
        "authoritative_evidence": [],
        "authoritative_attempt_count": 0,
        "authoritative_actual_inference_count": 0,
        "authoritative_fallback_count": 0,
        "authoritative_direction_aligned_count": 0,
        "authoritative_direction_mismatch_count": 0,
        "authoritative_return_sum_pct": 0.0,
        "authoritative_gross_profit_pct": 0.0,
        "authoritative_gross_loss_pct": 0.0,
        "authoritative_worst_return_pct": None,
        "authoritative_best_return_pct": None,
        "authoritative_tail_loss_count": 0,
    }


def _iso_row_datetime(row: Any, key: str) -> str | None:
    value = _as_utc(_row_get(row, key))
    return value.isoformat() if value is not None else None


def _compact_worst_sample(
    row: Any,
    *,
    tool_name: str,
    model_name: str,
    symbol: str,
    predicted_side: str,
    actual_side: str,
    actual_return: float,
    expected_return: float | None,
    sequence_length: int,
    legacy_mixed_shadow: bool,
) -> dict[str, Any]:
    return {
        "shadow_backtest_id": _row_get(row, "id"),
        "symbol": symbol,
        "tool": tool_name,
        "model": model_name,
        "predicted_side": predicted_side,
        "actual_best_side": actual_side,
        "actual_return_pct": round(float(actual_return), 6),
        "expected_return_pct": None if expected_return is None else round(float(expected_return), 6),
        "long_net_return_after_all_cost_pct": shadow_fee_after_return_labels(
            _shadow_quality_payload(row)
        ).get("long_net_return_after_all_cost_pct"),
        "short_net_return_after_all_cost_pct": shadow_fee_after_return_labels(
            _shadow_quality_payload(row)
        ).get("short_net_return_after_all_cost_pct"),
        "created_at": _iso_row_datetime(row, "created_at"),
        "due_at": _iso_row_datetime(row, "due_at"),
        "sequence_length": sequence_length,
        "legacy_mixed_shadow": bool(legacy_mixed_shadow),
    }


def _remember_worst_sample(metric: dict[str, Any], sample: dict[str, Any]) -> None:
    samples = metric.setdefault("worst_samples", [])
    samples.append(sample)
    samples.sort(key=lambda item: float(item.get("actual_return_pct") or 0.0))
    del samples[MAX_WORST_SAMPLE_COUNT:]


def _event_timestamp(event: dict[str, Any]) -> str:
    return str(event.get("label_timestamp") or event.get("created_at") or "")


def _canonical_net_return(event: dict[str, Any]) -> float | None:
    """Return only the current fee-after objective label.

    Legacy ``return_after_all_cost_pct`` data is deliberately not accepted here:
    silently consuming it would let old objective data influence promotion.
    """

    return _safe_float(event.get("net_return_after_all_cost_pct"), None)


def _walk_forward_report(events: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(events, key=lambda item: (_event_timestamp(item), str(item.get("id") or "")))
    canonical = [item for item in ordered if _canonical_net_return(item) is not None]
    missing_canonical_count = len(ordered) - len(canonical)
    if not canonical:
        return {
            "status": "insufficient_authoritative_evidence",
            "fold_count": 0,
            "positive_fold_count": 0,
            "sample_count": 0,
            "missing_canonical_return_count": missing_canonical_count,
            "folds": [],
        }
    fold_count = max(min(int(sqrt(len(canonical))), len(canonical)), 1)
    folds = []
    for fold_index in range(fold_count):
        start = len(canonical) * fold_index // fold_count
        end = len(canonical) * (fold_index + 1) // fold_count
        fold_events = canonical[start:end]
        returns = [float(_canonical_net_return(item)) for item in fold_events]
        fold_lcb = _mean_lcb(returns)
        folds.append(
            {
                "fold": fold_index + 1,
                "sample_count": len(fold_events),
                "start_at": _event_timestamp(fold_events[0]) if fold_events else None,
                "end_at": _event_timestamp(fold_events[-1]) if fold_events else None,
                "avg_return_after_all_cost_pct": round(sum(returns) / max(len(returns), 1), 6),
                "return_lcb_pct": round(fold_lcb, 6) if fold_lcb is not None else None,
                "win_rate": round(
                    sum(1 for value in returns if value > 0) / max(len(returns), 1),
                    6,
                ),
            }
        )
    positive_folds = sum(
        1
        for fold in folds
        if float(fold.get("return_lcb_pct") or 0.0) > 0.0
    )
    sufficient = bool(canonical)
    stable = bool(sufficient and positive_folds == fold_count)
    return {
        "status": "stable" if stable else "unstable" if sufficient else "insufficient_authoritative_evidence",
        "fold_count": fold_count,
        "positive_fold_count": positive_folds,
        "sample_count": len(canonical),
        "missing_canonical_return_count": missing_canonical_count,
        "folds": folds,
    }


def _regime_stability_report(events: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    missing_canonical_count = 0
    for event in events:
        net_return = _canonical_net_return(event)
        if net_return is None:
            missing_canonical_count += 1
            continue
        regime = _safe_str(event.get("market_regime")) or "unknown"
        grouped.setdefault(regime, []).append(float(net_return))
    rows = [
        {
            "regime": regime,
            "sample_count": len(values),
            "avg_return_after_all_cost_pct": round(sum(values) / max(len(values), 1), 6),
            "win_rate": round(sum(1 for value in values if value > 0) / max(len(values), 1), 6),
            "return_lcb_pct": round(_mean_lcb(values) or 0.0, 6),
        }
        for regime, values in sorted(grouped.items())
    ]
    eligible = rows
    stable = bool(eligible and all(float(row["return_lcb_pct"]) > 0.0 for row in eligible))
    return {
        "status": "stable" if stable else "unstable" if eligible else "insufficient_regime_evidence",
        "eligible_regime_count": len(eligible),
        "missing_canonical_return_count": missing_canonical_count,
        "regimes": rows,
    }


def _rolling_distribution_report(events: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(events, key=lambda item: (_event_timestamp(item), str(item.get("id") or "")))
    canonical = [item for item in ordered if _canonical_net_return(item) is not None]
    missing_canonical_count = len(ordered) - len(canonical)
    windows = []
    dynamic_window_sizes = sorted(
        {
            max(int(sqrt(len(canonical))), 1),
            max(len(canonical) // 2, 1),
            len(canonical),
        }
    ) if canonical else []
    for window_size in dynamic_window_sizes:
        if len(canonical) < window_size:
            continue
        values = [float(_canonical_net_return(item)) for item in canonical[-window_size:]]
        return_lcb = _mean_lcb(values)
        windows.append(
            {
                "window_size": window_size,
                "avg_return_after_all_cost_pct": round(sum(values) / window_size, 6),
                "win_rate": round(sum(1 for value in values if value > 0) / window_size, 6),
                "worst_return_after_all_cost_pct": round(min(values), 6),
                "return_lcb_pct": round(return_lcb, 6) if return_lcb is not None else None,
            }
        )
    stable = bool(windows and all(float(window["return_lcb_pct"] or 0.0) > 0.0 for window in windows))
    return {
        "status": "stable" if stable else "unstable" if canonical else "insufficient_evidence",
        "sample_count": len(canonical),
        "missing_canonical_return_count": missing_canonical_count,
        "dynamic_windows": dynamic_window_sizes,
        "windows": windows,
    }


def _record_fallback(
    metric: dict[str, Any],
    candidate: dict[str, Any],
    *,
    source: str,
) -> None:
    reason = _safe_str(candidate.get("fallback_reason")) or "specialist_inference_unavailable"
    metric["fallback_count"] += 1
    metric["fallback_reasons"][reason] += 1
    if source == "authoritative":
        metric["authoritative_fallback_count"] += 1
    else:
        metric["shadow_fallback_count"] += 1


def _update_return_extrema(
    metric: dict[str, Any],
    value: float,
    *,
    worst_key: str,
    best_key: str,
) -> None:
    worst = metric.get(worst_key)
    best = metric.get(best_key)
    metric[worst_key] = round(value if worst is None else min(float(worst), value), 6)
    metric[best_key] = round(value if best is None else max(float(best), value), 6)


def _finalize_metric(metric: dict[str, Any]) -> dict[str, Any]:
    direction_count = int(metric.get("direction_count") or 0)
    expected_count = int(metric.get("expected_return_count") or 0)
    signal_score_count = int(metric.get("signal_score_count") or 0)
    realized_sum = float(metric.get("realized_return_sum_pct") or 0.0)
    expected_sum = float(metric.get("expected_return_sum_pct") or 0.0)
    authoritative_count = int(metric.get("authoritative_direction_aligned_count") or 0)
    authoritative_sum = float(metric.get("authoritative_return_sum_pct") or 0.0)
    hit_rate = float(metric.get("direction_hit_count") or 0) / max(direction_count, 1)
    avg_realized = realized_sum / max(direction_count, 1)
    avg_expected = expected_sum / max(expected_count, 1)
    avg_signal_score = float(metric.get("signal_score_sum") or 0.0) / max(
        signal_score_count, 1
    )
    avg_authoritative = authoritative_sum / max(authoritative_count, 1)
    gross_profit = float(metric.get("realized_gross_profit_pct") or 0.0)
    gross_loss = float(metric.get("realized_gross_loss_pct") or 0.0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    authoritative_gross_profit = float(
        metric.get("authoritative_gross_profit_pct") or 0.0
    )
    authoritative_gross_loss = float(metric.get("authoritative_gross_loss_pct") or 0.0)
    authoritative_profit_factor = (
        authoritative_gross_profit / authoritative_gross_loss
        if authoritative_gross_loss > 0
        else None
    )
    authoritative_events = list(metric.get("authoritative_events") or [])
    walk_forward = _walk_forward_report(authoritative_events)
    regime_stability = _regime_stability_report(authoritative_events)
    rolling_distribution = _rolling_distribution_report(authoritative_events)
    authoritative_returns = [
        float(value)
        for event in authoritative_events
        if (value := _canonical_net_return(event)) is not None
    ]
    shadow_returns = [
        float(value)
        for event in list(metric.get("shadow_events") or [])
        if (value := _canonical_net_return(event)) is not None
    ]
    authoritative_return_lcb = _mean_lcb(authoritative_returns)
    shadow_return_lcb = _mean_lcb(shadow_returns)
    blockers = []
    if not authoritative_returns:
        blockers.append("authoritative_return_distribution_missing")
    if authoritative_return_lcb is None or authoritative_return_lcb <= 0.0:
        blockers.append("authoritative_fee_after_return_lcb_not_positive")
    if authoritative_profit_factor is None:
        blockers.append("authoritative_profit_factor_undefined")
    elif authoritative_profit_factor <= 1.0:
        blockers.append("authoritative_profit_factor_below_unity")
    authoritative_worst = metric.get("authoritative_worst_return_pct")
    blockers = list(dict.fromkeys(blockers))
    blocker_counts = dict(Counter(blockers))
    return {
        "tool": metric["tool"],
        "model": metric["model"],
        "sample_count": int(metric.get("sample_count") or 0),
        "actual_inference_count": int(metric.get("actual_inference_count") or 0),
        "fallback_count": int(metric.get("fallback_count") or 0),
        "shadow_fallback_count": int(metric.get("shadow_fallback_count") or 0),
        "fallback_reasons": dict(metric.get("fallback_reasons") or {}),
        "direction_count": direction_count,
        "direction_hit_count": int(metric.get("direction_hit_count") or 0),
        "direction_hit_rate": round(hit_rate, 6),
        "false_signal_count": int(metric.get("false_signal_count") or 0),
        "avg_realized_return_pct": round(avg_realized, 6),
        "avg_shadow_return_after_all_cost_pct": round(avg_realized, 6),
        "profit_factor": round(profit_factor, 6) if profit_factor is not None else None,
        "gross_profit_return_pct": round(gross_profit, 6),
        "gross_loss_return_pct": round(gross_loss, 6),
        "avg_expected_return_pct": round(avg_expected, 6),
        "avg_signal_score": round(avg_signal_score, 6),
        "worst_realized_return_pct": metric.get("worst_realized_return_pct"),
        "best_realized_return_pct": metric.get("best_realized_return_pct"),
        "legacy_mixed_shadow_count": int(metric.get("legacy_mixed_shadow_count") or 0),
        "legacy_quarantined_count": int(metric.get("legacy_quarantined_count") or 0),
        "tail_loss_count": int(metric.get("tail_loss_count") or 0),
        "tail_loss_symbols": [
            {"symbol": symbol, "count": count}
            for symbol, count in metric["tail_loss_symbols"].most_common(10)
        ],
        "worst_samples": list(metric.get("worst_samples") or [])[:MAX_WORST_SAMPLE_COUNT],
        "shadow_event_count": len(metric.get("shadow_events") or []),
        "shadow_events": list(metric.get("shadow_events") or [])[:MAX_WORST_SAMPLE_COUNT],
        "shadow_events_truncated": (
            len(metric.get("shadow_events") or []) > MAX_WORST_SAMPLE_COUNT
        ),
        "authoritative_attempt_count": int(metric.get("authoritative_attempt_count") or 0),
        "authoritative_actual_inference_count": int(
            metric.get("authoritative_actual_inference_count") or 0
        ),
        "total_actual_inference_count": int(metric.get("actual_inference_count") or 0)
        + int(metric.get("authoritative_actual_inference_count") or 0),
        "authoritative_fallback_count": int(metric.get("authoritative_fallback_count") or 0),
        "authoritative_direction_aligned_count": authoritative_count,
        "authoritative_direction_mismatch_count": int(
            metric.get("authoritative_direction_mismatch_count") or 0
        ),
        "authoritative_avg_return_after_all_cost_pct": round(avg_authoritative, 6),
        "authoritative_return_lcb_pct": (
            round(authoritative_return_lcb, 6)
            if authoritative_return_lcb is not None
            else None
        ),
        "shadow_return_lcb_pct": (
            round(shadow_return_lcb, 6) if shadow_return_lcb is not None else None
        ),
        "authoritative_profit_factor": (
            round(authoritative_profit_factor, 6)
            if authoritative_profit_factor is not None
            else None
        ),
        "authoritative_gross_profit_return_pct": round(authoritative_gross_profit, 6),
        "authoritative_gross_loss_return_pct": round(authoritative_gross_loss, 6),
        "authoritative_worst_return_after_all_cost_pct": authoritative_worst,
        "authoritative_best_return_after_all_cost_pct": metric.get(
            "authoritative_best_return_pct"
        ),
        "authoritative_tail_loss_count": int(
            metric.get("authoritative_tail_loss_count") or 0
        ),
        "authoritative_evidence": list(metric.get("authoritative_evidence") or []),
        "authoritative_events": authoritative_events,
        "walk_forward": walk_forward,
        "market_regime_stability": regime_stability,
        "rolling_distribution": rolling_distribution,
        "top_symbols": [
            {"symbol": symbol, "count": count}
            for symbol, count in metric["symbols"].most_common(10)
        ],
        "promotion_ready": not bool(blockers),
        "promotion_blockers": blockers,
        "blockers": blockers,
        "blocked_reasons": blockers,
        "blocked_reason_counts": blocker_counts,
        "promotion_gate": {
            "requires_authoritative_return_distribution": True,
            "minimum_authoritative_return_lcb_pct": 0.0,
            "minimum_authoritative_profit_factor": 1.0,
            "actual_inference_count": int(metric.get("actual_inference_count") or 0),
            "direction_count": direction_count,
            "direction_hit_rate": round(hit_rate, 6),
            "avg_realized_return_pct": round(avg_realized, 6),
            "profit_factor": round(profit_factor, 6) if profit_factor is not None else None,
            "authoritative_avg_return_after_all_cost_pct": round(avg_authoritative, 6),
            "authoritative_profit_factor": (
                round(authoritative_profit_factor, 6)
                if authoritative_profit_factor is not None
                else None
            ),
            "worst_realized_return_pct": metric.get("worst_realized_return_pct"),
            "tail_loss_count": int(metric.get("tail_loss_count") or 0),
            "legacy_mixed_shadow_count": int(metric.get("legacy_mixed_shadow_count") or 0),
            "legacy_quarantined_count": int(metric.get("legacy_quarantined_count") or 0),
            "training_cost_policy": "per_event_live_spread_fee_and_funding_complete",
            "walk_forward_status": walk_forward.get("status"),
            "market_regime_status": regime_stability.get("status"),
            "rolling_distribution_status": rolling_distribution.get("status"),
        },
    }


def summarize_specialist_shadow_evaluation(
    rows: Sequence[Any],
    *,
    authoritative_trade_samples: Sequence[Any] | None = None,
) -> dict[str, Any]:
    metrics: dict[tuple[str, str], dict[str, Any]] = {}
    completed_count = 0
    eligible_count = 0
    skipped_reasons: Counter[str] = Counter()

    for row in rows:
        if _safe_str(_row_get(row, "status")).lower() != "completed":
            skipped_reasons["not_completed"] += 1
            continue
        long_return = _safe_float(_row_get(row, "long_return_pct"), None)
        short_return = _safe_float(_row_get(row, "short_return_pct"), None)
        if long_return is None or short_return is None:
            skipped_reasons["missing_realized_returns"] += 1
            continue
        completed_count += 1
        local_shadow = _local_shadow(row)
        if not local_shadow:
            skipped_reasons["missing_local_ai_tools_shadow"] += 1
            continue
        actual_side = _actual_best_side(row)
        symbol = _safe_str(_row_get(row, "symbol")) or _safe_str(_features(row).get("symbol"))
        has_eligible_specialist = False
        for tool_name in ("profit_prediction", "time_series_prediction", "sentiment_analysis"):
            tool = local_shadow.get(tool_name)
            if not isinstance(tool, dict):
                continue
            if tool_name == "profit_prediction" and _baseline_only_shadow(tool):
                skipped_reasons[f"{tool_name}_baseline_only_shadow"] += 1
                continue
            candidates = _tool_shadow_candidates(tool_name, tool)
            if tool_name == "profit_prediction" and not any(
                bool(candidate.get("actual_inference")) for candidate in candidates
            ):
                skipped_reasons[f"{tool_name}_non_specialist_shadow"] += 1
                continue
            if not candidates:
                continue
            has_eligible_specialist = True
            for candidate in candidates:
                model_name = _safe_str(candidate.get("model")) or _tool_model_name(tool_name, tool)
                key = (tool_name, model_name)
                metric = metrics.setdefault(key, _empty_metric(tool_name, model_name))
                metric["sample_count"] += 1
                if symbol:
                    metric["symbols"][symbol] += 1
                if not candidate.get("actual_inference"):
                    _record_fallback(metric, candidate, source="shadow")
                    metric["shadow_events"].append(
                        {
                            "evidence_source": "shadow_counterfactual",
                            "shadow_backtest_id": _row_get(row, "id"),
                            "decision_id": _row_get(row, "decision_id"),
                            "symbol": symbol,
                            "actual_inference": False,
                            "fallback_reason": _safe_str(candidate.get("fallback_reason"))
                            or "specialist_inference_unavailable",
                            "created_at": _iso_row_datetime(row, "created_at"),
                            "label_timestamp": _iso_row_datetime(row, "due_at"),
                            "market_regime": _market_regime(row),
                        }
                    )
                    continue
                legacy_mixed_shadow = bool(candidate.get("legacy_mixed_shadow"))
                if legacy_mixed_shadow:
                    metric["legacy_mixed_shadow_count"] += 1
                sequence_length = int(candidate.get("sequence_length") or 0)
                if legacy_mixed_shadow:
                    metric["legacy_quarantined_count"] += 1
                    continue
                metric["actual_inference_count"] += 1
                predicted_side = _safe_str(candidate.get("direction")).lower()
                gross_return = _actual_return_for_side(row, predicted_side)
                if predicted_side in {"long", "short"} and gross_return is not None:
                    cost_complete_return = _cost_complete_shadow_return(
                        row,
                        side=predicted_side,
                        gross_return_pct=float(gross_return),
                    )
                    if cost_complete_return is None:
                        skipped_reasons["shadow_execution_cost_incomplete"] += 1
                        continue
                    actual_return, execution_cost = cost_complete_return
                    metric["direction_count"] += 1
                    metric["realized_return_sum_pct"] += actual_return
                    if actual_return > 0:
                        metric["realized_gross_profit_pct"] += actual_return
                    elif actual_return < 0:
                        metric["realized_gross_loss_pct"] += abs(actual_return)
                    if actual_side == predicted_side:
                        metric["direction_hit_count"] += 1
                    elif actual_return < 0:
                        metric["false_signal_count"] += 1
                    _update_return_extrema(
                        metric,
                        actual_return,
                        worst_key="worst_realized_return_pct",
                        best_key="best_realized_return_pct",
                    )
                    _remember_worst_sample(
                        metric,
                        _compact_worst_sample(
                            row,
                            tool_name=tool_name,
                            model_name=model_name,
                            symbol=symbol,
                            predicted_side=predicted_side,
                            actual_side=actual_side,
                            actual_return=actual_return,
                            expected_return=_safe_float(
                                candidate.get("expected_return_pct"),
                                None,
                            ),
                            sequence_length=sequence_length,
                            legacy_mixed_shadow=legacy_mixed_shadow,
                        ),
                    )
                    metric["shadow_events"].append(
                        {
                            "evidence_source": "shadow_counterfactual_after_cost",
                            "shadow_backtest_id": _row_get(row, "id"),
                            "decision_id": _row_get(row, "decision_id"),
                            "symbol": symbol,
                            "predicted_side": predicted_side,
                            "actual_best_side": actual_side,
                            "gross_return_pct": round(float(gross_return), 6),
                            "execution_cost": execution_cost,
                            "net_return_after_all_cost_pct": round(actual_return, 6),
                            "created_at": _iso_row_datetime(row, "created_at"),
                            "label_timestamp": _iso_row_datetime(row, "due_at"),
                            "market_regime": _market_regime(row),
                        }
                    )
                expected = _safe_float(candidate.get("expected_return_pct"), None)
                if expected is not None:
                    metric["expected_return_sum_pct"] += expected
                    metric["expected_return_count"] += 1
                signal_score = _safe_float(candidate.get("signal_score"), None)
                if signal_score is not None:
                    metric["signal_score_sum"] += signal_score
                    metric["signal_score_count"] += 1
        if has_eligible_specialist:
            eligible_count += 1

    authoritative_input_count = 0
    authoritative_eligible_count = 0
    authoritative_skipped_reasons: Counter[str] = Counter()
    seen_lifecycle_keys: set[str] = set()
    for sample in authoritative_trade_samples or []:
        authoritative_input_count += 1
        if _safe_str(_row_get(sample, "source")) != "okx_position_history":
            authoritative_skipped_reasons["non_authoritative_source"] += 1
            continue
        lifecycle_key = _safe_str(_row_get(sample, "lifecycle_key"))
        if not lifecycle_key:
            authoritative_skipped_reasons["missing_lifecycle_key"] += 1
            continue
        if lifecycle_key in seen_lifecycle_keys:
            authoritative_skipped_reasons["duplicate_lifecycle_key"] += 1
            continue
        seen_lifecycle_keys.add(lifecycle_key)
        if not bool(_row_get(sample, "trade_fact_trusted")):
            reason = _safe_str(_row_get(sample, "trade_fact_trust_reason"))
            authoritative_skipped_reasons[reason or "untrusted_trade_fact"] += 1
            continue
        decision_id = int(_safe_float(_row_get(sample, "decision_id"), 0.0) or 0)
        if decision_id <= 0:
            authoritative_skipped_reasons["missing_exact_entry_order_decision_link"] += 1
            continue
        actual_position_side = _safe_str(_row_get(sample, "side")).lower()
        if actual_position_side not in {"long", "short"}:
            authoritative_skipped_reasons["missing_authoritative_position_side"] += 1
            continue
        authoritative_return = _safe_float(
            _row_get(sample, PROFIT_TRAINING_TARGET), None
        )
        if authoritative_return is None:
            authoritative_skipped_reasons["missing_authoritative_after_cost_return"] += 1
            continue
        local_tools = _authoritative_local_tools(sample)
        if not local_tools:
            authoritative_skipped_reasons["missing_linked_local_ai_tools_evidence"] += 1
            continue
        authoritative_eligible_count += 1
        symbol = _safe_str(_row_get(sample, "symbol"))
        regime = _authoritative_market_regime(sample)
        for tool_name in ("time_series_prediction", "sentiment_analysis"):
            tool = local_tools.get(tool_name)
            if not isinstance(tool, dict):
                continue
            for candidate in _tool_shadow_candidates(tool_name, tool):
                model_name = _safe_str(candidate.get("model"))
                if not model_name:
                    continue
                metric = metrics.setdefault(
                    (tool_name, model_name), _empty_metric(tool_name, model_name)
                )
                metric["authoritative_attempt_count"] += 1
                evidence = {
                    "evidence_source": "okx_position_history",
                    "okx_history_id": _row_get(sample, "id"),
                    "lifecycle_key": lifecycle_key,
                    "decision_id": decision_id,
                    "symbol": symbol,
                    "predicted_side": _safe_str(candidate.get("direction")).lower(),
                    "actual_position_side": actual_position_side,
                    "actual_inference": bool(candidate.get("actual_inference")),
                    "observed_position_net_return_after_all_cost_pct": round(
                        float(authoritative_return), 6
                    ),
                    "label_timestamp": _safe_str(_row_get(sample, "label_timestamp")),
                    "market_regime": regime,
                }
                if not candidate.get("actual_inference"):
                    _record_fallback(metric, candidate, source="authoritative")
                    evidence["label_usable"] = False
                    evidence["label_reason"] = "specialist_inference_unavailable"
                    evidence["fallback_reason"] = _safe_str(
                        candidate.get("fallback_reason")
                    ) or "specialist_inference_unavailable"
                    metric["authoritative_evidence"].append(evidence)
                    continue
                metric["authoritative_actual_inference_count"] += 1
                predicted_side = _safe_str(candidate.get("direction")).lower()
                if predicted_side != actual_position_side:
                    metric["authoritative_direction_mismatch_count"] += 1
                    evidence["label_usable"] = False
                    evidence["label_reason"] = "prediction_not_aligned_with_observed_position"
                    metric["authoritative_evidence"].append(evidence)
                    continue
                metric["authoritative_direction_aligned_count"] += 1
                metric["authoritative_return_sum_pct"] += float(authoritative_return)
                if float(authoritative_return) > 0:
                    metric["authoritative_gross_profit_pct"] += float(authoritative_return)
                elif float(authoritative_return) < 0:
                    metric["authoritative_gross_loss_pct"] += abs(
                        float(authoritative_return)
                    )
                _update_return_extrema(
                    metric,
                    float(authoritative_return),
                    worst_key="authoritative_worst_return_pct",
                    best_key="authoritative_best_return_pct",
                )
                evidence["label_usable"] = True
                evidence["label_reason"] = "prediction_matches_observed_position_side"
                evidence["net_return_after_all_cost_pct"] = round(float(authoritative_return), 6)
                metric["authoritative_evidence"].append(evidence)
                metric["authoritative_events"].append(evidence)

    model_rows = [_finalize_metric(metric) for metric in metrics.values()]
    model_rows.sort(
        key=lambda item: (
            bool(item.get("promotion_ready")),
            float(item.get("avg_realized_return_pct") or 0.0),
            int(item.get("actual_inference_count") or 0),
        ),
        reverse=True,
    )
    top_blockers = Counter()
    for row in model_rows:
        top_blockers.update(_safe_str(reason) for reason in _safe_list(row.get("promotion_blockers")))
    return {
        "ok": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "policy": "phase3_specialist_shadow_evaluation_v2",
        "promotion_flow": "candidate_to_shadow_to_canary_to_active",
        "completed_count": completed_count,
        "eligible_shadow_count": eligible_count,
        "authoritative_input_count": authoritative_input_count,
        "authoritative_eligible_count": authoritative_eligible_count,
        "authoritative_skipped_reasons": dict(authoritative_skipped_reasons),
        "model_count": len(model_rows),
        "models": model_rows,
        "skipped_reasons": dict(skipped_reasons),
        "summary": {
            "promotion_ready_count": sum(1 for row in model_rows if row.get("promotion_ready")),
            "blocked_count": sum(1 for row in model_rows if not row.get("promotion_ready")),
            "top_blocked_reasons": [
                {"reason": reason, "count": count}
                for reason, count in top_blockers.most_common(8)
                if reason
            ],
        },
        "promotion_gate": {
            "requires_authoritative_return_distribution": True,
            "minimum_authoritative_return_lcb_pct": 0.0,
            "minimum_authoritative_profit_factor": 1.0,
            "training_cost_policy": "per_event_live_spread_fee_and_funding_complete",
            "requires_walk_forward_stability": True,
            "requires_market_regime_stability": True,
            "requires_rolling_distribution_stability": True,
            "shadow_only_promotion_allowed": False,
            "requires_at_least_one_promotion_ready_model": True,
        },
    }


class SpecialistShadowEvaluationService:
    def __init__(self, session_context_factory: Any | None = None) -> None:
        self._session_context_factory = session_context_factory

    async def report(
        self,
        *,
        hours: int = DEFAULT_WINDOW_HOURS,
        authoritative_trade_samples: Sequence[Any] | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        from sqlalchemy import select
        from sqlalchemy.orm import load_only

        from db.session import get_read_session_ctx
        from models.learning import ShadowBacktest

        capped_hours = max(1, min(int(hours or DEFAULT_WINDOW_HOURS), 24 * 90))
        epoch_start = load_training_epoch_start()
        since = max(datetime.now(UTC) - timedelta(hours=capped_hours), epoch_start)
        since_naive = since.replace(tzinfo=None)
        selected_mode = (
            "live" if str(mode or "").lower() == "live" else "paper" if mode else None
        )
        filters = [
            ShadowBacktest.created_at >= since_naive,
            ShadowBacktest.status == "completed",
            ShadowBacktest.long_return_pct.is_not(None),
            ShadowBacktest.short_return_pct.is_not(None),
        ]
        if selected_mode is not None:
            filters.append(ShadowBacktest.execution_mode == selected_mode)
        session_factory = self._session_context_factory or get_read_session_ctx
        async with session_factory() as session:
            result = await session.execute(
                select(ShadowBacktest)
                .where(*filters)
                .options(
                    load_only(
                        ShadowBacktest.id,
                        ShadowBacktest.decision_id,
                        ShadowBacktest.status,
                        ShadowBacktest.symbol,
                        ShadowBacktest.feature_snapshot,
                        ShadowBacktest.long_return_pct,
                        ShadowBacktest.short_return_pct,
                        ShadowBacktest.best_action,
                        ShadowBacktest.horizon_minutes,
                        ShadowBacktest.due_at,
                        ShadowBacktest.created_at,
                        ShadowBacktest.updated_at,
                    )
                )
                .order_by(ShadowBacktest.id.desc())
            )
            rows = list(result.scalars().all())
        report = summarize_specialist_shadow_evaluation(
            rows,
            authoritative_trade_samples=authoritative_trade_samples,
        )
        report["window_hours"] = capped_hours
        report["execution_mode"] = selected_mode or "all"
        report["mode_filter_applied"] = selected_mode is not None
        report["query_policy"] = {
            "read_only": True,
            "ordered_by_primary_key": True,
            "db_time_filter": True,
            "completed_cost_label_filter": True,
            "necessary_columns_only": True,
            "event_statistics_use_full_window": True,
            "event_evidence_rows_bounded": True,
            "row_limit": None,
        }
        return report
