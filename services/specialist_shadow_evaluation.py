from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

DEFAULT_WINDOW_HOURS = 168
DEFAULT_LIMIT = 2000
MIN_PROMOTION_SHADOW_SAMPLES = 30
MIN_DIRECTION_HIT_RATE = 0.48
MIN_AVG_REALIZED_RETURN_PCT = 0.02
MAX_FALSE_SIGNAL_LOSS_PCT = -0.18
MIN_TIMESERIES_SEQUENCE_LENGTH = 30
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


def _actual_best_side(row: Any) -> str:
    best = _safe_str(_row_get(row, "best_action")).lower()
    if best in {"long", "short"}:
        return best
    long_return = _safe_float(_row_get(row, "long_return_pct"), None)
    short_return = _safe_float(_row_get(row, "short_return_pct"), None)
    if long_return is None or short_return is None:
        return ""
    if long_return == short_return:
        return "flat"
    return "long" if long_return > short_return else "short"


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
    for key in ("primary_shadow_result", "challenger_shadow_result"):
        result = professional.get(key)
        if not isinstance(result, dict) or not _result_actual_inference(result):
            continue
        model = _safe_str(result.get("model"))
        if not model:
            continue
        rows.append(
            {
                "tool": "time_series_prediction",
                "model": model,
                "direction": _result_direction(result),
                "expected_return_pct": _result_expected_return(result),
                "actual_inference": True,
                "sequence_length": int(_safe_float(result.get("sequence_length"), 0.0) or 0),
                "legacy_mixed_shadow": False,
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
        }
    ]


def _tool_shadow_candidates(tool_name: str, tool: dict[str, Any]) -> list[dict[str, Any]]:
    if tool_name == "time_series_prediction":
        rows = _timeseries_shadow_candidates(tool)
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
        }
    ]


def _empty_metric(tool_name: str, model_name: str) -> dict[str, Any]:
    return {
        "tool": tool_name,
        "model": model_name,
        "sample_count": 0,
        "actual_inference_count": 0,
        "direction_count": 0,
        "direction_hit_count": 0,
        "false_signal_count": 0,
        "realized_return_sum_pct": 0.0,
        "expected_return_sum_pct": 0.0,
        "expected_return_count": 0,
        "worst_realized_return_pct": None,
        "best_realized_return_pct": None,
        "symbols": Counter(),
        "tail_loss_count": 0,
        "tail_loss_symbols": Counter(),
        "worst_samples": [],
        "blockers": [],
        "sequence_too_short_count": 0,
        "legacy_mixed_shadow_count": 0,
        "legacy_quarantined_count": 0,
        "legacy_sequence_too_short_count": 0,
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
        "long_return_pct": _safe_float(_row_get(row, "long_return_pct"), None),
        "short_return_pct": _safe_float(_row_get(row, "short_return_pct"), None),
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


def _finalize_metric(metric: dict[str, Any]) -> dict[str, Any]:
    direction_count = int(metric.get("direction_count") or 0)
    expected_count = int(metric.get("expected_return_count") or 0)
    realized_sum = float(metric.get("realized_return_sum_pct") or 0.0)
    expected_sum = float(metric.get("expected_return_sum_pct") or 0.0)
    hit_rate = float(metric.get("direction_hit_count") or 0) / max(direction_count, 1)
    avg_realized = realized_sum / max(direction_count, 1)
    avg_expected = expected_sum / max(expected_count, 1)
    blockers = []
    if int(metric.get("actual_inference_count") or 0) < MIN_PROMOTION_SHADOW_SAMPLES:
        blockers.append("specialist_shadow_sample_floor_not_met")
    if direction_count >= MIN_PROMOTION_SHADOW_SAMPLES and hit_rate < MIN_DIRECTION_HIT_RATE:
        blockers.append("direction_hit_rate_below_floor")
    if direction_count >= MIN_PROMOTION_SHADOW_SAMPLES and avg_realized < MIN_AVG_REALIZED_RETURN_PCT:
        blockers.append("avg_realized_return_below_floor")
    worst = metric.get("worst_realized_return_pct")
    if worst is not None and float(worst) <= MAX_FALSE_SIGNAL_LOSS_PCT:
        blockers.append("false_signal_loss_exceeds_floor")
    if int(metric.get("sequence_too_short_count") or 0) > 0:
        blockers.append("timeseries_sequence_too_short_for_promotion")
    blocker_counts = dict(Counter(blockers))
    return {
        "tool": metric["tool"],
        "model": metric["model"],
        "sample_count": int(metric.get("sample_count") or 0),
        "actual_inference_count": int(metric.get("actual_inference_count") or 0),
        "direction_count": direction_count,
        "direction_hit_count": int(metric.get("direction_hit_count") or 0),
        "direction_hit_rate": round(hit_rate, 6),
        "false_signal_count": int(metric.get("false_signal_count") or 0),
        "avg_realized_return_pct": round(avg_realized, 6),
        "avg_expected_return_pct": round(avg_expected, 6),
        "worst_realized_return_pct": metric.get("worst_realized_return_pct"),
        "best_realized_return_pct": metric.get("best_realized_return_pct"),
        "sequence_too_short_count": int(metric.get("sequence_too_short_count") or 0),
        "legacy_mixed_shadow_count": int(metric.get("legacy_mixed_shadow_count") or 0),
        "legacy_quarantined_count": int(metric.get("legacy_quarantined_count") or 0),
        "legacy_sequence_too_short_count": int(
            metric.get("legacy_sequence_too_short_count") or 0
        ),
        "tail_loss_count": int(metric.get("tail_loss_count") or 0),
        "tail_loss_symbols": [
            {"symbol": symbol, "count": count}
            for symbol, count in metric["tail_loss_symbols"].most_common(10)
        ],
        "worst_samples": list(metric.get("worst_samples") or [])[:MAX_WORST_SAMPLE_COUNT],
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
            "minimum_actual_inference_samples": MIN_PROMOTION_SHADOW_SAMPLES,
            "minimum_direction_hit_rate": MIN_DIRECTION_HIT_RATE,
            "minimum_avg_realized_return_pct": MIN_AVG_REALIZED_RETURN_PCT,
            "max_false_signal_loss_pct": MAX_FALSE_SIGNAL_LOSS_PCT,
            "actual_inference_count": int(metric.get("actual_inference_count") or 0),
            "direction_count": direction_count,
            "direction_hit_rate": round(hit_rate, 6),
            "avg_realized_return_pct": round(avg_realized, 6),
            "worst_realized_return_pct": metric.get("worst_realized_return_pct"),
            "tail_loss_count": int(metric.get("tail_loss_count") or 0),
            "minimum_timeseries_sequence_length": MIN_TIMESERIES_SEQUENCE_LENGTH,
            "sequence_too_short_count": int(metric.get("sequence_too_short_count") or 0),
            "legacy_mixed_shadow_count": int(metric.get("legacy_mixed_shadow_count") or 0),
            "legacy_quarantined_count": int(metric.get("legacy_quarantined_count") or 0),
            "legacy_sequence_too_short_count": int(
                metric.get("legacy_sequence_too_short_count") or 0
            ),
        },
    }


def summarize_specialist_shadow_evaluation(rows: Sequence[Any]) -> dict[str, Any]:
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
            if _baseline_only_shadow(tool):
                skipped_reasons[f"{tool_name}_baseline_only_shadow"] += 1
                continue
            if not _actual_inference(tool):
                skipped_reasons[f"{tool_name}_non_specialist_shadow"] += 1
                continue
            has_eligible_specialist = True
            for candidate in _tool_shadow_candidates(tool_name, tool):
                if not candidate.get("actual_inference"):
                    continue
                model_name = _safe_str(candidate.get("model")) or _tool_model_name(tool_name, tool)
                key = (tool_name, model_name)
                metric = metrics.setdefault(key, _empty_metric(tool_name, model_name))
                metric["sample_count"] += 1
                if symbol:
                    metric["symbols"][symbol] += 1
                legacy_mixed_shadow = bool(candidate.get("legacy_mixed_shadow"))
                if legacy_mixed_shadow:
                    metric["legacy_mixed_shadow_count"] += 1
                sequence_length = int(candidate.get("sequence_length") or 0)
                sequence_too_short = (
                    tool_name == "time_series_prediction"
                    and sequence_length < MIN_TIMESERIES_SEQUENCE_LENGTH
                )
                if sequence_too_short:
                    metric["legacy_sequence_too_short_count"] += 1
                if legacy_mixed_shadow or sequence_too_short:
                    metric["legacy_quarantined_count"] += 1
                    continue
                metric["actual_inference_count"] += 1
                predicted_side = _safe_str(candidate.get("direction")).lower()
                actual_return = _actual_return_for_side(row, predicted_side)
                if predicted_side in {"long", "short"} and actual_return is not None:
                    metric["direction_count"] += 1
                    metric["realized_return_sum_pct"] += actual_return
                    if actual_side == predicted_side:
                        metric["direction_hit_count"] += 1
                    elif actual_return < 0:
                        metric["false_signal_count"] += 1
                    worst = metric.get("worst_realized_return_pct")
                    best = metric.get("best_realized_return_pct")
                    metric["worst_realized_return_pct"] = (
                        round(actual_return, 6)
                        if worst is None
                        else round(min(float(worst), actual_return), 6)
                    )
                    metric["best_realized_return_pct"] = (
                        round(actual_return, 6)
                        if best is None
                        else round(max(float(best), actual_return), 6)
                    )
                    if actual_return <= MAX_FALSE_SIGNAL_LOSS_PCT:
                        metric["tail_loss_count"] += 1
                        if symbol:
                            metric["tail_loss_symbols"][symbol] += 1
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
                expected = _safe_float(candidate.get("expected_return_pct"), None)
                if expected is not None:
                    metric["expected_return_sum_pct"] += expected
                    metric["expected_return_count"] += 1
        if has_eligible_specialist:
            eligible_count += 1

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
        "policy": "phase3_specialist_shadow_evaluation_v1",
        "live_mutation": False,
        "promotion_flow": "shadow_to_canary_to_live",
        "completed_count": completed_count,
        "eligible_shadow_count": eligible_count,
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
            "minimum_actual_inference_samples": MIN_PROMOTION_SHADOW_SAMPLES,
            "minimum_direction_hit_rate": MIN_DIRECTION_HIT_RATE,
            "minimum_avg_realized_return_pct": MIN_AVG_REALIZED_RETURN_PCT,
            "max_false_signal_loss_pct": MAX_FALSE_SIGNAL_LOSS_PCT,
            "minimum_timeseries_sequence_length": MIN_TIMESERIES_SEQUENCE_LENGTH,
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
        limit: int = DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        from sqlalchemy import select

        from db.session import get_read_session_ctx
        from models.learning import ShadowBacktest

        capped_hours = max(1, min(int(hours or DEFAULT_WINDOW_HOURS), 24 * 90))
        capped_limit = max(50, min(int(limit or DEFAULT_LIMIT), 20000))
        since = datetime.now(UTC) - timedelta(hours=capped_hours)
        session_factory = self._session_context_factory or get_read_session_ctx
        async with session_factory() as session:
            result = await session.execute(
                select(ShadowBacktest).order_by(ShadowBacktest.id.desc()).limit(capped_limit)
            )
            rows = [row for row in result.scalars().all() if _row_in_window(row, since)]
        report = summarize_specialist_shadow_evaluation(rows)
        report["window_hours"] = capped_hours
        report["query_policy"] = {
            "read_only": True,
            "ordered_by_primary_key": True,
            "db_time_filter": False,
            "row_limit": capped_limit,
        }
        return report
