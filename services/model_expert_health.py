from __future__ import annotations

import math
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

from db.session import get_read_session_ctx
from models.decision import AIDecision
from models.learning import ExpertMemory, ShadowBacktest, StrategyLearningEvent
from services.strategy_learning import REQUIRED_ENTRY_EXPERTS, UNTRUSTED_EXPERT_STATUSES

MODEL_COMPONENTS = {
    "decision_maker",
    "ml_profit_model",
    "server_profit_model",
    "timeseries_model",
    "sentiment_model",
    "local_ai_tools",
}
DEFAULT_WINDOWS_HOURS = (24, 72)
MIN_DECISION_SAMPLES = 3
HIGH_JSON_ERROR_RATE = 0.25
HIGH_NO_RETURN_RATE = 0.25
NEGATIVE_PNL_DEGRADE_PCT = -0.01


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _created_at(row: Any) -> datetime | None:
    value = getattr(row, "created_at", None)
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _component_type(name: str) -> str:
    if name in REQUIRED_ENTRY_EXPERTS or name.endswith("_expert"):
        return "expert"
    return "model" if name in MODEL_COMPONENTS or name.endswith("_model") else "component"


def _raw(row: Any) -> dict[str, Any]:
    return _safe_dict(getattr(row, "raw_llm_response", None))


def _timing_rows(row: Any) -> list[dict[str, Any]]:
    raw = _raw(row)
    timings = raw.get("model_timings")
    if not isinstance(timings, list):
        timings = raw.get("_model_timings")
    return [item for item in _safe_list(timings) if isinstance(item, dict)]


def _expert_rows(row: Any) -> list[dict[str, Any]]:
    raw = _raw(row)
    experts = raw.get("experts")
    if not isinstance(experts, list):
        experts = raw.get("opinions")
    return [item for item in _safe_list(experts) if isinstance(item, dict)]


def _component_name(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return "unknown"


def _action(value: Any) -> str:
    text = str(value or "").lower().strip()
    if text in {"long", "short", "hold", "close_long", "close_short"}:
        return text
    return "unknown"


def _decision_action(row: Any) -> str:
    return _action(getattr(row, "action", None))


def _shadow_best_actions(shadows: list[Any]) -> dict[int, str]:
    best: dict[int, str] = {}
    for shadow in shadows:
        if str(getattr(shadow, "status", "") or "").lower() != "completed":
            continue
        decision_id = getattr(shadow, "decision_id", None)
        if decision_id is None:
            continue
        action = _action(getattr(shadow, "best_action", None))
        if action in {"long", "short", "hold"}:
            best[int(decision_id)] = action
    return best


def _is_json_error(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "").lower().strip()
    reason = str(row.get("reason") or row.get("error") or "").lower()
    return bool(
        status in {"failed", "invalid", "batch_fallback", "partial_batch_fallback"}
        or "json" in reason
        or "extract valid" in reason
        or "parse" in reason
    )


def _is_no_return(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "").lower().strip()
    return status in UNTRUSTED_EXPERT_STATUSES or status in {
        "failed",
        "invalid",
        "timeout",
        "timeout_fallback",
        "batch_fallback",
        "partial_batch_fallback",
    }


def _add_counter(bucket: dict[str, Any], key: str, amount: float = 1.0) -> None:
    bucket[key] = _safe_float(bucket.get(key), 0.0) + amount


def _empty_window() -> dict[str, Any]:
    return {
        "participation_count": 0,
        "recommendation_count": 0,
        "adopted_count": 0,
        "adopted_net_pnl_pct": 0.0,
        "wrong_recommendation_count": 0,
        "json_error_count": 0,
        "no_return_count": 0,
        "duration_total_sec": 0.0,
        "duration_count": 0,
        "action_counts": Counter(),
        "provider_models": Counter(),
    }


def _round_window(bucket: dict[str, Any]) -> dict[str, Any]:
    participation = int(bucket["participation_count"])
    adopted = int(bucket["adopted_count"])
    duration_count = int(bucket["duration_count"])
    wrong = int(bucket["wrong_recommendation_count"])
    return {
        "participation_count": participation,
        "recommendation_count": int(bucket["recommendation_count"]),
        "adopted_count": adopted,
        "adopted_net_pnl_pct": round(_safe_float(bucket["adopted_net_pnl_pct"]), 6),
        "avg_adopted_net_pnl_pct": (
            round(_safe_float(bucket["adopted_net_pnl_pct"]) / adopted, 6) if adopted else 0.0
        ),
        "wrong_recommendation_count": wrong,
        "wrong_recommendation_rate": round(wrong / participation, 4) if participation else 0.0,
        "json_error_count": int(bucket["json_error_count"]),
        "json_error_rate": (
            round(int(bucket["json_error_count"]) / participation, 4) if participation else 0.0
        ),
        "no_return_count": int(bucket["no_return_count"]),
        "no_return_rate": (
            round(int(bucket["no_return_count"]) / participation, 4) if participation else 0.0
        ),
        "avg_duration_sec": (
            round(_safe_float(bucket["duration_total_sec"]) / duration_count, 6)
            if duration_count
            else 0.0
        ),
        "action_counts": dict(bucket["action_counts"]),
        "provider_models": dict(bucket["provider_models"]),
    }


def _empty_component(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": _component_type(name),
        "windows": {f"{hours}h": _empty_window() for hours in DEFAULT_WINDOWS_HOURS},
        "memory": {"evidence_count": 0, "success_count": 0, "failure_count": 0},
    }


def _window_keys_for(row_time: datetime | None, now: datetime) -> list[str]:
    if row_time is None:
        return []
    age_hours = max((now - row_time).total_seconds() / 3600.0, 0.0)
    return [f"{hours}h" for hours in DEFAULT_WINDOWS_HOURS if age_hours <= hours]


def _recommendation_state(w24: dict[str, Any]) -> tuple[str, str, list[str]]:
    reasons: list[str] = []
    samples = int(w24["participation_count"])
    adopted = int(w24["adopted_count"])
    if samples < MIN_DECISION_SAMPLES:
        reasons.append("insufficient_samples")
        return "shadow_only", "observing", reasons
    pnl = _safe_float(w24["adopted_net_pnl_pct"])
    json_error_rate = _safe_float(w24["json_error_rate"])
    no_return_rate = _safe_float(w24["no_return_rate"])
    wrong_rate = _safe_float(w24["wrong_recommendation_rate"])
    if json_error_rate >= HIGH_JSON_ERROR_RATE:
        reasons.append("json_error_rate_high")
    if no_return_rate >= HIGH_NO_RETURN_RATE:
        reasons.append("no_return_rate_high")
    if adopted and pnl <= NEGATIVE_PNL_DEGRADE_PCT:
        reasons.append("negative_adopted_pnl")
    if wrong_rate >= 0.5:
        reasons.append("wrong_recommendation_rate_high")
    if "json_error_rate_high" in reasons and "no_return_rate_high" in reasons:
        return "disable", "needs_review", reasons
    if reasons:
        if "negative_adopted_pnl" in reasons or "wrong_recommendation_rate_high" in reasons:
            return "reduce", "needs_review", reasons
        return "shadow_only", "needs_review", reasons
    if adopted and pnl > 0:
        reasons.append("positive_adopted_pnl")
        return "keep", "supported", reasons
    reasons.append("no_adopted_outcome_yet")
    return "shadow_only", "observing", reasons


def _extract_component_rows(decision: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for timing in _timing_rows(decision):
        name = _component_name(timing, "name", "expert_name", "model_name")
        if name == "unknown":
            continue
        seen.add(name)
        rows.append(
            {
                "name": name,
                "action": _action(timing.get("action")),
                "duration_sec": _safe_float(timing.get("duration_sec"), 0.0),
                "provider_model": str(timing.get("provider_model") or ""),
                "json_error": _is_json_error(timing),
                "no_return": _is_no_return(timing),
            }
        )
    for expert in _expert_rows(decision):
        name = _component_name(expert, "expert_name", "model_name", "name")
        if name == "unknown" or name in seen:
            continue
        rows.append(
            {
                "name": name,
                "action": _action(expert.get("action")),
                "duration_sec": 0.0,
                "provider_model": str(expert.get("provider_model") or ""),
                "json_error": False,
                "no_return": False,
            }
        )
    raw = _raw(decision)
    if raw.get("ml_signal") or raw.get("local_ml_signal"):
        rows.append({"name": "ml_profit_model", "action": "unknown"})
    if raw.get("local_ai_tools"):
        rows.append({"name": "local_ai_tools", "action": "unknown"})
    return rows


def summarize_model_expert_health(
    decisions: list[Any],
    shadows: list[Any] | None = None,
    memories: list[Any] | None = None,
    strategy_events: list[Any] | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a read-only health report for models/experts from persisted evidence."""

    current = now or datetime.now(UTC)
    shadow_best = _shadow_best_actions(list(shadows or []))
    components: dict[str, dict[str, Any]] = {}

    for decision in decisions:
        row_time = _created_at(decision)
        window_keys = _window_keys_for(row_time, current)
        if not window_keys:
            continue
        decision_action = _decision_action(decision)
        adopted = bool(getattr(decision, "was_executed", False)) and decision_action in {
            "long",
            "short",
        }
        pnl = _safe_float(getattr(decision, "outcome_pnl_pct", None), 0.0)
        best_action = shadow_best.get(int(getattr(decision, "id", 0) or 0))
        for component_row in _extract_component_rows(decision):
            name = str(component_row.get("name") or "unknown")
            if not name or name == "unknown":
                continue
            component = components.setdefault(name, _empty_component(name))
            rec_action = _action(component_row.get("action"))
            for window_key in window_keys:
                bucket = component["windows"][window_key]
                _add_counter(bucket, "participation_count")
                if rec_action != "unknown":
                    _add_counter(bucket, "recommendation_count")
                    bucket["action_counts"][rec_action] += 1
                if adopted and rec_action == decision_action:
                    _add_counter(bucket, "adopted_count")
                    _add_counter(bucket, "adopted_net_pnl_pct", pnl)
                if best_action in {"long", "short"} and rec_action in {"long", "short", "hold"}:
                    wrong = rec_action != best_action
                    if wrong:
                        _add_counter(bucket, "wrong_recommendation_count")
                if bool(component_row.get("json_error")):
                    _add_counter(bucket, "json_error_count")
                if bool(component_row.get("no_return")):
                    _add_counter(bucket, "no_return_count")
                duration = _safe_float(component_row.get("duration_sec"), 0.0)
                if duration > 0:
                    _add_counter(bucket, "duration_total_sec", duration)
                    _add_counter(bucket, "duration_count")
                provider = str(component_row.get("provider_model") or "").strip()
                if provider:
                    bucket["provider_models"][provider] += 1

    for memory in memories or []:
        name = str(getattr(memory, "expert_name", "") or "").strip()
        if not name:
            continue
        component = components.setdefault(name, _empty_component(name))
        bucket = component["memory"]
        bucket["evidence_count"] += int(getattr(memory, "evidence_count", 0) or 0)
        bucket["success_count"] += int(getattr(memory, "success_count", 0) or 0)
        bucket["failure_count"] += int(getattr(memory, "failure_count", 0) or 0)

    for event in strategy_events or []:
        attribution = _safe_dict(getattr(event, "attribution", None))
        for key, value in attribution.items():
            if not isinstance(value, dict):
                continue
            name = str(value.get("source") or key or "").strip()
            if name:
                components.setdefault(name, _empty_component(name))

    final_components: dict[str, Any] = {}
    state_counts: Counter[str] = Counter()
    for name in sorted(components):
        item = components[name]
        rounded_windows = {key: _round_window(value) for key, value in item["windows"].items()}
        state, evidence_state, reasons = _recommendation_state(rounded_windows["24h"])
        state_counts[state] += 1
        final_components[name] = {
            "name": name,
            "type": item["type"],
            "recommended_state": state,
            "evidence_state": evidence_state,
            "state_reasons": reasons,
            "windows": rounded_windows,
            "stability": {
                "json_error_rate": rounded_windows["24h"]["json_error_rate"],
                "no_return_rate": rounded_windows["24h"]["no_return_rate"],
                "avg_duration_sec": rounded_windows["24h"]["avg_duration_sec"],
            },
            "memory": item["memory"],
        }

    return {
        "audit_only": True,
        "live_weight_mutation": False,
        "windows_hours": list(DEFAULT_WINDOWS_HOURS),
        "generated_at": current.isoformat(),
        "summary": {
            "components": len(final_components),
            "decisions": len(decisions),
            "shadows": len(shadows or []),
            "recommended_state_counts": dict(state_counts),
        },
        "components": final_components,
        "handling_states": ["keep", "reduce", "shadow_only", "disable", "replace", "add_candidate"],
    }


class ModelExpertHealthService:
    def __init__(self, session_context_factory: Any = get_read_session_ctx) -> None:
        self._session_context_factory = session_context_factory

    async def report(self, *, hours: int = 72, limit: int = 1000) -> dict[str, Any]:
        from sqlalchemy import select

        capped_hours = max(1, min(int(hours or 72), 168))
        capped_limit = max(50, min(int(limit or 1000), 5000))
        since = datetime.now(UTC) - timedelta(hours=capped_hours)
        async with self._session_context_factory() as session:
            decisions_result = await session.execute(
                select(AIDecision)
                .where(AIDecision.created_at >= since)
                .order_by(AIDecision.created_at.desc())
                .limit(capped_limit)
            )
            shadows_result = await session.execute(
                select(ShadowBacktest)
                .where(ShadowBacktest.created_at >= since)
                .order_by(ShadowBacktest.created_at.desc())
                .limit(capped_limit)
            )
            memories_result = await session.execute(
                select(ExpertMemory)
                .order_by(
                    ExpertMemory.updated_at.desc().nullslast(), ExpertMemory.created_at.desc()
                )
                .limit(500)
            )
            events_result = await session.execute(
                select(StrategyLearningEvent)
                .where(StrategyLearningEvent.created_at >= since)
                .order_by(StrategyLearningEvent.created_at.desc())
                .limit(capped_limit)
            )
        return summarize_model_expert_health(
            list(decisions_result.scalars().all()),
            list(shadows_result.scalars().all()),
            list(memories_result.scalars().all()),
            list(events_result.scalars().all()),
        )
