"""Strategy learning, candidate generation, and runtime scheduling.

The service turns existing records into structured feedback and a bounded
strategy profile.  It does not execute generated code and it does not call an
LLM directly; profiles are controlled parameter sets that can be scored,
shadow-validated, probed, scheduled, disabled, and rolled back.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import structlog
from sqlalchemy import select

from config.settings import DEFAULT_MAX_OPEN_POSITIONS_PER_MODEL, ENSEMBLE_TRADER_NAME, settings
from core.model_runtime import apply_non_thinking_request_controls, completion_token_limit
from core.safe_output import safe_error_text, safe_response_error_text
from db.session import get_read_session_ctx, get_session_ctx
from models.decision import AIDecision
from models.learning import (
    ExpertMemory,
    ShadowBacktest,
    StrategyLearningEvent,
    StrategyProfileSnapshot,
    TradeReflection,
)
from models.trade import Order, Position
from services.entry_priority import MIN_ENTRY_OPPORTUNITY_SCORE
from services.entry_strategy_mode import PORTFOLIO_ROSTER_FILL_MARKET_SYMBOL_MIN
from services.execution_result_classifier import ExecutionResultClassifier
from services.manual_close_marker import is_manual_close_order, position_has_manual_close_order
from services.okx_error_classifier import is_okx_temporary_service_error
from services.phase3_boundary import PHASE3_CLEAN_START_UTC
from services.position_open_time import parse_position_time, position_open_time
from services.position_quality import PositionQualityScorer
from services.profit_first_ranking import ProfitFirstRankingService
from services.runtime_entry_filters import RuntimeEntryFilters, default_entry_filters
from services.shadow_missed_opportunity_closed_loop import summarize_shadow_missed_opportunities
from services.text_integrity import sanitize_runtime_text
from services.trade_fact_trust import closed_position_trade_fact_untrusted_reason
from services.trading_params import DEFAULT_TRADING_PARAMS
from web_dashboard.api.text_sanitize import sanitize_payload, sanitize_text

logger = structlog.get_logger(__name__)
OKX_AUTHORITATIVE_LEDGER_MODEL = "okx_authoritative_sync"
EXECUTION_LEDGER_MODEL_NAMES = (ENSEMBLE_TRADER_NAME, OKX_AUTHORITATIVE_LEDGER_MODEL)

UNTRUSTED_EXPERT_STATUSES = {
    "batch_fallback",
    "partial_batch_fallback",
    "circuit_breaker_fallback",
    "fast_prefilter",
    "failed",
    "invalid",
    "timeout",
    "timeout_fallback",
    "independent_provider_fallback",
    "independent_provider_failed",
}
REQUIRED_ENTRY_EXPERTS = {
    "trend_expert",
    "momentum_expert",
    "sentiment_expert",
    "position_expert",
    "risk_expert",
}
CORE_ENTRY_EXPERTS = {"trend_expert", "momentum_expert", "risk_expert"}
NON_CORE_ENTRY_EXPERTS = REQUIRED_ENTRY_EXPERTS - CORE_ENTRY_EXPERTS
STRATEGY_LEARNING_PARAMS = DEFAULT_TRADING_PARAMS.strategy_learning
ENTRY_RISK_SIZING_PARAMS = DEFAULT_TRADING_PARAMS.entry_risk_sizing
DEFAULT_MIN_TRADE_TARGET_FALLBACK = STRATEGY_LEARNING_PARAMS.min_trade_count_target_baseline
DEFAULT_LOOKBACK_HOURS = STRATEGY_LEARNING_PARAMS.default_lookback_hours
STATE_FILE_NAME = "strategy_learning_state.json"
PROFILE_SNAPSHOT_MIN_INTERVAL_SECONDS = 600
LLM_CANDIDATE_CACHE_SECONDS = 6 * 60 * 60
LLM_CANDIDATE_MAX_COUNT = 2
LLM_CANDIDATE_PROMPT_VERSION = 4
LLM_CANDIDATE_PROMPT_MAX_CHARS = 9000
LLM_CANDIDATE_FAILURE_RETRY_SECONDS = 300
LLM_CANDIDATE_ERROR_TIMEOUT = "timeout"
LLM_CANDIDATE_ERROR_HTTP = "http_error"
LLM_CANDIDATE_ERROR_INVALID_JSON = "invalid_json"
LLM_CANDIDATE_ERROR_EMPTY = "empty_candidates"
LLM_CANDIDATE_ERROR_UNKNOWN = "unknown"
AUTO_DISABLED_PROFILE_RECONSIDER_SECONDS = 6 * 60 * 60
NON_ATTRIBUTABLE_EVENT_TYPES = {
    "manual_close",
    "position_snapshot",
    "position_sync",
    "scheduler_tick",
}
EXECUTION_REASON_CLASSIFIER = ExecutionResultClassifier()
ALLOWED_CANDIDATE_PARAM_KEYS = {
    "global_min_score_delta",
    "position_size_multiplier",
    "probe_fraction",
    "min_trade_count_target",
    "expert_integrity_mode",
    "max_probe_size_pct",
    "fallback_tolerance",
    "loss_exit_aggressiveness",
    "full_position_release",
    "position_review_priority_boost",
    "release_losing_positions_first",
    "winner_hold_extension",
    "profit_lock_min_usdt_multiplier",
    "winner_hold_dynamic",
    "payoff_repair_intensity",
    "pullback_lock_enabled",
    "side_overrides",
    "side_weights",
}
BOUNDED_FLOAT_PARAM_RANGES = {
    "global_min_score_delta": (-0.25, 0.35),
    "position_size_multiplier": (0.10, 1.25),
    "probe_fraction": (0.0, 0.10),
    "max_probe_size_pct": (0.0, ENTRY_RISK_SIZING_PARAMS.strategy_probe_cap_max_pct),
    "position_review_priority_boost": (0.70, 1.80),
    "profit_lock_min_usdt_multiplier": (0.80, 1.80),
    "payoff_repair_intensity": (0.0, 1.0),
}
ALLOWED_EXPERT_INTEGRITY_MODES = {
    "strict_all_required",
    "balanced_probe_allow_one_non_core_missing",
    "core_experts_required_probe_only",
}
ALLOWED_AGGRESSIVENESS = {"low", "normal", "high"}
ALLOWED_WINNER_HOLD = {"normal", "high"}
CONSUMED_RUNTIME_PARAM_KEYS = {
    "global_min_score_delta",
    "position_size_multiplier",
    "probe_fraction",
    "max_probe_size_pct",
    "expert_integrity_mode",
    "fallback_tolerance",
    "loss_exit_aggressiveness",
    "full_position_release",
    "position_review_priority_boost",
    "release_losing_positions_first",
    "winner_hold_extension",
    "profit_lock_min_usdt_multiplier",
    "winner_hold_dynamic",
    "payoff_repair_intensity",
    "pullback_lock_enabled",
    "side_overrides",
    "side_weights",
}


def _candidate_param_consumption(params: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(str(key) for key in _safe_dict(params))
    consumed = sorted(key for key in keys if key in CONSUMED_RUNTIME_PARAM_KEYS)
    unused = sorted(key for key in keys if key not in CONSUMED_RUNTIME_PARAM_KEYS)
    return {
        "consumed_runtime_params": consumed,
        "unused_runtime_params": unused,
        "has_consumed_runtime_params": bool(consumed),
    }


def default_min_trade_target() -> int:
    """Return the advisory training-sample target used by strategy learning."""

    params = STRATEGY_LEARNING_PARAMS
    return max(
        params.min_trade_count_target_min,
        min(
            _safe_int(
                getattr(
                    settings,
                    "strategy_learning_min_trade_count_target",
                    params.min_trade_count_target_baseline,
                ),
                params.min_trade_count_target_baseline,
            ),
            params.min_trade_count_target_settings_cap,
        ),
    )


def learning_trade_count_target(
    *,
    window_hours: int,
    market_scans: int = 0,
    entry_signals: int = 0,
    reflection_count: int = 0,
    shadow_opportunities: int = 0,
) -> int:
    """Return a dynamic, advisory sample target for strategy-confidence scoring."""

    params = STRATEGY_LEARNING_PARAMS
    baseline = default_min_trade_target()
    window_factor = (
        max(0, min(int(window_hours or DEFAULT_LOOKBACK_HOURS), params.max_lookback_hours))
        / params.dynamic_trade_target_reference_hours
    )
    activity_target = math.ceil(
        max(
            int(entry_signals or 0) * params.entry_signal_target_ratio,
            int(shadow_opportunities or 0) * params.shadow_opportunity_target_ratio,
        )
    )
    review_target = math.ceil(max(int(reflection_count or 0), 0) * params.reflection_target_ratio)
    scan_target = math.ceil(max(int(market_scans or 0), 0) * params.market_scan_target_ratio)
    dynamic_target = max(
        math.ceil(
            baseline
            * min(
                max(window_factor, params.dynamic_trade_target_min_window_factor),
                params.dynamic_trade_target_max_window_factor,
            )
        ),
        activity_target,
        review_target,
        scan_target,
    )
    return max(
        params.dynamic_trade_target_min,
        min(dynamic_target, params.dynamic_trade_target_max),
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _profit_first_feedback_can_influence_context(value: dict[str, Any]) -> bool:
    feedback = _safe_dict(value)
    return bool(
        feedback
        and feedback.get("can_influence_strategy_context")
        and not feedback.get("live_mutation")
        and not feedback.get("live_weight_mutation")
        and not feedback.get("live_sizing_mutation")
        and not feedback.get("can_submit_orders")
        and not feedback.get("can_increase_live_size")
    )


def _merge_profit_first_side_weights(
    base_weights: dict[str, Any],
    profit_first_feedback: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side in ("long", "short"):
        if side in base_weights:
            result[side] = round(_clamp(_safe_float(base_weights.get(side), 1.0), 0.25, 1.40), 6)
    if not _profit_first_feedback_can_influence_context(profit_first_feedback):
        return result
    feedback_weights = _safe_dict(profit_first_feedback.get("side_weights"))
    for side in ("long", "short"):
        if side not in feedback_weights:
            continue
        base = _safe_float(result.get(side), 1.0)
        feedback_weight = _clamp(_safe_float(feedback_weights.get(side), 1.0), 0.45, 1.12)
        result[side] = round(_clamp(base * feedback_weight, 0.25, 1.40), 6)
    return result


def _compact_profit_first_runtime_feedback(value: dict[str, Any]) -> dict[str, Any]:
    feedback = _safe_dict(value)
    if not feedback:
        return {}
    objective_basis = _safe_dict(feedback.get("objective_basis"))
    exit_reference = _safe_dict(feedback.get("exit_plan_reference"))
    local_ml = _safe_dict(feedback.get("local_ml_live_influence"))
    acceptance = _safe_dict(feedback.get("profit_acceptance"))
    missed = _safe_dict(feedback.get("missed_opportunity_feedback"))
    side_feedback = {
        side: {
            "count": row.get("count"),
            "realized_net_pnl": row.get("realized_net_pnl"),
            "profit_factor": row.get("profit_factor"),
            "recommended_stage": row.get("recommended_stage"),
            "weight_multiplier": row.get("weight_multiplier"),
            "hard_ban": bool(row.get("hard_ban")),
            "ranking_reasons": _safe_list(row.get("ranking_reasons"))[:6],
        }
        for side, row in _safe_dict(feedback.get("side_feedback")).items()
        if isinstance(row, dict)
    }
    strategy_profile_feedback = [
        {
            "strategy_profile_id": row.get("strategy_profile_id"),
            "symbol": row.get("symbol"),
            "side": row.get("side"),
            "decision_lane": row.get("decision_lane"),
            "recommended_stage": row.get("recommended_stage"),
            "weight_multiplier": row.get("weight_multiplier"),
            "realized_net_pnl": row.get("realized_net_pnl"),
            "profit_factor": row.get("profit_factor"),
            "ranking_reasons": _safe_list(row.get("ranking_reasons"))[:4],
        }
        for row in _safe_list(feedback.get("strategy_profile_feedback"))[:12]
        if isinstance(row, dict)
    ]
    source_weight_feedback = [
        {
            "source": row.get("source"),
            "recommended_stage": row.get("recommended_stage"),
            "weight_multiplier": row.get("weight_multiplier"),
            "realized_net_pnl": row.get("realized_net_pnl"),
            "count": row.get("count"),
            "ranking_reasons": _safe_list(row.get("ranking_reasons"))[:4],
        }
        for row in _safe_list(feedback.get("source_weight_feedback"))[:12]
        if isinstance(row, dict)
    ]
    lane_feedback = [
        {
            "lane": row.get("lane"),
            "recommendation": row.get("recommendation"),
            "reason": row.get("reason"),
            "entry_bias": row.get("entry_bias"),
            "count": row.get("count"),
            "realized_net_pnl": row.get("realized_net_pnl"),
            "profit_factor": row.get("profit_factor"),
        }
        for row in _safe_list(feedback.get("lane_feedback"))[:12]
        if isinstance(row, dict)
    ]
    size_feedback = [
        {
            "strategy_profile_id": row.get("strategy_profile_id"),
            "decision_lane": row.get("decision_lane"),
            "recommended_stage": row.get("recommended_stage"),
            "recommendation": row.get("recommendation"),
            "sizing_bias": row.get("sizing_bias"),
            "evidence": _safe_dict(row.get("evidence")),
        }
        for row in _safe_list(feedback.get("size_feedback"))[:12]
        if isinstance(row, dict)
    ]
    exit_feedback = [
        {
            "attribution": row.get("attribution"),
            "recommendation": row.get("recommendation"),
            "count": row.get("count"),
            "exit_bias": row.get("exit_bias"),
        }
        for row in _safe_list(feedback.get("exit_feedback"))[:12]
        if isinstance(row, dict)
    ]
    return {
        "status": feedback.get("status"),
        "objective": feedback.get("objective"),
        "objective_basis": {
            "metric": objective_basis.get("metric"),
            "cost_policy": objective_basis.get("cost_policy"),
            "window_policy": objective_basis.get("window_policy"),
        },
        "can_influence_strategy_context": bool(
            feedback.get("can_influence_strategy_context")
        ),
        "side_weights": _safe_dict(feedback.get("side_weights")),
        "side_feedback": side_feedback,
        "strategy_profile_feedback": strategy_profile_feedback,
        "source_weight_feedback": source_weight_feedback,
        "lane_feedback": lane_feedback,
        "size_feedback": size_feedback,
        "missed_opportunity_feedback": {
            "sample_count": missed.get("sample_count", 0),
            "diagnosis": missed.get("diagnosis"),
            "missed_positive_shadow_count": missed.get("missed_positive_shadow_count", 0),
            "missed_shadow_return_total_pct": missed.get("missed_shadow_return_total_pct", 0.0),
            "entry_bias": missed.get("entry_bias"),
            "reason_counts": _safe_list(missed.get("reason_counts"))[:8],
            "recommendations": _safe_list(missed.get("recommendations"))[:8],
        },
        "exit_feedback": exit_feedback,
        "exit_plan_reference": {
            "checked_count": exit_reference.get("checked_count", 0),
            "missing_count": exit_reference.get("missing_count", 0),
            "coverage_ratio": exit_reference.get("coverage_ratio", 0.0),
            "training_attribution_blocker": bool(
                exit_reference.get("training_attribution_blocker")
            ),
        },
        "local_ml_live_influence": {
            "allow_live_entry_influence": bool(local_ml.get("allow_live_entry_influence")),
            "eligible_for_shadow_to_canary_review": bool(
                local_ml.get("eligible_for_shadow_to_canary_review")
            ),
            "reason": local_ml.get("reason", ""),
        },
        "profit_acceptance": {
            "window_closed_trade_count": acceptance.get("window_closed_trade_count", 0),
            "net_pnl": acceptance.get("net_pnl", 0.0),
            "profit_factor": acceptance.get("profit_factor", 0.0),
            "avg_win": acceptance.get("avg_win", 0.0),
            "avg_loss": acceptance.get("avg_loss", 0.0),
        },
        "policy": _safe_dict(feedback.get("policy")),
    }


def _percentile(values: list[float], percentile: float) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    bounded = min(max(float(percentile or 0.0), 0.0), 1.0)
    index = (len(clean) - 1) * bounded
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return clean[int(index)]
    fraction = index - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def _payoff_distribution_profile(pnls: list[float]) -> dict[str, Any]:
    wins = [value for value in pnls if value > 0]
    losses = [abs(value) for value in pnls if value < 0]
    win_count = len(wins)
    loss_count = len(losses)
    total_count = len([value for value in pnls if value != 0])
    avg_win = sum(wins) / win_count if win_count else 0.0
    avg_loss = sum(losses) / loss_count if loss_count else 0.0
    win_median = _percentile(wins, 0.50)
    loss_median = _percentile(losses, 0.50)
    win_floor = min(avg_win, win_median) if win_count else 0.0
    loss_reference = max(avg_loss, loss_median)
    small_wins = [value for value in wins if win_floor > 0 and value <= win_floor]
    large_losses = [value for value in losses if loss_reference > 0 and value >= loss_reference]
    profit = sum(wins)
    loss = sum(losses)
    profit_factor = profit / loss if loss > 0 else (999.0 if profit > 0 else 0.0)
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else (999.0 if avg_win > 0 else 0.0)
    small_win_ratio = len(small_wins) / win_count if win_count else 0.0
    large_loss_ratio = len(large_losses) / loss_count if loss_count else 0.0
    imbalance = 0.0
    if total_count:
        low_payoff_pressure = max(0.0, 1.0 - min(payoff_ratio, 1.0))
        loss_share = loss_count / total_count
        imbalance = min(
            1.0,
            low_payoff_pressure * 0.45
            + small_win_ratio * 0.25
            + large_loss_ratio * 0.20
            + loss_share * 0.10,
        )
    triggered = bool(
        win_count > 0
        and loss_count > 0
        and (
            payoff_ratio < 1.0
            or profit_factor < 1.0
            or (small_wins and large_losses and sum(large_losses) > sum(small_wins))
        )
    )
    return {
        "sample_count": total_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "median_win": round(win_median, 6),
        "median_loss": round(loss_median, 6),
        "dynamic_small_win_reference": round(win_floor, 6),
        "dynamic_large_loss_reference": round(loss_reference, 6),
        "small_win_count": len(small_wins),
        "large_loss_count": len(large_losses),
        "small_win_ratio": round(small_win_ratio, 6),
        "large_loss_ratio": round(large_loss_ratio, 6),
        "profit_factor": round(profit_factor, 6),
        "payoff_ratio": round(payoff_ratio, 6),
        "imbalance_score": round(imbalance, 6),
        "triggered": triggered,
        "policy": "dynamic_window_distribution_not_fixed_usdt_thresholds",
    }


def _payoff_repair_profile(*profiles: dict[str, Any]) -> dict[str, Any]:
    active = [_safe_dict(profile) for profile in profiles if _safe_dict(profile)]
    if not active:
        return {
            "triggered": False,
            "imbalance_score": 0.0,
            "sample_count": 0,
            "profit_factor": 0.0,
            "payoff_ratio": 0.0,
            "small_win_count": 0,
            "large_loss_count": 0,
            "policy": "dynamic_window_distribution_not_fixed_usdt_thresholds",
        }
    triggered = any(bool(profile.get("triggered")) for profile in active)
    imbalance = max(_safe_float(profile.get("imbalance_score"), 0.0) for profile in active)
    sample_count = sum(_safe_int(profile.get("sample_count"), 0) for profile in active)
    small_wins = sum(_safe_int(profile.get("small_win_count"), 0) for profile in active)
    large_losses = sum(_safe_int(profile.get("large_loss_count"), 0) for profile in active)
    profit_factors = [
        _safe_float(profile.get("profit_factor"), 0.0)
        for profile in active
        if _safe_float(profile.get("profit_factor"), 0.0) > 0
    ]
    payoff_ratios = [
        _safe_float(profile.get("payoff_ratio"), 0.0)
        for profile in active
        if _safe_float(profile.get("payoff_ratio"), 0.0) > 0
    ]
    return {
        "triggered": triggered,
        "imbalance_score": round(_clamp(imbalance, 0.0, 1.0), 6),
        "sample_count": sample_count,
        "profit_factor": round(min(profit_factors), 6) if profit_factors else 0.0,
        "payoff_ratio": round(min(payoff_ratios), 6) if payoff_ratios else 0.0,
        "small_win_count": small_wins,
        "large_loss_count": large_losses,
        "policy": "dynamic_window_distribution_not_fixed_usdt_thresholds",
    }


def _compact_payoff_profile_value(value: Any) -> dict[str, Any]:
    source = _safe_dict(value)
    result: dict[str, Any] = {}
    for key in (
        "sample_count",
        "win_count",
        "loss_count",
        "avg_win",
        "avg_loss",
        "median_win",
        "median_loss",
        "dynamic_small_win_reference",
        "dynamic_large_loss_reference",
        "small_win_count",
        "large_loss_count",
        "small_win_ratio",
        "large_loss_ratio",
        "profit_factor",
        "payoff_ratio",
        "imbalance_score",
        "triggered",
        "policy",
    ):
        item = source.get(key)
        if isinstance(item, str):
            result[key] = item[:140]
        elif item is None or isinstance(item, (bool, int, float)):
            result[key] = item
    return result


class StrategyCandidateModelError(RuntimeError):
    """Structured error raised by the LLM strategy-candidate generator."""

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind or LLM_CANDIDATE_ERROR_UNKNOWN


def _material_low_quality_pressure(open_pressure: dict[str, Any]) -> bool:
    open_groups = max(0, _safe_int(open_pressure.get("open_group_count"), 0))
    max_open = max(1, _safe_int(open_pressure.get("max_open_positions"), open_groups or 1))
    low_quality = max(0, _safe_int(open_pressure.get("low_quality_open_count"), 0))
    if low_quality <= 0:
        return False
    ratio = _safe_float(open_pressure.get("low_quality_open_ratio"), 0.0)
    dynamic_threshold = max(2, math.ceil(max(open_groups, 1) * 0.40), math.ceil(max_open * 0.15))
    return bool(low_quality >= dynamic_threshold or (low_quality >= 2 and ratio >= 0.45))


def _material_release_pressure(
    open_pressure: dict[str, Any],
    problem_keys: set[str] | None = None,
    *,
    active_profile_id: str = "",
) -> bool:
    if open_pressure.get("full_position_pressure") or open_pressure.get("fragmentation_pressure"):
        return True
    if (
        active_profile_id == "loss_release"
        and _safe_int(open_pressure.get("low_quality_open_count"), 0) > 0
    ):
        return True
    return _material_low_quality_pressure(open_pressure)


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return str(value)


def _action_side(action: Any) -> str:
    text = _action(action)
    if text in {"long", "close_long"}:
        return "long"
    if text in {"short", "close_short"}:
        return "short"
    return "unknown"


def _aware(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_iso_datetime(value: Any) -> datetime | None:
    return parse_position_time(value)


def _hours_between(start: Any, end: Any) -> float:
    started = parse_position_time(start)
    ended = parse_position_time(end)
    if not started or not ended:
        return 0.0
    return max((ended - started).total_seconds() / 3600.0, 0.0)


def _action(value: Any) -> str:
    return str(value or "").lower().strip()


def _position_side(value: Any) -> str:
    text = str(value or "").lower()
    if "short" in text:
        return "short"
    if "long" in text:
        return "long"
    return "unknown"


def _created_at(row: Any) -> datetime | None:
    if isinstance(row, dict):
        return position_open_time(row)
    return parse_position_time(getattr(row, "created_at", None))


def _closed_at(row: Any) -> datetime | None:
    if isinstance(row, dict):
        return parse_position_time(row.get("closed_at"))
    return parse_position_time(getattr(row, "closed_at", None))


def _position_pnl(row: Any) -> float:
    return _safe_float(_row_get(row, "realized_pnl"), 0.0)


def _closed_position_dedupe_key(row: Any) -> tuple[Any, ...] | None:
    entry_order_id = str(_row_get(row, "entry_exchange_order_id") or "").strip()
    close_order_id = str(_row_get(row, "close_exchange_order_id") or "").strip()
    if not entry_order_id or not close_order_id:
        return None
    return (
        str(_row_get(row, "execution_mode") or "").strip().lower(),
        _normalized_symbol_key(_row_get(row, "symbol")),
        _position_side(_row_get(row, "side")),
        entry_order_id,
        close_order_id,
    )


def _closed_position_evidence_score(row: Any) -> tuple[int, float, float, int]:
    return (
        1 if str(_row_get(row, "close_exchange_order_id") or "").strip() else 0,
        _as_timestamp(_closed_at(row)),
        _as_timestamp(_created_at(row)),
        _safe_int(_row_get(row, "id"), 0),
    )


def _as_timestamp(value: datetime | None) -> float:
    if value is None:
        return 0.0
    return value.timestamp()


def _deduplicate_training_positions(positions: list[Any]) -> tuple[list[Any], dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[Any]] = {}
    passthrough: list[Any] = []
    for position in positions:
        key = _closed_position_dedupe_key(position)
        if key is None:
            passthrough.append(position)
        else:
            buckets.setdefault(key, []).append(position)
    deduped = list(passthrough)
    duplicate_groups: list[dict[str, Any]] = []
    for key, rows in buckets.items():
        if len(rows) == 1:
            deduped.append(rows[0])
            continue
        winner = max(rows, key=_closed_position_evidence_score)
        deduped.append(winner)
        duplicate_groups.append(
            {
                "key": {
                    "execution_mode": key[0],
                    "symbol": key[1],
                    "side": key[2],
                    "entry_exchange_order_id": key[3],
                    "close_exchange_order_id": key[4],
                },
                "kept_position_id": _safe_int(_row_get(winner, "id"), 0),
                "dropped_position_ids": [
                    _safe_int(_row_get(row, "id"), 0)
                    for row in rows
                    if row is not winner and _safe_int(_row_get(row, "id"), 0) > 0
                ],
                "duplicate_count": len(rows) - 1,
            }
        )
    deduped.sort(
        key=lambda row: (
            _closed_at(row) or _created_at(row) or datetime.min.replace(tzinfo=UTC),
            _safe_int(_row_get(row, "id"), 0),
        )
    )
    duplicate_count = sum(item["duplicate_count"] for item in duplicate_groups)
    duplicate_position_ids = [
        pid
        for item in duplicate_groups
        for pid in item["dropped_position_ids"]
        if pid > 0
    ]
    return deduped, {
        "deduplicated_position_count": duplicate_count,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_position_ids": sorted(duplicate_position_ids),
        "duplicate_groups": duplicate_groups[:50],
        "policy": (
            "strategy learning counts one closed trade per authoritative OKX entry/close order pair"
        ),
    }


def _open_position_pnl(row: Any) -> float:
    reported = _safe_float(_row_get(row, "unrealized_pnl"), 0.0)
    derived = _derived_open_position_pnl(row)
    if abs(reported) < 1e-9 and abs(derived) > 1e-9:
        return derived
    return reported


def _derived_open_position_pnl(row: Any) -> float:
    quantity = abs(
        _safe_float(
            _row_get(row, "quantity") or _row_get(row, "contracts") or _row_get(row, "sz"),
            0.0,
        )
    )
    entry = _safe_float(_row_get(row, "entry_price"), 0.0)
    current = _safe_float(_row_get(row, "current_price"), entry)
    if quantity <= 0 or entry <= 0 or current <= 0:
        return 0.0
    contract_size = _safe_float(
        _row_get(row, "contract_size") or _row_get(row, "contractSize"),
        1.0,
    )
    if contract_size <= 0:
        contract_size = 1.0
    side = _row_side(row)
    if side == "short":
        return (entry - current) * quantity * contract_size
    if side == "long":
        return (current - entry) * quantity * contract_size
    return 0.0


def _row_side(row: Any) -> str:
    if isinstance(row, dict):
        return _position_side(row.get("side"))
    return _position_side(getattr(row, "side", None))


def _row_symbol(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("symbol") or "")
    return str(getattr(row, "symbol", "") or "")


def _normalized_symbol_key(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if ":" in text:
        text = text.split(":", 1)[0]
    if text.endswith("-SWAP"):
        text = text[:-5]
    if "/" not in text and "-" in text:
        parts = [part for part in text.split("-") if part]
        if len(parts) >= 2:
            text = f"{parts[0]}/{parts[1]}"
    return text


def _row_model(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("model_name") or ENSEMBLE_TRADER_NAME)
    return str(getattr(row, "model_name", ENSEMBLE_TRADER_NAME) or ENSEMBLE_TRADER_NAME)


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _position_open_time_value(row: Any) -> Any:
    opened = position_open_time(row)
    if opened is not None:
        return opened.isoformat()
    return _row_get(row, "created_at") or _row_get(row, "opened_at") or _row_get(row, "timestamp")


def _row_strategy_profile_id(row: Any) -> str:
    direct = _row_get(row, "strategy_profile_id") or _row_get(row, "profile_id")
    if direct:
        return str(direct)
    raw = _row_get(row, "raw_response")
    context = _safe_dict(_safe_dict(raw).get("strategy_learning_context"))
    return str(context.get("strategy_profile_id") or "")


@dataclass(frozen=True, slots=True)
class StrategyProfile:
    """Versioned, bounded strategy profile consumed by the scheduler."""

    profile_id: str
    version: int
    label: str
    status: str
    source: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    promotion: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        consumption = _candidate_param_consumption(self.params)
        return {
            "id": self.profile_id,
            "version": self.version,
            "label": self.label,
            "status": self.status,
            "source": self.source,
            "description": self.description,
            "params": self.params,
            "promotion": self.promotion,
            "param_consumption": consumption,
            "consumed_runtime_params": consumption["consumed_runtime_params"],
            "unused_runtime_params": consumption["unused_runtime_params"],
        }


@dataclass(frozen=True, slots=True)
class StrategyFeedback:
    """Structured feedback compiled from trading records."""

    mode: str
    window_hours: int
    generated_at: str
    totals: dict[str, Any]
    side_performance: dict[str, dict[str, Any]]
    open_position_pressure: dict[str, Any]
    decision_quality: dict[str, Any]
    shadow_feedback: dict[str, Any]
    expert_memory: dict[str, Any]
    manual_intervention: dict[str, Any]
    trade_fact_quarantine: dict[str, Any]
    reflection_feedback: dict[str, Any]
    event_feedback: dict[str, Any]
    profit_first_runtime_feedback: dict[str, Any]
    problems: list[dict[str, Any]]
    root_causes: list[str]
    training_policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "window_hours": self.window_hours,
            "generated_at": self.generated_at,
            "totals": self.totals,
            "side_performance": self.side_performance,
            "open_position_pressure": self.open_position_pressure,
            "decision_quality": self.decision_quality,
            "shadow_feedback": self.shadow_feedback,
            "expert_memory": self.expert_memory,
            "manual_intervention": self.manual_intervention,
            "trade_fact_quarantine": self.trade_fact_quarantine,
            "reflection_feedback": self.reflection_feedback,
            "event_feedback": self.event_feedback,
            "profit_first_runtime_feedback": self.profit_first_runtime_feedback,
            "problems": self.problems,
            "root_causes": self.root_causes,
            "training_policy": self.training_policy,
        }


@dataclass(frozen=True, slots=True)
class StrategySchedule:
    """Selected profile plus runtime instructions."""

    active_profile: StrategyProfile
    reason: str
    runtime: dict[str, Any]
    rollback: dict[str, Any]
    candidates: list[dict[str, Any]]
    backtest: dict[str, Any]
    shadow_validation: dict[str, Any]
    probe: dict[str, Any]
    disabled_profiles: list[str]
    scheduler_mode: str
    manual_profile_id: str
    disabled_profile_reasons: dict[str, Any] = field(default_factory=dict)
    reconsidered_profiles: list[str] = field(default_factory=list)
    blocked_candidate_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_profile": self.active_profile.to_dict(),
            "reason": self.reason,
            "runtime": self.runtime,
            "rollback": self.rollback,
            "candidates": self.candidates,
            "backtest": self.backtest,
            "shadow_validation": self.shadow_validation,
            "probe": self.probe,
            "disabled_profiles": self.disabled_profiles,
            "scheduler_mode": self.scheduler_mode,
            "manual_profile_id": self.manual_profile_id,
            "disabled_profile_reasons": self.disabled_profile_reasons,
            "reconsidered_profiles": self.reconsidered_profiles,
            "blocked_candidate_count": self.blocked_candidate_count,
        }


class StrategyLearningStateStore:
    """Small JSON state store for disabled profiles and manual rollback."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (settings.data_dir / STATE_FILE_NAME)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"disabled_profiles": {}, "manual_active_profile": ""}
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                return {"disabled_profiles": {}, "manual_active_profile": ""}
            if state.get("manual_active_profile") == "baseline_current":
                state = dict(state)
                state["manual_active_profile"] = ""
            return state
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("strategy learning state load failed", error=safe_error_text(exc))
            return {"disabled_profiles": {}, "manual_active_profile": ""}

    def save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(self.path)

    def disabled_profiles(self) -> dict[str, Any]:
        state = self.load()
        disabled = state.get("disabled_profiles")
        if not isinstance(disabled, dict):
            return {}
        now = datetime.now(UTC)
        active: dict[str, Any] = {}
        expired: list[str] = []
        for profile_id, meta in disabled.items():
            row = _safe_dict(meta)
            reason = str(row.get("reason") or "")
            is_auto = bool(row.get("auto")) or reason.startswith("auto_runtime_guard:")
            disabled_until = _parse_iso_datetime(row.get("disabled_until"))
            updated_at = _parse_iso_datetime(row.get("updated_at"))
            reconsider_at = disabled_until
            if is_auto and reconsider_at is None and updated_at is not None:
                reconsider_at = updated_at + timedelta(
                    seconds=AUTO_DISABLED_PROFILE_RECONSIDER_SECONDS
                )
            if is_auto and reconsider_at is None:
                expired.append(str(profile_id))
                continue
            if is_auto and reconsider_at <= now:
                expired.append(str(profile_id))
                continue
            active[str(profile_id)] = row
        if expired:
            for profile_id in expired:
                disabled.pop(profile_id, None)
            state["disabled_profiles"] = disabled
            self.save(state)
        return active

    def set_profile_disabled(
        self,
        profile_id: str,
        *,
        disabled: bool,
        reason: str = "",
    ) -> dict[str, Any]:
        state = self.load()
        disabled_profiles = state.setdefault("disabled_profiles", {})
        if not isinstance(disabled_profiles, dict):
            disabled_profiles = {}
            state["disabled_profiles"] = disabled_profiles
        if disabled:
            now = datetime.now(UTC)
            is_auto = reason.startswith("auto_runtime_guard:")
            disabled_until = (
                now + timedelta(seconds=AUTO_DISABLED_PROFILE_RECONSIDER_SECONDS)
                if is_auto
                else None
            )
            disabled_profiles[profile_id] = {
                "reason": reason or "manual_disable",
                "updated_at": now.isoformat(),
                "auto": bool(is_auto),
                "disabled_until": disabled_until.isoformat() if disabled_until else "",
            }
            if state.get("manual_active_profile") == profile_id:
                state["manual_active_profile"] = ""
        else:
            disabled_profiles.pop(profile_id, None)
        self.save(state)
        return state

    def set_manual_active_profile(self, profile_id: str | None) -> dict[str, Any]:
        state = self.load()
        normalized_profile_id = (
            "" if profile_id in {None, "", "baseline_current"} else str(profile_id)
        )
        state["manual_active_profile"] = normalized_profile_id
        self.save(state)
        return state

    def mark_profile_snapshot_persisted(
        self,
        *,
        mode: str,
        signature: str,
        active_profile_id: str,
    ) -> dict[str, Any]:
        state = self.load()
        state["last_profile_snapshot"] = {
            "mode": mode,
            "signature": signature,
            "active_profile_id": active_profile_id,
            "persisted_at": datetime.now(UTC).isoformat(),
        }
        self.save(state)
        return state


class StrategyFeedbackCompiler:
    """Compile existing records into a single strategy feedback report."""

    def __init__(self, quality_scorer: PositionQualityScorer | None = None) -> None:
        self.quality_scorer = quality_scorer or PositionQualityScorer()

    def compile(
        self,
        *,
        mode: str,
        window_hours: int,
        positions: list[Any],
        open_positions: list[Any],
        orders: list[Any],
        decisions: list[Any],
        shadows: list[Any],
        memories: list[Any],
        strategy_events: list[Any] | None = None,
        reflections: list[Any] | None = None,
        max_open_positions: int = 20,
    ) -> StrategyFeedback:
        manual_orders = [order for order in orders if is_manual_close_order(order)]
        manual_position_ids: set[int] = set()
        untrusted_fact_position_ids: set[int] = set()
        untrusted_fact_reasons: dict[str, int] = {}
        training_positions: list[Any] = []
        for position in positions:
            if position_has_manual_close_order(position, manual_orders):
                manual_position_ids.add(_safe_int(getattr(position, "id", None), 0))
                continue
            untrusted_reason = closed_position_trade_fact_untrusted_reason(position)
            if untrusted_reason is not None:
                untrusted_fact_position_ids.add(_safe_int(getattr(position, "id", None), 0))
                untrusted_fact_reasons[untrusted_reason] = (
                    untrusted_fact_reasons.get(untrusted_reason, 0) + 1
                )
                continue
            training_positions.append(position)
        training_positions, duplicate_fact_quarantine = _deduplicate_training_positions(
            training_positions
        )

        side_performance = self._side_performance(training_positions)
        open_pressure = self._open_position_pressure(open_positions, max_open_positions)
        decision_quality = self._decision_quality(decisions)
        shadow_feedback = self._shadow_feedback(shadows, decisions)
        expert_memory = self._expert_memory(memories)
        event_feedback = self._event_feedback(strategy_events or [])
        duplicate_position_ids = {
            int(pid)
            for pid in duplicate_fact_quarantine.get("duplicate_position_ids", [])
            if _safe_int(pid, 0) > 0
        }
        reflection_feedback = self._reflection_feedback(
            reflections or [],
            excluded_position_ids=manual_position_ids
            | untrusted_fact_position_ids
            | duplicate_position_ids,
        )
        manual_intervention = {
            "manual_close_orders": len(manual_orders),
            "manual_closed_positions": len(manual_position_ids),
            "excluded_from_training": len(manual_position_ids),
            "policy": "manual closes are attribution and intervention signals, not model training samples",
        }
        trade_fact_quarantine = {
            "excluded_position_count": len(untrusted_fact_position_ids),
            "reason_counts": untrusted_fact_reasons,
            "position_ids": sorted(pid for pid in untrusted_fact_position_ids if pid > 0)[:50],
            "duplicate_position_count": duplicate_fact_quarantine["deduplicated_position_count"],
            "duplicate_group_count": duplicate_fact_quarantine["duplicate_group_count"],
            "duplicate_position_ids": duplicate_fact_quarantine["duplicate_position_ids"][:50],
            "duplicate_groups": duplicate_fact_quarantine["duplicate_groups"],
            "policy": (
                "closed positions missing authoritative OKX entry/close order links are kept for "
                "audit, and duplicate authoritative order pairs are counted once for strategy "
                "learning and reflection feedback"
            ),
        }

        trade_count = len(training_positions)
        training_pnls = [_position_pnl(row) for row in training_positions]
        payoff_profile = _payoff_distribution_profile(training_pnls)
        net_pnl = round(sum(training_pnls), 6)
        win_count = sum(1 for value in training_pnls if value > 0)
        loss_count = sum(1 for value in training_pnls if value < 0)
        small_win_count = _safe_int(payoff_profile.get("small_win_count"), 0)
        large_loss_count = _safe_int(payoff_profile.get("large_loss_count"), 0)
        avg_hold_hours = (
            sum(
                _hours_between(getattr(row, "created_at", None), getattr(row, "closed_at", None))
                for row in training_positions
            )
            / trade_count
            if trade_count
            else 0.0
        )
        loss_hold_hours = [
            _hours_between(getattr(row, "created_at", None), getattr(row, "closed_at", None))
            for row in training_positions
            if _position_pnl(row) < 0
        ]
        avg_loss_hold_hours = (
            sum(loss_hold_hours) / len(loss_hold_hours) if loss_hold_hours else 0.0
        )
        trade_count_target = learning_trade_count_target(
            window_hours=window_hours,
            market_scans=_safe_int(decision_quality.get("market_scans"), 0),
            entry_signals=_safe_int(decision_quality.get("entry_signals"), 0),
            reflection_count=_safe_int(reflection_feedback.get("training_count"), 0),
            shadow_opportunities=_safe_int(shadow_feedback.get("missed_opportunity_count"), 0),
        )
        problems, root_causes = self._problems(
            side_performance=side_performance,
            open_pressure=open_pressure,
            decision_quality=decision_quality,
            shadow_feedback=shadow_feedback,
            trade_count=trade_count,
            trade_count_target=trade_count_target,
            net_pnl=net_pnl,
            small_win_count=small_win_count,
            large_loss_count=large_loss_count,
            payoff_profile=payoff_profile,
            avg_loss_hold_hours=avg_loss_hold_hours,
            event_feedback=event_feedback,
            reflection_feedback=reflection_feedback,
        )
        profit_first_report = ProfitFirstRankingService().build_report(
            decisions=decisions,
            closed_positions=training_positions,
        )
        profit_first_runtime_feedback = _safe_dict(
            profit_first_report.get("runtime_feedback")
        )
        totals = {
            "closed_trade_count": len(positions),
            "training_trade_count": trade_count,
            "net_pnl": net_pnl,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": round(win_count / trade_count, 6) if trade_count else 0.0,
            "small_win_count": small_win_count,
            "large_loss_count": large_loss_count,
            "payoff_profile": payoff_profile,
            "avg_hold_hours": round(avg_hold_hours, 4),
            "avg_loss_hold_hours": round(avg_loss_hold_hours, 4),
            "trade_count_target": trade_count_target,
            "trade_count_target_policy": "dynamic_advisory_learning_confidence",
            "trade_count_target_is_entry_gate": False,
            "trade_count_target_baseline": default_min_trade_target(),
            "low_trade_count_penalty": trade_count < trade_count_target,
            "reflection_count": reflection_feedback.get("training_count", 0),
            "reflection_total_count": reflection_feedback.get("total_count", 0),
        }
        return StrategyFeedback(
            mode=mode,
            window_hours=int(window_hours),
            generated_at=datetime.now(UTC).isoformat(),
            totals=totals,
            side_performance=side_performance,
            open_position_pressure=open_pressure,
            decision_quality=decision_quality,
            shadow_feedback=shadow_feedback,
            expert_memory=expert_memory,
            manual_intervention=manual_intervention,
            trade_fact_quarantine=trade_fact_quarantine,
            reflection_feedback=reflection_feedback,
            event_feedback=event_feedback,
            profit_first_runtime_feedback=profit_first_runtime_feedback,
            problems=problems,
            root_causes=root_causes,
            training_policy={
                "manual_close_excluded": True,
                "low_trade_count_is_penalized": True,
                "arbitrary_code_generation_allowed": False,
                "candidate_profiles_only": True,
                "strategy_learning_params": DEFAULT_TRADING_PARAMS.to_dict().get(
                    "strategy_learning", {}
                ),
            },
        )

    def _side_performance(self, positions: list[Any]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {
            "long": self._empty_side_bucket(),
            "short": self._empty_side_bucket(),
        }
        for row in positions:
            side = _position_side(getattr(row, "side", None))
            if side not in result:
                continue
            pnl = _position_pnl(row)
            bucket = result[side]
            bucket["count"] += 1
            bucket["pnl"] += pnl
            bucket["wins"] += 1 if pnl > 0 else 0
            bucket["losses"] += 1 if pnl < 0 else 0
            bucket["profit"] += max(pnl, 0.0)
            bucket["loss"] += min(pnl, 0.0)
            bucket["largest_loss"] = min(bucket["largest_loss"], pnl)
            bucket["hold_hours_total"] += _hours_between(
                getattr(row, "created_at", None),
                getattr(row, "closed_at", None),
            )
        for bucket in result.values():
            count = bucket["count"]
            losses = bucket["losses"]
            profit = bucket["profit"]
            loss_abs = abs(bucket["loss"])
            bucket["pnl"] = round(bucket["pnl"], 6)
            bucket["avg_pnl"] = round(bucket["pnl"] / count, 6) if count else 0.0
            bucket["win_rate"] = round(bucket["wins"] / count, 6) if count else 0.0
            bucket["avg_hold_hours"] = (
                round(bucket["hold_hours_total"] / count, 4) if count else 0.0
            )
            bucket["profit_factor"] = round(profit / loss_abs, 6) if loss_abs > 0 else profit
            bucket["state"] = self._side_state(bucket)
            bucket["largest_loss"] = round(bucket["largest_loss"], 6)
            bucket["loss_pressure"] = round(abs(bucket["loss"]) / max(losses, 1), 6)
            bucket.pop("hold_hours_total", None)
        return result

    @staticmethod
    def _empty_side_bucket() -> dict[str, Any]:
        return {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0.0,
            "profit": 0.0,
            "loss": 0.0,
            "largest_loss": 0.0,
            "hold_hours_total": 0.0,
        }

    @staticmethod
    def _side_state(bucket: dict[str, Any]) -> str:
        count = _safe_int(bucket.get("count"), 0)
        pnl = _safe_float(bucket.get("pnl"), 0.0)
        wins = _safe_int(bucket.get("wins"), 0)
        losses = _safe_int(bucket.get("losses"), 0)
        win_rate = _safe_float(bucket.get("win_rate"), 0.0)
        if count >= 3 and pnl < 0 and (losses >= wins + 2 or win_rate <= 0.30):
            return "degraded"
        if count >= 3 and pnl > 0 and win_rate >= 0.45:
            return "working"
        return "neutral"

    def _open_position_pressure(
        self,
        open_positions: list[Any],
        max_open_positions: int,
    ) -> dict[str, Any]:
        max_open = max(1, int(max_open_positions or 1))
        rows = list(open_positions or [])
        part_count = len(rows)
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        side_counts = {"long": 0, "short": 0, "unknown": 0}
        side_pnl = {"long": 0.0, "short": 0.0, "unknown": 0.0}
        part_candidates: list[dict[str, Any]] = []

        for row in rows:
            symbol = _row_symbol(row)
            symbol_key = _normalized_symbol_key(symbol)
            side = _row_side(row)
            pnl = _open_position_pnl(row)
            side_counts[side] = side_counts.get(side, 0) + 1
            side_pnl[side] = side_pnl.get(side, 0.0) + pnl
            group_key = (symbol_key or symbol, side)
            group = grouped.setdefault(
                group_key,
                {
                    "symbol": symbol,
                    "symbol_key": symbol_key or symbol,
                    "side": side,
                    "model_name": _row_model(row),
                    "parts": 0,
                    "unrealized_pnl": 0.0,
                    "quantity": 0.0,
                    "entry_value": 0.0,
                    "current_value": 0.0,
                    "created_at": _position_open_time_value(row),
                    "strategy_profile_id": _row_strategy_profile_id(row),
                },
            )
            group["parts"] += 1
            group["unrealized_pnl"] += pnl
            qty = abs(_safe_float(_row_get(row, "quantity"), 0.0))
            entry = _safe_float(_row_get(row, "entry_price"), 0.0)
            current = _safe_float(_row_get(row, "current_price"), entry)
            if qty > 0 and entry > 0:
                group["quantity"] += qty
                group["entry_value"] += entry * qty
                group["current_value"] += (current if current > 0 else entry) * qty
            if not group.get("strategy_profile_id"):
                group["strategy_profile_id"] = _row_strategy_profile_id(row)
            if _created_at(row) and (
                not group.get("created_at")
                or _created_at(row) < _parse_iso_datetime(group.get("created_at"))
            ):
                group["created_at"] = _position_open_time_value(row)
            part_candidates.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "model_name": _row_model(row),
                    "unrealized_pnl": round(pnl, 6),
                }
            )

        group_rows = list(grouped.values())
        losing_groups = [
            row for row in group_rows if _safe_float(row.get("unrealized_pnl"), 0.0) < 0
        ]
        winner_groups = [
            row for row in group_rows if _safe_float(row.get("unrealized_pnl"), 0.0) > 0
        ]
        group_count = len(group_rows)
        usage_ratio = group_count / max_open
        part_usage_ratio = part_count / max_open
        duplicate_part_count = max(part_count - group_count, 0)
        near_full_threshold = max(1, math.ceil(max_open * 0.85))
        full_pressure = group_count >= max_open or group_count >= near_full_threshold
        fragment_pressure = duplicate_part_count >= 3 or part_usage_ratio >= 1.15
        for row in group_rows:
            qty = _safe_float(row.pop("quantity", 0.0), 0.0)
            entry_value = _safe_float(row.pop("entry_value", 0.0), 0.0)
            current_value = _safe_float(row.pop("current_value", 0.0), 0.0)
            if qty > 0:
                row["entry_price"] = entry_value / qty
                row["current_price"] = (
                    current_value / qty if current_value > 0 else row["entry_price"]
                )
                row["quantity"] = qty
                row["notional"] = abs(row["entry_price"] * qty)
            quality = self.quality_scorer.score(
                row,
                same_symbol_side_parts=_safe_int(row.get("parts"), 1),
            )
            row["position_quality"] = quality.as_dict()
            row["quality_score"] = round(quality.score, 4)
            row["quality_bucket"] = quality.bucket
            row["quality_reasons"] = list(quality.reasons)
            row["release_priority"] = round(quality.release_priority, 4)
            row["should_release"] = quality.should_release
        release_groups = sorted(
            (
                {
                    **row,
                    "unrealized_pnl": round(_safe_float(row.get("unrealized_pnl"), 0.0), 6),
                }
                for row in group_rows
            ),
            key=lambda item: (
                _safe_float(item.get("quality_score"), 100.0),
                _safe_float(item.get("unrealized_pnl"), 0.0),
                -_safe_float(_safe_dict(item.get("position_quality")).get("hold_hours"), 0.0),
            ),
        )
        low_quality_groups = [row for row in group_rows if bool(row.get("should_release"))]
        stale_groups = [
            row
            for row in group_rows
            if _safe_dict(row.get("position_quality")).get("bucket")
            in {"watch", "release_candidate", "release_now"}
        ]
        return {
            "open_count": group_count,
            "open_part_count": part_count,
            "open_group_count": group_count,
            "duplicate_part_count": duplicate_part_count,
            "max_open_positions": max_open,
            "usage_ratio": round(usage_ratio, 6),
            "part_usage_ratio": round(part_usage_ratio, 6),
            "full_position_pressure": bool(full_pressure),
            "fragmentation_pressure": bool(fragment_pressure),
            "low_quality_open_count": len(low_quality_groups),
            "low_quality_open_ratio": round(len(low_quality_groups) / max(group_count, 1), 6),
            "stale_open_count": len(stale_groups),
            "release_queue": release_groups[:8],
            "release_queue_count": len(release_groups),
            "losing_open_count": len(losing_groups),
            "losing_open_part_count": sum(_safe_int(row.get("parts"), 0) for row in losing_groups),
            "winner_open_count": len(winner_groups),
            "open_unrealized_pnl": round(
                sum(_safe_float(row.get("unrealized_pnl"), 0.0) for row in group_rows), 6
            ),
            "losing_unrealized_pnl": round(
                sum(_safe_float(row.get("unrealized_pnl"), 0.0) for row in losing_groups), 6
            ),
            "side_counts": side_counts,
            "side_unrealized_pnl": {key: round(value, 6) for key, value in side_pnl.items()},
            "release_candidates": release_groups[:8],
            "part_release_candidates": sorted(
                part_candidates, key=lambda item: item["unrealized_pnl"]
            )[:8],
        }

    def _decision_quality(self, decisions: list[Any]) -> dict[str, Any]:
        market_scans = 0
        entry_signals = 0
        executed_entries = 0
        expert_integrity_blocks = 0
        fallback_entry_decisions = 0
        zero_second_entry_decisions = 0
        recent_cutoff = datetime.now(UTC) - timedelta(hours=2)
        recent_market_scans = 0
        recent_entry_signals = 0
        recent_executed_entries = 0
        recent_expert_integrity_blocks = 0
        recent_fallback_entry_decisions = 0
        recent_zero_second_entry_decisions = 0
        status_counts: dict[str, int] = {}
        missing_expert_counts: dict[str, int] = {}
        for row in decisions or []:
            raw = _safe_dict(getattr(row, "raw_llm_response", None))
            action = _action(getattr(row, "action", None))
            row_created_at = _created_at(row)
            is_recent = bool(row_created_at and row_created_at >= recent_cutoff)
            analysis_type = str(getattr(row, "analysis_type", "") or raw.get("analysis_type") or "")
            if analysis_type.lower() == "position" or action in {"close_long", "close_short"}:
                continue
            market_scans += 1
            if is_recent:
                recent_market_scans += 1
            if action in {"long", "short"}:
                entry_signals += 1
                if is_recent:
                    recent_entry_signals += 1
                if bool(getattr(row, "was_executed", False)):
                    executed_entries += 1
                    if is_recent:
                        recent_executed_entries += 1
                fallback, zero_second, missing = self._expert_integrity_flags(raw)
                fallback_entry_decisions += 1 if fallback else 0
                zero_second_entry_decisions += 1 if zero_second else 0
                if is_recent:
                    recent_fallback_entry_decisions += 1 if fallback else 0
                    recent_zero_second_entry_decisions += 1 if zero_second else 0
                for name in missing:
                    missing_expert_counts[name] = missing_expert_counts.get(name, 0) + 1
            reason = str(getattr(row, "execution_reason", "") or "")
            if "expert_integrity" in reason:
                expert_integrity_blocks += 1
                if is_recent:
                    recent_expert_integrity_blocks += 1
            for status in self._timing_statuses(raw):
                status_counts[status] = status_counts.get(status, 0) + 1
        signal_rate = entry_signals / market_scans if market_scans else 0.0
        execution_rate = executed_entries / entry_signals if entry_signals else 0.0
        fallback_rate = fallback_entry_decisions / entry_signals if entry_signals else 0.0
        recent_fallback_rate = (
            recent_fallback_entry_decisions / recent_entry_signals if recent_entry_signals else 0.0
        )
        return {
            "market_scans": market_scans,
            "entry_signals": entry_signals,
            "executed_entries": executed_entries,
            "signal_rate": round(signal_rate, 6),
            "execution_rate": round(execution_rate, 6),
            "expert_integrity_blocks": expert_integrity_blocks,
            "fallback_entry_decisions": fallback_entry_decisions,
            "fallback_entry_rate": round(fallback_rate, 6),
            "zero_second_entry_decisions": zero_second_entry_decisions,
            "recent_window_hours": 2,
            "recent_market_scans": recent_market_scans,
            "recent_entry_signals": recent_entry_signals,
            "recent_executed_entries": recent_executed_entries,
            "recent_expert_integrity_blocks": recent_expert_integrity_blocks,
            "recent_fallback_entry_decisions": recent_fallback_entry_decisions,
            "recent_fallback_entry_rate": round(recent_fallback_rate, 6),
            "recent_zero_second_entry_decisions": recent_zero_second_entry_decisions,
            "model_health_recovered": bool(
                recent_entry_signals
                and recent_fallback_rate < 0.20
                and recent_expert_integrity_blocks == 0
                and recent_zero_second_entry_decisions == 0
            ),
            "model_timing_status_counts": status_counts,
            "missing_expert_counts": dict(
                sorted(missing_expert_counts.items(), key=lambda item: item[1], reverse=True)
            ),
        }

    def _expert_integrity_flags(self, raw: dict[str, Any]) -> tuple[bool, bool, list[str]]:
        trusted: set[str] = set()
        seen: set[str] = set()
        fallback = False
        zero_second = False
        for item in _safe_list(raw.get("model_timings")):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            if name not in REQUIRED_ENTRY_EXPERTS:
                continue
            seen.add(name)
            status = str(item.get("status") or "").lower()
            provider = str(item.get("provider_model") or "").lower()
            fallback_flag = bool(
                item.get("batch_expert_fallback")
                or item.get("fallback")
                or item.get("local_fallback")
            )
            seconds = self._timing_seconds(item)
            if status in UNTRUSTED_EXPERT_STATUSES or "fallback" in status or fallback_flag:
                fallback = True
            if seconds <= 0:
                zero_second = True
            if status == "completed" and provider != "local_fast_prefilter" and not fallback_flag:
                trusted.add(name)
        missing = sorted(REQUIRED_ENTRY_EXPERTS - trusted)
        if REQUIRED_ENTRY_EXPERTS - seen:
            fallback = True
        return fallback, zero_second, missing

    @staticmethod
    def _timing_seconds(item: dict[str, Any]) -> float:
        for key in (
            "seconds",
            "duration_sec",
            "duration_seconds",
            "elapsed_seconds",
            "latency_seconds",
        ):
            if key in item:
                return _safe_float(item.get(key), 0.0)
        return _safe_float(item.get("duration_ms"), 0.0) / 1000.0

    @staticmethod
    def _timing_statuses(raw: dict[str, Any]) -> list[str]:
        statuses: list[str] = []
        for item in _safe_list(raw.get("model_timings")):
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "unknown").lower()
            if status:
                statuses.append(status)
        return statuses

    def _shadow_feedback(
        self, shadows: list[Any], decisions: list[Any] | None = None
    ) -> dict[str, Any]:
        completed = [row for row in shadows if str(getattr(row, "status", "")) == "completed"]
        missed = [row for row in completed if bool(getattr(row, "missed_opportunity", False))]
        bad_signals = []
        good_signals = []
        for row in completed:
            action = _action(getattr(row, "decision_action", None))
            if action not in {"long", "short"}:
                continue
            realized = (
                _safe_float(getattr(row, "long_return_pct", None), 0.0)
                if action == "long"
                else _safe_float(getattr(row, "short_return_pct", None), 0.0)
            )
            if realized <= -0.40:
                bad_signals.append(row)
            elif realized >= 0.40:
                good_signals.append(row)
        missed_closed_loop = summarize_shadow_missed_opportunities(
            completed,
            decisions=decisions or [],
        )
        return {
            "completed_count": len(completed),
            "missed_opportunity_count": len(missed),
            "bad_signal_count": len(bad_signals),
            "good_signal_count": len(good_signals),
            "missed_opportunity_rate": round(len(missed) / len(completed), 6) if completed else 0.0,
            "bad_signal_rate": round(len(bad_signals) / len(completed), 6) if completed else 0.0,
            "missed_opportunity_closed_loop": missed_closed_loop,
        }

    @staticmethod
    def _expert_memory(memories: list[Any]) -> dict[str, Any]:
        active = [row for row in memories if bool(getattr(row, "is_active", True))]
        by_type: dict[str, int] = {}
        for row in active:
            memory_type = str(getattr(row, "memory_type", "lesson") or "lesson")
            by_type[memory_type] = by_type.get(memory_type, 0) + 1
        return {
            "active_count": len(active),
            "by_type": by_type,
            "missed_opportunity_lessons": by_type.get("shadow_missed_opportunity", 0),
            "bad_signal_lessons": by_type.get("shadow_bad_signal", 0),
        }

    def _reflection_feedback(
        self,
        reflections: list[Any],
        *,
        excluded_position_ids: set[int],
    ) -> dict[str, Any]:
        rows = list(reflections or [])
        training_rows: list[Any] = []
        excluded_rows: list[Any] = []
        for row in rows:
            position_id = _safe_int(getattr(row, "position_id", None), 0)
            source = str(getattr(row, "source", "") or "").lower()
            if position_id in excluded_position_ids or "manual" in source:
                excluded_rows.append(row)
                continue
            training_rows.append(row)

        outcome_counts: dict[str, int] = {}
        mistake_counts: dict[str, int] = {}
        improvement_counts: dict[str, int] = {}
        loss_hold_minutes: list[float] = []
        win_hold_minutes: list[float] = []
        reflection_pnls: list[float] = []
        total_hold_minutes = 0.0
        net_pnl = 0.0
        fee_estimate = 0.0
        recent: list[dict[str, Any]] = []

        for row in training_rows:
            outcome = str(getattr(row, "outcome", "flat") or "flat").lower()
            pnl = _safe_float(getattr(row, "realized_pnl", None), 0.0)
            fee = _safe_float(getattr(row, "fee_estimate", None), 0.0)
            reflection_pnls.append(pnl - fee)
            hold_minutes = _safe_float(getattr(row, "hold_minutes", None), 0.0)
            mistake = str(getattr(row, "mistake_summary", "") or "").strip()
            improvement = str(getattr(row, "improvement_summary", "") or "").strip()
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            net_pnl += pnl
            fee_estimate += fee
            total_hold_minutes += hold_minutes
            if pnl < 0:
                loss_hold_minutes.append(hold_minutes)
            elif pnl > 0:
                win_hold_minutes.append(hold_minutes)
            if mistake:
                key = mistake[:160]
                mistake_counts[key] = mistake_counts.get(key, 0) + 1
            if improvement:
                key = improvement[:160]
                improvement_counts[key] = improvement_counts.get(key, 0) + 1
            if len(recent) < 20:
                recent.append(
                    {
                        "id": getattr(row, "id", None),
                        "position_id": getattr(row, "position_id", None),
                        "closed_at": getattr(row, "closed_at", None),
                        "symbol": getattr(row, "symbol", None),
                        "side": _position_side(getattr(row, "side", None)),
                        "outcome": outcome,
                        "realized_pnl": round(pnl, 6),
                        "fee_estimate": round(fee, 6),
                        "hold_minutes": round(hold_minutes, 2),
                        "mistake_summary": mistake[:240],
                        "improvement_summary": improvement[:240],
                        "source": getattr(row, "source", None),
                        "exclude_from_training": False,
                    }
                )

        count = len(training_rows)
        top_mistakes = sorted(mistake_counts.items(), key=lambda item: item[1], reverse=True)[:8]
        top_improvements = sorted(
            improvement_counts.items(), key=lambda item: item[1], reverse=True
        )[:8]
        payoff_profile = _payoff_distribution_profile(reflection_pnls)
        return {
            "total_count": len(rows),
            "training_count": count,
            "excluded_manual_count": len(excluded_rows),
            "outcome_counts": dict(
                sorted(outcome_counts.items(), key=lambda item: item[1], reverse=True)
            ),
            "win_count": sum(
                1
                for row in training_rows
                if _safe_float(getattr(row, "realized_pnl", None), 0.0) > 0
            ),
            "loss_count": sum(
                1
                for row in training_rows
                if _safe_float(getattr(row, "realized_pnl", None), 0.0) < 0
            ),
            "net_reflection_pnl": round(net_pnl, 6),
            "fee_estimate": round(fee_estimate, 6),
            "fee_adjusted_pnl": round(net_pnl - fee_estimate, 6),
            "avg_hold_minutes": round(total_hold_minutes / count, 2) if count else 0.0,
            "avg_loss_hold_minutes": (
                round(sum(loss_hold_minutes) / len(loss_hold_minutes), 2)
                if loss_hold_minutes
                else 0.0
            ),
            "avg_win_hold_minutes": (
                round(sum(win_hold_minutes) / len(win_hold_minutes), 2) if win_hold_minutes else 0.0
            ),
            "small_win_count": _safe_int(payoff_profile.get("small_win_count"), 0),
            "large_loss_count": _safe_int(payoff_profile.get("large_loss_count"), 0),
            "payoff_profile": payoff_profile,
            "loss_sample_count": len(loss_hold_minutes),
            "win_sample_count": len(win_hold_minutes),
            "mistake_count": sum(mistake_counts.values()),
            "improvement_count": sum(improvement_counts.values()),
            "top_mistakes": [{"summary": key, "count": value} for key, value in top_mistakes],
            "top_improvements": [
                {"summary": key, "count": value} for key, value in top_improvements
            ],
            "recent_reflections": recent,
            "policy": "manual close reflections are excluded from training feedback",
        }

    def _event_feedback(self, events: list[Any]) -> dict[str, Any]:
        type_counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        profile_counts: dict[str, int] = {}
        block_reasons: dict[str, dict[str, Any]] = {}
        manual_close_count = 0
        max_position_blocks = 0
        fallback_blocks = 0
        execution_errors = 0
        execution_successes = 0
        execution_event_rows: list[dict[str, Any]] = []
        skip_kind_counts: dict[str, int] = {}
        defensive_probe_shadow_count = 0
        entry_evidence_shadow_only_count = 0
        covered = 0
        missing_profile = 0
        attributable_total = 0
        attributable_covered = 0
        attributable_missing_profile = 0
        non_attributable_events = 0
        recent: list[dict[str, Any]] = []
        for row in events or []:
            event_type = str(getattr(row, "event_type", "") or "unknown")
            status = str(getattr(row, "event_status", "") or "recorded")
            action = getattr(row, "action", None)
            profile_id = str(getattr(row, "profile_id", "") or "")
            reason = str(getattr(row, "reason", "") or "")
            attribution = _safe_dict(getattr(row, "attribution", None))
            skip_kind = self._event_skip_kind(reason=reason, attribution=attribution)
            if skip_kind:
                skip_kind_counts[skip_kind] = skip_kind_counts.get(skip_kind, 0) + 1
                if skip_kind == "profit_first_defensive_probe_shadow":
                    defensive_probe_shadow_count += 1
                if skip_kind in {"entry_evidence_shadow_only", "entry_evidence_wait"}:
                    entry_evidence_shadow_only_count += 1
            reason_info = self._event_reason_info(
                event_type=event_type,
                status=status,
                reason=reason,
                attribution=attribution,
            )
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
            if profile_id:
                covered += 1
                profile_counts[profile_id] = profile_counts.get(profile_id, 0) + 1
            else:
                missing_profile += 1
            if self._is_attributable_strategy_event(
                event_type=event_type,
                status=status,
                action=action,
            ):
                attributable_total += 1
                if profile_id:
                    attributable_covered += 1
                else:
                    attributable_missing_profile += 1
            else:
                non_attributable_events += 1
            if event_type == "manual_close":
                manual_close_count += 1
            lower_text = f"{event_type} {status} {reason}".lower()
            if any(
                token in lower_text
                for token in ("max_position", "capacity", "满仓", "仓位已满", "限制")
            ):
                max_position_blocks += 1
            if any(
                token in lower_text for token in ("fallback", "expert_integrity", "partial_batch")
            ):
                fallback_blocks += 1
            is_transient_exchange_error = is_okx_temporary_service_error(
                {
                    "event_type": event_type,
                    "status": status,
                    "reason": reason,
                    "attribution": attribution,
                }
            )
            is_execution_error = bool(
                status in {"error", "failed", "rejected"} or event_type == "execution_error"
            )
            if is_execution_error and not is_transient_exchange_error:
                execution_errors += 1
            is_execution_event = event_type in {
                "execution_attempt",
                "execution_result",
                "execution_error",
            }
            if event_type == "execution_result" and status in {
                "executed",
                "filled",
                "success",
            }:
                execution_successes += 1
            if is_execution_event:
                execution_event_rows.append(
                    {
                        "created_at": parse_position_time(getattr(row, "created_at", None)),
                        "event_type": event_type,
                        "status": status,
                        "reason_category": reason_info.get("category", "other"),
                        "skip_kind": skip_kind,
                        "blocks_execution_guard": self._event_blocks_execution_guard(
                            event_type=event_type,
                            status=status,
                            reason=reason,
                            attribution=attribution,
                            reason_info=reason_info,
                        ),
                        "is_error": bool(is_execution_error and not is_transient_exchange_error),
                        "is_success": bool(
                            event_type == "execution_result"
                            and status in {"executed", "filled", "success"}
                        ),
                    }
                )
            if status in {"blocked", "skipped", "rejected", "failed"}:
                key = str(reason_info.get("label") or event_type)[:160]
                if key:
                    item = block_reasons.setdefault(
                        key,
                        {
                            "reason": key,
                            "category": reason_info.get("category", "other"),
                            "raw_reason": str(attribution.get("blocker") or reason or event_type)[
                                :240
                            ],
                            "count": 0,
                        },
                    )
                    item["count"] = _safe_int(item.get("count"), 0) + 1
            if len(recent) < 40:
                recent.append(
                    {
                        "id": getattr(row, "id", None),
                        "created_at": getattr(row, "created_at", None),
                        "event_type": event_type,
                        "event_status": status,
                        "severity": getattr(row, "severity", "info"),
                        "symbol": getattr(row, "symbol", None),
                        "side": getattr(row, "side", None),
                        "action": action,
                        "profile_id": profile_id,
                        "order_id": getattr(row, "order_id", None),
                        "position_id": getattr(row, "position_id", None),
                        "reason": reason,
                        "reason_label": reason_info.get("label"),
                        "reason_category": reason_info.get("category"),
                        "skip_kind": skip_kind,
                        "exclude_from_training": bool(getattr(row, "exclude_from_training", False)),
                    }
                )
        total = len(events or [])
        coverage = covered / total if total else 0.0
        attributable_coverage = (
            attributable_covered / attributable_total if attributable_total else 0.0
        )
        top_blocks = sorted(
            block_reasons.values(),
            key=lambda item: _safe_int(item.get("count"), 0),
            reverse=True,
        )[:8]
        latest_execution_success_at = self._latest_execution_event_at(
            execution_event_rows,
            key="is_success",
        )
        latest_execution_error_at = self._latest_execution_event_at(
            execution_event_rows,
            key="is_error",
        )
        unresolved_execution_errors = self._unresolved_execution_errors(
            execution_event_rows,
            latest_success_at=latest_execution_success_at,
        )
        unresolved_execution_guard_errors = self._unresolved_execution_errors(
            execution_event_rows,
            latest_success_at=latest_execution_success_at,
            guard_only=True,
        )
        return {
            "total_events": total,
            "type_counts": dict(
                sorted(type_counts.items(), key=lambda item: item[1], reverse=True)
            ),
            "status_counts": dict(
                sorted(status_counts.items(), key=lambda item: item[1], reverse=True)
            ),
            "profile_counts": dict(
                sorted(profile_counts.items(), key=lambda item: item[1], reverse=True)
            ),
            "skip_kind_counts": dict(
                sorted(skip_kind_counts.items(), key=lambda item: item[1], reverse=True)
            ),
            "profit_first_defensive_probe_shadow_count": defensive_probe_shadow_count,
            "entry_evidence_shadow_only_count": entry_evidence_shadow_only_count,
            "attribution_coverage": round(coverage, 6),
            "attributable_event_coverage": round(attributable_coverage, 6),
            "attributable_events": attributable_total,
            "attributable_missing_profile_events": attributable_missing_profile,
            "non_attributable_events": non_attributable_events,
            "missing_profile_events": missing_profile,
            "manual_close_events": manual_close_count,
            "max_position_blocks": max_position_blocks,
            "fallback_blocks": fallback_blocks,
            "execution_errors": execution_errors,
            "execution_successes": execution_successes,
            "unresolved_execution_errors": unresolved_execution_errors,
            "unresolved_execution_guard_errors": unresolved_execution_guard_errors,
            "latest_execution_success_at": (
                latest_execution_success_at.isoformat() if latest_execution_success_at else None
            ),
            "latest_execution_error_at": (
                latest_execution_error_at.isoformat() if latest_execution_error_at else None
            ),
            "execution_recovered_after_error": bool(
                latest_execution_success_at
                and latest_execution_error_at
                and latest_execution_success_at >= latest_execution_error_at
            ),
            "top_block_reasons": top_blocks,
            "recent_events": recent,
        }

    @staticmethod
    def _event_skip_kind(*, reason: str, attribution: dict[str, Any]) -> str:
        direct = str(attribution.get("skip_kind") or "").strip().lower()
        if direct:
            return direct[:120]
        text = " ".join(
            str(value or "")
            for value in (
                reason,
                attribution.get("blocker"),
                attribution.get("policy_blocker"),
                attribution.get("reason"),
            )
        ).lower()
        known = {
            "profit_first_defensive_probe_shadow",
            "profit_first_probe_loss_brake",
            "entry_evidence_shadow_only",
            "entry_evidence_wait",
            "entry_pre_execution_skip",
            "round_unresolved_terminal_skip",
        }
        for token in known:
            if token in text:
                return token
        return ""

    @staticmethod
    def _latest_execution_event_at(
        rows: list[dict[str, Any]],
        *,
        key: str,
    ) -> datetime | None:
        timestamps = [
            parse_position_time(row.get("created_at"))
            for row in rows
            if row.get(key) and parse_position_time(row.get("created_at"))
        ]
        return max(timestamps) if timestamps else None

    @staticmethod
    def _unresolved_execution_errors(
        rows: list[dict[str, Any]],
        *,
        latest_success_at: datetime | None,
        guard_only: bool = False,
    ) -> int:
        unresolved = 0
        for row in rows:
            if not row.get("is_error"):
                continue
            if guard_only and not row.get("blocks_execution_guard"):
                continue
            created_at = parse_position_time(row.get("created_at"))
            if latest_success_at and created_at and created_at <= latest_success_at:
                continue
            unresolved += 1
        return unresolved

    @staticmethod
    def _event_blocks_execution_guard(
        *,
        event_type: str,
        status: str,
        reason: str,
        attribution: dict[str, Any],
        reason_info: dict[str, str],
    ) -> bool:
        normalized_type = str(event_type or "").lower()
        normalized_status = str(status or "").lower()
        if normalized_type not in {"execution_attempt", "execution_result", "execution_error"}:
            return False
        if (
            normalized_status not in {"error", "failed", "rejected"}
            and normalized_type != "execution_error"
        ):
            return False
        text = " ".join(
            str(value or "")
            for value in (
                reason,
                attribution.get("blocker"),
                attribution.get("error"),
                reason_info.get("label"),
            )
        ).lower()
        if is_okx_temporary_service_error(text):
            return False
        non_systemic_tokens = (
            "minimum contract size",
            "below okx minimum",
            "order size is below",
            "minsz",
            "51008",
            "insufficient usdt margin",
            "more than 5 open orders",
            "59670",
            "platform's limit",
            "open interest",
            "don't have any positions in this direction",
            "no matching position to close",
            "okx 最小",
            "保证金不足",
            "可用 usdt 保证金不足",
            "当前没有对应方向",
            "平台总持仓量",
        )
        if any(token in text for token in non_systemic_tokens):
            return False
        return True

    @staticmethod
    def _is_attributable_strategy_event(
        *,
        event_type: str,
        status: str,
        action: Any,
    ) -> bool:
        normalized_type = str(event_type or "").lower()
        normalized_status = str(status or "").lower()
        normalized_action = _action(action)
        if normalized_type in NON_ATTRIBUTABLE_EVENT_TYPES:
            return False
        if normalized_type in {"execution_attempt", "execution_result", "execution_error"}:
            return True
        if normalized_status in {"blocked", "skipped", "rejected", "failed", "error"}:
            return True
        if normalized_type == "decision_logged" and normalized_action in {
            "long",
            "short",
            "close_long",
            "close_short",
        }:
            return True
        return False

    @staticmethod
    def _event_reason_info(
        *,
        event_type: str,
        status: str,
        reason: str,
        attribution: dict[str, Any],
    ) -> dict[str, str]:
        raw = str(attribution.get("blocker") or reason or event_type or "").strip()
        combined = f"{event_type} {status} {raw}"
        lower = combined.lower()
        skip_kind = str(attribution.get("skip_kind") or "").lower()
        if (
            skip_kind == "profit_first_defensive_probe_shadow"
            or "profit_first_defensive_probe_shadow" in lower
        ):
            return {
                "category": "defensive_probe_shadow",
                "label": "Profit-First 防御探针把低收益小仓开仓转为影子样本。",
            }
        if skip_kind in {"entry_evidence_shadow_only", "entry_evidence_wait"}:
            return {
                "category": "entry_evidence_shadow",
                "label": "动态证据不足，候选只记录影子样本，暂不进入真实下单。",
            }
        if not raw:
            return {"category": "unknown", "label": "未记录事件原因"}
        if is_okx_temporary_service_error(raw):
            translated = EXECUTION_REASON_CLASSIFIER.translate_execution_error_text(raw)
            return {
                "category": "okx_transient_exchange_error",
                "label": translated or "OKX 交易所服务临时不可用，系统会稍后自动重试。",
            }
        translated = EXECUTION_REASON_CLASSIFIER.translate_execution_error_text(raw)
        if translated:
            return {"category": "okx_execution_error", "label": translated}
        if "missing execution result" in lower or "交易接口未返回执行结果" in raw:
            return {
                "category": "execution_missing_result",
                "label": "交易接口未返回执行结果，系统未拿到可确认的 OKX 订单回报。",
            }
        if any(token in lower for token in ("crowded_side_cap", "单边敞口", "单边敷口")):
            return {
                "category": "crowded_side_cap",
                "label": "单边持仓达到拥挤上限，本轮拒绝继续同方向开仓。",
            }
        if any(token in lower for token in ("max_position", "capacity", "满仓", "仓位已满")):
            return {
                "category": "capacity_block",
                "label": "仓位容量已满或接近上限，优先释放低质量持仓后再开新仓。",
            }
        if any(token in lower for token in ("fallback", "expert_integrity", "partial_batch")):
            return {
                "category": "expert_fallback_block",
                "label": "专家模型 fallback 或完整性不足，策略降低信任或阻断执行。",
            }
        label = str(sanitize_text(raw) or raw).strip()[:160]
        return {"category": "other", "label": label or "未记录事件原因"}

    def _problems(
        self,
        *,
        side_performance: dict[str, dict[str, Any]],
        open_pressure: dict[str, Any],
        decision_quality: dict[str, Any],
        shadow_feedback: dict[str, Any],
        trade_count: int,
        trade_count_target: int,
        net_pnl: float,
        small_win_count: int,
        large_loss_count: int,
        payoff_profile: dict[str, Any],
        avg_loss_hold_hours: float,
        event_feedback: dict[str, Any],
        reflection_feedback: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        problems: list[dict[str, Any]] = []
        root_causes: list[str] = []

        def add(key: str, severity: str, label: str, evidence: dict[str, Any]) -> None:
            label_overrides = {
                "reflection_negative_pnl": (
                    "\u7b56\u7565\u590d\u76d8\u7684\u8d39\u540e\u51c0\u6536\u76ca\u4e3a\u8d1f\uff0c"
                    "\u9700\u8981\u628a\u5f00\u4ed3\u8d28\u91cf\u548c\u5e73\u4ed3\u65f6\u673a\u5206\u5f00\u5f52\u56e0\u3002"
                ),
                "reflection_loss_hold_too_long": (
                    "\u7b56\u7565\u590d\u76d8\u663e\u793a\u4e8f\u635f\u4ed3\u62d6\u5ef6\u8fc7\u4e45\uff0c"
                    "\u6ee1\u4ed3\u65f6\u8981\u4f18\u5148\u91ca\u653e\u4f4e\u8d28\u91cf\u4e8f\u635f\u4ed3\u3002"
                ),
                "reflection_small_wins_large_losses": (
                    "\u7b56\u7565\u590d\u76d8\u540c\u6837\u6307\u5411\u5c0f\u76c8\u8fc7\u591a\u3001\u5927\u4e8f\u5b58\u5728\uff0c"
                    "\u8d62\u5bb6\u6301\u4ed3\u548c\u56de\u64a4\u4fdd\u62a4\u9700\u8981\u540c\u6b65\u4f18\u5316\u3002"
                ),
                "trade_reflection_mistakes": (
                    "\u7b56\u7565\u590d\u76d8\u5df2\u805a\u5408\u51fa\u91cd\u590d\u9519\u8bef\uff0c"
                    "\u5019\u9009\u7b56\u7565\u751f\u6210\u5fc5\u987b\u5438\u6536\u8fd9\u4e9b\u6539\u8fdb\u5efa\u8bae\u3002"
                ),
            }
            label = label_overrides.get(key, label)
            problems.append(
                {"key": key, "severity": severity, "label": label, "evidence": evidence}
            )
            root_causes.append(label)

        payoff_profile = _safe_dict(payoff_profile)
        reflection_payoff_profile = _safe_dict(reflection_feedback.get("payoff_profile"))
        payoff_profile_triggered = bool(payoff_profile.get("triggered"))
        reflection_payoff_triggered = bool(reflection_payoff_profile.get("triggered"))

        if trade_count < trade_count_target:
            add(
                "low_trade_count",
                "medium",
                "样本和开仓数量不足，不能让系统学成不开仓最安全",
                {
                    "trade_count": trade_count,
                    "target": trade_count_target,
                    "policy": "动态学习置信目标；不是开仓硬阈值",
                },
            )
        defensive_probe_shadow_count = _safe_int(
            event_feedback.get("profit_first_defensive_probe_shadow_count"),
            0,
        )
        if defensive_probe_shadow_count:
            add(
                "defensive_probe_shadow_loop",
                "high" if defensive_probe_shadow_count >= 3 else "medium",
                (
                    "Profit-First 防御探针正在把低收益小仓开仓转为影子样本；"
                    "策略生成不能继续只制造探针，应生成质量升级画像，让高质量信号走动态收益校验。"
                ),
                {
                    "profit_first_defensive_probe_shadow_count": defensive_probe_shadow_count,
                    "skip_kind_counts": _safe_dict(event_feedback.get("skip_kind_counts")),
                    "policy": (
                        "低收益探针继续变多不会带来真实训练样本；需要把生成目标从探针数量"
                        "切换到收益质量和正常 sizing 恢复"
                    ),
                },
            )
        if trade_count >= 3 and net_pnl < 0:
            add(
                "negative_realized_pnl",
                "high",
                "最近已平仓净收益为负，需要降低亏损仓占用并提高开仓质量",
                {"net_pnl": net_pnl, "trade_count": trade_count},
            )
        for side, bucket in side_performance.items():
            if bucket.get("state") == "degraded":
                add(
                    f"{side}_side_degraded",
                    "high",
                    f"{side} 方向近期真实平仓表现退化，开仓侧必须吸收反馈",
                    bucket,
                )
        if open_pressure.get("full_position_pressure") and open_pressure.get("losing_open_count"):
            add(
                "full_position_loss_pressure",
                "high",
                "满仓压力来自亏损持仓占位，应优先复盘并释放低质量亏损仓",
                open_pressure,
            )
        if open_pressure.get("low_quality_open_count"):
            add(
                "low_quality_position_pressure",
                "high",
                "低质量持仓正在占用仓位，应先释放低质量仓，再恢复新开仓探针。",
                open_pressure,
            )
        if (
            decision_quality.get("expert_integrity_blocks")
            or decision_quality.get("fallback_entry_rate", 0.0) >= 0.25
        ):
            add(
                "expert_fallback_overblocking",
                "medium",
                "专家 fallback 或完整性保护正在拦截开仓，需要用小仓探针而不是直接停摆",
                decision_quality,
            )
        missed_loop = _safe_dict(shadow_feedback.get("missed_opportunity_closed_loop"))
        if _safe_int(missed_loop.get("usable_group_count"), 0) > 0:
            add(
                "missed_opportunities",
                "medium",
                "影子复盘显示观望错过机会，需要让错过机会反馈影响后续开仓",
                shadow_feedback,
            )
        if (
            reflection_feedback.get("training_count", 0)
            and reflection_feedback.get("fee_adjusted_pnl", 0.0) < 0
        ):
            add(
                "reflection_negative_pnl",
                "high",
                "策略复盘的费后净收益为负，需要把开仓质量和平仓时机分开归因。",
                reflection_feedback,
            )
        if reflection_feedback.get(
            "avg_loss_hold_minutes", 0.0
        ) >= 180.0 and reflection_feedback.get("large_loss_count", 0):
            add(
                "reflection_loss_hold_too_long",
                "high",
                "策略复盘显示亏损仓拖延过久，满仓时要优先释放低质量亏损仓。",
                reflection_feedback,
            )
        if reflection_payoff_triggered:
            add(
                "reflection_small_wins_large_losses",
                "high",
                "策略复盘同样指向小盈过多、大亏存在，赢家持仓和回撤保护需要同步优化。",
                reflection_feedback,
            )
        if reflection_feedback.get("mistake_count", 0) >= 2:
            add(
                "trade_reflection_mistakes",
                "medium",
                "策略复盘已聚合出重复错误，候选策略生成必须吸收这些改进建议。",
                reflection_feedback,
            )
        if payoff_profile_triggered:
            add(
                "small_wins_large_losses",
                "high",
                "盈利仓过早小盈平仓，而亏损单损失更大",
                {
                    "small_win_count": small_win_count,
                    "large_loss_count": large_loss_count,
                    "payoff_profile": payoff_profile,
                    "policy": "dynamic_window_distribution_not_fixed_usdt_thresholds",
                },
            )
        if avg_loss_hold_hours >= 3.0 and large_loss_count:
            add(
                "loss_hold_too_long",
                "high",
                "亏损仓平均持有过久，平仓检查和亏损释放需要更主动",
                {"avg_loss_hold_hours": round(avg_loss_hold_hours, 4)},
            )
        if event_feedback.get("max_position_blocks"):
            add(
                "max_position_blocks",
                "high",
                "\u8fd1\u671f\u5f00\u4ed3\u673a\u4f1a\u88ab\u4ed3\u4f4d\u5bb9\u91cf\u963b\u65ad\uff0c\u9700\u8981\u4f18\u5148\u91ca\u653e\u4f4e\u8d28\u91cf\u4e8f\u635f\u4ed3\u518d\u627f\u63a5\u65b0\u98ce\u9669\u3002",
                event_feedback,
            )
        if event_feedback.get("fallback_blocks"):
            add(
                "event_fallback_blocks",
                "medium",
                "\u4e13\u5bb6 fallback \u6b63\u5728\u963b\u65ad\u6216\u964d\u7ea7\u51b3\u7b56\uff0c\u53ea\u80fd\u5728\u6838\u5fc3\u4e13\u5bb6\u53ef\u4fe1\u65f6\u4f7f\u7528\u63a2\u9488\u4ed3\u4f4d\u3002",
                event_feedback,
            )
        if event_feedback.get("execution_errors"):
            add(
                "execution_errors",
                "high",
                "\u6267\u884c\u5931\u8d25\u4e5f\u5c5e\u4e8e\u7b56\u7565\u53cd\u9988\uff0c\u9700\u8981\u548c AI \u5f00\u4ed3\u8d28\u91cf\u95ee\u9898\u5206\u5f00\u5f52\u56e0\u3002",
                event_feedback,
            )
        if (
            event_feedback.get("attributable_events")
            and event_feedback.get("attributable_event_coverage", 0.0) < 0.85
        ):
            add(
                "strategy_attribution_gap",
                "medium",
                "\u90e8\u5206\u7b56\u7565\u4e8b\u4ef6\u7f3a\u5c11\u753b\u50cf\u5f52\u56e0\uff0c\u5728\u4fe1\u4efb\u8c03\u5ea6\u7ed3\u8bba\u524d\u9700\u8981\u6301\u7eed\u8865\u9f50\u8bb0\u5f55\u4e0a\u4e0b\u6587\u3002",
                event_feedback,
            )
        return problems, root_causes


class StrategyCandidateGenerator:
    """Generate controlled profile candidates from feedback."""

    def generate(self, feedback: StrategyFeedback) -> list[StrategyProfile]:
        profiles = [self.baseline(feedback)]
        profiles.extend(self.rule_based_candidates(feedback))
        return self._dedupe(profiles)

    def rule_based_candidates(self, feedback: StrategyFeedback) -> list[StrategyProfile]:
        profiles: list[StrategyProfile] = []
        problem_keys = {item["key"] for item in feedback.problems}
        open_pressure = feedback.open_position_pressure
        totals = feedback.totals
        profit_first_feedback = _safe_dict(feedback.profit_first_runtime_feedback)
        missed_feedback = _safe_dict(profit_first_feedback.get("missed_opportunity_feedback"))
        lane_feedback = _safe_list(profit_first_feedback.get("lane_feedback"))
        exit_feedback = _safe_list(profit_first_feedback.get("exit_feedback"))
        trade_target = _safe_int(totals.get("trade_count_target"), default_min_trade_target())
        defensive_probe_shadow_loop = "defensive_probe_shadow_loop" in problem_keys
        missed_positive_shadow_pressure = bool(
            missed_feedback.get("diagnosis") == "system_over_conservative_review"
            or any(
                _safe_dict(row).get("entry_bias") == "expand_quality_entries"
                for row in lane_feedback
            )
        )
        tiny_probe_fee_drag = bool(
            any(_safe_dict(row).get("reason") == "position_too_small_fee_drag" for row in lane_feedback)
            or any(
                _safe_dict(row).get("exit_bias") == "keep_tiny_entries_shadow_only"
                for row in exit_feedback
            )
        )
        exit_too_early = any(
            _safe_dict(row).get("exit_bias") == "hold_winners_longer" for row in exit_feedback
        )
        exit_too_late = any(
            _safe_dict(row).get("exit_bias") == "cut_losers_faster" for row in exit_feedback
        )
        payoff_profile = _safe_dict(totals.get("payoff_profile"))
        reflection_payoff_profile = _safe_dict(
            feedback.reflection_feedback.get("payoff_profile")
        )
        payoff_repair_intensity = _clamp(
            max(
                _safe_float(payoff_profile.get("imbalance_score"), 0.0),
                _safe_float(reflection_payoff_profile.get("imbalance_score"), 0.0),
            ),
            0.0,
            1.0,
        )

        if defensive_probe_shadow_loop or tiny_probe_fee_drag or missed_positive_shadow_pressure:
            profiles.append(
                StrategyProfile(
                    profile_id="quality_entry_recovery",
                    version=1,
                    label="质量开仓恢复",
                    status="candidate",
                    source="feedback_generator",
                    description=(
                        "低收益探针已被 Profit-First 影子化或确认存在过度保守时，"
                        "不继续强制所有机会都走小仓；保留严格专家完整性，让收益质量"
                        "达标的信号恢复正常 sizing。"
                    ),
                    params={
                        "global_min_score_delta": 0.0,
                        "position_size_multiplier": 1.0,
                        "expert_integrity_mode": "strict_all_required",
                        "min_trade_count_target": trade_target,
                    },
                )
            )

        if (
            "expert_fallback_overblocking" in problem_keys
            or "missed_opportunities" in problem_keys
            or "trade_reflection_mistakes" in problem_keys
            or totals.get("training_trade_count", 0) < trade_target
            or (missed_positive_shadow_pressure and not tiny_probe_fee_drag)
        ):
            profiles.append(
                StrategyProfile(
                    profile_id="balanced_probe",
                    version=1,
                    label="\u5e73\u8861\u63a2\u9488",
                    status="candidate",
                    source="feedback_generator",
                    description=(
                        "\u5141\u8bb8\u6709\u9650\u975e\u6838\u5fc3\u4e13\u5bb6\u7f3a\u5931\uff0c"
                        "\u7528\u5c0f\u4ed3\u63a2\u9488\u6062\u590d\u6709\u6548\u5f00\u4ed3\u6837\u672c\uff0c"
                        "\u9632\u6b62\u7cfb\u7edf\u5b66\u6210\u4e0d\u5f00\u4ed3\u6700\u5b89\u5168\u3002"
                    ),
                    params={
                        "global_min_score_delta": -0.08,
                        "position_size_multiplier": (
                            ENTRY_RISK_SIZING_PARAMS.balanced_probe_position_size_multiplier
                        ),
                        "probe_fraction": 0.08,
                        "min_trade_count_target": trade_target,
                        "expert_integrity_mode": "balanced_probe_allow_one_non_core_missing",
                        "max_probe_size_pct": (
                            ENTRY_RISK_SIZING_PARAMS.balanced_probe_max_position_size_pct
                        ),
                        "fallback_tolerance": {
                            "allow_missing_non_core_experts": 1,
                            "core_experts_required": sorted(CORE_ENTRY_EXPERTS),
                            "non_core_experts": sorted(NON_CORE_ENTRY_EXPERTS),
                        },
                    },
                )
            )
        if (
            open_pressure.get("full_position_pressure")
            or _material_low_quality_pressure(open_pressure)
            or "loss_hold_too_long" in problem_keys
            or "reflection_loss_hold_too_long" in problem_keys
            or "reflection_negative_pnl" in problem_keys
            or exit_too_late
        ):
            profiles.append(
                StrategyProfile(
                    profile_id="loss_release",
                    version=1,
                    label="\u4e8f\u635f\u91ca\u653e",
                    status="candidate",
                    source="feedback_generator",
                    description=(
                        "\u6ee1\u4ed3\u6216\u4e8f\u635f\u4ed3\u5360\u7528\u65f6\uff0c"
                        "\u63d0\u9ad8\u6301\u4ed3\u590d\u76d8\u548c\u4f4e\u8d28\u91cf\u4e8f\u635f\u4ed3\u91ca\u653e\u4f18\u5148\u7ea7\u3002"
                    ),
                    params={
                        "global_min_score_delta": 0.02,
                        "loss_exit_aggressiveness": "high",
                        "full_position_release": True,
                        "position_review_priority_boost": 1.35,
                        "release_losing_positions_first": True,
                        "winner_hold_extension": "normal",
                    },
                )
            )
        if (
            "small_wins_large_losses" in problem_keys
            or "reflection_small_wins_large_losses" in problem_keys
            or exit_too_early
        ):
            profiles.append(
                StrategyProfile(
                    profile_id="winner_hold",
                    version=1,
                    label="\u8d62\u5bb6\u6301\u4ed3\u4f18\u5316",
                    status="candidate",
                    source="feedback_generator",
                    description=(
                        "\u51cf\u5c11\u4f18\u52bf\u4ed3\u4f4d\u8fc7\u65e9\u5c0f\u76c8\u5e73\u4ed3\uff0c"
                        "\u540c\u65f6\u7ee7\u7eed\u4fdd\u62a4\u56de\u64a4\u3002"
                    ),
                    params={
                        "global_min_score_delta": 0.0,
                        "winner_hold_extension": "high",
                        "profit_lock_min_usdt_multiplier": round(
                            _clamp(1.0 + payoff_repair_intensity * 0.72, 0.80, 1.80),
                            6,
                        ),
                        "payoff_repair_intensity": round(payoff_repair_intensity, 6),
                        "winner_hold_dynamic": {
                            "training": payoff_profile,
                            "reflection": reflection_payoff_profile,
                            "policy": "dynamic_window_distribution_not_fixed_usdt_thresholds",
                        },
                        "pullback_lock_enabled": True,
                        "loss_exit_aggressiveness": "normal",
                    },
                )
            )
        for side, bucket in feedback.side_performance.items():
            if bucket.get("state") == "degraded":
                opposite = "short" if side == "long" else "long"
                profiles.append(
                    StrategyProfile(
                        profile_id=f"{side}_side_recovery",
                        version=1,
                        label=f"{side} \u65b9\u5411\u6062\u590d",
                        status="candidate",
                        source="feedback_generator",
                        description=(
                            f"{side} \u4fa7\u771f\u5b9e\u8868\u73b0\u9000\u5316\uff0c"
                            "\u964d\u4f4e\u8be5\u65b9\u5411\u4ed3\u4f4d\u5e76\u63d0\u9ad8\u901a\u8fc7\u95e8\u69db\u3002"
                        ),
                        params={
                            "global_min_score_delta": 0.04,
                            "side_overrides": {
                                side: {
                                    "state": "degraded",
                                    "score_adjustment": -0.18,
                                    "min_score_delta": 0.22,
                                    "size_multiplier": 0.62,
                                    "reason": f"{side} side recent realized PnL is weak",
                                },
                                opposite: {
                                    "state": (
                                        "working"
                                        if feedback.side_performance.get(opposite, {}).get("pnl", 0)
                                        > 0
                                        else "neutral"
                                    ),
                                    "score_adjustment": 0.04,
                                    "min_score_delta": -0.03,
                                    "size_multiplier": 1.03,
                                    "reason": "opposite side can receive modest balance preference",
                                },
                            },
                            "side_weights": {side: 0.62, opposite: 1.03},
                        },
                    )
                )
        return profiles

    def from_structured_candidates(
        self,
        candidates: list[dict[str, Any]],
        feedback: StrategyFeedback,
    ) -> list[StrategyProfile]:
        profiles: list[StrategyProfile] = []
        for index, item in enumerate(candidates[:LLM_CANDIDATE_MAX_COUNT], start=1):
            if not isinstance(item, dict):
                continue
            params = self._sanitize_params(_safe_dict(item.get("params")))
            if not params:
                continue
            profile_id = self._sanitize_profile_id(
                item.get("profile_id") or item.get("id"), default=f"llm_candidate_{index}"
            )
            label = str(sanitize_text(item.get("label")) or f"LLM候选{index}")[:80]
            description = str(
                sanitize_text(item.get("description") or "大模型根据结构化反馈生成的受控参数候选。")
            )[:500]
            profiles.append(
                StrategyProfile(
                    profile_id=profile_id,
                    version=max(1, _safe_int(item.get("version"), 1)),
                    label=label,
                    status="candidate",
                    source="llm_structured_candidate",
                    description=description,
                    params={
                        **params,
                        "min_trade_count_target": max(
                            default_min_trade_target(),
                            _safe_int(
                                params.get("min_trade_count_target"),
                                _safe_int(
                                    feedback.totals.get("trade_count_target"),
                                    default_min_trade_target(),
                                ),
                            ),
                        ),
                    },
                    promotion=_safe_dict(item.get("promotion")),
                )
            )
        return self._dedupe(profiles)

    @staticmethod
    def _sanitize_profile_id(value: Any, *, default: str) -> str:
        text = str(value or default).strip().lower().replace(" ", "_")
        allowed = "".join(ch for ch in text if ch.isalnum() or ch in {"_", "-"})
        if not allowed:
            return default
        return allowed[:80]

    def _sanitize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        for key, value in params.items():
            if key not in ALLOWED_CANDIDATE_PARAM_KEYS:
                continue
            if key in BOUNDED_FLOAT_PARAM_RANGES:
                low, high = BOUNDED_FLOAT_PARAM_RANGES[key]
                clean[key] = round(min(max(_safe_float(value, 0.0), low), high), 6)
            elif key == "min_trade_count_target":
                clean[key] = max(default_min_trade_target(), min(_safe_int(value), 80))
            elif key == "expert_integrity_mode":
                mode = str(value or "")
                if mode in ALLOWED_EXPERT_INTEGRITY_MODES:
                    clean[key] = mode
            elif key == "loss_exit_aggressiveness":
                level = str(value or "")
                if level in ALLOWED_AGGRESSIVENESS:
                    clean[key] = level
            elif key == "winner_hold_extension":
                level = str(value or "")
                if level in ALLOWED_WINNER_HOLD:
                    clean[key] = level
            elif key == "winner_hold_dynamic":
                dynamic = self._sanitize_winner_hold_dynamic(_safe_dict(value))
                if dynamic:
                    clean[key] = dynamic
            elif key in {
                "full_position_release",
                "release_losing_positions_first",
                "pullback_lock_enabled",
            }:
                clean[key] = bool(value)
            elif key == "fallback_tolerance":
                clean[key] = self._sanitize_fallback_tolerance(_safe_dict(value))
            elif key == "side_overrides":
                sanitized = self._sanitize_side_overrides(_safe_dict(value))
                if sanitized:
                    clean[key] = sanitized
            elif key == "side_weights":
                sanitized_weights = self._sanitize_side_weights(_safe_dict(value))
                if sanitized_weights:
                    clean[key] = sanitized_weights
        if _safe_float(clean.get("probe_fraction"), 0.0) > 0:
            clean.setdefault(
                "position_size_multiplier",
                ENTRY_RISK_SIZING_PARAMS.balanced_probe_position_size_multiplier,
            )
            clean.setdefault(
                "max_probe_size_pct",
                ENTRY_RISK_SIZING_PARAMS.balanced_probe_max_position_size_pct,
            )
        return clean

    @staticmethod
    def _sanitize_winner_hold_dynamic(value: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in ("training", "reflection"):
            profile = _safe_dict(value.get(key))
            if not profile:
                continue
            result[key] = {
                profile_key: profile.get(profile_key)
                for profile_key in (
                    "sample_count",
                    "win_count",
                    "loss_count",
                    "avg_win",
                    "avg_loss",
                    "median_win",
                    "median_loss",
                    "dynamic_small_win_reference",
                    "dynamic_large_loss_reference",
                    "small_win_count",
                    "large_loss_count",
                    "small_win_ratio",
                    "large_loss_ratio",
                    "profit_factor",
                    "payoff_ratio",
                    "imbalance_score",
                    "triggered",
                    "policy",
                )
                if profile_key in profile
            }
        policy = str(value.get("policy") or "")[:120]
        if policy:
            result["policy"] = policy
        return result

    @staticmethod
    def _sanitize_fallback_tolerance(value: dict[str, Any]) -> dict[str, Any]:
        allowed_missing = max(0, min(_safe_int(value.get("allow_missing_non_core_experts"), 0), 2))
        core = [
            name
            for name in _safe_list(value.get("core_experts_required"))
            if name in CORE_ENTRY_EXPERTS
        ]
        non_core = [
            name
            for name in _safe_list(value.get("non_core_experts"))
            if name in NON_CORE_ENTRY_EXPERTS
        ]
        return {
            "allow_missing_non_core_experts": allowed_missing,
            "core_experts_required": sorted(set(core or CORE_ENTRY_EXPERTS)),
            "non_core_experts": sorted(set(non_core or NON_CORE_ENTRY_EXPERTS)),
        }

    @staticmethod
    def _sanitize_side_overrides(value: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for side in ("long", "short"):
            row = _safe_dict(value.get(side))
            if not row:
                continue
            result[side] = {
                "state": str(row.get("state") or "neutral")[:24],
                "score_adjustment": round(
                    min(max(_safe_float(row.get("score_adjustment"), 0.0), -0.35), 0.20),
                    6,
                ),
                "min_score_delta": round(
                    min(max(_safe_float(row.get("min_score_delta"), 0.0), -0.12), 0.35),
                    6,
                ),
                "size_multiplier": round(
                    min(max(_safe_float(row.get("size_multiplier"), 1.0), 0.10), 1.25),
                    6,
                ),
                "reason": str(row.get("reason") or "")[:240],
            }
        return result

    @staticmethod
    def _sanitize_side_weights(value: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for side in ("long", "short"):
            if side in value:
                result[side] = round(min(max(_safe_float(value.get(side), 1.0), 0.25), 1.40), 6)
        return result

    @staticmethod
    def _dedupe(profiles: list[StrategyProfile]) -> list[StrategyProfile]:
        result: list[StrategyProfile] = []
        seen: set[str] = set()
        for profile in profiles:
            if profile.profile_id in seen:
                continue
            result.append(profile)
            seen.add(profile.profile_id)
        return result

    @staticmethod
    def baseline(feedback: StrategyFeedback | None = None) -> StrategyProfile:
        target = default_min_trade_target()
        if feedback is not None:
            target = _safe_int(feedback.totals.get("trade_count_target"), target)
        return StrategyProfile(
            profile_id="baseline_current",
            version=1,
            label="\u7cfb\u7edf\u57fa\u7ebf",
            status="baseline",
            source="current_system",
            description=(
                "\u4fdd\u6301\u73b0\u6709\u7b56\u7565\uff0c"
                "\u53ea\u505a\u5f52\u56e0\u8bb0\u5f55\u548c\u4f4e\u4ea4\u6613\u91cf\u60e9\u7f5a\u8bc4\u4f30\u3002"
            ),
            params={
                "global_min_score_delta": 0.0,
                "position_size_multiplier": 1.0,
                "expert_integrity_mode": "strict_all_required",
                "min_trade_count_target": target,
            },
        )


class StrategyBacktester:
    """Score profiles with historical feedback and trade-count constraints."""

    def score(self, profile: StrategyProfile, feedback: StrategyFeedback) -> dict[str, Any]:
        totals = feedback.totals
        known_profiles = {
            "baseline_current",
            "balanced_probe",
            "loss_release",
            "winner_hold",
        }
        net_pnl = _safe_float(totals.get("net_pnl"), 0.0)
        trade_count = _safe_int(totals.get("training_trade_count"), 0)
        target = _safe_int(
            profile.params.get("min_trade_count_target"),
            _safe_int(totals.get("trade_count_target"), default_min_trade_target()),
        )
        trade_gap = max(target - trade_count, 0)
        low_trade_penalty = trade_gap * 1.25
        fee_estimate = max(trade_count * 0.08, abs(net_pnl) * 0.015)
        fee_adjusted_pnl = net_pnl - fee_estimate
        side_values = [_safe_dict(row) for row in feedback.side_performance.values()]
        losses = _safe_int(totals.get("loss_count"), 0)
        wins = _safe_int(totals.get("win_count"), 0)
        avg_pnl = net_pnl / trade_count if trade_count else 0.0
        avg_winner = (
            sum(_safe_float(row.get("profit"), 0.0) for row in side_values) / max(wins, 1)
            if wins
            else 0.0
        )
        avg_loser = (
            sum(_safe_float(row.get("loss"), 0.0) for row in side_values) / max(losses, 1)
            if losses
            else 0.0
        )
        side_pnls = [_safe_float(row.get("pnl"), 0.0) for row in side_values]
        max_drawdown = abs(min([0.0, net_pnl, *side_pnls]))
        consecutive_losses = max(
            (_safe_int(row.get("losses"), 0) for row in side_values), default=0
        )
        occupancy = _safe_float(feedback.open_position_pressure.get("usage_ratio"), 0.0)
        reflection = feedback.reflection_feedback
        missed_loop = _safe_dict(feedback.shadow_feedback.get("missed_opportunity_closed_loop"))
        missed_usable = _safe_int(missed_loop.get("usable_group_count"), 0)
        missed_opportunity_reduction = 0.0
        loss_release_speed = 0.0
        winner_avg_profit = avg_winner
        problem_keys = {item["key"] for item in feedback.problems}
        estimated_delta = 0.0
        matched_fixes: list[str] = []
        param_consumption = _candidate_param_consumption(profile.params)
        payoff_repair = _payoff_repair_profile(
            _safe_dict(totals.get("payoff_profile")),
            _safe_dict(reflection.get("payoff_profile")),
        )
        payoff_repair_intensity = _safe_float(
            profile.params.get("payoff_repair_intensity"),
            _safe_float(payoff_repair.get("imbalance_score"), 0.0),
        )

        if profile.profile_id == "balanced_probe":
            if "expert_fallback_overblocking" in problem_keys:
                estimated_delta += (
                    max(feedback.decision_quality.get("expert_integrity_blocks", 0), 1) * 0.35
                )
                matched_fixes.append("expert_fallback_overblocking")
            if "missed_opportunities" in problem_keys:
                estimated_delta += missed_usable * 0.18
                missed_opportunity_reduction = _safe_float(missed_usable, 0.0) * 0.18
                matched_fixes.append("missed_opportunities")
            if "trade_reflection_mistakes" in problem_keys:
                estimated_delta += min(_safe_int(reflection.get("mistake_count"), 0), 8) * 0.12
                matched_fixes.append("trade_reflection_mistakes")
            if trade_count < target:
                estimated_delta += (target - trade_count) * 0.42
                matched_fixes.append("low_trade_count")
                low_trade_penalty = max(low_trade_penalty - trade_gap * 0.95, 0.0)
        elif profile.profile_id == "loss_release":
            if "low_quality_position_pressure" in problem_keys:
                low_quality_count = _safe_int(
                    feedback.open_position_pressure.get("low_quality_open_count"),
                    0,
                )
                estimated_delta += max(low_quality_count, 1) * 0.9
                loss_release_speed = 1.0
                matched_fixes.append("low_quality_position_pressure")
            if "full_position_loss_pressure" in problem_keys:
                estimated_delta += (
                    abs(
                        _safe_float(
                            feedback.open_position_pressure.get("losing_unrealized_pnl"), 0.0
                        )
                    )
                    * 0.16
                )
                matched_fixes.append("full_position_loss_pressure")
            if "loss_hold_too_long" in problem_keys:
                estimated_delta += 1.4
                loss_release_speed = 1.0
                matched_fixes.append("loss_hold_too_long")
            if "reflection_loss_hold_too_long" in problem_keys:
                estimated_delta += min(
                    _safe_float(reflection.get("avg_loss_hold_minutes"), 0.0) / 180.0, 3.0
                )
                loss_release_speed = max(loss_release_speed, 1.0)
                matched_fixes.append("reflection_loss_hold_too_long")
            if "reflection_negative_pnl" in problem_keys:
                estimated_delta += min(
                    abs(_safe_float(reflection.get("fee_adjusted_pnl"), 0.0)) * 0.10, 2.5
                )
                matched_fixes.append("reflection_negative_pnl")
            low_trade_penalty = 0.0
        elif profile.profile_id == "winner_hold":
            if "small_wins_large_losses" in problem_keys:
                estimated_delta += min(
                    max(_safe_int(payoff_repair.get("sample_count"), 1), 1)
                    * max(_safe_float(payoff_repair.get("imbalance_score"), 0.0), 0.08)
                    * 0.55,
                    3.0,
                )
                winner_avg_profit = (
                    avg_winner * (1.0 + payoff_repair_intensity * 0.35)
                    if avg_winner
                    else 0.0
                )
                matched_fixes.append("small_wins_large_losses")
            if "reflection_small_wins_large_losses" in problem_keys:
                estimated_delta += min(
                    max(_safe_int(payoff_repair.get("sample_count"), 1), 1)
                    * max(_safe_float(payoff_repair.get("imbalance_score"), 0.0), 0.08)
                    * 0.45,
                    2.5,
                )
                winner_avg_profit = max(
                    winner_avg_profit,
                    avg_winner * (1.0 + payoff_repair_intensity * 0.28)
                    if avg_winner
                    else 0.0,
                )
                matched_fixes.append("reflection_small_wins_large_losses")
        elif profile.profile_id.endswith("_side_recovery"):
            side = profile.profile_id.removesuffix("_side_recovery")
            side_bucket = feedback.side_performance.get(side, {})
            if side_bucket.get("state") == "degraded":
                estimated_delta += abs(_safe_float(side_bucket.get("pnl"), 0.0)) * 0.12
                matched_fixes.append(f"{side}_side_degraded")
        elif profile.profile_id not in known_profiles:
            for fix_key, delta in self._generic_candidate_deltas(profile, feedback, problem_keys):
                estimated_delta += delta
                matched_fixes.append(fix_key)
            if (
                "controlled_entry_recovery" in matched_fixes
                or "defensive_probe_quality_recovery" in matched_fixes
            ):
                low_trade_penalty = max(low_trade_penalty - trade_gap * 0.95, 0.0)

        score = net_pnl + estimated_delta - low_trade_penalty
        score -= fee_estimate * 0.35
        score -= max_drawdown * 0.10
        score -= max(consecutive_losses - 3, 0) * 0.7
        score -= max(occupancy - 0.85, 0.0) * 2.0
        pass_gate = score >= fee_adjusted_pnl - 0.75 and (
            trade_count >= max(2, int(target * 0.35)) or profile.profile_id == "balanced_probe"
        )
        if profile.profile_id != "baseline_current" and not param_consumption[
            "has_consumed_runtime_params"
        ]:
            pass_gate = False
        if profile.profile_id in {"loss_release", "winner_hold"} and matched_fixes:
            pass_gate = True
        if profile.profile_id not in {"baseline_current", "balanced_probe"} and matched_fixes:
            pass_gate = True
        if profile.profile_id != "baseline_current" and not param_consumption[
            "has_consumed_runtime_params"
        ]:
            pass_gate = False
        if profile.profile_id == "baseline_current":
            pass_gate = True
            estimated_delta = 0.0
            score = net_pnl - low_trade_penalty
        return {
            "profile_id": profile.profile_id,
            "score": round(score, 6),
            "baseline_net_pnl": round(net_pnl, 6),
            "estimated_delta": round(estimated_delta, 6),
            "trade_count": trade_count,
            "trade_count_target": target,
            "low_trade_count_penalty": round(low_trade_penalty, 6),
            "trade_count_target_policy": "dynamic_advisory_learning_confidence",
            "trade_count_target_is_entry_gate": False,
            "matched_fixes": matched_fixes,
            "param_consumption": param_consumption,
            "consumed_runtime_params": param_consumption["consumed_runtime_params"],
            "unused_runtime_params": param_consumption["unused_runtime_params"],
            "fee_estimate": round(fee_estimate, 6),
            "fee_adjusted_pnl": round(fee_adjusted_pnl, 6),
            "max_drawdown": round(max_drawdown, 6),
            "consecutive_losses": consecutive_losses,
            "avg_pnl": round(avg_pnl, 6),
            "avg_winner": round(avg_winner, 6),
            "avg_loser": round(avg_loser, 6),
            "position_occupancy": round(occupancy, 6),
            "missed_opportunity_reduction": round(missed_opportunity_reduction, 6),
            "loss_release_speed": round(loss_release_speed, 6),
            "winner_avg_profit": round(winner_avg_profit, 6),
            "payoff_repair_profile": payoff_repair,
            "payoff_repair_intensity": round(payoff_repair_intensity, 6),
            "pass": bool(pass_gate),
            "notes": (
                "low trade count is penalized; a profile cannot win by refusing trades"
                if low_trade_penalty
                else "trade count constraint satisfied"
            ),
        }

    @staticmethod
    def _generic_candidate_deltas(
        profile: StrategyProfile,
        feedback: StrategyFeedback,
        problem_keys: set[str],
    ) -> list[tuple[str, float]]:
        """Estimate bounded LLM candidate impact from allowed parameters only."""

        params = profile.params
        deltas: list[tuple[str, float]] = []
        trade_count = _safe_int(feedback.totals.get("training_trade_count"), 0)
        target = _safe_int(params.get("min_trade_count_target"), default_min_trade_target())
        trade_gap = max(target - trade_count, 0)
        payoff_repair = _payoff_repair_profile(
            _safe_dict(feedback.totals.get("payoff_profile")),
            _safe_dict(feedback.reflection_feedback.get("payoff_profile")),
            *_safe_dict(params.get("winner_hold_dynamic")).values(),
        )
        opens_more = (
            _safe_float(params.get("global_min_score_delta"), 0.0) < 0
            or _safe_float(params.get("probe_fraction"), 0.0) > 0
            or _safe_float(params.get("position_size_multiplier"), 1.0) > 1.0
        )
        if opens_more and (
            problem_keys
            & {
                "low_trade_count",
                "missed_opportunities",
                "expert_fallback_overblocking",
                "event_fallback_blocks",
            }
        ):
            missed_loop = _safe_dict(feedback.shadow_feedback.get("missed_opportunity_closed_loop"))
            missed = _safe_int(missed_loop.get("usable_group_count"), 0)
            blocks = _safe_int(
                feedback.decision_quality.get("expert_integrity_blocks"), 0
            ) + _safe_int(feedback.event_feedback.get("fallback_blocks"), 0)
            deltas.append(
                (
                    "controlled_entry_recovery",
                    min(trade_gap * 0.32 + missed * 0.18 + blocks * 0.16, 4.0),
                )
            )
        defensive_probe_blocks = _safe_int(
            feedback.event_feedback.get("profit_first_defensive_probe_shadow_count"),
            0,
        )
        probe_cap_active = bool(
            _safe_float(params.get("probe_fraction"), 0.0) > 0
            or _safe_float(params.get("max_probe_size_pct"), 0.0) > 0
        )
        quality_recovery_profile = bool(
            profile.profile_id == "quality_entry_recovery"
            or (
                not probe_cap_active
                and _safe_float(params.get("position_size_multiplier"), 1.0) >= 1.0
                and str(params.get("expert_integrity_mode") or "strict_all_required")
                == "strict_all_required"
            )
        )
        if (
            quality_recovery_profile
            and "defensive_probe_shadow_loop" in problem_keys
            and defensive_probe_blocks > 0
        ):
            deltas.append(
                (
                    "defensive_probe_quality_recovery",
                    min(defensive_probe_blocks * 0.28 + trade_gap * 0.22, 4.0),
                )
            )
        releases_losers = (
            bool(
                params.get("full_position_release") or params.get("release_losing_positions_first")
            )
            or str(params.get("loss_exit_aggressiveness") or "") == "high"
        )
        if releases_losers and (
            problem_keys
            & {
                "full_position_loss_pressure",
                "max_position_blocks",
                "loss_hold_too_long",
                "reflection_loss_hold_too_long",
                "reflection_negative_pnl",
            }
        ):
            loss_pnl = abs(
                _safe_float(feedback.open_position_pressure.get("losing_unrealized_pnl"), 0.0)
            )
            loss_minutes = _safe_float(
                feedback.reflection_feedback.get("avg_loss_hold_minutes"), 0.0
            )
            deltas.append(
                (
                    "loss_release_from_candidate",
                    min(loss_pnl * 0.12 + loss_minutes / 240.0, 4.0),
                )
            )
        holds_winners = (
            str(params.get("winner_hold_extension") or "") == "high"
            or _safe_float(params.get("profit_lock_min_usdt_multiplier"), 1.0) > 1.05
        )
        if holds_winners and (
            problem_keys & {"small_wins_large_losses", "reflection_small_wins_large_losses"}
        ):
            deltas.append(
                (
                    "winner_hold_from_candidate",
                    min(
                        max(_safe_int(payoff_repair.get("sample_count"), 1), 1)
                        * max(
                            _safe_float(
                                params.get("payoff_repair_intensity"),
                                _safe_float(payoff_repair.get("imbalance_score"), 0.0),
                            ),
                            _safe_float(payoff_repair.get("imbalance_score"), 0.0),
                            0.08,
                        )
                        * 0.42,
                        3.0,
                    ),
                )
            )
        side_overrides = _safe_dict(params.get("side_overrides"))
        for side, override in side_overrides.items():
            bucket = _safe_dict(feedback.side_performance.get(str(side)))
            if (
                bucket.get("state") == "degraded"
                and _safe_float(_safe_dict(override).get("size_multiplier"), 1.0) < 1.0
            ):
                deltas.append(
                    (
                        f"{side}_side_candidate_recovery",
                        min(abs(_safe_float(bucket.get("pnl"), 0.0)) * 0.10, 3.0),
                    )
                )
        return [(key, round(max(delta, 0.0), 6)) for key, delta in deltas if delta > 0]


class StrategyScheduler:
    """Choose the active profile and runtime overrides."""

    def __init__(self, state_store: StrategyLearningStateStore | None = None) -> None:
        self.state_store = state_store or StrategyLearningStateStore()

    def schedule(
        self,
        profiles: list[StrategyProfile],
        feedback: StrategyFeedback,
        backtest_rows: list[dict[str, Any]],
    ) -> StrategySchedule:
        disabled = self.state_store.disabled_profiles()
        shadow_validation = self._shadow_validation(profiles, feedback)
        shadow_by_id = {
            str(row.get("profile_id")): row
            for row in _safe_list(shadow_validation.get("rows"))
            if isinstance(row, dict)
        }
        reconsidered_profiles: list[str] = []
        active_disabled: dict[str, Any] = {}
        for profile_id, meta in disabled.items():
            can_reconsider = self._auto_disabled_profile_can_be_reconsidered(
                profile=next((item for item in profiles if item.profile_id == profile_id), None),
                disabled_meta=_safe_dict(meta),
                feedback=feedback,
                backtest_rows=backtest_rows,
                shadow_by_id=shadow_by_id,
            )
            if can_reconsider:
                reconsidered_profiles.append(str(profile_id))
            else:
                active_disabled[str(profile_id)] = meta
        disabled = active_disabled
        disabled_ids = sorted(disabled.keys())
        disabled_reasons = {
            profile_id: {
                "reason": str(_safe_dict(meta).get("reason") or ""),
                "auto": bool(_safe_dict(meta).get("auto")),
                "disabled_until": str(_safe_dict(meta).get("disabled_until") or ""),
            }
            for profile_id, meta in disabled.items()
        }
        available = [profile for profile in profiles if profile.profile_id not in disabled]
        by_id = {profile.profile_id: profile for profile in available}
        state = self.state_store.load()
        manual_profile_id = str(state.get("manual_active_profile") or "")
        if manual_profile_id == "baseline_current":
            manual_profile_id = ""
        manual_lock_active = bool(manual_profile_id and manual_profile_id in by_id)

        selected = by_id.get("baseline_current") or StrategyCandidateGenerator.baseline()
        reason = "默认使用系统基线兜底。"
        problem_keys = {item["key"] for item in feedback.problems}
        blocked_candidate_count = sum(1 for profile in profiles if profile.profile_id in disabled)
        open_pressure = _safe_dict(feedback.open_position_pressure)
        material_release_pressure = _material_release_pressure(
            open_pressure,
            problem_keys,
        )
        pressure_guard_active = self._should_hold_baseline_due_to_pressure(
            feedback=feedback,
            disabled=disabled,
            problem_keys=problem_keys,
        )
        loss_release_profile = by_id.get("loss_release")

        if manual_lock_active:

            selected = by_id[manual_profile_id]
            reason = f"人工指定策略画像 {manual_profile_id}。"
        elif pressure_guard_active and loss_release_profile is not None:
            selected = loss_release_profile
            reason = (
                "Strategy guard is active: release low-quality positions while allowing "
                "bounded high-quality probes."
            )
        elif pressure_guard_active:
            selected = by_id.get("baseline_current", StrategyCandidateGenerator.baseline())
            reason = (
                "Strategy guard is active: keep baseline sizing, release low-quality positions, "
                "and allow only bounded high-quality probes."
            )
        elif (
            "low_quality_position_pressure" in problem_keys
            and material_release_pressure
            and "loss_release" in by_id
        ):
            selected = by_id["loss_release"]
            reason = "检测到低质量持仓占用仓位，优先调度亏损释放画像。"
        elif (
            ("full_position_loss_pressure" in problem_keys or "max_position_blocks" in problem_keys)
            and material_release_pressure
            and "loss_release" in by_id
        ):
            selected = by_id["loss_release"]
            reason = "检测到满仓和亏损仓占位，优先调度亏损释放画像。"
        elif (
            (
                "reflection_loss_hold_too_long" in problem_keys
                or "reflection_negative_pnl" in problem_keys
            )
            and material_release_pressure
            and "loss_release" in by_id
        ):
            selected = by_id["loss_release"]
            reason = "策略复盘显示费后亏损或亏损仓拖延过久，调度亏损释放画像。"
        elif "defensive_probe_shadow_loop" in problem_keys and "quality_entry_recovery" in by_id:
            selected = by_id["quality_entry_recovery"]
            reason = (
                "Profit-First 已多次把低收益探针转为影子样本，调度质量开仓恢复画像，"
                "不再让低样本问题继续生成探针闭环。"
            )
        else:
            degraded_sides = [
                side
                for side, bucket in feedback.side_performance.items()
                if bucket.get("state") == "degraded"
            ]
            for side in degraded_sides:
                candidate_id = f"{side}_side_recovery"
                if candidate_id in by_id:
                    selected = by_id[candidate_id]
                    reason = f"{side} 方向真实平仓表现退化，调度方向恢复画像。"
                    break
            else:
                if (
                    "expert_fallback_overblocking" in problem_keys
                    or "event_fallback_blocks" in problem_keys
                    or "missed_opportunities" in problem_keys
                    or "trade_reflection_mistakes" in problem_keys
                    or feedback.totals.get("training_trade_count", 0)
                    < _safe_int(
                        feedback.totals.get("trade_count_target"),
                        default_min_trade_target(),
                    )
                ) and "balanced_probe" in by_id:
                    selected = by_id["balanced_probe"]
                    reason = "开仓样本不足或专家 fallback 拦截偏多，调度平衡探针画像。"
                elif (
                    "small_wins_large_losses" in problem_keys
                    or "reflection_small_wins_large_losses" in problem_keys
                ) and "winner_hold" in by_id:
                    selected = by_id["winner_hold"]
                    reason = "盈利仓小盈过多且大亏存在，调度赢家持仓优化画像。"

        if not manual_lock_active and not pressure_guard_active:
            structured = self._best_structured_candidate(
                available=available,
                selected=selected,
                backtest_rows=backtest_rows,
                shadow_validation=shadow_validation,
            )
            if structured is not None:
                selected = structured

        selected_score = next(
            (row for row in backtest_rows if row.get("profile_id") == selected.profile_id),
            {},
        )
        selected_shadow = _safe_dict(shadow_by_id.get(selected.profile_id))
        selection_failed = (
            selected_score.get("pass") is False or selected_shadow.get("eligible") is False
        )
        if selected.profile_id != "baseline_current" and selection_failed:
            selected = by_id.get("baseline_current", StrategyCandidateGenerator.baseline())
            reason = "候选画像未通过交易数量/历史评分约束，自动回滚到系统基线兜底。"
            selected_score = next(
                (row for row in backtest_rows if row.get("profile_id") == selected.profile_id),
                {},
            )

        reason = self._readable_schedule_reason(
            selected_profile_id=selected.profile_id,
            manual_lock_active=manual_lock_active,
            manual_profile_id=manual_profile_id,
            problem_keys=problem_keys,
            score_failed=bool(selection_failed),
            pressure_guard_active=pressure_guard_active,
            selected_source=selected.source,
            matched_fixes=_safe_list(selected_score.get("matched_fixes")),
            disabled_profile_reasons=disabled_reasons,
            blocked_candidate_count=blocked_candidate_count,
            reconsidered_profiles=reconsidered_profiles,
        )
        runtime = self._runtime(selected, feedback)
        rollback = self._rollback(selected, feedback, selected_score)
        return StrategySchedule(
            active_profile=selected,
            reason=reason,
            runtime=runtime,
            rollback=rollback,
            candidates=[profile.to_dict() for profile in profiles],
            backtest={"rows": backtest_rows},
            shadow_validation=shadow_validation,
            probe=self._probe(selected, feedback),
            disabled_profiles=disabled_ids,
            scheduler_mode="manual" if manual_lock_active else "auto",
            manual_profile_id=manual_profile_id if manual_lock_active else "",
            disabled_profile_reasons=disabled_reasons,
            reconsidered_profiles=sorted(reconsidered_profiles),
            blocked_candidate_count=blocked_candidate_count,
        )

    @staticmethod
    def _should_hold_baseline_due_to_pressure(
        *,
        feedback: StrategyFeedback,
        disabled: dict[str, Any],
        problem_keys: set[str],
    ) -> bool:
        """Keep baseline while pressure remains after an automatic runtime rollback."""

        if not disabled:
            return False
        auto_disabled = any(
            bool(_safe_dict(meta).get("auto"))
            or str(_safe_dict(meta).get("reason") or "").startswith("auto_runtime_guard:")
            for meta in disabled.values()
        )
        if not auto_disabled:
            return False
        open_pressure = _safe_dict(feedback.open_position_pressure)
        pressure_active = _material_release_pressure(open_pressure, problem_keys)
        net_pnl = _safe_float(feedback.totals.get("net_pnl"), 0.0)
        has_current_pressure = bool(
            open_pressure.get("full_position_pressure")
            or open_pressure.get("fragmentation_pressure")
            or _material_low_quality_pressure(open_pressure)
            or _safe_int(open_pressure.get("low_quality_open_count"), 0) > 0
        )
        return bool(pressure_active and has_current_pressure and net_pnl < 0.0)

    @staticmethod
    def _auto_disabled_profile_can_be_reconsidered(
        *,
        profile: StrategyProfile | None,
        disabled_meta: dict[str, Any],
        feedback: StrategyFeedback,
        backtest_rows: list[dict[str, Any]],
        shadow_by_id: dict[str, Any],
    ) -> bool:
        """Let auto-disabled probe profiles re-enter when they target the active issue."""

        if profile is None or profile.profile_id == "baseline_current":
            return False
        reason = str(disabled_meta.get("reason") or "")
        is_auto = bool(disabled_meta.get("auto")) or reason.startswith("auto_runtime_guard:")
        if not is_auto:
            return False
        backtest = _safe_dict(
            next(
                (row for row in backtest_rows if row.get("profile_id") == profile.profile_id),
                {},
            )
        )
        shadow = _safe_dict(shadow_by_id.get(profile.profile_id))
        if backtest.get("pass") is False or shadow.get("eligible") is False:
            return False
        matched = {str(item) for item in _safe_list(backtest.get("matched_fixes"))}
        problem_keys = {
            str(item.get("key")) for item in feedback.problems if isinstance(item, dict)
        }
        open_pressure = _safe_dict(feedback.open_position_pressure)
        pressure_active = _material_release_pressure(open_pressure, problem_keys)
        if (
            "recent_net_pnl_guard" in reason
            and _safe_float(feedback.totals.get("net_pnl"), 0.0) < 0.0
            and pressure_active
        ):
            return False
        model_health_recovered = bool(
            _safe_dict(feedback.decision_quality).get("model_health_recovered")
        )
        fallback_reason_only = bool(
            any(token in reason for token in ("fallback_dependency_guard", "execution_error_guard"))
            and _safe_float(feedback.decision_quality.get("recent_fallback_entry_rate"), 1.0) < 0.20
            and _safe_int(feedback.decision_quality.get("recent_expert_integrity_blocks"), 0) == 0
            and _safe_int(feedback.decision_quality.get("recent_zero_second_entry_decisions"), 0)
            == 0
        )
        solves_fallback = bool(
            matched
            & {
                "expert_fallback_overblocking",
                "missed_opportunities",
                "low_trade_count",
                "controlled_entry_recovery",
                "trade_reflection_mistakes",
            }
            or shadow.get("would_reduce_blocks")
            or (
                shadow.get("would_increase_entries")
                and bool(
                    problem_keys
                    & {"expert_fallback_overblocking", "missed_opportunities", "low_trade_count"}
                )
            )
        )
        small_probe = _safe_float(profile.params.get("probe_fraction"), 0.0) > 0
        releases_losers = bool(
            profile.profile_id == "loss_release"
            or profile.params.get("full_position_release")
            or profile.params.get("release_losing_positions_first")
            or str(profile.params.get("loss_exit_aggressiveness") or "") == "high"
        )
        solves_loss_pressure = bool(
            releases_losers
            and (
                matched
                & {
                    "full_position_loss_pressure",
                    "max_position_blocks",
                    "low_quality_position_pressure",
                    "loss_hold_too_long",
                    "reflection_loss_hold_too_long",
                    "reflection_negative_pnl",
                    "loss_release_from_candidate",
                }
                or shadow.get("would_release_losers")
                or bool(
                    problem_keys
                    & {
                        "full_position_loss_pressure",
                        "max_position_blocks",
                        "loss_hold_too_long",
                        "reflection_loss_hold_too_long",
                    }
                )
            )
        )
        side_recovery = bool(profile.profile_id.endswith("_side_recovery"))
        solves_side_degradation = bool(
            side_recovery
            and any(key.endswith("_side_degraded") for key in problem_keys)
            and _safe_list(backtest.get("matched_fixes"))
        )
        solves_quality_recovery = bool(
            (
                "defensive_probe_quality_recovery" in matched
                or shadow.get("would_restore_quality_entries")
            )
            and "defensive_probe_shadow_loop" in problem_keys
        )
        return bool(
            (small_probe and solves_fallback)
            or solves_loss_pressure
            or solves_side_degradation
            or solves_quality_recovery
            or (
                model_health_recovered
                and fallback_reason_only
                and (solves_fallback or solves_loss_pressure or solves_quality_recovery)
            )
        )

    @staticmethod
    def _best_structured_candidate(
        *,
        available: list[StrategyProfile],
        selected: StrategyProfile,
        backtest_rows: list[dict[str, Any]],
        shadow_validation: dict[str, Any],
    ) -> StrategyProfile | None:
        """Allow bounded LLM-generated profiles to enter auto scheduling after validation."""

        backtest_by_id = {
            str(row.get("profile_id")): row for row in backtest_rows if isinstance(row, dict)
        }
        shadow_by_id = {
            str(row.get("profile_id")): row
            for row in _safe_list(_safe_dict(shadow_validation).get("rows"))
            if isinstance(row, dict)
        }
        selected_score = _safe_float(
            _safe_dict(backtest_by_id.get(selected.profile_id)).get("score"), 0.0
        )
        best: StrategyProfile | None = None
        best_score = selected_score
        for profile in available:
            if profile.source != "llm_structured_candidate":
                continue
            backtest = _safe_dict(backtest_by_id.get(profile.profile_id))
            shadow = _safe_dict(shadow_by_id.get(profile.profile_id))
            if backtest.get("pass") is False or shadow.get("eligible") is False:
                continue
            if not _safe_list(backtest.get("matched_fixes")):
                continue
            score = _safe_float(backtest.get("score"), 0.0) + _safe_float(
                shadow.get("shadow_score"), 0.0
            )
            if score > best_score + 0.15:
                best = profile
                best_score = score
        return best

    @staticmethod
    def _readable_schedule_reason(
        *,
        selected_profile_id: str,
        manual_lock_active: bool,
        manual_profile_id: str,
        problem_keys: set[str],
        score_failed: bool,
        pressure_guard_active: bool = False,
        selected_source: str = "",
        matched_fixes: list[Any] | None = None,
        disabled_profile_reasons: dict[str, Any] | None = None,
        blocked_candidate_count: int = 0,
        reconsidered_profiles: list[str] | None = None,
    ) -> str:
        if score_failed and selected_profile_id == "baseline_current":
            return "候选策略未通过交易数量或历史评分约束，自动回退到系统基线兜底。"
        if pressure_guard_active and selected_profile_id == "baseline_current":
            return (
                "策略护栏已触发自动回滚，且满仓/碎片化压力仍在；暂用系统基线，"
                "优先释放已有低质量仓位，只允许高质量小仓探针。"
            )
        if pressure_guard_active and selected_profile_id == "loss_release":
            return "策略护栏检测到满仓/碎片化压力，自动调度亏损释放画像，并保留高质量小仓探针。"
        if reconsidered_profiles and selected_profile_id in reconsidered_profiles:
            return (
                f"模型服务已恢复且候选重新通过回测/影子验证，自动解锁并调度 {selected_profile_id}。"
            )
        if manual_lock_active:
            return f"人工指定策略画像 {manual_profile_id}，自动调度暂不覆盖。"
        if selected_profile_id == "loss_release":
            if {"full_position_loss_pressure", "max_position_blocks"} & problem_keys:
                return "检测到满仓压力或亏损仓占位，自动调度到亏损释放画像。"
            return "策略复盘显示费后亏损或亏损仓拖延过久，自动调度到亏损释放画像。"
        if selected_profile_id.endswith("_side_recovery"):
            side = selected_profile_id.replace("_side_recovery", "")
            side_label = "多单" if side == "long" else "空单" if side == "short" else side
            return f"{side_label}方向近期表现退化，自动调度到方向恢复画像。"
        if selected_profile_id == "balanced_probe":
            return (
                "开仓样本不足、专家 fallback 拦截或影子复盘错过机会偏多，自动调度到平衡探针画像。"
            )
        if selected_profile_id == "quality_entry_recovery":
            return (
                "Profit-First 已把低收益探针转为影子样本；自动调度质量开仓恢复画像，"
                "取消策略学习层的探针仓位上限，让高质量信号继续接受动态收益和风控校验。"
            )
        if selected_profile_id == "winner_hold":
            return "盈利仓小盈过多且大亏存在，自动调度到赢家持仓优化画像。"
        if selected_source == "llm_structured_candidate":
            fixes = ", ".join(str(item) for item in (matched_fixes or [])[:4])
            suffix = f"，命中反馈：{fixes}" if fixes else ""
            return f"结构化候选已通过回测和影子验证，自动进入受控策略调度{suffix}。"
        if selected_profile_id == "baseline_current" and blocked_candidate_count:
            reasons = _safe_dict(disabled_profile_reasons)
            top = ", ".join(
                f"{profile_id}:{_safe_dict(meta).get('reason', '')}"
                for profile_id, meta in list(reasons.items())[:3]
            )
            if top:
                return f"检测到策略问题，但 {blocked_candidate_count} 个候选仍被护栏禁用，暂用系统基线；禁用原因：{top}"
            return f"检测到策略问题，但 {blocked_candidate_count} 个候选不可用，暂用系统基线。"
        return "自动调度未发现需要切换的高优先级问题，使用系统基线兜底。"

    def _runtime(self, profile: StrategyProfile, feedback: StrategyFeedback) -> dict[str, Any]:
        params = profile.params
        roster = self._runtime_roster(profile, feedback)
        entry_filters = self._runtime_entry_filters(profile, feedback, roster)
        runtime = {
            "profile_id": profile.profile_id,
            "profile_version": profile.version,
            "global_min_score_delta": _safe_float(params.get("global_min_score_delta"), 0.0),
            "position_size_multiplier": _safe_float(params.get("position_size_multiplier"), 1.0),
            "probe_fraction": _safe_float(params.get("probe_fraction"), 0.0),
            "max_probe_size_pct": _safe_float(params.get("max_probe_size_pct"), 0.0),
            "expert_integrity_mode": str(
                params.get("expert_integrity_mode") or "strict_all_required"
            ),
            "side_overrides": _safe_dict(params.get("side_overrides")),
            "side_weights": _safe_dict(params.get("side_weights")),
            "loss_exit_aggressiveness": params.get("loss_exit_aggressiveness", "normal"),
            "winner_hold_extension": params.get("winner_hold_extension", "normal"),
            "profit_lock_min_usdt_multiplier": _safe_float(
                params.get("profit_lock_min_usdt_multiplier"),
                1.0,
            ),
            "payoff_repair_intensity": _safe_float(
                params.get("payoff_repair_intensity"),
                0.0,
            ),
            "winner_hold_dynamic": _safe_dict(params.get("winner_hold_dynamic")),
            "pullback_lock_enabled": bool(params.get("pullback_lock_enabled")),
            "full_position_release": bool(params.get("full_position_release")),
            "release_losing_positions_first": bool(params.get("release_losing_positions_first")),
            "position_review_priority_boost": _safe_float(
                params.get("position_review_priority_boost"), 1.0
            ),
            "target_position_groups": roster["target_position_groups"],
            "target_open_position_groups": roster["target_position_groups"],
            "max_open_positions": roster["max_open_positions"],
            "rotation_slots": roster["rotation_slots"],
            "release_target_groups": roster["release_target_groups"],
            "position_review_max_groups": roster["position_review_max_groups"],
            "position_high_risk_max_groups": roster["position_high_risk_max_groups"],
            "position_urgent_exit_max_groups": roster["position_urgent_exit_max_groups"],
            "roster_fill_market_symbol_min": roster["roster_fill_market_symbol_min"],
            "capacity_policy_reason": roster["reason"],
            "analysis_budget": {
                "position_max_groups": roster["position_review_max_groups"],
                "position_high_risk_max_groups": roster["position_high_risk_max_groups"],
                "position_urgent_exit_max_groups": roster["position_urgent_exit_max_groups"],
                "roster_fill_market_symbol_min": roster["roster_fill_market_symbol_min"],
            },
            "entry_filters": entry_filters.to_dict(),
            "min_entry_volume_ratio": entry_filters.min_entry_volume_ratio,
            "min_entry_adx": entry_filters.min_entry_adx,
            "entry_filters_are_hard_gate": False,
            "training_trade_count": feedback.totals.get("training_trade_count", 0),
            "low_trade_count_penalty": bool(feedback.totals.get("low_trade_count_penalty")),
        }
        return self._apply_profit_first_runtime_feedback(runtime, profile, feedback)

    @staticmethod
    def _apply_profit_first_runtime_feedback(
        runtime: dict[str, Any],
        profile: StrategyProfile,
        feedback: StrategyFeedback,
    ) -> dict[str, Any]:
        profit_first = _safe_dict(feedback.profit_first_runtime_feedback)
        if not _profit_first_feedback_can_influence_context(profit_first):
            return runtime

        adjusted = dict(runtime)
        entry_filters = _safe_dict(adjusted.get("entry_filters"))
        params = STRATEGY_LEARNING_PARAMS
        reasons: list[str] = []
        missed = _safe_dict(profit_first.get("missed_opportunity_feedback"))
        lane_feedback = _safe_list(profit_first.get("lane_feedback"))
        size_feedback = _safe_list(profit_first.get("size_feedback"))
        exit_feedback = _safe_list(profit_first.get("exit_feedback"))

        if (
            missed.get("entry_bias") == "expand_quality_entries"
            or any(_safe_dict(row).get("entry_bias") == "expand_quality_entries" for row in lane_feedback)
        ) and entry_filters:
            volume_ratio = _clamp(
                _safe_float(entry_filters.get("min_entry_volume_ratio"), params.entry_volume_ratio_max)
                * 0.94,
                params.entry_volume_ratio_min,
                params.entry_volume_ratio_max,
            )
            adx = _clamp(
                _safe_float(entry_filters.get("min_entry_adx"), params.entry_adx_max) * 0.94,
                params.entry_adx_min,
                params.entry_adx_max,
            )
            entry_filters = {
                **entry_filters,
                "min_entry_volume_ratio": round(volume_ratio, 4),
                "min_entry_adx": round(adx, 2),
                "reason": ", ".join(
                    part
                    for part in (
                        str(entry_filters.get("reason") or "").strip(),
                        "profit_first_missed_positive_shadow_relaxation",
                    )
                    if part
                ),
            }
            adjusted["entry_filters"] = entry_filters
            adjusted["min_entry_volume_ratio"] = entry_filters["min_entry_volume_ratio"]
            adjusted["min_entry_adx"] = entry_filters["min_entry_adx"]
            reasons.append("missed_positive_shadow_relaxed_entry_reference")

        fee_drag_count = sum(
            _safe_int(_safe_dict(row).get("count"))
            for row in exit_feedback
            if _safe_dict(row).get("exit_bias") == "keep_tiny_entries_shadow_only"
        )
        fee_drag_count += sum(
            _safe_int(_safe_dict(_safe_dict(row).get("evidence")).get("position_too_small_fee_drag"))
            for row in size_feedback
            if _safe_dict(row).get("sizing_bias") == "reduce_weak_or_fee_drag_size"
        )
        if fee_drag_count > 0 and _safe_float(adjusted.get("probe_fraction"), 0.0) > 0:
            drag_pressure = min(0.30, fee_drag_count * 0.04)
            adjusted["probe_fraction"] = round(
                max(_safe_float(adjusted.get("probe_fraction"), 0.0) * (1.0 - drag_pressure), 0.0),
                6,
            )
            adjusted["max_probe_size_pct"] = round(
                max(
                    _safe_float(adjusted.get("max_probe_size_pct"), 0.0)
                    * (1.0 - min(0.35, fee_drag_count * 0.05)),
                    0.0,
                ),
                6,
            )
            reasons.append("fee_drag_feedback_keeps_weak_probe_small")

        hold_count = sum(
            _safe_int(_safe_dict(row).get("count"))
            for row in exit_feedback
            if _safe_dict(row).get("exit_bias") == "hold_winners_longer"
        )
        if hold_count > 0:
            hold_strength = min(1.0, hold_count / max(_safe_int(feedback.totals.get("training_trade_count"), 0), 1))
            adjusted["winner_hold_extension"] = "high"
            adjusted["pullback_lock_enabled"] = True
            adjusted["profit_lock_min_usdt_multiplier"] = round(
                _clamp(
                    max(
                        _safe_float(adjusted.get("profit_lock_min_usdt_multiplier"), 1.0),
                        1.0 + hold_strength * 0.18,
                    ),
                    0.80,
                    1.80,
                ),
                6,
            )
            adjusted["payoff_repair_intensity"] = round(
                _clamp(
                    max(
                        _safe_float(adjusted.get("payoff_repair_intensity"), 0.0),
                        hold_strength * 0.60,
                    ),
                    0.0,
                    1.0,
                ),
                6,
            )
            reasons.append("exit_feedback_requests_longer_winner_hold")

        cut_loss_count = sum(
            _safe_int(_safe_dict(row).get("count"))
            for row in exit_feedback
            if _safe_dict(row).get("exit_bias") == "cut_losers_faster"
        )
        if cut_loss_count > 0:
            cut_strength = min(1.0, cut_loss_count / max(_safe_int(feedback.totals.get("training_trade_count"), 0), 1))
            adjusted["loss_exit_aggressiveness"] = "high"
            adjusted["position_review_priority_boost"] = round(
                max(
                    _safe_float(adjusted.get("position_review_priority_boost"), 1.0),
                    1.0 + cut_strength * 0.30,
                ),
                6,
            )
            reasons.append("exit_feedback_requests_faster_loser_release")

        adjusted["profit_first_context"] = {
            "profile_id": profile.profile_id,
            "objective": profit_first.get("objective"),
            "objective_basis": _safe_dict(profit_first.get("objective_basis")),
            "strategy_profile_feedback": _safe_list(profit_first.get("strategy_profile_feedback"))[:8],
            "source_weight_feedback": _safe_list(profit_first.get("source_weight_feedback"))[:8],
            "lane_feedback": lane_feedback[:8],
            "size_feedback": size_feedback[:8],
            "missed_opportunity_feedback": missed,
            "exit_feedback": exit_feedback[:8],
            "applied_reasons": reasons,
            "policy": "bounded_strategy_context_feedback_only",
        }
        adjusted["profit_first_runtime_feedback_applied"] = bool(
            reasons or adjusted["profit_first_context"]
        )
        return adjusted

    @staticmethod
    def _runtime_roster(profile: StrategyProfile, feedback: StrategyFeedback) -> dict[str, Any]:
        open_pressure = _safe_dict(feedback.open_position_pressure)
        max_open = max(
            1,
            _safe_int(
                open_pressure.get("max_open_positions"),
                int(settings.max_open_positions_per_model or DEFAULT_MAX_OPEN_POSITIONS_PER_MODEL),
            ),
        )
        open_groups = max(0, _safe_int(open_pressure.get("open_group_count"), 0))
        low_quality = max(0, _safe_int(open_pressure.get("low_quality_open_count"), 0))
        losing = max(0, _safe_int(open_pressure.get("losing_open_count"), 0))
        release_queue = max(0, _safe_int(open_pressure.get("release_queue_count"), 0))
        problem_keys = {
            str(item.get("key")) for item in feedback.problems if isinstance(item, dict)
        }
        release_pressure = _material_release_pressure(
            open_pressure,
            problem_keys,
            active_profile_id=profile.profile_id,
        ) or bool(
            profile.params.get("full_position_release")
            or profile.params.get("release_losing_positions_first")
        )
        release_target = (
            max(low_quality, min(losing, max(release_queue, 1))) if release_pressure else 0
        )
        rotation_slots = 0
        if release_pressure and open_groups > 0:
            pressure_count = max(release_target, low_quality, release_queue, 1)
            pressure_slots = max(1, math.ceil(pressure_count * 0.20))
            envelope_slots = max(1, math.ceil(max(max_open, open_groups, 1) * 0.15))
            rotation_slots = min(pressure_slots, envelope_slots)
            max_open = max(max_open, open_groups + rotation_slots)
            target_groups = min(max_open, open_groups + rotation_slots)
            review_groups = min(max_open, max(open_groups, release_target + rotation_slots))
            reason = "release_low_quality_positions_with_rotation_slots"
        else:
            healthy_target = max(
                open_groups,
                min(max_open, max(1, math.ceil(max_open * 0.60))),
            )
            target_groups = min(max_open, healthy_target)
            review_groups = min(max_open, max(1, math.ceil(max(target_groups, 1) * 0.60)))
            reason = "expand_by_learned_positive_expectancy_capacity"
        high_risk_groups = min(max_open, max(review_groups, open_groups, target_groups))
        urgent_groups = min(
            max_open,
            max(high_risk_groups, open_groups + 1 if release_pressure else high_risk_groups),
        )
        roster_fill_market_symbol_min = max(
            1,
            min(
                int(settings.auto_scan_symbol_limit or 1),
                max(
                    PORTFOLIO_ROSTER_FILL_MARKET_SYMBOL_MIN,
                    1,
                    math.ceil(max(target_groups - open_groups, 1) * 0.30 * max_open),
                    math.ceil(max(target_groups - open_groups, 1) * 1.5),
                ),
            ),
        )
        return {
            "target_position_groups": target_groups,
            "max_open_positions": max_open,
            "rotation_slots": rotation_slots,
            "release_target_groups": release_target,
            "position_review_max_groups": max(1, review_groups),
            "position_high_risk_max_groups": max(1, high_risk_groups),
            "position_urgent_exit_max_groups": max(1, urgent_groups),
            "roster_fill_market_symbol_min": roster_fill_market_symbol_min,
            "reason": reason,
        }

    @staticmethod
    def _runtime_entry_filters(
        profile: StrategyProfile,
        feedback: StrategyFeedback,
        roster: dict[str, Any],
    ) -> RuntimeEntryFilters:
        params = STRATEGY_LEARNING_PARAMS
        base = default_entry_filters(reason="strategy_learning_runtime_default")
        volume_ratio = base.min_entry_volume_ratio
        adx = base.min_entry_adx
        reasons: list[str] = []

        net_pnl = _safe_float(feedback.totals.get("net_pnl"), 0.0)
        win_rate = _safe_float(feedback.totals.get("win_rate"), 0.0)
        low_trade_count = bool(feedback.totals.get("low_trade_count_penalty"))
        risk_mode = str(_safe_dict(feedback.totals).get("risk_mode") or "")
        open_pressure = _safe_dict(feedback.open_position_pressure)
        problem_keys = {
            str(item.get("key")) for item in feedback.problems if isinstance(item, dict)
        }
        release_pressure = (
            roster.get("reason") == "release_low_quality_positions_with_rotation_slots"
        )

        if net_pnl > 0 and win_rate >= 0.52 and not low_trade_count:
            factor = params.entry_filter_profit_tighten_factor
            volume_ratio *= 1.0 + (1.0 - factor) * 0.50
            adx *= 1.0 + (1.0 - factor) * 0.35
            reasons.append("profitable_recent_feedback_tightens_quality_reference")
        elif low_trade_count or "missed_opportunities" in problem_keys or net_pnl < 0:
            factor = params.entry_filter_loss_relax_factor
            volume_ratio *= factor
            adx *= 0.82
            reasons.append("low_sample_or_loss_feedback_relaxes_scan_reference")

        if release_pressure:
            volume_ratio *= params.entry_filter_release_relax_factor
            adx *= 0.90
            reasons.append("release_pressure_uses_advisory_filters_only")

        if risk_mode in {"drawdown", "hard_recovery"}:
            volume_ratio *= 1.08
            adx *= 1.06
            reasons.append("drawdown_mode_requires_clearer_reference_quality")

        if _safe_float(open_pressure.get("low_quality_open_ratio"), 0.0) > 0.30:
            volume_ratio *= 1.04
            reasons.append("low_quality_open_pressure_biases_quality_reference")

        bounded_volume = min(
            max(volume_ratio, params.entry_volume_ratio_min),
            params.entry_volume_ratio_max,
        )
        bounded_adx = min(max(adx, params.entry_adx_min), params.entry_adx_max)
        return RuntimeEntryFilters(
            min_entry_volume_ratio=round(bounded_volume, 4),
            min_entry_adx=round(bounded_adx, 2),
            source="strategy_learning_runtime",
            is_hard_entry_gate=False,
            reason=", ".join(reasons) or "baseline_dynamic_reference",
        )

    @staticmethod
    def _rollback(
        profile: StrategyProfile,
        feedback: StrategyFeedback,
        score: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "enabled": profile.profile_id != "baseline_current",
            "rollback_to": "baseline_current",
            "rules": [
                "探针画像连续亏损或评分低于基线时回滚",
                "交易数量低于目标时不能晋升为主策略",
                "专家 fallback 依赖继续升高时回滚",
                "满仓释放后净收益未改善时回滚",
            ],
            "current_profile_score": score,
            "recent_net_pnl": feedback.totals.get("net_pnl", 0.0),
        }

    @staticmethod
    def _shadow_validation(
        profiles: list[StrategyProfile],
        feedback: StrategyFeedback,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        missed_loop = _safe_dict(feedback.shadow_feedback.get("missed_opportunity_closed_loop"))
        missed = _safe_int(missed_loop.get("usable_group_count"), 0)
        raw_missed = _safe_int(feedback.shadow_feedback.get("missed_opportunity_count"), 0)
        bad = _safe_int(feedback.shadow_feedback.get("bad_signal_count"), 0)
        good = _safe_int(feedback.shadow_feedback.get("good_signal_count"), 0)
        fallback_blocks = _safe_int(feedback.event_feedback.get("fallback_blocks"), 0)
        integrity_blocks = _safe_int(feedback.decision_quality.get("expert_integrity_blocks"), 0)
        max_position_blocks = _safe_int(feedback.event_feedback.get("max_position_blocks"), 0)
        defensive_probe_blocks = _safe_int(
            feedback.event_feedback.get("profit_first_defensive_probe_shadow_count"),
            0,
        )
        losing_open_count = _safe_int(feedback.open_position_pressure.get("losing_open_count"), 0)
        low_quality_open_count = _safe_int(
            feedback.open_position_pressure.get("low_quality_open_count"),
            0,
        )
        small_wins = _safe_int(feedback.totals.get("small_win_count"), 0) + _safe_int(
            feedback.reflection_feedback.get("small_win_count"), 0
        )
        large_losses = _safe_int(feedback.totals.get("large_loss_count"), 0) + _safe_int(
            feedback.reflection_feedback.get("large_loss_count"), 0
        )
        trade_count = _safe_int(feedback.totals.get("training_trade_count"), 0)
        trade_target = _safe_int(
            feedback.totals.get("trade_count_target"),
            default_min_trade_target(),
        )
        baseline_score = 0.0

        for profile in profiles:
            params = profile.params
            param_consumption = _candidate_param_consumption(params)
            would_increase_entries = bool(
                profile.profile_id == "balanced_probe"
                or _safe_float(params.get("probe_fraction"), 0.0) > 0
                or _safe_float(params.get("global_min_score_delta"), 0.0) < 0
                or _safe_float(params.get("position_size_multiplier"), 1.0) > 1.0
            )
            would_reduce_blocks = bool(
                would_increase_entries
                and (missed > 0 or fallback_blocks > 0 or integrity_blocks > 0)
            )
            would_release_losers = bool(
                profile.profile_id == "loss_release"
                or params.get("full_position_release")
                or params.get("release_losing_positions_first")
                or str(params.get("loss_exit_aggressiveness") or "") == "high"
            )
            would_hold_winners = bool(
                profile.profile_id == "winner_hold"
                or str(params.get("winner_hold_extension") or "") == "high"
                or _safe_float(params.get("profit_lock_min_usdt_multiplier"), 1.0) > 1.05
            )
            fallback_safety = "strict"
            integrity_mode = str(params.get("expert_integrity_mode") or "strict_all_required")
            tolerance = _safe_dict(params.get("fallback_tolerance"))
            if integrity_mode == "balanced_probe_allow_one_non_core_missing":
                fallback_safety = "probe_core_required"
            if _safe_int(tolerance.get("allow_missing_non_core_experts"), 0) > 1:
                fallback_safety = "too_loose"
            probe_cap_active = bool(
                _safe_float(params.get("probe_fraction"), 0.0) > 0
                or _safe_float(params.get("max_probe_size_pct"), 0.0) > 0
            )
            would_restore_quality_entries = bool(
                defensive_probe_blocks > 0
                and (
                    profile.profile_id == "quality_entry_recovery"
                    or (
                        not probe_cap_active
                        and _safe_float(params.get("position_size_multiplier"), 1.0) >= 1.0
                        and integrity_mode == "strict_all_required"
                    )
                )
            )
            side_recovery = profile.profile_id.endswith("_side_recovery") or any(
                _safe_float(_safe_dict(item).get("size_multiplier"), 1.0) < 1.0
                for item in _safe_dict(params.get("side_overrides")).values()
            )

            score = 0.0
            if would_increase_entries:
                score += missed * 0.36 + max(trade_target - trade_count, 0) * 0.18
            if would_reduce_blocks:
                score += (fallback_blocks + integrity_blocks) * 0.14
            if would_release_losers:
                score += (
                    losing_open_count * 0.32
                    + low_quality_open_count * 0.42
                    + max_position_blocks * 0.25
                )
            if would_hold_winners:
                score += small_wins * 0.16 + good * 0.05
            if would_restore_quality_entries:
                score += defensive_probe_blocks * 0.34 + max(trade_target - trade_count, 0) * 0.14
            if side_recovery:
                score += bad * 0.22 + large_losses * 0.08
            if fallback_safety == "too_loose":
                score -= 0.7 + fallback_blocks * 0.08
            if would_increase_entries and bad > good + missed:
                score -= (bad - good - missed) * 0.20
            if profile.profile_id == "baseline_current":
                score = 0.0
                baseline_score = score

            target = _safe_int(params.get("min_trade_count_target"), default_min_trade_target())
            trade_count_guard = {
                "target": target,
                "current": trade_count,
                "penalized_if_low": True,
                "is_entry_gate": False,
                "policy": "dynamic_advisory_learning_confidence",
                "low_trade_count": trade_count < target,
            }
            eligible = profile.profile_id == "baseline_current" or (
                score >= -0.10
                and fallback_safety != "too_loose"
                and param_consumption["has_consumed_runtime_params"]
            )
            rows.append(
                {
                    "profile_id": profile.profile_id,
                    "shadow_score": round(score, 6),
                    "eligible": bool(eligible),
                    "would_increase_entries": would_increase_entries,
                    "would_reduce_blocks": would_reduce_blocks,
                    "would_release_losers": would_release_losers,
                    "would_hold_winners": would_hold_winners,
                    "would_restore_quality_entries": would_restore_quality_entries,
                    "fallback_safety": fallback_safety,
                    "param_consumption": param_consumption,
                    "consumed_runtime_params": param_consumption["consumed_runtime_params"],
                    "unused_runtime_params": param_consumption["unused_runtime_params"],
                    "trade_count_guard": trade_count_guard,
                    "probe_required": bool(
                        profile.profile_id != "baseline_current"
                        and _safe_float(params.get("probe_fraction"), 0.0) > 0
                    ),
                    "missed_opportunities_used": missed,
                    "missed_opportunity_raw_count": raw_missed,
                    "missed_opportunity_closed_loop": {
                        "usable_group_count": missed,
                        "summary": _safe_dict(missed_loop.get("summary")),
                        "blocked_reason_counts": _safe_dict(
                            missed_loop.get("blocked_reason_counts")
                        ),
                    },
                    "bad_signals_used": bad,
                    "good_signals_used": good,
                    "comparison_to_baseline": round(score - baseline_score, 6),
                    "shadow_decision_summary": {
                        "more_entries": "yes" if would_increase_entries else "no",
                        "release_losers": "yes" if would_release_losers else "no",
                        "hold_winners": "yes" if would_hold_winners else "no",
                        "quality_entry_recovery": (
                            "yes" if would_restore_quality_entries else "no"
                        ),
                        "block_reduction": "yes" if would_reduce_blocks else "no",
                    },
                }
            )
        return {
            "rows": rows,
            "completed_count": feedback.shadow_feedback.get("completed_count", 0),
            "baseline_profile_id": "baseline_current",
        }

    @staticmethod
    def _probe(profile: StrategyProfile, feedback: StrategyFeedback) -> dict[str, Any]:
        probe_fraction = _safe_float(profile.params.get("probe_fraction"), 0.0)
        missed_loop = _safe_dict(feedback.shadow_feedback.get("missed_opportunity_closed_loop"))
        closed_loop_probe_rules = [
            _safe_dict(item).get("probe_rules")
            for item in (
                _safe_list(missed_loop.get("probe_candidates"))
                + _safe_list(missed_loop.get("adopted"))
            )
            if _safe_dict(item).get("probe_rules")
        ]
        return {
            "profile_id": profile.profile_id,
            "enabled": profile.profile_id != "baseline_current" and probe_fraction > 0,
            "probe_fraction": probe_fraction,
            "small_position_first": True,
            "promotion_requirements": {
                "min_training_trades": default_min_trade_target(),
                "net_pnl_must_improve": True,
                "max_consecutive_losses": 3,
                "fallback_rate_must_not_increase": True,
            },
            "closed_loop_probe_rules": closed_loop_probe_rules,
            "missed_opportunity_closed_loop": {
                "usable_group_count": _safe_int(missed_loop.get("usable_group_count"), 0),
                "global_missed_count_can_drive_entries": bool(
                    missed_loop.get("global_missed_count_can_drive_entries")
                ),
            },
            "current_training_trades": feedback.totals.get("training_trade_count", 0),
        }


class StrategyLearningEngine:
    """Pure orchestration over compiler, generator, backtester, and scheduler."""

    def __init__(
        self,
        *,
        compiler: StrategyFeedbackCompiler | None = None,
        generator: StrategyCandidateGenerator | None = None,
        backtester: StrategyBacktester | None = None,
        scheduler: StrategyScheduler | None = None,
    ) -> None:
        self.compiler = compiler or StrategyFeedbackCompiler()
        self.generator = generator or StrategyCandidateGenerator()
        self.backtester = backtester or StrategyBacktester()
        self.scheduler = scheduler or StrategyScheduler()

    def build(
        self,
        *,
        mode: str,
        window_hours: int,
        positions: list[Any],
        open_positions: list[Any],
        orders: list[Any],
        decisions: list[Any],
        shadows: list[Any],
        memories: list[Any],
        strategy_events: list[Any] | None = None,
        reflections: list[Any] | None = None,
        max_open_positions: int = 20,
        extra_profiles: list[StrategyProfile] | None = None,
    ) -> dict[str, Any]:
        feedback = self.compiler.compile(
            mode=mode,
            window_hours=window_hours,
            positions=positions,
            open_positions=open_positions,
            orders=orders,
            decisions=decisions,
            shadows=shadows,
            memories=memories,
            strategy_events=strategy_events or [],
            reflections=reflections,
            max_open_positions=max_open_positions,
        )
        return self.build_from_feedback(feedback, extra_profiles=extra_profiles)

    def build_from_feedback(
        self,
        feedback: StrategyFeedback,
        *,
        extra_profiles: list[StrategyProfile] | None = None,
    ) -> dict[str, Any]:
        profiles = self.generator.generate(feedback)
        if extra_profiles:
            profiles = self.generator._dedupe([*profiles, *extra_profiles])
        backtest_rows = [self.backtester.score(profile, feedback) for profile in profiles]
        schedule = self.scheduler.schedule(profiles, feedback, backtest_rows)
        return {
            "feedback": feedback.to_dict(),
            "schedule": schedule.to_dict(),
            "active_profile": schedule.active_profile.to_dict(),
        }

    def apply_to_context(
        self,
        strategy_context: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        result = dict(strategy_context or {})
        schedule = _safe_dict(payload.get("schedule"))
        runtime = _safe_dict(schedule.get("runtime"))
        active_profile = _safe_dict(schedule.get("active_profile"))
        delta = _safe_float(runtime.get("global_min_score_delta"), 0.0)
        current_min_score = _safe_float(
            result.get("min_opportunity_score"), MIN_ENTRY_OPPORTUNITY_SCORE
        )
        if delta:
            result["min_opportunity_score"] = round(
                min(max(current_min_score + delta, 0.35), 2.80),
                4,
            )
        result.setdefault("side_quality", {})
        side_quality = result["side_quality"] if isinstance(result["side_quality"], dict) else {}
        for side, override in _safe_dict(runtime.get("side_overrides")).items():
            existing = _safe_dict(side_quality.get(side))
            side_quality[side] = {**existing, **_safe_dict(override)}
        result["side_quality"] = side_quality
        result["strategy_profile_id"] = active_profile.get("id") or runtime.get("profile_id")
        result["strategy_profile_version"] = active_profile.get("version") or runtime.get(
            "profile_version"
        )
        result["scheduler_reason"] = schedule.get("reason", "")
        result["expert_integrity_mode"] = runtime.get("expert_integrity_mode")
        result["position_size_multiplier"] = _safe_float(
            runtime.get("position_size_multiplier"), 1.0
        )
        result["probe_fraction"] = _safe_float(runtime.get("probe_fraction"), 0.0)
        result["max_probe_size_pct"] = _safe_float(runtime.get("max_probe_size_pct"), 0.0)
        profit_first_runtime_feedback = _safe_dict(
            _safe_dict(payload.get("feedback")).get("profit_first_runtime_feedback")
        )
        result["profit_first_runtime_feedback"] = _compact_profit_first_runtime_feedback(
            profit_first_runtime_feedback
        )
        result["side_weights"] = _merge_profit_first_side_weights(
            _safe_dict(runtime.get("side_weights")),
            profit_first_runtime_feedback,
        )
        result["profit_first_runtime_feedback_applied"] = bool(
            (
                _profit_first_feedback_can_influence_context(profit_first_runtime_feedback)
                and _safe_dict(profit_first_runtime_feedback.get("side_weights"))
            )
            or bool(runtime.get("profit_first_runtime_feedback_applied"))
            or bool(_safe_dict(runtime.get("profit_first_context")).get("applied_reasons"))
        )
        entry_filters = _safe_dict(runtime.get("entry_filters"))
        default_filters = default_entry_filters(reason="strategy_learning_context_default")
        result["entry_filters"] = entry_filters
        result["min_entry_volume_ratio"] = _safe_float(
            entry_filters.get("min_entry_volume_ratio"),
            _safe_float(
                runtime.get("min_entry_volume_ratio"),
                default_filters.min_entry_volume_ratio,
            ),
        )
        result["min_entry_adx"] = _safe_float(
            entry_filters.get("min_entry_adx"),
            _safe_float(runtime.get("min_entry_adx"), default_filters.min_entry_adx),
        )
        result["entry_filters_are_hard_gate"] = False
        result["strategy_learning_sizing"] = {
            "profile_id": active_profile.get("id") or runtime.get("profile_id"),
            "position_size_multiplier": result["position_size_multiplier"],
            "probe_fraction": result["probe_fraction"],
            "max_probe_size_pct": result["max_probe_size_pct"],
            "side_overrides": _safe_dict(runtime.get("side_overrides")),
            "side_weights": result["side_weights"],
            "profit_first_context": _safe_dict(runtime.get("profit_first_context")),
            "profit_first_runtime_feedback_applied": result[
                "profit_first_runtime_feedback_applied"
            ],
            "profit_first_runtime_feedback": result["profit_first_runtime_feedback"],
            "reason": schedule.get("reason", ""),
        }
        result["target_position_groups"] = _safe_int(runtime.get("target_position_groups"), 0)
        result["target_open_position_groups"] = _safe_int(
            runtime.get("target_open_position_groups"), result["target_position_groups"]
        )
        result["position_review_max_groups"] = _safe_int(
            runtime.get("position_review_max_groups"), 0
        )
        result["portfolio_roster"] = {
            **_safe_dict(result.get("portfolio_roster")),
            "target_position_groups": result["target_position_groups"],
            "rotation_slots": _safe_int(runtime.get("rotation_slots"), 0),
            "release_target_groups": _safe_int(runtime.get("release_target_groups"), 0),
            "max_open_positions": _safe_int(runtime.get("max_open_positions"), 0),
            "policy_source": "strategy_learning_runtime",
            "policy_reason": runtime.get("capacity_policy_reason"),
        }
        result["loss_exit_aggressiveness"] = str(
            runtime.get("loss_exit_aggressiveness") or "normal"
        )
        result["full_position_release"] = bool(runtime.get("full_position_release"))
        result["release_losing_positions_first"] = bool(
            runtime.get("release_losing_positions_first")
        )
        result["position_review_priority_boost"] = _safe_float(
            runtime.get("position_review_priority_boost"), 1.0
        )
        result["winner_hold_extension"] = str(runtime.get("winner_hold_extension") or "normal")
        result["profit_lock_min_usdt_multiplier"] = _safe_float(
            runtime.get("profit_lock_min_usdt_multiplier"),
            1.0,
        )
        result["payoff_repair_intensity"] = _safe_float(
            runtime.get("payoff_repair_intensity"),
            0.0,
        )
        result["winner_hold_dynamic"] = _safe_dict(runtime.get("winner_hold_dynamic"))
        result["pullback_lock_enabled"] = bool(runtime.get("pullback_lock_enabled"))
        guard = _safe_dict(payload.get("runtime_guard"))
        feedback_payload = _safe_dict(payload.get("feedback"))
        feedback_summary = self._compact_feedback(_safe_dict(payload.get("feedback")))
        open_pressure = _safe_dict(feedback_payload.get("open_position_pressure"))
        release_queue = _safe_list(open_pressure.get("release_queue")) or _safe_list(
            open_pressure.get("release_candidates")
        )
        active_profile_id = str(active_profile.get("id") or runtime.get("profile_id") or "")
        rebalance_queue = [
            _safe_dict(item)
            for item in release_queue
            if _safe_dict(item).get("should_release")
            or (
                active_profile_id
                and _safe_dict(item).get("strategy_profile_id")
                and _safe_dict(item).get("strategy_profile_id") != active_profile_id
            )
        ][:8]
        material_release_pressure = _material_release_pressure(
            open_pressure,
            {str(item) for item in _safe_list(feedback_summary.get("problem_keys"))},
            active_profile_id=active_profile_id,
        )
        has_current_release_pressure = bool(
            open_pressure.get("full_position_pressure")
            or open_pressure.get("fragmentation_pressure")
            or _material_low_quality_pressure(open_pressure)
            or _safe_int(open_pressure.get("low_quality_open_count"), 0) > 0
        )
        release_pressure_active = bool(
            material_release_pressure
            and has_current_release_pressure
            and (
                _safe_float(feedback_summary.get("net_pnl"), 0.0) < 0.0
                or _safe_int(open_pressure.get("low_quality_open_count"), 0) > 0
            )
        )
        guard_reasons = {str(token) for token in _safe_list(guard.get("reasons"))}
        model_health_recovered = bool(guard.get("model_health_recovered"))
        fallback_health_guard_active = bool(
            "fallback_dependency_guard" in guard_reasons and not model_health_recovered
        )
        execution_guard_active = bool(
            guard.get("should_rollback")
            and "execution_error_guard" in guard_reasons
            and not model_health_recovered
        )
        recovery_probe_allowed = bool(fallback_health_guard_active or execution_guard_active)
        entry_pause = False
        if release_pressure_active:
            sizing = result["strategy_learning_sizing"]
            sizing["release_pressure_active"] = True
            sizing["reason"] = (
                "满仓/低质量仓位压力存在：优先释放低质量仓位，同时只允许高质量小仓探针。"
            )
            sizing["probe_fraction"] = max(
                _safe_float(sizing.get("probe_fraction"), 0.0),
                ENTRY_RISK_SIZING_PARAMS.release_probe_fraction_floor,
            )
            sizing["max_probe_size_pct"] = min(
                max(
                    _safe_float(
                        sizing.get("max_probe_size_pct"),
                        ENTRY_RISK_SIZING_PARAMS.release_probe_default_cap_pct,
                    ),
                    ENTRY_RISK_SIZING_PARAMS.release_probe_min_cap_pct,
                ),
                ENTRY_RISK_SIZING_PARAMS.release_probe_max_cap_pct,
            )
            sizing["reason"] = (
                "满仓或低质量仓位压力存在：优先释放低质量仓位，" "同时只允许高质量小仓探针。"
            )
        if recovery_probe_allowed:
            sizing = result["strategy_learning_sizing"]
            sizing["health_guard_active"] = True
            sizing["recovery_probe_allowed"] = True
            sizing["reason"] = (
                "模型健康护栏激活：fallback 依赖偏高，系统不再硬停所有新开仓，"
                "改为质量驱动恢复探针来采集真实健康样本。"
            )
            sizing["position_size_multiplier"] = min(
                _safe_float(sizing.get("position_size_multiplier"), 1.0),
                ENTRY_RISK_SIZING_PARAMS.recovery_multiplier_cap,
            )
            sizing["probe_fraction"] = max(
                _safe_float(sizing.get("probe_fraction"), 0.0),
                ENTRY_RISK_SIZING_PARAMS.recovery_probe_fraction_floor,
            )
            sizing["max_probe_size_pct"] = min(
                max(
                    _safe_float(
                        sizing.get("max_probe_size_pct"),
                        ENTRY_RISK_SIZING_PARAMS.recovery_probe_default_cap_pct,
                    ),
                    ENTRY_RISK_SIZING_PARAMS.recovery_probe_min_cap_pct,
                ),
                ENTRY_RISK_SIZING_PARAMS.recovery_health_probe_max_cap_pct,
            )
            sizing["execution_guard_active"] = execution_guard_active
            sizing["reason"] = (
                "策略健康护栏已触发：系统不再硬停全部新开仓，"
                "改为质量驱动恢复探针；强信号可按收益质量动态放大，弱信号仍保持小仓。"
            )
        result["strategy_learning_entry_pause"] = entry_pause
        result["strategy_learning_entry_pause_reason"] = ""
        result["strategy_learning_execution_guard_active"] = execution_guard_active
        result["strategy_learning_execution_guard_reason"] = (
            "策略护栏检测到执行链路异常：已回滚异常画像并转入质量驱动恢复探针；"
            "不再阻断全部新开仓，真实下单仍会经过执行前风控和 OKX 规则校验。"
            if execution_guard_active
            else ""
        )
        result["strategy_learning_health_guard_active"] = fallback_health_guard_active
        result["strategy_learning_recovery_probe_allowed"] = recovery_probe_allowed
        result["strategy_learning_recovery_probe_reason"] = (
            "策略健康护栏已转为质量驱动恢复探针；强信号可动态放大，执行前风控和 OKX 规则仍会逐单校验。"
            if recovery_probe_allowed
            else ""
        )
        result["strategy_learning_release_pressure_active"] = release_pressure_active
        result["strategy_learning_release_pressure_reason"] = (
            "满仓/碎片化或低质量仓位压力存在：系统应先释放低质量仓位，并把新策略限制为小仓验证。"
            if release_pressure_active
            else ""
        )
        result["strategy_learning_release_pressure_detail"] = {
            "active": release_pressure_active,
            "material_pressure": material_release_pressure,
            "current_pressure": has_current_release_pressure,
            "policy": "current_position_pressure_only",
            "reason": result["strategy_learning_release_pressure_reason"],
            "open_position_pressure": {
                "open_count": _safe_int(open_pressure.get("open_count"), 0),
                "open_group_count": _safe_int(open_pressure.get("open_group_count"), 0),
                "max_open_positions": _safe_int(open_pressure.get("max_open_positions"), 0),
                "full_position_pressure": bool(open_pressure.get("full_position_pressure")),
                "fragmentation_pressure": bool(open_pressure.get("fragmentation_pressure")),
                "low_quality_open_count": _safe_int(open_pressure.get("low_quality_open_count"), 0),
                "low_quality_open_ratio": _safe_float(
                    open_pressure.get("low_quality_open_ratio"), 0.0
                ),
                "release_queue_count": _safe_int(open_pressure.get("release_queue_count"), 0),
            },
        }
        result["position_rebalance_queue"] = rebalance_queue
        result["low_quality_open_count"] = _safe_int(open_pressure.get("low_quality_open_count"), 0)
        result["low_quality_open_ratio"] = _safe_float(
            open_pressure.get("low_quality_open_ratio"), 0.0
        )
        result["strategy_learning"] = {
            "active_profile": active_profile,
            "runtime": runtime,
            "entry_filters": entry_filters,
            "reason": schedule.get("reason", ""),
            "rollback": schedule.get("rollback", {}),
            "feedback_summary": feedback_summary,
            "open_position_pressure": open_pressure,
            "runtime_guard": guard,
            "entry_pause": entry_pause,
            "entry_pause_reason": result["strategy_learning_entry_pause_reason"],
            "execution_guard_active": execution_guard_active,
            "execution_guard_reason": result["strategy_learning_execution_guard_reason"],
            "health_guard_active": fallback_health_guard_active,
            "recovery_probe_allowed": recovery_probe_allowed,
            "recovery_probe_reason": result["strategy_learning_recovery_probe_reason"],
            "release_pressure_active": release_pressure_active,
            "release_pressure_reason": result["strategy_learning_release_pressure_reason"],
            "release_pressure_detail": result["strategy_learning_release_pressure_detail"],
            "loss_exit_aggressiveness": result["loss_exit_aggressiveness"],
            "full_position_release": result["full_position_release"],
            "release_losing_positions_first": result["release_losing_positions_first"],
            "position_review_priority_boost": result["position_review_priority_boost"],
            "winner_hold_extension": result["winner_hold_extension"],
            "profit_lock_min_usdt_multiplier": result["profit_lock_min_usdt_multiplier"],
            "payoff_repair_intensity": result["payoff_repair_intensity"],
            "winner_hold_dynamic": result["winner_hold_dynamic"],
            "pullback_lock_enabled": result["pullback_lock_enabled"],
            "side_weights": result["side_weights"],
            "profit_first_context": _safe_dict(runtime.get("profit_first_context")),
            "profit_first_runtime_feedback": result["profit_first_runtime_feedback"],
            "profit_first_runtime_feedback_applied": result[
                "profit_first_runtime_feedback_applied"
            ],
            "release_queue": release_queue[:8],
            "rebalance_queue": rebalance_queue,
            "low_quality_open_count": result["low_quality_open_count"],
            "low_quality_open_ratio": result["low_quality_open_ratio"],
            "candidate_count": len(_safe_list(schedule.get("candidates"))),
            "scheduler_mode": schedule.get("scheduler_mode", "auto"),
            "manual_profile_id": schedule.get("manual_profile_id", ""),
            "disabled_profiles": _safe_list(schedule.get("disabled_profiles")),
            "shadow_validation": schedule.get("shadow_validation", {}),
            "probe": schedule.get("probe", {}),
            "backtest": schedule.get("backtest", {}),
            "dispatch_reason": schedule.get("reason", ""),
            "low_trade_count_penalized": True,
            "manual_close_excluded_from_training": True,
            "training_policy": _safe_dict(
                _safe_dict(payload.get("feedback")).get("training_policy")
            ),
        }
        result["execution_policy"] = (
            str(result.get("execution_policy") or "")
            + " Strategy learning scheduler is active: profiles adjust only bounded parameters; "
            "low-trade-count profiles are penalized and arbitrary code generation is forbidden."
        ).strip()
        return result

    @staticmethod
    def _compact_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
        totals = _safe_dict(feedback.get("totals"))
        decision_quality = _safe_dict(feedback.get("decision_quality"))
        open_pressure = _safe_dict(feedback.get("open_position_pressure"))
        reflection = _safe_dict(feedback.get("reflection_feedback"))
        return {
            "training_trade_count": totals.get("training_trade_count", 0),
            "net_pnl": totals.get("net_pnl", 0.0),
            "win_rate": totals.get("win_rate", 0.0),
            "expert_integrity_blocks": decision_quality.get("expert_integrity_blocks", 0),
            "fallback_entry_rate": decision_quality.get("fallback_entry_rate", 0.0),
            "full_position_pressure": open_pressure.get("full_position_pressure", False),
            "reflection_fee_adjusted_pnl": reflection.get("fee_adjusted_pnl", 0.0),
            "reflection_mistake_count": reflection.get("mistake_count", 0),
            "payoff_profile": _compact_payoff_profile_value(totals.get("payoff_profile")),
            "reflection_payoff_profile": _compact_payoff_profile_value(
                reflection.get("payoff_profile")
            ),
            "problem_keys": [item.get("key") for item in _safe_list(feedback.get("problems"))],
        }


class StrategyLearningService:
    """Database-backed strategy learning service used by API and trading loop."""

    def __init__(
        self,
        *,
        engine: StrategyLearningEngine | None = None,
        state_store: StrategyLearningStateStore | None = None,
    ) -> None:
        self.state_store = state_store or StrategyLearningStateStore()
        self.engine = engine or StrategyLearningEngine(
            scheduler=StrategyScheduler(self.state_store),
        )

    async def dashboard_payload(
        self,
        *,
        mode: str,
        hours: int = DEFAULT_LOOKBACK_HOURS,
        limit: int = STRATEGY_LEARNING_PARAMS.dashboard_default_limit,
        max_open_positions: int | None = None,
        detail: str = "summary",
    ) -> dict[str, Any]:
        selected_detail = "full" if str(detail or "").lower() == "full" else "summary"
        params = STRATEGY_LEARNING_PARAMS
        raw_limit = int(limit or params.dashboard_default_limit)
        max_limit = (
            params.dashboard_full_limit
            if selected_detail == "full"
            else params.dashboard_summary_limit
        )
        effective_limit = max(params.min_dashboard_limit, min(raw_limit, max_limit))
        rows = await self._load_rows(mode=mode, hours=hours, limit=effective_limit)
        feedback = self._compile_feedback(
            mode=mode,
            hours=hours,
            rows=rows,
            open_positions=rows["open_positions"],
            max_open_positions=max_open_positions,
        )
        payload = self._build_payload_from_feedback(
            feedback,
            extra_profiles=self._cached_llm_profiles(feedback),
        )
        payload.update(
            {
                "mode": mode,
                "window_hours": hours,
                "sample_limit": effective_limit,
                "detail": selected_detail,
                "state": self.state_store.load(),
            }
        )
        payload["runtime_guard"] = self._runtime_guard(payload, mutate=False)
        if selected_detail != "full":
            payload = self._compact_dashboard_payload(payload)
        return payload

    def _compact_dashboard_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Keep strategy dashboard fast while preserving scheduling evidence."""

        compact = dict(payload)
        feedback = _safe_dict(compact.get("feedback"))
        if feedback:
            compact["feedback"] = {
                "mode": feedback.get("mode"),
                "window_hours": feedback.get("window_hours"),
                "generated_at": feedback.get("generated_at"),
                "totals": feedback.get("totals", {}),
                "side_performance": feedback.get("side_performance", {}),
                "open_position_pressure": self._compact_open_position_pressure(
                    feedback.get("open_position_pressure")
                ),
                "decision_quality": self._compact_decision_quality(
                    feedback.get("decision_quality")
                ),
                "shadow_feedback": feedback.get("shadow_feedback", {}),
                "expert_memory": feedback.get("expert_memory", {}),
                "manual_intervention": feedback.get("manual_intervention", {}),
                "reflection_feedback": self._compact_reflection_feedback(
                    feedback.get("reflection_feedback")
                ),
                "event_feedback": self._compact_event_feedback(
                    feedback.get("event_feedback"),
                    include_recent_details=True,
                ),
                "profit_first_runtime_feedback": feedback.get(
                    "profit_first_runtime_feedback",
                    {},
                ),
                "problems": _safe_list(feedback.get("problems"))[:10],
                "root_causes": _safe_list(feedback.get("root_causes"))[:10],
                "training_policy": feedback.get("training_policy", {}),
            }
        schedule = _safe_dict(compact.get("schedule"))
        if schedule:
            schedule = dict(schedule)
            schedule["candidates"] = _safe_list(schedule.get("candidates"))[:12]
            backtest = _safe_dict(schedule.get("backtest"))
            schedule["backtest"] = {"rows": _safe_list(backtest.get("rows"))[:12]}
            shadow = _safe_dict(schedule.get("shadow_validation"))
            schedule["shadow_validation"] = {
                **shadow,
                "rows": _safe_list(shadow.get("rows"))[:12],
            }
            compact["schedule"] = schedule
        return compact

    async def apply_to_strategy_context(
        self,
        *,
        mode: str,
        strategy_context: dict[str, Any],
        open_positions: list[dict[str, Any]] | None,
        hours: int = DEFAULT_LOOKBACK_HOURS,
        limit: int = STRATEGY_LEARNING_PARAMS.dashboard_default_limit,
        max_open_positions: int | None = None,
    ) -> dict[str, Any]:
        rows = await self._load_rows(mode=mode, hours=hours, limit=limit)
        runtime_open_positions: list[Any] = list(open_positions or []) or rows["open_positions"]
        feedback = self._compile_feedback(
            mode=mode,
            hours=hours,
            rows=rows,
            open_positions=runtime_open_positions,
            max_open_positions=max_open_positions,
        )
        payload = await self._build_runtime_payload(
            mode=mode,
            feedback=feedback,
        )
        guard = self._runtime_guard(payload, mutate=True)
        max_rollback_rounds = LLM_CANDIDATE_MAX_COUNT + 4
        rollback_rounds = 0
        while guard.get("should_rollback") and rollback_rounds < max_rollback_rounds:
            rollback_rounds += 1
            payload = self._build_payload_from_feedback(
                feedback,
                extra_profiles=self._cached_llm_profiles(feedback),
            )
            guard = self._runtime_guard(payload, mutate=True)
        if rollback_rounds:
            guard["rollback_rounds"] = rollback_rounds
        payload["runtime_guard"] = guard
        await self._persist_profile_snapshots(mode, payload)
        return self.engine.apply_to_context(strategy_context, payload)

    def set_profile_disabled(
        self,
        profile_id: str,
        *,
        disabled: bool,
        reason: str = "",
    ) -> dict[str, Any]:
        return self.state_store.set_profile_disabled(profile_id, disabled=disabled, reason=reason)

    def rollback_to_baseline(self) -> dict[str, Any]:
        return self.state_store.set_manual_active_profile(None)

    def set_manual_active_profile(self, profile_id: str | None) -> dict[str, Any]:
        return self.state_store.set_manual_active_profile(profile_id)

    def _compile_feedback(
        self,
        *,
        mode: str,
        hours: int,
        rows: dict[str, list[Any]],
        open_positions: list[Any],
        max_open_positions: int | None,
    ) -> StrategyFeedback:
        return self.engine.compiler.compile(
            mode=mode,
            window_hours=hours,
            positions=rows["closed_positions"],
            open_positions=open_positions,
            orders=rows["orders"],
            decisions=rows["decisions"],
            shadows=rows["shadows"],
            memories=rows["memories"],
            strategy_events=rows["strategy_events"],
            reflections=rows["reflections"],
            max_open_positions=max_open_positions
            or int(settings.max_open_positions_per_model or DEFAULT_MAX_OPEN_POSITIONS_PER_MODEL),
        )

    def _build_payload_from_feedback(
        self,
        feedback: StrategyFeedback,
        *,
        extra_profiles: list[StrategyProfile] | None = None,
    ) -> dict[str, Any]:
        payload = self.engine.build_from_feedback(feedback, extra_profiles=extra_profiles)
        payload["llm_candidate_status"] = self._llm_candidate_status(feedback)
        return payload

    async def _build_runtime_payload(
        self,
        *,
        mode: str,
        feedback: StrategyFeedback,
    ) -> dict[str, Any]:
        extra_profiles = self._cached_llm_profiles(feedback)
        if self._should_refresh_llm_candidates(feedback):
            refreshed = await self._generate_llm_profiles(mode=mode, feedback=feedback)
            if refreshed:
                extra_profiles = refreshed
        return self._build_payload_from_feedback(feedback, extra_profiles=extra_profiles)

    def _feedback_signature(self, feedback: StrategyFeedback) -> str:
        data = {
            "totals": feedback.totals,
            "side_performance": feedback.side_performance,
            "open_position_pressure": feedback.open_position_pressure,
            "decision_quality": feedback.decision_quality,
            "shadow_feedback": feedback.shadow_feedback,
            "profit_first_runtime_feedback": feedback.profit_first_runtime_feedback,
            "event_feedback": {
                key: feedback.event_feedback.get(key)
                for key in (
                    "total_events",
                    "max_position_blocks",
                    "fallback_blocks",
                    "execution_errors",
                    "attribution_coverage",
                )
            },
            "reflection_feedback": {
                key: feedback.reflection_feedback.get(key)
                for key in (
                    "training_count",
                    "fee_adjusted_pnl",
                    "avg_loss_hold_minutes",
                    "small_win_count",
                    "large_loss_count",
                    "payoff_profile",
                    "mistake_count",
                )
            },
            "problems": [item.get("key") for item in feedback.problems],
        }
        raw = json.dumps(_json_safe(data), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _llm_candidate_status(self, feedback: StrategyFeedback) -> dict[str, Any]:
        state = self.state_store.load()
        entry = _safe_dict(state.get("llm_candidate_cache"))
        candidates = _safe_list(entry.get("candidates"))
        signature = self._feedback_signature(feedback)
        cache_matches = entry.get("signature") == signature
        cached_candidates = self._public_cached_llm_candidates(candidates)
        return {
            "enabled": bool(getattr(settings, "strategy_learning_llm_candidates_enabled", True)),
            "signature": signature,
            "cached_signature": entry.get("signature"),
            "cached_at": entry.get("generated_at"),
            "candidate_count": len(candidates),
            "visible_candidate_count": len(cached_candidates),
            "cache_matches_feedback": cache_matches,
            "cache_status": "current" if cache_matches else ("stale" if candidates else "empty"),
            "last_error": entry.get("last_error", ""),
            "last_error_kind": entry.get("last_error_kind", ""),
            "last_model": entry.get("model", ""),
            "attempts": _safe_list(entry.get("attempts")),
            "source": entry.get("source", "llm_structured_candidate" if candidates else "none"),
            "cached_candidates": cached_candidates,
        }

    @staticmethod
    def _public_cached_llm_candidates(candidates: list[Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, item in enumerate(candidates[:LLM_CANDIDATE_MAX_COUNT], start=1):
            data = _safe_dict(item)
            if not data:
                continue
            rows.append(
                {
                    "id": str(data.get("id") or data.get("profile_id") or f"llm_candidate_{index}"),
                    "label": str(sanitize_text(data.get("label")) or f"LLM候选{index}")[:80],
                    "description": str(sanitize_text(data.get("description")) or "")[:240],
                    "source": str(data.get("source") or "llm_structured_candidate"),
                    "params": sanitize_payload(_safe_dict(data.get("params"))),
                }
            )
        return rows

    def _cached_llm_profiles(self, feedback: StrategyFeedback) -> list[StrategyProfile]:
        entry = _safe_dict(self.state_store.load().get("llm_candidate_cache"))
        if entry.get("signature") != self._feedback_signature(feedback):
            return []
        return self.engine.generator.from_structured_candidates(
            _safe_list(entry.get("candidates")),
            feedback,
        )

    def _should_refresh_llm_candidates(self, feedback: StrategyFeedback) -> bool:
        if not bool(getattr(settings, "strategy_learning_llm_candidates_enabled", True)):
            return False
        if not self._llm_candidate_configs():
            return False
        problem_keys = {str(item.get("key") or "") for item in feedback.problems}
        if not problem_keys:
            return False
        signature = self._feedback_signature(feedback)
        entry = _safe_dict(self.state_store.load().get("llm_candidate_cache"))
        generated_at = _parse_iso_datetime(entry.get("generated_at"))
        interval = max(
            300,
            _safe_int(
                getattr(settings, "strategy_learning_llm_candidate_interval_seconds", 21600),
                LLM_CANDIDATE_CACHE_SECONDS,
            ),
        )
        age_seconds = (datetime.now(UTC) - generated_at).total_seconds() if generated_at else None
        if entry.get("signature") != signature:
            return True
        if _safe_int(entry.get("prompt_version"), 0) != LLM_CANDIDATE_PROMPT_VERSION:
            return True
        if _safe_list(entry.get("candidates")):
            return not bool(age_seconds is not None and age_seconds < interval)
        retry_after = min(interval, LLM_CANDIDATE_FAILURE_RETRY_SECONDS)
        return not bool(age_seconds is not None and age_seconds < retry_after)

    def _llm_candidate_configs(self) -> list[dict[str, str]]:
        configs = [cfg for cfg in settings.get_fixed_ai_models(False) if isinstance(cfg, dict)]
        ordered_names = [
            "decision_maker",
            "trend_expert",
            "momentum_expert",
            "risk_expert",
            "sentiment_expert",
            "position_expert",
        ]
        by_name = {str(item.get("name") or ""): item for item in configs}
        ordered = [by_name[name] for name in ordered_names if name in by_name]
        ordered.extend(item for item in configs if item not in ordered)
        result: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for cfg in ordered:
            if cfg.get("enabled") is False:
                continue
            api_base = str(cfg.get("api_base") or settings.ai_api_base or "").rstrip("/")
            api_key = str(cfg.get("api_key") or settings.ai_api_key or "")
            model = str(cfg.get("model") or settings.ai_model or "")
            if not api_base or not api_key or not model:
                continue
            key = (api_base, model)
            if key in seen:
                continue
            seen.add(key)
            result.append(
                {
                    "api_base": api_base,
                    "api_key": api_key,
                    "model": model,
                    "name": str(cfg.get("name") or model),
                }
            )
        return result

    async def _generate_llm_profiles(
        self,
        *,
        mode: str,
        feedback: StrategyFeedback,
    ) -> list[StrategyProfile]:
        signature = self._feedback_signature(feedback)
        attempts: list[dict[str, Any]] = []
        last_error = ""
        last_error_kind = ""
        for cfg in self._llm_candidate_configs():
            model = cfg["model"]
            try:
                candidates = await self._call_llm_candidate_model(
                    api_base=cfg["api_base"],
                    api_key=cfg["api_key"],
                    model=model,
                    feedback=feedback,
                )
            except Exception as exc:
                last_error = safe_error_text(exc, limit=260)
                last_error_kind = self._llm_candidate_error_kind(exc)
                attempts.append(
                    {
                        "name": cfg.get("name", ""),
                        "model": model,
                        "api_base": cfg.get("api_base", ""),
                        "status": "failed",
                        "error_kind": last_error_kind,
                        "error": last_error,
                    }
                )
                logger.debug(
                    "strategy learning llm candidate generation attempt failed",
                    model=model,
                    error=last_error,
                )
                continue
            profiles = self.engine.generator.from_structured_candidates(candidates, feedback)
            attempts.append(
                {
                    "name": cfg.get("name", ""),
                    "model": model,
                    "api_base": cfg.get("api_base", ""),
                    "status": "completed",
                    "candidate_count": len(profiles),
                }
            )
            self._store_llm_candidate_cache(
                mode=mode,
                signature=signature,
                candidates=[profile.to_dict() for profile in profiles],
                error="" if profiles else "LLM returned no valid bounded strategy candidates",
                error_kind="" if profiles else LLM_CANDIDATE_ERROR_EMPTY,
                attempts=attempts,
                model=model,
            )
            return profiles
        self._store_llm_candidate_cache(
            mode=mode,
            signature=signature,
            candidates=[],
            error=last_error or "all LLM candidate models failed",
            error_kind=last_error_kind or LLM_CANDIDATE_ERROR_UNKNOWN,
            attempts=attempts,
            model="",
        )
        return []

    @staticmethod
    def _top_text_items(
        items: Any,
        key: str,
        *,
        limit: int = 4,
        text_limit: int = 120,
        include_count: bool = True,
    ) -> list[Any]:
        rows: list[Any] = []
        for item in _safe_list(items)[:limit]:
            if isinstance(item, dict):
                value = item.get(key) or item.get("summary") or item.get("reason")
                count = _safe_int(item.get("count"), 0)
            else:
                value = item
                count = 0
            text = str(value or "").replace("\n", " ").strip()
            if text:
                if include_count:
                    rows.append({key: text[:text_limit], "count": count})
                else:
                    rows.append(text[:text_limit])
        return rows

    @staticmethod
    def _compact_count_map(value: Any, *, limit: int = 8) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        items = sorted(value.items(), key=lambda item: _safe_float(item[1], 0.0), reverse=True)
        result: dict[str, Any] = {}
        for key, count in items[:limit]:
            text_key = str(key or "").strip()[:80]
            if text_key:
                result[text_key] = count
        return result

    @staticmethod
    def _scalar_subset(value: Any, keys: tuple[str, ...]) -> dict[str, Any]:
        source = _safe_dict(value)
        result: dict[str, Any] = {}
        for key in keys:
            item = source.get(key)
            if isinstance(item, str):
                result[key] = item[:140]
            elif item is None or isinstance(item, (bool, int, float)):
                result[key] = item
        return result

    def _compact_payoff_profile(self, value: Any) -> dict[str, Any]:
        return _compact_payoff_profile_value(value)

    def _compact_open_position_pressure(self, value: Any) -> dict[str, Any]:
        result = self._scalar_subset(
            value,
            (
                "open_count",
                "open_part_count",
                "open_group_count",
                "duplicate_part_count",
                "max_open_positions",
                "usage_ratio",
                "part_usage_ratio",
                "full_position_pressure",
                "fragmentation_pressure",
                "low_quality_open_count",
                "low_quality_open_ratio",
                "stale_open_count",
                "release_queue_count",
                "losing_open_count",
                "losing_open_part_count",
                "winner_open_count",
                "open_unrealized_pnl",
                "losing_unrealized_pnl",
            ),
        )
        source = _safe_dict(value)
        result["side_counts"] = self._compact_count_map(source.get("side_counts"), limit=4)
        result["side_unrealized_pnl"] = self._compact_count_map(
            source.get("side_unrealized_pnl"), limit=4
        )
        result["release_candidates"] = [
            {
                "symbol": row.get("symbol"),
                "side": row.get("side"),
                "unrealized_pnl": row.get("unrealized_pnl"),
                "quality_score": row.get("quality_score"),
                "quality_bucket": row.get("quality_bucket"),
                "release_priority": row.get("release_priority"),
                "strategy_profile_id": row.get("strategy_profile_id"),
            }
            for row in (
                _safe_dict(item) for item in _safe_list(source.get("release_candidates"))[:4]
            )
        ]
        return result

    def _compact_decision_quality(self, value: Any) -> dict[str, Any]:
        result = self._scalar_subset(
            value,
            (
                "market_scans",
                "entry_signals",
                "executed_entries",
                "signal_rate",
                "execution_rate",
                "expert_integrity_blocks",
                "fallback_entry_decisions",
                "fallback_entry_rate",
                "zero_second_entry_decisions",
                "recent_window_hours",
                "recent_market_scans",
                "recent_entry_signals",
                "recent_executed_entries",
                "recent_expert_integrity_blocks",
                "recent_fallback_entry_decisions",
                "recent_fallback_entry_rate",
                "recent_zero_second_entry_decisions",
                "model_health_recovered",
            ),
        )
        source = _safe_dict(value)
        result["model_timing_status_counts"] = self._compact_count_map(
            source.get("model_timing_status_counts"), limit=8
        )
        result["missing_expert_counts"] = self._compact_count_map(
            source.get("missing_expert_counts"), limit=8
        )
        return result

    def _compact_event_feedback(
        self,
        value: Any,
        *,
        include_recent_details: bool = False,
    ) -> dict[str, Any]:
        source = _safe_dict(value)
        result = self._scalar_subset(
            source,
            (
                "total_events",
                "attribution_coverage",
                "attributable_event_coverage",
                "attributable_events",
                "attributable_missing_profile_events",
                "non_attributable_events",
                "missing_profile_events",
                "manual_close_events",
                "max_position_blocks",
                "fallback_blocks",
                "execution_errors",
                "execution_successes",
                "unresolved_execution_errors",
                "latest_execution_success_at",
                "latest_execution_error_at",
                "execution_recovered_after_error",
                "profit_first_defensive_probe_shadow_count",
                "entry_evidence_shadow_only_count",
            ),
        )
        result["type_counts"] = self._compact_count_map(source.get("type_counts"), limit=6)
        result["status_counts"] = self._compact_count_map(source.get("status_counts"), limit=6)
        result["profile_counts"] = self._compact_count_map(source.get("profile_counts"), limit=6)
        result["skip_kind_counts"] = self._compact_count_map(source.get("skip_kind_counts"), limit=8)
        result["top_block_reasons"] = [
            {
                "reason": str(_safe_dict(item).get("reason") or "")[:120],
                "category": str(_safe_dict(item).get("category") or "other")[:80],
                "raw_reason": str(sanitize_text(_safe_dict(item).get("raw_reason")) or "")[:160],
                "count": _safe_dict(item).get("count"),
            }
            for item in _safe_list(source.get("top_block_reasons"))[:5]
        ]
        recent_events = []
        for row in (_safe_dict(item) for item in _safe_list(source.get("recent_events"))[:5]):
            compact_row = {
                "event_type": row.get("event_type"),
                "event_status": row.get("event_status"),
                "symbol": row.get("symbol"),
                "side": row.get("side"),
                "profile_id": row.get("profile_id"),
                "reason": str(row.get("reason") or "")[:100],
                "reason_label": str(row.get("reason_label") or "")[:140],
                "reason_category": str(row.get("reason_category") or "")[:80],
                "skip_kind": str(row.get("skip_kind") or "")[:120],
            }
            if include_recent_details:
                created_at = row.get("created_at")
                compact_row.update(
                    {
                        "id": row.get("id"),
                        "created_at": (
                            created_at.isoformat()
                            if hasattr(created_at, "isoformat")
                            else created_at
                        ),
                        "severity": row.get("severity"),
                        "action": row.get("action"),
                        "order_id": row.get("order_id"),
                        "position_id": row.get("position_id"),
                        "exclude_from_training": bool(row.get("exclude_from_training")),
                    }
                )
            recent_events.append(compact_row)
        result["recent_events"] = recent_events
        return result

    def _compact_reflection_feedback(self, value: Any) -> dict[str, Any]:
        source = _safe_dict(value)
        result = self._scalar_subset(
            source,
            (
                "total_count",
                "training_count",
                "excluded_manual_count",
                "win_count",
                "loss_count",
                "net_reflection_pnl",
                "fee_estimate",
                "fee_adjusted_pnl",
                "avg_hold_minutes",
                "avg_loss_hold_minutes",
                "avg_win_hold_minutes",
                "small_win_count",
                "large_loss_count",
                "loss_sample_count",
                "win_sample_count",
                "mistake_count",
                "improvement_count",
            ),
        )
        result["outcome_counts"] = self._compact_count_map(source.get("outcome_counts"), limit=6)
        result["top_mistakes"] = self._top_text_items(
            source.get("top_mistakes"), "summary", limit=4, text_limit=100
        )
        result["top_improvements"] = self._top_text_items(
            source.get("top_improvements"), "summary", limit=4, text_limit=100
        )
        result["payoff_profile"] = self._compact_payoff_profile(source.get("payoff_profile"))
        return result

    def _compact_problem_items(self, problems: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in _safe_list(problems)[:8]:
            row = _safe_dict(item)
            rows.append(
                {
                    "key": str(row.get("key") or "")[:80],
                    "severity": str(row.get("severity") or "")[:40],
                    "label": str(row.get("label") or "")[:140],
                    "evidence": self._compact_problem_evidence(row.get("evidence")),
                }
            )
        return rows

    def _compact_problem_evidence(self, value: Any) -> dict[str, Any]:
        source = _safe_dict(value)
        result = self._scalar_subset(
            source,
            (
                "trade_count",
                "target",
                "net_pnl",
                "count",
                "pnl",
                "avg_pnl",
                "win_rate",
                "avg_hold_hours",
                "profit_factor",
                "state",
                "open_count",
                "open_part_count",
                "open_group_count",
                "duplicate_part_count",
                "usage_ratio",
                "part_usage_ratio",
                "full_position_pressure",
                "fragmentation_pressure",
                "losing_open_count",
                "market_scans",
                "entry_signals",
                "executed_entries",
                "expert_integrity_blocks",
                "fallback_entry_rate",
                "completed_count",
                "missed_opportunity_rate",
                "bad_signal_rate",
                "total_count",
                "training_count",
                "excluded_manual_count",
                "win_count",
                "loss_count",
                "fee_adjusted_pnl",
                "avg_loss_hold_minutes",
                "small_win_count",
                "large_loss_count",
                "mistake_count",
                "manual_close_events",
                "max_position_blocks",
                "fallback_blocks",
                "execution_errors",
                "attribution_coverage",
                "attributable_event_coverage",
                "attributable_events",
                "attributable_missing_profile_events",
                "non_attributable_events",
                "total_events",
            ),
        )
        for key in (
            "missing_expert_counts",
            "model_timing_status_counts",
            "outcome_counts",
            "side_counts",
            "side_unrealized_pnl",
            "type_counts",
            "status_counts",
            "profile_counts",
        ):
            compact = self._compact_count_map(source.get(key), limit=5)
            if compact:
                result[key] = compact
        if source.get("release_candidates"):
            result["release_candidates"] = self._compact_open_position_pressure(source).get(
                "release_candidates", []
            )
        if source.get("top_block_reasons"):
            result["top_block_reasons"] = self._compact_event_feedback(source).get(
                "top_block_reasons", []
            )
        top_mistakes = self._top_text_items(
            source.get("top_mistakes"), "summary", limit=3, text_limit=90
        )
        if top_mistakes:
            result["top_mistakes"] = top_mistakes
        top_improvements = self._top_text_items(
            source.get("top_improvements"), "summary", limit=3, text_limit=90
        )
        if top_improvements:
            result["top_improvements"] = top_improvements
        payoff_profile = self._compact_payoff_profile(source.get("payoff_profile"))
        if payoff_profile:
            result["payoff_profile"] = payoff_profile
        return result

    @staticmethod
    def _ensure_prompt_budget(prompt: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(_json_safe(prompt), ensure_ascii=False)
        if len(payload) <= LLM_CANDIDATE_PROMPT_MAX_CHARS:
            return prompt
        compact = json.loads(payload)
        summary = _safe_dict(compact.get("feedback_summary"))
        events = _safe_dict(summary.get("event_feedback"))
        events["recent_events"] = [
            {
                "event_type": _safe_dict(item).get("event_type"),
                "event_status": _safe_dict(item).get("event_status"),
                "reason": str(
                    _safe_dict(item).get("reason_label") or _safe_dict(item).get("reason") or ""
                )[:120],
                "reason_category": _safe_dict(item).get("reason_category"),
            }
            for item in _safe_list(events.get("recent_events"))[:3]
        ]
        events["top_block_reasons"] = _safe_list(events.get("top_block_reasons"))[:3]
        summary["event_feedback"] = events
        summary["root_causes"] = _safe_list(summary.get("root_causes"))[:4]
        summary["problems"] = _safe_list(summary.get("problems"))[:5]
        compact["feedback_summary"] = summary
        if len(json.dumps(compact, ensure_ascii=False)) <= LLM_CANDIDATE_PROMPT_MAX_CHARS:
            return compact

        minimal = dict(compact)
        minimal_summary = {
            "totals": summary.get("totals"),
            "side_performance": summary.get("side_performance"),
            "open_position_pressure": {
                key: _safe_dict(summary.get("open_position_pressure")).get(key)
                for key in (
                    "open_count",
                    "open_part_count",
                    "open_group_count",
                    "duplicate_part_count",
                    "max_open_positions",
                    "usage_ratio",
                    "part_usage_ratio",
                    "full_position_pressure",
                    "losing_open_count",
                    "open_unrealized_pnl",
                    "losing_unrealized_pnl",
                )
            },
            "decision_quality": {
                key: _safe_dict(summary.get("decision_quality")).get(key)
                for key in (
                    "market_scans",
                    "entry_signals",
                    "executed_entries",
                    "expert_integrity_blocks",
                    "fallback_entry_rate",
                    "zero_second_entry_decisions",
                )
            },
            "shadow_feedback": summary.get("shadow_feedback"),
            "event_feedback": {
                **{
                    key: events.get(key)
                    for key in (
                        "total_events",
                        "attribution_coverage",
                        "attributable_event_coverage",
                        "attributable_events",
                        "attributable_missing_profile_events",
                        "non_attributable_events",
                        "manual_close_events",
                        "max_position_blocks",
                        "fallback_blocks",
                        "execution_errors",
                        "skip_kind_counts",
                        "profit_first_defensive_probe_shadow_count",
                        "entry_evidence_shadow_only_count",
                    )
                },
                "recent_events": _safe_list(events.get("recent_events"))[:3],
                "top_block_reasons": _safe_list(events.get("top_block_reasons"))[:3],
            },
            "reflection_feedback": {
                key: _safe_dict(summary.get("reflection_feedback")).get(key)
                for key in (
                    "training_count",
                    "excluded_manual_count",
                    "win_count",
                    "loss_count",
                    "fee_adjusted_pnl",
                    "avg_loss_hold_minutes",
                    "small_win_count",
                    "large_loss_count",
                    "payoff_profile",
                    "mistake_count",
                )
            },
            "problems": [
                {
                    "key": _safe_dict(item).get("key"),
                    "severity": _safe_dict(item).get("severity"),
                    "label": str(_safe_dict(item).get("label") or "")[:80],
                }
                for item in _safe_list(summary.get("problems"))[:5]
            ],
            "root_causes": [str(item)[:80] for item in _safe_list(summary.get("root_causes"))[:3]],
            "training_policy": summary.get("training_policy"),
        }
        minimal["feedback_summary"] = minimal_summary
        return minimal

    def _candidate_generation_guidance(self, feedback: StrategyFeedback) -> dict[str, Any]:
        problem_keys = {str(item.get("key") or "") for item in feedback.problems}
        skip_kind_counts = _safe_dict(feedback.event_feedback.get("skip_kind_counts"))
        defensive_probe_shadow_count = _safe_int(
            feedback.event_feedback.get("profit_first_defensive_probe_shadow_count"),
            0,
        )
        profit_first_feedback = _safe_dict(feedback.profit_first_runtime_feedback)
        missed_feedback = _safe_dict(profit_first_feedback.get("missed_opportunity_feedback"))
        lane_feedback = _safe_list(profit_first_feedback.get("lane_feedback"))
        exit_feedback = _safe_list(profit_first_feedback.get("exit_feedback"))
        low_trade_count = bool(feedback.totals.get("low_trade_count_penalty"))
        fallback_or_missed = bool(
            problem_keys
            & {
                "expert_fallback_overblocking",
                "event_fallback_blocks",
                "missed_opportunities",
                "trade_reflection_mistakes",
            }
        )
        missed_positive_shadow_pressure = bool(
            missed_feedback.get("diagnosis") == "system_over_conservative_review"
            or any(
                _safe_dict(row).get("entry_bias") == "expand_quality_entries"
                for row in lane_feedback
            )
        )
        tiny_probe_fee_drag = any(
            _safe_dict(row).get("exit_bias") == "keep_tiny_entries_shadow_only"
            for row in exit_feedback
        )
        require_quality_recovery = bool(
            defensive_probe_shadow_count > 0
            or "defensive_probe_shadow_loop" in problem_keys
            or tiny_probe_fee_drag
            or missed_positive_shadow_pressure
        )
        payoff_repair = _payoff_repair_profile(
            _safe_dict(feedback.totals.get("payoff_profile")),
            _safe_dict(feedback.reflection_feedback.get("payoff_profile")),
        )
        return {
            "primary_issue_keys": sorted(key for key in problem_keys if key),
            "skip_kind_counts": skip_kind_counts,
            "defensive_probe_shadow_count": defensive_probe_shadow_count,
            "require_quality_entry_recovery_candidate": require_quality_recovery,
            "allow_recovery_probe_candidate": bool(
                low_trade_count and fallback_or_missed and not tiny_probe_fee_drag
            ),
            "profit_first_over_conservative_signal": missed_positive_shadow_pressure,
            "profit_first_tiny_probe_fee_drag": tiny_probe_fee_drag,
            "payoff_repair_profile": payoff_repair,
            "payoff_profile_policy": "dynamic_window_distribution_not_fixed_usdt_thresholds",
            "candidate_modes": {
                "quality_entry_recovery": (
                    "Use when low-payoff probes are being shadowed. Do not set "
                    "probe_fraction or max_probe_size_pct; keep strict expert integrity so "
                    "existing Profit-First quality gates decide whether real entries are large "
                    "enough to matter."
                ),
                "recovery_probe": (
                    "Use only when fallback, missed opportunity, or low-sample feedback is the "
                    "primary issue and there is no defensive-probe loop. Use bounded "
                    "probe_fraction/max_probe_size_pct."
                ),
                "loss_release": (
                    "Use when current low-quality losing positions or capacity pressure are the "
                    "primary issue."
                ),
                "winner_hold": (
                    "Use when the dynamic payoff profile shows low payoff ratio, weak profit "
                    "factor, or large-loss distribution pressure. Do not use fixed USDT cutoffs; "
                    "derive payoff_repair_intensity from generation_guidance.payoff_repair_profile."
                ),
            },
        }

    def _llm_candidate_prompt(self, feedback: StrategyFeedback) -> dict[str, Any]:
        event_feedback = feedback.event_feedback
        reflection_feedback = feedback.reflection_feedback
        generation_guidance = self._candidate_generation_guidance(feedback)
        recent_events = []
        for item in _safe_list(event_feedback.get("recent_events"))[:8]:
            if not isinstance(item, dict):
                continue
            recent_events.append(
                {
                    "event_type": item.get("event_type"),
                    "event_status": item.get("event_status"),
                    "symbol": item.get("symbol"),
                    "side": item.get("side"),
                    "profile_id": item.get("profile_id"),
                    "reason": str(item.get("reason") or "")[:140],
                }
            )
        return {
            "task": "generate_bounded_strategy_profile_candidates",
            "language": "zh-CN",
            "rules": [
                '只返回 JSON 对象，不要 Markdown。格式必须是 {"candidates": [...]}。',
                "不能生成 Python、SQL、shell 或任意可执行逻辑。只能设置 allowed_params 白名单参数。",
                (
                    "不要把所有候选都强制生成小仓探针；只有 recovery_probe 模式才设置 "
                    "probe_fraction/max_probe_size_pct。"
                ),
                (
                    "如果 generation_guidance.require_quality_entry_recovery_candidate=true，"
                    "至少生成一个 quality_entry_recovery 候选，不设置 probe_fraction 和 "
                    "max_probe_size_pct。"
                ),
                (
                    "如果 generation_guidance.payoff_repair_profile.triggered=true，"
                    "优先生成 winner_hold 候选；payoff_repair_intensity 必须来自动态收益分布画像，"
                    "不要使用固定 USDT 阈值。"
                ),
                "评分目标是手续费后净收益、交易次数、回撤、亏损释放和错过机会减少，不能用不开仓提高胜率。",
            ],
            "allowed_params": sorted(ALLOWED_CANDIDATE_PARAM_KEYS),
            "generation_guidance": generation_guidance,
            "feedback_summary": {
                "totals": {
                    "training_trade_count": feedback.totals.get("training_trade_count"),
                    "trade_count_target": feedback.totals.get("trade_count_target"),
                    "net_pnl": feedback.totals.get("net_pnl"),
                    "win_rate": feedback.totals.get("win_rate"),
                    "avg_loss_hold_hours": feedback.totals.get("avg_loss_hold_hours"),
                    "payoff_profile": self._compact_payoff_profile(
                        feedback.totals.get("payoff_profile")
                    ),
                },
                "side_performance": feedback.side_performance,
                "open_position_pressure": feedback.open_position_pressure,
                "decision_quality": {
                    "market_scans": feedback.decision_quality.get("market_scans"),
                    "entry_signals": feedback.decision_quality.get("entry_signals"),
                    "executed_entries": feedback.decision_quality.get("executed_entries"),
                    "expert_integrity_blocks": feedback.decision_quality.get(
                        "expert_integrity_blocks"
                    ),
                    "fallback_entry_rate": feedback.decision_quality.get("fallback_entry_rate"),
                    "missing_expert_counts": feedback.decision_quality.get("missing_expert_counts"),
                },
                "shadow_feedback": feedback.shadow_feedback,
                "event_feedback": {
                    "total_events": event_feedback.get("total_events"),
                    "attribution_coverage": event_feedback.get("attribution_coverage"),
                    "manual_close_events": event_feedback.get("manual_close_events"),
                    "max_position_blocks": event_feedback.get("max_position_blocks"),
                    "fallback_blocks": event_feedback.get("fallback_blocks"),
                    "execution_errors": event_feedback.get("execution_errors"),
                    "skip_kind_counts": event_feedback.get("skip_kind_counts"),
                    "profit_first_defensive_probe_shadow_count": event_feedback.get(
                        "profit_first_defensive_probe_shadow_count"
                    ),
                    "recent_events": recent_events,
                },
                "reflection_feedback": {
                    "training_count": reflection_feedback.get("training_count"),
                    "excluded_manual_count": reflection_feedback.get("excluded_manual_count"),
                    "outcome_counts": reflection_feedback.get("outcome_counts"),
                    "fee_adjusted_pnl": reflection_feedback.get("fee_adjusted_pnl"),
                    "avg_loss_hold_minutes": reflection_feedback.get("avg_loss_hold_minutes"),
                    "small_win_count": reflection_feedback.get("small_win_count"),
                    "large_loss_count": reflection_feedback.get("large_loss_count"),
                    "payoff_profile": self._compact_payoff_profile(
                        reflection_feedback.get("payoff_profile")
                    ),
                    "top_mistakes": self._top_text_items(
                        reflection_feedback.get("top_mistakes"), "summary", include_count=False
                    ),
                    "top_improvements": self._top_text_items(
                        reflection_feedback.get("top_improvements"), "summary", include_count=False
                    ),
                },
                "problems": feedback.problems,
                "training_policy": feedback.training_policy,
            },
            "response_schema": {
                "candidates": [
                    {
                        "profile_id": "string",
                        "label": "string",
                        "description": "string",
                        "params": {"only_allowed_keys": sorted(ALLOWED_CANDIDATE_PARAM_KEYS)},
                    }
                ]
            },
        }

    def _llm_candidate_prompt_v3(self, feedback: StrategyFeedback) -> dict[str, Any]:
        event_feedback = feedback.event_feedback
        reflection_feedback = feedback.reflection_feedback
        generation_guidance = self._candidate_generation_guidance(feedback)
        prompt = {
            "task": "generate_bounded_strategy_profile_candidates",
            "language": "zh-CN",
            "rules": [
                'Return one JSON object only, no Markdown. Format: {"candidates": [...] }.',
                "Do not generate Python, SQL, shell, or arbitrary executable logic.",
                "Only set whitelisted allowed_params. Keep candidates bounded and reversible.",
                (
                    "Do not force every candidate into probe mode. Use probe_fraction and "
                    "max_probe_size_pct only for recovery_probe candidates."
                ),
                (
                    "If generation_guidance.require_quality_entry_recovery_candidate is true, "
                    "include one quality_entry_recovery candidate without probe_fraction or "
                    "max_probe_size_pct so quality signals can be sized by existing dynamic "
                    "Profit-First gates."
                ),
                (
                    "If generation_guidance.payoff_repair_profile.triggered is true, include a "
                    "winner_hold candidate. Set payoff_repair_intensity from the dynamic payoff "
                    "distribution profile and pass winner_hold_dynamic evidence; do not use fixed "
                    "USDT cutoffs."
                ),
                "Do not optimize by avoiding trades; low trade count must be penalized.",
                "Use concise Chinese label and description fields.",
                "Return at most 2 candidates. label <= 12 chars, description <= 40 chars.",
                "Each candidate params must contain 3 to 5 keys only.",
            ],
            "allowed_params": sorted(ALLOWED_CANDIDATE_PARAM_KEYS),
            "generation_guidance": generation_guidance,
            "feedback_summary": {
                "totals": {
                    **self._scalar_subset(
                        feedback.totals,
                        (
                            "training_trade_count",
                            "trade_count_target",
                            "net_pnl",
                            "win_rate",
                            "small_win_count",
                            "large_loss_count",
                            "avg_hold_hours",
                            "avg_loss_hold_hours",
                            "low_trade_count_penalty",
                        ),
                    ),
                    "payoff_profile": self._compact_payoff_profile(
                        feedback.totals.get("payoff_profile")
                    ),
                },
                "side_performance": {
                    side: self._scalar_subset(
                        bucket,
                        (
                            "count",
                            "wins",
                            "losses",
                            "pnl",
                            "avg_pnl",
                            "win_rate",
                            "avg_hold_hours",
                            "profit_factor",
                            "largest_loss",
                            "loss_pressure",
                            "state",
                        ),
                    )
                    for side, bucket in feedback.side_performance.items()
                },
                "open_position_pressure": self._compact_open_position_pressure(
                    feedback.open_position_pressure
                ),
                "decision_quality": self._compact_decision_quality(feedback.decision_quality),
                "shadow_feedback": self._scalar_subset(
                    feedback.shadow_feedback,
                    (
                        "completed_count",
                        "missed_opportunity_count",
                        "bad_signal_count",
                        "good_signal_count",
                        "missed_opportunity_rate",
                        "bad_signal_rate",
                    ),
                ),
                "event_feedback": self._compact_event_feedback(event_feedback),
                "reflection_feedback": self._compact_reflection_feedback(reflection_feedback),
                "profit_first_runtime_feedback": _compact_profit_first_runtime_feedback(
                    feedback.profit_first_runtime_feedback
                ),
                "problems": self._compact_problem_items(feedback.problems),
                "root_causes": [str(item)[:140] for item in feedback.root_causes[:8]],
                "training_policy": self._scalar_subset(
                    feedback.training_policy,
                    (
                        "manual_close_excluded",
                        "low_trade_count_is_penalized",
                        "arbitrary_code_generation_allowed",
                        "candidate_profiles_only",
                    ),
                ),
            },
            "response_schema": {
                "candidates": [
                    {
                        "profile_id": "string",
                        "label": "string <= 12 chars",
                        "description": "string <= 40 chars",
                        "params": {"only_allowed_keys": sorted(ALLOWED_CANDIDATE_PARAM_KEYS)},
                    }
                ]
            },
        }
        return self._ensure_prompt_budget(prompt)

    def _llm_candidate_retry_prompt(self, prompt: dict[str, Any]) -> dict[str, Any]:
        summary = _safe_dict(prompt.get("feedback_summary"))
        return {
            "task": "generate_one_bounded_strategy_candidate",
            "language": "zh-CN",
            "rules": [
                "Return minified JSON only. No thinking, no Markdown, no explanation.",
                "Return exactly one candidate in candidates[].",
                "Use only 3 to 4 allowed params and keep strings short.",
                (
                    "If generation_guidance requires quality_entry_recovery, return a candidate "
                    "without probe_fraction/max_probe_size_pct; otherwise a bounded recovery "
                    "probe is allowed."
                ),
                'Valid shape: {"candidates":[{"profile_id":"llm_quality_recovery","label":"质量恢复","description":"恢复高质量正常开仓","params":{"position_size_multiplier":1.0,"expert_integrity_mode":"strict_all_required","global_min_score_delta":0.0}}]}',
            ],
            "allowed_params": sorted(ALLOWED_CANDIDATE_PARAM_KEYS),
            "generation_guidance": prompt.get("generation_guidance"),
            "feedback_summary": {
                "totals": summary.get("totals"),
                "side_performance": summary.get("side_performance"),
                "open_position_pressure": summary.get("open_position_pressure"),
                "decision_quality": summary.get("decision_quality"),
                "reflection_feedback": summary.get("reflection_feedback"),
                "profit_first_runtime_feedback": summary.get("profit_first_runtime_feedback"),
                "problems": [
                    {
                        "key": _safe_dict(item).get("key"),
                        "severity": _safe_dict(item).get("severity"),
                    }
                    for item in _safe_list(summary.get("problems"))[:5]
                ],
                "root_causes": _safe_list(summary.get("root_causes"))[:3],
            },
        }

    async def _call_llm_candidate_model(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        feedback: StrategyFeedback,
    ) -> list[dict[str, Any]]:
        prompt = self._llm_candidate_prompt_v3(feedback)
        max_tokens = completion_token_limit(
            "proxy",
            _safe_int(getattr(settings, "strategy_learning_llm_candidate_max_tokens", 360), 360),
            floor=160,
        )
        prompt = _json_safe(prompt)
        body = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是量化交易策略参数编译器。只输出 JSON 对象，不输出 Markdown，不输出代码。",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "stream": False,
        }
        body = apply_non_thinking_request_controls(model, body)
        timeout = max(
            5.0,
            _safe_float(
                getattr(settings, "strategy_learning_llm_candidate_timeout_seconds", 20.0),
                20.0,
            ),
        )
        try:
            response = await self._post_llm_candidate_request(
                api_base=api_base,
                api_key=api_key,
                body=body,
                timeout_seconds=timeout,
                retry=False,
            )
        except StrategyCandidateModelError as exc:
            if exc.kind != LLM_CANDIDATE_ERROR_TIMEOUT:
                raise
            retry_prompt = _json_safe(self._llm_candidate_retry_prompt(prompt))
            retry_body = self._llm_candidate_retry_body(
                model=model,
                prompt=retry_prompt,
                max_tokens=max_tokens,
            )
            response = await self._post_llm_candidate_request(
                api_base=api_base,
                api_key=api_key,
                body=retry_body,
                timeout_seconds=timeout,
                retry=True,
            )
        payload = self._response_json(response)
        content = self._extract_llm_content(payload)
        try:
            parsed = self._parse_json_object(content)
        except RuntimeError as exc:
            if "JSON was incomplete" not in str(exc) and "did not contain JSON" not in str(exc):
                raise StrategyCandidateModelError(
                    safe_error_text(exc, limit=220),
                    kind=LLM_CANDIDATE_ERROR_INVALID_JSON,
                ) from exc
            retry_prompt = _json_safe(self._llm_candidate_retry_prompt(prompt))
            retry_body = self._llm_candidate_retry_body(
                model=model,
                prompt=retry_prompt,
                max_tokens=max_tokens,
            )
            retry_response = await self._post_llm_candidate_request(
                api_base=api_base,
                api_key=api_key,
                body=retry_body,
                timeout_seconds=timeout,
                retry=True,
            )
            retry_payload = self._response_json(retry_response)
            retry_content = self._extract_llm_content(retry_payload)
            try:
                parsed = self._parse_json_object(retry_content)
            except RuntimeError as retry_exc:
                raise StrategyCandidateModelError(
                    safe_error_text(retry_exc, limit=220),
                    kind=LLM_CANDIDATE_ERROR_INVALID_JSON,
                ) from retry_exc
        return _safe_list(parsed.get("candidates"))[:LLM_CANDIDATE_MAX_COUNT]

    async def _post_llm_candidate_request(
        self,
        *,
        api_base: str,
        api_key: str,
        body: dict[str, Any],
        timeout_seconds: float,
        retry: bool,
    ) -> httpx.Response:
        label = "retry" if retry else "request"
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(
                    f"{api_base}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                )
        except httpx.TimeoutException as exc:
            raise StrategyCandidateModelError(
                f"strategy candidate {label} timed out after {timeout_seconds:.1f}s",
                kind=LLM_CANDIDATE_ERROR_TIMEOUT,
            ) from exc
        except httpx.HTTPError as exc:
            raise StrategyCandidateModelError(
                f"strategy candidate {label} transport failed: {safe_error_text(exc, limit=180)}",
                kind=LLM_CANDIDATE_ERROR_HTTP,
            ) from exc
        if not response.is_success:
            detail = safe_response_error_text(response, limit=300)
            message = f"strategy candidate {label} failed with HTTP {response.status_code}"
            if detail:
                message = f"{message}: {detail}"
            raise StrategyCandidateModelError(message, kind=LLM_CANDIDATE_ERROR_HTTP)
        return response

    @staticmethod
    def _response_json(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise StrategyCandidateModelError(
                "strategy candidate HTTP response body was not valid JSON",
                kind=LLM_CANDIDATE_ERROR_INVALID_JSON,
            ) from exc
        if not isinstance(payload, dict):
            raise StrategyCandidateModelError(
                "strategy candidate HTTP response body was not a JSON object",
                kind=LLM_CANDIDATE_ERROR_INVALID_JSON,
            )
        return payload

    @staticmethod
    def _llm_candidate_retry_body(
        *, model: str, prompt: dict[str, Any], max_tokens: int
    ) -> dict[str, Any]:
        retry_body = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是量化交易策略参数编译器。只输出一个完整 JSON 对象，"
                        "不要 Markdown，不要解释，不要代码。"
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "temperature": 0.1,
            "max_tokens": min(max_tokens, 260),
            "stream": False,
        }
        return apply_non_thinking_request_controls(model, retry_body)

    @staticmethod
    def _extract_llm_content(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") if isinstance(payload, dict) else []
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = _safe_dict(_safe_dict(choice).get("message"))
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
            return "\n".join(part for part in parts if part).strip()
        return ""

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        stripped = str(text or "").strip()
        think_end = stripped.lower().rfind("</think>")
        if think_end >= 0:
            stripped = stripped[think_end + len("</think>") :].strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = json.loads(StrategyLearningService._first_json_object(stripped))
        if not isinstance(parsed, dict):
            raise RuntimeError("strategy candidate response was not a JSON object")
        return parsed

    @staticmethod
    def _first_json_object(text: str) -> str:
        start = text.find("{")
        if start < 0:
            raise RuntimeError("strategy candidate response did not contain JSON")
        depth = 0
        in_string = False
        escaped = False
        for index, char in enumerate(text[start:], start=start):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        raise RuntimeError("strategy candidate response JSON was incomplete")

    def _store_llm_candidate_cache(
        self,
        *,
        mode: str,
        signature: str,
        candidates: list[dict[str, Any]],
        error: str,
        error_kind: str = "",
        attempts: list[dict[str, Any]] | None = None,
        model: str = "",
    ) -> None:
        state = self.state_store.load()
        sanitized_candidates = [self._sanitize_cached_candidate(item) for item in candidates]
        state["llm_candidate_cache"] = {
            "mode": "live" if str(mode).lower() == "live" else "paper",
            "signature": signature,
            "generated_at": datetime.now(UTC).isoformat(),
            "candidates": _json_safe(sanitized_candidates),
            "last_error": str(sanitize_text(error) or ""),
            "last_error_kind": str(error_kind or ""),
            "attempts": sanitize_payload(_json_safe(attempts or [])),
            "model": model,
            "prompt_version": LLM_CANDIDATE_PROMPT_VERSION,
            "source": "llm_structured_candidate" if candidates else "none",
        }
        self.state_store.save(state)

    @staticmethod
    def _llm_candidate_error_kind(exc: BaseException) -> str:
        kind = getattr(exc, "kind", "")
        if isinstance(kind, str) and kind:
            return kind
        if isinstance(exc, httpx.TimeoutException):
            return LLM_CANDIDATE_ERROR_TIMEOUT
        if isinstance(exc, httpx.HTTPError):
            return LLM_CANDIDATE_ERROR_HTTP
        return LLM_CANDIDATE_ERROR_UNKNOWN

    @staticmethod
    def _sanitize_cached_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        data = sanitize_payload(_json_safe(candidate))
        clean = _safe_dict(data)
        if "label" in clean:
            clean["label"] = str(clean.get("label") or "")[:80]
        if "description" in clean:
            clean["description"] = str(clean.get("description") or "")[:500]
        return clean

    async def record_event(
        self,
        *,
        mode: str,
        model_name: str = ENSEMBLE_TRADER_NAME,
        symbol: str | None = None,
        action: str | None = None,
        side: str | None = None,
        event_type: str,
        event_status: str = "recorded",
        reason: str | None = None,
        severity: str = "info",
        decision_id: int | None = None,
        order_id: int | None = None,
        position_id: int | None = None,
        strategy_context: dict[str, Any] | None = None,
        raw_response: dict[str, Any] | None = None,
        market_state: dict[str, Any] | None = None,
        attribution: dict[str, Any] | None = None,
        exclude_from_training: bool = False,
    ) -> int | None:
        selected_mode = "live" if str(mode or "").lower() == "live" else "paper"
        context = _safe_dict(strategy_context)
        learning = _safe_dict(context.get("strategy_learning"))
        active_profile = _safe_dict(learning.get("active_profile"))
        runtime = _safe_dict(learning.get("runtime"))
        profile_id = str(
            context.get("strategy_profile_id")
            or active_profile.get("id")
            or runtime.get("profile_id")
            or ""
        )
        profile_version = _safe_int(
            context.get("strategy_profile_version")
            or active_profile.get("version")
            or runtime.get("profile_version"),
            0,
        )
        event_side = side or _action_side(action)
        raw = _safe_dict(raw_response)
        event = StrategyLearningEvent(
            model_name=model_name or ENSEMBLE_TRADER_NAME,
            execution_mode=selected_mode,
            symbol=symbol,
            side=None if event_side == "unknown" else event_side,
            action=action,
            event_type=event_type,
            event_status=event_status,
            severity=severity,
            reason=str(sanitize_runtime_text(reason or "") or "")[:2000],
            decision_id=decision_id,
            order_id=order_id,
            position_id=position_id,
            profile_id=profile_id or None,
            profile_version=profile_version or None,
            scheduler_reason=str(
                sanitize_runtime_text(
                    context.get("scheduler_reason") or learning.get("reason") or ""
                )
                or ""
            )[:2000],
            strategy_snapshot=sanitize_runtime_text(
                _json_safe(
                    {
                        "active_profile": active_profile,
                        "runtime": runtime,
                        "feedback_summary": learning.get("feedback_summary"),
                        "rollback": learning.get("rollback"),
                        "scheduler_mode": learning.get("scheduler_mode", "auto"),
                        "manual_profile_id": learning.get("manual_profile_id", ""),
                        "candidate_count": learning.get("candidate_count", 0),
                        "dispatch_reason": learning.get("dispatch_reason")
                        or learning.get("reason"),
                        "disabled_profiles": learning.get("disabled_profiles", []),
                        "shadow_validation": learning.get("shadow_validation", {}),
                        "probe": learning.get("probe", {}),
                        "backtest": learning.get("backtest", {}),
                        "training_policy": learning.get("training_policy", {}),
                        "exclude_from_training": bool(exclude_from_training),
                    }
                ),
            ),
            market_state=sanitize_runtime_text(
                _json_safe(market_state or context.get("market_regime") or {})
            ),
            side_weights=sanitize_runtime_text(
                _json_safe(runtime.get("side_weights") or _safe_dict(context.get("side_quality")))
            ),
            expert_integrity=sanitize_runtime_text(
                _json_safe(self._expert_integrity_event_payload(raw, context))
            ),
            attribution=sanitize_runtime_text(_json_safe(attribution or {})),
            exclude_from_training=bool(exclude_from_training),
        )
        try:
            async with get_session_ctx() as session:
                session.add(event)
                await session.flush()
                return int(event.id)
        except Exception as exc:
            logger.debug(
                "failed to record strategy learning event",
                event_type=event_type,
                symbol=symbol,
                error=safe_error_text(exc),
            )
            return None

    async def _persist_profile_snapshots(self, mode: str, payload: dict[str, Any]) -> None:
        schedule = _safe_dict(payload.get("schedule"))
        active_id = str(_safe_dict(schedule.get("active_profile")).get("id") or "")
        candidates = _safe_list(schedule.get("candidates"))
        if not candidates:
            return
        selected_mode = "live" if str(mode or "").lower() == "live" else "paper"
        signature = self._profile_snapshot_signature(selected_mode, schedule)
        if not self._should_persist_profile_snapshots(selected_mode, signature, active_id):
            return
        backtest_rows = {
            str(row.get("profile_id")): row
            for row in _safe_list(_safe_dict(schedule.get("backtest")).get("rows"))
            if isinstance(row, dict)
        }
        shadow_rows = {
            str(row.get("profile_id")): row
            for row in _safe_list(_safe_dict(schedule.get("shadow_validation")).get("rows"))
            if isinstance(row, dict)
        }
        probe = _safe_dict(schedule.get("probe"))
        disabled = set(_safe_list(schedule.get("disabled_profiles")))
        try:
            async with get_session_ctx() as session:
                for profile in candidates:
                    if not isinstance(profile, dict):
                        continue
                    profile_id = str(profile.get("id") or "")
                    if not profile_id:
                        continue
                    session.add(
                        StrategyProfileSnapshot(
                            execution_mode=selected_mode,
                            profile_id=profile_id,
                            version=_safe_int(profile.get("version"), 1),
                            label=str(profile.get("label") or ""),
                            status=str(profile.get("status") or "candidate"),
                            source=str(profile.get("source") or "feedback_generator"),
                            description=str(profile.get("description") or ""),
                            params=_json_safe(_safe_dict(profile.get("params"))),
                            promotion=_json_safe(_safe_dict(profile.get("promotion"))),
                            backtest_metrics=_json_safe(backtest_rows.get(profile_id, {})),
                            shadow_validation=_json_safe(shadow_rows.get(profile_id, {})),
                            probe_state=_json_safe(
                                probe if probe.get("profile_id") == profile_id else {}
                            ),
                            scheduler_reason=str(schedule.get("reason") or "")[:2000],
                            is_active=profile_id == active_id,
                            is_disabled=profile_id in disabled,
                        )
                    )
            self.state_store.mark_profile_snapshot_persisted(
                mode=selected_mode,
                signature=signature,
                active_profile_id=active_id,
            )
        except Exception as exc:
            logger.debug(
                "failed to persist strategy profile snapshots",
                error=safe_error_text(exc),
            )

    @staticmethod
    def _profile_snapshot_signature(mode: str, schedule: dict[str, Any]) -> str:
        data = {
            "mode": mode,
            "active_profile": _safe_dict(schedule.get("active_profile")),
            "candidate_ids": [
                str(profile.get("id") or "")
                for profile in _safe_list(schedule.get("candidates"))
                if isinstance(profile, dict)
            ],
            "disabled_profiles": _safe_list(schedule.get("disabled_profiles")),
            "backtest": _safe_dict(schedule.get("backtest")).get("rows", []),
            "shadow": _safe_dict(schedule.get("shadow_validation")).get("rows", []),
        }
        raw = json.dumps(_json_safe(data), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _should_persist_profile_snapshots(
        self,
        mode: str,
        signature: str,
        active_profile_id: str,
    ) -> bool:
        entry = _safe_dict(self.state_store.load().get("last_profile_snapshot"))
        if entry.get("mode") != mode:
            return True
        if entry.get("signature") != signature:
            return True
        if entry.get("active_profile_id") != active_profile_id:
            return True
        persisted_at = _parse_iso_datetime(entry.get("persisted_at"))
        if not persisted_at:
            return True
        elapsed = (datetime.now(UTC) - persisted_at).total_seconds()
        return elapsed >= PROFILE_SNAPSHOT_MIN_INTERVAL_SECONDS

    def _runtime_guard(
        self,
        payload: dict[str, Any],
        *,
        mutate: bool = False,
    ) -> dict[str, Any]:
        schedule = _safe_dict(payload.get("schedule"))
        feedback = _safe_dict(payload.get("feedback"))
        active = _safe_dict(schedule.get("active_profile"))
        profile_id = str(active.get("id") or "baseline_current")
        totals = _safe_dict(feedback.get("totals"))
        decision_quality = _safe_dict(feedback.get("decision_quality"))
        event_feedback = _safe_dict(feedback.get("event_feedback"))
        backtest_rows = _safe_list(_safe_dict(schedule.get("backtest")).get("rows"))
        shadow_rows = _safe_list(_safe_dict(schedule.get("shadow_validation")).get("rows"))
        score = _safe_dict(
            next((row for row in backtest_rows if row.get("profile_id") == profile_id), {})
        )
        shadow = _safe_dict(
            next((row for row in shadow_rows if row.get("profile_id") == profile_id), {})
        )
        runtime = _safe_dict(schedule.get("runtime"))
        active_params = _safe_dict(active.get("params"))
        probe_fraction = max(
            _safe_float(runtime.get("probe_fraction"), 0.0),
            _safe_float(active_params.get("probe_fraction"), 0.0),
        )
        matched = {str(item) for item in _safe_list(score.get("matched_fixes"))}
        recent_fallback_rate = _safe_float(
            decision_quality.get("recent_fallback_entry_rate"),
            _safe_float(decision_quality.get("fallback_entry_rate"), 0.0),
        )
        recent_integrity_blocks = _safe_int(
            decision_quality.get("recent_expert_integrity_blocks"),
            _safe_int(decision_quality.get("expert_integrity_blocks"), 0),
        )
        recent_zero_second_entries = _safe_int(
            decision_quality.get("recent_zero_second_entry_decisions"),
            _safe_int(decision_quality.get("zero_second_entry_decisions"), 0),
        )
        unresolved_execution_errors = _safe_int(
            event_feedback.get("unresolved_execution_errors"),
            _safe_int(event_feedback.get("execution_errors"), 0),
        )
        unresolved_execution_guard_errors = _safe_int(
            event_feedback.get("unresolved_execution_guard_errors"),
            unresolved_execution_errors,
        )
        execution_recovered_after_error = bool(
            event_feedback.get("execution_recovered_after_error")
            or (
                event_feedback.get("latest_execution_success_at")
                and not unresolved_execution_errors
            )
        )
        model_health_recovered = bool(
            decision_quality.get("model_health_recovered")
            or execution_recovered_after_error
            or (
                _safe_int(decision_quality.get("recent_entry_signals"), 0) > 0
                and recent_fallback_rate < 0.20
                and recent_integrity_blocks == 0
                and recent_zero_second_entries == 0
            )
        )
        validated_probe = bool(
            probe_fraction > 0
            and score.get("pass") is not False
            and shadow.get("eligible") is not False
            and (
                matched
                & {
                    "expert_fallback_overblocking",
                    "missed_opportunities",
                    "low_trade_count",
                    "controlled_entry_recovery",
                    "trade_reflection_mistakes",
                }
                or shadow.get("would_reduce_blocks")
                or shadow.get("would_increase_entries")
            )
        )
        reasons: list[str] = []
        should_rollback = False
        if profile_id != "baseline_current":
            if _safe_float(totals.get("net_pnl"), 0.0) <= -8.0:
                reasons.append("recent_net_pnl_guard")
                if not validated_probe or _safe_int(totals.get("training_trade_count"), 0) >= max(
                    3, default_min_trade_target() // 2
                ):
                    should_rollback = True
            if _safe_int(totals.get("training_trade_count"), 0) < max(
                2, default_min_trade_target() // 3
            ):
                reasons.append("insufficient_trade_samples")
            if _safe_float(decision_quality.get("fallback_entry_rate"), 0.0) >= 0.55:
                reasons.append("fallback_dependency_guard")
            if unresolved_execution_guard_errors >= 3:
                reasons.append("execution_error_guard")
                if not model_health_recovered:
                    should_rollback = True
        if should_rollback and mutate:
            disabled_until = (
                datetime.now(UTC) + timedelta(seconds=AUTO_DISABLED_PROFILE_RECONSIDER_SECONDS)
            ).isoformat()
            self.state_store.set_profile_disabled(
                profile_id,
                disabled=True,
                reason="auto_runtime_guard:" + ",".join(reasons),
            )
            self.state_store.set_manual_active_profile(None)
        else:
            disabled_until = ""
        return {
            "profile_id": profile_id,
            "should_rollback": bool(should_rollback),
            "mutated": bool(should_rollback and mutate),
            "reasons": reasons,
            "auto_disable_reconsider_seconds": AUTO_DISABLED_PROFILE_RECONSIDER_SECONDS,
            "disabled_until": disabled_until,
            "model_health_recovered": model_health_recovered,
            "fallback_health_guard_active": bool(
                "fallback_dependency_guard" in reasons and not model_health_recovered
            ),
            "recovery_probe_allowed": bool(
                "fallback_dependency_guard" in reasons
                and not model_health_recovered
                and "execution_error_guard" not in reasons
            ),
            "unresolved_execution_errors": unresolved_execution_errors,
            "unresolved_execution_guard_errors": unresolved_execution_guard_errors,
            "execution_recovered_after_error": execution_recovered_after_error,
            "recent_fallback_entry_rate": recent_fallback_rate,
            "recent_expert_integrity_blocks": recent_integrity_blocks,
            "rules": [
                "\u63a2\u9488\u6216\u5019\u9009\u7b56\u7565\u8fd1\u671f\u51c0\u6536\u76ca\u663e\u8457\u6076\u5316\u65f6\u81ea\u52a8\u56de\u6eda",
                "fallback \u4f9d\u8d56\u6216\u6267\u884c\u5f02\u5e38\u5360\u4e3b\u5bfc\u65f6\u81ea\u52a8\u56de\u6eda",
                "\u4f4e\u4ea4\u6613\u91cf\u4f1a\u88ab\u60e9\u7f5a\uff0c\u8c03\u5ea6\u5668\u4e0d\u80fd\u7528\u4e0d\u5f00\u4ed3\u6765\u5236\u9020\u5b89\u5168\u611f",
            ],
        }

    def _expert_integrity_event_payload(
        self,
        raw_response: dict[str, Any],
        strategy_context: dict[str, Any],
    ) -> dict[str, Any]:
        compiler = getattr(self.engine, "compiler", None) or StrategyFeedbackCompiler()
        fallback, zero_second, missing = compiler._expert_integrity_flags(raw_response)
        return {
            "fallback_or_untrusted": bool(fallback),
            "zero_second_expert": bool(zero_second),
            "missing_experts": missing,
            "mode": strategy_context.get("expert_integrity_mode"),
            "model_timing_count": len(_safe_list(raw_response.get("model_timings"))),
        }

    async def _load_rows(self, *, mode: str, hours: int, limit: int) -> dict[str, list[Any]]:
        selected_mode = "live" if str(mode or "").lower() == "live" else "paper"
        is_paper = selected_mode == "paper"
        params = STRATEGY_LEARNING_PARAMS
        capped_hours = max(1, min(int(hours or DEFAULT_LOOKBACK_HOURS), params.max_lookback_hours))
        capped_limit = max(
            params.min_dashboard_limit,
            min(int(limit or params.dashboard_default_limit), params.dashboard_full_limit),
        )
        since = max(datetime.now(UTC) - timedelta(hours=capped_hours), PHASE3_CLEAN_START_UTC)
        async with get_read_session_ctx() as session:
            closed_result = await session.execute(
                select(Position)
                .where(
                    Position.model_name.in_(EXECUTION_LEDGER_MODEL_NAMES),
                    Position.execution_mode == selected_mode,
                    Position.is_open.is_(False),
                    Position.closed_at.is_not(None),
                    Position.closed_at >= since,
                )
                .order_by(Position.closed_at.desc(), Position.created_at.desc())
                .limit(capped_limit)
            )
            open_result = await session.execute(
                select(Position)
                .where(
                    Position.model_name.in_(EXECUTION_LEDGER_MODEL_NAMES),
                    Position.execution_mode == selected_mode,
                    Position.is_open.is_(True),
                    Position.created_at >= PHASE3_CLEAN_START_UTC,
                )
                .order_by(Position.created_at.desc())
                .limit(capped_limit)
            )
            order_result = await session.execute(
                select(Order)
                .where(
                    Order.model_name == ENSEMBLE_TRADER_NAME,
                    Order.execution_mode == selected_mode,
                    Order.created_at >= since - timedelta(hours=params.order_extra_lookback_hours),
                )
                .order_by(Order.created_at.desc())
                .limit(capped_limit * params.market_event_limit_multiplier)
            )
            decision_result = await session.execute(
                select(AIDecision)
                .where(
                    AIDecision.model_name == ENSEMBLE_TRADER_NAME,
                    AIDecision.is_paper.is_(is_paper),
                    AIDecision.created_at >= since,
                )
                .order_by(AIDecision.created_at.desc())
                .limit(capped_limit)
            )
            shadow_result = await session.execute(
                select(ShadowBacktest)
                .where(
                    ShadowBacktest.model_name == ENSEMBLE_TRADER_NAME,
                    ShadowBacktest.execution_mode == selected_mode,
                    ShadowBacktest.created_at >= since,
                )
                .order_by(ShadowBacktest.created_at.desc())
                .limit(capped_limit)
            )
            memory_result = await session.execute(
                select(ExpertMemory)
                .where(
                    ExpertMemory.created_at
                    >= since - timedelta(days=params.expert_memory_lookback_days)
                )
                .order_by(
                    ExpertMemory.updated_at.desc().nullslast(), ExpertMemory.created_at.desc()
                )
                .limit(params.expert_memory_limit)
            )
            reflection_result = await session.execute(
                select(TradeReflection)
                .where(
                    TradeReflection.model_name == ENSEMBLE_TRADER_NAME,
                    TradeReflection.execution_mode == selected_mode,
                    TradeReflection.closed_at >= since,
                )
                .order_by(TradeReflection.closed_at.desc().nullslast())
                .limit(capped_limit)
            )
            strategy_event_result = await session.execute(
                select(StrategyLearningEvent)
                .where(
                    StrategyLearningEvent.model_name == ENSEMBLE_TRADER_NAME,
                    StrategyLearningEvent.execution_mode == selected_mode,
                    StrategyLearningEvent.created_at >= since,
                )
                .order_by(StrategyLearningEvent.created_at.desc())
                .limit(capped_limit * params.market_event_limit_multiplier)
            )
            return {
                "closed_positions": list(closed_result.scalars().all()),
                "open_positions": list(open_result.scalars().all()),
                "orders": list(order_result.scalars().all()),
                "decisions": list(decision_result.scalars().all()),
                "shadows": list(shadow_result.scalars().all()),
                "memories": list(memory_result.scalars().all()),
                "reflections": list(reflection_result.scalars().all()),
                "strategy_events": list(strategy_event_result.scalars().all()),
            }
