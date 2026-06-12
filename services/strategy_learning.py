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

from config.settings import ENSEMBLE_TRADER_NAME, settings
from core.model_runtime import apply_non_thinking_request_controls, completion_token_limit
from core.safe_output import safe_error_text
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
from services.manual_close_marker import is_manual_close_order, position_has_manual_close_order

logger = structlog.get_logger(__name__)

UNTRUSTED_EXPERT_STATUSES = {
    "batch_fallback",
    "partial_batch_fallback",
    "circuit_breaker_fallback",
    "fast_prefilter",
    "failed",
    "invalid",
    "timeout",
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
DEFAULT_MIN_TRADE_TARGET = 8
DEFAULT_LOOKBACK_HOURS = 168
STATE_FILE_NAME = "strategy_learning_state.json"
PROFILE_SNAPSHOT_MIN_INTERVAL_SECONDS = 600
LLM_CANDIDATE_CACHE_SECONDS = 6 * 60 * 60
LLM_CANDIDATE_MAX_COUNT = 3
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
    "pullback_lock_enabled",
    "side_overrides",
    "side_weights",
}
BOUNDED_FLOAT_PARAM_RANGES = {
    "global_min_score_delta": (-0.25, 0.35),
    "position_size_multiplier": (0.10, 1.25),
    "probe_fraction": (0.0, 0.10),
    "max_probe_size_pct": (0.0, 0.03),
    "position_review_priority_boost": (0.70, 1.80),
    "profit_lock_min_usdt_multiplier": (0.80, 1.80),
}
ALLOWED_EXPERT_INTEGRITY_MODES = {
    "strict_all_required",
    "balanced_probe_allow_one_non_core_missing",
    "core_experts_required_probe_only",
}
ALLOWED_AGGRESSIVENESS = {"low", "normal", "high"}
ALLOWED_WINNER_HOLD = {"normal", "high"}


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
    if isinstance(value, datetime):
        return _aware(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _aware(parsed)


def _hours_between(start: Any, end: Any) -> float:
    started = _aware(start)
    ended = _aware(end)
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
    return _aware(getattr(row, "created_at", None))


def _closed_at(row: Any) -> datetime | None:
    return _aware(getattr(row, "closed_at", None))


def _position_pnl(row: Any) -> float:
    return _safe_float(getattr(row, "realized_pnl", None), 0.0)


def _open_position_pnl(row: Any) -> float:
    if isinstance(row, dict):
        return _safe_float(row.get("unrealized_pnl"), 0.0)
    return _safe_float(getattr(row, "unrealized_pnl", None), 0.0)


def _row_side(row: Any) -> str:
    if isinstance(row, dict):
        return _position_side(row.get("side"))
    return _position_side(getattr(row, "side", None))


def _row_symbol(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("symbol") or "")
    return str(getattr(row, "symbol", "") or "")


def _row_model(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("model_name") or ENSEMBLE_TRADER_NAME)
    return str(getattr(row, "model_name", ENSEMBLE_TRADER_NAME) or ENSEMBLE_TRADER_NAME)


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
        return {
            "id": self.profile_id,
            "version": self.version,
            "label": self.label,
            "status": self.status,
            "source": self.source,
            "description": self.description,
            "params": self.params,
            "promotion": self.promotion,
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
    reflection_feedback: dict[str, Any]
    event_feedback: dict[str, Any]
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
            "reflection_feedback": self.reflection_feedback,
            "event_feedback": self.event_feedback,
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
        return disabled if isinstance(disabled, dict) else {}

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
            disabled_profiles[profile_id] = {
                "reason": reason or "manual_disable",
                "updated_at": datetime.now(UTC).isoformat(),
            }
            if state.get("manual_active_profile") == profile_id:
                state["manual_active_profile"] = ""
        else:
            disabled_profiles.pop(profile_id, None)
        self.save(state)
        return state

    def set_manual_active_profile(self, profile_id: str | None) -> dict[str, Any]:
        state = self.load()
        normalized_profile_id = "" if profile_id in {None, "", "baseline_current"} else str(profile_id)
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
        training_positions: list[Any] = []
        for position in positions:
            if position_has_manual_close_order(position, manual_orders):
                manual_position_ids.add(_safe_int(getattr(position, "id", None), 0))
                continue
            training_positions.append(position)

        side_performance = self._side_performance(training_positions)
        open_pressure = self._open_position_pressure(open_positions, max_open_positions)
        decision_quality = self._decision_quality(decisions)
        shadow_feedback = self._shadow_feedback(shadows)
        expert_memory = self._expert_memory(memories)
        event_feedback = self._event_feedback(strategy_events or [])
        reflection_feedback = self._reflection_feedback(
            reflections or [], excluded_position_ids=manual_position_ids
        )
        manual_intervention = {
            "manual_close_orders": len(manual_orders),
            "manual_closed_positions": len(manual_position_ids),
            "excluded_from_training": len(manual_position_ids),
            "policy": "manual closes are attribution and intervention signals, not model training samples",
        }

        trade_count = len(training_positions)
        net_pnl = round(sum(_position_pnl(row) for row in training_positions), 6)
        win_count = sum(1 for row in training_positions if _position_pnl(row) > 0)
        loss_count = sum(1 for row in training_positions if _position_pnl(row) < 0)
        small_win_count = sum(1 for row in training_positions if 0 < _position_pnl(row) <= 1.0)
        large_loss_count = sum(1 for row in training_positions if _position_pnl(row) <= -3.0)
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
        problems, root_causes = self._problems(
            side_performance=side_performance,
            open_pressure=open_pressure,
            decision_quality=decision_quality,
            shadow_feedback=shadow_feedback,
            trade_count=trade_count,
            net_pnl=net_pnl,
            small_win_count=small_win_count,
            large_loss_count=large_loss_count,
            avg_loss_hold_hours=avg_loss_hold_hours,
            event_feedback=event_feedback,
            reflection_feedback=reflection_feedback,
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
            "avg_hold_hours": round(avg_hold_hours, 4),
            "avg_loss_hold_hours": round(avg_loss_hold_hours, 4),
            "trade_count_target": DEFAULT_MIN_TRADE_TARGET,
            "low_trade_count_penalty": trade_count < DEFAULT_MIN_TRADE_TARGET,
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
            reflection_feedback=reflection_feedback,
            event_feedback=event_feedback,
            problems=problems,
            root_causes=root_causes,
            training_policy={
                "manual_close_excluded": True,
                "low_trade_count_is_penalized": True,
                "arbitrary_code_generation_allowed": False,
                "candidate_profiles_only": True,
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
        count = len(open_positions or [])
        losing = [row for row in open_positions or [] if _open_position_pnl(row) < 0]
        winners = [row for row in open_positions or [] if _open_position_pnl(row) > 0]
        side_counts = {"long": 0, "short": 0, "unknown": 0}
        side_pnl = {"long": 0.0, "short": 0.0, "unknown": 0.0}
        worst: list[dict[str, Any]] = []
        for row in open_positions or []:
            side = _row_side(row)
            side_counts[side] = side_counts.get(side, 0) + 1
            side_pnl[side] = side_pnl.get(side, 0.0) + _open_position_pnl(row)
            worst.append(
                {
                    "symbol": _row_symbol(row),
                    "side": side,
                    "model_name": _row_model(row),
                    "unrealized_pnl": round(_open_position_pnl(row), 6),
                }
            )
        full_pressure = count >= max_open or count >= max(1, int(max_open * 0.85))
        return {
            "open_count": count,
            "max_open_positions": max_open,
            "usage_ratio": round(count / max_open, 6),
            "full_position_pressure": bool(full_pressure),
            "losing_open_count": len(losing),
            "winner_open_count": len(winners),
            "open_unrealized_pnl": round(sum(_open_position_pnl(row) for row in open_positions), 6),
            "losing_unrealized_pnl": round(sum(_open_position_pnl(row) for row in losing), 6),
            "side_counts": side_counts,
            "side_unrealized_pnl": {key: round(value, 6) for key, value in side_pnl.items()},
            "release_candidates": sorted(worst, key=lambda item: item["unrealized_pnl"])[:8],
        }

    def _decision_quality(self, decisions: list[Any]) -> dict[str, Any]:
        market_scans = 0
        entry_signals = 0
        executed_entries = 0
        expert_integrity_blocks = 0
        fallback_entry_decisions = 0
        zero_second_entry_decisions = 0
        status_counts: dict[str, int] = {}
        missing_expert_counts: dict[str, int] = {}
        for row in decisions or []:
            raw = _safe_dict(getattr(row, "raw_llm_response", None))
            action = _action(getattr(row, "action", None))
            analysis_type = str(getattr(row, "analysis_type", "") or raw.get("analysis_type") or "")
            if analysis_type.lower() == "position" or action in {"close_long", "close_short"}:
                continue
            market_scans += 1
            if action in {"long", "short"}:
                entry_signals += 1
                if bool(getattr(row, "was_executed", False)):
                    executed_entries += 1
                fallback, zero_second, missing = self._expert_integrity_flags(raw)
                fallback_entry_decisions += 1 if fallback else 0
                zero_second_entry_decisions += 1 if zero_second else 0
                for name in missing:
                    missing_expert_counts[name] = missing_expert_counts.get(name, 0) + 1
            reason = str(getattr(row, "execution_reason", "") or "")
            if "expert_integrity" in reason:
                expert_integrity_blocks += 1
            for status in self._timing_statuses(raw):
                status_counts[status] = status_counts.get(status, 0) + 1
        signal_rate = entry_signals / market_scans if market_scans else 0.0
        execution_rate = executed_entries / entry_signals if entry_signals else 0.0
        fallback_rate = fallback_entry_decisions / entry_signals if entry_signals else 0.0
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
            fallback_flag = bool(item.get("batch_expert_fallback") or item.get("fallback"))
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
        for key in ("seconds", "duration_seconds", "elapsed_seconds", "latency_seconds"):
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

    def _shadow_feedback(self, shadows: list[Any]) -> dict[str, Any]:
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
        return {
            "completed_count": len(completed),
            "missed_opportunity_count": len(missed),
            "bad_signal_count": len(bad_signals),
            "good_signal_count": len(good_signals),
            "missed_opportunity_rate": round(len(missed) / len(completed), 6) if completed else 0.0,
            "bad_signal_rate": round(len(bad_signals) / len(completed), 6) if completed else 0.0,
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
        total_hold_minutes = 0.0
        net_pnl = 0.0
        fee_estimate = 0.0
        small_win_count = 0
        large_loss_count = 0
        recent: list[dict[str, Any]] = []

        for row in training_rows:
            outcome = str(getattr(row, "outcome", "flat") or "flat").lower()
            pnl = _safe_float(getattr(row, "realized_pnl", None), 0.0)
            fee = _safe_float(getattr(row, "fee_estimate", None), 0.0)
            hold_minutes = _safe_float(getattr(row, "hold_minutes", None), 0.0)
            mistake = str(getattr(row, "mistake_summary", "") or "").strip()
            improvement = str(getattr(row, "improvement_summary", "") or "").strip()
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            net_pnl += pnl
            fee_estimate += fee
            total_hold_minutes += hold_minutes
            if 0 < pnl <= 1.0:
                small_win_count += 1
            if pnl <= -3.0:
                large_loss_count += 1
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
            "small_win_count": small_win_count,
            "large_loss_count": large_loss_count,
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
        block_reasons: dict[str, int] = {}
        manual_close_count = 0
        max_position_blocks = 0
        fallback_blocks = 0
        execution_errors = 0
        covered = 0
        missing_profile = 0
        recent: list[dict[str, Any]] = []
        for row in events or []:
            event_type = str(getattr(row, "event_type", "") or "unknown")
            status = str(getattr(row, "event_status", "") or "recorded")
            profile_id = str(getattr(row, "profile_id", "") or "")
            reason = str(getattr(row, "reason", "") or "")
            attribution = _safe_dict(getattr(row, "attribution", None))
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
            if profile_id:
                covered += 1
                profile_counts[profile_id] = profile_counts.get(profile_id, 0) + 1
            else:
                missing_profile += 1
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
            if status in {"error", "failed", "rejected"} or event_type == "execution_error":
                execution_errors += 1
            if status in {"blocked", "skipped", "rejected", "failed"}:
                key = str(attribution.get("blocker") or reason or event_type)[:120]
                if key:
                    block_reasons[key] = block_reasons.get(key, 0) + 1
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
                        "action": getattr(row, "action", None),
                        "profile_id": profile_id,
                        "order_id": getattr(row, "order_id", None),
                        "position_id": getattr(row, "position_id", None),
                        "reason": reason,
                        "exclude_from_training": bool(getattr(row, "exclude_from_training", False)),
                    }
                )
        total = len(events or [])
        coverage = covered / total if total else 0.0
        top_blocks = sorted(block_reasons.items(), key=lambda item: item[1], reverse=True)[:8]
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
            "attribution_coverage": round(coverage, 6),
            "missing_profile_events": missing_profile,
            "manual_close_events": manual_close_count,
            "max_position_blocks": max_position_blocks,
            "fallback_blocks": fallback_blocks,
            "execution_errors": execution_errors,
            "top_block_reasons": [{"reason": key, "count": count} for key, count in top_blocks],
            "recent_events": recent,
        }

    def _problems(
        self,
        *,
        side_performance: dict[str, dict[str, Any]],
        open_pressure: dict[str, Any],
        decision_quality: dict[str, Any],
        shadow_feedback: dict[str, Any],
        trade_count: int,
        net_pnl: float,
        small_win_count: int,
        large_loss_count: int,
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

        if trade_count < DEFAULT_MIN_TRADE_TARGET:
            add(
                "low_trade_count",
                "medium",
                "样本和开仓数量不足，不能让系统学成不开仓最安全",
                {"trade_count": trade_count, "target": DEFAULT_MIN_TRADE_TARGET},
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
        if shadow_feedback.get("missed_opportunity_rate", 0.0) >= 0.08:
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
        if reflection_feedback.get("small_win_count", 0) >= max(
            2, reflection_feedback.get("large_loss_count", 0)
        ) and reflection_feedback.get("large_loss_count", 0):
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
        if small_win_count >= max(2, large_loss_count) and large_loss_count:
            add(
                "small_wins_large_losses",
                "high",
                "盈利仓过早小盈平仓，而亏损单损失更大",
                {"small_win_count": small_win_count, "large_loss_count": large_loss_count},
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
            event_feedback.get("total_events")
            and event_feedback.get("attribution_coverage", 0.0) < 0.85
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
        profiles = [self.baseline()]
        profiles.extend(self.rule_based_candidates(feedback))
        return self._dedupe(profiles)

    def rule_based_candidates(self, feedback: StrategyFeedback) -> list[StrategyProfile]:
        profiles: list[StrategyProfile] = []
        problem_keys = {item["key"] for item in feedback.problems}
        open_pressure = feedback.open_position_pressure
        totals = feedback.totals

        if (
            "expert_fallback_overblocking" in problem_keys
            or "missed_opportunities" in problem_keys
            or "trade_reflection_mistakes" in problem_keys
            or totals.get("training_trade_count", 0) < DEFAULT_MIN_TRADE_TARGET
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
                        "position_size_multiplier": 0.62,
                        "probe_fraction": 0.08,
                        "min_trade_count_target": DEFAULT_MIN_TRADE_TARGET,
                        "expert_integrity_mode": "balanced_probe_allow_one_non_core_missing",
                        "max_probe_size_pct": 0.018,
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
            or "loss_hold_too_long" in problem_keys
            or "reflection_loss_hold_too_long" in problem_keys
            or "reflection_negative_pnl" in problem_keys
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
                        "profit_lock_min_usdt_multiplier": 1.25,
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
            profiles.append(
                StrategyProfile(
                    profile_id=profile_id,
                    version=max(1, _safe_int(item.get("version"), 1)),
                    label=str(item.get("label") or f"LLM\u5019\u9009{index}")[:80],
                    status="candidate",
                    source="llm_structured_candidate",
                    description=str(
                        item.get("description")
                        or "\u5927\u6a21\u578b\u6839\u636e\u7ed3\u6784\u5316\u53cd\u9988\u751f\u6210\u7684\u53d7\u63a7\u53c2\u6570\u5019\u9009\u3002"
                    )[:500],
                    params={
                        **params,
                        "min_trade_count_target": max(
                            DEFAULT_MIN_TRADE_TARGET,
                            _safe_int(
                                params.get("min_trade_count_target"),
                                _safe_int(
                                    feedback.totals.get("trade_count_target"),
                                    DEFAULT_MIN_TRADE_TARGET,
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
                clean[key] = max(DEFAULT_MIN_TRADE_TARGET, min(_safe_int(value), 80))
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
            clean.setdefault("position_size_multiplier", 0.62)
            clean.setdefault("max_probe_size_pct", 0.018)
        return clean

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
    def baseline() -> StrategyProfile:
        return StrategyProfile(
            profile_id="baseline_current",
            version=1,
            label="\u5f53\u524d\u57fa\u7ebf",
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
                "min_trade_count_target": DEFAULT_MIN_TRADE_TARGET,
            },
        )


class StrategyBacktester:
    """Score profiles with historical feedback and trade-count constraints."""

    def score(self, profile: StrategyProfile, feedback: StrategyFeedback) -> dict[str, Any]:
        totals = feedback.totals
        net_pnl = _safe_float(totals.get("net_pnl"), 0.0)
        trade_count = _safe_int(totals.get("training_trade_count"), 0)
        target = _safe_int(
            profile.params.get("min_trade_count_target"),
            DEFAULT_MIN_TRADE_TARGET,
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
        missed_opportunity_reduction = 0.0
        loss_release_speed = 0.0
        winner_avg_profit = avg_winner
        problem_keys = {item["key"] for item in feedback.problems}
        estimated_delta = 0.0
        matched_fixes: list[str] = []

        if profile.profile_id == "balanced_probe":
            if "expert_fallback_overblocking" in problem_keys:
                estimated_delta += (
                    max(feedback.decision_quality.get("expert_integrity_blocks", 0), 1) * 0.35
                )
                matched_fixes.append("expert_fallback_overblocking")
            if "missed_opportunities" in problem_keys:
                estimated_delta += (
                    feedback.shadow_feedback.get("missed_opportunity_count", 0) * 0.18
                )
                missed_opportunity_reduction = (
                    _safe_float(feedback.shadow_feedback.get("missed_opportunity_count"), 0.0)
                    * 0.18
                )
                matched_fixes.append("missed_opportunities")
            if "trade_reflection_mistakes" in problem_keys:
                estimated_delta += min(_safe_int(reflection.get("mistake_count"), 0), 8) * 0.12
                matched_fixes.append("trade_reflection_mistakes")
            if trade_count < target:
                estimated_delta += (target - trade_count) * 0.42
                matched_fixes.append("low_trade_count")
                low_trade_penalty = max(low_trade_penalty - trade_gap * 0.95, 0.0)
        elif profile.profile_id == "loss_release":
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
                estimated_delta += max(_safe_int(totals.get("small_win_count"), 0), 1) * 0.25
                winner_avg_profit = avg_winner * 1.15 if avg_winner else 0.0
                matched_fixes.append("small_wins_large_losses")
            if "reflection_small_wins_large_losses" in problem_keys:
                estimated_delta += max(_safe_int(reflection.get("small_win_count"), 0), 1) * 0.22
                winner_avg_profit = max(
                    winner_avg_profit,
                    avg_winner * 1.12 if avg_winner else 0.0,
                )
                matched_fixes.append("reflection_small_wins_large_losses")
        elif profile.profile_id.endswith("_side_recovery"):
            side = profile.profile_id.removesuffix("_side_recovery")
            side_bucket = feedback.side_performance.get(side, {})
            if side_bucket.get("state") == "degraded":
                estimated_delta += abs(_safe_float(side_bucket.get("pnl"), 0.0)) * 0.12
                matched_fixes.append(f"{side}_side_degraded")

        score = net_pnl + estimated_delta - low_trade_penalty
        score -= fee_estimate * 0.35
        score -= max_drawdown * 0.10
        score -= max(consecutive_losses - 3, 0) * 0.7
        score -= max(occupancy - 0.85, 0.0) * 2.0
        pass_gate = score >= fee_adjusted_pnl - 0.75 and (
            trade_count >= max(2, int(target * 0.35)) or profile.profile_id == "balanced_probe"
        )
        if profile.profile_id in {"loss_release", "winner_hold"} and matched_fixes:
            pass_gate = True
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
            "matched_fixes": matched_fixes,
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
            "pass": bool(pass_gate),
            "notes": (
                "low trade count is penalized; a profile cannot win by refusing trades"
                if low_trade_penalty
                else "trade count constraint satisfied"
            ),
        }


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
        disabled_ids = sorted(disabled.keys())
        available = [profile for profile in profiles if profile.profile_id not in disabled]
        by_id = {profile.profile_id: profile for profile in available}
        state = self.state_store.load()
        manual_profile_id = str(state.get("manual_active_profile") or "")
        if manual_profile_id == "baseline_current":
            manual_profile_id = ""
        manual_lock_active = bool(manual_profile_id and manual_profile_id in by_id)

        selected = by_id.get("baseline_current") or StrategyCandidateGenerator.baseline()
        reason = "默认使用当前基线。"
        problem_keys = {item["key"] for item in feedback.problems}

        if manual_lock_active:
            selected = by_id[manual_profile_id]
            reason = f"人工指定策略画像 {manual_profile_id}。"
        elif (
            "full_position_loss_pressure" in problem_keys or "max_position_blocks" in problem_keys
        ) and "loss_release" in by_id:
            selected = by_id["loss_release"]
            reason = "检测到满仓和亏损仓占位，优先调度亏损释放画像。"
        elif (
            "reflection_loss_hold_too_long" in problem_keys
            or "reflection_negative_pnl" in problem_keys
        ) and "loss_release" in by_id:
            selected = by_id["loss_release"]
            reason = "策略复盘显示费后亏损或亏损仓拖延过久，调度亏损释放画像。"
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
                    or feedback.totals.get("training_trade_count", 0) < DEFAULT_MIN_TRADE_TARGET
                ) and "balanced_probe" in by_id:
                    selected = by_id["balanced_probe"]
                    reason = "开仓样本不足或专家 fallback 拦截偏多，调度平衡探针画像。"
                elif (
                    "small_wins_large_losses" in problem_keys
                    or "reflection_small_wins_large_losses" in problem_keys
                ) and "winner_hold" in by_id:
                    selected = by_id["winner_hold"]
                    reason = "盈利仓小盈过多且大亏存在，调度赢家持仓优化画像。"

        selected_score = next(
            (row for row in backtest_rows if row.get("profile_id") == selected.profile_id),
            {},
        )
        if selected.profile_id != "baseline_current" and selected_score.get("pass") is False:
            selected = by_id.get("baseline_current", StrategyCandidateGenerator.baseline())
            reason = "候选画像未通过交易数量/历史评分约束，自动回滚到当前基线。"

        reason = self._readable_schedule_reason(
            selected_profile_id=selected.profile_id,
            manual_lock_active=manual_lock_active,
            manual_profile_id=manual_profile_id,
            problem_keys=problem_keys,
            score_failed=selected_score.get("pass") is False,
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
            shadow_validation=self._shadow_validation(profiles, feedback),
            probe=self._probe(selected, feedback),
            disabled_profiles=disabled_ids,
            scheduler_mode="manual" if manual_lock_active else "auto",
            manual_profile_id=manual_profile_id if manual_lock_active else "",
        )

    @staticmethod
    def _readable_schedule_reason(
        *,
        selected_profile_id: str,
        manual_lock_active: bool,
        manual_profile_id: str,
        problem_keys: set[str],
        score_failed: bool,
    ) -> str:
        if score_failed and selected_profile_id == "baseline_current":
            return "候选策略未通过交易数量或历史评分约束，自动回退到当前基线。"
        if manual_lock_active:
            return f"人工锁定策略画像 {manual_profile_id}，自动调度暂不覆盖。"
        if selected_profile_id == "loss_release":
            if {"full_position_loss_pressure", "max_position_blocks"} & problem_keys:
                return "检测到满仓压力或亏损仓占位，自动调度到亏损释放画像。"
            return "策略复盘显示费后亏损或亏损仓拖延过久，自动调度到亏损释放画像。"
        if selected_profile_id.endswith("_side_recovery"):
            side = selected_profile_id.replace("_side_recovery", "")
            side_label = "多单" if side == "long" else "空单" if side == "short" else side
            return f"{side_label}方向近期表现退化，自动调度到方向恢复画像。"
        if selected_profile_id == "balanced_probe":
            return "开仓样本不足、专家 fallback 拦截或影子复盘错过机会偏多，自动调度到平衡探针画像。"
        if selected_profile_id == "winner_hold":
            return "盈利仓小盈过多且大亏存在，自动调度到赢家持仓优化画像。"
        return "自动调度未发现需要切换的高优先级问题，使用当前基线。"

    def _runtime(self, profile: StrategyProfile, feedback: StrategyFeedback) -> dict[str, Any]:
        params = profile.params
        return {
            "profile_id": profile.profile_id,
            "profile_version": profile.version,
            "global_min_score_delta": _safe_float(params.get("global_min_score_delta"), 0.0),
            "position_size_multiplier": _safe_float(params.get("position_size_multiplier"), 1.0),
            "probe_fraction": _safe_float(params.get("probe_fraction"), 0.0),
            "expert_integrity_mode": str(
                params.get("expert_integrity_mode") or "strict_all_required"
            ),
            "side_overrides": _safe_dict(params.get("side_overrides")),
            "side_weights": _safe_dict(params.get("side_weights")),
            "loss_exit_aggressiveness": params.get("loss_exit_aggressiveness", "normal"),
            "winner_hold_extension": params.get("winner_hold_extension", "normal"),
            "full_position_release": bool(params.get("full_position_release")),
            "position_review_priority_boost": _safe_float(
                params.get("position_review_priority_boost"), 1.0
            ),
            "training_trade_count": feedback.totals.get("training_trade_count", 0),
            "low_trade_count_penalty": bool(feedback.totals.get("low_trade_count_penalty")),
        }

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
        rows = []
        missed = _safe_int(feedback.shadow_feedback.get("missed_opportunity_count"), 0)
        bad = _safe_int(feedback.shadow_feedback.get("bad_signal_count"), 0)
        for profile in profiles:
            score = 0.0
            if profile.profile_id == "balanced_probe":
                score += missed * 0.4 - bad * 0.15
            elif profile.profile_id.endswith("_side_recovery"):
                score += bad * 0.25
            elif profile.profile_id == "winner_hold":
                score += _safe_int(feedback.totals.get("small_win_count"), 0) * 0.2
            rows.append(
                {
                    "profile_id": profile.profile_id,
                    "shadow_score": round(score, 6),
                    "eligible": profile.profile_id == "baseline_current" or score >= 0.0,
                    "missed_opportunities_used": missed,
                    "bad_signals_used": bad,
                }
            )
        return {"rows": rows, "completed_count": feedback.shadow_feedback.get("completed_count", 0)}

    @staticmethod
    def _probe(profile: StrategyProfile, feedback: StrategyFeedback) -> dict[str, Any]:
        probe_fraction = _safe_float(profile.params.get("probe_fraction"), 0.0)
        return {
            "profile_id": profile.profile_id,
            "enabled": profile.profile_id != "baseline_current" and probe_fraction > 0,
            "probe_fraction": probe_fraction,
            "small_position_first": True,
            "promotion_requirements": {
                "min_training_trades": DEFAULT_MIN_TRADE_TARGET,
                "net_pnl_must_improve": True,
                "max_consecutive_losses": 3,
                "fallback_rate_must_not_increase": True,
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
        result["strategy_learning"] = {
            "active_profile": active_profile,
            "runtime": runtime,
            "reason": schedule.get("reason", ""),
            "rollback": schedule.get("rollback", {}),
            "feedback_summary": self._compact_feedback(_safe_dict(payload.get("feedback"))),
            "candidate_count": len(_safe_list(schedule.get("candidates"))),
            "scheduler_mode": schedule.get("scheduler_mode", "auto"),
            "manual_profile_id": schedule.get("manual_profile_id", ""),
            "low_trade_count_penalized": True,
            "manual_close_excluded_from_training": True,
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
        limit: int = 3000,
        max_open_positions: int | None = None,
    ) -> dict[str, Any]:
        rows = await self._load_rows(mode=mode, hours=hours, limit=limit)
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
                "sample_limit": limit,
                "state": self.state_store.load(),
            }
        )
        payload["runtime_guard"] = self._runtime_guard(payload, mutate=False)
        return payload

    async def apply_to_strategy_context(
        self,
        *,
        mode: str,
        strategy_context: dict[str, Any],
        open_positions: list[dict[str, Any]] | None,
        hours: int = DEFAULT_LOOKBACK_HOURS,
        limit: int = 3000,
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
        if guard.get("should_rollback"):
            payload = self._build_payload_from_feedback(
                feedback,
                extra_profiles=self._cached_llm_profiles(feedback),
            )
            guard = self._runtime_guard(payload, mutate=False)
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
            max_open_positions=max_open_positions or int(settings.max_open_positions_per_model),
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
        return {
            "enabled": bool(getattr(settings, "strategy_learning_llm_candidates_enabled", True)),
            "signature": self._feedback_signature(feedback),
            "cached_signature": entry.get("signature"),
            "cached_at": entry.get("generated_at"),
            "candidate_count": len(candidates),
            "last_error": entry.get("last_error", ""),
            "source": entry.get("source", "llm_structured_candidate" if candidates else "none"),
        }

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
        api_base, api_key, model = self._llm_candidate_config()
        if not api_base or not api_key or not model:
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
        fresh = bool(generated_at and (datetime.now(UTC) - generated_at).total_seconds() < interval)
        return not (fresh and entry.get("signature") == signature)

    def _llm_candidate_config(self) -> tuple[str, str, str]:
        configs = [cfg for cfg in settings.get_fixed_ai_models(False) if isinstance(cfg, dict)]
        cfg = next((item for item in configs if item.get("name") == "decision_maker"), {})
        if not cfg:
            cfg = next((item for item in configs if item.get("api_key")), {})
        api_base = str(cfg.get("api_base") or settings.ai_api_base or "").rstrip("/")
        api_key = str(cfg.get("api_key") or settings.ai_api_key or "")
        model = str(cfg.get("model") or settings.ai_model or "")
        return api_base, api_key, model

    async def _generate_llm_profiles(
        self,
        *,
        mode: str,
        feedback: StrategyFeedback,
    ) -> list[StrategyProfile]:
        api_base, api_key, model = self._llm_candidate_config()
        signature = self._feedback_signature(feedback)
        try:
            candidates = await self._call_llm_candidate_model(
                api_base=api_base,
                api_key=api_key,
                model=model,
                feedback=feedback,
            )
        except Exception as exc:
            self._store_llm_candidate_cache(
                mode=mode,
                signature=signature,
                candidates=[],
                error=safe_error_text(exc, limit=200),
            )
            logger.debug(
                "strategy learning llm candidate generation failed", error=safe_error_text(exc)
            )
            return []
        profiles = self.engine.generator.from_structured_candidates(candidates, feedback)
        self._store_llm_candidate_cache(
            mode=mode,
            signature=signature,
            candidates=[profile.to_dict() for profile in profiles],
            error="",
        )
        return profiles

    async def _call_llm_candidate_model(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        feedback: StrategyFeedback,
    ) -> list[dict[str, Any]]:
        prompt = {
            "task": "generate_bounded_strategy_profile_candidates",
            "language": "zh-CN",
            "rules": [
                "只返回 JSON 对象，不要 Markdown。",
                "不能生成 Python 代码、SQL、shell 或任意可执行逻辑。",
                "只能设置白名单参数，候选必须先小仓探针，不允许全量接管。",
                "低交易量策略必须被惩罚，不能用不开仓来提高胜率。",
            ],
            "allowed_params": sorted(ALLOWED_CANDIDATE_PARAM_KEYS),
            "feedback": {
                "totals": feedback.totals,
                "side_performance": feedback.side_performance,
                "open_position_pressure": feedback.open_position_pressure,
                "decision_quality": feedback.decision_quality,
                "shadow_feedback": feedback.shadow_feedback,
                "event_feedback": {
                    key: feedback.event_feedback.get(key)
                    for key in (
                        "total_events",
                        "attribution_coverage",
                        "manual_close_events",
                        "max_position_blocks",
                        "fallback_blocks",
                        "execution_errors",
                    )
                },
                "reflection_feedback": {
                    key: feedback.reflection_feedback.get(key)
                    for key in (
                        "training_count",
                        "excluded_manual_count",
                        "outcome_counts",
                        "fee_adjusted_pnl",
                        "avg_loss_hold_minutes",
                        "small_win_count",
                        "large_loss_count",
                        "top_mistakes",
                        "top_improvements",
                    )
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
        max_tokens = completion_token_limit(
            "proxy",
            _safe_int(getattr(settings, "strategy_learning_llm_candidate_max_tokens", 600), 600),
            floor=160,
        )
        body = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是量化交易策略参数编译器，只能输出受控 JSON 参数，不允许输出代码。",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
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
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{api_base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            )
        if not response.is_success:
            raise RuntimeError(
                f"strategy candidate request failed with HTTP {response.status_code}"
            )
        payload = response.json()
        content = self._extract_llm_content(payload)
        parsed = self._parse_json_object(content)
        return _safe_list(parsed.get("candidates"))[:LLM_CANDIDATE_MAX_COUNT]

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
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise RuntimeError("strategy candidate response was not a JSON object")
        return parsed

    def _store_llm_candidate_cache(
        self,
        *,
        mode: str,
        signature: str,
        candidates: list[dict[str, Any]],
        error: str,
    ) -> None:
        state = self.state_store.load()
        state["llm_candidate_cache"] = {
            "mode": "live" if str(mode).lower() == "live" else "paper",
            "signature": signature,
            "generated_at": datetime.now(UTC).isoformat(),
            "candidates": _json_safe(candidates),
            "last_error": error,
            "source": "llm_structured_candidate" if candidates else "none",
        }
        self.state_store.save(state)

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
            reason=str(reason or "")[:2000],
            decision_id=decision_id,
            order_id=order_id,
            position_id=position_id,
            profile_id=profile_id or None,
            profile_version=profile_version or None,
            scheduler_reason=str(context.get("scheduler_reason") or learning.get("reason") or "")[
                :2000
            ],
            strategy_snapshot=_json_safe(
                {
                    "active_profile": active_profile,
                    "runtime": runtime,
                    "feedback_summary": learning.get("feedback_summary"),
                    "rollback": learning.get("rollback"),
                }
            ),
            market_state=_json_safe(market_state or context.get("market_regime") or {}),
            side_weights=_json_safe(
                runtime.get("side_weights") or _safe_dict(context.get("side_quality"))
            ),
            expert_integrity=_json_safe(self._expert_integrity_event_payload(raw, context)),
            attribution=_json_safe(attribution or {}),
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
        reasons: list[str] = []
        should_rollback = False
        if profile_id != "baseline_current":
            if _safe_float(totals.get("net_pnl"), 0.0) <= -8.0:
                should_rollback = True
                reasons.append("recent_net_pnl_guard")
            if _safe_int(totals.get("training_trade_count"), 0) < max(
                2, DEFAULT_MIN_TRADE_TARGET // 3
            ):
                reasons.append("insufficient_trade_samples")
            if _safe_float(decision_quality.get("fallback_entry_rate"), 0.0) >= 0.55:
                should_rollback = True
                reasons.append("fallback_dependency_guard")
            if _safe_int(event_feedback.get("execution_errors"), 0) >= 3:
                should_rollback = True
                reasons.append("execution_error_guard")
        if should_rollback and mutate:
            self.state_store.set_profile_disabled(
                profile_id,
                disabled=True,
                reason="auto_runtime_guard:" + ",".join(reasons),
            )
            self.state_store.set_manual_active_profile(None)
        return {
            "profile_id": profile_id,
            "should_rollback": bool(should_rollback),
            "mutated": bool(should_rollback and mutate),
            "reasons": reasons,
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
        capped_hours = max(1, min(int(hours or DEFAULT_LOOKBACK_HOURS), 24 * 90))
        capped_limit = max(100, min(int(limit or 3000), 20000))
        since = datetime.now(UTC) - timedelta(hours=capped_hours)
        async with get_read_session_ctx() as session:
            closed_result = await session.execute(
                select(Position)
                .where(
                    Position.model_name == ENSEMBLE_TRADER_NAME,
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
                    Position.model_name == ENSEMBLE_TRADER_NAME,
                    Position.execution_mode == selected_mode,
                    Position.is_open.is_(True),
                )
                .order_by(Position.created_at.desc())
                .limit(capped_limit)
            )
            order_result = await session.execute(
                select(Order)
                .where(
                    Order.model_name == ENSEMBLE_TRADER_NAME,
                    Order.execution_mode == selected_mode,
                    Order.created_at >= since - timedelta(hours=2),
                )
                .order_by(Order.created_at.desc())
                .limit(capped_limit * 3)
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
                .where(ExpertMemory.created_at >= since - timedelta(days=30))
                .order_by(
                    ExpertMemory.updated_at.desc().nullslast(), ExpertMemory.created_at.desc()
                )
                .limit(2000)
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
                .limit(capped_limit * 3)
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
