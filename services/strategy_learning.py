"""Strategy learning, candidate generation, and runtime scheduling.

The service turns existing records into structured feedback and a bounded
strategy profile.  It does not execute generated code and it does not call an
LLM directly; profiles are controlled parameter sets that can be scored,
shadow-validated, probed, scheduled, disabled, and rolled back.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import select

from config.settings import ENSEMBLE_TRADER_NAME, settings
from core.safe_output import safe_error_text
from db.session import get_session_ctx
from models.decision import AIDecision
from models.learning import ExpertMemory, ShadowBacktest, TradeReflection
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


def _aware(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


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
        }


class StrategyLearningStateStore:
    """Small JSON state store for disabled profiles and manual rollback."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (settings.data_dir / STATE_FILE_NAME)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"disabled_profiles": {}, "manual_active_profile": ""}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
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
                state["manual_active_profile"] = "baseline_current"
        else:
            disabled_profiles.pop(profile_id, None)
        self.save(state)
        return state

    def set_manual_active_profile(self, profile_id: str | None) -> dict[str, Any]:
        state = self.load()
        state["manual_active_profile"] = profile_id or ""
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
            "reflection_count": len(reflections or []),
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
    ) -> tuple[list[dict[str, Any]], list[str]]:
        problems: list[dict[str, Any]] = []
        root_causes: list[str] = []

        def add(key: str, severity: str, label: str, evidence: dict[str, Any]) -> None:
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
        return problems, root_causes


class StrategyCandidateGenerator:
    """Generate controlled profile candidates from feedback."""

    def generate(self, feedback: StrategyFeedback) -> list[StrategyProfile]:
        profiles = [self.baseline()]
        problem_keys = {item["key"] for item in feedback.problems}
        open_pressure = feedback.open_position_pressure
        totals = feedback.totals

        if (
            "expert_fallback_overblocking" in problem_keys
            or "missed_opportunities" in problem_keys
            or totals.get("training_trade_count", 0) < DEFAULT_MIN_TRADE_TARGET
        ):
            profiles.append(
                StrategyProfile(
                    profile_id="balanced_probe",
                    version=1,
                    label="平衡探针",
                    status="candidate",
                    source="feedback_generator",
                    description="允许有限非核心专家缺失，用更小仓位恢复有效开仓样本。",
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
        if open_pressure.get("full_position_pressure") or "loss_hold_too_long" in problem_keys:
            profiles.append(
                StrategyProfile(
                    profile_id="loss_release",
                    version=1,
                    label="亏损释放",
                    status="candidate",
                    source="feedback_generator",
                    description="满仓或亏损仓占用时，提高持仓复盘和低质量亏损仓释放优先级。",
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
        if "small_wins_large_losses" in problem_keys:
            profiles.append(
                StrategyProfile(
                    profile_id="winner_hold",
                    version=1,
                    label="赢家持仓优化",
                    status="candidate",
                    source="feedback_generator",
                    description="减少优势仓位过早小盈平仓，同时继续保护回撤。",
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
                        label=f"{side} 方向恢复",
                        status="candidate",
                        source="feedback_generator",
                        description=f"{side} 侧真实表现退化，降低该方向仓位和通过门槛弹性。",
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

    @staticmethod
    def baseline() -> StrategyProfile:
        return StrategyProfile(
            profile_id="baseline_current",
            version=1,
            label="当前基线",
            status="baseline",
            source="current_system",
            description="保持现有策略，只做归因记录和低交易量惩罚评估。",
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
                matched_fixes.append("missed_opportunities")
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
                matched_fixes.append("loss_hold_too_long")
            low_trade_penalty = 0.0
        elif profile.profile_id == "winner_hold":
            if "small_wins_large_losses" in problem_keys:
                estimated_delta += max(_safe_int(totals.get("small_win_count"), 0), 1) * 0.25
                matched_fixes.append("small_wins_large_losses")
        elif profile.profile_id.endswith("_side_recovery"):
            side = profile.profile_id.removesuffix("_side_recovery")
            side_bucket = feedback.side_performance.get(side, {})
            if side_bucket.get("state") == "degraded":
                estimated_delta += abs(_safe_float(side_bucket.get("pnl"), 0.0)) * 0.12
                matched_fixes.append(f"{side}_side_degraded")

        score = net_pnl + estimated_delta - low_trade_penalty
        pass_gate = score >= net_pnl - 0.5 and (
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

        selected = by_id.get("baseline_current") or StrategyCandidateGenerator.baseline()
        reason = "默认使用当前基线。"
        problem_keys = {item["key"] for item in feedback.problems}

        if manual_profile_id and manual_profile_id in by_id:
            selected = by_id[manual_profile_id]
            reason = f"人工指定策略画像 {manual_profile_id}。"
        elif "full_position_loss_pressure" in problem_keys and "loss_release" in by_id:
            selected = by_id["loss_release"]
            reason = "检测到满仓和亏损仓占位，优先调度亏损释放画像。"
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
                    or "missed_opportunities" in problem_keys
                    or feedback.totals.get("training_trade_count", 0) < DEFAULT_MIN_TRADE_TARGET
                ) and "balanced_probe" in by_id:
                    selected = by_id["balanced_probe"]
                    reason = "开仓样本不足或专家 fallback 拦截偏多，调度平衡探针画像。"
                elif "small_wins_large_losses" in problem_keys and "winner_hold" in by_id:
                    selected = by_id["winner_hold"]
                    reason = "盈利仓小盈过多且大亏存在，调度赢家持仓优化画像。"

        selected_score = next(
            (row for row in backtest_rows if row.get("profile_id") == selected.profile_id),
            {},
        )
        if selected.profile_id != "baseline_current" and selected_score.get("pass") is False:
            selected = by_id.get("baseline_current", StrategyCandidateGenerator.baseline())
            reason = "候选画像未通过交易数量/历史评分约束，自动回滚到当前基线。"

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
        )

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
        reflections: list[Any] | None = None,
        max_open_positions: int = 20,
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
            reflections=reflections,
            max_open_positions=max_open_positions,
        )
        profiles = self.generator.generate(feedback)
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
        return {
            "training_trade_count": totals.get("training_trade_count", 0),
            "net_pnl": totals.get("net_pnl", 0.0),
            "win_rate": totals.get("win_rate", 0.0),
            "expert_integrity_blocks": decision_quality.get("expert_integrity_blocks", 0),
            "fallback_entry_rate": decision_quality.get("fallback_entry_rate", 0.0),
            "full_position_pressure": open_pressure.get("full_position_pressure", False),
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
        payload = self.engine.build(
            mode=mode,
            window_hours=hours,
            positions=rows["closed_positions"],
            open_positions=rows["open_positions"],
            orders=rows["orders"],
            decisions=rows["decisions"],
            shadows=rows["shadows"],
            memories=rows["memories"],
            reflections=rows["reflections"],
            max_open_positions=max_open_positions or int(settings.max_open_positions_per_model),
        )
        payload.update(
            {
                "mode": mode,
                "window_hours": hours,
                "sample_limit": limit,
                "state": self.state_store.load(),
            }
        )
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
        payload = self.engine.build(
            mode=mode,
            window_hours=hours,
            positions=rows["closed_positions"],
            open_positions=runtime_open_positions,
            orders=rows["orders"],
            decisions=rows["decisions"],
            shadows=rows["shadows"],
            memories=rows["memories"],
            reflections=rows["reflections"],
            max_open_positions=max_open_positions or int(settings.max_open_positions_per_model),
        )
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
        return self.state_store.set_manual_active_profile("baseline_current")

    def set_manual_active_profile(self, profile_id: str | None) -> dict[str, Any]:
        return self.state_store.set_manual_active_profile(profile_id)

    async def _load_rows(self, *, mode: str, hours: int, limit: int) -> dict[str, list[Any]]:
        selected_mode = "live" if str(mode or "").lower() == "live" else "paper"
        is_paper = selected_mode == "paper"
        capped_hours = max(1, min(int(hours or DEFAULT_LOOKBACK_HOURS), 24 * 90))
        capped_limit = max(100, min(int(limit or 3000), 20000))
        since = datetime.now(UTC) - timedelta(hours=capped_hours)
        async with get_session_ctx() as session:
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
            return {
                "closed_positions": list(closed_result.scalars().all()),
                "open_positions": list(open_result.scalars().all()),
                "orders": list(order_result.scalars().all()),
                "decisions": list(decision_result.scalars().all()),
                "shadows": list(shadow_result.scalars().all()),
                "memories": list(memory_result.scalars().all()),
                "reflections": list(reflection_result.scalars().all()),
            }
