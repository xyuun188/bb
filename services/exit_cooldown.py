"""Exit cooldown policy.

This module owns the short-term state that prevents repeated ordinary exits for
the same symbol and side.  It is intentionally independent from TradingService
so the execution and policy layers can depend on a small, testable component.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.exit_intent import (
    COOLDOWN_BYPASS_INTENTS,
    classify_exit_intent,
    is_low_quality_release_without_hard_risk,
)
from services.trading_params import DEFAULT_TRADING_PARAMS

EXIT_SYMBOL_SIDE_COOLDOWN_SECONDS = DEFAULT_TRADING_PARAMS.exit_cooldown.ordinary_seconds
ExitCooldownKey = tuple[str, str, str]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class ExitCooldownPolicy:
    """Prevent ordinary repeated exits for the same normalized symbol and side."""

    normalize_symbol: Callable[[Any], str]
    cooldown_seconds: float = EXIT_SYMBOL_SIDE_COOLDOWN_SECONDS
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    recent_exit_groups: MutableMapping[ExitCooldownKey, datetime] = field(default_factory=dict)
    volatile_cooldown_seconds: float = DEFAULT_TRADING_PARAMS.exit_cooldown.volatile_seconds
    elevated_volatility_cooldown_seconds: float = (
        DEFAULT_TRADING_PARAMS.exit_cooldown.elevated_volatility_seconds
    )
    stable_cooldown_seconds: float = DEFAULT_TRADING_PARAMS.exit_cooldown.stable_seconds
    untradable_exit_cooldown_seconds: float = 1800.0
    failed_untradable_exit_groups: MutableMapping[ExitCooldownKey, tuple[datetime, str]] = field(
        default_factory=dict
    )

    def group_key(self, model_name: str, decision: DecisionOutput) -> ExitCooldownKey:
        side = (
            "long"
            if decision.action == Action.CLOSE_LONG
            else "short" if decision.action == Action.CLOSE_SHORT else ""
        )
        # Keep model collapsed to "all" to preserve the existing global per-symbol/side policy.
        return ("all", self.normalize_symbol(decision.symbol), side)

    def bypasses_cooldown(self, decision: DecisionOutput) -> bool:
        if not decision.is_exit:
            return True
        if classify_exit_intent(decision) in COOLDOWN_BYPASS_INTENTS:
            return True
        raw = _safe_dict(decision.raw_response)
        fast_trigger = str(raw.get("fast_risk_trigger") or "")
        close_fraction = _safe_float(
            (
                raw.get("close_fraction")
                if raw.get("close_fraction") is not None
                else decision.position_size_pct
            ),
            1.0,
        )
        close_evidence = _safe_dict(raw.get("close_evidence"))
        exit_quality = _safe_dict(raw.get("exit_quality"))
        invalidation = _safe_dict(exit_quality.get("invalidation"))
        if is_low_quality_release_without_hard_risk(raw):
            return False
        if fast_trigger in {"stop_loss", "take_profit", "near_stop_progress", "hard_adverse_move"}:
            return True
        if fast_trigger == "fast_adverse_move" and close_fraction >= 0.999:
            return True
        if bool(
            raw.get("forced_exit")
            or close_evidence.get("hard_risk")
            or close_evidence.get("forced_exit")
        ):
            return True
        if bool(invalidation.get("severe")):
            return True
        return decision.model_name == "risk_engine"

    def cooldown_seconds_for(self, decision: DecisionOutput) -> float:
        """Return ordinary cooldown seconds adjusted by current volatility."""

        snapshot = _safe_dict(decision.feature_snapshot)
        volatility = _safe_float(snapshot.get("volatility_20"), 0.0)
        returns_5 = abs(_safe_float(snapshot.get("returns_5"), 0.0))
        returns_20 = abs(_safe_float(snapshot.get("returns_20"), 0.0))
        atr_14 = _safe_float(snapshot.get("atr_14"), 0.0)
        current_price = _safe_float(
            snapshot.get("current_price", snapshot.get("close", 0.0)),
            0.0,
        )
        atr_pct = atr_14 / current_price if atr_14 > 0 and current_price > 0 else 0.0
        if volatility <= 0 and returns_5 <= 0 and returns_20 <= 0 and atr_pct <= 0:
            return self.cooldown_seconds
        if volatility >= 0.08 or returns_5 >= 0.025 or returns_20 >= 0.045 or atr_pct >= 0.04:
            return min(self.cooldown_seconds, self.volatile_cooldown_seconds)
        if volatility >= 0.04 or returns_5 >= 0.012 or returns_20 >= 0.025 or atr_pct >= 0.025:
            return min(self.cooldown_seconds, self.elevated_volatility_cooldown_seconds)
        if volatility <= 0.015 and returns_5 <= 0.003 and returns_20 <= 0.008 and atr_pct <= 0.012:
            return max(self.cooldown_seconds, self.stable_cooldown_seconds)
        return self.cooldown_seconds

    def recent_exit_cooldown_reason(
        self,
        model_name: str,
        decision: DecisionOutput,
    ) -> str | None:
        if not decision.is_exit:
            return None
        key = self.group_key(model_name, decision)
        if not all(key):
            return None
        untradable_reason = self.recent_untradable_exit_reason(key, decision)
        if untradable_reason:
            return untradable_reason
        if self.bypasses_cooldown(decision):
            return None
        last_at = self.recent_exit_groups.get(key)
        if not isinstance(last_at, datetime):
            return None
        elapsed = max((self.clock() - last_at).total_seconds(), 0.0)
        effective_cooldown_seconds = self.cooldown_seconds_for(decision)
        if elapsed >= effective_cooldown_seconds:
            return None
        remaining = max(effective_cooldown_seconds - elapsed, 0.0)
        side_label = "做多" if key[2] == "long" else "做空"
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["recent_exit_cooldown"] = {
            "applied": True,
            "symbol": key[1],
            "side": key[2],
            "elapsed_seconds": round(elapsed, 3),
            "cooldown_seconds": effective_cooldown_seconds,
            "base_cooldown_seconds": self.cooldown_seconds,
            "bypass": False,
            "reason": "同一币种同方向刚发生过平仓，普通平仓信号短时间内不连续执行。",
        }
        decision.raw_response = raw
        return (
            f"连续平仓冷却：{key[1]} {side_label} 最近 {elapsed:.0f} 秒内已经执行过一次平仓，"
            f"普通减仓/策略平仓需再等待约 {remaining:.0f} 秒。"
            "硬止损、真实止盈、严重趋势失效或强制风险平仓不受此限制。"
        )

    def recent_untradable_exit_cooldown_reason(
        self,
        model_name: str,
        decision: DecisionOutput,
    ) -> str | None:
        if not decision.is_exit:
            return None
        key = self.group_key(model_name, decision)
        if not all(key):
            return None
        return self.recent_untradable_exit_reason(key, decision)

    def recent_untradable_exit_reason(
        self,
        key: ExitCooldownKey,
        decision: DecisionOutput,
    ) -> str | None:
        item = self.failed_untradable_exit_groups.get(key)
        if not item:
            return None
        last_at, reason = item
        elapsed = max((self.clock() - last_at).total_seconds(), 0.0)
        if elapsed >= self.untradable_exit_cooldown_seconds:
            self.failed_untradable_exit_groups.pop(key, None)
            return None
        remaining = max(self.untradable_exit_cooldown_seconds - elapsed, 0.0)
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["untradable_exit_cooldown"] = {
            "applied": True,
            "symbol": key[1],
            "side": key[2],
            "elapsed_seconds": round(elapsed, 3),
            "cooldown_seconds": self.untradable_exit_cooldown_seconds,
            "remaining_seconds": round(remaining, 3),
            "last_error": reason[:500],
        }
        decision.raw_response = raw
        side_label = "做多" if key[2] == "long" else "做空"
        return (
            f"不可交易平仓冷却：{key[1]} {side_label} 上一次平仓提交被 OKX 明确拒绝为交易对不可用，"
            f"系统暂停重复提交约 {remaining:.0f} 秒，等待交易所市场列表或仓位同步恢复。"
        )

    def remember_exit(self, model_name: str, decision: DecisionOutput) -> None:
        if not decision.is_exit:
            return
        key = self.group_key(model_name, decision)
        if not all(key):
            return
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        untradable_error = raw.get("untradable_exit_execution_error")
        if isinstance(untradable_error, dict):
            reason = str(untradable_error.get("reason") or "OKX symbol is not tradable")
            self.failed_untradable_exit_groups[key] = (self.clock(), reason)
            return
        self.recent_exit_groups[key] = self.clock()
