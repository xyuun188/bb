"""Same-symbol loss cooldown policy for entry execution gates."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.entry_priority import MIN_ENTRY_OPPORTUNITY_SCORE
from services.trading_params import DEFAULT_TRADING_PARAMS, EntryLossCooldownParams

NormalizeSymbol = Callable[[str], str | None]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class EntryLossCooldownPolicy:
    """Block same-symbol re-entry after recent real losses unless signal quality is high."""

    normalize_symbol: NormalizeSymbol | None = None
    params: EntryLossCooldownParams = DEFAULT_TRADING_PARAMS.entry_loss_cooldown

    def reason(self, decision: DecisionOutput) -> str | None:
        """Return a cooldown block reason, or ``None`` when entry can continue."""

        if not decision.is_entry:
            return None
        raw = _safe_dict(decision.raw_response)
        opportunity = _safe_dict(raw.get("opportunity_score"))
        side_profile = _safe_dict(opportunity.get("symbol_side_profile"))
        side_reason = self._profile_reason(decision, raw, opportunity, side_profile)
        if side_reason:
            return side_reason
        return None

    def override(
        self,
        decision: DecisionOutput,
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate whether high-quality evidence may unlock a recent-loss cooldown."""

        raw = _safe_dict(decision.raw_response)
        opportunity = _safe_dict(raw.get("opportunity_score"))
        score = _safe_float(opportunity.get("score"), float("nan"))
        min_score = _safe_float(opportunity.get("min_score_required"), MIN_ENTRY_OPPORTUNITY_SCORE)
        confidence = max(
            float(decision.confidence or 0.0),
            _safe_float(opportunity.get("confidence"), 0.0),
        )
        expected_net = _safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality = _safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        reward_risk = _safe_float(opportunity.get("reward_risk_ratio"), 0.0)
        server_expected = _safe_float(opportunity.get("server_profit_expected_return_pct"), 0.0)
        server_loss_probability = _safe_float(
            opportunity.get("server_profit_loss_probability"), 1.0
        )
        tail_risk = _safe_float(opportunity.get("tail_risk_score"), 0.0)
        score_required = max(
            self.params.override_min_score,
            max(min_score, MIN_ENTRY_OPPORTUNITY_SCORE) * self.params.override_score_multiple,
        )
        aligned_sources = [
            name
            for name in ("ml_aligned", "local_profit_aligned", "timeseries_aligned")
            if bool(opportunity.get(name))
        ]
        source_support = (
            server_expected >= self.params.override_min_server_expected
            and server_loss_probability <= self.params.override_max_loss_probability
        ) or (
            len(aligned_sources) >= 2
            and server_loss_probability <= self.params.override_max_loss_probability
        )
        checks = {
            "confidence": confidence >= self.params.override_min_confidence,
            "score": math.isfinite(score) and score >= score_required,
            "expected_net": expected_net >= self.params.override_min_expected_net,
            "profit_quality": profit_quality >= self.params.override_min_profit_quality,
            "reward_risk": reward_risk >= self.params.override_min_reward_risk,
            "tail_risk": tail_risk <= self.params.override_max_tail_risk,
            "source_support": source_support,
        }
        failed = [name for name, passed in checks.items() if not passed]
        metrics = {
            "confidence": round(confidence, 6),
            "score": round(score, 6) if math.isfinite(score) else None,
            "score_required": round(score_required, 6),
            "min_score_required": round(min_score, 6),
            "expected_net_return_pct": round(expected_net, 6),
            "profit_quality_ratio": round(profit_quality, 6),
            "reward_risk_ratio": round(reward_risk, 6),
            "server_profit_expected_return_pct": round(server_expected, 6),
            "server_profit_loss_probability": round(server_loss_probability, 6),
            "tail_risk_score": round(tail_risk, 6),
            "aligned_sources": aligned_sources,
            "profile_pnl": round(_safe_float(profile.get("pnl"), 0.0), 6),
            "profile_today_pnl": round(_safe_float(profile.get("today_pnl"), 0.0), 6),
            "profile_loss": round(_safe_float(profile.get("loss"), 0.0), 6),
            "profile_losses": int(profile.get("losses") or 0),
        }
        allowed = not failed
        return {
            "allowed": allowed,
            "failed": failed,
            "checks": checks,
            "metrics": metrics,
            "summary": (
                "真实亏损冷却已由高质量信号解锁：AI 置信度、机会评分、预期净收益、"
                "盈利质量和模型同向支持均达到放行条件。"
                if allowed
                else "真实亏损冷却未解锁：当前信号还不足以覆盖近期真实亏损。"
            ),
        }

    def _profile_reason(
        self,
        decision: DecisionOutput,
        raw: dict[str, Any],
        opportunity: dict[str, Any],
        profile: dict[str, Any],
    ) -> str | None:
        if not isinstance(profile, dict) or not profile.get("cooldown"):
            return None

        override = self.override(decision, profile)
        raw["loss_cooldown_override"] = override
        opportunity["loss_cooldown_override"] = {
            "allowed": bool(override.get("allowed")),
            "summary": str(override.get("summary") or ""),
            "metrics": override.get("metrics"),
            "failed": override.get("failed"),
        }
        raw["opportunity_score"] = opportunity
        decision.raw_response = raw
        if override.get("allowed"):
            return None

        pnl = _safe_float(profile.get("pnl"), 0.0)
        today_pnl = _safe_float(profile.get("today_pnl"), 0.0)
        loss = _safe_float(profile.get("loss"), 0.0)
        today_loss = _safe_float(profile.get("today_loss"), 0.0)
        largest_loss = _safe_float(profile.get("largest_loss"), 0.0)
        cooldown_remaining_hours = _safe_float(profile.get("cooldown_remaining_hours"), 0.0)
        count = int(profile.get("count") or 0)
        losses = int(profile.get("losses") or 0)
        wins = int(profile.get("wins") or 0)
        profit_factor = _safe_float(profile.get("profit_factor"), 0.0)
        cooldown_reason = str(profile.get("cooldown_reason") or "近期真实平仓表现偏弱")
        metrics = _safe_dict(override.get("metrics"))
        side = self._entry_side(decision, opportunity)
        side_label = self._side_label(side)
        symbol_key = self._normalized_symbol(decision.symbol)
        failed_text = self._failed_text(_safe_list(override.get("failed")))

        return (
            f"该币种{side_label}方向已进入真实亏损冷却：最近 {count} 笔平仓累计 "
            f"{pnl:.2f}U，今日 {today_pnl:.2f}U，总亏损 {loss:.2f}U，"
            f"今日亏损 {today_loss:.2f}U，最大单笔亏损 {largest_loss:.2f}U，"
            f"胜/负 {wins}/{losses}，盈利因子 {profit_factor:.2f}。原因：{cooldown_reason}。"
            f"本次尝试用高质量信号解锁冷却，但 {failed_text} 未达标；"
            f"当前机会评分 {self._metric(metrics, 'score'):.2f}/要求 "
            f"{self._metric(metrics, 'score_required'):.2f}，置信度 "
            f"{self._metric(metrics, 'confidence'):.0%}，预期净收益 "
            f"{self._metric(metrics, 'expected_net_return_pct'):.2f}%，盈利质量 "
            f"{self._metric(metrics, 'profit_quality_ratio'):.2f}，服务器预期 "
            f"{self._metric(metrics, 'server_profit_expected_return_pct'):.2f}%，亏损概率 "
            f"{self._metric(metrics, 'server_profit_loss_probability', 1.0):.0%}。"
            f"为避免在 {symbol_key} 上连续同向复亏，本次禁止{side_label}新开仓；"
            f"预计还需冷却约 {cooldown_remaining_hours:.1f} 小时，之后再按最新行情重新评估。"
        )

    def _normalized_symbol(self, symbol: str) -> str:
        if self.normalize_symbol is None:
            return symbol
        return self.normalize_symbol(symbol) or symbol

    @staticmethod
    def _metric(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
        return _safe_float(metrics.get(key), default)

    @staticmethod
    def _entry_side(decision: DecisionOutput, opportunity: dict[str, Any]) -> str:
        if decision.action == Action.LONG:
            return "long"
        if decision.action == Action.SHORT:
            return "short"
        return str(opportunity.get("side") or "")

    @staticmethod
    def _side_label(side: str) -> str:
        if side == "long":
            return "做多"
        if side == "short":
            return "做空"
        return side or "当前方向"

    @staticmethod
    def _failed_text(failed: list[Any]) -> str:
        failed_labels = {
            "confidence": "AI 置信度",
            "score": "机会评分强度",
            "expected_net": "预期净收益",
            "profit_quality": "盈利质量",
            "reward_risk": "盈亏比",
            "tail_risk": "尾部风险",
            "source_support": "服务器/本地模型同向支持",
        }
        return "、".join(failed_labels.get(str(item), str(item)) for item in failed) or (
            "高质量解锁条件"
        )
