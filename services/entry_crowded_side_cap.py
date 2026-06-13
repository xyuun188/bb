"""Crowded-side exposure hard cap for entry execution.

The portfolio repeatedly stacked one direction (for example 20+ concurrent
shorts) while every same-side entry only received an advisory score penalty
that other bonuses could cancel out. This policy adds an explicit hard cap:
once one side is both count-dominant and net-notional dominant, ordinary
same-side entries are blocked, and only genuinely strong, profit-aligned
signals may add one more controlled position.

It owns no database access and no sizing. It reads the exposure context that
EntryPositionExposurePolicy already builds (carried in opportunity_score and
strategy_mode) and returns a Chinese block reason when an entry should be
rejected, matching the existing entry gate contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.entry_priority import MIN_ENTRY_OPPORTUNITY_SCORE


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class EntryCrowdedSideCapPolicy:
    """Hard-block ordinary same-side entries once one direction is over-concentrated."""

    min_dominant_count: int = 8
    dominant_count_share: float = 0.72
    dominant_net_ratio: float = 0.55
    hard_max_side_count: int = 14
    strong_min_score_multiple: float = 1.6
    strong_min_score_floor: float = 2.6
    strong_min_expected_net_pct: float = 0.55
    strong_min_profit_quality_ratio: float = 1.4
    strong_max_loss_probability: float = 0.42
    hard_override_score_multiple: float = 2.4
    hard_override_score_floor: float = 4.2
    hard_override_min_expected_net_pct: float = 0.90
    hard_override_min_profit_quality_ratio: float = 1.8
    hard_override_max_loss_probability: float = 0.28
    hard_override_max_probe_fraction: float = 0.05
    hard_override_max_size_pct: float = 0.018

    def block_reason(self, decision: DecisionOutput) -> str | None:
        """Return a Chinese block reason when a crowded-side entry must be rejected."""

        if not decision.is_entry:
            return None
        side = self._entry_side(decision)
        if side not in {"long", "short"}:
            return None

        raw = _safe_dict(decision.raw_response)
        opportunity = _safe_dict(raw.get("opportunity_score"))
        exposure = self._resolve_exposure(raw, opportunity)
        if not exposure:
            return None

        side_count = _safe_int(exposure.get(side + "_count"), 0)
        count_share = _safe_float(exposure.get(side + "_count_share"), 0.0)
        net_ratio_abs = abs(_safe_float(exposure.get("net_ratio"), 0.0))
        dominant_side = str(exposure.get("dominant_side") or "neutral").lower()
        side_unrealized = _safe_float(exposure.get(side + "_unrealized_pnl"), 0.0)

        if side_count >= self.hard_max_side_count:
            if self._is_hard_ceiling_probe_override(decision, opportunity):
                self._annotate(decision, raw, exposure, side, mode="hard_ceiling_probe_override")
                return None
            self._annotate(decision, raw, exposure, side, mode="hard_ceiling")
            return (
                "\u5355\u8fb9\u6577\u53e3\u786c\u4e0a\u9650[crowded_side_cap]\uff1a\u5f53\u524d"
                + self._side_label(side)
                + "\u65b9\u5411\u5df2\u6709 "
                + str(side_count)
                + " \u4e2a\u6301\u4ed3\uff0c\u8fbe\u5230\u786c\u4e0a\u9650 "
                + str(self.hard_max_side_count)
                + "\uff0c\u672c\u8f6e\u62d2\u7edd\u7ee7\u7eed\u540c\u65b9\u5411\u5f00\u4ed3\u3002"
            )

        crowded = (
            dominant_side == side
            and net_ratio_abs >= self.dominant_net_ratio
            and side_count >= self.min_dominant_count
            and count_share >= self.dominant_count_share
        )
        if not crowded:
            return None

        if self._is_strong_aligned(decision, opportunity):
            self._annotate(decision, raw, exposure, side, mode="crowded_strong_override")
            return None

        self._annotate(decision, raw, exposure, side, mode="crowded_block")
        loss_hint = ""
        if side_unrealized < 0:
            loss_hint = (
                "\uff0c\u4e14\u8be5\u65b9\u5411\u6d6e\u4e8f " + f"{side_unrealized:.2f}" + "U"
            )
        return (
            "\u5355\u8fb9\u62e5\u6324\u786c\u4e0a\u9650[crowded_side_cap]\uff1a\u7ec4\u5408\u5df2\u9ad8\u5ea6\u96c6\u4e2d\u5728"
            + self._side_label(side)
            + "\u65b9\u5411\uff0c\u540c\u65b9\u5411 "
            + str(side_count)
            + " \u4ed3\uff08\u5360 "
            + f"{count_share * 100:.0f}%"
            + "\uff0c\u51c0\u6577\u53e3\u6bd4 "
            + f"{net_ratio_abs * 100:.0f}%"
            + "\uff09"
            + loss_hint
            + "\u3002\u666e\u901a\u540c\u65b9\u5411\u4fe1\u53f7\u4e0d\u518d\u5f00\u4ed3\uff0c\u53ea\u6709\u660e\u663e\u5f3a\u52bf\u4e14\u76c8\u5229\u8d28\u91cf\u8fbe\u6807\u7684\u4fe1\u53f7\u624d\u5141\u8bb8\u518d\u52a0\u4e00\u4ed3\u3002"
        )

    def _resolve_exposure(self, raw: dict[str, Any], opportunity: dict[str, Any]) -> dict[str, Any]:
        exposure = _safe_dict(opportunity.get("position_exposure"))
        if exposure:
            return exposure
        strategy_mode = _safe_dict(raw.get("strategy_mode"))
        return _safe_dict(strategy_mode.get("position_exposure"))

    def _is_strong_aligned(self, decision: DecisionOutput, opportunity: dict[str, Any]) -> bool:
        score = _safe_float(opportunity.get("score"), float("-inf"))
        min_score = _safe_float(opportunity.get("min_score_required"), MIN_ENTRY_OPPORTUNITY_SCORE)
        expected_net = _safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality = _safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        confidence = max(
            float(decision.confidence or 0.0),
            _safe_float(opportunity.get("confidence"), 0.0),
        )
        loss_probability = max(1.0 - confidence, 0.0)
        aligned = bool(
            opportunity.get("ml_aligned")
            or opportunity.get("local_profit_aligned")
            or opportunity.get("expert_aligned")
        )
        strong_score = score >= max(
            min_score * self.strong_min_score_multiple, self.strong_min_score_floor
        )
        return bool(
            aligned
            and strong_score
            and expected_net >= self.strong_min_expected_net_pct
            and profit_quality >= self.strong_min_profit_quality_ratio
            and loss_probability <= self.strong_max_loss_probability
            and not opportunity.get("high_disagreement")
        )

    def _is_hard_ceiling_probe_override(
        self,
        decision: DecisionOutput,
        opportunity: dict[str, Any],
    ) -> bool:
        score = _safe_float(opportunity.get("score"), float("-inf"))
        min_score = _safe_float(opportunity.get("min_score_required"), MIN_ENTRY_OPPORTUNITY_SCORE)
        expected_net = _safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality = _safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        confidence = max(
            float(decision.confidence or 0.0),
            _safe_float(opportunity.get("confidence"), 0.0),
        )
        probe_fraction = _safe_float(opportunity.get("probe_fraction"), 0.0)
        max_probe_size_pct = _safe_float(opportunity.get("max_probe_size_pct"), 0.0)
        if probe_fraction <= 0.0:
            probe_fraction = _safe_float(_safe_dict(decision.raw_response).get("probe_fraction"), 0.0)
        if max_probe_size_pct <= 0.0:
            max_probe_size_pct = _safe_float(
                _safe_dict(decision.raw_response).get("max_probe_size_pct"),
                0.0,
            )
        size_pct = float(decision.position_size_pct or 0.0)
        loss_probability = max(1.0 - confidence, 0.0)
        aligned = bool(
            opportunity.get("ml_aligned")
            and opportunity.get("local_profit_aligned")
            and (opportunity.get("expert_aligned") or opportunity.get("expert_consensus"))
        )
        probe_limited = bool(
            0.0 < probe_fraction <= self.hard_override_max_probe_fraction
            and 0.0 < max_probe_size_pct <= self.hard_override_max_size_pct
            and size_pct <= self.hard_override_max_size_pct
        )
        strong_score = score >= max(
            min_score * self.hard_override_score_multiple,
            self.hard_override_score_floor,
        )
        return bool(
            aligned
            and probe_limited
            and strong_score
            and expected_net >= self.hard_override_min_expected_net_pct
            and profit_quality >= self.hard_override_min_profit_quality_ratio
            and loss_probability <= self.hard_override_max_loss_probability
            and not opportunity.get("high_disagreement")
        )

    def _annotate(
        self,
        decision: DecisionOutput,
        raw: dict[str, Any],
        exposure: dict[str, Any],
        side: str,
        *,
        mode: str,
    ) -> None:
        raw["crowded_side_cap"] = {
            "mode": mode,
            "side": side,
            "side_count": _safe_int(exposure.get(side + "_count"), 0),
            "count_share": round(_safe_float(exposure.get(side + "_count_share"), 0.0), 6),
            "net_ratio": round(_safe_float(exposure.get("net_ratio"), 0.0), 6),
            "dominant_side": str(exposure.get("dominant_side") or "neutral"),
            "hard_max_side_count": self.hard_max_side_count,
            "min_dominant_count": self.min_dominant_count,
            "dominant_count_share": self.dominant_count_share,
            "dominant_net_ratio": self.dominant_net_ratio,
            "hard_override_max_probe_fraction": self.hard_override_max_probe_fraction,
            "hard_override_max_size_pct": self.hard_override_max_size_pct,
        }
        decision.raw_response = raw

    @staticmethod
    def _entry_side(decision: DecisionOutput) -> str:
        if decision.action == Action.LONG:
            return "long"
        if decision.action == Action.SHORT:
            return "short"
        return ""

    @staticmethod
    def _side_label(side: str) -> str:
        if side == "long":
            return "\u505a\u591a"
        if side == "short":
            return "\u505a\u7a7a"
        return side or "\u5f53\u524d"
