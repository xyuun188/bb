"""Entry execution priority policy.

This module decides whether an entry candidate is strong enough to skip
round-end sorting and builds the operator-facing reason text for entry sorting.
Keeping this outside TradingService makes the entry policy easier to test and
keeps the orchestrator focused on workflow coordination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput
from services.entry_direction_metrics import selected_entry_metrics
from services.trading_params import DEFAULT_TRADING_PARAMS

_PRIORITY_PARAMS = DEFAULT_TRADING_PARAMS.entry_execution_priority
MIN_ENTRY_OPPORTUNITY_SCORE = _PRIORITY_PARAMS.min_entry_opportunity_score


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class EntryExecutionPriorityPolicy:
    """Classify entry candidates for immediate execution versus normal sorting."""

    min_entry_opportunity_score: float = MIN_ENTRY_OPPORTUNITY_SCORE

    def immediate_execution_reason(self, decision: DecisionOutput) -> str | None:
        """Return a clear reason when an entry is strong enough to skip sorting."""
        if not decision.is_entry:
            return None
        raw = _safe_dict(decision.raw_response)
        opportunity = _safe_dict(raw.get("opportunity_score"))
        score = _safe_float(opportunity.get("score"), float("nan"))
        min_score = _safe_float(
            opportunity.get("min_score_required"),
            self.min_entry_opportunity_score,
        )
        expected_net = _safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality = _safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        selected_metrics = selected_entry_metrics(decision)
        if selected_metrics.has_selected_side:
            expected_net = selected_metrics.expected_net_return_pct
            profit_quality = selected_metrics.profit_quality_ratio
        confidence = max(
            float(decision.confidence or 0.0), _safe_float(opportunity.get("confidence"), 0.0)
        )
        entry_votes = int(_safe_float(opportunity.get("entry_vote_count"), 0.0))
        tail_risk_score = _safe_float(opportunity.get("tail_risk_score"), 0.0)
        high_disagreement = bool(opportunity.get("high_disagreement"))
        abnormal_volatility = bool(opportunity.get("abnormal_volatility"))
        quant_probe = _safe_dict(raw.get("quant_profit_probe"))
        quant_probe_triggered = bool(quant_probe.get("triggered"))
        strong_quant_probe = bool(quant_probe_triggered and quant_probe.get("strong_probe"))
        roster = _safe_dict(opportunity.get("portfolio_roster"))
        roster_underfilled = bool(roster.get("underfilled"))
        roster_fill_probe = bool(quant_probe_triggered and quant_probe.get("roster_fill_probe"))
        quant_loss_probability = _safe_float(
            (
                quant_probe.get("loss_probability")
                if quant_probe.get("loss_probability") is not None
                else opportunity.get("server_profit_loss_probability")
            ),
            1.0,
        )
        aligned = bool(
            opportunity.get("expert_aligned")
            or opportunity.get("ml_aligned")
            or opportunity.get("local_profit_aligned")
        )
        if not math.isfinite(score) or expected_net <= 0:
            return None
        if high_disagreement or abnormal_volatility:
            return None

        exceptional = (
            score
            >= max(
                min_score + _PRIORITY_PARAMS.exceptional_score_delta,
                _PRIORITY_PARAMS.exceptional_score_floor,
            )
            and confidence >= _PRIORITY_PARAMS.exceptional_min_confidence
            and expected_net >= _PRIORITY_PARAMS.exceptional_min_expected_net
            and profit_quality >= _PRIORITY_PARAMS.exceptional_min_profit_quality
        )
        strong_aligned = (
            score
            >= max(
                min_score + _PRIORITY_PARAMS.strong_score_delta,
                _PRIORITY_PARAMS.strong_score_floor,
            )
            and confidence >= _PRIORITY_PARAMS.strong_min_confidence
            and expected_net >= _PRIORITY_PARAMS.strong_min_expected_net
            and profit_quality >= _PRIORITY_PARAMS.strong_min_profit_quality
            and (aligned or entry_votes >= _PRIORITY_PARAMS.strong_min_entry_votes)
        )
        strong_quant = (
            strong_quant_probe
            and score
            >= max(
                min_score + _PRIORITY_PARAMS.strong_score_delta,
                _PRIORITY_PARAMS.strong_score_floor,
            )
            and confidence >= _PRIORITY_PARAMS.strong_quant_min_confidence
            and expected_net >= _PRIORITY_PARAMS.strong_quant_min_expected_net
            and profit_quality >= _PRIORITY_PARAMS.strong_quant_min_profit_quality
            and quant_loss_probability <= _PRIORITY_PARAMS.strong_quant_max_loss_probability
        )
        medium_quant = (
            quant_probe_triggered
            and not strong_quant_probe
            and score >= max(min_score, _PRIORITY_PARAMS.medium_quant_score_floor)
            and confidence >= _PRIORITY_PARAMS.medium_quant_min_confidence
            and expected_net >= _PRIORITY_PARAMS.medium_quant_min_expected_net
            and profit_quality >= _PRIORITY_PARAMS.medium_quant_min_profit_quality
            and quant_loss_probability <= _PRIORITY_PARAMS.medium_quant_max_loss_probability
            and float(decision.position_size_pct or 0.0)
            <= _PRIORITY_PARAMS.medium_quant_max_position_size
        )
        positive_expectancy = (
            score
            >= max(
                min_score + _PRIORITY_PARAMS.positive_expectancy_score_delta,
                _PRIORITY_PARAMS.positive_expectancy_score_floor,
            )
            and confidence >= _PRIORITY_PARAMS.positive_expectancy_min_confidence
            and expected_net >= _PRIORITY_PARAMS.positive_expectancy_min_expected_net
            and profit_quality >= _PRIORITY_PARAMS.positive_expectancy_min_profit_quality
            and quant_loss_probability <= _PRIORITY_PARAMS.positive_expectancy_max_loss_probability
            and tail_risk_score < _PRIORITY_PARAMS.positive_expectancy_max_tail_risk
            and (aligned or entry_votes >= 1 or quant_probe_triggered)
        )
        roster_fill_quant = (
            roster_underfilled
            and roster_fill_probe
            and score >= max(min_score, _PRIORITY_PARAMS.roster_fill_score_floor)
            and confidence >= _PRIORITY_PARAMS.roster_fill_min_confidence
            and expected_net >= _PRIORITY_PARAMS.roster_fill_min_expected_net
            and profit_quality >= _PRIORITY_PARAMS.roster_fill_min_profit_quality
            and quant_loss_probability <= _PRIORITY_PARAMS.roster_fill_max_loss_probability
            and float(decision.position_size_pct or 0.0)
            <= _PRIORITY_PARAMS.roster_fill_max_position_size
        )
        if not (
            exceptional
            or strong_aligned
            or strong_quant
            or medium_quant
            or positive_expectancy
            or roster_fill_quant
        ):
            return None
        signal_label = (
            "极强信号"
            if exceptional
            else (
                "强量化探针"
                if strong_quant
                else (
                    "组合补齐探针"
                    if roster_fill_quant
                    else (
                        "中等量化探针"
                        if medium_quant
                        else "正期望信号" if positive_expectancy else "强信号"
                    )
                )
            )
        )
        return (
            f"{signal_label}即时执行：机会评分 {score:.2f} 高于门槛 {min_score:.2f}，"
            f"AI置信度 {confidence:.0%}，预期净收益 {expected_net:.2f}%，"
            f"净盈亏比 {profit_quality:.2f}。为避免等待整轮排序错过价格，"
            "该信号通过风控后会立即进入下单前检查。"
        )

    def wait_sort_reason(
        self,
        decision: DecisionOutput,
        *,
        rank: int | None = None,
        candidate_count: int | None = None,
    ) -> str:
        raw = _safe_dict(decision.raw_response)
        opportunity = _safe_dict(raw.get("opportunity_score"))
        score = _safe_float(opportunity.get("score"), 0.0)
        min_score = _safe_float(
            opportunity.get("min_score_required"),
            self.min_entry_opportunity_score,
        )
        expected_net = _safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        selected_metrics = selected_entry_metrics(decision)
        if selected_metrics.has_selected_side:
            expected_net = selected_metrics.expected_net_return_pct
        confidence = max(
            float(decision.confidence or 0.0), _safe_float(opportunity.get("confidence"), 0.0)
        )
        prefix = (
            f"已进入开仓执行检查，历史候选排名参考 {rank}/{candidate_count}。"
            if rank is not None and candidate_count is not None
            else "已进入开仓执行检查。"
        )
        return (
            f"{prefix}这不是执行失败；开仓信号不会再等待整轮排序，会直接进入价格偏移、异常插针、保证金和 OKX 提交检查。"
            f"当前机会评分 {score:.2f}，执行门槛 {min_score:.2f}，"
            f"AI置信度 {confidence:.0%}，预期净收益 {expected_net:.2f}%。"
            "如果检查期间行情变化过大，会放弃本轮信号，下一轮重新分析。"
        )
