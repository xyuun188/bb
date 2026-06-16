"""Server-profit-model probe candidate policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.entry_signal_extraction import (
    expected_return_pct as signal_expected_return_pct,
    first_tool_payload,
    payload_side,
    signal_available,
)
from services.entry_probe_market_quality import EntryProbeMarketQualityPolicy
from services.trading_params import DEFAULT_TRADING_PARAMS, EntryQuantProfitProbeParams

ScoreCandidate = Callable[[DecisionOutput, dict[str, Any]], float]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _profit_signal(local_ai_tools_context: dict[str, Any] | None) -> dict[str, Any]:
    tools = local_ai_tools_context if isinstance(local_ai_tools_context, dict) else {}
    if not tools:
        return {}
    return first_tool_payload(
        {"local_ai_tools": tools},
        "profit_prediction",
        "profit_model",
        "server_profit",
        "server_profit_model",
        "profit",
    )


@dataclass(frozen=True, slots=True)
class EntryQuantProfitProbePolicy:
    """Create a small candidate when AI holds but the server profit model has edge."""

    market_quality: EntryProbeMarketQualityPolicy
    score_candidate: ScoreCandidate
    params: EntryQuantProfitProbeParams = DEFAULT_TRADING_PARAMS.entry_quant_profit_probe

    def create(
        self,
        original: DecisionOutput,
        feature_vector: Any,
        strategy: dict[str, Any] | None,
        ml_signal_context: dict[str, Any] | None,
        local_ai_tools_context: dict[str, Any] | None,
        direction_competition_context: dict[str, Any] | None,
    ) -> DecisionOutput | None:
        if not original.is_hold:
            return None
        tools = local_ai_tools_context if isinstance(local_ai_tools_context, dict) else {}
        profit = _profit_signal(local_ai_tools_context)
        if not signal_available(profit):
            return None
        side = payload_side(profit)
        if side not in {"long", "short"}:
            return None

        side_expected = signal_expected_return_pct(profit, side)
        opposite = "short" if side == "long" else "long"
        opposite_expected = signal_expected_return_pct(profit, opposite)
        edge = side_expected - opposite_expected
        loss_probability = _safe_float(profit.get(f"{side}_loss_probability"), 0.50)
        strategy_context = _safe_dict(strategy)
        roster = _safe_dict(strategy_context.get("portfolio_roster"))
        roster_underfilled = bool(roster.get("underfilled"))
        if not self._passes_primary_thresholds(
            side_expected, edge, loss_probability, roster_underfilled
        ):
            return None

        exposure = _safe_dict(strategy_context.get("position_exposure"))
        dominant_side = str(exposure.get("dominant_side") or "")
        if self._same_side_loss_is_crowded(exposure, dominant_side, side):
            return None

        strong_probe = self._is_strong_probe(side_expected, edge, loss_probability)
        market_quality_block = self.market_quality.block_reason(feature_vector, side)
        if market_quality_block and not strong_probe:
            self._record_block(
                original,
                reason=market_quality_block,
                side=side,
                side_expected=side_expected,
                edge=edge,
                loss_probability=loss_probability,
            )
            return None

        concentrated_loss_rebalance = self._has_concentrated_loss_rebalance(exposure, dominant_side)
        roster_fill_probe = bool(roster_underfilled and not strong_probe)
        candidate = self._candidate(
            original,
            feature_vector,
            tools,
            ml_signal_context,
            direction_competition_context,
            side=side,
            side_expected=side_expected,
            opposite_expected=opposite_expected,
            edge=edge,
            loss_probability=loss_probability,
            dominant_side=dominant_side,
            concentrated_loss_rebalance=concentrated_loss_rebalance,
            strong_probe=strong_probe,
            roster_fill_probe=roster_fill_probe,
            roster=roster,
        )
        self.score_candidate(candidate, strategy_context)
        if not self._post_score_is_valid(candidate):
            raw = original.raw_response if isinstance(original.raw_response, dict) else {}
            opportunity = _safe_dict(_safe_dict(candidate.raw_response).get("opportunity_score"))
            raw["quant_profit_probe_blocked"] = {
                "blocked": True,
                "reason": (
                    "服务端盈利模型弱正收益不足以覆盖本地 ML、时序、手续费滑点和尾部风险；"
                    "综合机会评分后预期净收益或盈利质量不足，因此不把 AI 观望强行转成开仓。"
                ),
                "side": side,
                "server_expected_return_pct": round(side_expected, 6),
                "expected_net_return_pct": round(
                    _safe_float(opportunity.get("expected_net_return_pct"), 0.0), 6
                ),
                "profit_quality_ratio": round(
                    _safe_float(opportunity.get("profit_quality_ratio"), 0.0), 6
                ),
                "tail_risk_score": round(_safe_float(opportunity.get("tail_risk_score"), 1.0), 6),
            }
            original.raw_response = raw
            return None
        return candidate

    def _passes_primary_thresholds(
        self,
        side_expected: float,
        edge: float,
        loss_probability: float,
        roster_underfilled: bool,
    ) -> bool:
        min_expected = (
            self.params.roster_fill_min_expected_pct
            if roster_underfilled
            else self.params.min_expected_pct
        )
        min_edge = (
            self.params.roster_fill_min_edge_pct if roster_underfilled else self.params.min_edge_pct
        )
        max_loss_probability = (
            self.params.roster_fill_max_loss_probability
            if roster_underfilled
            else self.params.default_max_loss_probability
        )
        if not roster_underfilled:
            min_expected = max(min_expected, self.params.min_expected_pct)
            min_edge = max(min_edge, self.params.min_edge_pct)
        return bool(
            side_expected >= min_expected
            and edge >= min_edge
            and loss_probability < max_loss_probability
        )

    def _same_side_loss_is_crowded(
        self, exposure: dict[str, Any], dominant_side: str, side: str
    ) -> bool:
        if dominant_side not in {"long", "short"} or side != dominant_side:
            return False
        count_share = _safe_float(exposure.get(f"{side}_count_share"), 0.0)
        side_unrealized = _safe_float(exposure.get(f"{side}_unrealized_pnl"), 0.0)
        return bool(count_share >= 0.80 and side_unrealized <= 0)

    def _has_concentrated_loss_rebalance(
        self, exposure: dict[str, Any], dominant_side: str
    ) -> bool:
        short_loss = (
            dominant_side == "short"
            and _safe_float(exposure.get("short_count_share"), 0.0) >= 0.80
            and _safe_float(exposure.get("short_unrealized_pnl"), 0.0)
            <= -self.params.min_concentrated_loss_usdt
        )
        long_loss = (
            dominant_side == "long"
            and _safe_float(exposure.get("long_count_share"), 0.0) >= 0.80
            and _safe_float(exposure.get("long_unrealized_pnl"), 0.0)
            <= -self.params.min_concentrated_loss_usdt
        )
        return bool(short_loss or long_loss)

    def _is_strong_probe(self, side_expected: float, edge: float, loss_probability: float) -> bool:
        return bool(
            side_expected >= max(self.params.min_expected_pct * 2.0, 0.45)
            and edge >= max(self.params.min_edge_pct * 2.0, 0.50)
            and loss_probability < 0.50
        )

    def _record_block(
        self,
        original: DecisionOutput,
        *,
        reason: str,
        side: str,
        side_expected: float,
        edge: float,
        loss_probability: float,
    ) -> None:
        raw = original.raw_response if isinstance(original.raw_response, dict) else {}
        raw["quant_profit_probe_blocked"] = {
            "blocked": True,
            "reason": reason,
            "side": side,
            "expected_return_pct": round(side_expected, 6),
            "edge_pct": round(edge, 6),
            "loss_probability": round(loss_probability, 6),
        }
        original.raw_response = raw

    def _candidate(
        self,
        original: DecisionOutput,
        feature_vector: Any,
        tools: dict[str, Any],
        ml_signal_context: dict[str, Any] | None,
        direction_competition_context: dict[str, Any] | None,
        *,
        side: str,
        side_expected: float,
        opposite_expected: float,
        edge: float,
        loss_probability: float,
        dominant_side: str,
        concentrated_loss_rebalance: bool,
        strong_probe: bool,
        roster_fill_probe: bool,
        roster: dict[str, Any],
    ) -> DecisionOutput:
        confidence = min(
            max(0.60 + min(side_expected, 1.2) * 0.11 + min(edge, 2.0) * 0.05, 0.60),
            0.80,
        )
        stop_loss_pct = 0.012
        min_reward_risk = 3.60 if strong_probe else 2.80
        take_profit_cap = 0.085 if strong_probe else 0.065
        take_profit_pct = max(
            stop_loss_pct * min_reward_risk,
            min(take_profit_cap, stop_loss_pct * min_reward_risk + side_expected / 100.0 * 0.70),
        )
        probe_size = 0.060 if strong_probe else (0.020 if roster_fill_probe else 0.025)
        probe_leverage = 5.0 if strong_probe else 3.0
        raw_response = original.raw_response if isinstance(original.raw_response, dict) else {}
        raw_response = dict(raw_response)
        raw_response.update(
            {
                "analysis_type": "market",
                "ml_signal": ml_signal_context or {},
                "local_ai_tools": tools,
                "direction_competition": direction_competition_context or {},
                "quant_profit_probe": {
                    "triggered": True,
                    "source": "server_profit_model",
                    "side": side,
                    "expected_return_pct": round(side_expected, 6),
                    "opposite_expected_return_pct": round(opposite_expected, 6),
                    "edge_pct": round(edge, 6),
                    "loss_probability": round(loss_probability, 6),
                    "dominant_side": dominant_side,
                    "concentrated_loss_rebalance": concentrated_loss_rebalance,
                    "strong_probe": strong_probe,
                    "roster_fill_probe": roster_fill_probe,
                    "portfolio_roster": roster,
                    "position_size_pct": round(probe_size, 6),
                    "suggested_leverage": round(probe_leverage, 6),
                    "stop_loss_pct": round(stop_loss_pct, 6),
                    "take_profit_pct": round(take_profit_pct, 6),
                    "reward_risk_ratio": round(take_profit_pct / max(stop_loss_pct, 1e-12), 6),
                    "reason": (
                        "当前组合低于目标持仓数，AI 观望但服务端盈利模型给出正期望；"
                        "生成补仓小仓候选并继续走完整风控。"
                        if roster_fill_probe
                        else "AI 观望，但服务端盈利模型给出正期望；生成小仓候选并继续走完整风控。"
                    ),
                },
            }
        )
        probe_label = "补仓" if roster_fill_probe else "正期望"
        return DecisionOutput(
            model_name=original.model_name,
            symbol=original.symbol,
            action=Action.LONG if side == "long" else Action.SHORT,
            confidence=confidence,
            reasoning=(
                f"服务端盈利模型触发{probe_label}小仓候选：{side} 调整后预期收益 "
                f"{side_expected:.2f}%，相对另一方向优势 {edge:.2f}%，"
                f"亏损概率 {loss_probability:.1%}。"
            ),
            position_size_pct=probe_size,
            suggested_leverage=probe_leverage,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            raw_response=raw_response,
            feature_snapshot=(
                feature_vector.to_dict()
                if hasattr(feature_vector, "to_dict")
                else (original.feature_snapshot or {})
            ),
        )

    @staticmethod
    def _post_score_is_valid(candidate: DecisionOutput) -> bool:
        candidate_raw = _safe_dict(candidate.raw_response)
        opportunity = _safe_dict(candidate_raw.get("opportunity_score"))
        expected_net = _safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality = _safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        tail_risk = _safe_float(opportunity.get("tail_risk_score"), 1.0)
        return bool(expected_net > 0 and profit_quality > 0 and tail_risk < 0.95)
