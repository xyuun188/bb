"""Entry opportunity scoring policy.

This module owns the expected-net-return score used to rank entry candidates.
It is intentionally dependency-injected so TradingService wires data and
orchestration, while the scoring algorithm stays testable here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from config.settings import settings
from services.entry_evidence import build_entry_evidence_score
from services.entry_priority import MIN_ENTRY_OPPORTUNITY_SCORE
from services.entry_signal_extraction import (
    expected_return_pct as signal_expected_return_pct,
    first_tool_payload,
    payload_side,
    signal_available,
)
from services.entry_stop_loss_budget import ENTRY_MAX_STOP_LOSS_NORMAL_USDT
from services.entry_symbol_winner import EntrySymbolWinnerDecayPolicy
from services.trading_params import ESTIMATED_TAKER_FEE_PCT

ML_EXPECTED_RETURN_SCORE_CAP_PCT = 3.0
ENTRY_NET_WEIGHT_AI = 0.25
ENTRY_NET_WEIGHT_LOCAL_ML = 0.40
ENTRY_NET_WEIGHT_SERVER_PROFIT = 0.08
ENTRY_NET_WEIGHT_TIMESERIES = 0.22
ENTRY_SMALL_WIN_BIG_LOSS_PENALTY_CAP = 0.90
ENTRY_REALIZED_EDGE_BONUS_CAP = 0.85
ENTRY_REALIZED_EDGE_PENALTY_CAP = 1.15
ENTRY_MIN_NET_PROFIT_QUALITY_RATIO = 1.50
ENTRY_WEAK_HISTORY_MIN_PROFIT_QUALITY_RATIO = 2.00
ENTRY_STRONG_ALIGNED_MIN_PROFIT_QUALITY_RATIO = 0.85
ENTRY_WEAK_HISTORY_STRONG_ALIGNED_MIN_PROFIT_QUALITY_RATIO = 1.05
ENTRY_WEAK_HISTORY_MIN_SCORE = 3.20
DYNAMIC_ENTRY_SCORE_ML_ALIGNED_STRONG = 0.75
DYNAMIC_ENTRY_SCORE_ML_ALIGNED = 0.85
DYNAMIC_ENTRY_SCORE_EXPERT_ALIGNED = 0.90
QUANT_PROFIT_PROBE_MIN_EXPECTED_PCT = 0.18
QUANT_PROFIT_PROBE_MIN_SCORE = 0.35
ABNORMAL_WICK_TAIL_RISK_MAX_PCT = 60.0

NormalizeSymbol = Callable[[Any], str | None]
ContributionAdjuster = Callable[[list[str], dict[str, Any]], dict[str, Any]]
DecisionAnnotator = Callable[[DecisionOutput], dict[str, Any]]


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


def _tool_signal(raw: dict[str, Any], *keys: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return first_tool_payload(raw, *keys)


@dataclass(slots=True)
class EntryOpportunityScoringPolicy:
    """Score entry candidates using explicit dependencies instead of TradingService state."""

    normalize_symbol: NormalizeSymbol
    model_contribution_score_adjustment: ContributionAdjuster
    annotate_decision_source: DecisionAnnotator
    entry_symbol_winner_decay: EntrySymbolWinnerDecayPolicy

    @staticmethod
    def _safe_dict(value: Any) -> dict[str, Any]:
        return _safe_dict(value)

    @staticmethod
    def _safe_list(value: Any) -> list[Any]:
        return _safe_list(value)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        return _safe_float(value, default)

    def _memory_habit_adjustment(
        self,
        raw: dict[str, Any],
        *,
        side: str,
        expected_net_return_pct: float,
        loss_probability: float,
        tail_risk_score: float,
        profit_quality_ratio: float,
        base_min_score_required: float,
    ) -> dict[str, Any]:
        feedback = self._safe_dict(raw.get("memory_feedback"))
        habit = self._safe_dict(feedback.get("decision_habit"))
        by_side = self._safe_dict(habit.get("by_side"))
        side_habit = self._safe_dict(by_side.get(side))
        if not side_habit:
            legacy_side = self._safe_dict(self._safe_dict(feedback.get("by_side")).get(side))
            action_bias = str(legacy_side.get("action_bias") or "")
            if action_bias == "prefer_small_probe_when_current_ev_positive":
                side_habit = {
                    "stance": "probe_when_ev_ok",
                    "proactive_level": 0.35,
                    "probe_budget_pct": legacy_side.get("max_probe_size_pct", 0.015),
                    "min_expected_net_pct": 0.12,
                    "max_loss_probability": 0.58,
                    "max_tail_risk": 0.98,
                }
            elif action_bias == "require_stronger_confirmation":
                side_habit = {
                    "stance": "strict_confirm",
                    "proactive_level": 0.0,
                    "probe_budget_pct": 0.0,
                    "min_expected_net_pct": 0.35,
                    "max_loss_probability": 0.42,
                    "max_tail_risk": 0.82,
                }
        stance = str(side_habit.get("stance") or "neutral")
        if stance == "probe_when_ev_ok":
            min_expected = self._safe_float(side_habit.get("min_expected_net_pct"), 0.12)
            max_loss_probability = self._safe_float(side_habit.get("max_loss_probability"), 0.58)
            max_tail_risk = self._safe_float(side_habit.get("max_tail_risk"), 0.98)
            quality_ok = bool(
                expected_net_return_pct >= min_expected
                and loss_probability <= max_loss_probability
                and tail_risk_score <= max_tail_risk
                and profit_quality_ratio > 0
            )
            proactive_level = min(
                max(self._safe_float(side_habit.get("proactive_level"), 0.35), 0.0),
                1.0,
            )
            if quality_ok:
                score_adjustment = min(0.30, 0.08 + proactive_level * 0.18)
                relaxed_min = max(
                    0.35,
                    base_min_score_required - min(0.22, 0.08 + proactive_level * 0.10),
                )
                return {
                    "applied": True,
                    "stance": stance,
                    "quality_ok": True,
                    "score_adjustment": round(score_adjustment, 6),
                    "min_score_required": round(relaxed_min, 6),
                    "max_size_pct": round(
                        self._safe_float(side_habit.get("probe_budget_pct"), 0.015), 6
                    ),
                    "reason": (
                        "review memory shows repeated missed opportunities; current EV and "
                        "tail risk allow a small probe"
                    ),
                }
            return {
                "applied": False,
                "stance": stance,
                "quality_ok": False,
                "score_adjustment": 0.0,
                "reason": "missed-opportunity memory exists but current quality gates failed",
            }
        if stance == "strict_confirm":
            return {
                "applied": True,
                "stance": stance,
                "quality_ok": False,
                "score_adjustment": -0.28,
                "min_score_required": round(max(base_min_score_required + 0.22, 1.05), 6),
                "max_size_pct": 0.0,
                "reason": "review memory says realized or shadow losses dominate this side",
            }
        return {
            "applied": False,
            "stance": stance,
            "quality_ok": False,
            "score_adjustment": 0.0,
            "reason": "no memory habit adjustment",
        }

    def _side_quality_adjustment(
        self,
        strategy: dict[str, Any],
        *,
        side: str,
        strong_aligned_profit_evidence: bool,
    ) -> dict[str, Any]:
        side_quality = self._safe_dict(strategy.get("side_quality"))
        item = self._safe_dict(side_quality.get(side))
        state = str(item.get("state") or "neutral")
        if state == "degraded":
            raw_score = self._safe_float(item.get("score_adjustment"), -0.25)
            raw_delta = self._safe_float(item.get("min_score_delta"), 0.22)
            raw_size = self._safe_float(item.get("size_multiplier"), 0.65)
            score_adjustment = (
                max(raw_score, -0.12) if strong_aligned_profit_evidence else raw_score
            )
            min_score_delta = min(raw_delta, 0.10) if strong_aligned_profit_evidence else raw_delta
            size_multiplier = max(raw_size, 0.82) if strong_aligned_profit_evidence else raw_size
            return {
                "applied": True,
                "state": state,
                "score_adjustment": round(score_adjustment, 6),
                "min_score_delta": round(min_score_delta, 6),
                "size_multiplier": round(size_multiplier, 6),
                "strong_current_evidence_relief": bool(strong_aligned_profit_evidence),
                "reason": item.get("reason") or "side realized performance is weak",
            }
        if state == "working":
            return {
                "applied": True,
                "state": state,
                "score_adjustment": round(self._safe_float(item.get("score_adjustment"), 0.08), 6),
                "min_score_delta": round(self._safe_float(item.get("min_score_delta"), -0.05), 6),
                "size_multiplier": round(self._safe_float(item.get("size_multiplier"), 1.05), 6),
                "strong_current_evidence_relief": False,
                "reason": item.get("reason") or "side realized performance is positive",
            }
        return {
            "applied": False,
            "state": state,
            "score_adjustment": 0.0,
            "min_score_delta": 0.0,
            "size_multiplier": 1.0,
            "strong_current_evidence_relief": False,
            "reason": "no side-quality adjustment",
        }

    def score_candidate(
        self,
        decision: DecisionOutput,
        strategy: dict[str, Any] | None = None,
    ) -> float:
        """Rank entry candidates by expected net opportunity, not just confidence."""
        if not decision.is_entry:
            return -1e9

        side = "long" if decision.action == Action.LONG else "short"
        raw = self._safe_dict(decision.raw_response)
        strategy = self._safe_dict(strategy)
        ml_signal = self._safe_dict(raw.get("ml_signal"))
        predictions = self._safe_list(ml_signal.get("predictions"))
        primary = self._safe_dict(predictions[0] if predictions else {})

        influence_enabled = bool(ml_signal.get("influence_enabled", True))
        influence_policy = self._safe_dict(ml_signal.get("influence_policy"))
        side_policy: dict[str, Any] = self._safe_dict(influence_policy.get(side))
        side_influence_enabled = influence_enabled and (
            not isinstance(side_policy, dict) or side_policy.get("enabled", True)
        )
        raw_expected_pct = self._safe_float(primary.get(f"{side}_expected_return_pct"), 0.0)
        expected_pct = max(
            min(raw_expected_pct, ML_EXPECTED_RETURN_SCORE_CAP_PCT),
            -ML_EXPECTED_RETURN_SCORE_CAP_PCT,
        )
        opposite = "short" if side == "long" else "long"
        raw_opposite_expected_pct = self._safe_float(
            primary.get(f"{opposite}_expected_return_pct"), 0.0
        )
        opposite_expected_pct = max(
            min(raw_opposite_expected_pct, ML_EXPECTED_RETURN_SCORE_CAP_PCT),
            -ML_EXPECTED_RETURN_SCORE_CAP_PCT,
        )
        edge_pct = expected_pct - opposite_expected_pct
        win_rate = self._safe_float(primary.get(f"{side}_win_rate"), 0.50)
        ml_quality = self._safe_float(primary.get("profit_quality_score"), 0.0)
        if not side_influence_enabled:
            expected_pct = 0.0
            opposite_expected_pct = 0.0
            edge_pct = 0.0
            win_rate = 0.50
            ml_quality = 0.0

        confidence = max(min(float(decision.confidence or 0.0), 1.0), 0.0)
        size = max(float(decision.position_size_pct or 0.0), 0.0)
        leverage = max(float(decision.suggested_leverage or 1.0), 1.0)
        stop_loss_pct = max(float(decision.stop_loss_pct or 0.0), 0.0)
        take_profit_pct = max(float(decision.take_profit_pct or 0.0), 0.0)
        loss_probability = max(1.0 - confidence, 0.0)
        reward_risk_ratio = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 0.0
        ai_expected_return_pct = (
            confidence * take_profit_pct - loss_probability * stop_loss_pct
        ) * 100

        fee_pct = ESTIMATED_TAKER_FEE_PCT * 2 * 100
        slippage_pct = max(float(settings.max_slippage_pct or 0.0), 0.0) * 100
        confidence_bonus = max(confidence - 0.55, 0.0) * 0.45
        rr_bonus = max(min(reward_risk_ratio - 1.0, 2.0), 0.0) * 0.16
        risk_penalty = max(0.58 - confidence, 0.0) * 0.85
        weak_rr_penalty = max(1.0 - reward_risk_ratio, 0.0) * 0.75

        exposure_penalty = 0.0
        exposure_balance_bonus = 0.0
        exposure = self._safe_dict(strategy.get("position_exposure"))
        if exposure.get("dominant_side") == side:
            net_ratio_abs = abs(self._safe_float(exposure.get("net_ratio"), 0.0))
            count_share = self._safe_float(exposure.get(f"{side}_count_share"), 0.0)
            exposure_penalty = net_ratio_abs * 0.35 + max(count_share - 0.70, 0.0) * 0.45
        elif exposure.get("dominant_side") in {"long", "short"}:
            dominant = str(exposure.get("dominant_side") or "")
            opposite_dominant = "short" if dominant == "long" else "long"
            if side == opposite_dominant:
                exposure_balance_bonus = (
                    abs(self._safe_float(exposure.get("net_ratio"), 0.0)) * 0.10
                )
        base_min_score_required = self._safe_float(
            strategy.get("min_opportunity_score", MIN_ENTRY_OPPORTUNITY_SCORE),
            MIN_ENTRY_OPPORTUNITY_SCORE,
        )
        min_score_required = base_min_score_required
        dynamic_score_reason = (
            f"分歧大、波动异常或没有盈利模型同向确认，保持 {base_min_score_required:.2f}+ 基础门槛。"
        )
        local_tools = self._safe_dict(raw.get("local_ai_tools"))
        local_profit = _tool_signal(
            raw,
            "profit_prediction",
            "profit_model",
            "server_profit",
            "server_profit_model",
            "profit",
        )
        local_best_side = payload_side(local_profit)
        local_expected = signal_expected_return_pct(local_profit, side)
        local_available = signal_available(local_profit)
        ml_aligned = (
            side_influence_enabled
            and expected_pct > 0
            and (edge_pct >= 0 or str(primary.get("best_side") or "").lower() == side)
        )
        local_aligned = local_available and local_best_side == side and local_expected > 0
        local_conflicts = local_available and (
            local_expected <= 0
            or (local_best_side in {"long", "short"} and local_best_side != side)
        )
        local_loss_probability = self._safe_float(
            local_profit.get(f"{side}_loss_probability"), 0.50
        )
        local_quality = self._safe_float(local_profit.get("profit_quality_score"), 0.0)
        ts_prediction = _tool_signal(
            raw,
            "time_series_prediction",
            "timeseries_prediction",
            "sequence_prediction",
            "timeseries",
            "time_series",
        )
        ts_best_side = payload_side(ts_prediction)
        ts_expected = signal_expected_return_pct(ts_prediction, side)
        ts_aligned = signal_available(ts_prediction) and ts_best_side == side and ts_expected > 0
        if exposure.get("dominant_side") in {"long", "short"}:
            dominant = str(exposure.get("dominant_side") or "")
            opposite_dominant = "short" if dominant == "long" else "long"
            if side == opposite_dominant and (expected_pct > 0 or local_expected > 0 or ts_aligned):
                # Prefer portfolio balance only when this side has profit evidence.
                exposure_balance_bonus += min(
                    abs(self._safe_float(exposure.get("net_ratio"), 0.0)) * 0.18, 0.22
                )
        experts = self._safe_list(raw.get("experts"))
        entry_votes = 0
        opposite_votes = 0
        hold_votes = 0
        for expert in experts:
            if not isinstance(expert, dict):
                continue
            action_value = str(expert.get("action") or "").lower()
            if action_value == side:
                entry_votes += 1
            elif action_value == opposite:
                opposite_votes += 1
            elif action_value == "hold":
                hold_votes += 1
        entry_support = self._safe_dict(raw.get("entry_signal_support"))
        if entry_support.get("side") == side:
            support_experts = entry_support.get("same_direction_experts")
            technical_support = entry_support.get("technical_support")
            if isinstance(support_experts, list):
                entry_votes = max(entry_votes, len(support_experts))
            if isinstance(technical_support, list) and len(technical_support) >= 2:
                opposite_votes = 0
        expert_aligned = entry_votes >= 2 and opposite_votes == 0
        high_disagreement = opposite_votes > 0 or (hold_votes >= 3 and entry_votes < 2)
        direction_competition = self._safe_dict(raw.get("direction_competition"))
        if not direction_competition and isinstance(strategy.get("direction_competition"), dict):
            direction_competition = self._safe_dict(strategy.get("direction_competition"))
        direction_preferred_side = str(
            direction_competition.get("preferred_side") or "neutral"
        ).lower()
        direction_gap = self._safe_float(direction_competition.get("score_gap"), 0.0)
        direction_side_score = self._safe_float(
            (self._safe_dict(direction_competition.get(side)).get("score")),
            0.0,
        )
        direction_opposite_score = self._safe_float(
            (self._safe_dict(direction_competition.get(opposite)).get("score")),
            0.0,
        )
        direction_alignment_bonus = 0.0
        direction_conflict_penalty = 0.0
        if direction_preferred_side == side and direction_gap >= 0.08:
            direction_alignment_bonus = min(direction_gap, 1.8) * 0.32
        elif direction_preferred_side == opposite and direction_gap >= 0.12:
            direction_conflict_penalty = min(direction_gap, 2.0) * 0.55
            high_disagreement = True
        elif (
            direction_preferred_side == "neutral"
            and abs(direction_side_score - direction_opposite_score) < 0.08
        ):
            direction_conflict_penalty = 0.10
        contribution_sources: list[str] = []
        if ml_aligned:
            contribution_sources.append("ml_profit_model")
        if local_aligned:
            contribution_sources.append("server_profit_model")
        if ts_aligned:
            contribution_sources.append("timeseries_model")
        if expert_aligned:
            contribution_sources.append("expert_alignment")
        if not any(
            source in contribution_sources
            for source in ("ml_profit_model", "server_profit_model", "timeseries_model")
        ):
            contribution_sources.append("ai_only_without_quant")
        contribution_perf = self._safe_dict(strategy.get("model_contribution_performance"))
        contribution_adjustment = self.model_contribution_score_adjustment(
            contribution_sources,
            contribution_perf,
        )
        portfolio_roster = self._safe_dict(strategy.get("portfolio_roster"))
        contribution_score_multiplier = self._safe_float(
            contribution_adjustment.get("score_multiplier"),
            1.0,
        )
        contribution_size_multiplier = self._safe_float(
            contribution_adjustment.get("size_multiplier"),
            1.0,
        )
        contribution_score_adjustment = self._safe_float(
            contribution_adjustment.get("score_adjustment"),
            0.0,
        )
        if contribution_size_multiplier != 1.0:
            previous_adjustment = raw.get("model_contribution_adjustment")
            size_already_applied = (
                isinstance(previous_adjustment, dict)
                and previous_adjustment.get("size_applied") is True
            )
            if not size_already_applied:
                original_size = size
                size = max(min(size * contribution_size_multiplier, 1.0), 0.0)
                decision.position_size_pct = size
                contribution_adjustment["original_position_size"] = round(original_size, 6)
                contribution_adjustment["adjusted_position_size"] = round(size, 6)
                contribution_adjustment["size_applied"] = True
        feature_snapshot = (
            decision.feature_snapshot if isinstance(decision.feature_snapshot, dict) else {}
        )
        volatility = self._safe_float(feature_snapshot.get("volatility_20"), 0.0)
        day_change = abs(self._safe_float(feature_snapshot.get("change_24h_pct"), 0.0))
        abnormal_volatility = volatility >= 0.08 or day_change >= 18.0
        if not high_disagreement and not abnormal_volatility:
            if ml_aligned and local_aligned:
                min_score_required = min(min_score_required, DYNAMIC_ENTRY_SCORE_ML_ALIGNED_STRONG)
                dynamic_score_reason = (
                    "ML 与服务器盈利模型同向且预期收益为正，允许 0.75+ 小仓开仓。"
                )
            elif ml_aligned or local_aligned:
                min_score_required = min(min_score_required, DYNAMIC_ENTRY_SCORE_ML_ALIGNED)
                dynamic_score_reason = (
                    "ML 或服务器盈利模型与 AI 方向同向且预期收益为正，允许 0.85+ 小仓开仓。"
                )
            elif expert_aligned and expected_pct > 0:
                min_score_required = min(min_score_required, DYNAMIC_ENTRY_SCORE_EXPERT_ALIGNED)
                dynamic_score_reason = "专家方向一致且预期收益为正，允许 0.90+ 开仓。"
        if contribution_score_multiplier >= 1.06:
            min_score_required = max(min(min_score_required, base_min_score_required - 0.08), 0.72)
            dynamic_score_reason = (
                f"{dynamic_score_reason} 最近真实平仓贡献为正，闭环调权后放宽 0.08。"
            )
        elif contribution_score_multiplier <= 0.94:
            min_score_required = max(min_score_required + 0.18, base_min_score_required)
            dynamic_score_reason = (
                f"{dynamic_score_reason} 最近真实平仓贡献为负，闭环调权后提高门槛并缩小仓位。"
            )
        symbol_key = self.normalize_symbol(decision.symbol) or decision.symbol
        profiles = strategy.get("symbol_side_performance") if isinstance(strategy, dict) else {}
        if not isinstance(profiles, dict):
            profiles = {}
        side_profile = (
            profiles.get(f"{symbol_key}|{side}")
            if isinstance(profiles.get(f"{symbol_key}|{side}"), dict)
            else {}
        )
        symbol_profile = (
            profiles.get(f"{symbol_key}|all")
            if isinstance(profiles.get(f"{symbol_key}|all"), dict)
            else {}
        )
        historical_adjustment = 0.0
        historical_block = False
        historical_reason = "今天还没有该币种方向的真实平仓记录。"
        for profile, weight, label in (
            (symbol_profile, 0.55, "symbol"),
            (side_profile, 1.00, "symbol-side"),
        ):
            if not isinstance(profile, dict) or int(profile.get("count") or 0) <= 0:
                continue
            pnl = self._safe_float(profile.get("pnl"), 0.0)
            avg_pnl = self._safe_float(profile.get("avg_pnl"), 0.0)
            profit_factor = self._safe_float(profile.get("profit_factor"), 0.0)
            losses = int(profile.get("losses") or 0)
            wins = int(profile.get("wins") or 0)
            if profile.get("cooldown"):
                label_cn = "symbol" if label == "symbol" else "symbol-side"
                historical_block = True
                historical_reason = (
                    f"{label_cn} recent realized PnL is weak: pnl={pnl:.2f} U, "
                    f"losses={losses}, wins={wins}, profit_factor={profit_factor:.2f}."
                )
            if pnl > 0 and profit_factor >= 1.25:
                historical_adjustment += min(pnl / 32.0, ENTRY_REALIZED_EDGE_BONUS_CAP) * weight
            if avg_pnl < 0 or profit_factor < 0.75:
                loss_count_penalty = min(losses, 10) * 0.06
                avg_loss_penalty = abs(avg_pnl) / 12.0
                historical_adjustment -= (
                    min(
                        avg_loss_penalty + loss_count_penalty,
                        ENTRY_REALIZED_EDGE_PENALTY_CAP,
                    )
                    * weight
                )
            if losses >= wins + 2 and pnl < 0:
                historical_adjustment -= 0.25 * weight

        side_losses = int(side_profile.get("losses") or 0) if isinstance(side_profile, dict) else 0
        side_wins = int(side_profile.get("wins") or 0) if isinstance(side_profile, dict) else 0
        side_avg_pnl = (
            self._safe_float(side_profile.get("avg_pnl"), 0.0)
            if isinstance(side_profile, dict)
            else 0.0
        )
        side_profit_factor = (
            self._safe_float(side_profile.get("profit_factor"), 1.0)
            if isinstance(side_profile, dict)
            else 1.0
        )
        side_largest_loss = (
            abs(self._safe_float(side_profile.get("largest_loss"), 0.0))
            if isinstance(side_profile, dict)
            else 0.0
        )
        side_profit = (
            self._safe_float(side_profile.get("profit"), 0.0)
            if isinstance(side_profile, dict)
            else 0.0
        )
        side_count = int(side_profile.get("count") or 0) if isinstance(side_profile, dict) else 0
        side_pnl = (
            self._safe_float(side_profile.get("pnl"), 0.0)
            if isinstance(side_profile, dict)
            else 0.0
        )
        side_loss = (
            self._safe_float(side_profile.get("loss"), 0.0)
            if isinstance(side_profile, dict)
            else 0.0
        )
        symbol_pnl = (
            self._safe_float(symbol_profile.get("pnl"), 0.0)
            if isinstance(symbol_profile, dict)
            else 0.0
        )
        winner_policy = getattr(self, "entry_symbol_winner_decay", None)
        if winner_policy is None:
            winner_policy = EntrySymbolWinnerDecayPolicy()
        winner_adjustment = winner_policy.evaluate(
            side=side,
            side_profile=side_profile,
            symbol_profile=symbol_profile,
            base_min_score_required=base_min_score_required,
            current_min_score_required=min_score_required,
            side_loss=side_loss,
            side_profit=side_profit,
            side_losses=side_losses,
        )
        symbol_profit_tier = winner_adjustment.tier
        symbol_tier_reason = winner_adjustment.reason
        symbol_tier_score_adjustment = winner_adjustment.score_adjustment
        min_score_required = winner_adjustment.min_score_required
        small_win_big_loss_penalty = 0.0
        if side_count >= 2 and side_largest_loss > 0:
            avg_win = side_profit / max(side_wins, 1)
            loss_to_win_ratio = side_largest_loss / max(avg_win, 0.25)
            if loss_to_win_ratio >= 3.0 or side_profit_factor < 0.80:
                small_win_big_loss_penalty = min(
                    ENTRY_SMALL_WIN_BIG_LOSS_PENALTY_CAP,
                    (loss_to_win_ratio - 2.0) * 0.12 + max(0.80 - side_profit_factor, 0.0) * 0.55,
                )
                historical_adjustment -= small_win_big_loss_penalty
        tail_history_component = 0.0
        if side_losses > 0 and (side_avg_pnl < 0 or side_profit_factor < 0.80):
            tail_history_component = min(
                0.35,
                side_losses * 0.035
                + max(abs(side_avg_pnl) / 18.0, 0.0)
                + (0.08 if side_losses >= side_wins + 2 else 0.0),
            )
        stop_risk_component = min(max(stop_loss_pct / 0.055, 0.0), 1.0) * 0.22
        loss_probability_component = min(max(local_loss_probability, 0.0), 1.0) * 0.36
        volatility_component = min(max(volatility / 0.08, 0.0), 1.0) * 0.17
        abnormal_wick_max_pct = self._safe_float(feature_snapshot.get("abnormal_wick_max_pct"), 0.0)
        abnormal_wick_count = int(
            self._safe_float(feature_snapshot.get("abnormal_wick_count_72h"), 0.0)
        )
        abnormal_wick_recent_hours = self._safe_float(
            feature_snapshot.get("abnormal_wick_recent_hours"), 9999.0
        )
        abnormal_wick_component = 0.0
        if abnormal_wick_max_pct >= ABNORMAL_WICK_TAIL_RISK_MAX_PCT and abnormal_wick_count > 0:
            recency_weight = (
                1.0
                if abnormal_wick_recent_hours <= 24.0
                else 0.70 if abnormal_wick_recent_hours <= 72.0 else 0.45
            )
            abnormal_wick_component = min(abnormal_wick_max_pct / 300.0, 0.55) * recency_weight
        disagreement_component = (0.16 if high_disagreement else 0.0) + (
            0.09 if abnormal_volatility else 0.0
        )
        tail_risk_score = min(
            max(
                loss_probability_component
                + stop_risk_component
                + volatility_component
                + abnormal_wick_component
                + tail_history_component
                + disagreement_component,
                0.0,
            ),
            1.35,
        )
        tail_risk_penalty = tail_risk_score * 0.92
        quant_conflict_penalty = 0.0
        same_side_loss_concentration = False
        if local_available and local_expected <= 0:
            quant_conflict_penalty += min(abs(local_expected) * 1.35 + 0.45, 1.75)
        if local_available and local_best_side in {"long", "short"} and local_best_side != side:
            quant_conflict_penalty += 0.55
        if local_loss_probability >= 0.64 and not local_aligned:
            quant_conflict_penalty += min((local_loss_probability - 0.60) * 1.6, 0.55)
        if isinstance(exposure, dict) and exposure.get("dominant_side") == side:
            count_share = self._safe_float(exposure.get(f"{side}_count_share"), 0.0)
            side_unrealized = self._safe_float(exposure.get(f"{side}_unrealized_pnl"), 0.0)
            if count_share >= 0.85 and side_unrealized < 0:
                same_side_loss_concentration = True
                quant_conflict_penalty += min(abs(side_unrealized) / 30.0, 0.65)
                dynamic_score_reason = (
                    f"当前组合已经高度集中在 {side}，且该方向浮亏 {side_unrealized:.2f}U；"
                    "本轮只作为风险扣分，不再直接禁止同方向开仓。"
                )

        strong_current_profit_support = (
            local_aligned
            and local_expected > 0
            and local_quality >= 0.35
            and local_loss_probability < 0.62
        )
        historical_adjustment_cap = -0.85 if strong_current_profit_support else -1.80
        if historical_adjustment < historical_adjustment_cap:
            historical_adjustment = historical_adjustment_cap
        historical_adjustment += symbol_tier_score_adjustment

        expected_net_return_pct = (
            ai_expected_return_pct * ENTRY_NET_WEIGHT_AI
            + expected_pct * ENTRY_NET_WEIGHT_LOCAL_ML
            + local_expected * ENTRY_NET_WEIGHT_SERVER_PROFIT
            + ts_expected * ENTRY_NET_WEIGHT_TIMESERIES
            - fee_pct
            - slippage_pct
        )
        expected_loss_pct = max(
            stop_loss_pct * 100 * max(1.0 - confidence, 0.0),
            max(local_loss_probability - 0.50, 0.0) * stop_loss_pct * 100 * 2.0,
            fee_pct + slippage_pct,
        )
        success_probability = min(
            max(
                win_rate * 0.45
                + confidence * 0.30
                + (1.0 - min(max(local_loss_probability, 0.0), 1.0)) * 0.20
                + (0.05 if local_aligned or ts_aligned else 0.0),
                0.0,
            ),
            1.0,
        )
        profit_quality_ratio = expected_net_return_pct / max(
            expected_loss_pct + fee_pct + slippage_pct, 0.05
        )
        downside_asymmetry_penalty = 0.0
        if expected_net_return_pct <= 0:
            downside_asymmetry_penalty = min(
                abs(expected_net_return_pct) * 0.75 + expected_loss_pct * 0.22, 1.25
            )
        elif expected_loss_pct > expected_net_return_pct * 1.8:
            downside_asymmetry_penalty = min(
                (expected_loss_pct - expected_net_return_pct * 1.8) * 0.32,
                0.75,
            )
        strong_aligned_profit_evidence = (
            expected_net_return_pct > 0
            and not high_disagreement
            and not abnormal_volatility
            and tail_risk_score < 0.88
            and (
                strong_current_profit_support
                or (ml_aligned and expected_pct >= 0.05 and edge_pct >= 0)
                or (ts_aligned and ts_expected > 0)
            )
        )
        min_profit_quality_ratio_required = (
            ENTRY_WEAK_HISTORY_MIN_PROFIT_QUALITY_RATIO
            if historical_block and not strong_aligned_profit_evidence
            else ENTRY_MIN_NET_PROFIT_QUALITY_RATIO
        )
        if strong_aligned_profit_evidence:
            if historical_block:
                min_profit_quality_ratio_required = min(
                    min_profit_quality_ratio_required,
                    ENTRY_WEAK_HISTORY_STRONG_ALIGNED_MIN_PROFIT_QUALITY_RATIO,
                )
            else:
                min_profit_quality_ratio_required = min(
                    min_profit_quality_ratio_required,
                    ENTRY_STRONG_ALIGNED_MIN_PROFIT_QUALITY_RATIO,
                )
        quant_probe = self._safe_dict(raw.get("quant_profit_probe"))
        if (
            quant_probe.get("triggered")
            and local_aligned
            and local_expected >= QUANT_PROFIT_PROBE_MIN_EXPECTED_PCT
            and local_loss_probability < 0.58
            and (direction_preferred_side in {side, "neutral", ""} or local_expected >= 0.45)
        ):
            min_score_required = min(min_score_required, QUANT_PROFIT_PROBE_MIN_SCORE)
            min_profit_quality_ratio_required = min(
                min_profit_quality_ratio_required,
                0.0,
            )
            dynamic_score_reason = (
                "AI 原始观望，但服务器盈利模型给出正期望且亏损概率可控；"
                "按小仓盈利探针门槛执行完整风控，净盈亏比只记录不硬拦截。"
            )
        capital_efficiency_score = (
            expected_net_return_pct * max(leverage, 1.0) / max(size * 100.0, 1.0)
        )
        memory_habit_adjustment = self._memory_habit_adjustment(
            raw,
            side=side,
            expected_net_return_pct=expected_net_return_pct,
            loss_probability=local_loss_probability,
            tail_risk_score=tail_risk_score,
            profit_quality_ratio=profit_quality_ratio,
            base_min_score_required=base_min_score_required,
        )
        habit_score_adjustment = self._safe_float(
            memory_habit_adjustment.get("score_adjustment"), 0.0
        )
        habit_min_score = memory_habit_adjustment.get("min_score_required")
        if habit_min_score is not None:
            min_score_required = self._safe_float(habit_min_score, min_score_required)
        habit_size_cap = self._safe_float(memory_habit_adjustment.get("max_size_pct"), 0.0)
        if (
            habit_size_cap > 0
            and memory_habit_adjustment.get("stance") == "probe_when_ev_ok"
            and size > habit_size_cap
        ):
            original_size = size
            size = habit_size_cap
            decision.position_size_pct = size
            memory_habit_adjustment["original_position_size"] = round(original_size, 6)
            memory_habit_adjustment["adjusted_position_size"] = round(size, 6)
        side_quality_adjustment = self._side_quality_adjustment(
            strategy,
            side=side,
            strong_aligned_profit_evidence=strong_aligned_profit_evidence,
        )
        side_quality_score_adjustment = self._safe_float(
            side_quality_adjustment.get("score_adjustment"), 0.0
        )
        side_quality_min_delta = self._safe_float(
            side_quality_adjustment.get("min_score_delta"), 0.0
        )
        if side_quality_min_delta:
            min_score_required = max(0.35, min_score_required + side_quality_min_delta)
        side_quality_size_multiplier = self._safe_float(
            side_quality_adjustment.get("size_multiplier"), 1.0
        )
        if side_quality_size_multiplier < 0.999 and size > 0:
            original_size = size
            size = max(min(size * side_quality_size_multiplier, 1.0), 0.0)
            decision.position_size_pct = size
            side_quality_adjustment["original_position_size"] = round(original_size, 6)
            side_quality_adjustment["adjusted_position_size"] = round(size, 6)
        score = (
            expected_net_return_pct * 2.35
            + profit_quality_ratio * 1.20
            + success_probability * 0.25
            + edge_pct * 0.25
            + local_quality * 0.18
            + confidence * 0.10
            + confidence_bonus
            + rr_bonus
            + min(size * leverage, 1.0) * 0.05
            - expected_loss_pct * 0.90
            - tail_risk_penalty
            - risk_penalty
            - weak_rr_penalty
            - downside_asymmetry_penalty
            - exposure_penalty
            - quant_conflict_penalty
            + exposure_balance_bonus
            + direction_alignment_bonus
            - direction_conflict_penalty
            + historical_adjustment
            + contribution_score_adjustment
            + habit_score_adjustment
            + side_quality_score_adjustment
        )
        if historical_block:
            score -= 0.20 if strong_current_profit_support else 0.45
            if not strong_current_profit_support:
                min_score_required = max(min_score_required, ENTRY_WEAK_HISTORY_MIN_SCORE)
                dynamic_score_reason = (
                    "该币种方向近期真实盈亏偏弱；只有净盈亏比足够高且模型证据改善时才允许继续试。"
                )

        raw["opportunity_score"] = {
            "score": round(score, 6),
            "side": side,
            "expected_return_pct": round(expected_pct, 6),
            "raw_expected_return_pct": round(raw_expected_pct, 6),
            "opposite_expected_return_pct": round(opposite_expected_pct, 6),
            "raw_opposite_expected_return_pct": round(raw_opposite_expected_pct, 6),
            "ml_expected_return_score_cap_pct": ML_EXPECTED_RETURN_SCORE_CAP_PCT,
            "profit_edge_pct": round(edge_pct, 6),
            "win_rate": round(win_rate, 6),
            "ml_profit_quality_score": round(ml_quality, 6),
            "server_profit_expected_return_pct": round(local_expected, 6),
            "server_profit_best_side": local_best_side,
            "server_profit_conflict": bool(local_conflicts),
            "server_profit_loss_probability": round(local_loss_probability, 6),
            "server_profit_quality_score": round(local_quality, 6),
            "timeseries_expected_return_pct": round(ts_expected, 6),
            "timeseries_aligned": bool(ts_aligned),
            "confidence": round(confidence, 6),
            "ai_expected_return_pct": round(ai_expected_return_pct, 6),
            "expected_net_return_pct": round(expected_net_return_pct, 6),
            "expected_loss_pct": round(expected_loss_pct, 6),
            "expected_net_weights": {
                "ai_expected_return": ENTRY_NET_WEIGHT_AI,
                "local_ml_expected_return": ENTRY_NET_WEIGHT_LOCAL_ML,
                "server_profit_expected_return": ENTRY_NET_WEIGHT_SERVER_PROFIT,
                "timeseries_expected_return": ENTRY_NET_WEIGHT_TIMESERIES,
            },
            "downside_asymmetry_penalty": round(downside_asymmetry_penalty, 6),
            "tail_risk_score": round(tail_risk_score, 6),
            "tail_risk_penalty": round(tail_risk_penalty, 6),
            "quant_conflict_penalty": round(quant_conflict_penalty, 6),
            "same_side_loss_concentration": bool(same_side_loss_concentration),
            "tail_history_component": round(tail_history_component, 6),
            "stop_risk_component": round(stop_risk_component, 6),
            "abnormal_wick_component": round(abnormal_wick_component, 6),
            "abnormal_wick_count_72h": int(abnormal_wick_count),
            "abnormal_wick_max_pct": round(abnormal_wick_max_pct, 6),
            "abnormal_wick_recent_hours": round(abnormal_wick_recent_hours, 6),
            "success_probability": round(success_probability, 6),
            "profit_quality_ratio": round(profit_quality_ratio, 6),
            "min_profit_quality_ratio_required": round(min_profit_quality_ratio_required, 6),
            "strong_aligned_profit_evidence": bool(strong_aligned_profit_evidence),
            "capital_efficiency_score": round(capital_efficiency_score, 6),
            "reward_risk_ratio": round(reward_risk_ratio, 6),
            "confidence_bonus": round(confidence_bonus, 6),
            "reward_risk_bonus": round(rr_bonus, 6),
            "size_x_leverage": round(size * leverage, 6),
            "fee_pct": round(fee_pct, 6),
            "slippage_pct": round(slippage_pct, 6),
            "risk_penalty": round(risk_penalty, 6),
            "weak_rr_penalty": round(weak_rr_penalty, 6),
            "exposure_penalty": round(exposure_penalty, 6),
            "position_exposure": exposure if isinstance(exposure, dict) else {},
            "exposure_balance_bonus": round(exposure_balance_bonus, 6),
            "direction_competition": direction_competition,
            "direction_preferred_side": direction_preferred_side,
            "direction_side_score": round(direction_side_score, 6),
            "direction_opposite_score": round(direction_opposite_score, 6),
            "direction_alignment_bonus": round(direction_alignment_bonus, 6),
            "direction_conflict_penalty": round(direction_conflict_penalty, 6),
            "historical_adjustment": round(historical_adjustment, 6),
            "small_win_big_loss_penalty": round(small_win_big_loss_penalty, 6),
            "side_largest_loss_usdt": round(side_largest_loss, 6),
            "side_profit_factor": round(side_profit_factor, 6),
            "symbol_profit_tier": symbol_profit_tier,
            "symbol_profit_tier_reason": symbol_tier_reason,
            "symbol_tier_score_adjustment": round(symbol_tier_score_adjustment, 6),
            "symbol_winner_decay": winner_adjustment.to_dict(),
            "side_realized_pnl_usdt": round(side_pnl, 6),
            "symbol_realized_pnl_usdt": round(symbol_pnl, 6),
            "model_contribution_adjustment": contribution_adjustment,
            "model_contribution_sources": contribution_sources,
            "model_contribution_score_adjustment": round(contribution_score_adjustment, 6),
            "memory_habit_adjustment": memory_habit_adjustment,
            "side_quality_adjustment": side_quality_adjustment,
            "portfolio_roster": portfolio_roster,
            "historical_adjustment_cap": round(historical_adjustment_cap, 6),
            "strong_current_profit_support": bool(strong_current_profit_support),
            "historical_block": bool(historical_block),
            "historical_reason": historical_reason,
            "weak_history_requires_stronger_edge": bool(
                historical_block and not strong_current_profit_support
            ),
            "symbol_side_profile": side_profile,
            "symbol_profile": symbol_profile,
            "base_min_score_required": round(base_min_score_required, 6),
            "min_score_required": round(min_score_required, 6),
            "dynamic_score_reason": dynamic_score_reason,
            "ml_aligned": bool(ml_aligned),
            "local_profit_aligned": bool(local_aligned),
            "expert_aligned": bool(expert_aligned),
            "high_disagreement": bool(high_disagreement),
            "abnormal_volatility": bool(abnormal_volatility),
            "entry_vote_count": int(entry_votes),
            "opposite_vote_count": int(opposite_votes),
            "risk_mode": (
                str(strategy.get("risk_mode") or "normal")
                if isinstance(strategy, dict)
                else "normal"
            ),
            "max_entry_stop_loss_usdt": round(
                self._safe_float(
                    (
                        strategy.get("max_entry_stop_loss_usdt")
                        if isinstance(strategy, dict)
                        else ENTRY_MAX_STOP_LOSS_NORMAL_USDT
                    ),
                    ENTRY_MAX_STOP_LOSS_NORMAL_USDT,
                ),
                6,
            ),
            "ml_influence_enabled": bool(side_influence_enabled),
            "ml_influence_reason": (
                "ML 当前达标，参与机会评分。"
                if side_influence_enabled
                else "ML 当前处于学习观察中，或该方向未达标，本次机会评分不使用 ML 加减分。"
            ),
            "rule": (
                "auto entries are ranked by expected net return, possible loss, fees, "
                "success probability, and capital efficiency before execution"
            ),
        }
        evidence_score = build_entry_evidence_score(decision, raw["opportunity_score"])
        if evidence_score:
            evidence_adjustment = max(
                min(
                    (self._safe_float(evidence_score.get("effective_score"), 0.0) - 65.0) / 35.0,
                    0.75,
                ),
                -1.25,
            )
            if evidence_score.get("hard_block"):
                evidence_adjustment -= 2.0
            score += evidence_adjustment
            raw["opportunity_score"]["score"] = round(score, 6)
            raw["opportunity_score"]["evidence_score"] = evidence_score
            raw["opportunity_score"]["evidence_score_adjustment"] = round(evidence_adjustment, 6)
            raw["opportunity_score"][
                "server_profit_weight_policy"
            ] = "server_profit 只作为辅助证据，不能单独覆盖 ML/时序/AI 的方向冲突。"
        decision.raw_response = raw
        self.annotate_decision_source(decision)
        return score
