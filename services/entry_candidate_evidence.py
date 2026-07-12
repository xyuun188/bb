"""Build structured long/short entry evidence before asking the LLM."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.entry_priority import MIN_ENTRY_OPPORTUNITY_SCORE
from services.entry_profit_risk_sizing import ENTRY_MEANINGFUL_SIZE_MAX_TAIL_RISK
from services.trading_params import DEFAULT_TRADING_PARAMS

CandidateScorer = Callable[[DecisionOutput, dict[str, Any] | None], float]
FeatureOpportunityScorer = Callable[[Any], float]
_ENTRY_EVIDENCE_PARAMS = DEFAULT_TRADING_PARAMS.entry_evidence


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


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _feature_snapshot(feature_vector: Any) -> dict[str, Any]:
    to_dict = getattr(feature_vector, "to_dict", None)
    if callable(to_dict):
        snapshot = to_dict()
        if isinstance(snapshot, dict):
            return snapshot
    return {}


@dataclass(frozen=True, slots=True)
class EntryCandidateEvidencePolicy:
    """Create prompt evidence for both entry sides without executing a trade."""

    model_name: str
    score_candidate: CandidateScorer
    feature_opportunity_score: FeatureOpportunityScorer

    def build(
        self,
        feature_vector: Any,
        strategy: dict[str, Any] | None,
        ml_signal_context: dict[str, Any] | None,
        local_ai_tools_context: dict[str, Any] | None,
        direction_competition_context: dict[str, Any] | None,
        memory_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        symbol = str(getattr(feature_vector, "symbol", "") or "")
        base_raw = {
            "analysis_type": "market",
            "ml_signal": ml_signal_context or {},
            "local_ai_tools": local_ai_tools_context or {},
            "direction_competition": direction_competition_context or {},
            "memory_feedback": memory_feedback or {},
            "pre_ai_candidate_evidence": True,
        }

        long_evidence = self._build_side(
            "long",
            symbol,
            feature_vector,
            strategy,
            base_raw,
            memory_feedback,
        )
        short_evidence = self._build_side(
            "short",
            symbol,
            feature_vector,
            strategy,
            base_raw,
            memory_feedback,
        )
        preferred_side = self._preferred_side(long_evidence, short_evidence)
        return {
            "enabled": True,
            "symbol": symbol,
            "feature_opportunity_score": round(
                _safe_float(self.feature_opportunity_score(feature_vector), 0.0),
                4,
            ),
            "preferred_side_by_evidence": preferred_side,
            "memory_feedback": self._compact_memory_feedback(memory_feedback),
            "long": long_evidence,
            "short": short_evidence,
            "policy": (
                "This is prompt evidence, not an execution veto. AI must compare "
                "long/short expected net profit, loss probability, payoff quality, "
                "recent realized performance, and tail risk before choosing action, "
                "size, leverage, stop loss, and take profit."
            ),
        }

    def _build_side(
        self,
        side: str,
        symbol: str,
        feature_vector: Any,
        strategy: dict[str, Any] | None,
        base_raw: dict[str, Any],
        memory_feedback: dict[str, Any] | None,
    ) -> dict[str, Any]:
        action = Action.LONG if side == "long" else Action.SHORT
        decision = DecisionOutput(
            model_name=self.model_name,
            symbol=symbol,
            action=action,
            confidence=0.62,
            reasoning="pre_ai_candidate_evidence",
            position_size_pct=0.03,
            suggested_leverage=3.0,
            stop_loss_pct=0.015,
            take_profit_pct=0.045,
            raw_response=dict(base_raw),
            feature_snapshot=_feature_snapshot(feature_vector),
        )
        score_before_memory = self.score_candidate(decision, strategy)
        side_feedback = self._side_memory_feedback(memory_feedback, side)
        memory_bonus = _safe_float(side_feedback.get("candidate_score_bonus"), 0.0)
        score = score_before_memory + memory_bonus
        raw = _safe_dict(decision.raw_response)
        opportunity = _safe_dict(raw.get("opportunity_score"))
        expected_net = _safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        tail_risk = _safe_float(opportunity.get("tail_risk_score"), 0.0)
        loss_probability = _safe_float(
            opportunity.get("server_profit_loss_probability"),
            0.5,
        )
        profit_quality = _safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        min_score = _safe_float(
            opportunity.get("min_score_required"),
            MIN_ENTRY_OPPORTUNITY_SCORE,
        )
        high_profit_potential = bool(
            expected_net >= 1.20
            and profit_quality >= 1.20
            and loss_probability <= 0.38
            and tail_risk <= ENTRY_MEANINGFUL_SIZE_MAX_TAIL_RISK
            and (
                opportunity.get("ml_aligned")
                or opportunity.get("local_profit_aligned")
                or opportunity.get("timeseries_aligned")
            )
        )
        probe_block_reasons = self._probe_conversion_block_reasons(
            expected_net=expected_net,
            profit_quality=profit_quality,
            loss_probability=loss_probability,
            tail_risk=tail_risk,
        )
        recommendation = self._recommendation(
            score,
            min_score,
            expected_net,
            profit_quality,
            loss_probability,
            tail_risk,
            high_profit_potential,
            side_feedback,
            probe_conversion_ready=not probe_block_reasons,
        )
        return {
            "side": side,
            "score": round(score, 6),
            "score_before_memory_feedback": round(score_before_memory, 6),
            "memory_candidate_score_bonus": round(memory_bonus, 6),
            "min_score_reference": round(min_score, 6),
            "expected_net_return_pct": round(expected_net, 6),
            "expected_loss_pct": opportunity.get("expected_loss_pct"),
            "success_probability": opportunity.get("success_probability"),
            "loss_probability": round(loss_probability, 6),
            "profit_quality_ratio": round(profit_quality, 6),
            "tail_risk_score": round(tail_risk, 6),
            "high_profit_potential": high_profit_potential,
            "sizing_hint": (
                "profit_potential_large: AI may use higher size/leverage if thesis is clear"
                if high_profit_potential
                else "normal_or_small: do not enlarge unless AI finds stronger evidence"
            ),
            "reward_risk_ratio": opportunity.get("reward_risk_ratio"),
            "ml_expected_return_pct": opportunity.get("expected_return_pct"),
            "ml_win_rate": opportunity.get("diagnostic_win_rate"),
            "server_profit_expected_return_pct": opportunity.get(
                "server_profit_expected_return_pct"
            ),
            "server_profit_best_side": opportunity.get("server_profit_best_side"),
            "server_profit_conflict": bool(opportunity.get("server_profit_conflict")),
            "timeseries_expected_return_pct": opportunity.get("timeseries_expected_return_pct"),
            "timeseries_aligned": bool(opportunity.get("timeseries_aligned")),
            "direction_side_score": opportunity.get("direction_side_score"),
            "direction_opposite_score": opportunity.get("direction_opposite_score"),
            "historical_reason": opportunity.get("historical_reason"),
            "historical_block": bool(opportunity.get("historical_block")),
            "review_feedback": side_feedback,
            "symbol_profile": self._compact_profile(opportunity.get("symbol_profile")),
            "symbol_side_profile": self._compact_profile(opportunity.get("symbol_side_profile")),
            "abnormal_wick_count_72h": opportunity.get("abnormal_wick_count_72h"),
            "abnormal_wick_max_pct": opportunity.get("abnormal_wick_max_pct"),
            "abnormal_wick_recent_hours": opportunity.get("abnormal_wick_recent_hours"),
            "probe_conversion_ready": bool(not probe_block_reasons),
            "probe_conversion_block_reasons": probe_block_reasons,
            "probe_conversion_thresholds": self._probe_conversion_thresholds(),
            "recommendation": recommendation,
        }

    @staticmethod
    def _recommendation(
        score: float,
        min_score: float,
        expected_net: float,
        profit_quality: float,
        loss_probability: float,
        tail_risk: float,
        high_profit_potential: bool,
        side_feedback: dict[str, Any],
        *,
        probe_conversion_ready: bool,
    ) -> str:
        action_bias = str(side_feedback.get("action_bias") or "")
        allow_probe = bool(side_feedback.get("allow_probe"))
        if action_bias == "require_stronger_confirmation":
            return "memory_risk_requires_stronger_confirmation"
        if high_profit_potential:
            return "high_profit_candidate_allow_larger_size_and_leverage"
        if allow_probe and probe_conversion_ready:
            return "memory_supported_probe_candidate"
        if (
            allow_probe
            and expected_net > 0
            and profit_quality >= _ENTRY_EVIDENCE_PARAMS.positive_net_probe_min_profit_quality
            and loss_probability <= _ENTRY_EVIDENCE_PARAMS.positive_net_probe_max_loss_probability
            and tail_risk <= _ENTRY_EVIDENCE_PARAMS.positive_net_probe_max_tail_risk
        ):
            return "memory_watchlist_needs_probe_threshold"
        if expected_net <= 0 or profit_quality <= 0.12 or tail_risk >= 1.15:
            return "hold_or_tiny_probe_only"
        if score >= min_score and expected_net > 0 and tail_risk < 0.95:
            return "tradable_if_ai_thesis_confirms"
        return "needs_stronger_ai_confirmation"

    @staticmethod
    def _probe_conversion_block_reasons(
        *,
        expected_net: float,
        profit_quality: float,
        loss_probability: float,
        tail_risk: float,
    ) -> list[str]:
        reasons: list[str] = []
        if expected_net < _ENTRY_EVIDENCE_PARAMS.positive_net_probe_min_expected_pct:
            reasons.append("expected_net_below_probe_threshold")
        if profit_quality < _ENTRY_EVIDENCE_PARAMS.positive_net_probe_min_profit_quality:
            reasons.append("profit_quality_below_probe_threshold")
        if loss_probability > _ENTRY_EVIDENCE_PARAMS.positive_net_probe_max_loss_probability:
            reasons.append("loss_probability_above_probe_threshold")
        if tail_risk > _ENTRY_EVIDENCE_PARAMS.positive_net_probe_max_tail_risk:
            reasons.append("tail_risk_above_probe_threshold")
        return reasons

    @staticmethod
    def _probe_conversion_thresholds() -> dict[str, float]:
        return {
            "min_expected_net_return_pct": round(
                _ENTRY_EVIDENCE_PARAMS.positive_net_probe_min_expected_pct,
                6,
            ),
            "min_profit_quality_ratio": round(
                _ENTRY_EVIDENCE_PARAMS.positive_net_probe_min_profit_quality,
                6,
            ),
            "max_loss_probability": round(
                _ENTRY_EVIDENCE_PARAMS.positive_net_probe_max_loss_probability,
                6,
            ),
            "max_tail_risk_score": round(
                _ENTRY_EVIDENCE_PARAMS.positive_net_probe_max_tail_risk,
                6,
            ),
        }

    @staticmethod
    def _side_memory_feedback(
        memory_feedback: dict[str, Any] | None,
        side: str,
    ) -> dict[str, Any]:
        feedback = _safe_dict(memory_feedback)
        by_side = _safe_dict(feedback.get("by_side"))
        item = _safe_dict(by_side.get(side))
        if not item:
            return {
                "side": side,
                "memory_count": 0,
                "missed_opportunity_count": 0,
                "positive_evidence_count": 0,
                "risk_evidence_count": 0,
                "score_adjustment": 0.0,
                "candidate_score_bonus": 0.0,
                "allow_probe": False,
                "action_bias": "neutral",
            }
        return item

    @staticmethod
    def _compact_memory_feedback(memory_feedback: dict[str, Any] | None) -> dict[str, Any]:
        feedback = _safe_dict(memory_feedback)
        if not feedback:
            return {}
        by_side = _safe_dict(feedback.get("by_side"))
        return {
            "enabled": bool(feedback.get("enabled")),
            "preferred_side_by_memory": feedback.get("preferred_side_by_memory"),
            "vector_memory": EntryCandidateEvidencePolicy._compact_vector_memory(
                _safe_dict(feedback.get("vector_memory"))
            ),
            "decision_habit": EntryCandidateEvidencePolicy._compact_decision_habit(
                _safe_dict(feedback.get("decision_habit"))
            ),
            "long": EntryCandidateEvidencePolicy._compact_side_feedback(
                _safe_dict(by_side.get("long"))
            ),
            "short": EntryCandidateEvidencePolicy._compact_side_feedback(
                _safe_dict(by_side.get("short"))
            ),
            "policy": str(feedback.get("policy") or "")[:180],
        }

    @staticmethod
    def _compact_vector_memory(item: dict[str, Any]) -> dict[str, Any]:
        if not item:
            return {}
        return {
            "enabled": bool(item.get("enabled")),
            "status": str(item.get("status") or ""),
            "matched_count": _safe_int(item.get("matched_count"), 0),
            "policy": str(item.get("policy") or "")[:120],
            "hits": [
                {
                    "score": round(_safe_float(hit.get("score"), 0.0), 6),
                    "action": str(hit.get("action") or ""),
                    "outcome": str(hit.get("outcome") or ""),
                    "pnl_pct": hit.get("pnl_pct"),
                }
                for hit in (item.get("hits") if isinstance(item.get("hits"), list) else [])[:3]
                if isinstance(hit, dict)
            ],
        }

    @staticmethod
    def _compact_side_feedback(item: dict[str, Any]) -> dict[str, Any]:
        if not item:
            return {}
        return {
            "action_bias": item.get("action_bias"),
            "allow_probe": bool(item.get("allow_probe")),
            "missed_opportunity_count": _safe_int(item.get("missed_opportunity_count"), 0),
            "positive_evidence_count": _safe_int(item.get("positive_evidence_count"), 0),
            "risk_evidence_count": _safe_int(item.get("risk_evidence_count"), 0),
            "candidate_score_bonus": round(_safe_float(item.get("candidate_score_bonus"), 0.0), 6),
            "max_probe_size_pct": round(_safe_float(item.get("max_probe_size_pct"), 0.0), 6),
        }

    @staticmethod
    def _compact_decision_habit(item: dict[str, Any]) -> dict[str, Any]:
        if not item:
            return {}
        by_side = _safe_dict(item.get("by_side"))
        return {
            "posture": item.get("posture"),
            "preferred_side": item.get("preferred_side"),
            "active_probe_sides": item.get("active_probe_sides") or [],
            "conservative_sides": item.get("conservative_sides") or [],
            "long": EntryCandidateEvidencePolicy._compact_side_habit(
                _safe_dict(by_side.get("long"))
            ),
            "short": EntryCandidateEvidencePolicy._compact_side_habit(
                _safe_dict(by_side.get("short"))
            ),
        }

    @staticmethod
    def _compact_side_habit(item: dict[str, Any]) -> dict[str, Any]:
        if not item:
            return {}
        return {
            "stance": item.get("stance"),
            "proactive_level": round(_safe_float(item.get("proactive_level"), 0.0), 6),
            "probe_budget_pct": round(_safe_float(item.get("probe_budget_pct"), 0.0), 6),
            "min_expected_net_pct": round(_safe_float(item.get("min_expected_net_pct"), 0.0), 6),
            "max_loss_probability": round(_safe_float(item.get("max_loss_probability"), 0.0), 6),
            "max_tail_risk": round(_safe_float(item.get("max_tail_risk"), 0.0), 6),
        }

    @staticmethod
    def _preferred_side(
        long_evidence: dict[str, Any],
        short_evidence: dict[str, Any],
    ) -> str:
        long_score = _safe_float(long_evidence.get("score"), 0.0)
        short_score = _safe_float(short_evidence.get("score"), 0.0)
        if long_score > short_score + 0.08:
            return "long"
        if short_score > long_score + 0.08:
            return "short"
        return "neutral"

    @staticmethod
    def _compact_profile(profile: Any) -> dict[str, Any]:
        if not isinstance(profile, dict):
            return {}
        return {
            "count": _safe_int(profile.get("count"), 0),
            "pnl": round(_safe_float(profile.get("pnl"), 0.0), 4),
            "today_pnl": round(_safe_float(profile.get("today_pnl"), 0.0), 4),
            "wins": _safe_int(profile.get("wins"), 0),
            "losses": _safe_int(profile.get("losses"), 0),
            "profit_factor": round(_safe_float(profile.get("profit_factor"), 0.0), 4),
            "largest_loss": round(_safe_float(profile.get("largest_loss"), 0.0), 4),
            "first_closed_at": profile.get("first_closed_at"),
            "last_closed_at": profile.get("last_closed_at"),
            "last_loss_at": profile.get("last_loss_at"),
            "last_loss_age_hours": profile.get("last_loss_age_hours"),
            "lookback_days": profile.get("lookback_days"),
            "cooldown": bool(profile.get("cooldown")),
            "cooldown_reason": str(profile.get("cooldown_reason") or "")[:120],
            "cooldown_kind": str(profile.get("cooldown_kind") or "")[:80],
            "cooldown_time_based": bool(profile.get("cooldown_time_based")),
            "cooldown_remaining_hours": round(
                _safe_float(profile.get("cooldown_remaining_hours"), 0.0),
                6,
            ),
            "profile_scope": str(profile.get("profile_scope") or "")[:40],
        }
