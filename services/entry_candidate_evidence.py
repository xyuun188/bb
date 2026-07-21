"""Read-only side-by-side authoritative return evidence for AI context."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.paper_exploration import select_paper_exploration_side

CandidateScorer = Callable[[DecisionOutput, dict[str, Any] | None], float]
FeatureOpportunityScorer = Callable[[Any], float]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if isfinite(result) else default


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _feature_snapshot(feature_vector: Any) -> dict[str, Any]:
    to_dict = getattr(feature_vector, "to_dict", None)
    if not callable(to_dict):
        return {}
    snapshot = to_dict()
    return snapshot if isinstance(snapshot, dict) else {}


def _scheduled_return_prior(
    strategy: dict[str, Any] | None,
    *,
    symbol: str,
    side: str,
) -> dict[str, Any]:
    strategy_context = _safe_dict(strategy)
    learning = _safe_dict(strategy_context.get("strategy_learning"))
    runtime = _safe_dict(learning.get("runtime"))
    current_regime = str(runtime.get("current_market_regime") or "").lower()
    applicable: list[dict[str, Any]] = []
    for profile in runtime.get("governed_profiles") or []:
        item = _safe_dict(profile)
        selector = _safe_dict(item.get("selector"))
        if str(selector.get("side") or "").lower() != side:
            continue
        selected_symbol = str(selector.get("symbol") or "").upper()
        selected_regime = str(selector.get("market_regime") or "").lower()
        if selected_symbol and selected_symbol != symbol.upper():
            continue
        if selected_regime and selected_regime != current_regime:
            continue
        applicable.append(item)
    if not applicable:
        return {
            "available": False,
            "role": "historical_prior_only",
            "match_status": "not_matched",
            "reason": "no_governed_historical_prior_matches_context",
            "context_fields_influenced": [],
            "can_authorize_entry": False,
            "can_change_size_or_leverage": False,
        }
    applicable.sort(
        key=lambda item: (
            bool(_safe_dict(item.get("selector")).get("symbol")),
            bool(_safe_dict(item.get("selector")).get("market_regime")),
            -int(_safe_float(item.get("rank"), 0.0)),
        ),
        reverse=True,
    )
    selected = applicable[0]
    return {
        "available": True,
        "match_status": "matched_historical_prior",
        "profile_id": selected.get("id"),
        "profile_version": selected.get("version"),
        "rank": selected.get("rank"),
        "selector": _safe_dict(selected.get("selector")),
        "historical_return_distribution": _safe_dict(
            selected.get("historical_return_distribution")
        ),
        "walk_forward": _safe_dict(selected.get("walk_forward")),
        "shadow_validation": _safe_dict(selected.get("shadow_validation")),
        "role": "historical_prior_only",
        "context_fields_influenced": ["scheduled_return_prior"],
        "current_return_contract_required": True,
        "can_authorize_entry": False,
        "can_change_size_or_leverage": False,
    }


@dataclass(frozen=True, slots=True)
class EntryCandidateEvidencePolicy:
    """Compare both sides without granting execution or probe permission."""

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
            "long", symbol, feature_vector, strategy, base_raw
        )
        short_evidence = self._build_side(
            "short", symbol, feature_vector, strategy, base_raw
        )
        feature_score = _safe_float(self.feature_opportunity_score(feature_vector), 0.0)
        eligible_sides = [
            item
            for item in (long_evidence, short_evidence)
            if item["production_eligible"]
        ]
        preferred = (
            max(eligible_sides, key=lambda item: item["return_lcb_pct"])["side"]
            if eligible_sides
            else "neutral"
        )
        exploration = select_paper_exploration_side(
            {"long": long_evidence, "short": short_evidence},
            feature_opportunity_score=feature_score,
        )
        generated_at = datetime.now(UTC).isoformat()
        return {
            "enabled": True,
            "read_only": True,
            "is_entry_gate": False,
            "symbol": symbol,
            "preferred_side_by_evidence": preferred,
            "preferred_exploration_side": exploration["preferred_side"],
            "feature_opportunity_score": round(feature_score, 8),
            "long": long_evidence,
            "short": short_evidence,
            "paper_exploration": exploration,
            "memory_feedback_observation": _safe_dict(memory_feedback),
            "policy_provenance": {
                "source": "authoritative_side_return_opportunity_snapshots",
                "observation_window": "current_pre_ai_candidate_round",
                "sample_count": sum(
                    int(item["production_source_count"])
                    for item in (long_evidence, short_evidence)
                ),
                "generated_at": generated_at,
                "strategy_version": "2026-07-12.candidate-return-evidence.v1",
                "fallback_reason": (
                    ""
                    if eligible_sides
                    else "bounded_paper_exploration_available"
                    if exploration["preferred_side"] in {"long", "short"}
                    else "no_production_or_bounded_exploration_side"
                ),
            },
            "policy": (
                "Compare fee-after return LCB for long and short. This context cannot grant "
                "execution, position size, or leverage."
            ),
        }

    def _build_side(
        self,
        side: str,
        symbol: str,
        feature_vector: Any,
        strategy: dict[str, Any] | None,
        base_raw: dict[str, Any],
    ) -> dict[str, Any]:
        decision = DecisionOutput(
            model_name=self.model_name,
            symbol=symbol,
            action=Action.LONG if side == "long" else Action.SHORT,
            confidence=0.0,
            reasoning="authoritative_pre_ai_return_evidence",
            position_size_pct=0.0,
            suggested_leverage=1.0,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            raw_response=dict(base_raw),
            feature_snapshot=_feature_snapshot(feature_vector),
        )
        score = self.score_candidate(decision, strategy)
        finite_score = _optional_float(score)
        opportunity = _safe_dict(_safe_dict(decision.raw_response).get("opportunity_score"))
        expected_net = _safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        return_lcb = _safe_float(opportunity.get("return_lcb_pct"), 0.0)
        source_count = int(
            _safe_float(
                _safe_dict(opportunity.get("policy_provenance")).get("sample_count"),
                0.0,
            )
        )
        production_eligible = bool(
            opportunity.get("production_eligible") is True
            and expected_net > 0.0
            and return_lcb > 0.0
        )
        return {
            "side": side,
            "score": round(finite_score, 8) if finite_score is not None else None,
            "score_policy": opportunity.get("score_policy"),
            "expected_net_return_pct": round(expected_net, 8),
            "return_lcb_pct": round(return_lcb, 8),
            "return_uncertainty_pct": round(
                _safe_float(opportunity.get("return_uncertainty_pct"), 0.0),
                8,
            ),
            "expected_loss_pct": round(
                _safe_float(opportunity.get("expected_loss_pct"), 0.0),
                8,
            ),
            "profit_quality_ratio": round(
                _safe_float(opportunity.get("profit_quality_ratio"), 0.0),
                8,
            ),
            "loss_probability": round(
                _safe_float(opportunity.get("server_profit_loss_probability"), 1.0),
                8,
            ),
            "tail_risk_score": round(
                _safe_float(opportunity.get("tail_risk_score"), 1.0),
                8,
            ),
            "execution_cost": _safe_dict(opportunity.get("execution_cost")),
            "scheduled_return_prior": _scheduled_return_prior(
                strategy,
                symbol=symbol,
                side=side,
            ),
            "production_source_count": source_count,
            "return_distribution_ready": opportunity.get("production_eligible") is True,
            "production_eligible": production_eligible,
            "recommendation": (
                "positive_fee_after_return_lcb"
                if production_eligible
                else "observation_only_or_non_positive_return_lcb"
            ),
            "policy_provenance": _safe_dict(opportunity.get("policy_provenance")),
        }
