"""Policies that keep batched LLM experts meaningfully independent.

The batch expert path is useful for latency, but one model call can collapse
five roles into a single low-information consensus.  This module detects that
specific failure mode and asks the registry to independently retry only the
experts that can validate the objective evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import pstdev
from typing import TYPE_CHECKING, Any

from ai_brain.base_model import Action, DecisionOutput
from services.entry_signal_extraction import (
    expected_return_pct as signal_expected_return_pct,
    first_tool_payload,
    payload_side,
    signal_available,
)

if TYPE_CHECKING:
    from data_feed.feature_vector import FeatureVector

BATCH_EXPERT_NAMES = (
    "trend_expert",
    "momentum_expert",
    "sentiment_expert",
    "position_expert",
    "risk_expert",
)
MARKET_RETRY_EXPERTS = ("trend_expert", "momentum_expert", "sentiment_expert")
POSITION_RETRY_EXPERTS = ("position_expert", "risk_expert", "trend_expert", "momentum_expert")

LOW_INFO_CONFIDENCE_MIN = 0.42
LOW_INFO_CONFIDENCE_MAX = 0.58
LOW_INFO_CONFIDENCE_STDEV_MAX = 0.045
MIN_OBJECTIVE_EVIDENCE_SCORE = 2.0
HARD_WICK_MAX_PCT = 80.0
HARD_WICK_RECENT_HOURS = 96.0


@dataclass(frozen=True, slots=True)
class ObjectiveEvidence:
    """Aggregated directional evidence outside the batch LLM answer."""

    side: str | None
    score: float
    reasons: tuple[str, ...] = ()
    hard_risk: bool = False
    hard_risk_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExpertDiversityReview:
    """Result of reviewing a batched expert response for consensus collapse."""

    should_retry: bool
    reason: str
    low_information_consensus: bool
    objective_evidence: ObjectiveEvidence
    target_experts: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["objective_evidence"] = self.objective_evidence.to_dict()
        return payload


def review_batch_expert_consensus(
    features: FeatureVector,
    context: dict[str, Any],
    decisions: dict[str, DecisionOutput],
) -> ExpertDiversityReview:
    """Decide whether a batched all-hold answer needs independent expert retry.

    The retry is deliberately narrow:
    - no retry for normal mixed answers;
    - no retry for weak objective evidence;
    - no retry when hard market risk already explains holding;
    - retry only role experts that can check the evidence.
    """

    low_info = _is_low_information_hold_consensus(decisions)
    evidence = _objective_directional_evidence(features, context)
    analysis_type = "position" if context.get("review_positions") else "market"
    if not low_info:
        return ExpertDiversityReview(
            should_retry=False,
            reason="batch experts were not a low-information hold consensus",
            low_information_consensus=False,
            objective_evidence=evidence,
        )
    if evidence.hard_risk:
        return ExpertDiversityReview(
            should_retry=False,
            reason=f"hold consensus is explained by hard risk: {evidence.hard_risk_reason}",
            low_information_consensus=True,
            objective_evidence=evidence,
        )
    if not evidence.side or evidence.score < MIN_OBJECTIVE_EVIDENCE_SCORE:
        return ExpertDiversityReview(
            should_retry=False,
            reason="objective directional evidence is too weak for independent retry",
            low_information_consensus=True,
            objective_evidence=evidence,
        )

    targets = POSITION_RETRY_EXPERTS if analysis_type == "position" else MARKET_RETRY_EXPERTS
    return ExpertDiversityReview(
        should_retry=True,
        reason=(
            "batched experts returned low-information all-hold while objective "
            f"evidence favors {evidence.side}"
        ),
        low_information_consensus=True,
        objective_evidence=evidence,
        target_experts=targets,
    )


def _is_low_information_hold_consensus(decisions: dict[str, DecisionOutput]) -> bool:
    selected = [decisions.get(name) for name in BATCH_EXPERT_NAMES]
    if any(not isinstance(item, DecisionOutput) for item in selected):
        return False
    typed = [item for item in selected if isinstance(item, DecisionOutput)]
    if any(decision.action != Action.HOLD for decision in typed):
        return False
    if any(decision.cross_check_for for decision in typed):
        return False

    confidences = [float(decision.confidence or 0.0) for decision in typed]
    if any(
        confidence < LOW_INFO_CONFIDENCE_MIN or confidence > LOW_INFO_CONFIDENCE_MAX
        for confidence in confidences
    ):
        return False
    return pstdev(confidences) <= LOW_INFO_CONFIDENCE_STDEV_MAX


def _objective_directional_evidence(
    features: FeatureVector,
    context: dict[str, Any],
) -> ObjectiveEvidence:
    hard_risk_reason = _hard_risk_reason(features)
    if hard_risk_reason:
        return ObjectiveEvidence(
            side=None,
            score=0.0,
            reasons=(),
            hard_risk=True,
            hard_risk_reason=hard_risk_reason,
        )

    side_scores: dict[str, float] = {"long": 0.0, "short": 0.0}
    side_reasons: dict[str, list[str]] = {"long": [], "short": []}

    _score_ml_signal(context, side_scores, side_reasons)
    _score_local_profit_model(context, side_scores, side_reasons)
    _score_entry_candidate_evidence(context, side_scores, side_reasons)
    _score_direction_competition(context, side_scores, side_reasons)
    _score_feature_momentum(features, side_scores, side_reasons)
    _score_position_review_context(context, side_scores, side_reasons)

    best_side = max(side_scores, key=side_scores.get)
    best_score = side_scores[best_side]
    opposite = "short" if best_side == "long" else "long"
    if best_score <= 0 or best_score - side_scores[opposite] < 0.75:
        return ObjectiveEvidence(side=None, score=best_score, reasons=())
    return ObjectiveEvidence(
        side=best_side,
        score=round(best_score, 3),
        reasons=tuple(side_reasons[best_side][:8]),
    )


def _score_ml_signal(
    context: dict[str, Any],
    side_scores: dict[str, float],
    side_reasons: dict[str, list[str]],
) -> None:
    ml_signal = context.get("ml_signal")
    if not isinstance(ml_signal, dict):
        return
    if ml_signal.get("available") is False or ml_signal.get("influence_enabled") is False:
        return
    predictions = ml_signal.get("predictions")
    primary = predictions[0] if isinstance(predictions, list) and predictions else {}
    if not isinstance(primary, dict):
        return

    long_expected = _float(primary.get("long_expected_return_pct"))
    short_expected = _float(primary.get("short_expected_return_pct"))
    side = _normal_side(primary.get("best_side")) or (
        "long" if long_expected >= short_expected else "short"
    )
    expected = long_expected if side == "long" else short_expected
    opposite_expected = short_expected if side == "long" else long_expected
    edge = expected - opposite_expected
    if expected >= 0.12 and edge >= 0.08:
        points = 2.0 if expected >= 0.25 or edge >= 0.18 else 1.5
        side_scores[side] += points
        side_reasons[side].append(f"ml_signal:{expected:.3f}/{edge:.3f}")


def _score_local_profit_model(
    context: dict[str, Any],
    side_scores: dict[str, float],
    side_reasons: dict[str, list[str]],
) -> None:
    tools = context.get("local_ai_tools")
    profit = (
        first_tool_payload(
            {"local_ai_tools": tools},
            "profit_prediction",
            "profit_model",
            "server_profit",
            "server_profit_model",
            "profit",
        )
        if isinstance(tools, dict)
        else {}
    )
    if not signal_available(profit):
        return
    side = _normal_side(payload_side(profit) or profit.get("direction"))
    if side is None:
        long_expected = signal_expected_return_pct(profit, "long")
        short_expected = signal_expected_return_pct(profit, "short")
        side = "long" if long_expected >= short_expected else "short"
    opposite = "short" if side == "long" else "long"
    expected = signal_expected_return_pct(profit, side)
    opposite_expected = signal_expected_return_pct(profit, opposite)
    loss_probability = _float(profit.get(f"{side}_loss_probability"), 0.50)
    if expected >= 0.03 and expected >= opposite_expected and loss_probability <= 0.62:
        points = 2.0 if expected >= 0.12 else 1.25
        side_scores[side] += points
        side_reasons[side].append(f"local_profit:{expected:.3f}/loss={loss_probability:.2f}")


def _score_entry_candidate_evidence(
    context: dict[str, Any],
    side_scores: dict[str, float],
    side_reasons: dict[str, list[str]],
) -> None:
    evidence = context.get("entry_candidate_evidence")
    if not isinstance(evidence, dict):
        return
    for side in ("long", "short"):
        side_payload = evidence.get(side)
        if not isinstance(side_payload, dict):
            continue
        score = max(
            _float(side_payload.get("score")),
            _float(side_payload.get("effective_score")),
            _float(side_payload.get("dynamic_evidence_score")),
        )
        expected_net = _float(side_payload.get("expected_net_return_pct"))
        profit_quality = _float(side_payload.get("profit_quality_ratio"))
        tail_risk = _float(side_payload.get("tail_risk_score"), 1.0)
        aligned = int(_float(side_payload.get("aligned_source_count"), 0.0))
        if score >= 45 or (expected_net >= 0.35 and profit_quality >= 0.55 and tail_risk <= 0.85):
            points = 2.0 if score >= 60 or expected_net >= 0.75 else 1.5
            if aligned >= 3:
                points += 0.5
            side_scores[side] += points
            side_reasons[side].append(f"entry_evidence:{score:.1f}/{expected_net:.2f}")


def _score_direction_competition(
    context: dict[str, Any],
    side_scores: dict[str, float],
    side_reasons: dict[str, list[str]],
) -> None:
    competition = context.get("direction_competition")
    if not isinstance(competition, dict):
        return
    side = _normal_side(
        competition.get("preferred_side")
        or competition.get("winner_side")
        or competition.get("best_side")
    )
    if side is None:
        return
    gap = abs(_float(competition.get("score_gap")))
    if gap >= 0.08:
        side_scores[side] += 1.0
        side_reasons[side].append(f"direction_competition:gap={gap:.2f}")


def _score_feature_momentum(
    features: FeatureVector,
    side_scores: dict[str, float],
    side_reasons: dict[str, list[str]],
) -> None:
    returns_1 = _float(getattr(features, "returns_1", 0.0))
    returns_5 = _float(getattr(features, "returns_5", 0.0))
    returns_20 = _float(getattr(features, "returns_20", 0.0))
    price_vs_sma20 = _float(getattr(features, "price_vs_sma20", 0.0))
    price_vs_sma50 = _float(getattr(features, "price_vs_sma50", 0.0))
    adx = _float(getattr(features, "adx_14", 0.0))
    volume_ratio = _float(getattr(features, "volume_ratio", 1.0), 1.0)

    long_setup = (
        returns_5 >= 0.002
        and returns_20 >= 0.004
        and price_vs_sma20 >= 0.006
        and price_vs_sma50 >= 0.004
    )
    short_setup = (
        returns_5 <= -0.002
        and returns_20 <= -0.004
        and price_vs_sma20 <= -0.006
        and price_vs_sma50 <= -0.004
    )
    if adx >= 16 and volume_ratio >= 1.05:
        if long_setup:
            side_scores["long"] += 1.0
            side_reasons["long"].append("feature_momentum:trend_up")
        if short_setup:
            side_scores["short"] += 1.0
            side_reasons["short"].append("feature_momentum:trend_down")
    if abs(returns_1) >= 0.003 and volume_ratio >= 1.20:
        side = "long" if returns_1 > 0 else "short"
        side_scores[side] += 0.5
        side_reasons[side].append(f"feature_burst:{returns_1:.3f}")


def _score_position_review_context(
    context: dict[str, Any],
    side_scores: dict[str, float],
    side_reasons: dict[str, list[str]],
) -> None:
    if not context.get("review_positions"):
        return
    for position in context.get("open_positions") or []:
        if not isinstance(position, dict):
            continue
        side = _normal_side(position.get("side"))
        if side not in {"long", "short"}:
            continue
        pnl = _float(
            position.get("unrealized_pnl"),
            _float(position.get("unrealized_pnl_usdt")),
        )
        risk_usage = _float(position.get("risk_usage"), _float(position.get("risk_usage_pct")))
        opposite = "short" if side == "long" else "long"
        if pnl < -1.0 or risk_usage >= 0.55:
            side_scores[opposite] += 1.0
            side_reasons[opposite].append(f"position_review:pnl={pnl:.2f}/risk={risk_usage:.2f}")


def _hard_risk_reason(features: FeatureVector) -> str:
    wick_count = int(_float(getattr(features, "abnormal_wick_count_72h", 0.0)))
    wick_max = _float(getattr(features, "abnormal_wick_max_pct", 0.0))
    wick_recent = _float(getattr(features, "abnormal_wick_recent_hours", 9999.0), 9999.0)
    if wick_count > 0 and wick_max >= HARD_WICK_MAX_PCT and wick_recent <= HARD_WICK_RECENT_HOURS:
        return f"recent abnormal wick {wick_max:.1f}% within {wick_recent:.1f}h"
    return ""


def _side_expected(payload: dict[str, Any], side: str) -> float:
    return max(
        _float(payload.get(f"adjusted_{side}_return_pct")),
        _float(payload.get(f"{side}_expected_return_pct")),
        _float(payload.get(f"{side}_return_pct")),
    )


def _normal_side(value: Any) -> str | None:
    side = str(value or "").strip().lower()
    if side in {"buy", "open_long"}:
        return "long"
    if side in {"sell", "open_short"}:
        return "short"
    if side in {"long", "short"}:
        return side
    return None


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
