"""Entry evidence scoring for new position decisions.

This module owns the model-evidence policy.  TradingService should only pass
decision context in and consume the returned score/tier/block result.
"""

from __future__ import annotations

from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.entry_signal_extraction import (
    directional_expected_return_pct,
    entry_signal_payloads,
    expected_return_pct,
    payload_side,
    safe_float,
    signal_available,
)
from services.trading_params import DEFAULT_TRADING_PARAMS

_ENTRY_TIER_PARAMS = DEFAULT_TRADING_PARAMS.entry_tiers
_ENTRY_EVIDENCE_PARAMS = DEFAULT_TRADING_PARAMS.entry_evidence
ENTRY_EVIDENCE_SCORE_NORMAL = _ENTRY_TIER_PARAMS.normal_score
ENTRY_EVIDENCE_SCORE_MEDIUM = _ENTRY_TIER_PARAMS.medium_score
ENTRY_EVIDENCE_SCORE_SMALL = _ENTRY_TIER_PARAMS.small_score
ENTRY_EVIDENCE_SCORE_PROBE = _ENTRY_TIER_PARAMS.exploration_score
ENTRY_EVIDENCE_SCORE_WEAK_PROBE = _ENTRY_TIER_PARAMS.weak_probe_score
ENTRY_EVIDENCE_SCORE_HARD_BLOCK = _ENTRY_TIER_PARAMS.weak_probe_score
ENTRY_EVIDENCE_WEAK_PROBE_MIN_ALIGNED_SOURCES = _ENTRY_TIER_PARAMS.weak_probe_min_aligned_sources
ENTRY_EVIDENCE_SHORT_SCORE_OFFSET = _ENTRY_EVIDENCE_PARAMS.short_score_offset
ENTRY_EVIDENCE_SHORT_SIZE_MULTIPLIER = _ENTRY_EVIDENCE_PARAMS.short_size_multiplier
ENTRY_EVIDENCE_MAJOR_CONFLICT_SIZE_CAP = _ENTRY_EVIDENCE_PARAMS.major_conflict_size_cap
ENTRY_EVIDENCE_EXPLORATION_SIZE_CAP = _ENTRY_TIER_PARAMS.exploration_size_cap
ENTRY_EVIDENCE_WEAK_CONFLICT_SIZE_CAP = _ENTRY_TIER_PARAMS.weak_probe_size_cap
ENTRY_EVIDENCE_MISSING_KEY_SIZE_CAP = _ENTRY_EVIDENCE_PARAMS.missing_key_size_cap
ENTRY_EVIDENCE_WEAK_OPPOSITE_RETURN_PCT = _ENTRY_EVIDENCE_PARAMS.weak_opposite_return_pct
ENTRY_EVIDENCE_STRONG_OPPOSITE_RETURN_PCT = _ENTRY_EVIDENCE_PARAMS.strong_opposite_return_pct
ENTRY_EVIDENCE_WEAK_OPPOSITE_PENALTY_RATIO = _ENTRY_EVIDENCE_PARAMS.weak_opposite_penalty_ratio
ENTRY_EVIDENCE_NORMAL_OPPOSITE_PENALTY_RATIO = _ENTRY_EVIDENCE_PARAMS.normal_opposite_penalty_ratio
ENTRY_EVIDENCE_AI_SUPPORT_EXCLUDED_EXPERTS = {"position_expert", "risk_expert"}
ENTRY_EVIDENCE_SHORT_PROBE_RELIEF_MIN_BASE_SCORE = (
    _ENTRY_EVIDENCE_PARAMS.short_probe_relief_min_base_score
)
ENTRY_EVIDENCE_SHORT_PROBE_RELIEF_MIN_EFFECTIVE_SCORE = (
    _ENTRY_EVIDENCE_PARAMS.short_probe_relief_min_effective_score
)
ENTRY_EVIDENCE_SHORT_PROBE_RELIEF_MIN_DIRECTION_GAP = (
    _ENTRY_EVIDENCE_PARAMS.short_probe_relief_min_direction_gap
)
ENTRY_EVIDENCE_SHORT_PROBE_RELIEF_MAX_LOSS_PROBABILITY = (
    _ENTRY_EVIDENCE_PARAMS.short_probe_relief_max_loss_probability
)


def _signal_component(
    *,
    label: str,
    source: str,
    available: bool,
    side: str,
    entry_side: str,
    weight: float,
    expected_return_pct: float = 0.0,
    missing_penalty: float = 0.0,
    opposite_penalty_ratio: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    side = str(side or "").lower()
    entry_side = str(entry_side or "").lower()
    status = "missing"
    points = -abs(missing_penalty) if not available or side not in {"long", "short"} else 0.0
    conflict_strength = "none"
    if available and side in {"long", "short"}:
        if side == entry_side:
            if source == "server_profit" and expected_return_pct <= 0.0:
                status = "ignored_negative_expected"
                points = 0.0
            else:
                status = "aligned"
                strength = 1.0 + min(max(expected_return_pct, 0.0) / 3.0, 0.20)
                points = weight * strength
        else:
            abs_expected = abs(expected_return_pct)
            if abs_expected < ENTRY_EVIDENCE_WEAK_OPPOSITE_RETURN_PCT:
                status = "weak_opposite"
                conflict_strength = "weak"
                points = (
                    -weight * opposite_penalty_ratio * ENTRY_EVIDENCE_WEAK_OPPOSITE_PENALTY_RATIO
                )
            elif abs_expected < ENTRY_EVIDENCE_STRONG_OPPOSITE_RETURN_PCT:
                status = "opposite"
                conflict_strength = "normal"
                strength = 1.0 + min(abs_expected / 3.0, 0.20)
                points = (
                    -weight
                    * opposite_penalty_ratio
                    * ENTRY_EVIDENCE_NORMAL_OPPOSITE_PENALTY_RATIO
                    * strength
                )
            else:
                status = "opposite"
                conflict_strength = "strong"
                strength = 1.0 + min(abs_expected / 3.0, 0.20)
                points = -weight * opposite_penalty_ratio * strength
    return points, {
        "source": source,
        "label": label,
        "available": bool(available),
        "side": side,
        "entry_side": entry_side,
        "expected_return_pct": round(expected_return_pct, 6),
        "weight": round(weight, 6),
        "points": round(points, 6),
        "status": status,
        "conflict_strength": conflict_strength,
    }


def _ignored_signal_component(
    *,
    label: str,
    source: str,
    side: str,
    entry_side: str,
    expected_return_pct: float = 0.0,
    reason: str,
) -> tuple[float, dict[str, Any]]:
    return 0.0, {
        "source": source,
        "label": label,
        "available": False,
        "side": str(side or "").lower(),
        "entry_side": str(entry_side or "").lower(),
        "expected_return_pct": round(expected_return_pct, 6),
        "weight": 0.0,
        "points": 0.0,
        "status": "ignored",
        "influence_enabled": False,
        "reason": reason,
    }


def _directional_expert_support(raw: dict[str, Any], entry_side: str) -> tuple[bool, list[str]]:
    opinions = raw.get("opinions")
    if not isinstance(opinions, list):
        opinions = raw.get("experts")
    if not isinstance(opinions, list):
        return False, []

    support: list[str] = []
    for opinion in opinions:
        if not isinstance(opinion, dict):
            continue
        name = str(opinion.get("model_name") or opinion.get("name") or "")
        if name in ENTRY_EVIDENCE_AI_SUPPORT_EXCLUDED_EXPERTS:
            continue
        action = str(opinion.get("action") or "").lower()
        confidence = safe_float(opinion.get("confidence"), 0.0)
        if action == entry_side and confidence >= 0.55:
            support.append(name or "unknown")
    return True, support


def _probe_origin(raw: dict[str, Any]) -> tuple[bool, bool, list[str]]:
    origins: list[str] = []
    evidence_probe = raw.get("evidence_profit_probe")
    original_hold_probe = False
    if isinstance(evidence_probe, dict) and evidence_probe.get("triggered"):
        origins.append("evidence_profit_probe")
        original_hold_probe = str(evidence_probe.get("ai_original_action") or "").lower() == "hold"

    for key in (
        "quant_only_probe_entry",
        "quant_validation_probe_entry",
        "quant_reversal_probe_entry",
    ):
        probe = raw.get(key)
        if isinstance(probe, dict) and probe.get("allow"):
            origins.append(key)

    if raw.get("probe_entry") is True:
        origins.append("probe_entry")
    if raw.get("profit_first_probe_entry") is True:
        origins.append("profit_first_probe_entry")

    return bool(origins), original_hold_probe, origins


def _ai_component(
    *,
    decision: DecisionOutput,
    raw: dict[str, Any],
    entry_side: str,
) -> tuple[float, dict[str, Any]]:
    full_points = 25.0 * min(max(float(decision.confidence or 0.0), 0.0), 1.0)
    opinions_present, support_sources = _directional_expert_support(raw, entry_side)
    probe_derived, original_hold_probe, probe_origins = _probe_origin(raw)

    points = full_points
    status = "aligned"
    reason = ""
    if original_hold_probe and not support_sources:
        points = 0.0
        status = "probe_derived_no_expert_support"
        reason = "Original AI committee held; probe direction cannot count as AI alignment."
    elif probe_derived:
        points = min(full_points, 8.0)
        status = "probe_derived_limited"
        reason = "Probe-derived entry gets capped AI evidence points."
    elif opinions_present and not support_sources:
        points = min(full_points, 8.0)
        status = "limited_no_expert_support"
        reason = "Final entry has no directional expert support; AI evidence is capped."
    elif opinions_present and len(support_sources) == 1:
        points = min(full_points, 14.0)
        status = "limited_single_expert_support"
        reason = "Only one directional expert supports this side; AI evidence is capped."

    item = {
        "source": "ai",
        "label": "AI",
        "available": True,
        "side": entry_side,
        "entry_side": entry_side,
        "weight": 25.0,
        "points": round(points, 6),
        "status": status,
        "confidence": round(float(decision.confidence or 0.0), 6),
        "directional_support_count": len(support_sources),
        "directional_support_sources": support_sources,
        "probe_derived": probe_derived,
        "probe_origins": probe_origins,
    }
    if reason:
        item["reason"] = reason
    return points, item


def _server_profit_component(
    *,
    profit: dict[str, Any],
    profit_side: str,
    entry_side: str,
) -> tuple[float, dict[str, Any]]:
    available = signal_available(profit)
    expected = expected_return_pct(profit, profit_side or entry_side)
    if available and profit_side == entry_side:
        expected = expected_return_pct(profit, entry_side)
        if expected <= 0.0:
            return 0.0, {
                "source": "server_profit",
                "label": "盈利模型",
                "available": True,
                "side": profit_side,
                "entry_side": entry_side,
                "expected_return_pct": round(expected, 6),
                "weight": 5.0,
                "points": 0.0,
                "status": "ignored_negative_expected",
                "conflict_strength": "none",
                "reason": (
                    "Server profit model best_side matches, but adjusted expected return is "
                    "not positive, so it cannot add aligned entry evidence."
                ),
            }
    return _signal_component(
        label="盈利模型",
        source="server_profit",
        available=available,
        side=profit_side,
        entry_side=entry_side,
        weight=5.0,
        expected_return_pct=expected,
        missing_penalty=0.0,
        opposite_penalty_ratio=0.60,
    )


def _memory_component(raw: dict[str, Any], entry_side: str) -> tuple[float, dict[str, Any]]:
    summary_raw = raw.get("memory_summary")
    summary = summary_raw if isinstance(summary_raw, dict) else {}
    feedback_raw = raw.get("memory_feedback")
    feedback = feedback_raw if isinstance(feedback_raw, dict) else {}
    side_feedback = {}
    by_side = feedback.get("by_side") if isinstance(feedback.get("by_side"), dict) else {}
    if isinstance(by_side, dict):
        side_feedback = by_side.get(entry_side) if isinstance(by_side.get(entry_side), dict) else {}
    memory_adjustment = safe_float(raw.get("memory_adjustment"), 0.0)
    positive = int(safe_float(summary.get("positive_lessons"), 0.0))
    risk = int(safe_float(summary.get("risk_lessons"), 0.0))
    used = int(safe_float(summary.get("used"), 0.0))
    feedback_available = bool(side_feedback)
    if used <= 0 and not feedback_available:
        points = 0.0
        status = "missing"
    else:
        points = max(min(memory_adjustment * 45.0 + (positive - risk) * 1.5, 10.0), -10.0)
        if feedback_available:
            feedback_points = safe_float(side_feedback.get("score_adjustment"), 0.0) * 18.0
            if side_feedback.get("allow_probe"):
                feedback_points += 1.25
            if side_feedback.get("action_bias") == "require_stronger_confirmation":
                feedback_points -= 2.0
            points = max(min(points + feedback_points, 10.0), -10.0)
        status = "aligned" if points > 1.0 else "opposite" if points < -1.0 else "neutral"
    return points, {
        "source": "shadow_memory",
        "label": "影子/交易记忆",
        "available": used > 0 or feedback_available,
        "side": entry_side,
        "entry_side": entry_side,
        "weight": 10.0,
        "points": round(points, 6),
        "status": status,
        "memory_adjustment": round(memory_adjustment, 6),
        "used": used,
        "positive_lessons": positive,
        "risk_lessons": risk,
        "review_feedback": {
            "action_bias": side_feedback.get("action_bias"),
            "allow_probe": bool(side_feedback.get("allow_probe")),
            "missed_opportunity_count": int(
                safe_float(side_feedback.get("missed_opportunity_count"), 0.0)
            ),
            "risk_evidence_count": int(safe_float(side_feedback.get("risk_evidence_count"), 0.0)),
            "score_adjustment": round(safe_float(side_feedback.get("score_adjustment"), 0.0), 6),
        },
    }


def build_entry_evidence_score(
    decision: DecisionOutput,
    opportunity: dict[str, Any],
) -> dict[str, Any]:
    """Score whether an entry has enough aligned evidence and how large it may be."""
    if not decision.is_entry:
        return {}
    raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
    entry_side = (
        "long"
        if decision.action == Action.LONG
        else "short" if decision.action == Action.SHORT else str(opportunity.get("side") or "")
    )
    opposite = "short" if entry_side == "long" else "long"
    payloads = entry_signal_payloads(raw)
    ml_signal = payloads["ml"]
    primary_ml = payloads["primary_ml"]
    profit = payloads["server_profit"]
    timeseries = payloads["timeseries"]
    sentiment = payloads["sentiment"]

    ml_influence_enabled = bool(opportunity.get("ml_influence_enabled", True))
    ml_available = bool(ml_signal) and ml_influence_enabled
    ml_side = payload_side(primary_ml)
    ml_expected = safe_float(
        (
            primary_ml.get(
                "best_expected_return_pct", primary_ml.get(f"{entry_side}_expected_return_pct", 0.0)
            )
            if isinstance(primary_ml, dict)
            else 0.0
        ),
        0.0,
    )
    profit_side = payload_side(profit)
    timeseries_side = payload_side(timeseries)
    sentiment_side = payload_side(sentiment)
    components: list[dict[str, Any]] = []
    total = 0.0

    ai_points, ai_item = _ai_component(decision=decision, raw=raw, entry_side=entry_side)
    total += ai_points
    components.append(ai_item)
    ml_component = (
        _signal_component(
            label="ML",
            source="ml",
            available=ml_available,
            side=ml_side,
            entry_side=entry_side,
            weight=30.0,
            expected_return_pct=ml_expected,
            missing_penalty=8.0,
            opposite_penalty_ratio=1.10,
        )
        if ml_influence_enabled
        else _ignored_signal_component(
            label="ML",
            source="ml",
            side=ml_side,
            entry_side=entry_side,
            expected_return_pct=ml_expected,
            reason="ML 当前为学习观察模式，不参与动态证据评分、缺失惩罚或方向冲突硬拦。",
        )
    )

    for points, item in (
        ml_component,
        _signal_component(
            label="时序",
            source="timeseries",
            available=signal_available(timeseries),
            side=timeseries_side,
            entry_side=entry_side,
            weight=20.0,
            expected_return_pct=directional_expected_return_pct(
                timeseries,
                timeseries_side or entry_side,
            ),
            missing_penalty=5.0,
            opposite_penalty_ratio=1.05,
        ),
        _signal_component(
            label="情绪",
            source="sentiment",
            available=signal_available(sentiment),
            side=sentiment_side,
            entry_side=entry_side,
            weight=10.0,
            expected_return_pct=expected_return_pct(sentiment, sentiment_side or entry_side),
            missing_penalty=0.0,
            opposite_penalty_ratio=0.80,
        ),
        _signal_component(
            label="盈利模型",
            source="server_profit",
            available=signal_available(profit),
            side=profit_side,
            entry_side=entry_side,
            weight=5.0,
            expected_return_pct=expected_return_pct(profit, profit_side or entry_side),
            missing_penalty=0.0,
            opposite_penalty_ratio=0.60,
        ),
        _memory_component(raw, entry_side),
    ):
        total += points
        components.append(item)

    side_profile_raw = opportunity.get("symbol_side_profile")
    side_profile = side_profile_raw if isinstance(side_profile_raw, dict) else {}
    side_pnl = safe_float(side_profile.get("pnl"), 0.0)
    side_pf = safe_float(side_profile.get("profit_factor"), 1.0)
    side_count = int(side_profile.get("count") or 0)
    history_points = 0.0
    if side_count >= 2:
        if side_pnl > 0 and side_pf >= 1.15:
            history_points = min(side_pnl / 4.0, 10.0)
        elif side_pnl < 0 or side_pf < 0.85:
            history_points = -min(abs(side_pnl) / 5.0 + max(0.85 - side_pf, 0.0) * 8.0, 14.0)
    total += history_points
    components.append(
        {
            "source": "symbol_side_history",
            "label": "币种方向",
            "available": side_count > 0,
            "side": entry_side,
            "entry_side": entry_side,
            "weight": 10.0,
            "points": round(history_points, 6),
            "status": (
                "aligned" if history_points > 0 else "opposite" if history_points < 0 else "neutral"
            ),
            "count": side_count,
            "pnl": round(side_pnl, 6),
            "profit_factor": round(side_pf, 6),
        }
    )

    score = min(max(total, 0.0), 100.0)
    effective_score = score - (ENTRY_EVIDENCE_SHORT_SCORE_OFFSET if entry_side == "short" else 0.0)
    major_opposites = [
        item["source"]
        for item in components
        if item.get("source") in {"ml", "timeseries", "shadow_memory"}
        and item.get("status") == "opposite"
    ]
    weak_opposites = [
        item["source"]
        for item in components
        if item.get("source") in {"ml", "timeseries", "shadow_memory"}
        and item.get("status") == "weak_opposite"
    ]
    strong_opposites = [
        item["source"]
        for item in components
        if item.get("source") in {"ml", "timeseries", "shadow_memory"}
        and item.get("status") == "opposite"
        and item.get("conflict_strength") == "strong"
    ]
    missing_key_sources = [
        item["source"]
        for item in components
        if item.get("source") in {"ml", "timeseries"} and item.get("status") == "missing"
    ]
    aligned_support_sources = [
        item["source"]
        for item in components
        if item.get("source")
        in {
            "ai",
            "ml",
            "timeseries",
            "sentiment",
            "server_profit",
            "shadow_memory",
            "symbol_side_history",
        }
        and item.get("status") == "aligned"
        and safe_float(item.get("points"), 0.0) > 0
    ]
    short_probe_relief: dict[str, Any] = {"applied": False}
    positive_net_probe_relief: dict[str, Any] = {"applied": False}
    strong_positive_net_relief: dict[str, Any] = {"applied": False}
    memory_missed_opportunity_relief: dict[str, Any] = {"applied": False}
    tradeable_probe = False
    if entry_side == "short" and (
        ENTRY_EVIDENCE_SHORT_PROBE_RELIEF_MIN_EFFECTIVE_SCORE
        <= effective_score
        < ENTRY_EVIDENCE_SCORE_HARD_BLOCK
        <= score
    ):
        direction_competition = (
            opportunity.get("direction_competition")
            if isinstance(opportunity.get("direction_competition"), dict)
            else raw.get("direction_competition")
        )
        direction_competition = (
            direction_competition if isinstance(direction_competition, dict) else {}
        )
        direction_preferred_side = str(
            opportunity.get("direction_preferred_side")
            or direction_competition.get("preferred_side")
            or ""
        ).lower()
        direction_gap = safe_float(direction_competition.get("score_gap"), 0.0)
        server_expected = max(
            safe_float(opportunity.get("server_profit_expected_return_pct"), 0.0),
            expected_return_pct(profit, "short"),
        )
        server_loss_probability = safe_float(
            opportunity.get("server_profit_loss_probability"), 0.50
        )
        server_aligned = "server_profit" in aligned_support_sources or bool(
            opportunity.get("local_profit_aligned")
        )
        timeseries_aligned = "timeseries" in aligned_support_sources or bool(
            opportunity.get("timeseries_aligned")
        )
        short_relief_allowed = bool(
            score >= ENTRY_EVIDENCE_SHORT_PROBE_RELIEF_MIN_BASE_SCORE
            and server_aligned
            and server_expected > 0.0
            and timeseries_aligned
            and direction_preferred_side == "short"
            and direction_gap >= ENTRY_EVIDENCE_SHORT_PROBE_RELIEF_MIN_DIRECTION_GAP
            and server_loss_probability <= ENTRY_EVIDENCE_SHORT_PROBE_RELIEF_MAX_LOSS_PROBABILITY
            and not strong_opposites
            and "timeseries" not in major_opposites
        )
        if short_relief_allowed:
            original_effective_score = effective_score
            effective_score = ENTRY_EVIDENCE_SCORE_WEAK_PROBE
            short_probe_relief = {
                "applied": True,
                "tradeable_probe": False,
                "shadow_only": True,
                "from_effective_score": round(original_effective_score, 6),
                "to_effective_score": round(effective_score, 6),
                "server_expected_return_pct": round(server_expected, 6),
                "server_loss_probability": round(server_loss_probability, 6),
                "direction_preferred_side": direction_preferred_side,
                "direction_gap": round(direction_gap, 6),
                "reason": (
                    "Short base evidence cleared the weak-probe floor, but weak-conflict "
                    "signals remain shadow-only until they lift into exploration/small tier. "
                    "This prevents meaningless micro orders from feeding fast-loss churn."
                ),
            }
    hard_block_reasons: list[str] = []
    advisory_wait_reasons: list[str] = []
    if {"ml", "timeseries"}.issubset(set(strong_opposites)):
        hard_block_reasons.append("ML 和时序同时强反向")
    elif {"ml", "timeseries"}.issubset(set(major_opposites)):
        hard_block_reasons.append("ML 和时序同时明确反向")
    if "ml" in strong_opposites and "shadow_memory" in major_opposites:
        hard_block_reasons.append("ML 反向且影子/交易记忆偏负")
    missing_key_degraded_relief: dict[str, Any] = {"applied": False}
    missing_key_degraded = bool(
        len(missing_key_sources) >= 2 and not major_opposites and not strong_opposites
    )
    if missing_key_degraded:
        ai_confidence = safe_float(ai_item.get("confidence"), 0.0)
        ai_points = safe_float(ai_item.get("points"), 0.0)
        allow_degraded_probe = bool(
            ai_points >= ENTRY_EVIDENCE_SCORE_WEAK_PROBE / 3.0 or aligned_support_sources
        )
        if allow_degraded_probe:
            original_effective_score = effective_score
            if effective_score < ENTRY_EVIDENCE_SCORE_WEAK_PROBE:
                effective_score = ENTRY_EVIDENCE_SCORE_WEAK_PROBE
            missing_key_degraded_relief = {
                "applied": True,
                "tradeable_probe": False,
                "shadow_only": True,
                "missing_key_sources": list(missing_key_sources),
                "from_effective_score": round(original_effective_score, 6),
                "to_effective_score": round(effective_score, 6),
                "ai_confidence": round(ai_confidence, 6),
                "aligned_support_sources": list(aligned_support_sources),
                "reason": (
                    "ML/time-series services are unavailable; missing key model data is "
                    "recorded as shadow-only evidence until the model chain recovers or "
                    "other evidence lifts the signal out of the weak tier."
                ),
            }
    expected_net_return = safe_float(opportunity.get("expected_net_return_pct"), 0.0)
    opportunity_score = safe_float(opportunity.get("score"), 0.0)
    min_score_required = safe_float(opportunity.get("min_score_required"), 0.95)
    profit_quality_ratio = safe_float(opportunity.get("profit_quality_ratio"), 0.0)
    loss_probability = safe_float(opportunity.get("server_profit_loss_probability"), 1.0)
    tail_risk_score = safe_float(opportunity.get("tail_risk_score"), 1.0)
    confidence = max(
        safe_float(decision.confidence, 0.0),
        safe_float(opportunity.get("confidence"), 0.0),
    )
    positive_net_probe_allowed = bool(
        not hard_block_reasons
        and expected_net_return >= _ENTRY_EVIDENCE_PARAMS.positive_net_probe_min_expected_pct
        and opportunity_score >= max(min_score_required - 0.55, 0.35)
        and confidence >= _ENTRY_EVIDENCE_PARAMS.positive_net_probe_min_confidence
        and profit_quality_ratio >= _ENTRY_EVIDENCE_PARAMS.positive_net_probe_min_profit_quality
        and loss_probability <= _ENTRY_EVIDENCE_PARAMS.positive_net_probe_max_loss_probability
        and tail_risk_score <= _ENTRY_EVIDENCE_PARAMS.positive_net_probe_max_tail_risk
        and not {"ml", "timeseries"}.issubset(set(strong_opposites))
        and not ("ml" in strong_opposites and "timeseries" in major_opposites)
    )
    aligned_support_count = len(set(aligned_support_sources))
    strong_positive_relief_allowed = bool(
        positive_net_probe_allowed
        and expected_net_return >= _ENTRY_EVIDENCE_PARAMS.strong_positive_relief_min_expected_pct
        and confidence >= _ENTRY_EVIDENCE_PARAMS.strong_positive_relief_min_confidence
        and profit_quality_ratio >= _ENTRY_EVIDENCE_PARAMS.strong_positive_relief_min_profit_quality
        and loss_probability <= _ENTRY_EVIDENCE_PARAMS.strong_positive_relief_max_loss_probability
        and tail_risk_score <= _ENTRY_EVIDENCE_PARAMS.strong_positive_relief_max_tail_risk
        and opportunity_score
        >= max(
            min_score_required + 1.0,
            _ENTRY_EVIDENCE_PARAMS.strong_positive_relief_min_opportunity_score,
        )
        and aligned_support_count
        >= _ENTRY_EVIDENCE_PARAMS.strong_positive_relief_min_aligned_sources
    )
    elite_positive_relief_allowed = bool(
        strong_positive_relief_allowed
        and expected_net_return >= _ENTRY_EVIDENCE_PARAMS.elite_positive_relief_min_expected_pct
        and confidence >= _ENTRY_EVIDENCE_PARAMS.elite_positive_relief_min_confidence
        and profit_quality_ratio >= _ENTRY_EVIDENCE_PARAMS.elite_positive_relief_min_profit_quality
        and loss_probability <= _ENTRY_EVIDENCE_PARAMS.elite_positive_relief_max_loss_probability
        and tail_risk_score <= _ENTRY_EVIDENCE_PARAMS.elite_positive_relief_max_tail_risk
        and opportunity_score
        >= max(
            min_score_required + 2.0,
            _ENTRY_EVIDENCE_PARAMS.elite_positive_relief_min_opportunity_score,
        )
    )
    memory_component = next(
        (item for item in components if item.get("source") == "shadow_memory"),
        {},
    )
    memory_review = (
        memory_component.get("review_feedback")
        if isinstance(memory_component.get("review_feedback"), dict)
        else {}
    )
    memory_missed_count = int(safe_float(memory_review.get("missed_opportunity_count"), 0.0))
    memory_score_adjustment = safe_float(memory_review.get("score_adjustment"), 0.0)
    memory_relief_allowed = bool(
        positive_net_probe_allowed
        and bool(memory_review.get("allow_probe"))
        and memory_missed_count >= 6
        and memory_score_adjustment >= 0.12
        and expected_net_return >= _ENTRY_EVIDENCE_PARAMS.positive_net_probe_min_expected_pct
        and profit_quality_ratio >= _ENTRY_EVIDENCE_PARAMS.positive_net_probe_min_profit_quality
        and loss_probability <= _ENTRY_EVIDENCE_PARAMS.positive_net_probe_max_loss_probability
        and tail_risk_score <= _ENTRY_EVIDENCE_PARAMS.positive_net_probe_max_tail_risk
        and "shadow_memory" not in major_opposites
        and not strong_opposites
    )
    if positive_net_probe_allowed:
        original_effective_score = effective_score
        if effective_score < ENTRY_EVIDENCE_SCORE_WEAK_PROBE:
            effective_score = ENTRY_EVIDENCE_SCORE_WEAK_PROBE
        positive_net_probe_relief = {
            "applied": True,
            "tradeable_probe": False,
            "shadow_only": True,
            "from_effective_score": round(original_effective_score, 6),
            "to_effective_score": round(effective_score, 6),
            "expected_net_return_pct": round(expected_net_return, 6),
            "opportunity_score": round(opportunity_score, 6),
            "profit_quality_ratio": round(profit_quality_ratio, 6),
            "loss_probability": round(loss_probability, 6),
            "tail_risk_score": round(tail_risk_score, 6),
            "reason": (
                "机会评分为正但仍处于弱冲突档；本轮只沉淀影子样本和复盘证据，"
                "不再提交微小真实/模拟订单。只有净收益、盈利质量、置信度和多源同向证据"
                "继续增强并抬升到 exploration/small 档后才允许执行。"
            ),
        }
    if memory_relief_allowed:
        original_effective_score = effective_score
        effective_score = max(effective_score, ENTRY_EVIDENCE_SCORE_PROBE)
        tradeable_probe = True
        memory_missed_opportunity_relief = {
            "applied": True,
            "tradeable_probe": True,
            "shadow_only": False,
            "from_effective_score": round(original_effective_score, 6),
            "to_effective_score": round(effective_score, 6),
            "missed_opportunity_count": memory_missed_count,
            "memory_score_adjustment": round(memory_score_adjustment, 6),
            "expected_net_return_pct": round(expected_net_return, 6),
            "profit_quality_ratio": round(profit_quality_ratio, 6),
            "loss_probability": round(loss_probability, 6),
            "tail_risk_score": round(tail_risk_score, 6),
            "reason": (
                "影子复盘多次证明观望错过同方向机会，且当前预期净收益、盈利质量、"
                "亏损概率和尾部风险达标；允许受控小仓质量试单，但不绕过硬风控。"
            ),
        }
    if strong_positive_relief_allowed:
        original_effective_score = effective_score
        target_score = (
            ENTRY_EVIDENCE_SCORE_SMALL
            if elite_positive_relief_allowed
            else ENTRY_EVIDENCE_SCORE_PROBE
        )
        effective_score = max(effective_score, target_score)
        tradeable_probe = True
        strong_positive_net_relief = {
            "applied": True,
            "tier_floor": "small" if elite_positive_relief_allowed else "exploration",
            "from_effective_score": round(original_effective_score, 6),
            "to_effective_score": round(effective_score, 6),
            "expected_net_return_pct": round(expected_net_return, 6),
            "opportunity_score": round(opportunity_score, 6),
            "profit_quality_ratio": round(profit_quality_ratio, 6),
            "loss_probability": round(loss_probability, 6),
            "tail_risk_score": round(tail_risk_score, 6),
            "confidence": round(confidence, 6),
            "aligned_support_sources": list(aligned_support_sources),
            "reason": (
                "净收益、盈利质量、置信度、亏损概率和尾部风险同时达标，"
                "且存在多源同向证据；该信号不再按弱冲突极小仓处理，"
                "但仍保留方向冲突、最大仓位和执行前行情复核。"
            ),
        }
    if effective_score < ENTRY_EVIDENCE_SCORE_HARD_BLOCK:
        advisory_wait_reasons.append("动态证据评分低于可交易底线，当前仅保留观望或极小探针")

    if effective_score >= ENTRY_EVIDENCE_SCORE_NORMAL:
        tier = "normal"
        size_multiplier = 1.0
    elif effective_score >= ENTRY_EVIDENCE_SCORE_MEDIUM:
        tier = "medium"
        size_multiplier = 0.60
    elif effective_score >= ENTRY_EVIDENCE_SCORE_SMALL:
        tier = "small"
        size_multiplier = 0.30
    elif effective_score >= ENTRY_EVIDENCE_SCORE_PROBE:
        tier = "exploration"
        size_multiplier = 0.10
    elif effective_score >= ENTRY_EVIDENCE_SCORE_WEAK_PROBE:
        tier = (
            "degraded_missing_probe"
            if missing_key_degraded_relief.get("applied")
            and not major_opposites
            and not weak_opposites
            else "weak_conflict_probe"
        )
        size_multiplier = 0.05
    else:
        tier = "blocked"
        size_multiplier = 0.0

    if (
        tier == "weak_conflict_probe"
        and not positive_net_probe_relief.get("applied")
        and len(aligned_support_sources) < ENTRY_EVIDENCE_WEAK_PROBE_MIN_ALIGNED_SOURCES
    ):
        advisory_wait_reasons.append("当前仅保留观望或极小探针，等待更多同向证据")
        advisory_wait_reasons.append(
            "weak conflict probe requires at least three aligned evidence sources"
        )
        tier = "blocked"
        size_multiplier = 0.0
    if entry_side == "short" and size_multiplier > 0:
        size_multiplier *= ENTRY_EVIDENCE_SHORT_SIZE_MULTIPLIER

    max_size_pct = None
    if tier == "exploration":
        max_size_pct = ENTRY_EVIDENCE_EXPLORATION_SIZE_CAP
    elif tier in {"weak_conflict_probe", "degraded_missing_probe"}:
        max_size_pct = ENTRY_EVIDENCE_WEAK_CONFLICT_SIZE_CAP
    elif missing_key_sources:
        max_size_pct = ENTRY_EVIDENCE_MISSING_KEY_SIZE_CAP
    if major_opposites and tier not in {"blocked", "exploration", "weak_conflict_probe"}:
        max_size_pct = min(
            max_size_pct if max_size_pct is not None else ENTRY_EVIDENCE_MAJOR_CONFLICT_SIZE_CAP,
            ENTRY_EVIDENCE_MAJOR_CONFLICT_SIZE_CAP,
        )
    if weak_opposites and tier not in {"blocked", "weak_conflict_probe"}:
        max_size_pct = min(
            max_size_pct if max_size_pct is not None else ENTRY_EVIDENCE_MAJOR_CONFLICT_SIZE_CAP,
            ENTRY_EVIDENCE_MAJOR_CONFLICT_SIZE_CAP,
        )

    return {
        "score": round(score, 6),
        "effective_score": round(effective_score, 6),
        "side": entry_side,
        "opposite_side": opposite,
        "tier": tier,
        "size_multiplier": round(size_multiplier, 6),
        "max_size_pct": round(max_size_pct, 6) if max_size_pct is not None else None,
        "hard_block": bool(hard_block_reasons),
        "hard_block_reasons": hard_block_reasons,
        "advisory_wait_reasons": advisory_wait_reasons,
        "major_opposites": major_opposites,
        "weak_opposites": weak_opposites,
        "strong_opposites": strong_opposites,
        "missing_key_sources": missing_key_sources,
        "aligned_support_sources": aligned_support_sources,
        "missing_key_degraded_relief": missing_key_degraded_relief,
        "positive_net_probe_relief": positive_net_probe_relief,
        "memory_missed_opportunity_relief": memory_missed_opportunity_relief,
        "strong_positive_net_relief": strong_positive_net_relief,
        "short_probe_relief": short_probe_relief,
        "tradeable_probe": bool(tradeable_probe),
        "shadow_only": bool(
            tier in {"weak_conflict_probe", "degraded_missing_probe"} and not tradeable_probe
        ),
        "components": components,
        "policy": (
            "硬风控只拦严重方向冲突和交易安全风险；"
            "模型服务缺失或证据不足先按观望/影子学习处理，不等同于方向错误；"
            "其余按 AI/ML/时序/情绪/影子记忆/server_profit/币种方向历史的动态证据分映射仓位。"
        ),
    }
