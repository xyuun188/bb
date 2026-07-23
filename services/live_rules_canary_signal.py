"""Deterministic market-direction contract for live rules-canary entries."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from math import isfinite
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.production_trade_gate import validate_production_trade_gate

LIVE_RULES_CANARY_SIGNAL_VERSION = "2026-07-23.live-rules-canary-signal.v1"
_DIRECTION_FIELDS = (
    "returns_1",
    "returns_5",
    "returns_20",
    "macd_diff",
    "price_vs_sma20",
    "price_vs_sma50",
)
_MIN_DIRECTION_VOTES = 4
_MIN_DOMINANT_VOTES = 4
_MIN_CONSENSUS_RATIO = 0.75


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _model_shadow_decision(decision: DecisionOutput) -> dict[str, Any]:
    return {
        "action": decision.action.value,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "position_size_pct": decision.position_size_pct,
        "suggested_leverage": decision.suggested_leverage,
        "stop_loss_pct": decision.stop_loss_pct,
        "take_profit_pct": decision.take_profit_pct,
        "suggested_holding_minutes": decision.suggested_holding_minutes,
        "maximum_holding_minutes": decision.maximum_holding_minutes,
        "observation_only": True,
        "can_authorize_entry": False,
        "can_change_size_or_leverage": False,
    }


def build_live_rules_canary_signal(
    feature_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    """Choose a side only when current technical directions strongly agree."""

    snapshot = _safe_dict(feature_snapshot)
    votes: dict[str, str] = {}
    inputs: dict[str, float] = {}
    for field in _DIRECTION_FIELDS:
        value = _finite_float(snapshot.get(field))
        if value is None or value == 0:
            continue
        inputs[field] = value
        votes[field] = "long" if value > 0 else "short"

    long_votes = sum(side == "long" for side in votes.values())
    short_votes = sum(side == "short" for side in votes.values())
    valid_vote_count = len(votes)
    dominant_side = (
        "long"
        if long_votes > short_votes
        else "short"
        if short_votes > long_votes
        else "hold"
    )
    dominant_votes = max(long_votes, short_votes)
    consensus_ratio = (
        dominant_votes / valid_vote_count if valid_vote_count > 0 else 0.0
    )
    current_price = _finite_float(
        snapshot.get("current_price", snapshot.get("close"))
    )
    volatility = max(
        _finite_float(snapshot.get("volatility_20")) or 0.0,
        0.0,
    )
    atr = max(_finite_float(snapshot.get("atr_14")) or 0.0, 0.0)
    blockers: list[str] = []
    if current_price is None or current_price <= 0:
        blockers.append("rules_canary_signal_price_missing")
    if atr <= 0 and volatility <= 0:
        blockers.append("rules_canary_signal_risk_anchor_missing")
    if valid_vote_count < _MIN_DIRECTION_VOTES:
        blockers.append("rules_canary_signal_votes_insufficient")
    if dominant_votes < _MIN_DOMINANT_VOTES:
        blockers.append("rules_canary_signal_consensus_weak")
    if consensus_ratio < _MIN_CONSENSUS_RATIO:
        blockers.append("rules_canary_signal_consensus_weak")
    if dominant_side == "hold":
        blockers.append("rules_canary_signal_direction_tied")

    blockers = list(dict.fromkeys(blockers))
    eligible = not blockers
    action = dominant_side if eligible else "hold"
    generated_at = datetime.now(UTC).isoformat()
    policy_inputs = {
        "direction_inputs": inputs,
        "votes": votes,
        "current_price": current_price,
        "atr_14": atr,
        "volatility_20": volatility,
        "minimum_direction_votes": _MIN_DIRECTION_VOTES,
        "minimum_dominant_votes": _MIN_DOMINANT_VOTES,
        "minimum_consensus_ratio": _MIN_CONSENSUS_RATIO,
    }
    return {
        "version": LIVE_RULES_CANARY_SIGNAL_VERSION,
        "execution_scope": "live_rules_canary",
        "decision_authority": "rules",
        "model_can_influence": False,
        "production_eligible": eligible,
        "action": action,
        "score": round(consensus_ratio, 8) if eligible else 0.0,
        "long_votes": long_votes,
        "short_votes": short_votes,
        "valid_vote_count": valid_vote_count,
        "consensus_ratio": round(consensus_ratio, 8),
        "direction_inputs": inputs,
        "direction_votes": votes,
        "blockers": blockers,
        "reason": "rules_canary_technical_consensus_ready"
        if eligible
        else ",".join(blockers),
        "policy_provenance": {
            "source": "current_market_technical_feature_consensus",
            "observation_window": "current_market_feature_snapshot",
            "sample_count": valid_vote_count,
            "generated_at": generated_at,
            "strategy_version": LIVE_RULES_CANARY_SIGNAL_VERSION,
            "fallback_reason": "" if eligible else ",".join(blockers),
            "input_fingerprint": _fingerprint(policy_inputs),
        },
    }


def apply_live_rules_canary_signal(
    decision: DecisionOutput,
    gate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Replace model execution fields when the authoritative gate selects rules."""

    gate_validation = validate_production_trade_gate(
        gate,
        required_mode="live_rules_canary",
    )
    if not gate_validation.valid:
        return None
    trade_gate = gate_validation.gate

    raw = dict(_safe_dict(decision.raw_response))
    raw["production_trade_gate"] = trade_gate
    raw["model_shadow_decision"] = _model_shadow_decision(decision)
    signal = build_live_rules_canary_signal(decision.feature_snapshot)
    raw["live_rules_canary_signal"] = signal
    decision.raw_response = raw
    decision.action = (
        Action.LONG
        if signal["action"] == "long"
        else Action.SHORT
        if signal["action"] == "short"
        else Action.HOLD
    )
    decision.confidence = float(signal["score"])
    decision.reasoning = str(signal["reason"])
    decision.position_size_pct = 0.0
    decision.suggested_leverage = 1.0
    decision.stop_loss_pct = 0.0
    decision.take_profit_pct = 0.0
    decision.suggested_holding_minutes = 0.0
    decision.maximum_holding_minutes = 0.0
    decision.cross_check_for = None
    return signal
