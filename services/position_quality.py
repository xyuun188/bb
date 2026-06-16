"""Quality scoring for open positions and release triage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.position_open_time import parse_position_time, position_hold_hours
from services.trading_params import ESTIMATED_TAKER_FEE_PCT


@dataclass(frozen=True, slots=True)
class PositionQualityScore:
    """A bounded score where lower values are better release candidates."""

    score: float
    bucket: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    hold_hours: float = 0.0
    pnl_ratio: float = 0.0
    notional: float = 0.0
    estimated_round_trip_fee: float = 0.0
    release_priority: float = 0.0

    @property
    def should_release(self) -> bool:
        return self.score < 48.0 or self.release_priority >= 72.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "bucket": self.bucket,
            "reasons": list(self.reasons),
            "hold_hours": round(self.hold_hours, 4),
            "pnl_ratio": round(self.pnl_ratio, 6),
            "notional": round(self.notional, 6),
            "estimated_round_trip_fee": round(self.estimated_round_trip_fee, 6),
            "release_priority": round(self.release_priority, 4),
            "should_release": self.should_release,
        }


class PositionQualityScorer:
    """Scores time cost, capital efficiency, fee drag, crowding, and reversal risk."""

    def score(
        self,
        position: dict[str, Any],
        *,
        feature_vector: Any | None = None,
        active_strategy_profile_id: str | None = None,
        same_symbol_side_parts: int = 1,
    ) -> PositionQualityScore:
        entry = _safe_float(position.get("entry_price"), 0.0)
        current = _safe_float(position.get("current_price"), entry)
        qty = abs(
            _safe_float(
                position.get("quantity") or position.get("contracts") or position.get("sz"),
                0.0,
            )
        )
        notional = _position_notional(position, qty, current if current > 0 else entry)
        if notional <= 0 and entry > 0 and qty > 0:
            notional = abs(entry * qty)
        reported_unrealized = _safe_float(position.get("unrealized_pnl"), 0.0)
        derived_unrealized = _derived_unrealized_pnl(
            position,
            quantity=qty,
            entry=entry,
            current=current,
        )
        unrealized = (
            derived_unrealized
            if abs(reported_unrealized) < 1e-9 and abs(derived_unrealized) > 1e-9
            else reported_unrealized
        )
        pnl_ratio = unrealized / max(notional, 1e-9)
        hold_hours = position_hold_hours(position)
        estimated_fee = max(notional * ESTIMATED_TAKER_FEE_PCT * 2.0, 0.0)

        score = 100.0
        reasons: list[str] = []

        if hold_hours >= 12 and abs(pnl_ratio) < 0.003:
            score -= 34.0
            reasons.append("time_cost_flat_12h")
        elif hold_hours >= 6 and abs(pnl_ratio) < 0.002:
            score -= 20.0
            reasons.append("time_cost_flat_6h")
        elif hold_hours >= 3 and abs(pnl_ratio) < 0.0012:
            score -= 12.0
            reasons.append("time_cost_flat_3h")

        if estimated_fee > 0:
            fee_multiple = abs(unrealized) / max(estimated_fee, 1e-9)
            if hold_hours >= 3 and abs(unrealized) < estimated_fee * 1.5:
                score -= 22.0
                reasons.append("fee_drag_dominates")
            elif hold_hours >= 1.5 and fee_multiple < 1.0:
                score -= 10.0
                reasons.append("fee_efficiency_weak")

        if unrealized < 0:
            if pnl_ratio <= -0.05:
                score -= 66.0
                reasons.append("severe_loss_pressure")
            elif pnl_ratio <= -0.025:
                score -= 44.0
                reasons.append("hard_loss_pressure")
            elif pnl_ratio <= -0.015:
                score -= 26.0
                reasons.append("loss_pressure")
            elif pnl_ratio <= -0.006:
                score -= 14.0
                reasons.append("loss_watch")
        elif unrealized > 0 and pnl_ratio >= 0.008:
            score += 8.0
            reasons.append("winner_has_edge")

        if same_symbol_side_parts >= 3:
            score -= 18.0
            reasons.append("fragmented_capital")
        elif same_symbol_side_parts == 2:
            score -= 8.0
            reasons.append("duplicate_symbol_side")

        margin = _safe_float(
            position.get("margin")
            or position.get("initial_margin")
            or position.get("initialMargin")
            or position.get("margin_used"),
            0.0,
        )
        if margin > 0 and hold_hours >= 4 and abs(unrealized) < max(margin * 0.003, estimated_fee):
            score -= 14.0
            reasons.append("capital_efficiency_low")

        strategy_profile_id = str(
            position.get("strategy_profile_id")
            or position.get("profile_id")
            or _nested(position, "strategy_learning_context", "strategy_profile_id")
            or ""
        ).strip()
        active_profile = str(active_strategy_profile_id or "").strip()
        if active_profile and strategy_profile_id and strategy_profile_id != active_profile:
            score -= 16.0
            reasons.append("old_strategy_profile")

        reversal = self._signal_reversal_score(position, feature_vector)
        if reversal >= 2:
            score -= 24.0
            reasons.append("signal_reversal")
        elif reversal == 1:
            score -= 12.0
            reasons.append("signal_reversal_watch")

        score = min(max(score, 0.0), 100.0)
        release_priority = min(max(100.0 - score, 0.0), 100.0)
        bucket = "high"
        if score < 35.0:
            bucket = "release_now"
        elif score < 48.0:
            bucket = "release_candidate"
        elif score < 68.0:
            bucket = "watch"

        return PositionQualityScore(
            score=score,
            bucket=bucket,
            reasons=tuple(dict.fromkeys(reasons)),
            hold_hours=hold_hours,
            pnl_ratio=pnl_ratio,
            notional=notional,
            estimated_round_trip_fee=estimated_fee,
            release_priority=release_priority,
        )

    def _signal_reversal_score(self, position: dict[str, Any], feature_vector: Any | None) -> int:
        if feature_vector is None:
            return 0
        side = str(position.get("side") or "").lower()
        if side not in {"long", "short"}:
            return 0
        returns_5 = _feature_float(feature_vector, "returns_5")
        returns_20 = _feature_float(feature_vector, "returns_20")
        macd_diff = _feature_float(feature_vector, "macd_diff")
        bb_pct = _feature_float(feature_vector, "bb_pct", 0.5)
        score = 0
        if side == "long":
            score += 1 if returns_5 < -0.006 or returns_20 < -0.012 else 0
            score += 1 if macd_diff < 0 and bb_pct < 0.35 else 0
        else:
            score += 1 if returns_5 > 0.006 or returns_20 > 0.012 else 0
            score += 1 if macd_diff > 0 and bb_pct > 0.65 else 0
        return score


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _hold_hours(opened_at: Any) -> float:
    opened = parse_position_time(opened_at)
    if opened is None:
        return 0.0
    return position_hold_hours({"created_at": opened})


def _feature_float(feature_vector: Any, name: str, default: float = 0.0) -> float:
    if isinstance(feature_vector, dict):
        return _safe_float(feature_vector.get(name), default)
    return _safe_float(getattr(feature_vector, name, default), default)


def _nested(source: dict[str, Any], *keys: str) -> Any:
    value: Any = source
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _position_notional(position: dict[str, Any], quantity: float, price: float) -> float:
    contract_size = _position_contract_size(position)
    info = position.get("info") if isinstance(position.get("info"), dict) else {}
    direct = _safe_float(
        position.get("notional")
        or position.get("notional_usd")
        or position.get("notionalUsd")
        or position.get("position_value")
        or info.get("notionalUsd")
        or info.get("notional")
        or info.get("posValue"),
        0.0,
    )
    if direct > 0:
        return abs(direct)
    return abs(quantity * max(price, 0.0) * (contract_size if contract_size > 0 else 1.0))


def _position_contract_size(position: dict[str, Any]) -> float:
    info = position.get("info") if isinstance(position.get("info"), dict) else {}
    contract_size = _safe_float(
        position.get("contract_size") or position.get("contractSize") or info.get("ctVal"),
        1.0,
    )
    return contract_size if contract_size > 0 else 1.0


def _derived_unrealized_pnl(
    position: dict[str, Any],
    *,
    quantity: float,
    entry: float,
    current: float,
) -> float:
    if quantity <= 0 or entry <= 0 or current <= 0:
        return 0.0
    side = str(position.get("side") or "").lower()
    contract_size = _position_contract_size(position)
    if side == "short":
        return (entry - current) * quantity * contract_size
    if side == "long":
        return (current - entry) * quantity * contract_size
    return 0.0
