"""Time-decayed symbol winner/loser adjustment for entry scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from services.trading_params import DEFAULT_TRADING_PARAMS, EntryWinnerParams


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


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class EntrySymbolWinnerAdjustment:
    tier: str
    reason: str
    score_adjustment: float
    min_score_required: float
    side_decay_weight: float
    symbol_decay_weight: float
    side_effective_pnl: float
    symbol_effective_pnl: float
    side_age_days: float | None
    symbol_age_days: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "reason": self.reason,
            "score_adjustment": round(self.score_adjustment, 6),
            "min_score_required": round(self.min_score_required, 6),
            "side_decay_weight": round(self.side_decay_weight, 6),
            "symbol_decay_weight": round(self.symbol_decay_weight, 6),
            "side_effective_pnl": round(self.side_effective_pnl, 6),
            "symbol_effective_pnl": round(self.symbol_effective_pnl, 6),
            "side_age_days": (
                round(self.side_age_days, 6) if self.side_age_days is not None else None
            ),
            "symbol_age_days": (
                round(self.symbol_age_days, 6) if self.symbol_age_days is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class _DecayedProfile:
    profile: dict[str, Any]
    pnl: float
    effective_pnl: float
    decay_weight: float
    age_days: float | None
    count: int
    profit_factor: float


class EntrySymbolWinnerDecayPolicy:
    """Apply recency-aware historical winner relief without overfitting stale PnL."""

    def __init__(
        self,
        params: EntryWinnerParams | None = None,
        *,
        clock: Any | None = None,
    ) -> None:
        self.params = params or DEFAULT_TRADING_PARAMS.entry_winner
        self.clock = clock or (lambda: datetime.now(UTC))

    def evaluate(
        self,
        *,
        side: str,
        side_profile: dict[str, Any],
        symbol_profile: dict[str, Any],
        base_min_score_required: float,
        current_min_score_required: float,
        side_loss: float,
        side_profit: float,
        side_losses: int,
    ) -> EntrySymbolWinnerAdjustment:
        side_decayed = self._decayed(side_profile)
        symbol_decayed = self._decayed(symbol_profile)
        side_label = "做多" if side == "long" else "做空"

        tier = "neutral"
        reason = "该币种/方向近期真实盈亏样本不足，按中性处理。"
        score_adjustment = 0.0
        min_score_required = current_min_score_required

        if self._is_winner(side_decayed):
            tier = "side_winner"
            score_adjustment = min(
                side_decayed.effective_pnl / 28.0,
                self.params.score_bonus_cap,
            )
            min_score_required = max(
                min(
                    current_min_score_required,
                    base_min_score_required - self.params.score_relief * side_decayed.decay_weight,
                ),
                0.68,
            )
            reason = (
                f"该币种{side_label}方向近期真实盈利 {side_decayed.pnl:.2f}U，"
                f"时间衰减后有效盈利 {side_decayed.effective_pnl:.2f}U，"
                f"盈利因子 {side_decayed.profit_factor:.2f}，按近期赢家轻微放宽。"
            )
        elif self._is_winner(symbol_decayed):
            tier = "symbol_winner"
            score_adjustment = min(
                symbol_decayed.effective_pnl / 44.0,
                self.params.score_bonus_cap * 0.65,
            )
            min_score_required = max(
                min(
                    current_min_score_required,
                    base_min_score_required
                    - self.params.score_relief * 0.5 * symbol_decayed.decay_weight,
                ),
                0.72,
            )
            reason = (
                f"该币种近期真实盈利 {symbol_decayed.pnl:.2f}U，"
                f"时间衰减后有效盈利 {symbol_decayed.effective_pnl:.2f}U，"
                f"盈利因子 {symbol_decayed.profit_factor:.2f}；方向仍按当前证据判断，只给轻微优先级。"
            )
        elif (
            side_decayed.count >= 2
            and side_decayed.pnl < 0
            and (side_loss > side_profit * 1.15 or side_decayed.profit_factor < 0.80)
        ):
            tier = "side_loser"
            score_adjustment = -min(abs(side_decayed.pnl) / 32.0 + side_losses * 0.05, 0.75)
            reason = (
                f"该币种{side_label}方向近期真实净亏 {side_decayed.pnl:.2f}U，"
                "不永久禁用，但降低评分和仓位，等待新证据证明值得再试。"
            )

        return EntrySymbolWinnerAdjustment(
            tier=tier,
            reason=reason,
            score_adjustment=score_adjustment,
            min_score_required=min_score_required,
            side_decay_weight=side_decayed.decay_weight,
            symbol_decay_weight=symbol_decayed.decay_weight,
            side_effective_pnl=side_decayed.effective_pnl,
            symbol_effective_pnl=symbol_decayed.effective_pnl,
            side_age_days=side_decayed.age_days,
            symbol_age_days=symbol_decayed.age_days,
        )

    def _is_winner(self, decayed: _DecayedProfile) -> bool:
        return bool(
            decayed.count >= self.params.min_count
            and decayed.effective_pnl >= self.params.min_pnl_usdt
            and decayed.profit_factor >= self.params.min_profit_factor
            and decayed.decay_weight > 0.0
        )

    def _decayed(self, profile: dict[str, Any]) -> _DecayedProfile:
        if not isinstance(profile, dict):
            profile = {}
        pnl = _safe_float(profile.get("pnl"), 0.0)
        count = _safe_int(profile.get("count"), 0)
        profit_factor = _safe_float(profile.get("profit_factor"), 1.0)
        age_days = self._profile_age_days(profile)
        decay_weight = self._decay_weight(age_days, count)
        return _DecayedProfile(
            profile=profile,
            pnl=pnl,
            effective_pnl=pnl * decay_weight if pnl > 0 else pnl,
            decay_weight=decay_weight,
            age_days=age_days,
            count=count,
            profit_factor=profit_factor,
        )

    def _profile_age_days(self, profile: dict[str, Any]) -> float | None:
        last_closed = _parse_datetime(profile.get("last_closed_at"))
        if last_closed is None:
            return None
        now = self.clock()
        now = now if now.tzinfo else now.replace(tzinfo=UTC)
        return max((now.astimezone(UTC) - last_closed.astimezone(UTC)).total_seconds(), 0.0) / (
            24.0 * 3600.0
        )

    def _decay_weight(self, age_days: float | None, count: int) -> float:
        if count <= 0:
            return 0.0
        if age_days is None:
            return max(min(self.params.missing_timestamp_weight, 1.0), 0.0)
        if age_days > self.params.max_age_days:
            return 0.0
        half_life = max(self.params.half_life_days, 0.1)
        return max(min(0.5 ** (age_days / half_life), 1.0), 0.0)
