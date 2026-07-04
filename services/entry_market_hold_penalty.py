"""Market-analysis hold memory and rotation penalty policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

NormalizeSymbol = Callable[[Any], str | None]
FeatureOpportunityScore = Callable[[Any], float]
FloatProvider = Callable[[], float]
DatetimeProvider = Callable[[], datetime]

MARKET_NO_OPPORTUNITY_WINDOW_MINUTES = 45.0
MARKET_NO_OPPORTUNITY_RECHECK_MINUTES = 18.0
MARKET_NO_OPPORTUNITY_STREAK_THRESHOLD = 2
MARKET_NO_OPPORTUNITY_MAX_PENALTY = 240.0
MARKET_RECENT_HOLD_DECAY_MINUTES = 45.0
MARKET_RECENT_HOLD_MAX_PENALTY = 42.0
MARKET_RECENT_LOSS_DEFAULT_DECAY_MINUTES = 75.0
MARKET_RECENT_LOSS_MAX_PENALTY = 180.0
MARKET_RECENT_ANALYSIS_DECAY_MINUTES = 30.0
MARKET_RECENT_ANALYSIS_DEDUPE_SECONDS = 75.0
MARKET_RECENT_ANALYSIS_MAX_PENALTY = 18.0
MARKET_NO_OPPORTUNITY_SCORE_RESET_DELTA = 28.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class EntryMarketHoldPenaltyPolicy:
    """Track recent AI holds and reduce stale repeat-analysis priority."""

    normalize_symbol: NormalizeSymbol
    feature_opportunity_score: FeatureOpportunityScore
    min_entry_volume_ratio_provider: FloatProvider
    min_entry_adx_provider: FloatProvider
    now_provider: DatetimeProvider = _utcnow
    recent_hold_symbols: dict[str, datetime] = field(default_factory=dict)
    recent_loss_symbols: dict[str, dict[str, Any]] = field(default_factory=dict)
    no_opportunity_symbols: dict[str, dict[str, Any]] = field(default_factory=dict)
    recent_analyzed_symbols: dict[str, datetime] = field(default_factory=dict)

    def remember_hold_symbol(
        self,
        symbol: str,
        feature_vector: Any | None = None,
        reason: str | None = None,
    ) -> None:
        normalized = self._normalized(symbol)
        if not normalized:
            return

        now = self.now_provider()
        self.recent_hold_symbols[normalized] = now
        self._prune_recent_holds(now)

        previous = self.no_opportunity_symbols.get(normalized) or {}
        first_seen = previous.get("first_seen_at")
        if not isinstance(first_seen, datetime) or first_seen < now - timedelta(
            minutes=MARKET_NO_OPPORTUNITY_WINDOW_MINUTES
        ):
            first_seen = now
            hold_count = 0
        else:
            hold_count = int(previous.get("hold_count") or 0)

        snapshot = self._feature_snapshot(feature_vector)
        self.no_opportunity_symbols[normalized] = {
            "first_seen_at": first_seen,
            "last_hold_at": now,
            "hold_count": hold_count + 1,
            "last_feature_score": (
                self.feature_opportunity_score(feature_vector)
                if feature_vector is not None
                else None
            ),
            "last_volume_ratio": snapshot["volume_ratio"],
            "last_returns_5": snapshot["returns_5"],
            "last_returns_20": snapshot["returns_20"],
            "last_price_vs_sma20": snapshot["price_vs_sma20"],
            "last_price_vs_sma50": snapshot["price_vs_sma50"],
            "last_adx_14": snapshot["adx_14"],
            "reason": str(reason or "")[:220],
        }
        self.prune_no_opportunity_symbols()

    def remember_analyzed_symbol(self, symbol: str) -> None:
        normalized = self._normalized(symbol)
        if not normalized:
            return

        now = self.now_provider()
        self.recent_analyzed_symbols[normalized] = now
        cutoff = now - timedelta(minutes=MARKET_RECENT_ANALYSIS_DECAY_MINUTES)
        self.recent_analyzed_symbols = {
            key: seen_at
            for key, seen_at in self.recent_analyzed_symbols.items()
            if seen_at >= cutoff
        }

    def prune_no_opportunity_symbols(self) -> None:
        if not self.no_opportunity_symbols:
            return
        cutoff = self.now_provider() - timedelta(minutes=MARKET_NO_OPPORTUNITY_WINDOW_MINUTES * 2)
        self.no_opportunity_symbols = {
            symbol: state
            for symbol, state in self.no_opportunity_symbols.items()
            if isinstance(state.get("last_hold_at"), datetime) and state["last_hold_at"] >= cutoff
        }

    def clear_symbol(self, symbol: str) -> None:
        normalized = self._normalized(symbol)
        if normalized:
            self.no_opportunity_symbols.pop(normalized, None)
            self.recent_hold_symbols.pop(normalized, None)
            self.recent_loss_symbols.pop(normalized, None)

    def sync_recent_loss_profiles(self, profiles: dict[str, Any] | None) -> None:
        if not isinstance(profiles, dict):
            return
        now = self.now_provider()
        active_symbols: set[str] = set()
        for key, profile in profiles.items():
            if not isinstance(profile, dict):
                continue
            symbol = self._symbol_from_profile_key(str(key or ""))
            normalized = self._normalized(symbol)
            if not normalized or not self._profile_has_recent_loss_pressure(profile):
                continue
            remaining_minutes = self._profile_cooldown_remaining_minutes(profile)
            loss = abs(_safe_float(profile.get("loss"), 0.0))
            today_loss = abs(_safe_float(profile.get("today_loss"), 0.0))
            largest_loss = abs(_safe_float(profile.get("largest_loss"), 0.0))
            severity = min(max(loss + today_loss + largest_loss, 0.0) / 5.0, 1.0)
            self.recent_loss_symbols[normalized] = {
                "last_seen_at": now,
                "expires_at": now + timedelta(minutes=remaining_minutes),
                "penalty": MARKET_RECENT_LOSS_MAX_PENALTY * max(severity, 0.35),
                "reason": str(profile.get("cooldown_reason") or "")[:220],
                "loss": loss,
                "today_loss": today_loss,
                "largest_loss": largest_loss,
            }
            active_symbols.add(normalized)
        self.recent_loss_symbols = {
            symbol: state
            for symbol, state in self.recent_loss_symbols.items()
            if symbol in active_symbols or self._recent_loss_state_active(state, now)
        }

    def recent_hold_penalty(self, symbol: str) -> float:
        normalized = self._normalized(symbol)
        seen_at = self.recent_hold_symbols.get(normalized or "")
        loss_penalty = self.recent_loss_penalty(symbol)
        if not seen_at:
            return loss_penalty
        age_minutes = (self.now_provider() - seen_at).total_seconds() / 60.0
        if age_minutes >= MARKET_RECENT_HOLD_DECAY_MINUTES:
            return loss_penalty
        hold_penalty = max(
            0.0,
            MARKET_RECENT_HOLD_MAX_PENALTY * (1.0 - age_minutes / MARKET_RECENT_HOLD_DECAY_MINUTES),
        )
        return max(hold_penalty, loss_penalty)

    def recent_loss_penalty(self, symbol: str) -> float:
        normalized = self._normalized(symbol)
        state = self.recent_loss_symbols.get(normalized or "")
        if not isinstance(state, dict):
            return 0.0
        now = self.now_provider()
        if not self._recent_loss_state_active(state, now):
            self.recent_loss_symbols.pop(normalized or "", None)
            return 0.0
        expires_at = state.get("expires_at")
        last_seen_at = state.get("last_seen_at")
        if not isinstance(expires_at, datetime) or not isinstance(last_seen_at, datetime):
            return 0.0
        total_seconds = max((expires_at - last_seen_at).total_seconds(), 1.0)
        remaining_seconds = max((expires_at - now).total_seconds(), 0.0)
        decay = min(max(remaining_seconds / total_seconds, 0.0), 1.0)
        return max(_safe_float(state.get("penalty"), 0.0), 0.0) * decay

    def recent_analysis_penalty(self, symbol: str) -> float:
        normalized = self._normalized(symbol)
        seen_at = self.recent_analyzed_symbols.get(normalized or "")
        if not seen_at:
            return 0.0
        age_minutes = (self.now_provider() - seen_at).total_seconds() / 60.0
        if age_minutes >= MARKET_RECENT_ANALYSIS_DECAY_MINUTES:
            return 0.0
        return max(
            0.0,
            MARKET_RECENT_ANALYSIS_MAX_PENALTY
            * (1.0 - age_minutes / MARKET_RECENT_ANALYSIS_DECAY_MINUTES),
        )

    def recently_analyzed(
        self,
        symbol: str,
        min_interval_seconds: float = MARKET_RECENT_ANALYSIS_DEDUPE_SECONDS,
    ) -> bool:
        normalized = self._normalized(symbol)
        seen_at = self.recent_analyzed_symbols.get(normalized or "")
        if not seen_at:
            return False
        age_seconds = (self.now_provider() - seen_at).total_seconds()
        return 0.0 <= age_seconds < max(float(min_interval_seconds), 0.0)

    def no_opportunity_rotation_penalty(
        self,
        symbol: str,
        feature_vector: Any | None = None,
    ) -> float:
        normalized = self._normalized(symbol)
        if not normalized:
            return 0.0

        state = self.no_opportunity_symbols.get(normalized)
        if not state:
            return 0.0

        last_hold_at = state.get("last_hold_at")
        if not isinstance(last_hold_at, datetime):
            self.no_opportunity_symbols.pop(normalized, None)
            return 0.0

        now = self.now_provider()
        age_minutes = (now - last_hold_at).total_seconds() / 60.0
        if age_minutes >= MARKET_NO_OPPORTUNITY_WINDOW_MINUTES:
            self.no_opportunity_symbols.pop(normalized, None)
            return 0.0

        hold_count = max(0, int(state.get("hold_count") or 0))
        if hold_count < MARKET_NO_OPPORTUNITY_STREAK_THRESHOLD:
            return 0.0

        snapshot = self._feature_snapshot(feature_vector)
        previous_score = _safe_float(state.get("last_feature_score"), 0.0)
        current_score = (
            self.feature_opportunity_score(feature_vector) if feature_vector is not None else 0.0
        )
        if self._opportunity_improved(current_score, previous_score, snapshot, state):
            self.clear_symbol(normalized)
            return 0.0

        if age_minutes >= MARKET_NO_OPPORTUNITY_RECHECK_MINUTES:
            decay = max(
                0.0,
                1.0
                - (age_minutes - MARKET_NO_OPPORTUNITY_RECHECK_MINUTES)
                / max(
                    MARKET_NO_OPPORTUNITY_WINDOW_MINUTES - MARKET_NO_OPPORTUNITY_RECHECK_MINUTES,
                    1.0,
                ),
            )
        else:
            decay = 1.0

        streak_multiplier = min(
            1.0,
            (hold_count - MARKET_NO_OPPORTUNITY_STREAK_THRESHOLD + 1) / 4.0,
        )
        return MARKET_NO_OPPORTUNITY_MAX_PENALTY * streak_multiplier * decay

    def _opportunity_improved(
        self,
        current_score: float,
        previous_score: float,
        snapshot: dict[str, float],
        state: dict[str, Any],
    ) -> bool:
        raw_returns_5 = snapshot["returns_5"]
        raw_returns_20 = snapshot["returns_20"]
        returns_5 = abs(raw_returns_5)
        returns_20 = abs(raw_returns_20)
        volume_ratio = snapshot["volume_ratio"]
        adx_14 = snapshot["adx_14"]
        price_vs_sma20 = snapshot["price_vs_sma20"]

        previous_volume_ratio = _safe_float(state.get("last_volume_ratio"), 0.0)
        previous_returns_5 = _safe_float(state.get("last_returns_5"), 0.0)
        previous_returns_20 = _safe_float(state.get("last_returns_20"), 0.0)
        previous_price_vs_sma20 = _safe_float(state.get("last_price_vs_sma20"), 0.0)

        volume_regime_changed = bool(
            previous_volume_ratio > 0
            and volume_ratio >= max(previous_volume_ratio * 1.70, previous_volume_ratio + 0.35)
        )
        momentum_regime_changed = bool(
            (
                raw_returns_5 * previous_returns_5 < 0
                and abs(raw_returns_5) >= 0.003
                and abs(previous_returns_5) >= 0.001
            )
            or (
                raw_returns_20 * previous_returns_20 < 0
                and abs(raw_returns_20) >= 0.006
                and abs(previous_returns_20) >= 0.002
            )
        )
        moving_average_crossed = bool(
            price_vs_sma20 * previous_price_vs_sma20 < 0 and abs(price_vs_sma20) >= 0.002
        )
        min_volume_ratio = _safe_float(self.min_entry_volume_ratio_provider(), 0.0)
        min_adx = _safe_float(self.min_entry_adx_provider(), 0.0)
        return (
            current_score >= previous_score + MARKET_NO_OPPORTUNITY_SCORE_RESET_DELTA
            or volume_regime_changed
            or momentum_regime_changed
            or moving_average_crossed
            or (
                volume_ratio >= max(min_volume_ratio, 0.8)
                and (returns_5 >= 0.004 or returns_20 >= 0.010)
            )
            or (adx_14 >= max(min_adx, 24.0) and (returns_5 >= 0.003 or returns_20 >= 0.008))
        )

    def _prune_recent_holds(self, now: datetime) -> None:
        cutoff = now - timedelta(minutes=MARKET_RECENT_HOLD_DECAY_MINUTES)
        self.recent_hold_symbols = {
            key: seen_at for key, seen_at in self.recent_hold_symbols.items() if seen_at >= cutoff
        }

    def _normalized(self, symbol: Any) -> str | None:
        normalized = self.normalize_symbol(symbol)
        if normalized is None:
            return None
        return str(normalized)

    @staticmethod
    def _feature_snapshot(feature_vector: Any | None) -> dict[str, float]:
        if feature_vector is None:
            return {
                "volume_ratio": 0.0,
                "returns_5": 0.0,
                "returns_20": 0.0,
                "price_vs_sma20": 0.0,
                "price_vs_sma50": 0.0,
                "adx_14": 0.0,
            }
        return {
            "volume_ratio": _safe_float(getattr(feature_vector, "volume_ratio", 0.0), 0.0),
            "returns_5": _safe_float(getattr(feature_vector, "returns_5", 0.0), 0.0),
            "returns_20": _safe_float(getattr(feature_vector, "returns_20", 0.0), 0.0),
            "price_vs_sma20": _safe_float(
                getattr(feature_vector, "price_vs_sma20", 0.0),
                0.0,
            ),
            "price_vs_sma50": _safe_float(
                getattr(feature_vector, "price_vs_sma50", 0.0),
                0.0,
            ),
            "adx_14": _safe_float(getattr(feature_vector, "adx_14", 0.0), 0.0),
        }

    @staticmethod
    def _symbol_from_profile_key(key: str) -> str:
        if "|" not in key:
            return key.strip()
        symbol, _side = key.split("|", 1)
        return symbol.strip()

    @staticmethod
    def _profile_has_recent_loss_pressure(profile: dict[str, Any]) -> bool:
        if not bool(profile.get("cooldown")):
            return False
        if _safe_float(profile.get("pnl"), 0.0) < 0:
            return True
        return any(
            _safe_float(profile.get(field), 0.0) > 0.0
            for field in ("loss", "today_loss", "largest_loss")
        )

    @staticmethod
    def _profile_cooldown_remaining_minutes(profile: dict[str, Any]) -> float:
        remaining_hours = _safe_float(profile.get("cooldown_remaining_hours"), 0.0)
        if remaining_hours > 0:
            return max(remaining_hours * 60.0, 1.0)
        last_loss_age_hours = _safe_float(profile.get("last_loss_age_hours"), 0.0)
        remaining = MARKET_RECENT_LOSS_DEFAULT_DECAY_MINUTES - max(
            last_loss_age_hours * 60.0,
            0.0,
        )
        return max(remaining, 1.0)

    @staticmethod
    def _recent_loss_state_active(state: dict[str, Any], now: datetime) -> bool:
        expires_at = state.get("expires_at")
        return isinstance(expires_at, datetime) and expires_at > now
