"""Auto-scan feature ranking for entry candidates."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from services.entry_wick_guard import (
    ABNORMAL_WICK_ENTRY_BLOCK_MAX_PCT,
    ABNORMAL_WICK_ENTRY_BLOCK_MIN_COUNT,
    ABNORMAL_WICK_ENTRY_BLOCK_RECENT_HOURS,
)
from services.trading_params import DEFAULT_TRADING_PARAMS

SuspiciousSymbolReason = Callable[[str], str | None]
FloatProvider = Callable[[], float]
PenaltyProvider = Callable[[str], float]
RotationPenaltyProvider = Callable[[str, Any], float]


def _feature_float(feature: Any, key: str, default: float = 0.0) -> float:
    try:
        return float(getattr(feature, key, default) or default)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class EntryFeatureRankResult:
    selected: dict[str, Any]
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EntryFeatureRankerPolicy:
    """Rank symbols after K-line indicators are available, before spending AI tokens."""

    suspicious_symbol_reason: SuspiciousSymbolReason
    min_entry_volume_ratio_provider: FloatProvider
    min_entry_adx_provider: FloatProvider
    major_symbols: frozenset[str]
    params: Any = DEFAULT_TRADING_PARAMS.entry_feature_ranker

    def feature_opportunity_score(self, feature: Any) -> float:
        params = self.params
        try:
            volume_24h = float(getattr(feature, "volume_24h", 0) or 0)
            volume_ratio = float(getattr(feature, "volume_ratio", 0) or 0)
            adx_14 = float(getattr(feature, "adx_14", 0) or 0)
            returns_1 = abs(float(getattr(feature, "returns_1", 0) or 0))
            returns_5 = abs(float(getattr(feature, "returns_5", 0) or 0))
            returns_20 = abs(float(getattr(feature, "returns_20", 0) or 0))
            volatility_20 = float(getattr(feature, "volatility_20", 0) or 0)
            change_24h = abs(float(getattr(feature, "change_24h_pct", 0) or 0))
            bb_pct = float(getattr(feature, "bb_pct", 0.5) or 0.5)
            price_vs_sma20 = abs(float(getattr(feature, "price_vs_sma20", 0) or 0))
            price_vs_sma50 = abs(float(getattr(feature, "price_vs_sma50", 0) or 0))
            current_price = float(
                getattr(feature, "current_price", 0) or getattr(feature, "close", 0) or 0
            )
        except (TypeError, ValueError):
            return 0.0

        notional_24h = max(volume_24h * max(current_price, 0.0), 0.0)
        liquidity = math.log10(notional_24h + 1.0) * params.liquidity_log_weight
        participation = (
            min(max(volume_ratio, 0.0), params.volume_ratio_score_cap) * params.participation_weight
        )
        trend_quality = min(max(adx_14, 0.0), params.adx_score_cap) * params.adx_weight
        momentum = min(
            (returns_1 * params.momentum_returns_1_weight)
            + (returns_5 * params.momentum_returns_5_weight)
            + (returns_20 * params.momentum_returns_20_weight),
            params.momentum_score_cap,
        )
        day_move = min(change_24h, params.day_move_cap_pct) * params.day_move_weight
        volatility_bonus = min(
            max(volatility_20, 0.0) * params.volatility_weight,
            params.volatility_score_cap,
        )
        trend_distance = min(
            (price_vs_sma20 + price_vs_sma50) * params.trend_distance_weight,
            params.trend_distance_cap,
        )
        band_bonus = (
            params.bollinger_extreme_bonus
            if bb_pct <= params.bollinger_extreme_low or bb_pct >= params.bollinger_extreme_high
            else 0.0
        )
        low_activity_penalty = (
            params.low_activity_penalty
            if volume_ratio < self.min_entry_volume_ratio_provider()
            else 0.0
        )
        extreme_vol_penalty = (
            params.extreme_volatility_penalty
            if volatility_20 > params.extreme_volatility_threshold
            and change_24h > params.extreme_volatility_day_move_pct
            else (
                params.elevated_volatility_penalty
                if volatility_20 > params.elevated_volatility_threshold
                else 0.0
            )
        )

        return (
            liquidity
            + participation
            + trend_quality
            + momentum
            + day_move
            + volatility_bonus
            + trend_distance
            + band_bonus
            - low_activity_penalty
            - extreme_vol_penalty
        )

    def is_auto_tradeable_feature(self, feature: Any) -> bool:
        params = self.params
        parsed = self._parse_filter_inputs(feature)
        if parsed is None:
            return False
        symbol, current_price, volume_24h, volume_ratio, volatility_20, change_24h, adx_14 = parsed
        if self._has_recent_abnormal_wick(feature):
            return False

        notional_24h = current_price * volume_24h
        min_notional = (
            params.tradable_major_min_notional_usdt
            if symbol in self.major_symbols
            else params.tradable_alt_min_notional_usdt
        )
        analysis_volume_floor = max(
            min(
                max(
                    float(self.min_entry_volume_ratio_provider() or 0.0),
                    params.tradable_volume_provider_floor,
                )
                * params.tradable_volume_multiplier,
                params.tradable_volume_cap,
            ),
            params.tradable_volume_floor,
        )
        analysis_adx_floor = max(
            min(
                max(
                    float(self.min_entry_adx_provider() or 0.0)
                    - params.tradable_adx_provider_offset,
                    params.tradable_adx_provider_floor,
                ),
                params.tradable_adx_cap,
            ),
            params.tradable_adx_floor,
        )
        if volume_ratio < analysis_volume_floor:
            return False
        if notional_24h < min_notional:
            return False
        if volatility_20 > params.tradable_max_volatility:
            return False
        if change_24h > params.tradable_max_day_change_pct:
            return False
        return not (symbol not in self.major_symbols and adx_14 < analysis_adx_floor)

    def is_auto_analysis_candidate_feature(self, feature: Any) -> bool:
        params = self.params
        parsed = self._parse_filter_inputs(feature)
        if parsed is None:
            return False
        symbol, current_price, volume_24h, volume_ratio, volatility_20, change_24h, adx_14 = parsed
        if self._has_recent_abnormal_wick(feature):
            return False

        notional_24h = current_price * volume_24h
        min_notional = (
            params.analysis_major_min_notional_usdt
            if symbol in self.major_symbols
            else params.analysis_alt_min_notional_usdt
        )
        soft_volume_floor = max(
            min(
                max(
                    float(self.min_entry_volume_ratio_provider() or 0.0),
                    params.analysis_volume_provider_floor,
                )
                * params.analysis_volume_multiplier,
                params.analysis_volume_cap,
            ),
            params.analysis_volume_floor,
        )
        soft_adx_floor = max(
            min(
                max(
                    float(self.min_entry_adx_provider() or 0.0)
                    - params.analysis_adx_provider_offset,
                    params.analysis_adx_provider_floor,
                ),
                params.analysis_adx_cap,
            ),
            params.analysis_adx_floor,
        )
        if volume_ratio < soft_volume_floor:
            return False
        if notional_24h < min_notional:
            return False
        if volatility_20 > params.analysis_max_volatility:
            return False
        if change_24h > params.analysis_max_day_change_pct:
            return False
        return not (symbol not in self.major_symbols and adx_14 < soft_adx_floor)

    def rank(
        self,
        feature_vectors: dict[str, Any],
        limit: int,
        *,
        recent_hold_penalty: PenaltyProvider,
        recent_analysis_penalty: PenaltyProvider,
        no_opportunity_rotation_penalty: RotationPenaltyProvider,
    ) -> EntryFeatureRankResult:
        all_items = list(feature_vectors.items())
        tradable_items = [
            item for item in feature_vectors.items() if self.is_auto_tradeable_feature(item[1])
        ]
        soft_items = [
            item
            for item in all_items
            if item not in tradable_items and self.is_auto_analysis_candidate_feature(item[1])
        ]

        def ranking_score(item: tuple[str, Any]) -> float:
            symbol, feature = item
            rotation_penalty = no_opportunity_rotation_penalty(symbol, feature)
            return (
                self.feature_opportunity_score(feature)
                - recent_hold_penalty(symbol)
                - recent_analysis_penalty(symbol)
                - rotation_penalty
            )

        tradable_symbols = {symbol for symbol, _ in tradable_items}
        ranked_tradable = sorted(
            tradable_items,
            key=lambda item: (ranking_score(item),),
            reverse=True,
        )
        ranked_soft = sorted(
            soft_items,
            key=ranking_score,
            reverse=True,
        )

        selected_items = list(ranked_tradable[:limit])
        if len(selected_items) < limit:
            selected_items.extend(ranked_soft[: max(limit - len(selected_items), 0)])
        if not selected_items:
            selected_items = sorted(all_items, key=ranking_score, reverse=True)[:limit]

        selected = dict(selected_items)
        diagnostics = {
            "selected": len(selected),
            "candidates": len(feature_vectors),
            "tradable_candidates": len(tradable_items),
            "secondary_candidates": len(soft_items),
            "symbols": [
                {
                    "symbol": symbol,
                    "score": round(self.feature_opportunity_score(feature), 2),
                    "recent_hold_penalty": round(recent_hold_penalty(symbol), 2),
                    "recent_analysis_penalty": round(recent_analysis_penalty(symbol), 2),
                    "rotation_penalty": round(no_opportunity_rotation_penalty(symbol, feature), 2),
                    "selection_tier": (
                        "hard_filter" if symbol in tradable_symbols else "secondary_fill"
                    ),
                    "volume_ratio": round(_feature_float(feature, "volume_ratio"), 2),
                    "adx": round(_feature_float(feature, "adx_14"), 1),
                    "change_24h": round(_feature_float(feature, "change_24h_pct"), 2),
                }
                for symbol, feature in selected_items[: min(8, len(selected_items))]
            ],
        }
        return EntryFeatureRankResult(selected=selected, diagnostics=diagnostics)

    def _parse_filter_inputs(
        self,
        feature: Any,
    ) -> tuple[str, float, float, float, float, float, float] | None:
        try:
            symbol = str(getattr(feature, "symbol", "") or "").upper()
            if self.suspicious_symbol_reason(symbol):
                return None
            current_price = float(
                getattr(feature, "current_price", 0) or getattr(feature, "close", 0) or 0
            )
            volume_24h = float(getattr(feature, "volume_24h", 0) or 0)
            volume_ratio = float(getattr(feature, "volume_ratio", 0) or 0)
            volatility_20 = float(getattr(feature, "volatility_20", 0) or 0)
            change_24h = abs(float(getattr(feature, "change_24h_pct", 0) or 0))
            adx_14 = float(getattr(feature, "adx_14", 0) or 0)
        except (TypeError, ValueError):
            return None
        return symbol, current_price, volume_24h, volume_ratio, volatility_20, change_24h, adx_14

    @staticmethod
    def _has_recent_abnormal_wick(feature: Any) -> bool:
        try:
            abnormal_wick_count = int(float(getattr(feature, "abnormal_wick_count_72h", 0) or 0))
            abnormal_wick_max_pct = float(getattr(feature, "abnormal_wick_max_pct", 0) or 0)
            abnormal_wick_recent_hours = float(
                getattr(feature, "abnormal_wick_recent_hours", 9999) or 9999
            )
        except (TypeError, ValueError):
            return False
        return (
            abnormal_wick_count >= ABNORMAL_WICK_ENTRY_BLOCK_MIN_COUNT
            and abnormal_wick_max_pct >= ABNORMAL_WICK_ENTRY_BLOCK_MAX_PCT
            and abnormal_wick_recent_hours <= ABNORMAL_WICK_ENTRY_BLOCK_RECENT_HOURS
        )
