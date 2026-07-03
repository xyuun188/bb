"""Auto-scan feature ranking for entry candidates."""

from __future__ import annotations

import math
from collections import Counter
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


def _feature_text(feature: Any, key: str) -> str:
    return str(getattr(feature, key, "") or "").strip()


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
        if self._missing_indicator_snapshot(feature):
            return 0.0
        params = self.params
        try:
            volume_24h = float(getattr(feature, "volume_24h", 0) or 0)
            volume_ratio = self._entry_activity_volume_ratio(feature)
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

        notional_24h = self._feature_notional_24h_usdt(feature, current_price, volume_24h)
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

        notional_24h = self._feature_notional_24h_usdt(feature, current_price, volume_24h)
        min_notional = (
            params.tradable_major_min_notional_usdt
            if symbol in self.major_symbols
            else params.tradable_alt_min_notional_usdt
        )
        tradable_volume_floor = params.tradable_volume_floor
        tradable_adx_floor = params.tradable_adx_floor
        if volume_ratio < tradable_volume_floor:
            return False
        if notional_24h < min_notional:
            return False
        if volatility_20 > params.tradable_max_volatility:
            return False
        if change_24h > params.tradable_max_day_change_pct:
            return False
        return not (symbol not in self.major_symbols and adx_14 < tradable_adx_floor)

    def is_auto_analysis_candidate_feature(self, feature: Any) -> bool:
        params = self.params
        parsed = self._parse_filter_inputs(feature)
        if parsed is None:
            return False
        symbol, current_price, volume_24h, volume_ratio, volatility_20, change_24h, adx_14 = parsed
        if self._has_recent_abnormal_wick(feature):
            return False

        notional_24h = self._feature_notional_24h_usdt(feature, current_price, volume_24h)
        min_notional = (
            params.analysis_major_min_notional_usdt
            if symbol in self.major_symbols
            else params.analysis_alt_min_notional_usdt
        )
        analysis_volume_floor = params.analysis_volume_floor
        analysis_adx_floor = params.analysis_adx_floor
        if volume_ratio < analysis_volume_floor:
            return False
        if notional_24h < min_notional:
            return False
        if volatility_20 > params.analysis_max_volatility:
            return False
        if change_24h > params.analysis_max_day_change_pct:
            return False
        return not (symbol not in self.major_symbols and adx_14 < analysis_adx_floor)

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
        tradable_symbols = {symbol for symbol, _ in tradable_items}
        soft_symbols = {symbol for symbol, _ in soft_items}
        filtered_items = [
            (symbol, feature)
            for symbol, feature in all_items
            if symbol not in tradable_symbols and symbol not in soft_symbols
        ]
        filter_diagnostics = {
            symbol: self._feature_filter_diagnostic(feature) for symbol, feature in all_items
        }

        def ranking_score(item: tuple[str, Any]) -> float:
            symbol, feature = item
            rotation_penalty = no_opportunity_rotation_penalty(symbol, feature)
            return (
                self.feature_opportunity_score(feature)
                - recent_hold_penalty(symbol)
                - recent_analysis_penalty(symbol)
                - rotation_penalty
            )

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
        ranked_filtered = sorted(
            filtered_items,
            key=ranking_score,
            reverse=True,
        )

        def split_recent_analysis(
            items: list[tuple[str, Any]],
        ) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
            fresh_items: list[tuple[str, Any]] = []
            recent_items: list[tuple[str, Any]] = []
            for symbol, feature in items:
                if recent_analysis_penalty(symbol) > 0:
                    recent_items.append((symbol, feature))
                else:
                    fresh_items.append((symbol, feature))
            return fresh_items, recent_items

        fresh_tradable, recent_tradable = split_recent_analysis(ranked_tradable)
        fresh_soft, recent_soft = split_recent_analysis(ranked_soft)
        selected_items: list[tuple[str, Any]] = []
        for bucket in (fresh_tradable, fresh_soft, recent_tradable, recent_soft):
            if len(selected_items) >= limit:
                break
            selected_items.extend(bucket[: max(limit - len(selected_items), 0)])
        underfill_fallback_symbols: set[str] = set()
        if len(selected_items) < max(limit, 0):
            fallback_fill_items = [
                item
                for item in ranked_filtered
                if self._can_fill_underfilled_analysis(
                    item[1],
                    filter_diagnostics.get(item[0], {}),
                )
            ]
            selected_fallback_items = fallback_fill_items[: max(limit - len(selected_items), 0)]
            selected_items.extend(selected_fallback_items)
            underfill_fallback_symbols = {symbol for symbol, _feature in selected_fallback_items}

        selected = dict(selected_items)
        selected_symbols = {symbol for symbol, _ in selected_items}
        recent_candidate_symbols = {
            symbol
            for symbol, _feature in [*ranked_tradable, *ranked_soft]
            if recent_analysis_penalty(symbol) > 0
        }
        recent_selected_symbols = {
            symbol for symbol in selected_symbols if symbol in recent_candidate_symbols
        }
        recent_deferred_symbols = sorted(recent_candidate_symbols - selected_symbols)

        def symbol_diagnostic(
            symbol: str,
            feature: Any,
            *,
            selected_item: bool,
            fallback_item: bool = False,
        ) -> dict[str, Any]:
            raw_score = self.feature_opportunity_score(feature)
            hold_penalty = recent_hold_penalty(symbol)
            analysis_penalty = recent_analysis_penalty(symbol)
            rotation_penalty = no_opportunity_rotation_penalty(symbol, feature)
            if symbol in tradable_symbols:
                tier = "hard_filter"
            elif self.is_auto_analysis_candidate_feature(feature):
                tier = "secondary_fill"
            elif fallback_item:
                tier = "fallback_score"
            else:
                tier = "filtered_out"
            filter_diag = filter_diagnostics.get(symbol, {})
            if selected_item:
                reason = "selected_for_market_analysis"
            elif symbol in recent_deferred_symbols:
                reason = "recent_analysis_diversity_deferred"
            elif tier == "filtered_out":
                reason = "feature_filter_rejected"
            else:
                reason = "outside_market_symbol_budget"
            return {
                "symbol": symbol,
                "score": round(raw_score, 2),
                "net_score": round(
                    raw_score - hold_penalty - analysis_penalty - rotation_penalty, 2
                ),
                "recent_hold_penalty": round(hold_penalty, 2),
                "recent_analysis_penalty": round(analysis_penalty, 2),
                "rotation_penalty": round(rotation_penalty, 2),
                "selection_tier": tier,
                "selected": selected_item,
                "non_selected_reason": reason,
                "filter_reasons": list(
                    filter_diag.get(
                        (
                            "analysis_reasons"
                            if tier in {"filtered_out", "fallback_score"}
                            else "tradable_reasons"
                        ),
                        [],
                    )
                ),
                "filter_metrics": dict(filter_diag.get("metrics") or {}),
                "volume_ratio": round(self._entry_activity_volume_ratio(feature), 2),
                "trend_volume_ratio": round(_feature_float(feature, "volume_ratio"), 2),
                "volume_ratio_source": self._entry_activity_volume_ratio_source(feature),
                "trend_volume_ratio_timeframe": _feature_text(feature, "volume_ratio_timeframe"),
                "entry_activity_volume_ratio": round(
                    _feature_float(feature, "entry_activity_volume_ratio"),
                    4,
                ),
                "entry_activity_volume_timeframe": _feature_text(
                    feature,
                    "entry_activity_volume_timeframe",
                ),
                "adx": round(_feature_float(feature, "adx_14"), 1),
                "change_24h": round(_feature_float(feature, "change_24h_pct"), 2),
            }

        ranked_candidates = [*ranked_tradable, *ranked_soft]
        if not ranked_candidates:
            fallback_items = [
                item for item in all_items if not self._missing_indicator_snapshot(item[1])
            ]
            ranked_candidates = sorted(
                fallback_items or all_items,
                key=ranking_score,
                reverse=True,
            )
            fallback_symbols = {symbol for symbol, _ in ranked_candidates}
        else:
            fallback_symbols = set()
        fallback_symbols |= underfill_fallback_symbols
        rank_sample_items = []
        seen_symbols: set[str] = set()
        for symbol, feature in [*selected_items, *ranked_candidates]:
            if symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            rank_sample_items.append((symbol, feature))
            if len(rank_sample_items) >= 12:
                break
        filtered_reason_counts = Counter()
        for symbol, _feature in filtered_items:
            reasons = list(filter_diagnostics.get(symbol, {}).get("analysis_reasons") or [])
            filtered_reason_counts.update(reasons or ["filtered_without_reason"])

        rank_underfilled = len(selected) < max(0, int(limit or 0))
        missing_indicator_count = sum(
            1 for _symbol, feature in all_items if self._missing_indicator_snapshot(feature)
        )
        if rank_underfilled and all_items and missing_indicator_count == len(all_items):
            rank_underfill_reason = "missing_indicator_snapshot"
        elif rank_underfilled and ranked_candidates:
            rank_underfill_reason = "insufficient_tradeable_or_secondary_candidates"
        elif rank_underfilled and all_items:
            rank_underfill_reason = "fallback_selected_filtered_candidates"
        else:
            rank_underfill_reason = ""
        diagnostics = {
            "selected": len(selected),
            "candidates": len(feature_vectors),
            "tradable_candidates": len(tradable_items),
            "secondary_candidates": len(soft_items),
            "filtered_out_candidates": len(filtered_items),
            "market_symbol_limit": max(0, int(limit or 0)),
            "rank_underfilled": rank_underfilled,
            "rank_underfill_reason": rank_underfill_reason,
            "fallback_filtered_fill_count": len(underfill_fallback_symbols),
            "fallback_filtered_fill_policy": {
                "read_only": True,
                "is_entry_gate": False,
                "applied": bool(underfill_fallback_symbols),
                "symbols": sorted(underfill_fallback_symbols)[:12],
                "reason": (
                    "When qualified market-analysis candidates underfill the available budget, "
                    "the ranker may spend unused analysis capacity on the best non-severe "
                    "filtered candidates that still have positive opportunity score and at "
                    "least one market-structure anchor. This only broadens analysis coverage; "
                    "entry evidence, sizing, leverage, ML readiness, OKX state, and risk checks "
                    "still decide whether a real order can be submitted."
                ),
            },
            "recent_analysis_diversity": {
                "read_only": True,
                "is_entry_gate": False,
                "applied": bool(recent_deferred_symbols),
                "recent_candidate_count": len(recent_candidate_symbols),
                "recent_deferred_count": len(recent_deferred_symbols),
                "recent_selected_count": len(recent_selected_symbols),
                "recent_deferred_symbols": recent_deferred_symbols[:20],
                "recent_selected_symbols": sorted(recent_selected_symbols)[:20],
                "reason": (
                    "recently analyzed symbols are deferred while fresh qualified "
                    "market-analysis candidates are available; execution gates, sizing, "
                    "leverage, and risk checks are unchanged"
                ),
            },
            "filtered_out_reason_counts": [
                {"reason": reason, "count": int(count)}
                for reason, count in filtered_reason_counts.most_common(12)
            ],
            "ranked_symbol_sample": [
                symbol_diagnostic(
                    symbol,
                    feature,
                    selected_item=symbol in selected_symbols,
                    fallback_item=symbol in fallback_symbols,
                )
                for symbol, feature in rank_sample_items
            ],
            "filtered_symbol_sample": [
                symbol_diagnostic(
                    symbol,
                    feature,
                    selected_item=symbol in selected_symbols,
                    fallback_item=symbol in fallback_symbols,
                )
                for symbol, feature in ranked_filtered[: min(8, len(ranked_filtered))]
            ],
            "symbols": [
                symbol_diagnostic(
                    symbol,
                    feature,
                    selected_item=True,
                    fallback_item=symbol in fallback_symbols,
                )
                for symbol, feature in selected_items[: min(8, len(selected_items))]
            ],
        }
        return EntryFeatureRankResult(selected=selected, diagnostics=diagnostics)

    def _can_fill_underfilled_analysis(self, feature: Any, diagnostic: dict[str, Any]) -> bool:
        if self._missing_indicator_snapshot(feature):
            return False
        symbol = str(getattr(feature, "symbol", "") or "").upper()
        if symbol and self.suspicious_symbol_reason(symbol):
            return False
        reasons = set(diagnostic.get("analysis_reasons") or [])
        severe_reasons = {
            "missing_indicator_snapshot",
            "suspicious_symbol",
            "invalid_feature_values",
            "recent_abnormal_wick",
            "analysis_volatility_above_cap",
            "analysis_day_change_above_cap",
        }
        if reasons & severe_reasons:
            return False
        if self.feature_opportunity_score(feature) <= 0:
            return False
        return self._has_underfilled_market_support(diagnostic)

    @staticmethod
    def _has_underfilled_market_support(diagnostic: dict[str, Any]) -> bool:
        metrics = dict(diagnostic.get("metrics") or {})

        def as_float(key: str) -> float:
            try:
                return float(metrics.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        notional = as_float("notional_24h")
        min_notional = as_float("analysis_min_notional")
        volume_ratio = as_float("volume_ratio")
        volume_floor = as_float("analysis_volume_floor")
        adx = as_float("adx")
        adx_floor = as_float("analysis_adx_floor")
        support_count = 0
        if min_notional <= 0 or notional >= min_notional:
            support_count += 1
        if volume_floor <= 0 or volume_ratio >= volume_floor:
            support_count += 1
        if adx_floor <= 0 or adx >= adx_floor:
            support_count += 1
        return support_count > 0

    def _feature_filter_diagnostic(self, feature: Any) -> dict[str, Any]:
        params = self.params
        parsed = self._parse_filter_inputs(feature)
        symbol = str(getattr(feature, "symbol", "") or "").upper()
        if parsed is None:
            reason = (
                "missing_indicator_snapshot"
                if self._missing_indicator_snapshot(feature)
                else (
                    "suspicious_symbol"
                    if symbol and self.suspicious_symbol_reason(symbol)
                    else "invalid_feature_values"
                )
            )
            return {
                "symbol": symbol,
                "tradable_reasons": [reason],
                "analysis_reasons": [reason],
                "metrics": {},
            }

        symbol, current_price, volume_24h, volume_ratio, volatility_20, change_24h, adx_14 = parsed
        abnormal_wick = self._has_recent_abnormal_wick(feature)
        notional_24h = self._feature_notional_24h_usdt(feature, current_price, volume_24h)
        tradable_min_notional = (
            params.tradable_major_min_notional_usdt
            if symbol in self.major_symbols
            else params.tradable_alt_min_notional_usdt
        )
        analysis_min_notional = (
            params.analysis_major_min_notional_usdt
            if symbol in self.major_symbols
            else params.analysis_alt_min_notional_usdt
        )
        runtime_entry_volume_ratio = max(float(self.min_entry_volume_ratio_provider() or 0.0), 0.0)
        runtime_entry_adx = max(float(self.min_entry_adx_provider() or 0.0), 0.0)
        tradable_volume_floor = params.tradable_volume_floor
        analysis_volume_floor = params.analysis_volume_floor
        tradable_adx_floor = params.tradable_adx_floor
        analysis_adx_floor = params.analysis_adx_floor

        tradable_reasons: list[str] = []
        analysis_reasons: list[str] = []
        if abnormal_wick:
            tradable_reasons.append("recent_abnormal_wick")
            analysis_reasons.append("recent_abnormal_wick")
        if volume_ratio < tradable_volume_floor:
            tradable_reasons.append("tradable_volume_ratio_below_floor")
        if notional_24h < tradable_min_notional:
            tradable_reasons.append("tradable_notional_below_floor")
        if volatility_20 > params.tradable_max_volatility:
            tradable_reasons.append("tradable_volatility_above_cap")
        if change_24h > params.tradable_max_day_change_pct:
            tradable_reasons.append("tradable_day_change_above_cap")
        if symbol not in self.major_symbols and adx_14 < tradable_adx_floor:
            tradable_reasons.append("tradable_adx_below_floor")

        if volume_ratio < analysis_volume_floor:
            analysis_reasons.append("analysis_volume_ratio_below_floor")
        if notional_24h < analysis_min_notional:
            analysis_reasons.append("analysis_notional_below_floor")
        if volatility_20 > params.analysis_max_volatility:
            analysis_reasons.append("analysis_volatility_above_cap")
        if change_24h > params.analysis_max_day_change_pct:
            analysis_reasons.append("analysis_day_change_above_cap")
        if symbol not in self.major_symbols and adx_14 < analysis_adx_floor:
            analysis_reasons.append("analysis_adx_below_floor")

        return {
            "symbol": symbol,
            "tradable_reasons": tradable_reasons,
            "analysis_reasons": analysis_reasons,
            "metrics": {
                "notional_24h": round(notional_24h, 2),
                "notional_24h_source": str(
                    getattr(feature, "volume_24h_source", "") or "price_x_volume_24h"
                ),
                "volume_ratio": round(volume_ratio, 4),
                "volume_ratio_source": self._entry_activity_volume_ratio_source(feature),
                "trend_volume_ratio": round(_feature_float(feature, "volume_ratio"), 4),
                "trend_volume_ratio_timeframe": _feature_text(feature, "volume_ratio_timeframe"),
                "entry_activity_volume_ratio": round(
                    _feature_float(feature, "entry_activity_volume_ratio"),
                    4,
                ),
                "entry_activity_volume_timeframe": _feature_text(
                    feature,
                    "entry_activity_volume_timeframe",
                ),
                "runtime_entry_volume_ratio_advisory": round(runtime_entry_volume_ratio, 4),
                "runtime_entry_adx_advisory": round(runtime_entry_adx, 2),
                "adx": round(adx_14, 2),
                "volatility_20": round(volatility_20, 4),
                "change_24h": round(change_24h, 4),
                "tradable_volume_floor": round(tradable_volume_floor, 4),
                "analysis_volume_floor": round(analysis_volume_floor, 4),
                "tradable_min_notional": round(tradable_min_notional, 2),
                "analysis_min_notional": round(analysis_min_notional, 2),
                "tradable_adx_floor": round(tradable_adx_floor, 2),
                "analysis_adx_floor": round(analysis_adx_floor, 2),
            },
        }

    @staticmethod
    def _feature_notional_24h_usdt(
        feature: Any,
        current_price: float,
        volume_24h: float,
    ) -> float:
        explicit_notional = _feature_float(feature, "notional_24h_usdt", 0.0)
        if explicit_notional > 0:
            return explicit_notional
        quote_volume = _feature_float(feature, "volume_24h_quote", 0.0)
        if quote_volume > 0:
            return quote_volume
        base_volume = _feature_float(feature, "volume_24h_base", 0.0)
        if base_volume > 0:
            return max(base_volume * max(current_price, 0.0), 0.0)
        return max(volume_24h * max(current_price, 0.0), 0.0)

    def _parse_filter_inputs(
        self,
        feature: Any,
    ) -> tuple[str, float, float, float, float, float, float] | None:
        try:
            symbol = str(getattr(feature, "symbol", "") or "").upper()
            if self.suspicious_symbol_reason(symbol):
                return None
            if self._missing_indicator_snapshot(feature):
                return None
            current_price = float(
                getattr(feature, "current_price", 0) or getattr(feature, "close", 0) or 0
            )
            volume_24h = float(getattr(feature, "volume_24h", 0) or 0)
            volume_ratio = self._entry_activity_volume_ratio(feature)
            volatility_20 = float(getattr(feature, "volatility_20", 0) or 0)
            change_24h = abs(float(getattr(feature, "change_24h_pct", 0) or 0))
            adx_14 = float(getattr(feature, "adx_14", 0) or 0)
        except (TypeError, ValueError):
            return None
        return symbol, current_price, volume_24h, volume_ratio, volatility_20, change_24h, adx_14

    @staticmethod
    def _missing_indicator_snapshot(feature: Any) -> bool:
        marker = getattr(feature, "indicator_snapshot_available", None)
        if marker is None:
            return False
        if isinstance(marker, str):
            marker = marker.strip().lower() in {"1", "true", "yes", "y"}
        return not bool(marker)

    @staticmethod
    def _entry_activity_volume_ratio(feature: Any) -> float:
        if _feature_text(feature, "entry_activity_volume_timeframe"):
            return max(_feature_float(feature, "entry_activity_volume_ratio"), 0.0)
        return max(_feature_float(feature, "volume_ratio"), 0.0)

    @staticmethod
    def _entry_activity_volume_ratio_source(feature: Any) -> str:
        if _feature_text(feature, "entry_activity_volume_timeframe"):
            return "entry_activity_volume_ratio"
        return "volume_ratio"

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
