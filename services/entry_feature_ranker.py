"""Auto-scan feature ranking for entry candidates."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from services.dynamic_policy_values import (
    DynamicPolicyValue,
    empirical_policy_value,
)

SuspiciousSymbolReason = Callable[[str], str | None]


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
    major_symbols: frozenset[str]

    def feature_opportunity_score(self, feature: Any) -> float:
        if self._missing_indicator_snapshot(feature) and not self._has_basic_market_anchor(feature):
            return 0.0
        try:
            volume_24h = float(getattr(feature, "volume_24h", 0) or 0)
            volume_ratio = self._entry_activity_volume_ratio(feature)
            adx_14 = float(getattr(feature, "adx_14", 0) or 0)
            returns_1 = abs(float(getattr(feature, "returns_1", 0) or 0))
            returns_5 = abs(float(getattr(feature, "returns_5", 0) or 0))
            returns_20 = abs(float(getattr(feature, "returns_20", 0) or 0))
            volatility_20 = float(getattr(feature, "volatility_20", 0) or 0)
            change_24h = abs(float(getattr(feature, "change_24h_pct", 0) or 0))
            current_price = float(
                getattr(feature, "current_price", 0) or getattr(feature, "close", 0) or 0
            )
        except (TypeError, ValueError):
            return 0.0

        notional_24h = self._feature_notional_24h_usdt(feature, current_price, volume_24h)
        liquidity = math.log1p(max(notional_24h, 0.0))
        participation = math.log1p(max(volume_ratio, 0.0))
        trend_quality = math.log1p(max(adx_14, 0.0))
        realized_move = returns_1 + returns_5 + returns_20 + change_24h
        move_efficiency = realized_move / max(volatility_20, abs(returns_1), 1e-12)
        quality_inputs = [liquidity, participation, trend_quality, move_efficiency]
        return sum(quality_inputs) / len(quality_inputs)

    def is_auto_tradeable_feature(self, feature: Any) -> bool:
        parsed = self._parse_filter_inputs(feature, allow_incomplete_indicator=False)
        return parsed is not None

    def is_auto_analysis_candidate_feature(self, feature: Any) -> bool:
        parsed = self._parse_filter_inputs(feature, allow_incomplete_indicator=True)
        return parsed is not None

    def _cross_sectional_policy(
        self,
        feature_vectors: dict[str, Any],
    ) -> dict[str, DynamicPolicyValue]:
        metrics: dict[str, list[float]] = {
            "volume_ratio": [],
            "adx": [],
            "volatility": [],
            "day_change": [],
            "major_notional": [],
            "alt_notional": [],
        }
        for feature in feature_vectors.values():
            parsed = self._parse_filter_inputs(feature, allow_incomplete_indicator=True)
            if parsed is None:
                continue
            symbol, current_price, volume_24h, volume_ratio, volatility, day_change, adx = parsed
            notional = self._feature_notional_24h_usdt(feature, current_price, volume_24h)
            metrics["volume_ratio"].append(volume_ratio)
            metrics["adx"].append(adx)
            metrics["volatility"].append(volatility)
            metrics["day_change"].append(day_change)
            metrics["major_notional" if symbol in self.major_symbols else "alt_notional"].append(
                notional
            )

        window = "current_market_feature_cross_section"
        return {
            "analysis_volume_floor": empirical_policy_value(
                "analysis_volume_floor",
                metrics["volume_ratio"],
                selector="lower_hinge",
                observation_window=window,
            ),
            "tradable_volume_floor": empirical_policy_value(
                "tradable_volume_floor",
                metrics["volume_ratio"],
                selector="median",
                observation_window=window,
            ),
            "analysis_adx_floor": empirical_policy_value(
                "analysis_adx_floor",
                metrics["adx"],
                selector="lower_hinge",
                observation_window=window,
            ),
            "tradable_adx_floor": empirical_policy_value(
                "tradable_adx_floor",
                metrics["adx"],
                selector="median",
                observation_window=window,
            ),
            "analysis_volatility_cap": empirical_policy_value(
                "analysis_volatility_cap",
                metrics["volatility"],
                selector="upper_hinge",
                observation_window=window,
            ),
            "tradable_volatility_cap": empirical_policy_value(
                "tradable_volatility_cap",
                metrics["volatility"],
                selector="median",
                observation_window=window,
            ),
            "analysis_day_change_cap": empirical_policy_value(
                "analysis_day_change_cap",
                metrics["day_change"],
                selector="upper_hinge",
                observation_window=window,
            ),
            "tradable_day_change_cap": empirical_policy_value(
                "tradable_day_change_cap",
                metrics["day_change"],
                selector="median",
                observation_window=window,
            ),
            **{
                f"{tier}_{group}_notional_floor": empirical_policy_value(
                    f"{tier}_{group}_notional_floor",
                    metrics[f"{group}_notional"],
                    selector="median" if tier == "tradable" else "lower_hinge",
                    observation_window=window,
                )
                for tier in ("analysis", "tradable")
                for group in ("major", "alt")
            },
        }

    @staticmethod
    def _policy_number(
        policy: dict[str, DynamicPolicyValue],
        name: str,
    ) -> float | None:
        item = policy.get(name)
        if item is None or not item.production_eligible or item.value is None:
            return None
        return float(item.value)

    def _passes_cross_sectional_policy(
        self,
        feature: Any,
        policy: dict[str, DynamicPolicyValue],
        *,
        tier: str,
    ) -> bool:
        parsed = self._parse_filter_inputs(
            feature,
            allow_incomplete_indicator=tier == "analysis",
        )
        if parsed is None:
            return False
        symbol, current_price, volume_24h, volume_ratio, volatility, day_change, adx = parsed
        group = "major" if symbol in self.major_symbols else "alt"
        notional = self._feature_notional_24h_usdt(feature, current_price, volume_24h)
        thresholds = {
            "volume": self._policy_number(policy, f"{tier}_volume_floor"),
            "notional": self._policy_number(policy, f"{tier}_{group}_notional_floor"),
            "adx": self._policy_number(policy, f"{tier}_adx_floor"),
            "volatility": self._policy_number(policy, f"{tier}_volatility_cap"),
            "day_change": self._policy_number(policy, f"{tier}_day_change_cap"),
        }
        if any(value is None for value in thresholds.values()):
            return False
        return bool(
            volume_ratio >= float(thresholds["volume"])
            and notional >= float(thresholds["notional"])
            and adx >= float(thresholds["adx"])
            and volatility <= float(thresholds["volatility"])
            and day_change <= float(thresholds["day_change"])
        )

    def rank(
        self,
        feature_vectors: dict[str, Any],
        limit: int,
    ) -> EntryFeatureRankResult:
        all_items = list(feature_vectors.items())
        dynamic_policy = self._cross_sectional_policy(feature_vectors)
        tradable_items = [
            item
            for item in feature_vectors.items()
            if self._passes_cross_sectional_policy(item[1], dynamic_policy, tier="tradable")
        ]
        soft_items = [
            item
            for item in all_items
            if item not in tradable_items
            and self._passes_cross_sectional_policy(item[1], dynamic_policy, tier="analysis")
        ]
        tradable_symbols = {symbol for symbol, _ in tradable_items}
        soft_symbols = {symbol for symbol, _ in soft_items}
        filtered_items = [
            (symbol, feature)
            for symbol, feature in all_items
            if symbol not in tradable_symbols and symbol not in soft_symbols
        ]
        filter_diagnostics = {
            symbol: self._feature_filter_diagnostic(feature, dynamic_policy)
            for symbol, feature in all_items
        }

        def ranking_score(item: tuple[str, Any]) -> float:
            _symbol, feature = item
            return self.feature_opportunity_score(feature)

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

        selected_items: list[tuple[str, Any]] = []
        for bucket in (ranked_tradable, ranked_soft):
            if len(selected_items) >= limit:
                break
            selected_items.extend(bucket[: max(limit - len(selected_items), 0)])
        selected = dict(selected_items)
        selected_symbols = {symbol for symbol, _ in selected_items}

        def symbol_diagnostic(
            symbol: str,
            feature: Any,
            *,
            selected_item: bool,
        ) -> dict[str, Any]:
            raw_score = self.feature_opportunity_score(feature)
            if symbol in tradable_symbols:
                tier = "hard_filter"
            elif symbol in soft_symbols:
                tier = "secondary_fill"
            else:
                tier = "filtered_out"
            filter_diag = filter_diagnostics.get(symbol, {})
            if selected_item:
                reason = "selected_for_market_analysis"
            elif tier == "filtered_out":
                reason = "feature_filter_rejected"
            else:
                reason = "outside_market_symbol_budget"
            return {
                "symbol": symbol,
                "score": round(raw_score, 2),
                "net_score": round(raw_score, 2),
                "selection_tier": tier,
                "selected": selected_item,
                "non_selected_reason": reason,
                "filter_reasons": list(
                    filter_diag.get(
                        ("analysis_reasons" if tier == "filtered_out" else "tradable_reasons"),
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
        elif rank_underfilled and (ranked_candidates or all_items):
            rank_underfill_reason = "insufficient_tradeable_or_secondary_candidates"
        else:
            rank_underfill_reason = ""
        diagnostics = {
            "selected": len(selected),
            "candidates": len(feature_vectors),
            "tradable_candidates": len(tradable_items),
            "secondary_candidates": len(soft_items),
            "filtered_out_candidates": len(filtered_items),
            "dynamic_policy": {
                "version": "2026-07-12.dynamic-market-cross-section.v1",
                "values": {name: item.to_dict() for name, item in dynamic_policy.items()},
            },
            "market_symbol_limit": max(0, int(limit or 0)),
            "rank_underfilled": rank_underfilled,
            "rank_underfill_reason": rank_underfill_reason,
            "filtered_out_reason_counts": [
                {"reason": reason, "count": int(count)}
                for reason, count in filtered_reason_counts.most_common(12)
            ],
            "ranked_symbol_sample": [
                symbol_diagnostic(
                    symbol,
                    feature,
                    selected_item=symbol in selected_symbols,
                )
                for symbol, feature in rank_sample_items
            ],
            "filtered_symbol_sample": [
                symbol_diagnostic(
                    symbol,
                    feature,
                    selected_item=symbol in selected_symbols,
                )
                for symbol, feature in ranked_filtered[: min(8, len(ranked_filtered))]
            ],
            "symbols": [
                symbol_diagnostic(
                    symbol,
                    feature,
                    selected_item=True,
                )
                for symbol, feature in selected_items[: min(8, len(selected_items))]
            ],
        }
        return EntryFeatureRankResult(selected=selected, diagnostics=diagnostics)

    def _feature_filter_diagnostic(
        self,
        feature: Any,
        policy: dict[str, DynamicPolicyValue],
    ) -> dict[str, Any]:
        parsed = self._parse_filter_inputs(feature, allow_incomplete_indicator=True)
        symbol = str(getattr(feature, "symbol", "") or "").upper()
        if parsed is None:
            reason = (
                "missing_market_anchor_snapshot"
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
        notional_24h = self._feature_notional_24h_usdt(feature, current_price, volume_24h)
        group = "major" if symbol in self.major_symbols else "alt"

        def value(name: str) -> float | None:
            return self._policy_number(policy, name)

        thresholds = {
            "tradable_volume_floor": value("tradable_volume_floor"),
            "analysis_volume_floor": value("analysis_volume_floor"),
            "tradable_min_notional": value(f"tradable_{group}_notional_floor"),
            "analysis_min_notional": value(f"analysis_{group}_notional_floor"),
            "tradable_adx_floor": value("tradable_adx_floor"),
            "analysis_adx_floor": value("analysis_adx_floor"),
            "tradable_volatility_cap": value("tradable_volatility_cap"),
            "analysis_volatility_cap": value("analysis_volatility_cap"),
            "tradable_day_change_cap": value("tradable_day_change_cap"),
            "analysis_day_change_cap": value("analysis_day_change_cap"),
        }

        tradable_reasons: list[str] = []
        analysis_reasons: list[str] = []
        if self._uses_fallback_indicator_snapshot(feature):
            tradable_reasons.append("fallback_indicator_snapshot")
        if any(item is None for item in thresholds.values()):
            tradable_reasons.append("dynamic_market_distribution_unavailable")
            analysis_reasons.append("dynamic_market_distribution_unavailable")
        if thresholds["tradable_volume_floor"] is not None and volume_ratio < float(
            thresholds["tradable_volume_floor"]
        ):
            tradable_reasons.append("tradable_volume_ratio_below_floor")
        if thresholds["tradable_min_notional"] is not None and notional_24h < float(
            thresholds["tradable_min_notional"]
        ):
            tradable_reasons.append("tradable_notional_below_floor")
        if thresholds["tradable_volatility_cap"] is not None and volatility_20 > float(
            thresholds["tradable_volatility_cap"]
        ):
            tradable_reasons.append("tradable_volatility_above_cap")
        if thresholds["tradable_day_change_cap"] is not None and change_24h > float(
            thresholds["tradable_day_change_cap"]
        ):
            tradable_reasons.append("tradable_day_change_above_cap")
        if thresholds["tradable_adx_floor"] is not None and adx_14 < float(
            thresholds["tradable_adx_floor"]
        ):
            tradable_reasons.append("tradable_adx_below_floor")

        if thresholds["analysis_volume_floor"] is not None and volume_ratio < float(
            thresholds["analysis_volume_floor"]
        ):
            analysis_reasons.append("analysis_volume_ratio_below_floor")
        if thresholds["analysis_min_notional"] is not None and notional_24h < float(
            thresholds["analysis_min_notional"]
        ):
            analysis_reasons.append("analysis_notional_below_floor")
        if thresholds["analysis_volatility_cap"] is not None and volatility_20 > float(
            thresholds["analysis_volatility_cap"]
        ):
            analysis_reasons.append("analysis_volatility_above_cap")
        if thresholds["analysis_day_change_cap"] is not None and change_24h > float(
            thresholds["analysis_day_change_cap"]
        ):
            analysis_reasons.append("analysis_day_change_above_cap")
        if thresholds["analysis_adx_floor"] is not None and adx_14 < float(
            thresholds["analysis_adx_floor"]
        ):
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
                "adx": round(adx_14, 2),
                "volatility_20": round(volatility_20, 4),
                "change_24h": round(change_24h, 4),
                **{
                    name: None if item is None else round(item, 6)
                    for name, item in thresholds.items()
                },
                "threshold_source": "current_market_feature_cross_section",
                "indicator_snapshot_quality": (
                    "fallback_market_anchor"
                    if self._uses_fallback_indicator_snapshot(feature)
                    else "full_indicator_snapshot"
                ),
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
        *,
        allow_incomplete_indicator: bool = False,
    ) -> tuple[str, float, float, float, float, float, float] | None:
        try:
            symbol = str(getattr(feature, "symbol", "") or "").upper()
            if self.suspicious_symbol_reason(symbol):
                return None
            if self._missing_indicator_snapshot(feature) and not (
                allow_incomplete_indicator and self._has_basic_market_anchor(feature)
            ):
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
        if current_price <= 0:
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

    def _uses_fallback_indicator_snapshot(self, feature: Any) -> bool:
        return self._missing_indicator_snapshot(feature) and self._has_basic_market_anchor(feature)

    def _has_basic_market_anchor(self, feature: Any) -> bool:
        current_price = _feature_float(feature, "current_price") or _feature_float(feature, "close")
        if current_price <= 0:
            return False
        notional_24h = self._feature_notional_24h_usdt(
            feature,
            current_price,
            _feature_float(feature, "volume_24h"),
        )
        if notional_24h <= 0:
            return False
        return bool(
            self._entry_activity_volume_ratio(feature) > 0
            or abs(_feature_float(feature, "change_24h_pct")) > 0
            or _feature_float(feature, "adx_14") > 0
        )

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
