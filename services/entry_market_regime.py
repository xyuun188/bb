"""Entry market-regime advisory policy."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput

NormalizeSymbol = Callable[[Any], str | None]
FeatureValidator = Callable[[Any], bool]

ALT_LONG_BTC_ETH_5M_FLOOR = -0.0015
ALT_LONG_BTC_ETH_20M_FLOOR = -0.004
ALT_LONG_BTC_ETH_ADX_FLOOR = 16.0


def _feature_float(feature: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(feature, name, default) or default)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class EntryMarketRegimeContextPolicy:
    """Build market-regime context before per-symbol entry analysis."""

    is_valid_feature_vector: FeatureValidator
    alt_long_5m_floor: float = ALT_LONG_BTC_ETH_5M_FLOOR
    alt_long_20m_floor: float = ALT_LONG_BTC_ETH_20M_FLOOR
    alt_long_adx_floor: float = ALT_LONG_BTC_ETH_ADX_FLOOR

    def context(self, feature_vectors: dict[str, Any]) -> dict[str, Any]:
        rows = [
            feature
            for feature in (feature_vectors or {}).values()
            if self.is_valid_feature_vector(feature)
        ]
        if not rows:
            return {
                "mode": "unknown",
                "confidence": 0.0,
                "avoid_long": False,
                "avoid_short": False,
            }

        total = len(rows)
        up_5 = sum(1 for feature in rows if _feature_float(feature, "returns_5") > 0.002)
        down_5 = sum(1 for feature in rows if _feature_float(feature, "returns_5") < -0.002)
        up_20 = sum(1 for feature in rows if _feature_float(feature, "returns_20") > 0.006)
        down_20 = sum(1 for feature in rows if _feature_float(feature, "returns_20") < -0.006)
        above_sma = sum(
            1
            for feature in rows
            if _feature_float(feature, "price_vs_sma20") > 0
            and _feature_float(feature, "price_vs_sma50") > 0
        )
        below_sma = sum(
            1
            for feature in rows
            if _feature_float(feature, "price_vs_sma20") < 0
            and _feature_float(feature, "price_vs_sma50") < 0
        )
        high_adx = sum(1 for feature in rows if _feature_float(feature, "adx_14") >= 25)
        avg_ret_5 = sum(_feature_float(feature, "returns_5") for feature in rows) / total
        avg_ret_20 = sum(_feature_float(feature, "returns_20") for feature in rows) / total

        majors = [
            feature
            for feature in rows
            if str(getattr(feature, "symbol", "")).upper() in {"BTC/USDT", "ETH/USDT"}
        ]
        major_score = 0.0
        for feature in majors:
            major_score += (
                _feature_float(feature, "returns_5") * 0.55
                + _feature_float(feature, "returns_20") * 0.45
            )
        if majors:
            major_score /= len(majors)
        btc_eth_filter = self.btc_eth_alt_long_filter(majors)

        up_breadth = max(up_5 / total, up_20 / total)
        down_breadth = max(down_5 / total, down_20 / total)
        trend_breadth = max(above_sma / total, below_sma / total)
        confidence = min(
            max(abs(up_breadth - down_breadth) + abs(avg_ret_20) * 12 + trend_breadth * 0.25, 0.0),
            0.95,
        )

        mode = "mixed"
        avoid_long = False
        avoid_short = False
        reason = "Market direction is mixed; trade only symbol-level high-quality signals."
        if up_breadth >= 0.55 and avg_ret_5 > 0 and major_score >= -0.001:
            mode = "rebound_squeeze_up"
            avoid_short = True
            reason = "Broad short-term rebound; shorts need stronger symbol-level evidence."
        elif down_breadth >= 0.55 and avg_ret_5 < 0 and major_score <= 0.001:
            mode = "selloff_squeeze_down"
            avoid_long = True
            reason = "Broad short-term selloff; longs need stronger symbol-level evidence."
        elif above_sma / total >= 0.55 and avg_ret_20 > 0.003:
            mode = "uptrend_continuation"
            avoid_short = True
            reason = "Broad uptrend; counter-trend shorts need stronger evidence."
        elif below_sma / total >= 0.55 and avg_ret_20 < -0.003:
            mode = "downtrend_continuation"
            avoid_long = True
            reason = "Broad downtrend; counter-trend longs need stronger evidence."

        return {
            "mode": mode,
            "confidence": round(confidence, 4),
            "avoid_long": avoid_long,
            "avoid_short": avoid_short,
            "reason": reason,
            "sample_count": total,
            "up_5_ratio": round(up_5 / total, 4),
            "down_5_ratio": round(down_5 / total, 4),
            "up_20_ratio": round(up_20 / total, 4),
            "down_20_ratio": round(down_20 / total, 4),
            "above_sma_ratio": round(above_sma / total, 4),
            "below_sma_ratio": round(below_sma / total, 4),
            "high_adx_ratio": round(high_adx / total, 4),
            "avg_returns_5": round(avg_ret_5, 6),
            "avg_returns_20": round(avg_ret_20, 6),
            "major_score": round(major_score, 6),
            "btc_eth_filter": btc_eth_filter,
        }

    def btc_eth_alt_long_filter(self, majors: list[Any]) -> dict[str, Any]:
        if not majors:
            return {
                "allow_alt_long": True,
                "reason": "BTC/ETH context unavailable; do not add an extra alt-long block.",
            }

        avg_ret_5 = sum(_feature_float(feature, "returns_5") for feature in majors) / len(majors)
        avg_ret_20 = sum(_feature_float(feature, "returns_20") for feature in majors) / len(majors)
        avg_adx = sum(_feature_float(feature, "adx_14") for feature in majors) / len(majors)
        allow = not (
            avg_ret_5 <= self.alt_long_5m_floor
            and avg_ret_20 <= self.alt_long_20m_floor
            and avg_adx >= self.alt_long_adx_floor
        )
        reason = f"BTC/ETH avg returns: 5={avg_ret_5:.4f}, 20={avg_ret_20:.4f}, ADX={avg_adx:.1f}."
        if not allow:
            reason = f"{reason} Broad market is falling and trend has not recovered."
        return {
            "allow_alt_long": allow,
            "avg_returns_5": round(avg_ret_5, 6),
            "avg_returns_20": round(avg_ret_20, 6),
            "avg_adx_14": round(avg_adx, 4),
            "reason": reason,
        }


@dataclass(frozen=True, slots=True)
class EntryMarketRegimePolicy:
    """Annotate entry decisions with market-regime warnings without hard-blocking AI."""

    normalize_symbol: NormalizeSymbol
    alt_long_allowed_symbols: frozenset[str]

    def __init__(
        self,
        normalize_symbol: NormalizeSymbol,
        alt_long_allowed_symbols: Iterable[str],
    ) -> None:
        object.__setattr__(self, "normalize_symbol", normalize_symbol)
        object.__setattr__(
            self,
            "alt_long_allowed_symbols",
            frozenset(str(symbol).upper() for symbol in alt_long_allowed_symbols),
        )

    def reason(
        self,
        decision: DecisionOutput,
        market_regime: dict[str, Any],
    ) -> str | None:
        if not decision.is_entry or not isinstance(market_regime, dict):
            return None
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        symbol = (self.normalize_symbol(decision.symbol) or decision.symbol).upper()
        if decision.action == Action.LONG and symbol not in self.alt_long_allowed_symbols:
            btc_eth = (
                market_regime.get("btc_eth_filter")
                if isinstance(market_regime.get("btc_eth_filter"), dict)
                else {}
            )
            raw["alt_long_style_filter"] = {
                "blocked": False,
                "soft_warning": True,
                "reason": self._reason_text(btc_eth),
                "btc_eth_filter": btc_eth,
            }
            decision.raw_response = raw
        return None

    @staticmethod
    def _reason_text(btc_eth: dict[str, Any]) -> str:
        if btc_eth and not bool(btc_eth.get("allow_alt_long", True)):
            return (
                "BTC/ETH is weak, but this filter is advisory only; single-symbol long "
                "entries still go through AI, ML, time-series, price guard, and account "
                "risk checks."
            )
        return "Alt-long style filter is advisory; hard checks happen later."
