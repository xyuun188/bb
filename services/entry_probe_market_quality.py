"""Market-quality guard for AI-hold probe entry candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.trading_params import DEFAULT_TRADING_PARAMS, EntryProbeMarketQualityParams


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class EntryProbeMarketQualityPolicy:
    """Block probe entries when a fresh market snapshot contradicts the probe side."""

    params: EntryProbeMarketQualityParams = DEFAULT_TRADING_PARAMS.entry_probe_market_quality

    def block_reason(self, feature_vector: Any, side: str) -> str | None:
        current_price = _safe_float(getattr(feature_vector, "current_price", 0.0), 0.0)
        close_price = _safe_float(getattr(feature_vector, "close", 0.0), 0.0)
        returns_20 = _safe_float(getattr(feature_vector, "returns_20", 0.0), 0.0)
        volume_ratio = _safe_float(getattr(feature_vector, "volume_ratio", 0.0), 0.0)
        price_vs_sma20 = _safe_float(getattr(feature_vector, "price_vs_sma20", 0.0), 0.0)
        price_vs_sma50 = _safe_float(getattr(feature_vector, "price_vs_sma50", 0.0), 0.0)

        if current_price > 0 and close_price > 0:
            gap = abs(current_price - close_price) / max(close_price, 1e-12)
            if gap >= self.params.max_price_field_gap:
                return (
                    f"行情快照价格字段自相矛盾：current_price 与 close 相差 "
                    f"{gap * 100:.2f}%，该快照不能触发服务端盈利模型补仓开仓。"
                )

        normalized_side = str(side or "").lower()
        if normalized_side == "long":
            if (
                returns_20 <= -self.params.strong_contra_20m_pct
                and price_vs_sma20 < 0
                and price_vs_sma50 < 0
            ):
                return (
                    f"做多探针被阻止：20 分钟收益 {returns_20 * 100:.2f}% "
                    "且价格仍在短中期均线下方，短线结构没有支持追多。"
                )
        elif normalized_side == "short":
            if (
                returns_20 >= self.params.strong_contra_20m_pct
                and price_vs_sma20 > 0
                and price_vs_sma50 > 0
            ):
                return (
                    f"做空探针被阻止：20 分钟收益 {returns_20 * 100:.2f}% "
                    "且价格仍在短中期均线上方，短线结构没有支持追空。"
                )

        if 0 < volume_ratio < self.params.min_volume_ratio:
            return (
                f"当前成交量相对均值过低（volume_ratio={volume_ratio:.4f}），"
                "服务端盈利模型弱信号不能在低活跃盘口触发补仓开仓。"
            )
        return None
