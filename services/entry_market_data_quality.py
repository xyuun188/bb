from __future__ import annotations

from collections.abc import Callable
from typing import Any

from services.entry_wick_guard import ABNORMAL_WICK_ENTRY_BLOCK_MAX_PCT

ENTRY_PRICE_FIELD_SPLIT_BLOCK_PCT = 0.08
ENTRY_PRICE_24H_RANGE_TOLERANCE_PCT = 0.03
ENTRY_DATA_STALE_ZERO_RETURNS_MIN_24H_CHANGE = 0.003
DEFAULT_MAX_SLIPPAGE_PCT = 0.005


class MarketValueReader:
    """Read dict-like or object-like market payload values consistently."""

    @staticmethod
    def read(source: Any, key: str, default: Any = None) -> Any:
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)


class EntryMarketDataQualityPolicy:
    """Reject entry analysis/execution when market data is clearly unusable."""

    def __init__(
        self,
        *,
        market_value_reader: Callable[[Any, str, Any], Any] | None = None,
        max_slippage_pct_provider: Callable[[], float] | None = None,
    ) -> None:
        self.market_value_reader = market_value_reader or MarketValueReader().read
        self.max_slippage_pct_provider = max_slippage_pct_provider or (
            lambda: DEFAULT_MAX_SLIPPAGE_PCT
        )

    def reason(self, source: Any, *, stage_label: str = "下单前") -> str | None:
        try:
            snapshot = self._snapshot(source)
        except (TypeError, ValueError, AttributeError):
            return f"{stage_label}行情数据异常，无法确认真实价格和盘口，本次不执行新开仓。"

        price = (
            snapshot["current_price"]
            or snapshot["close_price"]
            or snapshot["bid"]
            or snapshot["ask"]
        )
        if price <= 0:
            return f"{stage_label}没有有效价格，本次不执行新开仓。"

        split_reason = self._price_source_split_reason(snapshot, stage_label)
        if split_reason:
            return split_reason

        range_reason = self._outside_24h_range_reason(snapshot, price, stage_label)
        if range_reason:
            return range_reason

        spread_reason = self._spread_reason(snapshot, stage_label)
        if spread_reason:
            return spread_reason

        depth_reason = self._depth_reason(snapshot, stage_label)
        if depth_reason:
            return depth_reason

        stale_reason = self._stale_zero_returns_reason(snapshot, stage_label)
        if stale_reason:
            return stale_reason

        if (
            snapshot["abnormal_wick_count"] > 0
            and snapshot["abnormal_wick_max"] >= ABNORMAL_WICK_ENTRY_BLOCK_MAX_PCT
        ):
            return (
                f"{stage_label}检测到近72小时异常插针，"
                f"最大振幅约 {snapshot['abnormal_wick_max']:.2f}%，"
                "该币种容易出现非连续成交价格，本次不执行新开仓。"
            )

        return None

    def _snapshot(self, source: Any) -> dict[str, float | int]:
        return {
            "current_price": self._read_float(source, "current_price"),
            "close_price": self._read_float(source, "close"),
            "bid": self._read_float(source, "bid"),
            "ask": self._read_float(source, "ask"),
            "returns_1": self._read_float(source, "returns_1"),
            "returns_5": self._read_float(source, "returns_5"),
            "returns_20": self._read_float(source, "returns_20"),
            "volatility_20": self._read_float(source, "volatility_20"),
            "change_24h_pct": self._read_float(source, "change_24h_pct"),
            "high_24h": self._read_float(source, "high_24h"),
            "low_24h": self._read_float(source, "low_24h"),
            "bid_depth": self._read_float(source, "orderbook_bid_depth"),
            "ask_depth": self._read_float(source, "orderbook_ask_depth"),
            "imbalance": self._read_float(source, "orderbook_imbalance"),
            "abnormal_wick_count": int(self._read_float(source, "abnormal_wick_count_72h")),
            "abnormal_wick_max": self._read_float(source, "abnormal_wick_max_pct"),
        }

    def _read_float(self, source: Any, key: str) -> float:
        return _safe_float(self.market_value_reader(source, key, 0.0), 0.0)

    @staticmethod
    def _price_source_split_reason(
        snapshot: dict[str, float | int],
        stage_label: str,
    ) -> str | None:
        reference_prices = [
            float(value)
            for value in (
                snapshot["current_price"],
                snapshot["close_price"],
                snapshot["bid"],
                snapshot["ask"],
            )
            if float(value) > 0
        ]
        if len(reference_prices) < 2:
            return None

        min_ref = min(reference_prices)
        max_ref = max(reference_prices)
        split = (max_ref - min_ref) / max(min_ref, 1e-12)
        if split < ENTRY_PRICE_FIELD_SPLIT_BLOCK_PCT:
            return None

        return (
            f"{stage_label}行情价格源分裂："
            f"current={snapshot['current_price']:g}、close={snapshot['close_price']:g}、"
            f"bid={snapshot['bid']:g}、ask={snapshot['ask']:g}，"
            f"最大差异约 {split * 100:.2f}%。"
            "这会导致止盈止损价格和交易所主订单价格不匹配，本次不执行新开仓。"
        )

    @staticmethod
    def _outside_24h_range_reason(
        snapshot: dict[str, float | int],
        price: float,
        stage_label: str,
    ) -> str | None:
        high_24h = float(snapshot["high_24h"])
        low_24h = float(snapshot["low_24h"])
        if high_24h <= 0 or low_24h <= 0 or high_24h < low_24h:
            return None

        range_floor = low_24h * (1.0 - ENTRY_PRICE_24H_RANGE_TOLERANCE_PCT)
        range_ceiling = high_24h * (1.0 + ENTRY_PRICE_24H_RANGE_TOLERANCE_PCT)
        if range_floor <= price <= range_ceiling:
            return None

        return (
            f"{stage_label}行情价格与24小时区间矛盾：当前价 {price:g}，"
            f"24小时低点 {low_24h:g}、高点 {high_24h:g}。"
            "当前价已明显落在交易所24小时区间之外，行情快照可能串币、延迟或来源异常，"
            "本次不执行新开仓。"
        )

    def _spread_reason(self, snapshot: dict[str, float | int], stage_label: str) -> str | None:
        bid = float(snapshot["bid"])
        ask = float(snapshot["ask"])
        if bid <= 0 or ask <= 0 or ask < bid:
            return None

        spread = (ask - bid) / max((ask + bid) / 2.0, 1e-12)
        threshold = max(
            _safe_float(self.max_slippage_pct_provider(), DEFAULT_MAX_SLIPPAGE_PCT) * 2.0, 0.012
        )
        if spread <= threshold:
            return None

        return (
            f"{stage_label}盘口价差过大：买一 {bid:g} / 卖一 {ask:g}，"
            f"价差约 {spread * 100:.2f}%，容易产生明显滑点，本次不执行新开仓。"
        )

    @staticmethod
    def _depth_reason(snapshot: dict[str, float | int], stage_label: str) -> str | None:
        bid_depth = float(snapshot["bid_depth"])
        ask_depth = float(snapshot["ask_depth"])
        imbalance = float(snapshot["imbalance"])
        if (bid_depth > 0 and ask_depth > 0) or abs(imbalance) < 0.98:
            return None

        return (
            f"{stage_label}盘口深度异常：买盘深度 {bid_depth:.4g}、"
            f"卖盘深度 {ask_depth:.4g}，盘口失衡 {imbalance:.2f}。"
            "该币种当前流动性或盘口数据不可靠，本次不执行新开仓。"
        )

    @staticmethod
    def _stale_zero_returns_reason(
        snapshot: dict[str, float | int],
        stage_label: str,
    ) -> str | None:
        all_short_returns_zero = (
            abs(float(snapshot["returns_1"])) < 1e-12
            and abs(float(snapshot["returns_5"])) < 1e-12
            and abs(float(snapshot["returns_20"])) < 1e-12
            and abs(float(snapshot["volatility_20"])) < 1e-12
        )
        if not all_short_returns_zero:
            return None

        change_24h_pct = float(snapshot["change_24h_pct"])
        if abs(change_24h_pct) < ENTRY_DATA_STALE_ZERO_RETURNS_MIN_24H_CHANGE * 100:
            return None

        return (
            f"{stage_label}短周期行情特征疑似缺失：1/5/20周期收益率和波动率都为0，"
            f"但24小时涨跌幅为 {change_24h_pct:.2f}%。"
            "本次不把不完整行情送入开仓执行。"
        )


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default
