from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.market_facts import market_fact_reasons, market_source_consistency_reasons


@dataclass(frozen=True, slots=True)
class MarketDataQualityIssue:
    """Structured market-data quality issue used for audit and training isolation."""

    code: str
    severity: str
    reason: str
    stage_label: str
    details: dict[str, float | int | str | bool]

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "reason": self.reason,
            "stage_label": self.stage_label,
            "details": self.details,
            "exclude_from_training": True,
            "training_quality_reason": f"market_data_quality:{self.code}",
        }


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
    ) -> None:
        self.market_value_reader = market_value_reader or MarketValueReader().read

    def reason(self, source: Any, *, stage_label: str = "下单前") -> str | None:
        issue = self.issue(source, stage_label=stage_label)
        return issue.reason if issue else None

    def issue(self, source: Any, *, stage_label: str = "下单前") -> MarketDataQualityIssue | None:
        try:
            market_fact_issue = self._market_fact_issue(source, stage_label)
            snapshot = self._snapshot(source)
        except (TypeError, ValueError, AttributeError):
            return self._issue(
                "market_payload_invalid",
                stage_label,
                "行情数据异常，无法确认真实价格和盘口，本次不执行新开仓。",
                {},
            )

        if market_fact_issue:
            return market_fact_issue

        price = self._reference_price(snapshot)
        if price <= 0:
            return self._issue(
                "missing_valid_price",
                stage_label,
                "没有有效价格，本次不执行新开仓。",
                snapshot,
            )

        for checker in (
            self._indicator_snapshot_issue,
            self._price_source_split_issue,
            self._outside_24h_range_issue,
            self._depth_issue,
            self._stale_zero_returns_issue,
        ):
            issue = checker(snapshot, price, stage_label)
            if issue:
                return issue
        return None

    @staticmethod
    def _reference_price(snapshot: dict[str, float | int]) -> float:
        return float(
            snapshot["current_price"]
            or snapshot["close_price"]
            or snapshot["bid"]
            or snapshot["ask"]
        )

    def _snapshot(self, source: Any) -> dict[str, Any]:
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
            "indicator_snapshot_available": self.market_value_reader(
                source,
                "indicator_snapshot_available",
                None,
            ),
            "indicator_snapshot_quality": str(
                self.market_value_reader(source, "indicator_snapshot_quality", "") or ""
            ),
            "indicator_snapshot_reason": str(
                self.market_value_reader(source, "indicator_snapshot_reason", "") or ""
            ),
        }

    def _read_float(self, source: Any, key: str) -> float:
        return _safe_float(self.market_value_reader(source, key, 0.0), 0.0)

    def _indicator_snapshot_issue(
        self,
        snapshot: dict[str, Any],
        _price: float,
        stage_label: str,
    ) -> MarketDataQualityIssue | None:
        marker = snapshot.get("indicator_snapshot_available")
        if marker is None:
            return None
        if isinstance(marker, str):
            marker = marker.strip().lower() in {"1", "true", "yes", "y"}
        if bool(marker):
            return None
        reason_code = str(
            snapshot.get("indicator_snapshot_reason") or "indicator_snapshot_unavailable"
        )
        return self._issue(
            "indicator_snapshot_unavailable",
            stage_label,
            (
                "技术指标快照不可用，当前 RSI、ADX、波动率和成交量比仅是数据结构默认值，"
                "不能作为方向判断或开仓证据，本次不送入开仓专家分析。"
            ),
            {
                "indicator_snapshot_available": False,
                "indicator_snapshot_quality": str(
                    snapshot.get("indicator_snapshot_quality") or "unavailable"
                ),
                "indicator_snapshot_reason": reason_code,
            },
        )

    def _price_source_split_issue(
        self,
        snapshot: dict[str, float | int],
        _price: float,
        stage_label: str,
    ) -> MarketDataQualityIssue | None:
        bid = float(snapshot["bid"])
        ask = float(snapshot["ask"])
        if bid > 0 and ask > 0 and bid > ask:
            return self._issue(
                "crossed_bid_ask",
                stage_label,
                f"盘口结构无效：买一 {bid:g} 高于卖一 {ask:g}，本次不执行新开仓。",
                snapshot,
            )

        # The indicator close belongs to a completed candle while current/bid/ask
        # belong to the live quote. Their ordinary time-basis difference is priced
        # by the live spread and pre-order refresh, not treated as corrupted data.
        return None

    def _market_fact_issue(
        self,
        source: Any,
        stage_label: str,
    ) -> MarketDataQualityIssue | None:
        fact = self.market_value_reader(source, "market_fact", None)
        if not isinstance(fact, dict):
            return self._issue(
                "native_market_fact_missing",
                stage_label,
                "缺少绑定 OKX 原生合约身份的市场事实，本次不执行新开仓。",
                {"market_fact_present": False},
            )

        quality = fact.get("quality")
        quality = quality if isinstance(quality, dict) else {}
        declared_status = str(quality.get("status") or "").strip().lower()
        declared_reasons = [
            str(reason).strip()
            for reason in quality.get("reasons") or []
            if str(reason).strip()
        ]
        verified_reasons = market_fact_reasons(fact)
        source_consistency = fact.get("source_consistency")
        if not isinstance(source_consistency, dict):
            verified_reasons.append("source_consistency_contract_missing")
        else:
            verified_reasons.extend(
                f"source_consistency:{reason}"
                for reason in market_source_consistency_reasons(source_consistency)
            )
        if not str(fact.get("fact_id") or "").startswith("sha256:"):
            verified_reasons.append("market_fact_id_missing")
        if not quality:
            verified_reasons.append("market_fact_quality_missing")
        reasons = list(dict.fromkeys([*verified_reasons, *declared_reasons]))
        if declared_status == "clean" and not reasons:
            return None
        if not reasons:
            reasons.append("market_fact_quality_not_clean")

        identity = fact.get("native_identity")
        identity = identity if isinstance(identity, dict) else {}
        return self._issue(
            "native_market_fact_invalid",
            stage_label,
            (
                "OKX 原生市场事实未通过执行资格校验："
                f"{', '.join(reasons)}。本次不执行新开仓。"
            ),
            {
                "market_fact_id": str(fact.get("fact_id") or ""),
                "market_fact_schema_version": str(fact.get("schema_version") or ""),
                "market_fact_quality_status": declared_status,
                "market_fact_reason_codes": ";".join(reasons),
                "inst_id": str(identity.get("inst_id") or ""),
                "inst_type": str(identity.get("inst_type") or ""),
                "uly": str(identity.get("uly") or ""),
                "contract_spec_version": str(
                    identity.get("contract_spec_version") or ""
                ),
                "source_interface": str(fact.get("source_interface") or ""),
                "source_endpoint": str(fact.get("source_endpoint") or ""),
                "source_channel": str(fact.get("source_channel") or ""),
                "source_timestamp_ms": int(fact.get("source_timestamp_ms") or 0),
            },
        )

    def _outside_24h_range_issue(
        self,
        snapshot: dict[str, float | int],
        price: float,
        stage_label: str,
    ) -> MarketDataQualityIssue | None:
        high_24h = float(snapshot["high_24h"])
        low_24h = float(snapshot["low_24h"])
        if high_24h <= 0 or low_24h <= 0 or high_24h < low_24h:
            return None
        if low_24h <= price <= high_24h:
            return None
        return self._issue(
            "price_outside_24h_range",
            stage_label,
            (
                f"行情价格与24小时区间矛盾：当前价 {price:g}，"
                f"24小时低点 {low_24h:g}、高点 {high_24h:g}。"
                "当前价已明显落在交易所24小时区间之外，行情快照可能串币、延迟或来源异常，"
                "本次不执行新开仓。"
            ),
            {
                **snapshot,
                "reference_price": price,
                "range_floor": round(low_24h, 12),
                "range_ceiling": round(high_24h, 12),
            },
        )

    def _depth_issue(
        self,
        snapshot: dict[str, float | int],
        _price: float,
        stage_label: str,
    ) -> MarketDataQualityIssue | None:
        bid_depth = float(snapshot["bid_depth"])
        ask_depth = float(snapshot["ask_depth"])
        imbalance = float(snapshot["imbalance"])
        if bid_depth > 0 and ask_depth > 0:
            return None
        return self._issue(
            "orderbook_depth_invalid",
            stage_label,
            (
                f"盘口深度异常：买盘深度 {bid_depth:.4g}、卖盘深度 {ask_depth:.4g}，"
                f"盘口失衡 {imbalance:.2f}。该币种当前流动性或盘口数据不可靠，本次不执行新开仓。"
            ),
            snapshot,
        )

    def _stale_zero_returns_issue(
        self,
        snapshot: dict[str, float | int],
        _price: float,
        stage_label: str,
    ) -> MarketDataQualityIssue | None:
        all_short_returns_zero = (
            abs(float(snapshot["returns_1"])) < 1e-12
            and abs(float(snapshot["returns_5"])) < 1e-12
            and abs(float(snapshot["returns_20"])) < 1e-12
            and abs(float(snapshot["volatility_20"])) < 1e-12
        )
        if not all_short_returns_zero:
            return None
        change_24h_pct = float(snapshot["change_24h_pct"])
        if change_24h_pct == 0:
            return None
        return self._issue(
            "short_cycle_features_missing",
            stage_label,
            (
                "短周期行情特征疑似缺失：1/5/20周期收益率和波动率都为0，"
                f"但24小时涨跌幅为 {change_24h_pct:.2f}%。"
                "本次不把不完整行情送入开仓执行。"
            ),
            snapshot,
        )

    @staticmethod
    def _issue(
        code: str,
        stage_label: str,
        reason: str,
        details: dict[str, Any],
    ) -> MarketDataQualityIssue:
        return MarketDataQualityIssue(
            code=code,
            severity="block_entry",
            reason=f"{stage_label}{reason}",
            stage_label=stage_label,
            details={key: value for key, value in details.items() if _json_scalar(value)},
        )


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None
