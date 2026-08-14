"""Deterministic market candidate-pool filter with a replayable funnel."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from core.reason_codes import ReasonCode, reason_evidence


@dataclass(frozen=True, slots=True)
class CandidatePoolConfig:
    whitelist: frozenset[str] = frozenset()
    blacklist: frozenset[str] = frozenset()
    allowed_asset_types: frozenset[str] = frozenset({"swap", "spot"})
    min_listing_age: timedelta = timedelta(days=7)
    min_history_coverage: float = 0.95
    min_quote_volume_usdt: float = 100_000.0
    max_spread_bps: float = 30.0
    min_price: float = 0.0
    max_price: float | None = None
    min_order_notional_usdt: float = 5.0
    max_volatility_pct: float | None = None
    strategy_id: str = "default"


@dataclass(frozen=True, slots=True)
class CandidatePoolResult:
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    funnel: list[dict[str, Any]]
    generated_at: str
    version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "read_only": True,
            "is_entry_gate": False,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "funnel": self.funnel,
            "generated_at": self.generated_at,
            "candidate_count": len(self.accepted) + len(self.rejected),
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "diagnostic_boundary": (
                "Candidate generation diagnostics only; execution and OKX instrument authority remain unchanged."
            ),
        }


class CandidatePool:
    """Run the documented filter order and retain every removal reason."""

    VERSION = "bb.candidate-pool.v1"
    FILTERS = (
        "exchange_status",
        "whitelist_blacklist",
        "asset_type",
        "listing_age",
        "data_completeness",
        "liquidity",
        "spread",
        "price_order_size",
        "volatility_anomaly",
        "strategy_compatibility",
        "protection",
    )

    def __init__(self, config: CandidatePoolConfig | None = None) -> None:
        self.config = config or CandidatePoolConfig()

    def build(
        self,
        candidates: list[dict[str, Any]],
        *,
        now: datetime | None = None,
    ) -> CandidatePoolResult:
        checked_at = (now or datetime.now(UTC)).astimezone(UTC)
        current = [dict(item) for item in candidates if isinstance(item, dict)]
        rejected: list[dict[str, Any]] = []
        funnel: list[dict[str, Any]] = []
        for name in self.FILTERS:
            passed: list[dict[str, Any]] = []
            removed: list[dict[str, Any]] = []
            for item in current:
                reason = self._reason(name, item, checked_at)
                if reason is None:
                    passed.append(item)
                    continue
                symbol = str(item.get("symbol") or "")
                removal = {
                    "symbol": symbol,
                    "filter": name,
                    "reason_code": reason["code"],
                    "reason": reason["summary"],
                    "reason_evidence": reason_evidence(
                        reason["code"],
                        stage="candidate_pool",
                        blocker=True,
                        observed_value=reason.get("observed_value"),
                        threshold=reason.get("threshold"),
                        triggered_at=checked_at,
                        source_event=name,
                        evidence_summary=reason["summary"],
                    ),
                }
                removed.append(removal)
                rejected.append(removal)
            funnel.append(
                {
                    "filter": name,
                    "input_count": len(current),
                    "output_count": len(passed),
                    "removed_count": len(removed),
                    "removed_symbols": [row["symbol"] for row in removed],
                    "removed_reason_codes": [row["reason_code"] for row in removed],
                }
            )
            current = passed
        accepted = [
            {
                **item,
                "candidate_pool": {"accepted": True, "generated_at": checked_at.isoformat()},
            }
            for item in current
        ]
        return CandidatePoolResult(
            accepted=accepted,
            rejected=rejected,
            funnel=funnel,
            generated_at=checked_at.isoformat(),
            version=self.VERSION,
        )

    def _reason(self, name: str, item: dict[str, Any], now: datetime) -> dict[str, Any] | None:
        config = self.config
        symbol = str(item.get("symbol") or "").strip()
        if name == "exchange_status":
            if item.get("exchange_available", True) is not True:
                return self._r(ReasonCode.MARKET_EXCHANGE_UNAVAILABLE, "exchange does not expose this instrument")
        elif name == "whitelist_blacklist":
            if config.whitelist and symbol not in config.whitelist:
                return self._r(ReasonCode.MARKET_NOT_WHITELISTED, "symbol is outside the configured whitelist")
            if symbol in config.blacklist or item.get("blacklisted") is True:
                return self._r(ReasonCode.MARKET_BLACKLISTED, "symbol is blacklisted")
        elif name == "asset_type":
            if str(item.get("asset_type") or "swap") not in config.allowed_asset_types:
                return self._r(ReasonCode.MARKET_PRICE_CONSTRAINT, "asset type is not strategy-supported")
        elif name == "listing_age":
            listed = _as_utc(item.get("listed_at"))
            if listed and now - listed < config.min_listing_age:
                return self._r(ReasonCode.MARKET_DATA_INCOMPLETE, "instrument has not met minimum listing age", now - listed, config.min_listing_age)
        elif name == "data_completeness":
            coverage = _number(item.get("history_coverage"))
            if coverage is not None and coverage < config.min_history_coverage:
                return self._r(ReasonCode.MARKET_DATA_INCOMPLETE, "historical data coverage is below threshold", coverage, config.min_history_coverage)
            if item.get("data_complete") is False:
                return self._r(ReasonCode.MARKET_DATA_INCOMPLETE, "historical data is incomplete")
        elif name == "liquidity":
            volume = _number(item.get("quote_volume_24h_usdt"))
            if volume is not None and volume < config.min_quote_volume_usdt:
                return self._r(ReasonCode.MARKET_LIQUIDITY, "24h quote volume is below threshold", volume, config.min_quote_volume_usdt)
        elif name == "spread":
            spread = _number(item.get("spread_bps"))
            if spread is not None and spread > config.max_spread_bps:
                return self._r(ReasonCode.MARKET_SPREAD, "spread is above threshold", spread, config.max_spread_bps)
        elif name == "price_order_size":
            price = _number(item.get("price"))
            min_notional = _number(item.get("min_order_notional_usdt")) or 0.0
            if price is not None and price < config.min_price:
                return self._r(ReasonCode.MARKET_PRICE_CONSTRAINT, "price is below configured minimum", price, config.min_price)
            if config.max_price is not None and price is not None and price > config.max_price:
                return self._r(ReasonCode.MARKET_PRICE_CONSTRAINT, "price is above configured maximum", price, config.max_price)
            if min_notional > config.min_order_notional_usdt:
                return self._r(ReasonCode.MARKET_PRICE_CONSTRAINT, "minimum order notional exceeds account policy", min_notional, config.min_order_notional_usdt)
        elif name == "volatility_anomaly":
            volatility = _number(item.get("volatility_pct"))
            if config.max_volatility_pct is not None and volatility is not None and volatility > config.max_volatility_pct:
                return self._r(ReasonCode.MARKET_VOLATILITY_ANOMALY, "volatility is above anomaly threshold", volatility, config.max_volatility_pct)
        elif name == "strategy_compatibility":
            compatible = item.get("strategy_compatible")
            if compatible is False or (
                isinstance(item.get("compatible_strategies"), (list, tuple, set))
                and config.strategy_id not in item["compatible_strategies"]
            ):
                return self._r(ReasonCode.SIGNAL_UNAVAILABLE, "strategy does not support this instrument")
        elif name == "protection":
            if item.get("protection_allowed") is False:
                return self._r(ReasonCode.RISK_COOLDOWN, "protection chain currently excludes this instrument")
        return None

    @staticmethod
    def _r(code: str, summary: str, observed_value: Any = None, threshold: Any = None) -> dict[str, Any]:
        return {"code": code, "summary": summary, "observed_value": observed_value, "threshold": threshold}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _as_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, str) and value.strip():
        try:
            value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return None
