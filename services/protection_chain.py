"""Side-effect-free protection diagnostics for new-position eligibility.

The production risk engine remains the authority that blocks orders.  This
module evaluates the same categories into a stable, replayable evidence shape
so a protection can be inspected before it is enabled as an enforcement rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.reason_codes import ReasonCode, reason_evidence


@dataclass(frozen=True, slots=True)
class ProtectionThresholds:
    max_consecutive_losses: int = 3
    max_drawdown_pct: float = 10.0
    min_pair_fee_adjusted_return_pct: float = 0.0
    min_pair_trade_count: int = 10
    max_spread_bps: float = 30.0
    min_liquidity_usdt: float = 100_000.0


@dataclass(frozen=True, slots=True)
class ProtectionObservation:
    scope: str
    symbol: str | None = None
    strategy_id: str | None = None
    cooldown_until: datetime | str | None = None
    consecutive_losses: int = 0
    loss_lock_until: datetime | str | None = None
    drawdown_pct: float = 0.0
    drawdown_release_at: datetime | str | None = None
    pair_fee_adjusted_return_pct: float | None = None
    pair_trade_count: int = 0
    pair_reevaluate_at: datetime | str | None = None
    spread_bps: float | None = None
    liquidity_usdt: float | None = None
    exchange_healthy: bool = True
    exchange_health_reason: str = ""
    exchange_recheck_at: datetime | str | None = None
    source_event: str = ""
    observed_at: datetime | str | None = None
    manual_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


class ProtectionChain:
    """Evaluate protections in a deterministic order without mutating state."""

    VERSION = "bb.protection-chain.v1"
    ORDER = (
        "cooldown",
        "consecutive_loss",
        "max_drawdown",
        "low_profit_pair",
        "spread_liquidity",
        "exchange_health",
    )

    def __init__(self, thresholds: ProtectionThresholds | None = None) -> None:
        self.thresholds = thresholds or ProtectionThresholds()

    def evaluate(
        self,
        observation: ProtectionObservation,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        checked_at = _as_utc(now) or datetime.now(UTC)
        states = [
            self._cooldown(observation, checked_at),
            self._consecutive_loss(observation, checked_at),
            self._max_drawdown(observation, checked_at),
            self._low_profit_pair(observation, checked_at),
            self._spread_liquidity(observation, checked_at),
            self._exchange_health(observation, checked_at),
        ]
        active = [state for state in states if state["active"]]
        primary = active[0] if active else None
        return {
            "version": self.VERSION,
            "read_only": True,
            "is_entry_gate": False,
            "scope": observation.scope,
            "symbol": observation.symbol,
            "strategy_id": observation.strategy_id,
            "checked_at": checked_at.isoformat(),
            "allowed_by_diagnostics": not active,
            "active_count": len(active),
            "primary_reason_code": primary["reason_code"] if primary else None,
            "states": states,
            "boundary": (
                "Diagnostic only. The existing risk engine remains authoritative for order blocking."
            ),
        }

    def _state(
        self,
        observation: ProtectionObservation,
        *,
        rule: str,
        active: bool,
        code: str,
        observed_value: Any,
        threshold: Any,
        checked_at: datetime,
        release_at: datetime | str | None,
        action: str,
        evidence_summary: str,
    ) -> dict[str, Any]:
        override = observation.manual_overrides.get(rule)
        effective_active = bool(active and not _valid_override(override))
        evidence = reason_evidence(
            code,
            stage="risk_check",
            blocker=effective_active,
            observed_value=observed_value,
            threshold=threshold,
            triggered_at=observation.observed_at or checked_at,
            release_at=release_at,
            source_event=observation.source_event or rule,
            evidence_summary=evidence_summary,
        )
        return {
            "rule": rule,
            "scope": observation.scope,
            "active": effective_active,
            "triggered": bool(active),
            "reason_code": code,
            "action": action if effective_active else "none",
            "observed_value": observed_value,
            "threshold": threshold,
            "started_at": evidence["triggered_at"],
            "release_at": evidence["release_at"],
            "source_event": evidence["source_event"],
            "evidence_summary": evidence_summary,
            "manual_override": dict(override) if isinstance(override, dict) else None,
            "reason_evidence": evidence,
        }

    def _cooldown(self, observation: ProtectionObservation, checked_at: datetime) -> dict[str, Any]:
        release = _as_utc(observation.cooldown_until)
        active = bool(release and checked_at < release)
        return self._state(
            observation,
            rule="cooldown",
            active=active,
            code=ReasonCode.RISK_COOLDOWN,
            observed_value=checked_at.isoformat(),
            threshold=release.isoformat() if release else None,
            checked_at=checked_at,
            release_at=release,
            action="block_new_entry",
            evidence_summary="symbol or strategy cooldown is still active",
        )

    def _consecutive_loss(
        self,
        observation: ProtectionObservation,
        checked_at: datetime,
    ) -> dict[str, Any]:
        threshold = max(int(self.thresholds.max_consecutive_losses), 1)
        return self._state(
            observation,
            rule="consecutive_loss",
            active=int(observation.consecutive_losses) >= threshold,
            code=ReasonCode.RISK_CONSECUTIVE_LOSS,
            observed_value=int(observation.consecutive_losses),
            threshold=threshold,
            checked_at=checked_at,
            release_at=observation.loss_lock_until,
            action="lock_scope",
            evidence_summary="consecutive loss threshold reached",
        )

    def _max_drawdown(
        self,
        observation: ProtectionObservation,
        checked_at: datetime,
    ) -> dict[str, Any]:
        threshold = max(float(self.thresholds.max_drawdown_pct), 0.0)
        return self._state(
            observation,
            rule="max_drawdown",
            active=float(observation.drawdown_pct) >= threshold,
            code=ReasonCode.RISK_MAX_DRAWDOWN,
            observed_value=float(observation.drawdown_pct),
            threshold=threshold,
            checked_at=checked_at,
            release_at=observation.drawdown_release_at,
            action="halt_or_reduce_risk",
            evidence_summary="rolling drawdown threshold reached",
        )

    def _low_profit_pair(
        self,
        observation: ProtectionObservation,
        checked_at: datetime,
    ) -> dict[str, Any]:
        enough_samples = int(observation.pair_trade_count) >= max(
            int(self.thresholds.min_pair_trade_count), 1
        )
        value = observation.pair_fee_adjusted_return_pct
        active = bool(
            enough_samples
            and value is not None
            and float(value) < float(self.thresholds.min_pair_fee_adjusted_return_pct)
        )
        return self._state(
            observation,
            rule="low_profit_pair",
            active=active,
            code=ReasonCode.RISK_LOW_PROFIT_PAIR,
            observed_value={"return_pct": value, "trade_count": observation.pair_trade_count},
            threshold={
                "min_return_pct": self.thresholds.min_pair_fee_adjusted_return_pct,
                "min_trade_count": self.thresholds.min_pair_trade_count,
            },
            checked_at=checked_at,
            release_at=observation.pair_reevaluate_at,
            action="remove_pair",
            evidence_summary="pair fee-adjusted return is below threshold with enough samples",
        )

    def _spread_liquidity(
        self,
        observation: ProtectionObservation,
        checked_at: datetime,
    ) -> dict[str, Any]:
        spread_failed = (
            observation.spread_bps is not None
            and float(observation.spread_bps) > float(self.thresholds.max_spread_bps)
        )
        liquidity_failed = (
            observation.liquidity_usdt is not None
            and float(observation.liquidity_usdt) < float(self.thresholds.min_liquidity_usdt)
        )
        code = ReasonCode.MARKET_SPREAD if spread_failed else ReasonCode.MARKET_LIQUIDITY
        return self._state(
            observation,
            rule="spread_liquidity",
            active=bool(spread_failed or liquidity_failed),
            code=code,
            observed_value={
                "spread_bps": observation.spread_bps,
                "liquidity_usdt": observation.liquidity_usdt,
            },
            threshold={
                "max_spread_bps": self.thresholds.max_spread_bps,
                "min_liquidity_usdt": self.thresholds.min_liquidity_usdt,
            },
            checked_at=checked_at,
            release_at=None,
            action="reject_current_entry",
            evidence_summary="current spread or liquidity does not satisfy market quality limits",
        )

    def _exchange_health(
        self,
        observation: ProtectionObservation,
        checked_at: datetime,
    ) -> dict[str, Any]:
        return self._state(
            observation,
            rule="exchange_health",
            active=not bool(observation.exchange_healthy),
            code=ReasonCode.MARKET_EXCHANGE_UNAVAILABLE,
            observed_value={
                "healthy": bool(observation.exchange_healthy),
                "reason": observation.exchange_health_reason,
            },
            threshold={"healthy": True},
            checked_at=checked_at,
            release_at=observation.exchange_recheck_at,
            action="halt_new_risk",
            evidence_summary=observation.exchange_health_reason or "exchange health check failed",
        )


def _as_utc(value: datetime | str | None) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    return None


def _valid_override(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return bool(value.get("approved") and value.get("approved_by") and value.get("approved_at"))
