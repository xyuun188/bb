"""Stable, machine-readable reasons for decisions and policy outcomes.

The human ``reason`` text on a decision is intentionally preserved for
backwards compatibility.  This module supplies the structured companion used
by audit and dashboard consumers.  It contains no trading side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


class ReasonCode:
    """Namespaced reason codes shared by the decision lifecycle."""

    SIGNAL_HOLD = "SIGNAL_HOLD"
    SIGNAL_ENTRY = "SIGNAL_ENTRY"
    SIGNAL_EXIT = "SIGNAL_EXIT"
    SIGNAL_UNAVAILABLE = "SIGNAL_UNAVAILABLE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_INVALID_OUTPUT = "MODEL_INVALID_OUTPUT"
    PROFIT_GATE_NON_POSITIVE = "PROFIT_GATE_NON_POSITIVE"
    PROFIT_GATE_INSUFFICIENT_EVIDENCE = "PROFIT_GATE_INSUFFICIENT_EVIDENCE"
    RISK_COOLDOWN = "RISK_COOLDOWN"
    RISK_CONSECUTIVE_LOSS = "RISK_CONSECUTIVE_LOSS"
    RISK_MAX_DRAWDOWN = "RISK_MAX_DRAWDOWN"
    RISK_LOW_PROFIT_PAIR = "RISK_LOW_PROFIT_PAIR"
    RISK_CAPACITY = "RISK_CAPACITY"
    MARKET_EXCHANGE_UNAVAILABLE = "MARKET_EXCHANGE_UNAVAILABLE"
    MARKET_NOT_WHITELISTED = "MARKET_NOT_WHITELISTED"
    MARKET_BLACKLISTED = "MARKET_BLACKLISTED"
    MARKET_DATA_INCOMPLETE = "MARKET_DATA_INCOMPLETE"
    MARKET_LIQUIDITY = "MARKET_LIQUIDITY"
    MARKET_SPREAD = "MARKET_SPREAD"
    MARKET_PRICE_CONSTRAINT = "MARKET_PRICE_CONSTRAINT"
    MARKET_VOLATILITY_ANOMALY = "MARKET_VOLATILITY_ANOMALY"
    ACCOUNT_BALANCE = "ACCOUNT_BALANCE"
    ACCOUNT_POSITION_LIMIT = "ACCOUNT_POSITION_LIMIT"
    EXCHANGE_REJECTED = "EXCHANGE_REJECTED"
    EXCHANGE_TIMEOUT = "EXCHANGE_TIMEOUT"
    EXCHANGE_CONFIRMATION_MISSING = "EXCHANGE_CONFIRMATION_MISSING"
    RECONCILIATION_PENDING = "RECONCILIATION_PENDING"
    RECONCILIATION_MISMATCH = "RECONCILIATION_MISMATCH"


@dataclass(frozen=True, slots=True)
class ReasonEvidence:
    """Evidence attached to a reason code, safe to persist as JSON."""

    code: str
    stage: str
    blocker: bool = True
    observed_value: Any = None
    threshold: Any = None
    triggered_at: str | None = None
    release_at: str | None = None
    source_event: str | None = None
    evidence_summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "blocker": bool(self.blocker),
            "observed_value": self.observed_value,
            "threshold": self.threshold,
            "triggered_at": self.triggered_at,
            "release_at": self.release_at,
            "source_event": self.source_event,
            "evidence_summary": self.evidence_summary,
        }


def reason_evidence(
    code: str,
    *,
    stage: str,
    blocker: bool = True,
    observed_value: Any = None,
    threshold: Any = None,
    triggered_at: datetime | str | None = None,
    release_at: datetime | str | None = None,
    source_event: str | None = None,
    evidence_summary: str = "",
) -> dict[str, Any]:
    """Build a normalized reason evidence object.

    Datetimes are normalized to UTC ISO-8601 strings.  Values are deliberately
    left opaque so callers can include the exact observation and threshold
    used by their policy without changing the contract.
    """

    normalized = str(code or "").strip().upper()
    if not normalized or "_" not in normalized:
        raise ValueError("reason code must be a non-empty namespaced identifier")
    return ReasonEvidence(
        code=normalized,
        stage=str(stage or "unknown"),
        blocker=bool(blocker),
        observed_value=observed_value,
        threshold=threshold,
        triggered_at=_timestamp(triggered_at),
        release_at=_timestamp(release_at),
        source_event=str(source_event or "") or None,
        evidence_summary=str(evidence_summary or "").strip(),
    ).as_dict()


def reason_code_for_blocker(blocker: str | None, reason: str | None = None) -> str:
    """Map existing blocker labels to stable codes without changing behavior."""

    key = str(blocker or "").strip().lower()
    mapping = {
        "entry_gate": ReasonCode.SIGNAL_UNAVAILABLE,
        "market_regime": ReasonCode.PROFIT_GATE_INSUFFICIENT_EVIDENCE,
        "entry_capacity": ReasonCode.RISK_CAPACITY,
        "cooldown": ReasonCode.RISK_COOLDOWN,
        "consecutive_loss": ReasonCode.RISK_CONSECUTIVE_LOSS,
        "max_drawdown": ReasonCode.RISK_MAX_DRAWDOWN,
        "low_profit_pair": ReasonCode.RISK_LOW_PROFIT_PAIR,
        "exchange": ReasonCode.EXCHANGE_REJECTED,
        "reconciliation": ReasonCode.RECONCILIATION_MISMATCH,
    }
    if key in mapping:
        return mapping[key]
    text = str(reason or "").lower()
    if "cooldown" in text:
        return ReasonCode.RISK_COOLDOWN
    if "timeout" in text or "timed out" in text:
        return ReasonCode.MODEL_TIMEOUT
    if "liquid" in text:
        return ReasonCode.MARKET_LIQUIDITY
    if "spread" in text:
        return ReasonCode.MARKET_SPREAD
    if "balance" in text or "margin" in text:
        return ReasonCode.ACCOUNT_BALANCE
    return "SYSTEM_POLICY_BLOCKED"


def infer_reason_code(stage: str, status: str, reason: str | None = None) -> str:
    """Provide a stable fallback for legacy callers that only pass text."""

    text = str(reason or "")
    lowered = text.lower()
    if "timeout" in lowered or "timed out" in lowered:
        return ReasonCode.MODEL_TIMEOUT
    if "cooldown" in lowered:
        return ReasonCode.RISK_COOLDOWN
    if "spread" in lowered:
        return ReasonCode.MARKET_SPREAD
    if "liquid" in lowered:
        return ReasonCode.MARKET_LIQUIDITY
    if stage == "ai_analysis":
        return ReasonCode.SIGNAL_ENTRY if status in {"completed", "passed"} else ReasonCode.SIGNAL_UNAVAILABLE
    if stage == "strategy_arbitration":
        return ReasonCode.SIGNAL_ENTRY if status in {"completed", "passed"} else ReasonCode.SIGNAL_HOLD
    if stage == "risk_check":
        return ReasonCode.RISK_CAPACITY if status in {"blocked", "failed"} else ReasonCode.SIGNAL_ENTRY
    if stage == "exchange_submit":
        return ReasonCode.EXCHANGE_REJECTED if status in {"blocked", "failed"} else ReasonCode.SIGNAL_ENTRY
    if stage == "exchange_confirm":
        return ReasonCode.EXCHANGE_CONFIRMATION_MISSING if status in {"blocked", "failed"} else ReasonCode.SIGNAL_ENTRY
    if stage == "local_sync":
        return ReasonCode.RECONCILIATION_MISMATCH if status in {"blocked", "failed"} else ReasonCode.RECONCILIATION_PENDING
    return "SYSTEM_POLICY_BLOCKED"


def _timestamp(value: datetime | str | None) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return normalized.isoformat()
    return str(value)
