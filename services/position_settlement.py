"""Shared helpers for storing closed-position settlement snapshots.

The position row is the durable settlement cache.  Reconciliation may improve
the cache later, but dashboards and training should not have to re-derive fees
and PnL from scratch every time they read history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

SETTLEMENT_FORMULA = "close_fill_pnl + funding_fee - entry_fee - close_fee"
SETTLEMENT_STATUS_SETTLING = "settling"
SETTLEMENT_STATUS_EXCEPTION = "settlement_exception"
FINAL_SETTLEMENT_STATUSES = frozenset({"reconciled", "settled", "okx_position_history"})


def final_settlement_status_values() -> tuple[str, ...]:
    return tuple(sorted(FINAL_SETTLEMENT_STATUSES))


def is_final_settlement_status(value: Any) -> bool:
    return str(value or "").strip() in FINAL_SETTLEMENT_STATUSES


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def proportional_signed_value(value: float | None, close_qty: float, total_qty: float) -> float:
    amount = safe_float(value, 0.0)
    close = safe_float(close_qty, 0.0)
    total = safe_float(total_qty, 0.0)
    if amount == 0.0 or close <= 0:
        return 0.0
    if total <= 0:
        return amount
    return amount * min(close / total, 1.0)


@dataclass(frozen=True, slots=True)
class PositionSettlementSnapshot:
    close_fill_pnl: float
    entry_fee: float
    close_fee: float
    funding_fee: float = 0.0
    status: str = "provisional"
    source: str = "system_execution"
    synced_at: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def realized_pnl(self) -> float:
        return self.close_fill_pnl + self.funding_fee - self.entry_fee - self.close_fee

    def as_position_payload(self) -> dict[str, Any]:
        raw = {
            "formula": SETTLEMENT_FORMULA,
            "status": self.status,
            "source": self.source,
            **dict(self.raw or {}),
        }
        return {
            "realized_pnl": self.realized_pnl,
            "close_fill_pnl": self.close_fill_pnl,
            "entry_fee": self.entry_fee,
            "close_fee": self.close_fee,
            "funding_fee": self.funding_fee,
            "settlement_status": self.status,
            "settlement_source": self.source,
            "settlement_synced_at": self.synced_at or datetime.now(UTC),
            "settlement_raw": raw,
        }


def build_position_settlement_snapshot(
    *,
    close_fill_pnl: float,
    entry_fee: float,
    close_fee: float,
    funding_fee: float | None = 0.0,
    status: str = "provisional",
    source: str = "system_execution",
    synced_at: datetime | None = None,
    raw: dict[str, Any] | None = None,
) -> PositionSettlementSnapshot:
    return PositionSettlementSnapshot(
        close_fill_pnl=safe_float(close_fill_pnl, 0.0),
        entry_fee=abs(safe_float(entry_fee, 0.0)),
        close_fee=abs(safe_float(close_fee, 0.0)),
        funding_fee=safe_float(funding_fee, 0.0),
        status=str(status or "provisional").strip() or "provisional",
        source=str(source or "system_execution").strip() or "system_execution",
        synced_at=synced_at,
        raw=dict(raw or {}),
    )


def apply_position_settlement_snapshot(
    position: Any,
    snapshot: PositionSettlementSnapshot,
) -> None:
    for key, value in snapshot.as_position_payload().items():
        setattr(position, key, value)


def settlement_payload_fields(snapshot: PositionSettlementSnapshot) -> dict[str, Any]:
    return snapshot.as_position_payload()


def funding_fee_from_payload(payload: Any) -> tuple[float, str]:
    """Extract a funding fee from an execution payload when the executor has one.

    Most OKX close order callbacks do not include funding; in that case callers
    store zero with source ``not_available_at_close`` and later reconciliation
    can update the same snapshot from account bills.
    """

    candidates: list[tuple[Any, str]] = []
    if isinstance(payload, dict):
        candidates.extend(
            [
                (payload.get("funding_fee"), "payload.funding_fee"),
                (payload.get("fundingFee"), "payload.fundingFee"),
                (payload.get("funding"), "payload.funding"),
            ]
        )
        native = payload.get("native_close_fill")
        if isinstance(native, dict):
            candidates.extend(
                [
                    (native.get("funding_fee"), "native_close_fill.funding_fee"),
                    (native.get("fundingFee"), "native_close_fill.fundingFee"),
                ]
            )
        info = payload.get("info")
        if isinstance(info, dict):
            candidates.extend(
                [
                    (info.get("funding_fee"), "info.funding_fee"),
                    (info.get("fundingFee"), "info.fundingFee"),
                ]
            )
    for value, source in candidates:
        if value is None:
            continue
        parsed = safe_float(value, 0.0)
        if abs(parsed) > 1e-12:
            return parsed, source
        return 0.0, source
    return 0.0, "not_available_at_close"
