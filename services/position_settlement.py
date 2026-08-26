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
SETTLEMENT_STATUS_UNRESOLVED = "settlement_unresolved"
FINAL_SETTLEMENT_STATUSES = frozenset({"reconciled", "settled", "okx_position_history"})
TERMINAL_SETTLEMENT_STATUSES = frozenset({SETTLEMENT_STATUS_UNRESOLVED})
SETTLEMENT_DISPLAY_STATE_SETTLED = "settled"
SETTLEMENT_DISPLAY_STATE_LIFECYCLE_OPEN = "lifecycle_open"
SETTLEMENT_DISPLAY_STATE_PENDING = "pending_authority"
SETTLEMENT_DISPLAY_STATE_IDENTITY_UNRESOLVED = "identity_unresolved"
SETTLEMENT_DISPLAY_STATE_EVIDENCE_UNRESOLVED = "evidence_unresolved"
SETTLEMENT_DISPLAY_STATE_STOPPED = "stopped_waiting"

_IDENTITY_UNRESOLVED_REASONS = frozenset(
    {
        "official_position_history_identity_unresolved",
        "positions_history_no_matching_row",
        "positions_history_ambiguous_match",
        "official_history_identity_unresolved",
    }
)
_EVIDENCE_UNRESOLVED_REASONS = frozenset(
    {
        "lifecycle_fragment_contract_conservation_unresolved",
        "lifecycle_fragment_allocation_incomplete",
        "settlement_algebra_mismatch",
        "close_fill_contracts_history_mismatch",
        "entry_fill_contracts_history_mismatch",
    }
)


def final_settlement_status_values() -> tuple[str, ...]:
    return tuple(sorted(FINAL_SETTLEMENT_STATUSES))


def is_final_settlement_status(value: Any) -> bool:
    return str(value or "").strip() in FINAL_SETTLEMENT_STATUSES


def is_terminal_settlement_status(value: Any) -> bool:
    """Return whether settlement ended without an authoritative fact.

    Terminal unresolved rows remain visible for audit, but are excluded from
    retry loops, official PnL, expert memory, and training until explicitly
    requeued by a repair operation.
    """

    return str(value or "").strip() in TERMINAL_SETTLEMENT_STATUSES


def settlement_display_state(
    status: Any,
    source: Any = "",
    raw: Any = None,
) -> dict[str, Any]:
    """Classify a settlement row for UI and downstream eligibility.

    The persisted status is intentionally kept backward compatible; this
    derived state prevents ``settling`` and identity quarantine rows from
    being presented as the same kind of wait.
    """

    status_text = str(status or "").strip()
    source_text = str(source or "").strip()
    details = dict(raw) if isinstance(raw, dict) else {}
    # Failure payloads may retain an older lifecycle-open ``reason`` for
    # audit.  The latest error/quarantine reason owns the current display.
    current_reason = next(
        (
            str(details.get(key) or "").strip()
            for key in ("last_error_code", "quarantine_reason", "reason")
            if str(details.get(key) or "").strip()
        ),
        "",
    )
    reason_values = {current_reason} if current_reason else set()
    if is_final_settlement_status(status_text):
        return {
            "code": SETTLEMENT_DISPLAY_STATE_SETTLED,
            "label": "\u5df2\u5b8c\u6210\u6743\u5a01\u7ed3\u7b97",
            "retryable": False,
            "trainable": True,
            "explanation": "\u5df2\u83b7\u53d6 OKX \u5b8c\u6574\u751f\u547d\u5468\u671f\u7ed3\u7b97\u4e8b\u5b9e",
        }
    if is_terminal_settlement_status(status_text):
        return {
            "code": SETTLEMENT_DISPLAY_STATE_STOPPED,
            "label": "\u6743\u5a01\u7ed3\u7b97\u672a\u5b8c\u6210\uff0c\u5df2\u505c\u6b62\u81ea\u52a8\u7b49\u5f85",
            "retryable": False,
            "trainable": False,
            "explanation": "\u8d85\u8fc7\u81ea\u52a8\u7b49\u5f85\u65f6\u9650\uff0c\u672a\u4f2a\u9020\u76c8\u4e8f\uff1b\u9700\u8981\u663e\u5f0f\u4fee\u590d\u540e\u624d\u80fd\u91cd\u65b0\u5165\u961f",
        }
    if status_text == "superseded_position_residual":
        return {
            "code": SETTLEMENT_DISPLAY_STATE_STOPPED,
            "label": "\u91cd\u590d\u4ed3\u4f4d\u6b8b\u7559\uff0c\u5df2\u505c\u6b62\u7b49\u5f85",
            "retryable": False,
            "trainable": False,
            "explanation": "\u8be5\u8bb0\u5f55\u662f\u540c\u4e00 OKX \u751f\u547d\u5468\u671f\u7684\u91cd\u590d\u6b8b\u7559\uff0c\u4e0d\u4f5c\u4e3a\u5355\u72ec\u6743\u5a01\u7ed3\u7b97\u4e5f\u4e0d\u518d\u91cd\u8bd5",
        }
    if (
        source_text == "okx_position_lifecycle_still_open"
        or "position_lifecycle_still_open" in reason_values
    ):
        return {
            "code": SETTLEMENT_DISPLAY_STATE_LIFECYCLE_OPEN,
            "label": "OKX \u4ed3\u4f4d\u751f\u547d\u5468\u671f\u4ecd\u5f00\u653e",
            "retryable": True,
            "trainable": False,
            "explanation": "\u540c\u4e00 posId \u4ecd\u6709\u5f00\u653e\u4ed3\u4f4d\uff0c\u5c40\u90e8\u5e73\u4ed3\u4e0d\u80fd\u63d0\u524d\u7ed3\u7b97",
        }
    if reason_values.intersection(_EVIDENCE_UNRESOLVED_REASONS):
        return {
            "code": SETTLEMENT_DISPLAY_STATE_EVIDENCE_UNRESOLVED,
            "label": "\u7ed3\u7b97\u8bc1\u636e\u65e0\u6cd5\u5b88\u6052",
            "retryable": True,
            "trainable": False,
            "explanation": "\u5c40\u90e8\u5e73\u4ed3\u5206\u7247\u4e0e OKX \u5b98\u65b9\u6570\u91cf\u6216\u7ecf\u6d4e\u4e8b\u5b9e\u5c1a\u672a\u5b88\u6052",
        }
    if (
        status_text == "settlement_quarantined"
        or source_text == "okx_position_history_identity_quarantine"
        or reason_values.intersection(_IDENTITY_UNRESOLVED_REASONS)
    ):
        return {
            "code": SETTLEMENT_DISPLAY_STATE_IDENTITY_UNRESOLVED,
            "label": "OKX \u6743\u5a01\u4ed3\u4f4d\u5386\u53f2\u8eab\u4efd\u672a\u786e\u8ba4",
            "retryable": True,
            "trainable": False,
            "explanation": "OKX \u5e73\u4ed3\u6210\u4ea4\u5df2\u786e\u8ba4\uff0c\u4f46\u65e0\u6cd5\u4e0e\u672c\u5730\u751f\u547d\u5468\u671f\u552f\u4e00\u5173\u8054",
        }
    return {
        "code": SETTLEMENT_DISPLAY_STATE_PENDING,
        "label": "\u7b49\u5f85 OKX \u6743\u5a01\u7ed3\u7b97",
        "retryable": status_text not in {"superseded_position_residual"},
        "trainable": False,
        "explanation": "\u5df2\u5e73\u4ed3\uff0c\u5c1a\u672a\u83b7\u53d6\u53ef\u4fe1\u7684 OKX \u6743\u5a01\u7ed3\u7b97\u4e8b\u5b9e",
    }


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
