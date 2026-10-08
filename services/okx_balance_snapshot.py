"""Shared validity rules for native and cached OKX account snapshots."""

from __future__ import annotations

from math import isfinite
from typing import Any

from services.okx_error_classifier import (
    extract_okx_error,
    is_okx_temporary_service_error,
)


def finite_balance_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def balance_snapshot_verified(snapshot: Any) -> bool:
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("verified") is False
        or snapshot.get("error")
        or snapshot.get("stale")
    ):
        return False
    equity = finite_balance_value(snapshot.get("equity"))
    free = finite_balance_value(snapshot.get("free"))
    if equity is None or free is None:
        return False
    # A native response with explicit zero values is authoritative, not an
    # endpoint outage. Unmarked all-zero placeholders carry no such evidence.
    return snapshot.get("verified") is True or equity > 0 or free > 0


def balance_snapshot_recovery_reason(snapshot: Any) -> str | None:
    """Allow bounded intermediate recovery, never order submission authority."""

    if balance_snapshot_verified(snapshot):
        return None
    error = snapshot.get("error") if isinstance(snapshot, dict) else None
    if error:
        code, _message = extract_okx_error(error)
        if code not in (None, "0") and not is_okx_temporary_service_error(error):
            return None
        text = str(error).lower()
        transport_failure = any(
            marker in text
            for marker in (
                "requesttimeout",
                "request timeout",
                "request timed out",
                "read timed out",
                "connection reset",
                "connection aborted",
                "connection closed",
                "balance snapshot request timed out",
            )
        )
        if is_okx_temporary_service_error(error) or transport_failure:
            return "okx_private_balance_temporary_service_error"
        if isinstance(snapshot, dict) and snapshot.get("error_kind") == "empty_response":
            return "okx_private_balance_empty_snapshot"
        return None
    if isinstance(snapshot, dict) and snapshot.get("verified") is True:
        return None
    return "okx_private_balance_empty_snapshot"
