"""Pure checks for live-analysis versus execution-environment compatibility."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Any

ENTRY_ENVIRONMENT_PRICE_MAX_DRIFT_FRACTION = 0.01

_TEXT_FIELDS = (
    "instId",
    "uly",
    "ctValCcy",
    "settleCcy",
    "ctType",
)
_IDENTITY_NUMERIC_FIELDS = (
    "ctVal",
    "ctMult",
)
_EXECUTION_RULE_FIELDS = (
    "lotSz",
    "minSz",
    "tickSz",
)


def _row(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return str(value or "").strip().upper()


def _decimal(value: Any, *, default: str = "0") -> Decimal:
    try:
        return Decimal(str(value if value not in (None, "") else default))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def _ticker_price(value: Any) -> float:
    row = _row(value)
    rows = row.get("data")
    if isinstance(rows, list) and rows:
        row = _row(rows[0])
    for key in ("markPx", "last", "lastPx", "bidPx", "askPx"):
        try:
            parsed = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if isfinite(parsed) and parsed > 0:
            return parsed
    return 0.0


def assess_okx_entry_environment_compatibility(
    *,
    live_instrument: Any,
    execution_instrument: Any,
    live_ticker: Any,
    execution_ticker: Any,
    max_price_drift_fraction: float = ENTRY_ENVIRONMENT_PRICE_MAX_DRIFT_FRACTION,
) -> dict[str, Any]:
    """Return an auditable compatibility decision for one candidate symbol.

    Analysis uses the live public market while paper orders use the demo
    execution environment. A paper candidate is executable only when both
    environments address the same contract and their current prices are close
    enough that the analysis is still meaningful.
    """

    live = _row(live_instrument)
    execution = _row(execution_instrument)
    blockers: list[str] = []
    if not live:
        blockers.append("live_instrument_missing")
    if not execution:
        blockers.append("execution_instrument_missing")

    if live and execution:
        for field in _TEXT_FIELDS:
            left = _text(live.get(field))
            right = _text(execution.get(field))
            if left != right:
                blockers.append(f"{field}_mismatch")
        for field in _IDENTITY_NUMERIC_FIELDS:
            # OKX omits ctMult for some linear contracts; the SDK treats it as 1.
            default = "1" if field == "ctMult" else "0"
            left = _decimal(live.get(field), default=default)
            right = _decimal(execution.get(field), default=default)
            if left != right:
                blockers.append(f"{field}_mismatch")
        if _decimal(live.get("ctVal")) <= 0 or _decimal(execution.get("ctVal")) <= 0:
            blockers.append("ctVal_missing_or_invalid")
        for field in _EXECUTION_RULE_FIELDS:
            if _decimal(execution.get(field)) <= 0:
                blockers.append(f"execution_{field}_missing_or_invalid")

    live_price = _ticker_price(live_ticker)
    execution_price = _ticker_price(execution_ticker)
    price_drift = None
    if live_price <= 0 or execution_price <= 0:
        blockers.append("environment_price_missing")
    else:
        price_drift = abs(execution_price - live_price) / max(live_price, execution_price)
        if price_drift > max(float(max_price_drift_fraction), 0.0):
            blockers.append("environment_price_drift_exceeded")

    # Keep one blocker per category in diagnostics while preserving the exact
    # list for operators and tests.
    deduped_blockers = list(dict.fromkeys(blockers))
    operational_differences = [
        f"{field}_mismatch"
        for field in _EXECUTION_RULE_FIELDS
        if live and execution and _decimal(live.get(field)) != _decimal(execution.get(field))
    ]
    return {
        "compatible": not deduped_blockers,
        "reason": (
            "live_analysis_and_execution_environment_compatible"
            if not deduped_blockers
            else "okx_live_execution_environment_incompatible"
        ),
        "blockers": deduped_blockers,
        "price_drift_fraction": price_drift,
        "max_price_drift_fraction": max(float(max_price_drift_fraction), 0.0),
        "live_price": live_price,
        "execution_price": execution_price,
        "live_inst_id": _text(live.get("instId")),
        "execution_inst_id": _text(execution.get("instId")),
        "live_uly": _text(live.get("uly")),
        "execution_uly": _text(execution.get("uly")),
        "operational_rule_differences": operational_differences,
        "spec_fields": {
            field: {
                "live": live.get(field),
                "execution": execution.get(field),
            }
            for field in (*_TEXT_FIELDS, *_IDENTITY_NUMERIC_FIELDS, *_EXECUTION_RULE_FIELDS)
        },
    }
