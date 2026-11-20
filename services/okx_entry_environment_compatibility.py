"""Pure checks for live-analysis versus execution-environment compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Any

ENTRY_ENVIRONMENT_PRICE_MAX_DRIFT_FRACTION = 0.01
ENTRY_ENVIRONMENT_COMPATIBILITY_VERSION = (
    "2026-11-20.okx-entry-environment-compatibility.v3"
)
ENTRY_INSTRUMENT_UNAVAILABLE_CACHE_SECONDS = 6 * 60 * 60
ENTRY_ENVIRONMENT_PRICE_DRIFT_RETRY_SECONDS = 10 * 60
ENTRY_ENVIRONMENT_COMPATIBILITY_RETRY_SECONDS = 60


def entry_instrument_definitively_unavailable(facts: Mapping[str, Any] | None) -> bool:
    """Only an explicit OKX capability rejection warrants the long retry window."""

    payload = facts if isinstance(facts, Mapping) else {}
    if payload.get("reason") == "okx_pre_order_execution_market_unavailable":
        compatibility = payload.get("environment_compatibility")
        compatibility = compatibility if isinstance(compatibility, Mapping) else {}
        blockers = compatibility.get("blockers")
        if isinstance(blockers, (list, tuple, set, frozenset)) and set(blockers) == {
            "execution_market_unavailable"
        }:
            return True
    return (
        str(payload.get("error_code") or "") == "51001"
        or str(payload.get("reason") or "") == "okx_private_entry_instrument_unavailable"
    )


def entry_instrument_unavailable_cache_ttl(facts: Mapping[str, Any] | None) -> float | None:
    """Use one retry policy for the executor and candidate capability cache."""

    payload = facts if isinstance(facts, Mapping) else {}
    if entry_instrument_definitively_unavailable(payload):
        return ENTRY_INSTRUMENT_UNAVAILABLE_CACHE_SECONDS
    if payload.get("reason") != "okx_entry_live_execution_environment_incompatible":
        return None
    compatibility = payload.get("environment_compatibility")
    compatibility = compatibility if isinstance(compatibility, Mapping) else {}
    blockers = set(compatibility.get("blockers") or ())
    if not blockers:
        return None
    if "environment_price_drift_exceeded" in blockers and blockers <= {
        "environment_price_drift_exceeded",
        "environment_executable_quote_drift_exceeded",
    }:
        return ENTRY_ENVIRONMENT_PRICE_DRIFT_RETRY_SECONDS
    return ENTRY_ENVIRONMENT_COMPATIBILITY_RETRY_SECONDS


_HARD_TEXT_FIELDS = (
    "instId",
    "ctValCcy",
    "settleCcy",
    "ctType",
)
_ALIAS_TEXT_FIELDS = ("uly",)
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


def _ticker_row(value: Any) -> dict[str, Any]:
    row = _row(value)
    rows = row.get("data")
    if isinstance(rows, list) and rows:
        return _row(rows[0])
    return row


def _positive_number(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if isfinite(parsed) and parsed > 0 else 0.0


def _ticker_price(row: dict[str, Any]) -> float:
    for key in ("markPx", "last", "lastPx", "bidPx", "askPx"):
        if parsed := _positive_number(row.get(key)):
            return parsed
    return 0.0


def _ticker_quote(row: dict[str, Any]) -> tuple[float, float] | None:
    bid = _positive_number(row.get("bidPx"))
    ask = _positive_number(row.get("askPx"))
    if not 0 < bid <= ask or any(
        key in row and not _positive_number(row[key]) for key in ("bidSz", "askSz")
    ):
        return None
    return bid, ask


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
    warnings: list[str] = []
    if not live:
        blockers.append("live_instrument_missing")
    if not execution:
        blockers.append("execution_instrument_missing")

    if live and execution:
        for field in _HARD_TEXT_FIELDS:
            left = _text(live.get(field))
            right = _text(execution.get(field))
            if left != right:
                blockers.append(f"{field}_mismatch")
        for field in _ALIAS_TEXT_FIELDS:
            left = _text(live.get(field))
            right = _text(execution.get(field))
            if left != right:
                warnings.append(f"{field}_alias_mismatch")
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

    live_quote_row = _ticker_row(live_ticker)
    execution_quote_row = _ticker_row(execution_ticker)
    live_price = _ticker_price(live_quote_row)
    execution_price = _ticker_price(execution_quote_row)
    max_drift = max(float(max_price_drift_fraction), 0.0)
    price_drift = None
    if live_price <= 0 or execution_price <= 0:
        blockers.append("environment_price_missing")
    else:
        price_drift = abs(execution_price - live_price) / max(live_price, execution_price)
        if price_drift > max_drift:
            blockers.append("environment_price_drift_exceeded")

    live_quote = _ticker_quote(live_quote_row)
    execution_quote = _ticker_quote(execution_quote_row)
    quote_drift: dict[str, float] = {}
    if execution_quote_row and execution_quote is None:
        blockers.append("execution_ticker_quote_incomplete")
    if live_quote is not None and execution_quote is not None:
        quote_drift = {
            side: abs(execution_px - live_px) / max(live_px, execution_px)
            for side, live_px, execution_px in zip(
                ("bid", "ask"), live_quote, execution_quote, strict=True
            )
        }
        if max(quote_drift.values()) > max_drift:
            blockers.append("environment_executable_quote_drift_exceeded")

    # Keep one blocker per category in diagnostics while preserving the exact
    # list for operators and tests.
    deduped_blockers = list(dict.fromkeys(blockers))
    operational_differences = [
        f"{field}_mismatch"
        for field in _EXECUTION_RULE_FIELDS
        if live and execution and _decimal(live.get(field)) != _decimal(execution.get(field))
    ]
    alias_differences = [
        f"{field}_alias_mismatch"
        for field in _ALIAS_TEXT_FIELDS
        if live and execution and _text(live.get(field)) != _text(execution.get(field))
    ]
    return {
        "contract_version": ENTRY_ENVIRONMENT_COMPATIBILITY_VERSION,
        "compatible": not deduped_blockers,
        "reason": (
            "live_analysis_and_execution_environment_compatible"
            if not deduped_blockers
            else "okx_live_execution_environment_incompatible"
        ),
        "blockers": deduped_blockers,
        "warnings": list(dict.fromkeys(warnings)),
        "price_drift_fraction": price_drift,
        "max_price_drift_fraction": max_drift,
        "live_price": live_price,
        "execution_price": execution_price,
        "quote_drift_fraction": quote_drift,
        "live_ticker_timestamp_ms": live_quote_row.get("ts"),
        "execution_ticker_timestamp_ms": execution_quote_row.get("ts"),
        "live_inst_id": _text(live.get("instId")),
        "execution_inst_id": _text(execution.get("instId")),
        "live_uly": _text(live.get("uly")),
        "execution_uly": _text(execution.get("uly")),
        "identity_alias_differences": alias_differences,
        "operational_rule_differences": operational_differences,
        "spec_fields": {
            field: {
                "live": live.get(field),
                "execution": execution.get(field),
            }
            for field in (
                *_HARD_TEXT_FIELDS,
                *_ALIAS_TEXT_FIELDS,
                *_IDENTITY_NUMERIC_FIELDS,
                *_EXECUTION_RULE_FIELDS,
            )
        },
    }
