"""Profit-loop training sample contract.

The contract accepts both profitable and losing closed trades.  A negative
return is a valid label; missing authoritative lifecycle facts are not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isclose, isfinite
from typing import Any, Literal

PROFIT_TRAINING_CONTRACT_VERSION = "2026-07-23.profit-loop-training.v1"
PROFIT_TRAINING_TARGET = "net_return_after_all_cost_pct"

DecisionAuthority = Literal["rules", "model", "manual", "system"]

REQUIRED_TEXT_FIELDS = (
    "symbol",
    "side",
    "entry_order_id",
    "close_order_id",
    "notional_source",
    "entry_fee_source",
    "close_fee_source",
    "pnl_source",
    "funding_fee_source",
    "slippage_source",
)
REQUIRED_NUMERIC_FIELDS = (
    "entry_price",
    "close_price",
    "quantity",
    "notional",
    "entry_fee",
    "close_fee",
    "funding_fee",
    "slippage",
    "realized_pnl",
    "net_return_after_all_cost_pct",
    "holding_minutes",
)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _fingerprint_payload(sample: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "symbol",
        "side",
        "entry_order_id",
        "close_order_id",
        "entry_price",
        "close_price",
        "quantity",
        "notional",
        "realized_pnl",
        "net_return_after_all_cost_pct",
    )
    return {key: sample.get(key) for key in keys}


def profit_sample_fingerprint(sample: dict[str, Any]) -> str:
    payload = json.dumps(
        _fingerprint_payload(sample),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ProfitTrainingContract:
    eligible: bool
    reason: str
    target: str
    target_value: float | None
    outcome: Literal["profit", "loss", "flat", "invalid"]
    decision_authority: str
    evidence_fingerprint: str
    blockers: tuple[str, ...]
    model_shadow_alignment: str
    version: str = PROFIT_TRAINING_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blockers"] = list(self.blockers)
        return payload


def _model_shadow_alignment(sample: dict[str, Any]) -> str:
    prediction = _safe_dict(sample.get("model_shadow_prediction"))
    predicted_side = _text(
        prediction.get("side")
        or prediction.get("action")
        or prediction.get("predicted_side")
    ).lower()
    actual_side = _text(sample.get("side")).lower()
    net_return = _safe_float(sample.get(PROFIT_TRAINING_TARGET))
    if not predicted_side:
        return "no_model_shadow_prediction"
    if predicted_side in {"buy", "open_long"}:
        predicted_side = "long"
    elif predicted_side in {"sell", "open_short"}:
        predicted_side = "short"
    if predicted_side not in {"long", "short"}:
        return "invalid_model_shadow_prediction"
    if net_return is None:
        return "unknown"
    if predicted_side == actual_side and net_return < 0:
        return "supported_losing_side"
    if predicted_side != actual_side and net_return < 0:
        return "avoided_losing_side"
    if predicted_side == actual_side and net_return > 0:
        return "supported_winning_side"
    if predicted_side != actual_side and net_return > 0:
        return "missed_winning_side"
    return "flat_trade"


def validate_profit_training_sample(sample: dict[str, Any]) -> ProfitTrainingContract:
    sample = dict(sample)
    blockers: list[str] = []
    for field in REQUIRED_TEXT_FIELDS:
        if not _text(sample.get(field)):
            blockers.append(f"{field}_missing")
    for field in REQUIRED_NUMERIC_FIELDS:
        value = _safe_float(sample.get(field))
        if value is None:
            blockers.append(f"{field}_missing_or_invalid")

    side = _text(sample.get("side")).lower()
    if side not in {"long", "short"}:
        blockers.append("side_invalid")
    if (_safe_float(sample.get("entry_price")) or 0.0) <= 0:
        blockers.append("entry_price_not_positive")
    if (_safe_float(sample.get("close_price")) or 0.0) <= 0:
        blockers.append("close_price_not_positive")
    if (_safe_float(sample.get("quantity")) or 0.0) <= 0:
        blockers.append("quantity_not_positive")
    if (_safe_float(sample.get("notional")) or 0.0) <= 0:
        blockers.append("notional_not_positive")
    if (_safe_float(sample.get("holding_minutes")) or 0.0) < 0:
        blockers.append("holding_minutes_negative")

    for field in (
        "notional_source",
        "entry_fee_source",
        "close_fee_source",
        "pnl_source",
        "funding_fee_source",
    ):
        if _text(sample.get(field)).lower() in {"", "missing", "estimated", "fallback"}:
            blockers.append(f"{field}_not_authoritative")
    slippage_source = _text(sample.get("slippage_source"))
    if slippage_source != "okx_configured_stop_trigger_to_fills_vwap":
        blockers.append("slippage_source_not_authoritative")

    entry_fee = _safe_float(sample.get("entry_fee"))
    close_fee = _safe_float(sample.get("close_fee"))
    slippage = _safe_float(sample.get("slippage"))
    if entry_fee is not None and entry_fee < 0:
        blockers.append("entry_fee_negative")
    if close_fee is not None and close_fee < 0:
        blockers.append("close_fee_negative")
    if slippage is not None and slippage < 0:
        blockers.append("slippage_negative")

    authority = _text(sample.get("decision_authority")).lower()
    if authority not in {"rules", "model", "manual", "system"}:
        blockers.append("decision_authority_invalid")

    target_value = _safe_float(sample.get(PROFIT_TRAINING_TARGET))
    if target_value is None:
        outcome: Literal["profit", "loss", "flat", "invalid"] = "invalid"
    elif target_value > 0:
        outcome = "profit"
    elif target_value < 0:
        outcome = "loss"
    else:
        outcome = "flat"

    realized_pnl = _safe_float(sample.get("realized_pnl"))
    notional = _safe_float(sample.get("notional"))
    if (
        target_value is not None
        and realized_pnl is not None
        and notional is not None
        and notional > 0
        and not isclose(
            target_value,
            realized_pnl / notional * 100.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        blockers.append("net_return_target_algebra_mismatch")

    fingerprint = _text(sample.get("evidence_fingerprint")) or profit_sample_fingerprint(
        sample
    )
    unique_blockers = tuple(dict.fromkeys(blockers))
    return ProfitTrainingContract(
        eligible=not unique_blockers,
        reason="profit_training_sample_ready" if not unique_blockers else unique_blockers[0],
        target=PROFIT_TRAINING_TARGET,
        target_value=target_value,
        outcome=outcome if not unique_blockers else "invalid",
        decision_authority=authority,
        evidence_fingerprint=fingerprint,
        blockers=unique_blockers,
        model_shadow_alignment=_model_shadow_alignment(sample),
    )
