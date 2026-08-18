"""Canonical OKX contract quantity and notional valuation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

OKX_CONTRACT_VALUATION_VERSION = "2026-08-17.okx-contract-valuation.v1"
DEFAULT_NOTIONAL_RELATIVE_TOLERANCE = 0.05
DEFAULT_NOTIONAL_ABSOLUTE_TOLERANCE_USD = 1.0


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def _currency_pair(inst_id: str) -> tuple[str, str]:
    parts = str(inst_id or "").strip().upper().split("-")
    return (
        parts[0] if parts else "",
        parts[1] if len(parts) > 1 else "",
    )


@dataclass(frozen=True, slots=True)
class OkxContractValuation:
    inst_id: str
    contracts: float
    ct_val: float
    ct_mult: float
    ct_val_ccy: str
    settle_ccy: str
    ct_type: str
    mark_price: float
    base_quantity: float
    calculated_notional_usd: float
    reported_notional_usd: float
    notional_difference_usd: float
    notional_difference_ratio: float | None
    notional_consistent: bool | None
    valuation_formula: str
    contract_spec_complete: bool
    contract_spec_source: str
    valuation_timestamp: str | None
    version: str = OKX_CONTRACT_VALUATION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def okx_contract_valuation(
    *,
    contracts: Any,
    mark_price: Any,
    reported_notional_usd: Any,
    contract_spec: dict[str, Any] | None,
    inst_id: Any = "",
    valuation_timestamp: Any = None,
    relative_tolerance: float = DEFAULT_NOTIONAL_RELATIVE_TOLERANCE,
    absolute_tolerance_usd: float = DEFAULT_NOTIONAL_ABSOLUTE_TOLERANCE_USD,
) -> OkxContractValuation:
    """Value a linear or quote-denominated OKX derivative from one snapshot."""

    spec = _safe_dict(contract_spec)
    normalized_inst_id = str(spec.get("instId") or inst_id or "").strip().upper()
    base_ccy, quote_ccy = _currency_pair(normalized_inst_id)
    contract_count = abs(_safe_float(contracts))
    mark = max(_safe_float(mark_price), 0.0)
    reported = abs(_safe_float(reported_notional_usd))
    ct_val = max(_safe_float(spec.get("ctVal"), 0.0), 0.0)
    ct_mult = max(_safe_float(spec.get("ctMult"), 1.0), 0.0)
    raw_ct_val_ccy = str(spec.get("ctValCcy") or "").strip().upper()
    settle_ccy = str(spec.get("settleCcy") or quote_ccy or "").strip().upper()
    ct_type = str(spec.get("ctType") or "").strip().lower()
    ct_val_ccy = raw_ct_val_ccy
    currency_source_complete = bool(raw_ct_val_ccy)
    if not ct_val_ccy and base_ccy and settle_ccy and base_ccy != settle_ccy:
        ct_val_ccy = base_ccy

    quote_denominated = bool(
        ct_type == "inverse"
        or (
            ct_val_ccy
            and ct_val_ccy in {quote_ccy, settle_ccy}
            and ct_val_ccy != base_ccy
        )
    )
    unit_value = ct_val * ct_mult
    if quote_denominated:
        calculated_notional = contract_count * unit_value
        base_quantity = calculated_notional / mark if mark > 0.0 else 0.0
        formula = "contracts_x_ctVal_x_ctMult"
    else:
        base_quantity = contract_count * unit_value
        calculated_notional = base_quantity * mark
        formula = "contracts_x_ctVal_x_ctMult_x_mark"

    spec_source = str(spec.get("source") or "").strip() or "position_snapshot"
    spec_complete = bool(
        normalized_inst_id
        and contract_count > 0.0
        and ct_val > 0.0
        and ct_mult > 0.0
        and ct_val_ccy
        and settle_ccy
        and (quote_denominated or mark > 0.0)
    )
    if not currency_source_complete and spec_source == "okx_public_instruments":
        spec_complete = False

    difference = abs(calculated_notional - reported) if reported > 0.0 else 0.0
    difference_ratio = (
        difference / max(calculated_notional, reported)
        if calculated_notional > 0.0 and reported > 0.0
        else None
    )
    consistent = None
    if spec_complete and calculated_notional > 0.0 and reported > 0.0:
        allowed = max(
            max(float(absolute_tolerance_usd), 0.0),
            max(calculated_notional, reported) * max(float(relative_tolerance), 0.0),
        )
        consistent = difference <= allowed

    timestamp_text = str(valuation_timestamp or "").strip() or None
    return OkxContractValuation(
        inst_id=normalized_inst_id,
        contracts=round(contract_count, 12),
        ct_val=round(ct_val, 12),
        ct_mult=round(ct_mult, 12),
        ct_val_ccy=ct_val_ccy,
        settle_ccy=settle_ccy,
        ct_type=ct_type or ("inverse" if quote_denominated else "linear"),
        mark_price=round(mark, 12),
        base_quantity=round(base_quantity, 12),
        calculated_notional_usd=round(calculated_notional, 12),
        reported_notional_usd=round(reported, 12),
        notional_difference_usd=round(difference, 12),
        notional_difference_ratio=(
            round(difference_ratio, 12) if difference_ratio is not None else None
        ),
        notional_consistent=consistent,
        valuation_formula=formula,
        contract_spec_complete=spec_complete,
        contract_spec_source=spec_source,
        valuation_timestamp=timestamp_text,
    )
