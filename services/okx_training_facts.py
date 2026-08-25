"""Build model-training samples from authoritative OKX SWAP lifecycles."""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from services.normal_paper_trade import (
    HISTORICAL_NORMAL_PAPER_TRADE_VERSION,
    LEGACY_NORMAL_PAPER_TRADE_V3_VERSION,
    LEGACY_NORMAL_PAPER_TRADE_V4_VERSION,
    LEGACY_NORMAL_PAPER_TRADE_V5_VERSION,
    LEGACY_NORMAL_PAPER_TRADE_V6_VERSION,
    LEGACY_NORMAL_PAPER_TRADE_V7_VERSION,
    LEGACY_NORMAL_PAPER_TRADE_VERSION,
    NORMAL_PAPER_TRADE_VERSION,
    historical_normal_paper_trade_contract_reasons,
    legacy_normal_paper_v2_trade_contract_reasons,
    legacy_normal_paper_v3_trade_contract_reasons,
    legacy_normal_paper_v4_trade_contract_reasons,
    legacy_normal_paper_v5_trade_contract_reasons,
    legacy_normal_paper_v6_trade_contract_reasons,
    legacy_normal_paper_v7_trade_contract_reasons,
    normal_paper_trade_contract_reasons,
)
from services.okx_execution_slippage import (
    OKX_FILL_MARK_SLIPPAGE_SOURCE,
    OKX_FILL_MARK_SLIPPAGE_VERSION,
    OKX_ROUND_TRIP_SLIPPAGE_SOURCE,
)
from services.okx_lifecycle_order_allocations import (
    apply_lifecycle_order_allocation,
    lifecycle_order_allocation,
)
from services.okx_native_facts import OKX_ACCOUNT_BILLS_TRADE_SOURCE
from services.okx_order_fact_sync import authoritative_order_fee_fact_source
from services.paper_exploration import paper_exploration_contract_reasons
from services.paper_training import paper_training_contract_reasons
from services.production_trade_gate import validate_production_trade_gate
from services.profit_training_contract import validate_profit_training_sample

EXTREME_FUNDING_TO_NOTIONAL_RATIO = 0.20
REALIZED_NET_PNL_FORMULA = (
    "gross_pnl_usdt + official_fee_signed_usdt + funding_fee_usdt + liquidation_penalty_usdt"
)


def _value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _text(value: Any) -> str:
    return str(value or "").strip()


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_list(item))
        return list(dict.fromkeys(token for token in result if token))
    text = _text(value)
    if not text:
        return []
    tokens = {text}
    for separator in (",", ";", "|", "\n", "\t", " "):
        pieces: set[str] = set()
        for token in tokens:
            pieces.update(part.strip() for part in token.split(separator) if part.strip())
        tokens = pieces
    return sorted(tokens)


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def build_funding_bill_lifecycle_facts(
    histories: Iterable[Any],
    account_bills: Iterable[Any],
) -> dict[str, dict[str, Any]]:
    """Attribute mirrored funding bills to non-overlapping OKX lifecycles."""

    history_rows = list(histories)
    bill_rows = list(account_bills)
    matched_by_lifecycle: dict[str, list[Any]] = {}
    lifecycle_by_bill: dict[str, set[str]] = {}
    for history in history_rows:
        lifecycle_key = _text(_value(history, "row_identity"))
        opened_at = _as_utc(_value(history, "opened_at"))
        closed_at = _as_utc(_value(history, "updated_at_okx"))
        inst_id = _text(_value(history, "inst_id")).upper()
        side = _text(_value(history, "side")).lower()
        mode = _canonical_execution_mode(_value(history, "mode"))
        matches: list[Any] = []
        if lifecycle_key and opened_at and closed_at and inst_id and mode:
            for bill in bill_rows:
                bill_mode = _canonical_execution_mode(_value(bill, "mode"))
                bill_inst_id = _text(_value(bill, "inst_id")).upper()
                bill_side = _text(_value(bill, "pos_side")).lower()
                bill_at = _as_utc(_value(bill, "bill_ts"))
                funding_fee = _safe_float(_value(bill, "funding_fee"), 0.0) or 0.0
                if (
                    bill_mode != mode
                    or bill_inst_id != inst_id
                    or bill_at is None
                    or bill_at < opened_at
                    or bill_at > closed_at
                    or abs(funding_fee) <= 1e-12
                    or (bill_side in {"long", "short"} and bill_side != side)
                ):
                    continue
                matches.append(bill)
                bill_id = _text(_value(bill, "bill_id")) or f"db:{_value(bill, 'id', 0)}"
                lifecycle_by_bill.setdefault(bill_id, set()).add(lifecycle_key)
        matched_by_lifecycle[lifecycle_key] = matches

    result: dict[str, dict[str, Any]] = {}
    for lifecycle_key, matches in matched_by_lifecycle.items():
        bill_ids = [
            _text(_value(bill, "bill_id")) or f"db:{_value(bill, 'id', 0)}" for bill in matches
        ]
        shared_bill_ids = [
            bill_id for bill_id in bill_ids if len(lifecycle_by_bill.get(bill_id, set())) > 1
        ]
        result[lifecycle_key] = {
            "mirror_available": True,
            "bill_count": len(matches),
            "bill_ids": list(dict.fromkeys(bill_ids)),
            "signed_funding_fee_usdt": sum(
                _safe_float(_value(bill, "funding_fee"), 0.0) or 0.0 for bill in matches
            ),
            "shared_bill_ids": list(dict.fromkeys(shared_bill_ids)),
            "attribution_complete": not shared_bill_ids,
            "source": "okx_account_bills",
        }
    return result


def _funding_training_evidence(
    *,
    funding_fee: float,
    notional: float | None,
    official_funding_present: bool,
    bill_facts: dict[str, Any] | None,
) -> dict[str, Any]:
    ratio = funding_fee / notional if notional is not None and notional > 0 else None
    extreme = bool(ratio is not None and abs(ratio) >= EXTREME_FUNDING_TO_NOTIONAL_RATIO)
    facts = dict(bill_facts or {})
    legacy_direct_build = bill_facts is None
    bill_fee = _safe_float(facts.get("signed_funding_fee_usdt"), 0.0) or 0.0
    bill_count = max(int(_safe_float(facts.get("bill_count"), 0.0) or 0), 0)
    shared_bill_ids = list(facts.get("shared_bill_ids") or [])
    tolerance = max(1e-8, abs(funding_fee) * 1e-5)
    bill_matches = bool(bill_count > 0 and abs(bill_fee - funding_fee) <= tolerance)
    gaps: list[str] = []
    if not official_funding_present:
        status = "unavailable"
    elif shared_bill_ids:
        status = "pending_review_lifecycle_conflict"
        gaps.append("funding_bill_lifecycle_conflict")
    elif legacy_direct_build:
        # Direct unit/integration builders may not have loaded the bill mirror;
        # the production loader always supplies a mirror fact (including empty).
        status = "verified_position_history"
    elif abs(funding_fee) <= 1e-12 and bill_count <= 0:
        status = "verified_position_history"
    elif bill_matches and facts.get("attribution_complete") is True:
        status = "verified_extreme_account_bills" if extreme else "verified_account_bills"
    elif bill_count <= 0:
        status = (
            "pending_review_extreme_missing_account_bills"
            if extreme
            else "pending_review_missing_account_bills"
        )
        gaps.append(
            "extreme_funding_missing_account_bill_reconciliation"
            if extreme
            else "funding_missing_account_bill_reconciliation"
        )
    else:
        status = (
            "pending_review_extreme_account_bill_mismatch"
            if extreme
            else "pending_review_account_bill_mismatch"
        )
        gaps.append(
            "extreme_funding_account_bill_mismatch" if extreme else "funding_account_bill_mismatch"
        )
    eligible = status in {
        "verified_position_history",
        "verified_account_bills",
        "verified_extreme_account_bills",
    }
    return {
        "status": status,
        "eligible": eligible,
        "extreme": extreme,
        "extreme_threshold_ratio": EXTREME_FUNDING_TO_NOTIONAL_RATIO,
        "funding_fee_to_notional_ratio": ratio,
        "position_history_funding_fee_usdt": funding_fee,
        "account_bill_funding_fee_usdt": bill_fee,
        "account_bill_count": bill_count,
        "account_bill_ids": list(facts.get("bill_ids") or []),
        "shared_bill_ids": shared_bill_ids,
        "account_bill_difference_usdt": bill_fee - funding_fee,
        "attribution_complete": bool(eligible and not shared_bill_ids),
        "gaps": gaps,
    }


def _raw_contract_spec(raw: dict[str, Any]) -> dict[str, Any]:
    value = raw.get("_bb_contract_spec")
    return dict(value) if isinstance(value, dict) else {}


def _order_trade_ids(orders: Iterable[Any]) -> list[str]:
    return list(
        dict.fromkeys(
            trade_id
            for order in orders
            for trade_id in _list(_value(order, "okx_trade_ids"))
            if trade_id
        )
    )


def _authoritative_fill_fact(order: Any, *, order_id: str) -> dict[str, Any]:
    if order is None:
        return {}
    source = authoritative_order_fee_fact_source(order, order_id=order_id)
    raw = _dict(_value(order, "okx_raw_fills", {}))
    base_quantity = _safe_float(raw.get("base_quantity"), None)
    average_price = _safe_float(raw.get("avg_price"), None)
    contracts = _safe_float(raw.get("contracts"), None)
    fee = _safe_float(raw.get("fee_abs"), None)
    fill_pnl = _safe_float(raw.get("fill_pnl"), None)
    if fill_pnl is None:
        fill_pnl = _safe_float(_value(order, "okx_fill_pnl"), None)
    execution_slippage = _dict(raw.get("execution_slippage"))
    execution_slippage_usdt = _safe_float(
        execution_slippage.get("adverse_slippage_usdt"),
        None,
    )
    execution_slippage_contracts = _safe_float(
        execution_slippage.get("contracts"),
        None,
    )
    execution_slippage_contract_size = _safe_float(
        execution_slippage.get("contract_size"),
        None,
    )
    execution_slippage_fill_vwap = _safe_float(
        execution_slippage.get("fill_vwap"),
        None,
    )
    execution_slippage_actual_notional = _safe_float(
        execution_slippage.get("actual_notional_usdt"),
        None,
    )
    raw_contract_size = _safe_float(raw.get("contract_size"), None)
    verified_public_contract_size = (
        raw_contract_size
        if raw.get("contract_size_verified") is True
        and _text(raw.get("contract_size_source")) == "okx_public_instruments"
        else None
    )
    raw_trade_ids = set(_list(raw.get("trade_ids")))
    execution_slippage_trade_ids = set(_list(execution_slippage.get("trade_ids")))
    execution_slippage_reasons = _execution_slippage_validation_reasons(
        execution_slippage=execution_slippage,
        fill_fact_origin=_fill_fact_origin(raw),
        expected_order_id=order_id,
        expected_inst_id=_text(raw.get("inst_id")).upper(),
        expected_side=_text(_value(order, "side")).lower(),
        expected_trade_ids=raw_trade_ids,
        expected_contracts=contracts,
        expected_contract_size=raw_contract_size,
        expected_fill_vwap=average_price,
        expected_actual_notional=(
            base_quantity * average_price
            if base_quantity is not None and average_price is not None
            else None
        ),
        actual_trade_ids=execution_slippage_trade_ids,
        actual_contracts=execution_slippage_contracts,
        actual_contract_size=execution_slippage_contract_size,
        actual_fill_vwap=execution_slippage_fill_vwap,
        actual_notional=execution_slippage_actual_notional,
        adverse_slippage_usdt=execution_slippage_usdt,
    )
    execution_slippage_complete = not execution_slippage_reasons
    if (
        source is None
        or base_quantity is None
        or base_quantity <= 0
        or average_price is None
        or average_price <= 0
        or contracts is None
        or contracts <= 0
        or fee is None
        or fee < 0
    ):
        return {}
    return {
        "order_id": order_id,
        "base_quantity": base_quantity,
        "average_price": average_price,
        "contracts": contracts,
        "verified_public_contract_size": verified_public_contract_size,
        "fee": fee,
        "fill_pnl": fill_pnl,
        "fee_source": source,
        "execution_slippage_complete": execution_slippage_complete,
        "execution_slippage_usdt": (
            execution_slippage_usdt if execution_slippage_complete else None
        ),
        "execution_slippage_source": (
            OKX_FILL_MARK_SLIPPAGE_SOURCE if execution_slippage_complete else ""
        ),
        "execution_slippage_reasons": execution_slippage_reasons,
    }


def _execution_slippage_validation_reasons(
    *,
    execution_slippage: dict[str, Any],
    fill_fact_origin: str,
    expected_order_id: str,
    expected_inst_id: str,
    expected_side: str,
    expected_trade_ids: set[str],
    expected_contracts: float | None,
    expected_contract_size: float | None,
    expected_fill_vwap: float | None,
    expected_actual_notional: float | None,
    actual_trade_ids: set[str],
    actual_contracts: float | None,
    actual_contract_size: float | None,
    actual_fill_vwap: float | None,
    actual_notional: float | None,
    adverse_slippage_usdt: float | None,
) -> list[str]:
    if execution_slippage.get("complete") is not True:
        stored_reasons = _list(execution_slippage.get("reasons"))
        if stored_reasons:
            return [f"stored_slippage:{reason}" for reason in stored_reasons]
        state = "incomplete_without_reason" if execution_slippage else "fact_missing"
        return [f"stored_slippage:{state}:{fill_fact_origin}"]

    reasons: list[str] = []
    if execution_slippage.get("version") != OKX_FILL_MARK_SLIPPAGE_VERSION:
        reasons.append("slippage_version_invalid")
    if execution_slippage.get("source") != OKX_FILL_MARK_SLIPPAGE_SOURCE:
        reasons.append("slippage_source_invalid")
    if _text(execution_slippage.get("order_id")) != expected_order_id:
        reasons.append("slippage_order_id_mismatch")
    if _text(execution_slippage.get("inst_id")).upper() != expected_inst_id:
        reasons.append("slippage_instrument_id_mismatch")
    if _text(execution_slippage.get("side")).lower() != expected_side:
        reasons.append("slippage_side_mismatch")
    if not expected_trade_ids:
        reasons.append("fill_trade_ids_missing")
    elif actual_trade_ids != expected_trade_ids:
        reasons.append("slippage_trade_ids_mismatch")
    for name, actual, expected, abs_tol in (
        ("contracts", actual_contracts, expected_contracts, 1e-12),
        ("contract_size", actual_contract_size, expected_contract_size, 1e-12),
        ("fill_vwap", actual_fill_vwap, expected_fill_vwap, 1e-12),
        ("actual_notional", actual_notional, expected_actual_notional, 1e-8),
    ):
        if actual is None or expected is None:
            reasons.append(f"slippage_{name}_missing")
        elif not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=abs_tol):
            reasons.append(f"slippage_{name}_mismatch")
    if adverse_slippage_usdt is None or adverse_slippage_usdt < 0:
        reasons.append("slippage_adverse_usdt_invalid")
    return reasons


def _fill_fact_origin(raw: dict[str, Any]) -> str:
    if raw.get("fills_history_confirmed") is True:
        return "fills_history"
    if (
        raw.get("account_bills_trade_confirmed") is True
        and _text(raw.get("source")) == OKX_ACCOUNT_BILLS_TRADE_SOURCE
    ):
        return "account_bills_trade"
    if raw.get("order_detail_confirmed") is True:
        return "order_detail"
    if raw.get("execution_result_confirmed") is True:
        return "execution_result"
    return "unconfirmed"


def _authoritative_fill_group(
    order_ids: list[str],
    orders_by_exchange_id: dict[str, Any],
    *,
    raw_row: dict[str, Any] | None = None,
    lifecycle_role: str = "",
) -> dict[str, Any]:
    facts: list[dict[str, Any]] = []
    allocation_failures: dict[str, list[str]] = {}
    for order_id in order_ids:
        fact = _authoritative_fill_fact(
            orders_by_exchange_id.get(order_id),
            order_id=order_id,
        )
        if fact and raw_row is not None and lifecycle_role:
            allocation, allocation_error = lifecycle_order_allocation(
                raw_row,
                role=lifecycle_role,
                order_id=order_id,
                order_contracts=float(fact["contracts"]),
            )
            if allocation_error:
                allocation_failures[order_id] = [allocation_error]
                fact = {}
            elif allocation is not None:
                fact = apply_lifecycle_order_allocation(
                    fact,
                    allocation=allocation,
                    role=lifecycle_role,
                )
        facts.append(fact)
    execution_slippage_failures = {
        order_id: (
            list(fact.get("execution_slippage_reasons") or [])
            if fact
            else ["authoritative_fill_fact_missing"]
        )
        for order_id, fact in zip(order_ids, facts, strict=True)
        if not fact or fact.get("execution_slippage_complete") is not True
    }
    complete = bool(order_ids and all(facts) and len(facts) == len(order_ids))
    if not complete:
        return {
            "complete": False,
            "missing_order_ids": [
                order_id for order_id, fact in zip(order_ids, facts, strict=True) if not fact
            ],
            "execution_slippage_complete": False,
            "execution_slippage_failures": execution_slippage_failures,
            "lifecycle_order_allocation_failures": allocation_failures,
        }
    base_quantity = sum(float(fact["base_quantity"]) for fact in facts)
    notional = sum(float(fact["base_quantity"]) * float(fact["average_price"]) for fact in facts)
    sources = sorted({str(fact["fee_source"]) for fact in facts})
    execution_slippage_complete = all(
        fact.get("execution_slippage_complete") is True for fact in facts
    )
    fill_pnl_complete = all(_safe_float(fact.get("fill_pnl"), None) is not None for fact in facts)
    verified_contract_size_values = [
        float(value)
        for fact in facts
        if (value := _safe_float(fact.get("verified_public_contract_size"), None)) is not None
        and value > 0
    ]
    verified_contract_sizes = set(verified_contract_size_values)
    verified_public_contract_size = (
        next(iter(verified_contract_sizes))
        if len(verified_contract_sizes) == 1 and len(verified_contract_size_values) == len(facts)
        else None
    )
    return {
        "complete": True,
        "facts": facts,
        "base_quantity": base_quantity,
        "contracts": sum(float(fact["contracts"]) for fact in facts),
        "average_price": notional / base_quantity,
        "notional": notional,
        "fee": sum(float(fact["fee"]) for fact in facts),
        "verified_public_contract_size": verified_public_contract_size,
        "fill_pnl_complete": fill_pnl_complete,
        "fill_pnl": (sum(float(fact["fill_pnl"]) for fact in facts) if fill_pnl_complete else None),
        "fee_source": "+".join(sources),
        "execution_slippage_complete": execution_slippage_complete,
        "execution_slippage_usdt": (
            sum(float(fact["execution_slippage_usdt"]) for fact in facts)
            if execution_slippage_complete
            else None
        ),
        "execution_slippage_source": (
            OKX_FILL_MARK_SLIPPAGE_SOURCE if execution_slippage_complete else ""
        ),
        "execution_slippage_failures": execution_slippage_failures,
        "lifecycle_order_allocation_failures": allocation_failures,
        "missing_order_ids": [],
    }


def _reconcile_allocated_lifecycle_fees(
    *,
    entry_fill_group: dict[str, Any],
    close_fill_group: dict[str, Any],
    entry_fee: float | None,
    close_fee: float | None,
    official_fee_signed: float,
) -> tuple[float | None, float | None, dict[str, Any]]:
    """Apply OKX lifecycle fee rounding to the one proportionally allocated side."""

    if entry_fee is None or close_fee is None:
        return entry_fee, close_fee, {"applied": False, "reason": "fill_fees_incomplete"}
    official_total = abs(official_fee_signed)
    if official_total <= 0:
        return entry_fee, close_fee, {"applied": False, "reason": "official_fee_unavailable"}
    allocated_roles = [
        role
        for role, group in (("entry", entry_fill_group), ("close", close_fill_group))
        if any(
            isinstance(fact.get("lifecycle_order_allocation"), dict)
            for fact in group.get("facts", [])
            if isinstance(fact, dict)
        )
    ]
    if len(allocated_roles) != 1:
        return entry_fee, close_fee, {
            "applied": False,
            "reason": "single_allocated_fee_side_required",
            "allocated_roles": allocated_roles,
        }
    calculated_total = entry_fee + close_fee
    delta = official_total - calculated_total
    if math.isclose(calculated_total, official_total, rel_tol=1e-6, abs_tol=1e-8):
        return entry_fee, close_fee, {
            "applied": False,
            "reason": "allocated_fee_already_consistent",
            "allocated_role": allocated_roles[0],
            "difference_usdt": delta,
        }
    tolerance = max(1e-8, official_total * 0.001)
    allocated_role = allocated_roles[0]
    adjusted_entry = entry_fee + delta if allocated_role == "entry" else entry_fee
    adjusted_close = close_fee + delta if allocated_role == "close" else close_fee
    if abs(delta) > tolerance or adjusted_entry < 0 or adjusted_close < 0:
        return entry_fee, close_fee, {
            "applied": False,
            "reason": "allocated_fee_difference_exceeds_rounding_tolerance",
            "allocated_role": allocated_role,
            "difference_usdt": delta,
            "tolerance_usdt": tolerance,
        }
    return adjusted_entry, adjusted_close, {
        "applied": True,
        "source_authority": "okx_position_history_total_fee",
        "allocated_role": allocated_role,
        "calculated_total_before_usdt": calculated_total,
        "official_total_usdt": official_total,
        "difference_usdt": delta,
        "tolerance_usdt": tolerance,
    }


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _protection_execution(order: Any) -> dict[str, Any]:
    raw = _dict(_value(order, "okx_raw_fills", {}))
    execution = _dict(raw.get("protection_execution"))
    if (
        execution.get("lifecycle_complete") is True
        and _text(execution.get("source_authority")) == "okx_algo_history_plus_fills_history"
        and _text(execution.get("actual_side")).lower() in {"sl", "tp"}
    ):
        return execution
    return {}


def _protection_submission(order: Any) -> dict[str, Any]:
    raw = _dict(_value(order, "okx_raw_fills", {}))
    submission = _dict(raw.get("protection_submission"))
    if (
        submission.get("exchange_confirmation_recorded") is True
        and _text(submission.get("source_authority"))
        == "local_submit_plus_okx_create_order_response"
    ):
        return submission
    return {}


def _first_protection_execution(orders: Iterable[Any]) -> dict[str, Any]:
    return next(
        (execution for order in orders if (execution := _protection_execution(order))),
        {},
    )


def _first_protection_submission(orders: Iterable[Any]) -> dict[str, Any]:
    return next(
        (submission for order in orders if (submission := _protection_submission(order))),
        {},
    )


def _iso_from_ms(value: Any) -> str | None:
    timestamp_ms = _safe_float(value, None)
    if timestamp_ms is None or timestamp_ms <= 0:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC).isoformat()


def _execution_budget_facts(
    *,
    raw_llm_response: dict[str, Any],
    realized_pnl: float,
) -> dict[str, float | None]:
    sizing = _dict(raw_llm_response.get("profit_risk_sizing"))
    risk_budget = _safe_float(sizing.get("risk_budget_usdt"), None)
    planned_loss = _safe_float(sizing.get("planned_stressed_loss_usdt"), None)
    actual_loss = max(-realized_pnl, 0.0)
    return {
        "risk_budget_usdt": risk_budget if risk_budget is not None and risk_budget > 0 else None,
        "planned_stressed_loss_usdt": (
            planned_loss if planned_loss is not None and planned_loss >= 0 else None
        ),
        "actual_loss_usdt": actual_loss,
        "actual_over_budget_loss_usdt": (
            max(actual_loss - risk_budget, 0.0)
            if risk_budget is not None and risk_budget > 0
            else None
        ),
    }


def _has_raw_key(raw: dict[str, Any], *keys: str) -> bool:
    return any(key in raw and raw.get(key) not in (None, "") for key in keys)


def _canonical_execution_mode(value: Any) -> str:
    mode = _text(value).lower()
    if mode in {"paper", "demo", "sim", "simulation"}:
        return "paper"
    if mode in {"live", "real", "production"}:
        return "live"
    return ""


def _decision_authority(
    *,
    raw_llm_response: dict[str, Any],
    execution_mode: str,
    valid_normal_paper: bool,
    valid_legacy_paper: bool,
    strategy_training_role: str,
) -> str:
    normal_paper = _dict(raw_llm_response.get("normal_paper_trade"))
    if execution_mode == "paper" and valid_normal_paper:
        authority = _text(normal_paper.get("decision_authority")).lower()
        if authority:
            return authority
        if (
            normal_paper.get("version") == HISTORICAL_NORMAL_PAPER_TRADE_VERSION
            and normal_paper.get("order_creation_owner") == "ensemble_trader_unified_decision"
        ):
            return "ensemble"
    gate_validation = validate_production_trade_gate(raw_llm_response.get("production_trade_gate"))
    if gate_validation.valid:
        return _text(gate_validation.gate.get("decision_authority")).lower()
    if execution_mode == "paper" and valid_legacy_paper:
        return "system"
    if strategy_training_role != "entry_strategy":
        return "system"
    return ""


def _model_shadow_prediction(
    raw_llm_response: dict[str, Any],
    *,
    decision_authority: str,
) -> dict[str, Any]:
    if decision_authority != "rules":
        return {}
    gate_validation = validate_production_trade_gate(
        raw_llm_response.get("production_trade_gate"),
        required_mode="live_rules_canary",
    )
    signal = _dict(raw_llm_response.get("live_rules_canary_signal"))
    shadow = _dict(raw_llm_response.get("model_shadow_decision"))
    if (
        not gate_validation.valid
        or signal.get("production_eligible") is not True
        or signal.get("decision_authority") != "rules"
        or signal.get("model_can_influence") is not False
        or signal.get("action") not in {"long", "short"}
        or shadow.get("observation_only") is not True
        or shadow.get("can_authorize_entry") is not False
        or shadow.get("can_change_size_or_leverage") is not False
    ):
        return {}
    action = _text(shadow.get("action")).lower()
    if action in {"buy", "open_long"}:
        action = "long"
    elif action in {"sell", "open_short"}:
        action = "short"
    if action not in {"long", "short"}:
        return {}
    return {
        "action": action,
        "confidence": _safe_float(shadow.get("confidence"), None),
        "source": "live_rules_canary_model_shadow_decision",
        "observation_only": True,
        "can_authorize_entry": False,
        "rules_execution_action": signal.get("action"),
        "signal_version": signal.get("version"),
    }


def _directional_price_return_pct(
    *,
    side: str,
    entry_price: float,
    close_price: float,
) -> float | None:
    if entry_price <= 0 or close_price <= 0 or side not in {"long", "short"}:
        return None
    raw_return = (close_price - entry_price) / entry_price * 100.0
    return raw_return if side == "long" else -raw_return


def _return_consistency_facts(
    *,
    side: str,
    entry_price: float,
    close_price: float,
    gross_pnl: float,
    notional: float | None,
) -> dict[str, Any]:
    price_return_pct = _directional_price_return_pct(
        side=side,
        entry_price=entry_price,
        close_price=close_price,
    )
    gross_return_pct = (
        gross_pnl / notional * 100.0 if notional is not None and notional > 0 else None
    )
    return_consistent = bool(
        price_return_pct is not None
        and gross_return_pct is not None
        and math.isclose(
            gross_return_pct,
            price_return_pct,
            rel_tol=0.01,
            abs_tol=0.05,
        )
    )
    return {
        "gross_price_return_pct": price_return_pct,
        "gross_return_on_notional_pct": gross_return_pct,
        "gross_return_price_consistent": return_consistent,
    }


def _historical_contract_notional_reconciliation(
    *,
    side: str,
    entry_price: float,
    close_price: float,
    gross_pnl: float,
    realized_pnl: float,
    pnl_ratio: float | None,
    leverage: float,
    current_notional: float | None,
    contracts: float,
    contract_ct_mult: float | None,
    close_fill_group: dict[str, Any],
) -> dict[str, Any]:
    """Recover the historical face value when today's instrument spec changed.

    OKX position-history gross PnL and the matching fills are both authoritative.
    Their price path determines the historical entry notional without borrowing
    today's public ``ctVal`` for an older lifecycle.
    """

    price_return_pct = _directional_price_return_pct(
        side=side,
        entry_price=entry_price,
        close_price=close_price,
    )
    close_fill_pnl = _safe_float(close_fill_group.get("fill_pnl"), None)
    price_path_notional: float | None = None
    if (
        price_return_pct is not None
        and abs(price_return_pct) > 1e-12
        and abs(gross_pnl) > 1e-12
        and gross_pnl * price_return_pct > 0
        and close_fill_group.get("fill_pnl_complete") is True
        and close_fill_pnl is not None
        and math.isclose(
            close_fill_pnl,
            gross_pnl,
            rel_tol=1e-6,
            abs_tol=max(1e-8, abs(gross_pnl) * 1e-8),
        )
    ):
        candidate = gross_pnl / (price_return_pct / 100.0)
        if candidate > 0 and math.isfinite(candidate):
            price_path_notional = candidate

    margin_ratio_notional: float | None = None
    if (
        pnl_ratio is not None
        and abs(pnl_ratio) > 1e-12
        and abs(realized_pnl) > 1e-12
        and realized_pnl * pnl_ratio > 0
        and leverage > 0
        and math.isfinite(leverage)
    ):
        candidate = realized_pnl / pnl_ratio * leverage
        if candidate > 0 and math.isfinite(candidate):
            margin_ratio_notional = candidate

    if (
        price_path_notional is not None
        and margin_ratio_notional is not None
        and not math.isclose(
            price_path_notional,
            margin_ratio_notional,
            rel_tol=0.02,
            abs_tol=max(1e-8, price_path_notional * 1e-6),
        )
    ):
        return {
            "applied": False,
            "reason": "authoritative_historical_notional_sources_conflict",
            "authority_conflict": True,
            "price_path_notional": price_path_notional,
            "margin_ratio_notional": margin_ratio_notional,
            "position_history_pnl_ratio": pnl_ratio,
            "position_history_leverage": leverage,
        }

    derived_notional = price_path_notional or margin_ratio_notional
    if derived_notional is None:
        return {
            "applied": False,
            "reason": "authoritative_historical_notional_derivation_unavailable",
        }
    source_authority = (
        "okx_fills_history_pnl_and_position_history_price_path"
        if price_path_notional is not None
        else "okx_position_history_realized_pnl_pnl_ratio_and_leverage"
    )
    if current_notional is None or current_notional <= 0:
        return {"applied": False, "reason": "current_fill_notional_unavailable"}
    if math.isclose(
        current_notional,
        derived_notional,
        rel_tol=0.01,
        abs_tol=max(1e-8, derived_notional * 1e-6),
    ):
        return {"applied": False, "reason": "current_contract_spec_already_consistent"}
    if contracts <= 0:
        return {"applied": False, "reason": "historical_contract_count_unavailable"}
    if entry_price <= 0:
        return {"applied": False, "reason": "historical_entry_price_unavailable"}
    ct_mult = contract_ct_mult if contract_ct_mult is not None and contract_ct_mult > 0 else 1.0
    effective_base_quantity = derived_notional / entry_price
    effective_ct_val = effective_base_quantity / contracts / ct_mult
    if effective_ct_val <= 0 or not math.isfinite(effective_ct_val):
        return {"applied": False, "reason": "derived_historical_contract_value_invalid"}
    return {
        "applied": True,
        "reason": "current_public_contract_value_mismatches_authoritative_history",
        "source_authority": source_authority,
        "current_notional": current_notional,
        "historical_notional": derived_notional,
        "notional_scale": derived_notional / current_notional,
        "historical_base_quantity": effective_base_quantity,
        "historical_ct_val": effective_ct_val,
        "close_fill_pnl": close_fill_pnl,
        "position_history_gross_pnl": gross_pnl,
        "gross_price_return_pct": price_return_pct,
        "price_path_notional": price_path_notional,
        "margin_ratio_notional": margin_ratio_notional,
        "position_history_pnl_ratio": pnl_ratio,
        "position_history_leverage": leverage,
    }


def build_okx_history_training_sample(
    history: Any,
    *,
    positions_by_id: dict[int, Any] | None = None,
    orders_by_exchange_id: dict[str, Any] | None = None,
    decision_raw_by_position_id: dict[int, dict[str, Any]] | None = None,
    decision_raw_by_order_id: dict[str, dict[str, Any]] | None = None,
    decision_feature_by_order_id: dict[str, dict[str, Any]] | None = None,
    decision_execution_by_order_id: dict[str, dict[str, Any]] | None = None,
    funding_bill_lifecycle_facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one mirrored OKX positions-history lifecycle into one sample."""

    positions_by_id = positions_by_id or {}
    orders_by_exchange_id = orders_by_exchange_id or {}
    decision_raw_by_position_id = decision_raw_by_position_id or {}
    decision_raw_by_order_id = decision_raw_by_order_id or {}
    decision_feature_by_order_id = decision_feature_by_order_id or {}
    decision_execution_by_order_id = decision_execution_by_order_id or {}
    raw = dict(_value(history, "raw_row", {}) or {})
    position_ids = [
        int(value) for value in _list(_value(history, "position_ids")) if value.isdigit()
    ]
    entry_order_ids = _list(_value(history, "entry_order_ids"))
    close_order_ids = _list(_value(history, "close_order_ids"))
    linked_order_ids = list(dict.fromkeys([*entry_order_ids, *close_order_ids]))
    entry_orders = [
        orders_by_exchange_id[value] for value in entry_order_ids if value in orders_by_exchange_id
    ]
    close_orders = [
        orders_by_exchange_id[value] for value in close_order_ids if value in orders_by_exchange_id
    ]
    linked_orders = [
        orders_by_exchange_id[value] for value in linked_order_ids if value in orders_by_exchange_id
    ]
    local_positions = [positions_by_id[value] for value in position_ids if value in positions_by_id]
    local_position = local_positions[0] if local_positions else None

    entry_fill_group = _authoritative_fill_group(
        entry_order_ids,
        orders_by_exchange_id,
        raw_row=raw,
        lifecycle_role="entry",
    )
    close_fill_group = _authoritative_fill_group(
        close_order_ids,
        orders_by_exchange_id,
        raw_row=raw,
        lifecycle_role="close",
    )
    canonical_entry_order_id = entry_order_ids[0] if entry_order_ids else ""
    canonical_close_order_id = close_order_ids[-1] if close_order_ids else ""
    canonical_notional = (
        _safe_float(entry_fill_group.get("notional"), None)
        if entry_fill_group.get("complete") is True
        else None
    )
    canonical_entry_fee = (
        _safe_float(entry_fill_group.get("fee"), None)
        if entry_fill_group.get("complete") is True
        else None
    )
    canonical_close_fee = (
        _safe_float(close_fill_group.get("fee"), None)
        if close_fill_group.get("complete") is True
        else None
    )

    opened_at = _as_utc(_value(history, "opened_at"))
    closed_at = _as_utc(_value(history, "updated_at_okx"))
    holding_minutes = (
        max((closed_at - opened_at).total_seconds() / 60.0, 0.0)
        if opened_at and closed_at
        else None
    )
    entry_price = _safe_float(_value(history, "open_avg_px"), 0.0) or 0.0
    close_price = _safe_float(_value(history, "close_avg_px"), 0.0) or 0.0
    side = _text(_value(history, "side")).lower()
    fill_contracts = (
        _safe_float(entry_fill_group.get("contracts"), None)
        if entry_fill_group.get("complete") is True
        else None
    )
    contracts = fill_contracts or 0.0
    open_max_contracts = (
        _safe_float(raw.get("openMaxPos"), None)
        or _safe_float(_value(history, "open_max_pos"), 0.0)
        or 0.0
    )
    close_total_contracts = (
        _safe_float(raw.get("closeTotalPos"), None)
        or _safe_float(_value(history, "close_total_pos"), 0.0)
        or 0.0
    )
    entry_target_contracts = (
        close_total_contracts
        if _text(_value(history, "close_status")).lower() == "full" and close_total_contracts > 0
        else open_max_contracts
    )
    spec = _raw_contract_spec(raw)
    public_or_stored_ct_val = _safe_float(spec.get("ctVal"), None)
    ct_mult = _safe_float(spec.get("ctMult"), None)
    lot_size = _safe_float(spec.get("lotSz"), None)

    realized_pnl = _safe_float(_value(history, "realized_pnl"), 0.0) or 0.0
    gross_pnl = _safe_float(_value(history, "pnl"), 0.0) or 0.0
    fee_signed = _safe_float(_value(history, "fee"), 0.0) or 0.0
    funding_fee = _safe_float(_value(history, "funding_fee"), 0.0) or 0.0
    liquidation_penalty = (
        _safe_float(raw.get("liqPenalty") or raw.get("liquidationPenalty"), 0.0) or 0.0
    )
    source_execution_mode = _text(_value(history, "mode")).lower()
    execution_mode = _canonical_execution_mode(source_execution_mode)
    history_contract_source = _text(raw.get("_bb_contract_spec_source"))
    ct_val = public_or_stored_ct_val
    contract_ct_val_source = history_contract_source
    entry_verified_contract_size = _safe_float(
        entry_fill_group.get("verified_public_contract_size"),
        None,
    )
    close_verified_contract_size = _safe_float(
        close_fill_group.get("verified_public_contract_size"),
        None,
    )
    verified_fill_contract_sizes = {
        value
        for value in (entry_verified_contract_size, close_verified_contract_size)
        if value is not None and value > 0
    }
    if len(verified_fill_contract_sizes) == 1 and ct_mult is not None and ct_mult > 0:
        verified_fill_ct_val = next(iter(verified_fill_contract_sizes)) / ct_mult
        if ct_val is None or not math.isclose(
            ct_val,
            verified_fill_ct_val,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            ct_val = verified_fill_ct_val
            contract_ct_val_source = "okx_public_instruments_verified_order_fills"
    historical_contract_reconciliation = _historical_contract_notional_reconciliation(
        side=side,
        entry_price=entry_price,
        close_price=close_price,
        gross_pnl=gross_pnl,
        realized_pnl=realized_pnl,
        pnl_ratio=_safe_float(_value(history, "pnl_ratio"), None),
        leverage=_safe_float(_value(history, "leverage"), 1.0) or 1.0,
        current_notional=canonical_notional,
        contracts=contracts,
        contract_ct_mult=ct_mult,
        close_fill_group=close_fill_group,
    )
    if historical_contract_reconciliation.get("applied") is True:
        canonical_notional = _safe_float(
            historical_contract_reconciliation.get("historical_notional"),
            canonical_notional,
        )
        ct_val = _safe_float(
            historical_contract_reconciliation.get("historical_ct_val"),
            ct_val,
        )
        contract_ct_val_source = str(
            historical_contract_reconciliation.get("source_authority") or ""
        )
    canonical_entry_fee, canonical_close_fee, lifecycle_fee_reconciliation = (
        _reconcile_allocated_lifecycle_fees(
            entry_fill_group=entry_fill_group,
            close_fill_group=close_fill_group,
            entry_fee=canonical_entry_fee,
            close_fee=canonical_close_fee,
            official_fee_signed=fee_signed,
        )
    )
    return_facts = _return_consistency_facts(
        side=side,
        entry_price=entry_price,
        close_price=close_price,
        gross_pnl=gross_pnl,
        notional=canonical_notional,
    )
    settlement_expected = gross_pnl + fee_signed + funding_fee + liquidation_penalty
    settlement_tolerance = max(1e-6, abs(realized_pnl) * 1e-5)
    funding_training_evidence = _funding_training_evidence(
        funding_fee=funding_fee,
        notional=canonical_notional,
        official_funding_present=_has_raw_key(raw, "fundingFee", "funding_fee"),
        bill_facts=funding_bill_lifecycle_facts,
    )
    gaps: list[str] = []
    if historical_contract_reconciliation.get("authority_conflict") is True:
        gaps.append("historical_contract_notional_authorities_conflict")
    if not execution_mode:
        gaps.append("missing_or_invalid_execution_mode")
    if _text(_value(history, "sync_status")).lower() != "synced":
        gaps.append("history_sync_not_confirmed")
    if _text(_value(history, "close_status")).lower() != "full":
        gaps.append("lifecycle_not_fully_closed")
    if not _text(_value(history, "pos_id")):
        gaps.append("missing_okx_pos_id")
    if _text(_value(history, "side")).lower() not in {"long", "short"}:
        gaps.append("missing_position_side")
    if entry_price <= 0:
        gaps.append("missing_open_average_price")
    if close_price <= 0:
        gaps.append("missing_close_average_price")
    if not _has_raw_key(raw, "realizedPnl", "realized_pnl"):
        gaps.append("missing_official_realized_pnl")
    if not _has_raw_key(raw, "fee", "totalFee", "total_fee"):
        gaps.append("missing_official_fee")
    if not _has_raw_key(raw, "fundingFee", "funding_fee"):
        gaps.append("missing_official_funding_fee")
    gaps.extend(funding_training_evidence["gaps"])
    if ct_val is None or ct_val <= 0:
        gaps.append("missing_contract_ct_val")
    if ct_mult is None or ct_mult <= 0:
        gaps.append("missing_contract_ct_mult")
    if lot_size is None or lot_size <= 0:
        gaps.append("missing_contract_lot_size")
    if contracts <= 0:
        gaps.append("missing_fill_or_open_contracts")
    if history_contract_source != "okx_public_instruments":
        gaps.append("contract_spec_source_not_okx_public_instruments")
    if return_facts.get("gross_return_price_consistent") is not True:
        gaps.append("gross_return_price_path_mismatch")
    if abs(realized_pnl - settlement_expected) > settlement_tolerance:
        gaps.append("settlement_algebra_mismatch")
    if entry_fill_group.get("complete") is not True:
        gaps.append("missing_authoritative_entry_fill_facts")
    if close_fill_group.get("complete") is not True:
        gaps.append("missing_authoritative_close_fill_facts")
    if (
        entry_fill_group.get("complete") is True
        and entry_target_contracts > 0
        and not math.isclose(
            _safe_float(entry_fill_group.get("contracts"), 0.0) or 0.0,
            entry_target_contracts,
            rel_tol=0.02,
            abs_tol=1e-12,
        )
    ):
        gaps.append("entry_fill_contracts_history_mismatch")
    if (
        close_fill_group.get("complete") is True
        and close_total_contracts > 0
        and not math.isclose(
            _safe_float(close_fill_group.get("contracts"), 0.0) or 0.0,
            close_total_contracts,
            rel_tol=0.02,
            abs_tol=1e-12,
        )
    ):
        gaps.append("close_fill_contracts_history_mismatch")
    if (
        entry_fill_group.get("complete") is True
        and ct_val is not None
        and ct_val > 0
        and ct_mult is not None
        and ct_mult > 0
        and historical_contract_reconciliation.get("applied") is not True
        and not math.isclose(
            _safe_float(entry_fill_group.get("base_quantity"), 0.0) or 0.0,
            (_safe_float(entry_fill_group.get("contracts"), 0.0) or 0.0) * ct_val * ct_mult,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        gaps.append("entry_fill_contract_quantity_mismatch")
    if (
        close_fill_group.get("complete") is True
        and ct_val is not None
        and ct_val > 0
        and ct_mult is not None
        and ct_mult > 0
        and historical_contract_reconciliation.get("applied") is not True
        and not math.isclose(
            _safe_float(close_fill_group.get("base_quantity"), 0.0) or 0.0,
            (_safe_float(close_fill_group.get("contracts"), 0.0) or 0.0) * ct_val * ct_mult,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        gaps.append("close_fill_contract_quantity_mismatch")
    if canonical_notional is None or canonical_notional <= 0:
        gaps.append("missing_authoritative_entry_notional")
    if canonical_entry_fee is None:
        gaps.append("missing_authoritative_entry_fee")
    if canonical_close_fee is None:
        gaps.append("missing_authoritative_close_fee")
    entry_fill_price = _safe_float(entry_fill_group.get("average_price"), None)
    close_fill_price = _safe_float(close_fill_group.get("average_price"), None)
    if (
        entry_fill_price is not None
        and entry_price > 0
        and not math.isclose(entry_fill_price, entry_price, rel_tol=0.001, abs_tol=1e-12)
    ):
        gaps.append("entry_fill_price_history_mismatch")
    if (
        close_fill_price is not None
        and close_price > 0
        and not math.isclose(close_fill_price, close_price, rel_tol=0.001, abs_tol=1e-12)
    ):
        gaps.append("close_fill_price_history_mismatch")
    if holding_minutes is None:
        gaps.append("missing_authoritative_holding_minutes")
    if canonical_entry_fee is not None and canonical_close_fee is not None:
        order_fee_total = canonical_entry_fee + canonical_close_fee
        if not math.isclose(
            order_fee_total,
            abs(fee_signed),
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            gaps.append("order_fee_total_mismatch")
    gaps = list(dict.fromkeys(gaps))

    position_id = position_ids[0] if position_ids else 0
    entry_decision_ids = sorted(
        {
            int(_value(orders_by_exchange_id.get(order_id), "decision_id", 0) or 0)
            for order_id in entry_order_ids
            if int(_value(orders_by_exchange_id.get(order_id), "decision_id", 0) or 0) > 0
        }
    )
    raw_llm_response: dict[str, Any] = {}
    entry_feature_snapshot: dict[str, Any] = {}
    decision_id = entry_decision_ids[0] if len(entry_decision_ids) == 1 else 0
    decision_lineage_source = (
        "exact_entry_order_decision_id"
        if len(entry_decision_ids) == 1
        else "multiple_entry_decisions"
        if len(entry_decision_ids) > 1
        else "missing"
    )
    if len(entry_decision_ids) == 1:
        for order_id in entry_order_ids:
            order = orders_by_exchange_id.get(order_id)
            if int(_value(order, "decision_id", 0) or 0) != decision_id:
                continue
            candidate = decision_raw_by_order_id.get(order_id)
            if isinstance(candidate, dict) and candidate:
                raw_llm_response = candidate
                decision_lineage_source = "exact_entry_order_decision_payload"
            feature_candidate = decision_feature_by_order_id.get(order_id)
            if isinstance(feature_candidate, dict) and feature_candidate:
                entry_feature_snapshot = dict(feature_candidate)
            if raw_llm_response and entry_feature_snapshot:
                break
    if not raw_llm_response:
        raw_llm_response = next(
            (
                decision_raw_by_position_id[value]
                for value in position_ids
                if value in decision_raw_by_position_id
            ),
            {},
        )
        if raw_llm_response:
            decision_lineage_source = "position_time_fallback_payload"

    stop_loss_price = _safe_float(_value(local_position, "stop_loss_price"), None)
    take_profit_price = _safe_float(_value(local_position, "take_profit_price"), None)
    exact_execution = next(
        (
            decision_execution_by_order_id[order_id]
            for order_id in entry_order_ids
            if order_id in decision_execution_by_order_id
        ),
        {},
    )
    if entry_price > 0 and side in {"long", "short"}:
        stop_loss_pct = _safe_float(exact_execution.get("stop_loss_pct"), None)
        take_profit_pct = _safe_float(exact_execution.get("take_profit_pct"), None)
        if (
            (stop_loss_price is None or stop_loss_price <= 0)
            and stop_loss_pct
            and stop_loss_pct > 0
        ):
            stop_loss_price = (
                entry_price * (1 - stop_loss_pct)
                if side == "long"
                else entry_price * (1 + stop_loss_pct)
            )
        if (
            (take_profit_price is None or take_profit_price <= 0)
            and take_profit_pct
            and take_profit_pct > 0
        ):
            take_profit_price = (
                entry_price * (1 + take_profit_pct)
                if side == "long"
                else entry_price * (1 - take_profit_pct)
            )
    protection_execution = _first_protection_execution(close_orders)
    protection_submission = _first_protection_submission(entry_orders)
    stop_loss_fill_confirmed = bool(
        protection_execution and _text(protection_execution.get("actual_side")).lower() == "sl"
    )
    entry_execution_slippage_usdt = _safe_float(
        entry_fill_group.get("execution_slippage_usdt"),
        None,
    )
    close_execution_slippage_usdt = _safe_float(
        close_fill_group.get("execution_slippage_usdt"),
        None,
    )
    slippage_notional_scale = (
        _safe_float(
            historical_contract_reconciliation.get("notional_scale"),
            1.0,
        )
        or 1.0
    )
    if historical_contract_reconciliation.get("applied") is True:
        if entry_execution_slippage_usdt is not None:
            entry_execution_slippage_usdt *= slippage_notional_scale
        if close_execution_slippage_usdt is not None:
            close_execution_slippage_usdt *= slippage_notional_scale
    execution_slippage_usdt = (
        entry_execution_slippage_usdt + close_execution_slippage_usdt
        if entry_fill_group.get("execution_slippage_complete") is True
        and close_fill_group.get("execution_slippage_complete") is True
        and entry_execution_slippage_usdt is not None
        and close_execution_slippage_usdt is not None
        else None
    )
    canonical_slippage = (
        execution_slippage_usdt / canonical_notional * 100.0
        if execution_slippage_usdt is not None
        and canonical_notional is not None
        and canonical_notional > 0
        else None
    )
    canonical_slippage_source = (
        OKX_ROUND_TRIP_SLIPPAGE_SOURCE if canonical_slippage is not None else ""
    )
    if canonical_slippage is None:
        gaps.append("missing_authoritative_slippage")
    gaps = list(dict.fromkeys(gaps))
    protection_execution_gaps: list[str] = []
    if protection_execution:
        if not protection_submission:
            protection_execution_gaps.append("missing_client_protection_submission_confirmation")
        if protection_execution.get("actual_trigger_market_price_available") is not True:
            protection_execution_gaps.append("actual_trigger_market_price_unavailable")
        if protection_execution.get("trigger_path_extrema_available") is not True:
            protection_execution_gaps.append("trigger_path_extrema_unavailable")
        if protection_execution.get("trigger_orderbook_snapshot_available") is not True:
            protection_execution_gaps.append("trigger_orderbook_snapshot_unavailable")
    budget_facts = _execution_budget_facts(
        raw_llm_response=raw_llm_response,
        realized_pnl=realized_pnl,
    )
    lineage_gaps: list[str] = []
    if not entry_order_ids:
        lineage_gaps.append("missing_position_history_entry_orders")
    elif not entry_orders:
        lineage_gaps.append("missing_loaded_entry_order_facts")
    if not close_order_ids:
        lineage_gaps.append("missing_position_history_close_orders")
    elif not any(order_id in orders_by_exchange_id for order_id in close_order_ids):
        lineage_gaps.append("missing_loaded_close_order_facts")
    if decision_id <= 0:
        lineage_gaps.append("missing_exact_entry_order_decision_link")
    if len(entry_decision_ids) > 1:
        lineage_gaps.append("multiple_entry_decision_lineage")
    if decision_lineage_source != "exact_entry_order_decision_payload":
        lineage_gaps.append("missing_exact_entry_order_decision_payload")
    if local_position is None:
        lineage_gaps.append("missing_local_position_strategy_lineage")
    if stop_loss_price is None or stop_loss_price <= 0:
        lineage_gaps.append("missing_planned_stop_loss_lineage")
    if take_profit_price is None or take_profit_price <= 0:
        lineage_gaps.append("missing_planned_take_profit_lineage")
    paper_canary = _dict(raw_llm_response.get("paper_bootstrap_canary"))
    obsolete_sampling_entry = bool(
        paper_canary
        and (
            _text(paper_canary.get("trade_kind")) != "normal_strategy_trade"
            or paper_canary.get("continuous_training_after_settlement") is not True
        )
    )
    if obsolete_sampling_entry:
        lineage_gaps.append("obsolete_sampling_entry_not_strategy_trainable")
    normal_paper = _dict(raw_llm_response.get("normal_paper_trade"))
    normal_paper_version = _text(normal_paper.get("version"))
    current_normal_paper = bool(normal_paper and normal_paper_version == NORMAL_PAPER_TRADE_VERSION)
    legacy_v7_normal_paper = bool(
        normal_paper and normal_paper_version == LEGACY_NORMAL_PAPER_TRADE_V7_VERSION
    )
    legacy_v6_normal_paper = bool(
        normal_paper and normal_paper_version == LEGACY_NORMAL_PAPER_TRADE_V6_VERSION
    )
    legacy_v5_normal_paper = bool(
        normal_paper and normal_paper_version == LEGACY_NORMAL_PAPER_TRADE_V5_VERSION
    )
    legacy_v4_normal_paper = bool(
        normal_paper and normal_paper_version == LEGACY_NORMAL_PAPER_TRADE_V4_VERSION
    )
    legacy_v3_normal_paper = bool(
        normal_paper and normal_paper_version == LEGACY_NORMAL_PAPER_TRADE_V3_VERSION
    )
    legacy_v2_normal_paper = bool(
        normal_paper and normal_paper_version == LEGACY_NORMAL_PAPER_TRADE_VERSION
    )
    historical_normal_paper = bool(
        normal_paper and normal_paper_version == HISTORICAL_NORMAL_PAPER_TRADE_VERSION
    )
    normal_paper_gaps = []
    if current_normal_paper:
        normal_paper_gaps = normal_paper_trade_contract_reasons(normal_paper)
    elif legacy_v7_normal_paper:
        normal_paper_gaps = legacy_normal_paper_v7_trade_contract_reasons(normal_paper)
    elif legacy_v6_normal_paper:
        normal_paper_gaps = legacy_normal_paper_v6_trade_contract_reasons(normal_paper)
    elif legacy_v5_normal_paper:
        normal_paper_gaps = legacy_normal_paper_v5_trade_contract_reasons(normal_paper)
    elif legacy_v4_normal_paper:
        normal_paper_gaps = legacy_normal_paper_v4_trade_contract_reasons(normal_paper)
    elif legacy_v3_normal_paper:
        normal_paper_gaps = legacy_normal_paper_v3_trade_contract_reasons(normal_paper)
    elif legacy_v2_normal_paper:
        normal_paper_gaps = legacy_normal_paper_v2_trade_contract_reasons(normal_paper)
    elif historical_normal_paper:
        normal_paper_gaps = historical_normal_paper_trade_contract_reasons(normal_paper)
    elif normal_paper:
        normal_paper_gaps = normal_paper_trade_contract_reasons(normal_paper)
    if normal_paper and execution_mode != "paper":
        normal_paper_gaps.append("normal_paper_trade_non_paper_execution_mode")
    paper_exploration = _dict(raw_llm_response.get("paper_exploration"))
    paper_exploration_gaps = (
        paper_exploration_contract_reasons(paper_exploration) if paper_exploration else []
    )
    paper_training = _dict(raw_llm_response.get("paper_training"))
    paper_training_gaps = paper_training_contract_reasons(paper_training) if paper_training else []
    if paper_training and execution_mode != "paper":
        paper_training_gaps.append("paper_training_non_paper_execution_mode")
    if paper_training and (paper_exploration or paper_canary):
        paper_training_gaps.append("paper_training_conflicting_entry_contract")
    if (
        current_normal_paper
        or legacy_v7_normal_paper
        or legacy_v6_normal_paper
        or legacy_v5_normal_paper
        or legacy_v4_normal_paper
        or legacy_v3_normal_paper
        or legacy_v2_normal_paper
    ) and (paper_exploration or paper_training or paper_canary):
        normal_paper_gaps.append("normal_paper_trade_conflicting_legacy_contract")
    normal_paper_gaps = list(dict.fromkeys(normal_paper_gaps))
    paper_training_gaps = list(dict.fromkeys(paper_training_gaps))
    valid_paper_exploration = bool(paper_exploration and not paper_exploration_gaps)
    valid_paper_training = bool(paper_training and not paper_training_gaps)
    valid_normal_paper = bool(normal_paper and not normal_paper_gaps)
    if normal_paper_gaps:
        lineage_gaps.append("invalid_normal_paper_trade_contract")
    if paper_exploration_gaps:
        lineage_gaps.append("invalid_paper_exploration_contract")
    if paper_training_gaps:
        lineage_gaps.append("invalid_paper_training_contract")
    lineage_gaps = list(dict.fromkeys(lineage_gaps))
    model_name = _text(_value(local_position, "model_name")) if local_position else ""
    lifecycle_key = _text(_value(history, "row_identity"))
    sample_source = "okx_position_history"
    strategy_training_role = (
        "aggregate_position_research_only"
        if len(entry_decision_ids) > 1
        else "obsolete_sampling_research_only"
        if obsolete_sampling_entry
        else "invalid_exploration_research_only"
        if paper_exploration_gaps
        else "invalid_paper_training_research_only"
        if paper_training_gaps
        else "invalid_normal_paper_research_only"
        if normal_paper_gaps
        else "entry_strategy"
    )
    sample = {
        "source": sample_source,
        "id": int(_value(history, "id", 0) or 0),
        "lifecycle_key": lifecycle_key,
        "position_id": position_id,
        "decision_id": decision_id,
        "entry_decision_ids": entry_decision_ids,
        "entry_decision_count": len(entry_decision_ids),
        "decision_lineage_source": decision_lineage_source,
        "features": entry_feature_snapshot,
        "decision_timestamp": opened_at.isoformat() if opened_at else None,
        "position_ids": position_ids,
        "okx_pos_id": _text(_value(history, "pos_id")),
        "entry_order_ids": entry_order_ids,
        "close_order_ids": close_order_ids,
        "entry_order_id": canonical_entry_order_id,
        "close_order_id": canonical_close_order_id,
        "linked_order_ids": linked_order_ids,
        "okx_trade_ids": _order_trade_ids(linked_orders),
        "model_name": model_name,
        "execution_mode": execution_mode,
        "source_execution_mode": source_execution_mode,
        "symbol": _text(_value(history, "symbol")),
        "inst_id": _text(_value(history, "inst_id")),
        "side": side,
        "close_status": _text(_value(history, "close_status")).lower(),
        "entry_price": entry_price,
        "close_price": close_price,
        "quantity": contracts,
        "quantity_unit": "contracts",
        "fill_contracts": fill_contracts,
        "contract_ct_val": ct_val,
        "contract_ct_val_source": contract_ct_val_source,
        "historical_contract_reconciliation": historical_contract_reconciliation,
        "lifecycle_fee_reconciliation": lifecycle_fee_reconciliation,
        "public_or_stored_contract_ct_val": public_or_stored_ct_val,
        "contract_ct_mult": ct_mult,
        "contract_lot_size": lot_size,
        "notional": canonical_notional,
        "notional_source": (
            str(historical_contract_reconciliation.get("source_authority") or "")
            if historical_contract_reconciliation.get("applied") is True
            else "okx_entry_fill_base_quantity_and_average_price"
            if canonical_notional is not None and canonical_notional > 0
            else ""
        ),
        **return_facts,
        "realized_pnl": realized_pnl,
        "gross_pnl": gross_pnl,
        "entry_fee": canonical_entry_fee,
        "close_fee": canonical_close_fee,
        "entry_fee_source": _text(entry_fill_group.get("fee_source")),
        "close_fee_source": _text(close_fill_group.get("fee_source")),
        "funding_fee": funding_fee,
        "funding_evidence_status": funding_training_evidence["status"],
        "funding_training_evidence": funding_training_evidence,
        "funding_fee_to_notional_ratio": funding_training_evidence["funding_fee_to_notional_ratio"],
        "funding_bill_count": funding_training_evidence["account_bill_count"],
        "funding_bill_ids": funding_training_evidence["account_bill_ids"],
        "funding_attribution_complete": funding_training_evidence["attribution_complete"],
        "liquidation_penalty": liquidation_penalty,
        "official_fee_signed": fee_signed,
        "realized_net_pnl_formula": REALIZED_NET_PNL_FORMULA,
        "realized_net_pnl_components": {
            "gross_pnl_usdt": gross_pnl,
            "official_fee_signed_usdt": fee_signed,
            "funding_fee_usdt": funding_fee,
            "liquidation_penalty_usdt": liquidation_penalty,
            "components_total_usdt": settlement_expected,
            "reported_realized_net_pnl_usdt": realized_pnl,
            "formula_consistent": abs(realized_pnl - settlement_expected) <= settlement_tolerance,
        },
        "settlement_components_total": settlement_expected,
        "holding_minutes": holding_minutes,
        "leverage": _safe_float(_value(history, "leverage"), 1.0) or 1.0,
        "position_history_pnl_ratio": _safe_float(_value(history, "pnl_ratio"), None),
        "planned_stop_loss_price": stop_loss_price,
        "planned_take_profit_price": take_profit_price,
        "stop_loss_fill_confirmed": stop_loss_fill_confirmed,
        "slippage": canonical_slippage,
        "slippage_source": canonical_slippage_source,
        "entry_execution_slippage_usdt": entry_execution_slippage_usdt,
        "close_execution_slippage_usdt": close_execution_slippage_usdt,
        "execution_slippage_usdt": execution_slippage_usdt,
        "entry_execution_slippage_complete": (
            entry_fill_group.get("execution_slippage_complete") is True
        ),
        "close_execution_slippage_complete": (
            close_fill_group.get("execution_slippage_complete") is True
        ),
        "execution_slippage_failures": {
            "entry": dict(entry_fill_group.get("execution_slippage_failures") or {}),
            "close": dict(close_fill_group.get("execution_slippage_failures") or {}),
        },
        "lifecycle_order_allocation_failures": {
            "entry": dict(entry_fill_group.get("lifecycle_order_allocation_failures") or {}),
            "close": dict(close_fill_group.get("lifecycle_order_allocation_failures") or {}),
        },
        "protection_execution_supervision_ready": bool(protection_execution),
        "protection_lifecycle_complete": bool(protection_execution and protection_submission),
        "protection_execution_gaps": protection_execution_gaps,
        "protection_algo_id": _text(protection_execution.get("algo_id")) or None,
        "protection_generated_order_id": (
            _text(protection_execution.get("generated_order_id")) or None
        ),
        "protection_actual_side": (_text(protection_execution.get("actual_side")) or None),
        "exchange_configured_trigger_price": _safe_float(
            protection_execution.get("configured_trigger_price"),
            None,
        ),
        "actual_trigger_market_price": _safe_float(
            protection_execution.get("actual_trigger_market_price"),
            None,
        ),
        "actual_trigger_market_price_available": (
            protection_execution.get("actual_trigger_market_price_available") is True
        ),
        "protection_exchange_confirmed_at": (
            protection_submission.get("exchange_confirmed_at")
            or _iso_from_ms(protection_execution.get("exchange_confirmed_at_ms"))
        ),
        "protection_triggered_at": _iso_from_ms(protection_execution.get("triggered_at_ms")),
        "protection_fill_started_at": _iso_from_ms(protection_execution.get("fill_started_at_ms")),
        "protection_fill_completed_at": _iso_from_ms(
            protection_execution.get("fill_completed_at_ms")
        ),
        "trigger_to_first_fill_ms": _safe_float(
            protection_execution.get("trigger_to_first_fill_ms"),
            None,
        ),
        "protection_fill_mark_price": _safe_float(
            protection_execution.get("fill_mark_price"),
            None,
        ),
        "protection_fill_index_price": _safe_float(
            protection_execution.get("fill_index_price"),
            None,
        ),
        "protection_fill_path_min_price": _safe_float(
            protection_execution.get("fill_path_min_price"),
            None,
        ),
        "protection_fill_path_max_price": _safe_float(
            protection_execution.get("fill_path_max_price"),
            None,
        ),
        "protection_fill_mark_slippage_pct": _safe_float(
            protection_execution.get("fill_mark_slippage_pct"),
            None,
        ),
        "execution_risk_budget_usdt": budget_facts["risk_budget_usdt"],
        "execution_planned_stressed_loss_usdt": budget_facts["planned_stressed_loss_usdt"],
        "execution_actual_loss_usdt": budget_facts["actual_loss_usdt"],
        "execution_actual_over_budget_loss_usdt": budget_facts["actual_over_budget_loss_usdt"],
        "strategy_entry_kind": "normal_strategy_trade",
        "historical_entry_contract_kind": (
            "normal_paper_v8"
            if valid_normal_paper and current_normal_paper
            else "paper_training"
            if valid_paper_training
            else "paper_exploration"
            if valid_paper_exploration
            else "normal_paper_v1"
            if valid_normal_paper and historical_normal_paper
            else "normal_paper_v2"
            if valid_normal_paper and legacy_v2_normal_paper
            else "normal_paper_v3"
            if valid_normal_paper and legacy_v3_normal_paper
            else "normal_paper_v4"
            if valid_normal_paper and legacy_v4_normal_paper
            else "normal_paper_v5"
            if valid_normal_paper and legacy_v5_normal_paper
            else "normal_paper_v6"
            if valid_normal_paper and legacy_v6_normal_paper
            else "normal_paper_v7"
            if valid_normal_paper and legacy_v7_normal_paper
            else None
        ),
        "strategy_selection_reason": (
            _text(normal_paper.get("selection_reason"))
            if valid_normal_paper
            and (
                current_normal_paper
                or legacy_v7_normal_paper
                or legacy_v6_normal_paper
                or legacy_v5_normal_paper
                or legacy_v4_normal_paper
                or legacy_v3_normal_paper
                or legacy_v2_normal_paper
            )
            else _text(normal_paper.get("route_kind"))
            if valid_normal_paper and historical_normal_paper
            else _text(paper_training.get("selection_reason"))
            if valid_paper_training
            else _text(paper_exploration.get("selection_reason"))
            if valid_paper_exploration
            else "governed_fee_after_return_strategy"
        ),
        "normal_paper_trade_evidence": (
            {
                "version": normal_paper.get("version"),
                "contract_generation": (
                    "current_quality_observation_v8"
                    if current_normal_paper
                    and normal_paper.get("selection_reason")
                    == "paper_quality_observation"
                    else "current_validated_v8"
                    if current_normal_paper
                    else "historical_quality_v7"
                    if legacy_v7_normal_paper
                    else "historical_quality_v6"
                    if legacy_v6_normal_paper
                    else "historical_quality_v5"
                    if legacy_v5_normal_paper
                    else "historical_expected_net_v4"
                    if legacy_v4_normal_paper
                    else "historical_dynamic_v3"
                    if legacy_v3_normal_paper
                    else "historical_fixed_v2"
                    if legacy_v2_normal_paper
                    else "historical_normal_v1"
                ),
                "entry_type": normal_paper.get("entry_type"),
                "route_kind": normal_paper.get("route_kind"),
                "decision_authority": normal_paper.get("decision_authority"),
                "selection_reason": normal_paper.get("selection_reason"),
                "expected_net_return_pct": normal_paper.get("expected_net_return_pct"),
                "objective_net_return_pct": normal_paper.get("objective_net_return_pct"),
                "loss_probability": normal_paper.get("loss_probability"),
                "prediction_horizon_minutes": normal_paper.get("prediction_horizon_minutes"),
                "production_permission": normal_paper.get("production_permission"),
            }
            if valid_normal_paper
            else {}
        ),
        "paper_exploration_evidence": (
            {
                "version": paper_exploration.get("version"),
                "selected_side": paper_exploration.get("selected_side"),
                "expected_net_return_pct": paper_exploration.get("expected_net_return_pct"),
                "return_lcb_pct": paper_exploration.get("return_lcb_pct"),
                "information_value_score": paper_exploration.get("information_value_score"),
                "single_trade_risk_fraction_cap": paper_exploration.get(
                    "single_trade_risk_fraction_cap"
                ),
                "portfolio_risk_fraction_cap": paper_exploration.get("portfolio_risk_fraction_cap"),
                "sample_target": paper_exploration.get("sample_target"),
                "daily_sample_quota": paper_exploration.get("daily_sample_quota"),
            }
            if valid_paper_exploration
            else {}
        ),
        "paper_training_evidence": (
            {
                "version": paper_training.get("version"),
                "trade_kind": paper_training.get("trade_kind"),
                "selected_side": paper_training.get("selected_side"),
                "signal_source": paper_training.get("signal_source"),
                "expected_net_return_pct": paper_training.get("expected_net_return_pct"),
                "return_lcb_pct": paper_training.get("return_lcb_pct"),
                "loss_tolerant_for_training": paper_training.get("loss_tolerant_for_training"),
                "continuous_training_after_settlement": paper_training.get(
                    "continuous_training_after_settlement"
                ),
                "sample_target": paper_training.get("sample_target"),
                "daily_sample_quota": paper_training.get("daily_sample_quota"),
            }
            if valid_paper_training
            else {}
        ),
        "close_order_types": sorted(
            {
                _text(_value(order, "order_type")).lower()
                for order in close_orders
                if _text(_value(order, "order_type"))
            }
        ),
        "raw_llm_response": raw_llm_response,
        "outcome": "profit" if realized_pnl > 0 else "loss" if realized_pnl < 0 else "flat",
        "pnl_source": "okx_position_history_realized_pnl",
        "settlement_source": "okx_position_history_realized_pnl",
        "funding_fee_source": "okx_positions_history.fundingFee",
        "trade_fact_trusted": not gaps,
        "trade_fact_trust_reason": gaps[0] if gaps else "",
        "strategy_lineage_complete": not lineage_gaps,
        "strategy_lineage_gaps": lineage_gaps,
        "strategy_entry_supervision_eligible": bool(
            len(entry_decision_ids) <= 1
            and not obsolete_sampling_entry
            and not normal_paper_gaps
            and not paper_exploration_gaps
            and not paper_training_gaps
        ),
        "strategy_training_role": strategy_training_role,
        "training_evidence_gaps": list(dict.fromkeys([*gaps, *lineage_gaps])),
        "label_timestamp": closed_at.isoformat() if closed_at else None,
    }
    decision_authority = _decision_authority(
        raw_llm_response=raw_llm_response,
        execution_mode=execution_mode,
        valid_normal_paper=valid_normal_paper,
        valid_legacy_paper=(valid_paper_exploration or valid_paper_training),
        strategy_training_role=strategy_training_role,
    )
    sample["decision_authority"] = decision_authority
    model_shadow_prediction = _model_shadow_prediction(
        raw_llm_response,
        decision_authority=decision_authority,
    )
    if model_shadow_prediction:
        sample["model_shadow_prediction"] = model_shadow_prediction
    if canonical_notional is not None and canonical_notional > 0:
        sample["net_return_after_all_cost_pct"] = realized_pnl / canonical_notional * 100.0
    sample["profit_training_contract"] = validate_profit_training_sample(sample).to_dict()
    return sample
