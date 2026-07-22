"""Current-position takeover contract built from authoritative live facts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from math import isclose, isfinite
from typing import Any

from core.symbols import normalize_trading_symbol

CURRENT_POSITION_MANAGEMENT_VERSION = "2026-07-15.current-position-management.v1"
CURRENT_POSITION_MANAGEMENT_KIND = "current_position_takeover"
ALLOWED_MANAGEMENT_ACTIONS = (
    "hold",
    "reduce",
    "close",
    "protection_repair",
)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _row_value(row: Any, *names: str) -> Any:
    if isinstance(row, dict):
        for name in names:
            if name in row and row.get(name) is not None:
                return row.get(name)
        return None
    for name in names:
        value = getattr(row, name, None)
        if value is not None:
            return value
    return None


def build_current_position_management_contract(
    facts: dict[str, Any],
    *,
    previous_contract: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a reduce-only management contract without reconstructing entry intent."""

    generated_at = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
    previous = _safe_dict(previous_contract)
    previous_canary_lifecycle = _safe_dict(previous.get("paper_canary_lifecycle"))
    refreshed_canary_lifecycle = _safe_dict(facts.get("paper_canary_lifecycle"))
    paper_canary_lifecycle = dict(
        previous_canary_lifecycle or refreshed_canary_lifecycle
    )
    if paper_canary_lifecycle and refreshed_canary_lifecycle:
        for key in ("decision_id", "artifact_version", "executed_at"):
            if paper_canary_lifecycle.get(key) in (None, ""):
                refreshed_value = refreshed_canary_lifecycle.get(key)
                if refreshed_value not in (None, ""):
                    paper_canary_lifecycle[key] = refreshed_value
    previous_training_lifecycle = _safe_dict(previous.get("paper_training_lifecycle"))
    refreshed_training_lifecycle = _safe_dict(facts.get("paper_training_lifecycle"))
    paper_training_lifecycle = dict(
        previous_training_lifecycle or refreshed_training_lifecycle
    )
    if paper_training_lifecycle and refreshed_training_lifecycle:
        for key in ("decision_id", "executed_at"):
            if paper_training_lifecycle.get(key) in (None, ""):
                refreshed_value = refreshed_training_lifecycle.get(key)
                if refreshed_value not in (None, ""):
                    paper_training_lifecycle[key] = refreshed_value
    symbol = normalize_trading_symbol(facts.get("symbol"))
    side = str(facts.get("side") or "").lower().strip()
    quantity = abs(_safe_float(facts.get("quantity")))
    contracts = abs(_safe_float(facts.get("contracts")))
    entry_price = max(_safe_float(facts.get("entry_price")), 0.0)
    current_price = max(_safe_float(facts.get("current_price")), 0.0)
    entry_fee = max(_safe_float(facts.get("entry_fee_usdt")), 0.0)
    full_entry_fee = max(_safe_float(facts.get("full_entry_fee_usdt")), 0.0)
    full_entry_notional = max(_safe_float(facts.get("full_entry_notional_usdt")), 0.0)
    stop_loss = max(_safe_float(facts.get("stop_loss_price")), 0.0)
    take_profit = max(_safe_float(facts.get("take_profit_price")), 0.0)
    position_stressed_loss = max(
        _safe_float(facts.get("position_stressed_loss_usdt")),
        0.0,
    )
    portfolio_stressed_loss = max(
        _safe_float(facts.get("portfolio_stressed_loss_usdt")),
        0.0,
    )
    portfolio_gross_notional = max(
        _safe_float(facts.get("portfolio_gross_notional_usdt")),
        0.0,
    )
    account_equity = max(_safe_float(facts.get("account_equity_usdt")), 0.0)
    open_position_count = max(_safe_int(facts.get("open_position_count")), 0)
    entry_order_ids = sorted(
        {
            str(value).strip()
            for value in facts.get("entry_order_ids") or []
            if str(value or "").strip()
        }
    )
    decision_ids = sorted(
        {
            _safe_int(value)
            for value in facts.get("entry_decision_ids") or []
            if _safe_int(value) > 0
        }
    )
    entry_contract_gaps = list(
        dict.fromkeys(
            str(value)
            for value in facts.get("original_entry_contract_gaps") or []
            if str(value or "").strip()
        )
    )
    protection_orders = [
        dict(value)
        for value in facts.get("protection_orders") or []
        if isinstance(value, dict)
    ]

    blockers: list[str] = []
    if not symbol or side not in {"long", "short"}:
        blockers.append("current_position_identity_incomplete")
    if quantity <= 0 or contracts <= 0 or entry_price <= 0 or current_price <= 0:
        blockers.append("current_position_valuation_incomplete")
    if facts.get("entry_fee_evidence_complete") is not True:
        blockers.append("authoritative_entry_fee_evidence_incomplete")
    if not entry_order_ids:
        blockers.append("entry_order_lineage_missing")
    if facts.get("protection_evidence_complete") is not True:
        blockers.append("okx_protection_evidence_incomplete")
    if stop_loss <= 0 or take_profit <= 0 or position_stressed_loss <= 0:
        blockers.append("current_oco_stressed_loss_incomplete")
    elif not (
        (side == "long" and stop_loss < entry_price < take_profit)
        or (side == "short" and take_profit < entry_price < stop_loss)
    ):
        blockers.append("current_oco_direction_invalid")
    if account_equity <= 0:
        blockers.append("current_account_equity_missing")
    if portfolio_gross_notional <= 0 or portfolio_stressed_loss <= 0:
        blockers.append("current_portfolio_state_incomplete")
    if open_position_count <= 0:
        blockers.append("current_portfolio_inventory_missing")

    stressed_loss_share = (
        min(max(position_stressed_loss / portfolio_stressed_loss, 0.0), 1.0)
        if portfolio_stressed_loss > 0
        else 0.0
    )
    equal_risk_share = 1.0 / open_position_count if open_position_count > 0 else 0.0
    concentration_pressure = (
        min(
            max(
                (stressed_loss_share - equal_risk_share) / (1.0 - equal_risk_share),
                0.0,
            ),
            1.0,
        )
        if open_position_count > 1
        else 0.0
    )
    exit_fee_rate_proxy = (
        full_entry_fee / full_entry_notional if full_entry_notional > 0 else 0.0
    )
    if facts.get("entry_fee_evidence_complete") is True and full_entry_notional <= 0:
        blockers.append("entry_fee_rate_basis_missing")

    blockers = list(dict.fromkeys(blockers))
    original_entry_status = (
        "complete_at_entry"
        if facts.get("original_entry_contract_complete") is True and not entry_contract_gaps
        else "historical_entry_contract_incomplete_preserved"
    )
    takeover_at = (
        str(previous.get("takeover_at") or "").strip()
        if previous.get("contract_version") == CURRENT_POSITION_MANAGEMENT_VERSION
        and previous.get("kind") == CURRENT_POSITION_MANAGEMENT_KIND
        else ""
    ) or generated_at

    input_facts = {
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "contracts": contracts,
        "entry_price": entry_price,
        "current_price": current_price,
        "entry_fee_usdt": entry_fee,
        "full_entry_fee_usdt": full_entry_fee,
        "full_entry_notional_usdt": full_entry_notional,
        "stop_loss_price": stop_loss,
        "take_profit_price": take_profit,
        "position_stressed_loss_usdt": position_stressed_loss,
        "portfolio_stressed_loss_usdt": portfolio_stressed_loss,
        "portfolio_gross_notional_usdt": portfolio_gross_notional,
        "account_equity_usdt": account_equity,
        "open_position_count": open_position_count,
        "entry_order_ids": entry_order_ids,
        "entry_decision_ids": decision_ids,
        "original_entry_contract_gaps": entry_contract_gaps,
        "protection_orders": protection_orders,
        "entry_fee_evidence_complete": facts.get("entry_fee_evidence_complete") is True,
        "protection_evidence_complete": facts.get("protection_evidence_complete") is True,
        "paper_canary_lifecycle": paper_canary_lifecycle,
        "paper_training_lifecycle": paper_training_lifecycle,
    }
    provenance = {
        "source": "okx_current_position_fills_protection_account_and_portfolio_facts",
        "observation_window": "current_open_position_takeover_refresh",
        "sample_count": len(entry_order_ids) + len(protection_orders) + open_position_count + 1,
        "generated_at": generated_at,
        "strategy_version": CURRENT_POSITION_MANAGEMENT_VERSION,
        "fallback_reason": ",".join(blockers),
        "input_fingerprint": _fingerprint(input_facts),
    }
    contract = {
        "contract_version": CURRENT_POSITION_MANAGEMENT_VERSION,
        "kind": CURRENT_POSITION_MANAGEMENT_KIND,
        "management_eligible": not blockers,
        "blockers": blockers,
        "takeover_at": takeover_at,
        "refreshed_at": generated_at,
        "symbol": symbol,
        "side": side,
        "quantity": round(quantity, 12),
        "contracts": round(contracts, 12),
        "entry_price": round(entry_price, 12),
        "current_price": round(current_price, 12),
        "entry_fee_usdt": round(entry_fee, 12),
        "full_entry_fee_usdt": round(full_entry_fee, 12),
        "entry_fee_evidence_complete": facts.get("entry_fee_evidence_complete") is True,
        "entry_fee_source": facts.get("entry_fee_source"),
        "exit_fee_rate_proxy": round(exit_fee_rate_proxy, 12),
        "exit_fee_rate_source": "authoritative_entry_fill_fee_rate_proxy",
        "stop_loss_price": round(stop_loss, 12),
        "take_profit_price": round(take_profit, 12),
        "protection_evidence_complete": facts.get("protection_evidence_complete") is True,
        "protection_orders": protection_orders,
        "position_notional_usdt": round(quantity * current_price, 12),
        "position_stressed_loss_usdt": round(position_stressed_loss, 12),
        "portfolio_stressed_loss_usdt": round(portfolio_stressed_loss, 12),
        "portfolio_gross_notional_usdt": round(portfolio_gross_notional, 12),
        "account_equity_usdt": round(account_equity, 12),
        "portfolio_stressed_loss_ratio": round(
            portfolio_stressed_loss / account_equity if account_equity > 0 else 0.0,
            12,
        ),
        "portfolio_gross_notional_ratio": round(
            portfolio_gross_notional / account_equity if account_equity > 0 else 0.0,
            12,
        ),
        "position_stressed_loss_share": round(stressed_loss_share, 12),
        "dynamic_equal_risk_share": round(equal_risk_share, 12),
        "portfolio_concentration_pressure": round(concentration_pressure, 12),
        "open_position_count": open_position_count,
        "can_expand_position": False,
        "can_increase_leverage": False,
        "allowed_actions": list(ALLOWED_MANAGEMENT_ACTIONS),
        "replaces_original_entry_contract": False,
        "original_entry_contract_status": original_entry_status,
        "original_entry_order_ids": entry_order_ids,
        "original_entry_decision_ids": decision_ids,
        "original_entry_contract_gaps": entry_contract_gaps,
        "policy_provenance": provenance,
    }
    if paper_canary_lifecycle:
        contract["paper_canary_lifecycle"] = paper_canary_lifecycle
    if paper_training_lifecycle:
        contract["paper_training_lifecycle"] = paper_training_lifecycle
    contract["policy_provenance"]["contract_fingerprint"] = _fingerprint(contract)
    return contract


def current_position_management_contract_complete(
    position: Any,
    contract: Any | None = None,
) -> bool:
    """Validate that a persisted takeover contract still matches the open position."""

    value = _safe_dict(
        contract
        if contract is not None
        else getattr(position, "current_management_contract", None)
    )
    if (
        value.get("contract_version") != CURRENT_POSITION_MANAGEMENT_VERSION
        or value.get("kind") != CURRENT_POSITION_MANAGEMENT_KIND
        or value.get("management_eligible") is not True
        or value.get("entry_fee_evidence_complete") is not True
        or value.get("protection_evidence_complete") is not True
        or value.get("can_expand_position") is not False
        or value.get("can_increase_leverage") is not False
        or value.get("blockers")
        or tuple(value.get("allowed_actions") or ()) != ALLOWED_MANAGEMENT_ACTIONS
    ):
        return False
    provenance = _safe_dict(value.get("policy_provenance"))
    if not all(
        str(provenance.get(key) or "").strip()
        for key in (
            "source",
            "observation_window",
            "generated_at",
            "strategy_version",
            "input_fingerprint",
            "contract_fingerprint",
        )
    ):
        return False
    if _safe_int(provenance.get("sample_count")) <= 0 or provenance.get("fallback_reason"):
        return False

    position_quantity = abs(_safe_float(_row_value(position, "quantity", "base_quantity")))
    position_entry = _safe_float(_row_value(position, "entry_price", "entryPrice"))
    position_fee = max(
        _safe_float(_row_value(position, "entry_fee", "entry_fee_usdt")),
        0.0,
    )
    position_symbol = normalize_trading_symbol(_row_value(position, "symbol"))
    position_side = str(_row_value(position, "side", "position_side") or "").lower()
    position_stop = _safe_float(_row_value(position, "stop_loss_price", "stop_loss"))
    position_take_profit = _safe_float(
        _row_value(position, "take_profit_price", "take_profit")
    )
    protection_orders = [
        item for item in value.get("protection_orders") or [] if isinstance(item, dict)
    ]
    protection_quantity = sum(
        abs(_safe_float(item.get("contracts"))) for item in protection_orders
    )
    protection_rows_complete = bool(
        protection_orders
        and all(
            str(item.get("algo_id") or "").strip()
            and str(item.get("state") or "").lower() in {
                "live",
                "effective",
                "partially_effective",
                "open",
                "pending",
            }
            and item.get("reduce_only") is True
            and _safe_float(item.get("contracts")) > 0
            and _safe_float(item.get("stop_loss_price")) > 0
            and _safe_float(item.get("take_profit_price")) > 0
            for item in protection_orders
        )
        and isclose(
            protection_quantity,
            abs(_safe_float(value.get("contracts"))),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    )
    scalar_protection_matches = bool(
        len(protection_orders) > 1
        or (
            isclose(
                position_stop,
                _safe_float(value.get("stop_loss_price")),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            and isclose(
                position_take_profit,
                _safe_float(value.get("take_profit_price")),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        )
    )
    return bool(
        position_symbol == normalize_trading_symbol(value.get("symbol"))
        and position_side == str(value.get("side") or "").lower()
        and position_quantity > 0
        and position_entry > 0
        and isclose(
            position_quantity,
            abs(_safe_float(value.get("quantity"))),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        and isclose(
            position_entry,
            _safe_float(value.get("entry_price")),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        and isclose(
            position_fee,
            max(_safe_float(value.get("entry_fee_usdt")), 0.0),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        and protection_rows_complete
        and scalar_protection_matches
    )
