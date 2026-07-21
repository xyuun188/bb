"""Authoritative dynamic risk budget and entry sizing.

Risk budget is generated before position size.  The final notional is solved
from that independent budget and a stressed loss fraction; downstream callers
may only reconcile the contract to a smaller executable notional.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite, sqrt
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from core.symbols import normalize_trading_symbol, okx_inst_id_from_symbol
from core.training_contracts import AUTHORITATIVE_TRADE_OUTCOME_SOURCES
from services.dynamic_leverage_allocator import DynamicLeverageAllocator, DynamicLeverageInput
from services.paper_exploration import (
    PAPER_EXPLORATION_MAX_LCB_GAP_RATIO,
    PAPER_EXPLORATION_MAX_PORTFOLIO_RISK_FRACTION,
    PAPER_EXPLORATION_MAX_SINGLE_TRADE_RISK_FRACTION,
    PAPER_EXPLORATION_SIZING_VERSION,
    is_paper_exploration_decision,
    paper_exploration_selection_reasons,
)

RISK_SIZING_VERSION = "2026-07-15.independent-profit-risk-budget.v3"
LEVERAGE_TIER_SELECTION_VERSION = "2026-07-15.okx-target-notional-tier.v1"

EntryProfitRiskSizingEvaluator = Callable[
    [DecisionOutput, str, list[dict[str, Any]]],
    Awaitable[None],
]
EntryBalanceProvider = Callable[[str, DecisionOutput | None], Awaitable[float | None]]
EntryExchangeRiskFactsProvider = Callable[
    [str, DecisionOutput, list[dict[str, Any]]],
    Awaitable[dict[str, Any]],
]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _normalized_ratio(value: Any) -> float:
    number = max(_safe_float(value, 0.0), 0.0)
    return number / 100.0 if number > 1.0 else number


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tier_value(tier: dict[str, Any], *keys: str) -> float:
    info = _safe_dict(tier.get("info"))
    for key in keys:
        value = _safe_float(tier.get(key), float("nan"))
        if isfinite(value) and value > 0:
            return value
        value = _safe_float(info.get(key), float("nan"))
        if isfinite(value) and value > 0:
            return value
    return 0.0


def select_okx_leverage_tier(
    leverage_tiers: Any,
    *,
    target_notional_usdt: float,
    mark_price: float,
    contract_spec: dict[str, Any] | None,
    current_position_notional_usdt: float = 0.0,
    current_position_contracts: float = 0.0,
) -> dict[str, Any]:
    """Select the authoritative OKX leverage tier for the projected position."""

    rows = [dict(row) for row in _safe_list(leverage_tiers) if isinstance(row, dict)]
    valid_rows = [row for row in rows if _tier_value(row, "maxLeverage", "max_leverage") >= 1]
    target_notional = max(_safe_float(target_notional_usdt, 0.0), 0.0)
    current_notional = max(_safe_float(current_position_notional_usdt, 0.0), 0.0)
    current_contracts = max(_safe_float(current_position_contracts, 0.0), 0.0)
    price = max(_safe_float(mark_price, 0.0), 0.0)
    spec = _safe_dict(contract_spec)
    ct_val = max(_safe_float(spec.get("ctVal"), 0.0), 0.0)
    ct_mult = max(_safe_float(spec.get("ctMult"), 0.0), 0.0)
    contract_unit_notional = ct_val * ct_mult * price
    target_contracts = (
        target_notional / contract_unit_notional if contract_unit_notional > 0 else 0.0
    )
    projected_notional = current_notional + target_notional
    projected_contracts = current_contracts + target_contracts
    reasons: list[str] = []
    selected: dict[str, Any] = {}
    selection_source = ""

    if target_notional <= 0:
        reasons.append("target_notional_missing_for_leverage_tier")
    if not valid_rows:
        reasons.append("okx_leverage_tiers_missing")

    notional_bounded = [
        row for row in valid_rows if _tier_value(row, "maxNotional", "max_notional") > 0
    ]
    size_bounded = [row for row in valid_rows if _tier_value(row, "maxSz", "max_size") > 0]
    if not reasons and notional_bounded:
        selected = min(
            (
                row
                for row in notional_bounded
                if projected_notional
                <= _tier_value(row, "maxNotional", "max_notional")
            ),
            key=lambda row: _tier_value(row, "maxNotional", "max_notional"),
            default={},
        )
        selection_source = "okx_normalized_notional_bounds"
    elif not reasons and size_bounded:
        if contract_unit_notional <= 0:
            reasons.append("okx_contract_unit_notional_missing_for_leverage_tier")
        else:
            selected = min(
                (
                    row
                    for row in size_bounded
                    if projected_contracts <= _tier_value(row, "maxSz", "max_size")
                ),
                key=lambda row: _tier_value(row, "maxSz", "max_size"),
                default={},
            )
            selection_source = "okx_contract_size_bounds"
    elif not reasons:
        if len(valid_rows) == 1:
            selected = valid_rows[0]
            selection_source = "okx_single_reported_tier_maximum"
        else:
            reasons.append("okx_leverage_tier_bounds_missing")

    if not reasons and not selected:
        reasons.append("projected_position_exceeds_reported_leverage_tiers")
    max_leverage = _tier_value(selected, "maxLeverage", "max_leverage")
    if not reasons and max_leverage < 1:
        reasons.append("selected_okx_leverage_tier_missing_maximum")

    eligible = not reasons
    generated_at = datetime.now(UTC).isoformat()
    audit_inputs = {
        "target_notional_usdt": target_notional,
        "current_position_notional_usdt": current_notional,
        "current_position_contracts": current_contracts,
        "mark_price": price,
        "ct_val": ct_val,
        "ct_mult": ct_mult,
        "tier_count": len(rows),
        "tiers_fingerprint": _fingerprint(rows),
    }
    return {
        "production_eligible": eligible,
        "reason": "okx_target_notional_tier_selected" if eligible else ",".join(reasons),
        "selection_source": selection_source,
        "max_leverage": max_leverage if eligible else 0.0,
        "selected_tier": selected if eligible else {},
        "target_notional_usdt": target_notional,
        "current_position_notional_usdt": current_notional,
        "projected_position_notional_usdt": projected_notional,
        "mark_price": price,
        "contract_spec": spec,
        "contract_unit_notional_usdt": contract_unit_notional,
        "target_contracts": target_contracts,
        "current_position_contracts": current_contracts,
        "projected_position_contracts": projected_contracts,
        "tier_count": len(rows),
        "tiers_fingerprint": audit_inputs["tiers_fingerprint"],
        "policy_provenance": {
            "source": "okx_position_tiers_and_native_contract_spec",
            "observation_window": "current_pre_entry_projected_position",
            "sample_count": len(valid_rows),
            "generated_at": generated_at,
            "strategy_version": LEVERAGE_TIER_SELECTION_VERSION,
            "fallback_reason": "" if eligible else ",".join(reasons),
            "input_fingerprint": _fingerprint(audit_inputs),
        },
    }


def _production_source_count(opportunity: dict[str, Any]) -> int:
    breakdown = _safe_dict(opportunity.get("expected_net_breakdown"))
    return sum(
        1
        for component in _safe_list(breakdown.get("components"))
        if (
            _safe_dict(component).get("included_in_return_distribution") is True
            or (
                "included_in_return_distribution" not in _safe_dict(component)
                and _safe_dict(component).get("production_eligible") is True
            )
        )
    )


def _side(decision: DecisionOutput) -> str:
    return "long" if decision.action == Action.LONG else "short"


def _atr_ratio(decision: DecisionOutput) -> float:
    snapshot = _safe_dict(decision.feature_snapshot)
    explicit = _normalized_ratio(snapshot.get("atr_pct"))
    if explicit > 0:
        return explicit
    atr = max(_safe_float(snapshot.get("atr_14"), 0.0), 0.0)
    price = max(_safe_float(snapshot.get("current_price", snapshot.get("close")), 0.0), 0.0)
    return atr / price if atr > 0 and price > 0 else 0.0


def _path_adverse_fraction(decision: DecisionOutput) -> float:
    values = [
        _safe_float(value, float("nan"))
        for value in _safe_list(_safe_dict(decision.feature_snapshot).get("close_sequence"))
    ]
    prices = [value for value in values if isfinite(value) and value > 0]
    if len(prices) < 2:
        return 0.0
    adverse = 0.0
    anchor = prices[0]
    if _side(decision) == "long":
        for price in prices[1:]:
            anchor = max(anchor, price)
            adverse = max(adverse, (anchor - price) / anchor)
    else:
        for price in prices[1:]:
            anchor = min(anchor, price)
            adverse = max(adverse, (price - anchor) / anchor)
    return adverse


def _actual_trade_calibration(opportunity: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for component in _safe_list(_safe_dict(opportunity.get("expected_net_breakdown")).get("components")):
        item = _safe_dict(component)
        if item.get("included_in_return_distribution") is not True:
            continue
        calibration = _safe_dict(item.get("actual_trade_calibration"))
        if calibration.get("source_authority") in AUTHORITATIVE_TRADE_OUTCOME_SOURCES:
            rows.append(calibration)
    if not rows:
        return {}
    symbol_rows = [row for row in rows if row.get("profile_source") == "symbol_side"]
    return (symbol_rows or rows)[0]


def _distribution_tail(value: Any) -> float:
    distribution = _safe_dict(value)
    return max(
        _safe_float(distribution.get("expected"), 0.0),
        _safe_float(distribution.get("upper_hinge"), 0.0),
        _safe_float(distribution.get("maximum"), 0.0),
        0.0,
    )


def _rolling_realized_lcb_pct(calibration: dict[str, Any]) -> float | None:
    distribution = _safe_dict(calibration.get("net_return_after_cost_pct"))
    count = int(_safe_float(distribution.get("count"), 0.0))
    lower = _safe_float(distribution.get("lower_hinge"), float("nan"))
    return lower if count > 0 and isfinite(lower) else None


def _position_notional(
    position: dict[str, Any],
    contract_specs: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    info = _safe_dict(position.get("info"))
    direct = abs(
        _safe_float(
            position.get("notional")
            or position.get("notional_usd")
            or position.get("notionalUsd")
            or info.get("notionalUsd")
            or info.get("notional")
            or info.get("posValue"),
            0.0,
        )
    )
    inst_id = str(info.get("instId") or position.get("okx_inst_id") or "").upper()
    spec = _safe_dict(contract_specs.get(inst_id))
    contracts = abs(
        _safe_float(
            position.get("contracts") or position.get("sz") or info.get("pos") or info.get("qty"),
            0.0,
        )
    )
    ct_val = max(
        _safe_float(spec.get("ctVal"), 0.0),
        _safe_float(position.get("contract_size") or position.get("contractSize"), 0.0),
        _safe_float(info.get("ctVal"), 0.0),
    )
    ct_mult = max(_safe_float(spec.get("ctMult") or info.get("ctMult"), 0.0), 0.0)
    mark = max(
        _safe_float(position.get("current_price") or position.get("markPrice"), 0.0),
        _safe_float(info.get("markPx"), 0.0),
    )
    calculated = contracts * ct_val * ct_mult * mark
    notional = direct if direct > 0 else calculated
    return notional, {
        "inst_id": inst_id,
        "contracts": contracts,
        "ct_val": ct_val,
        "ct_mult": ct_mult,
        "mark_price": mark,
        "notional_source": "okx_notional_usd" if direct > 0 else "okx_contract_spec_and_mark",
    }


def _candidate_existing_exposure(
    decision: DecisionOutput,
    positions: list[dict[str, Any]],
    contract_specs: dict[str, Any],
) -> dict[str, float]:
    candidate_symbol = normalize_trading_symbol(decision.symbol)
    candidate_side = _side(decision)
    notional = 0.0
    contracts = 0.0
    for position in positions:
        item = _safe_dict(position)
        if item.get("is_open", True) is False:
            continue
        info = _safe_dict(item.get("info"))
        position_symbol = normalize_trading_symbol(
            item.get("symbol") or info.get("instId") or item.get("okx_inst_id")
        )
        position_side = str(item.get("side") or info.get("posSide") or "").lower()
        if position_symbol != candidate_symbol or position_side != candidate_side:
            continue
        position_notional, valuation = _position_notional(item, contract_specs)
        notional += position_notional
        contracts += max(_safe_float(valuation.get("contracts"), 0.0), 0.0)
    return {
        "notional_usdt": notional,
        "contracts": contracts,
    }


def _portfolio_risk_snapshot(
    positions: list[dict[str, Any]],
    *,
    candidate_side: str,
    contract_specs: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    current_stressed_loss = 0.0
    current_margin = 0.0
    gross_notional = 0.0
    same_side_notional = 0.0
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for position in positions:
        item = _safe_dict(position)
        if item.get("is_open", True) is False:
            continue
        notional, valuation = _position_notional(item, contract_specs)
        side = str(item.get("side") or "").lower()
        mark = valuation["mark_price"]
        stop = max(_safe_float(item.get("stop_loss") or item.get("stop_loss_price"), 0.0), 0.0)
        leverage = max(_safe_float(item.get("leverage") or _safe_dict(item.get("info")).get("lever"), 1.0), 1.0)
        direct_margin = _safe_float(
                item.get("margin")
                or item.get("initial_margin")
                or item.get("initialMargin")
                or _safe_dict(item.get("info")).get("imr"),
                0.0,
        )
        margin = direct_margin if direct_margin > 0 else notional / leverage if notional > 0 else 0.0
        adverse_stop = (
            max(stop - mark, 0.0) / mark
            if side == "short" and mark > 0
            else max(mark - stop, 0.0) / mark
            if side == "long" and mark > 0
            else 0.0
        )
        if notional <= 0 or mark <= 0:
            blockers.append("open_position_native_valuation_incomplete")
        if adverse_stop <= 0:
            blockers.append("open_position_stress_stop_incomplete")
        stressed_loss = notional * adverse_stop
        gross_notional += notional
        current_margin += margin
        current_stressed_loss += stressed_loss
        if side == candidate_side:
            same_side_notional += notional
        rows.append(
            {
                **valuation,
                "symbol": normalize_trading_symbol(item.get("symbol")),
                "side": side,
                "margin_mode": item.get("marginMode") or _safe_dict(item.get("info")).get("mgnMode"),
                "leverage": leverage,
                "stop_price": stop,
                "stress_loss_fraction": adverse_stop,
                "stressed_loss_usdt": stressed_loss,
                "margin_usdt": margin,
            }
        )
    concentration = same_side_notional / gross_notional if gross_notional > 0 else 0.0
    return {
        "current_stressed_loss_usdt": current_stressed_loss,
        "current_margin_usdt": current_margin,
        "gross_notional_usdt": gross_notional,
        "same_side_notional_usdt": same_side_notional,
        "direction_concentration": concentration,
        "positions": rows,
    }, list(dict.fromkeys(blockers))


def build_portfolio_risk_snapshot(
    positions: list[dict[str, Any]],
    *,
    candidate_side: str,
    contract_specs: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Expose the shared native portfolio valuation contract to bounded entry policies."""

    return _portfolio_risk_snapshot(
        positions,
        candidate_side=candidate_side,
        contract_specs=contract_specs,
    )


def _correlation_pressure(
    decision: DecisionOutput,
    positions: list[dict[str, Any]],
) -> tuple[float, str]:
    if not positions:
        return 0.0, "empty_portfolio"
    strategy = _safe_dict(_safe_dict(decision.raw_response).get("strategy_mode"))
    context = _safe_dict(strategy.get("portfolio_correlation"))
    key = f"{normalize_trading_symbol(decision.symbol)}|{_side(decision)}"
    row = _safe_dict(context.get(key))
    value = _safe_float(row.get("weighted_adverse_correlation"), float("nan"))
    if isfinite(value):
        return _clamp(value), "current_feature_return_correlation"
    return 0.0, "missing"


def _reconciliation_history(sizing: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in _safe_list(sizing.get("execution_reconciliations")) if isinstance(item, dict)]


def reconcile_profit_risk_sizing(
    decision: DecisionOutput,
    *,
    final_notional_usdt: float,
    final_leverage: float,
    source: str,
    execution_facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild the authoritative contract for a smaller executable notional."""

    raw = _safe_dict(decision.raw_response)
    sizing = dict(_safe_dict(raw.get("profit_risk_sizing")))
    original_target = max(_safe_float(sizing.get("target_notional_usdt"), 0.0), 0.0)
    fill_notional_ceiling = max(
        _safe_float(sizing.get("fill_notional_ceiling_usdt"), 0.0),
        0.0,
    )
    risk_budget = max(_safe_float(sizing.get("risk_budget_usdt"), 0.0), 0.0)
    stress = max(_safe_float(sizing.get("stressed_loss_fraction"), 0.0), 0.0)
    margin_basis = max(_safe_float(sizing.get("available_margin_usdt"), 0.0), 0.0)
    expected_net = max(_safe_float(sizing.get("expected_net_return_pct"), 0.0), 0.0)
    leverage_tier = _safe_dict(sizing.get("leverage_tier_selection"))
    tier_max_leverage = max(_safe_float(leverage_tier.get("max_leverage"), 0.0), 0.0)
    notional = max(_safe_float(final_notional_usdt, 0.0), 0.0)
    leverage = max(_safe_float(final_leverage, 0.0), 0.0)
    reasons: list[str] = []
    if sizing.get("production_eligible") is not True:
        reasons.append("upstream_sizing_ineligible")
    if notional <= 0 or leverage < 1 or stress <= 0 or margin_basis <= 0:
        reasons.append("execution_reconciliation_inputs_incomplete")
    if leverage_tier.get("production_eligible") is not True or tier_max_leverage < 1:
        reasons.append("execution_leverage_tier_contract_missing")
    elif leverage > tier_max_leverage + 1e-8:
        reasons.append("execution_leverage_exceeds_selected_okx_tier")
    notional_ceiling = (
        fill_notional_ceiling
        if source == "okx_confirmed_entry_fill" and fill_notional_ceiling > 0
        else original_target
    )
    if original_target <= 0 or notional > notional_ceiling + 1e-8:
        reasons.append("execution_notional_exceeds_authoritative_target")
    planned_loss = notional * stress
    if risk_budget <= 0 or planned_loss > risk_budget + 1e-8:
        reasons.append("execution_stressed_loss_exceeds_risk_budget")
    eligible = not reasons
    generated_at = datetime.now(UTC).isoformat()
    facts = dict(execution_facts or {})
    history = _reconciliation_history(sizing)
    history.append(
        {
            "source": source,
            "generated_at": generated_at,
            "final_notional_usdt": round(notional, 8),
            "final_leverage": round(leverage, 8),
            "facts": facts,
            "facts_fingerprint": _fingerprint(facts),
            "eligible": eligible,
            "reasons": reasons,
        }
    )
    position_size = notional / max(margin_basis * leverage, 1e-12) if eligible else 0.0
    sizing.update(
        {
            "production_eligible": eligible,
            "reason": "execution_notional_reconciled" if eligible else ",".join(reasons),
            "position_size_pct": round(position_size, 8),
            "final_notional_usdt": round(notional, 8),
            "final_margin_usdt": round(notional / leverage, 8) if leverage > 0 else 0.0,
            "final_leverage": round(leverage, 8),
            "planned_stressed_loss_usdt": round(planned_loss, 8),
            "expected_profit_usdt": round(notional * expected_net / 100.0, 8),
            "execution_reconciliations": history,
        }
    )
    provenance = dict(_safe_dict(sizing.get("policy_provenance")))
    strategy_version = str(
        sizing.get("contract_version")
        or provenance.get("strategy_version")
        or RISK_SIZING_VERSION
    )
    provenance.update(
        {
            "generated_at": generated_at,
            "strategy_version": strategy_version,
            "fallback_reason": "" if eligible else ",".join(reasons),
            "contract_fingerprint": _fingerprint(
                {
                    "risk_budget_usdt": sizing.get("risk_budget_usdt"),
                    "stressed_loss_fraction": sizing.get("stressed_loss_fraction"),
                    "final_notional_usdt": sizing.get("final_notional_usdt"),
                    "final_leverage": leverage,
                    "leverage_tier_input_fingerprint": _safe_dict(
                        leverage_tier.get("policy_provenance")
                    ).get("input_fingerprint"),
                    "execution_reconciliations": history,
                }
            ),
        }
    )
    sizing["policy_provenance"] = provenance
    raw["profit_risk_sizing"] = sizing
    return_policy = raw.get("production_return_policy")
    if isinstance(return_policy, dict):
        return_policy = dict(return_policy)
        return_policy["position_size_pct"] = round(position_size, 8)
        raw["production_return_policy"] = return_policy
    decision.raw_response = raw
    decision.position_size_pct = position_size if eligible else 0.0
    decision.suggested_leverage = leverage if eligible else 1.0
    return {"eligible": eligible, "reasons": reasons, "sizing": sizing}


@dataclass(slots=True)
class EntryProfitRiskSizingPolicy:
    """Generate the single production risk budget and position contract."""

    evaluator: EntryProfitRiskSizingEvaluator | None = None
    allocated_order_balance: EntryBalanceProvider | None = None
    exchange_risk_facts: EntryExchangeRiskFactsProvider | None = None
    dynamic_leverage_allocator: Any | None = None

    async def apply(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]] | None = None,
    ) -> None:
        if self.evaluator is not None:
            await self.evaluator(decision, model_mode, open_positions or [])
            return
        if not decision.is_entry:
            return
        if self.allocated_order_balance is None:
            raise RuntimeError("EntryProfitRiskSizingPolicy requires allocated_order_balance")

        raw = _safe_dict(decision.raw_response)
        paper_exploration = is_paper_exploration_decision(decision)
        exploration_selection_reasons = (
            paper_exploration_selection_reasons(decision, model_mode)
            if paper_exploration
            else []
        )
        opportunity = _safe_dict(raw.get("opportunity_score"))
        distribution = _safe_dict(opportunity.get("return_distribution_contract"))
        execution_cost = _safe_dict(opportunity.get("execution_cost"))
        strategy = _safe_dict(raw.get("strategy_mode"))
        positions = open_positions or []
        facts = (
            await self.exchange_risk_facts(model_mode, decision, positions)
            if self.exchange_risk_facts is not None
            else _safe_dict(raw.get("exchange_risk_facts"))
        )
        allocated_margin = max(
            _safe_float(await self.allocated_order_balance(model_mode, decision), 0.0),
            0.0,
        )
        fact_available_margin = max(
            _safe_float(facts.get("available_margin_usdt"), 0.0),
            0.0,
        )
        available_margin = (
            min(allocated_margin, fact_available_margin)
            if allocated_margin > 0 and fact_available_margin > 0
            else fact_available_margin
        )
        account_equity = max(_safe_float(facts.get("account_equity_usdt"), 0.0), 0.0)
        expected_net = _safe_float(distribution.get("raw_expected_return_pct"), float("nan"))
        if not isfinite(expected_net):
            expected_net = _safe_float(opportunity.get("expected_net_return_pct"), float("nan"))
        return_lcb = _safe_float(distribution.get("objective_expected_return_pct"), float("nan"))
        if not isfinite(return_lcb):
            return_lcb = _safe_float(opportunity.get("return_lcb_pct"), float("nan"))
        uncertainty_pct = max(_safe_float(distribution.get("uncertainty_penalty_pct"), 0.0), 0.0)
        expected_loss_pct = max(_safe_float(distribution.get("tail_loss_penalty_pct"), 0.0), 0.0)
        if expected_loss_pct <= 0:
            expected_loss_pct = max(_safe_float(opportunity.get("expected_loss_pct"), 0.0), 0.0)
        loss_probability = _clamp(_safe_float(distribution.get("tail_loss_probability"), 1.0))
        tail_risk = _clamp(_safe_float(opportunity.get("tail_risk_score"), loss_probability))
        profit_quality = max(_safe_float(opportunity.get("profit_quality_ratio"), 0.0), 0.0)
        source_count = _production_source_count(opportunity)
        calibration = _actual_trade_calibration(opportunity)
        rolling_realized_lcb = _rolling_realized_lcb_pct(calibration)

        snapshot = _safe_dict(decision.feature_snapshot)
        declared_stop = _normalized_ratio(decision.stop_loss_pct)
        declared_take_profit = _normalized_ratio(decision.take_profit_pct)
        atr_ratio = _atr_ratio(decision)
        path_adverse = _path_adverse_fraction(decision)
        volatility = _normalized_ratio(snapshot.get("volatility_20"))
        wick = max(_safe_float(snapshot.get("abnormal_wick_max_pct"), 0.0) / 100.0, 0.0)
        stop_slippage_tail = _distribution_tail(calibration.get("stop_loss_slippage_pct")) / 100.0
        general_slippage_tail = _distribution_tail(calibration.get("slippage_pct")) / 100.0
        market_impact = max(
            _safe_float(execution_cost.get("slippage_pct"), 0.0),
            _safe_float(execution_cost.get("liquidity_penalty_pct"), 0.0),
            _safe_float(execution_cost.get("imbalance_penalty_pct"), 0.0),
            _safe_float(execution_cost.get("market_impact_pct"), 0.0),
            0.0,
        ) / 100.0
        stressed_loss_fraction = max(
            declared_stop,
            atr_ratio,
            path_adverse,
            volatility,
            wick,
            expected_loss_pct / 100.0,
            stop_slippage_tail,
            general_slippage_tail,
            market_impact,
        )
        cost_pct = max(_safe_float(execution_cost.get("total_pct"), 0.0), 0.0)
        positive_return = max(expected_net, 0.0) if isfinite(expected_net) else 0.0
        positive_lcb = max(return_lcb, 0.0) if isfinite(return_lcb) else 0.0
        dynamic_take_profit = max(declared_take_profit, (positive_return + cost_pct) / 100.0)

        contract_specs = _safe_dict(facts.get("contract_specs"))
        portfolio, portfolio_blockers = _portfolio_risk_snapshot(
            positions,
            candidate_side=_side(decision),
            contract_specs=contract_specs,
        )
        correlation_pressure, correlation_source = _correlation_pressure(decision, positions)
        dependency_capacity = 1.0 / (
            1.0 + portfolio["direction_concentration"] + correlation_pressure
        )
        drawdown_pressure = _clamp(_safe_float(strategy.get("drawdown_pressure"), 0.0))
        drawdown_capacity = 1.0 - drawdown_pressure
        return_quality_basis = positive_return if paper_exploration else positive_lcb
        return_quality = return_quality_basis / max(
            return_quality_basis + uncertainty_pct + expected_loss_pct + cost_pct,
            1e-12,
        )
        survival_quality = (1.0 - loss_probability) * (1.0 - tail_risk)
        realized_downside = max(-(rolling_realized_lcb or 0.0), 0.0) / 100.0
        realized_return_basis = positive_return if paper_exploration else positive_lcb
        realized_history_capacity = (realized_return_basis / 100.0) / max(
            realized_return_basis / 100.0 + realized_downside,
            1e-12,
        )
        side_depth = max(
            _safe_float(
                snapshot.get(
                    "orderbook_ask_depth" if _side(decision) == "long" else "orderbook_bid_depth"
                ),
                0.0,
            ),
            0.0,
        )
        liquidity_budget_share = _clamp(side_depth / max(account_equity, 1e-12))
        single_trade_budget_fraction = _clamp(
            return_quality
            * survival_quality
            * drawdown_capacity
            * realized_history_capacity
            * liquidity_budget_share
        )
        portfolio_budget_fraction = _clamp(single_trade_budget_fraction * dependency_capacity)
        if paper_exploration:
            single_trade_budget_fraction = min(
                single_trade_budget_fraction,
                PAPER_EXPLORATION_MAX_SINGLE_TRADE_RISK_FRACTION,
            )
            portfolio_budget_fraction = min(
                portfolio_budget_fraction,
                PAPER_EXPLORATION_MAX_PORTFOLIO_RISK_FRACTION,
            )
        single_trade_budget = account_equity * single_trade_budget_fraction
        portfolio_risk_budget = account_equity * portfolio_budget_fraction
        remaining_portfolio_budget = max(
            portfolio_risk_budget - portfolio["current_stressed_loss_usdt"],
            0.0,
        )
        risk_budget = min(single_trade_budget, remaining_portfolio_budget)
        target_notional = (
            risk_budget / stressed_loss_fraction if stressed_loss_fraction > 0 else 0.0
        )
        target_inst_id = str(
            facts.get("target_inst_id") or okx_inst_id_from_symbol(decision.symbol)
        ).upper()
        target_contract_spec = _safe_dict(contract_specs.get(target_inst_id))
        existing_exposure = _candidate_existing_exposure(decision, positions, contract_specs)
        target_price = max(
            _safe_float(snapshot.get("current_price", snapshot.get("close")), 0.0),
            0.0,
        )
        leverage_tier_selection = select_okx_leverage_tier(
            facts.get("leverage_tiers"),
            target_notional_usdt=target_notional,
            mark_price=target_price,
            contract_spec=target_contract_spec,
            current_position_notional_usdt=existing_exposure["notional_usdt"],
            current_position_contracts=existing_exposure["contracts"],
        )
        system_max_leverage = max(
            _safe_float(leverage_tier_selection.get("max_leverage"), 0.0),
            0.0,
        )

        reasons: list[str] = []
        reasons.extend(exploration_selection_reasons)
        if facts.get("production_eligible") is not True:
            reasons.append("exchange_risk_facts_ineligible")
        if leverage_tier_selection.get("production_eligible") is not True:
            reasons.append(
                str(leverage_tier_selection.get("reason") or "okx_leverage_tier_ineligible")
            )
        if account_equity <= 0:
            reasons.append("account_equity_missing")
        if available_margin <= 0:
            reasons.append("available_margin_missing")
        if system_max_leverage < 1:
            reasons.append("exchange_system_max_leverage_missing")
        if source_count <= 0:
            reasons.append("production_return_observations_missing")
        if not isfinite(expected_net) or expected_net <= 0:
            reasons.append("fee_after_expected_return_not_positive")
        if paper_exploration and (not isfinite(return_lcb) or return_lcb > 0):
            reasons.append("paper_exploration_return_lcb_not_uncertain")
        elif (
            paper_exploration
            and isfinite(expected_net)
            and expected_net > 0
            and max(-return_lcb, 0.0) / expected_net
            > PAPER_EXPLORATION_MAX_LCB_GAP_RATIO
        ):
            reasons.append("paper_exploration_not_close_to_profitable_threshold")
        elif not paper_exploration and (not isfinite(return_lcb) or return_lcb <= 0):
            reasons.append("fee_after_return_lcb_not_positive")
        if rolling_realized_lcb is None:
            reasons.append("rolling_authoritative_return_distribution_missing")
        if execution_cost.get("production_eligible") is not True or cost_pct <= 0:
            reasons.append("production_execution_cost_missing")
        if side_depth <= 0:
            reasons.append("live_orderbook_depth_missing")
        if stressed_loss_fraction <= 0:
            reasons.append("stressed_loss_fraction_missing")
        if correlation_source == "missing":
            reasons.append("portfolio_correlation_missing")
        reasons.extend(portfolio_blockers)
        if risk_budget <= 0 or target_notional <= 0:
            reasons.append("independent_risk_budget_zero")

        eligible = not reasons
        allocator = self.dynamic_leverage_allocator or DynamicLeverageAllocator()
        leverage_decision = allocator.allocate(
            DynamicLeverageInput(
                symbol=decision.symbol,
                requested_leverage=_safe_float(decision.suggested_leverage, 1.0),
                system_max_leverage=system_max_leverage,
                target_notional_usdt=target_notional,
                available_margin_usdt=available_margin,
                stressed_loss_fraction=stressed_loss_fraction,
                expected_net_return_pct=positive_return,
                return_lcb_pct=positive_lcb,
                expected_loss_pct=expected_loss_pct,
                profit_quality_ratio=profit_quality,
                loss_probability=loss_probability,
                tail_risk_score=tail_risk,
                aligned_source_count=source_count,
                atr_pct=atr_ratio,
                execution_cost=execution_cost,
                portfolio_capacity_fraction=dependency_capacity,
            )
        )
        if leverage_decision.policy_provenance.get("production_eligible") is not True:
            reasons.extend(leverage_decision.reasons)
            eligible = False
        leverage = (
            1.0
            if paper_exploration
            else float(leverage_decision.final_integer_leverage)
            if eligible
            else 1.0
        )
        final_notional = (
            min(target_notional, side_depth, available_margin * leverage) if eligible else 0.0
        )
        final_margin = final_notional / leverage if leverage > 0 else 0.0
        final_size = final_margin / available_margin if available_margin > 0 else 0.0
        planned_loss = final_notional * stressed_loss_fraction
        if final_notional <= 0 or planned_loss > risk_budget + 1e-8:
            reasons.append("final_notional_exceeds_independent_risk_contract")
            eligible = False
            final_notional = final_margin = final_size = planned_loss = 0.0

        generated_at = datetime.now(UTC).isoformat()
        audit_inputs = {
            "account_equity_usdt": account_equity,
            "available_margin_usdt": available_margin,
            "drawdown_pressure": drawdown_pressure,
            "expected_net_return_pct": expected_net,
            "return_lcb_pct": return_lcb,
            "rolling_realized_return_lcb_pct": rolling_realized_lcb,
            "return_uncertainty_pct": uncertainty_pct,
            "expected_loss_pct": expected_loss_pct,
            "loss_probability": loss_probability,
            "tail_risk_score": tail_risk,
            "liquidity_capacity_usdt": side_depth,
            "direction_concentration": portfolio["direction_concentration"],
            "correlation_pressure": correlation_pressure,
            "current_portfolio_stressed_loss_usdt": portfolio["current_stressed_loss_usdt"],
            "system_max_leverage": system_max_leverage,
            "leverage_tier_input_fingerprint": _safe_dict(
                leverage_tier_selection.get("policy_provenance")
            ).get("input_fingerprint"),
        }
        provenance = {
            "source": (
                "bounded_paper_exploration_return_and_account_risk_budget"
                if paper_exploration
                else "independent_return_drawdown_liquidity_tail_and_portfolio_risk_budget"
            ),
            "observation_window": "current_decision_account_and_okx_native_portfolio",
            "sample_count": source_count,
            "generated_at": generated_at,
            "strategy_version": (
                PAPER_EXPLORATION_SIZING_VERSION
                if paper_exploration
                else RISK_SIZING_VERSION
            ),
            "fallback_reason": "" if eligible else ",".join(dict.fromkeys(reasons)),
            "input_fingerprint": _fingerprint(audit_inputs),
        }
        sizing = {
            "contract_version": (
                PAPER_EXPLORATION_SIZING_VERSION
                if paper_exploration
                else RISK_SIZING_VERSION
            ),
            "contract_lifecycle": (
                "paper_exploration" if paper_exploration else "production_return"
            ),
            "execution_scope": "paper_only" if paper_exploration else "mode_authoritative",
            "production_permission": False if paper_exploration else None,
            "production_eligible": eligible,
            "reason": "independent_dynamic_risk_budget_ready" if eligible else provenance["fallback_reason"],
            "account_equity_usdt": round(account_equity, 8),
            "available_margin_usdt": round(available_margin, 8),
            "position_size_pct": round(final_size, 8),
            "risk_budget_usdt": round(risk_budget, 8),
            "single_trade_risk_budget_usdt": round(single_trade_budget, 8),
            "portfolio_risk_budget_usdt": round(portfolio_risk_budget, 8),
            "remaining_portfolio_risk_budget_usdt": round(remaining_portfolio_budget, 8),
            "current_portfolio_stressed_loss_usdt": round(
                portfolio["current_stressed_loss_usdt"], 8
            ),
            "planned_stressed_loss_usdt": round(planned_loss, 8),
            "target_notional_usdt": round(target_notional, 8),
            "final_notional_usdt": round(final_notional, 8),
            "final_margin_usdt": round(final_margin, 8),
            "final_leverage": round(leverage, 8),
            "expected_net_return_pct": round(positive_return, 8),
            "expected_profit_usdt": round(final_notional * positive_return / 100.0, 8),
            "declared_stop_loss_fraction": round(declared_stop, 8),
            "stressed_loss_fraction": round(stressed_loss_fraction, 8),
            "declared_take_profit_fraction": round(declared_take_profit, 8),
            "dynamic_take_profit_fraction": round(dynamic_take_profit, 8),
            "stress_components": {
                "planned_stop_fraction": round(declared_stop, 8),
                "atr_fraction": round(atr_ratio, 8),
                "path_adverse_fraction": round(path_adverse, 8),
                "volatility_fraction": round(volatility, 8),
                "abnormal_wick_fraction": round(wick, 8),
                "expected_tail_loss_fraction": round(expected_loss_pct / 100.0, 8),
                "actual_stop_slippage_tail_fraction": round(stop_slippage_tail, 8),
                "actual_execution_slippage_tail_fraction": round(general_slippage_tail, 8),
                "current_orderbook_impact_fraction": round(market_impact, 8),
            },
            "budget_factors": {
                "return_quality": round(return_quality, 8),
                "survival_quality": round(survival_quality, 8),
                "drawdown_capacity": round(drawdown_capacity, 8),
                "realized_history_capacity": round(realized_history_capacity, 8),
                "liquidity_budget_share": round(liquidity_budget_share, 8),
                "portfolio_dependency_capacity": round(dependency_capacity, 8),
            },
            "paper_exploration_risk_caps": (
                {
                    "single_trade_equity_fraction": (
                        PAPER_EXPLORATION_MAX_SINGLE_TRADE_RISK_FRACTION
                    ),
                    "portfolio_equity_fraction": (
                        PAPER_EXPLORATION_MAX_PORTFOLIO_RISK_FRACTION
                    ),
                    "leverage_cap": 1,
                    "sample_target": None,
                    "daily_sample_quota": None,
                }
                if paper_exploration
                else {}
            ),
            "portfolio_risk_snapshot": portfolio,
            "exchange_contract_specs": contract_specs,
            "exchange_risk_facts_provenance": facts.get("policy_provenance"),
            "leverage_tier_selection": leverage_tier_selection,
            "dynamic_leverage_decision": leverage_decision.to_dict(),
            "audit_inputs": audit_inputs,
            "units": {
                "money": "USDT",
                "returns": "percentage_points",
                "fractions": "decimal_ratio",
                "position_size_pct": "available_margin_fraction",
                "notional": "USDT",
            },
            "policy_provenance": provenance,
        }
        sizing["policy_provenance"]["contract_fingerprint"] = _fingerprint(sizing)
        raw["profit_risk_sizing"] = sizing
        raw["dynamic_leverage_decision"] = leverage_decision.to_dict()
        decision.raw_response = raw
        decision.position_size_pct = final_size if eligible else 0.0
        decision.suggested_leverage = leverage if eligible else 1.0
        decision.stop_loss_pct = stressed_loss_fraction if eligible else decision.stop_loss_pct
        decision.take_profit_pct = dynamic_take_profit if eligible else decision.take_profit_pct


def build_portfolio_correlation_context(
    feature_vectors: dict[str, Any],
    positions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build current return correlations for each candidate against open exposure."""

    def sequence(value: Any) -> list[float]:
        raw = value.to_dict() if hasattr(value, "to_dict") else _safe_dict(value)
        prices = [
            _safe_float(item, float("nan")) for item in _safe_list(raw.get("close_sequence"))
        ]
        usable = [item for item in prices if isfinite(item) and item > 0]
        return [usable[index] / usable[index - 1] - 1.0 for index in range(1, len(usable))]

    def correlation(left: list[float], right: list[float]) -> float | None:
        count = min(len(left), len(right))
        if count < 2:
            return None
        x = left[-count:]
        y = right[-count:]
        x_mean = sum(x) / count
        y_mean = sum(y) / count
        covariance = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
        x_scale = sqrt(sum((a - x_mean) ** 2 for a in x))
        y_scale = sqrt(sum((b - y_mean) ** 2 for b in y))
        if x_scale <= 0 or y_scale <= 0:
            return None
        return min(max(covariance / (x_scale * y_scale), -1.0), 1.0)

    features = {normalize_trading_symbol(key): value for key, value in feature_vectors.items()}
    result: dict[str, Any] = {}
    for candidate_symbol, candidate_feature in features.items():
        candidate_returns = sequence(candidate_feature)
        for candidate_side, candidate_sign in (("long", 1.0), ("short", -1.0)):
            weighted = 0.0
            total_weight = 0.0
            rows: list[dict[str, Any]] = []
            for position in positions:
                position_symbol = normalize_trading_symbol(position.get("symbol"))
                position_feature = features.get(position_symbol)
                corr = (
                    correlation(candidate_returns, sequence(position_feature))
                    if position_feature
                    else None
                )
                notional, _valuation = _position_notional(_safe_dict(position), {})
                position_sign = (
                    1.0 if str(position.get("side") or "").lower() == "long" else -1.0
                )
                if corr is None or notional <= 0:
                    continue
                adverse = max(corr * candidate_sign * position_sign, 0.0)
                weighted += adverse * notional
                total_weight += notional
                rows.append(
                    {
                        "symbol": position_symbol,
                        "price_return_correlation": round(corr, 8),
                        "exposure_adverse_correlation": round(adverse, 8),
                        "notional_weight_usdt": round(notional, 8),
                    }
                )
            result[f"{candidate_symbol}|{candidate_side}"] = {
                "weighted_adverse_correlation": round(weighted / total_weight, 8)
                if total_weight > 0
                else None,
                "matched_position_count": len(rows),
                "positions": rows,
            }
    return result
