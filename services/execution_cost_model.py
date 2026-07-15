"""Execution-cost estimation used by entry scoring.

The configured maximum slippage is a safety ceiling, not the ordinary cost of
every order. Entry scoring estimates realistic executable cost from current
market microstructure and only uses the configured max as a cap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

EXECUTION_COST_FACT_FIELDS = (
    "taker_fee_rate",
    "entry_fee_rate",
    "exit_fee_rate",
    "round_trip_fee_pct",
    "fee_rate_source",
    "fee_rate_observed_at",
)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def attach_execution_cost_facts(feature_vector: Any, facts: dict[str, Any] | None) -> Any:
    """Attach account fee facts to one live market snapshot without guessing values."""

    payload = facts if isinstance(facts, dict) else {}
    for key in EXECUTION_COST_FACT_FIELDS:
        value = payload.get(key)
        if value in {None, ""}:
            continue
        if isinstance(feature_vector, dict):
            feature_vector[key] = value
        else:
            setattr(feature_vector, key, value)
    provenance = payload.get("policy_provenance")
    if isinstance(provenance, dict):
        if isinstance(feature_vector, dict):
            feature_vector["fee_policy_provenance"] = dict(provenance)
        else:
            feature_vector.fee_policy_provenance = dict(provenance)
    return feature_vector


@dataclass(frozen=True, slots=True)
class ExecutionCostEstimate:
    """Round-trip fee and dynamic slippage estimate in percentage points."""

    fee_pct: float
    slippage_pct: float
    total_pct: float
    spread_pct: float
    spread_source: str
    bid_depth_usdt: float
    ask_depth_usdt: float
    orderbook_imbalance: float
    liquidity_penalty_pct: float
    imbalance_penalty_pct: float
    market_impact_pct: float
    order_notional_usdt: float
    order_side: str
    order_size_complete: bool
    observed_side_depth_usdt: float
    depth_consumption_fraction: float
    estimated_vwap: float
    reference_price: float
    book_levels_consumed: int
    slippage_source: str
    production_eligible: bool
    reason: str
    policy_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def execution_cost_estimate(feature_snapshot: dict[str, Any] | None) -> ExecutionCostEstimate:
    """Estimate round-trip execution costs from the current market snapshot."""

    snapshot = feature_snapshot if isinstance(feature_snapshot, dict) else {}
    spread_pct, spread_source = _normalized_spread_pct(snapshot)
    bid_depth = max(_safe_float(snapshot.get("orderbook_bid_depth"), 0.0), 0.0)
    ask_depth = max(_safe_float(snapshot.get("orderbook_ask_depth"), 0.0), 0.0)
    imbalance = min(max(_safe_float(snapshot.get("orderbook_imbalance"), 0.0), -1.0), 1.0)

    half_spread_pct = max(spread_pct / 2.0, 0.0)
    depth_asymmetry = (
        abs(bid_depth - ask_depth) / (bid_depth + ask_depth)
        if bid_depth > 0 and ask_depth > 0
        else 0.0
    )
    liquidity_penalty_pct = half_spread_pct * depth_asymmetry
    imbalance_penalty_pct = half_spread_pct * abs(imbalance)
    impact = _orderbook_market_impact(snapshot)
    market_impact_pct = max(_safe_float(impact.get("market_impact_pct"), 0.0), 0.0)
    order_size_slippage_pct = max(
        _safe_float(impact.get("execution_slippage_pct"), 0.0),
        0.0,
    )
    raw_slippage_pct = (
        order_size_slippage_pct
        if impact.get("eligible") is True
        else half_spread_pct + liquidity_penalty_pct + imbalance_penalty_pct
    )
    slippage_pct = raw_slippage_pct
    fee_pct, fee_source = round_trip_fee_pct(snapshot)
    order_notional = max(_safe_float(snapshot.get("planned_order_notional_usdt"), 0.0), 0.0)
    order_size_complete = order_notional > 0 and impact.get("eligible") is True
    base_eligible = bool(spread_source != "missing" and spread_pct > 0 and fee_pct > 0)
    production_eligible = bool(
        base_eligible and (order_notional <= 0 or order_size_complete)
    )
    failure_reason = (
        "fee_rate_missing"
        if fee_pct <= 0
        else "live_spread_missing"
        if spread_source == "missing" or spread_pct <= 0
        else str(impact.get("reason") or "order_size_market_impact_incomplete")
        if order_notional > 0 and not order_size_complete
        else ""
    )
    slippage_pct = round(slippage_pct, 6)
    return ExecutionCostEstimate(
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
        total_pct=fee_pct + slippage_pct,
        spread_pct=round(spread_pct, 6),
        spread_source=spread_source,
        bid_depth_usdt=round(bid_depth, 6),
        ask_depth_usdt=round(ask_depth, 6),
        orderbook_imbalance=round(imbalance, 6),
        liquidity_penalty_pct=round(liquidity_penalty_pct, 6),
        imbalance_penalty_pct=round(imbalance_penalty_pct, 6),
        market_impact_pct=round(market_impact_pct, 6),
        order_notional_usdt=round(order_notional, 8),
        order_side=str(snapshot.get("planned_order_side") or "").lower(),
        order_size_complete=order_size_complete,
        observed_side_depth_usdt=round(
            max(_safe_float(impact.get("observed_side_depth_usdt"), 0.0), 0.0),
            8,
        ),
        depth_consumption_fraction=round(
            max(_safe_float(impact.get("depth_consumption_fraction"), 0.0), 0.0),
            8,
        ),
        estimated_vwap=round(max(_safe_float(impact.get("estimated_vwap"), 0.0), 0.0), 12),
        reference_price=round(
            max(_safe_float(impact.get("reference_price"), 0.0), 0.0),
            12,
        ),
        book_levels_consumed=max(int(_safe_float(impact.get("book_levels_consumed"), 0.0)), 0),
        slippage_source=(
            "dynamic_live_spread_depth_imbalance_and_order_vwap"
            if order_size_complete and production_eligible
            else "pre_sizing_live_spread_depth_imbalance"
            if production_eligible
            else "observation_only_missing_live_spread"
        ),
        production_eligible=production_eligible,
        reason=(
            "live_order_size_microstructure_cost_ready"
            if order_size_complete and production_eligible
            else "pre_sizing_microstructure_cost_ready"
            if production_eligible
            else failure_reason
        ),
        policy_provenance={
            "source": f"{fee_source}+live_bid_ask_depth_imbalance_and_order_vwap",
            "observation_window": "current_pre_order_native_orderbook_snapshot",
            "sample_count": (
                int(spread_source != "missing")
                + int(bid_depth > 0 and ask_depth > 0)
                + max(int(_safe_float(impact.get("book_levels_consumed"), 0.0)), 0)
            ),
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy_version": "2026-07-15.order-size-execution-cost.v2",
            "fallback_reason": "" if production_eligible else failure_reason,
            "orderbook_fingerprint": str(
                _safe_dict(snapshot.get("pre_order_execution_facts")).get(
                    "input_fingerprint"
                )
                or ""
            ),
        },
    )


def _orderbook_market_impact(snapshot: dict[str, Any]) -> dict[str, Any]:
    order_notional = max(_safe_float(snapshot.get("planned_order_notional_usdt"), 0.0), 0.0)
    side = str(snapshot.get("planned_order_side") or "").lower()
    if order_notional <= 0:
        return {"eligible": False, "reason": "planned_order_notional_missing"}
    if side not in {"long", "short"}:
        return {"eligible": False, "reason": "planned_order_side_missing"}

    bid = max(_safe_float(snapshot.get("bid"), 0.0), 0.0)
    ask = max(_safe_float(snapshot.get("ask"), 0.0), 0.0)
    reference = (bid + ask) / 2.0 if bid > 0 and ask >= bid else 0.0
    contract_base = max(
        _safe_float(snapshot.get("contract_value_base"), 0.0),
        0.0,
    )
    levels = snapshot.get("orderbook_asks" if side == "long" else "orderbook_bids")
    levels = levels if isinstance(levels, list) else []
    if reference <= 0:
        return {"eligible": False, "reason": "orderbook_reference_price_missing"}
    if contract_base <= 0:
        return {"eligible": False, "reason": "contract_value_base_missing"}
    if not levels:
        return {"eligible": False, "reason": "executable_orderbook_levels_missing"}

    target_base = order_notional / reference
    remaining_base = target_base
    filled_base = 0.0
    quote_cost = 0.0
    observed_depth = 0.0
    consumed = 0
    best_price = 0.0
    for level in levels:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue
        price = max(_safe_float(level[0], 0.0), 0.0)
        contracts = max(_safe_float(level[1], 0.0), 0.0)
        level_base = contracts * contract_base
        if price <= 0 or level_base <= 0:
            continue
        if best_price <= 0:
            best_price = price
        observed_depth += level_base * price
        if remaining_base <= 0:
            continue
        used_base = min(remaining_base, level_base)
        filled_base += used_base
        quote_cost += used_base * price
        remaining_base -= used_base
        consumed += 1

    if filled_base <= 0 or remaining_base > 1e-12:
        return {
            "eligible": False,
            "reason": "planned_order_exceeds_observed_orderbook_depth",
            "observed_side_depth_usdt": observed_depth,
            "depth_consumption_fraction": (
                order_notional / observed_depth if observed_depth > 0 else 0.0
            ),
            "book_levels_consumed": consumed,
            "reference_price": reference,
        }

    vwap = quote_cost / filled_base
    adverse_impact = (
        max(vwap - best_price, 0.0)
        if side == "long"
        else max(best_price - vwap, 0.0)
    )
    return {
        "eligible": True,
        "reason": "order_size_vwap_ready",
        "market_impact_pct": adverse_impact / reference * 100.0,
        "execution_slippage_pct": (
            max(vwap - reference, 0.0) / reference * 100.0
            if side == "long"
            else max(reference - vwap, 0.0) / reference * 100.0
        ),
        "observed_side_depth_usdt": observed_depth,
        "depth_consumption_fraction": (
            order_notional / observed_depth if observed_depth > 0 else 0.0
        ),
        "estimated_vwap": vwap,
        "reference_price": reference,
        "book_levels_consumed": consumed,
    }


def round_trip_fee_pct(snapshot: dict[str, Any]) -> tuple[float, str]:
    explicit = _safe_float(snapshot.get("round_trip_fee_pct"), 0.0)
    if explicit > 0:
        return explicit, "snapshot_round_trip_fee_pct"
    entry_rate = _normalized_fee_rate(
        snapshot.get("entry_fee_rate", snapshot.get("taker_fee_rate", snapshot.get("fee_rate")))
    )
    exit_rate = _normalized_fee_rate(
        snapshot.get("exit_fee_rate", snapshot.get("taker_fee_rate", snapshot.get("fee_rate")))
    )
    if entry_rate <= 0 or exit_rate <= 0:
        return 0.0, "missing_exchange_fee_rate"
    return round((entry_rate + exit_rate) * 100.0, 6), "snapshot_exchange_fee_rates"


def _normalized_fee_rate(value: Any) -> float:
    # Exchange fee-rate fields use decimal ratios. Percent values must use
    # the explicit round_trip_fee_pct field instead of unit guessing here.
    return max(_safe_float(value, 0.0), 0.0)


def _normalized_spread_pct(snapshot: dict[str, Any]) -> tuple[float, str]:
    spread = _safe_float(snapshot.get("spread_pct"), 0.0)
    if spread > 0:
        return spread, "spread_pct"

    bid = _safe_float(snapshot.get("bid"), 0.0)
    ask = _safe_float(snapshot.get("ask"), 0.0)
    if bid > 0 and ask >= bid:
        mid = (bid + ask) / 2.0
        if mid > 0:
            return (ask - bid) / mid * 100.0, "bid_ask"
    return 0.0, "missing"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
