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
    raw_slippage_pct = half_spread_pct + liquidity_penalty_pct + imbalance_penalty_pct
    slippage_pct = raw_slippage_pct
    fee_pct, fee_source = round_trip_fee_pct(snapshot)
    production_eligible = bool(
        spread_source != "missing"
        and spread_pct > 0
        and fee_pct > 0
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
        slippage_source=(
            "dynamic_live_spread_depth_imbalance"
            if production_eligible
            else "observation_only_missing_live_spread"
        ),
        production_eligible=production_eligible,
        reason=(
            "live_microstructure_cost_ready"
            if production_eligible
            else "fee_rate_missing"
            if fee_pct <= 0
            else "live_spread_missing"
        ),
        policy_provenance={
            "source": f"{fee_source}+live_bid_ask_depth_and_orderbook_imbalance",
            "observation_window": "current_orderbook_snapshot",
            "sample_count": int(spread_source != "missing") + int(bid_depth > 0 and ask_depth > 0),
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy_version": "2026-07-12.dynamic-execution-cost.v1",
            "fallback_reason": (
                ""
                if production_eligible
                else "fee_rate_missing"
                if fee_pct <= 0
                else "live_spread_missing"
            ),
        },
    )


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
