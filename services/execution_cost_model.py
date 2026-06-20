"""Execution-cost estimation used by entry scoring.

The configured maximum slippage is a safety ceiling, not the ordinary cost of
every order. Entry scoring estimates realistic executable cost from current
market microstructure and only uses the configured max as a cap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from config.settings import settings
from services.trading_params import DEFAULT_TRADING_PARAMS, ESTIMATED_TAKER_FEE_PCT

_EXECUTION_COST_PARAMS = DEFAULT_TRADING_PARAMS.execution_cost
DEFAULT_MAX_SLIPPAGE_PCT = _EXECUTION_COST_PARAMS.default_max_slippage_pct
DEFAULT_PAPER_SLIPPAGE_PCT = _EXECUTION_COST_PARAMS.default_paper_slippage_pct
MIN_EXECUTION_SLIPPAGE_PCT = _EXECUTION_COST_PARAMS.min_execution_slippage_pct
LIQUIDITY_DEPTH_REFERENCE_USDT = _EXECUTION_COST_PARAMS.liquidity_depth_reference_usdt


@dataclass(frozen=True, slots=True)
class ExecutionCostEstimate:
    """Round-trip fee and dynamic slippage estimate in percentage points."""

    fee_pct: float
    slippage_pct: float
    total_pct: float
    configured_max_slippage_pct: float
    spread_pct: float
    spread_source: str
    bid_depth_usdt: float
    ask_depth_usdt: float
    orderbook_imbalance: float
    liquidity_penalty_pct: float
    imbalance_penalty_pct: float
    slippage_cap_pct: float
    slippage_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def execution_cost_estimate(feature_snapshot: dict[str, Any] | None) -> ExecutionCostEstimate:
    """Estimate round-trip execution costs for opportunity scoring.

    ``settings.max_slippage_pct`` is stored as a decimal ratio (0.005 = 0.5%).
    It is treated as an upper bound for estimated slippage, not as the default
    slippage for every candidate.
    """

    snapshot = feature_snapshot if isinstance(feature_snapshot, dict) else {}
    configured_max_slippage_pct = max(
        _safe_float(settings.max_slippage_pct, DEFAULT_MAX_SLIPPAGE_PCT) * 100.0,
        MIN_EXECUTION_SLIPPAGE_PCT,
    )
    spread_pct, spread_source = _normalized_spread_pct(snapshot)
    bid_depth = max(_safe_float(snapshot.get("orderbook_bid_depth"), 0.0), 0.0)
    ask_depth = max(_safe_float(snapshot.get("orderbook_ask_depth"), 0.0), 0.0)
    min_depth = min(bid_depth, ask_depth) if bid_depth > 0 and ask_depth > 0 else 0.0
    imbalance = min(max(_safe_float(snapshot.get("orderbook_imbalance"), 0.0), -1.0), 1.0)

    half_spread_pct = max(spread_pct * 0.50, 0.0)
    liquidity_penalty_pct = _liquidity_penalty_pct(min_depth)
    imbalance_penalty_pct = max(abs(imbalance) - 0.55, 0.0) * 0.06
    raw_slippage_pct = max(
        DEFAULT_PAPER_SLIPPAGE_PCT,
        MIN_EXECUTION_SLIPPAGE_PCT,
        half_spread_pct + liquidity_penalty_pct + imbalance_penalty_pct,
    )
    slippage_pct = min(raw_slippage_pct, configured_max_slippage_pct)
    cap_used = raw_slippage_pct > configured_max_slippage_pct
    fee_pct = round(ESTIMATED_TAKER_FEE_PCT * 2.0 * 100.0, 6)
    slippage_pct = round(slippage_pct, 6)
    return ExecutionCostEstimate(
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
        total_pct=fee_pct + slippage_pct,
        configured_max_slippage_pct=round(configured_max_slippage_pct, 6),
        spread_pct=round(spread_pct, 6),
        spread_source=spread_source,
        bid_depth_usdt=round(bid_depth, 6),
        ask_depth_usdt=round(ask_depth, 6),
        orderbook_imbalance=round(imbalance, 6),
        liquidity_penalty_pct=round(liquidity_penalty_pct, 6),
        imbalance_penalty_pct=round(imbalance_penalty_pct, 6),
        slippage_cap_pct=round(configured_max_slippage_pct, 6),
        slippage_source="capped_by_configured_max" if cap_used else "dynamic_microstructure",
    )


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


def _liquidity_penalty_pct(min_depth_usdt: float) -> float:
    if min_depth_usdt <= 0:
        return 0.04
    depth_gap = max(LIQUIDITY_DEPTH_REFERENCE_USDT - min_depth_usdt, 0.0)
    return min(depth_gap / LIQUIDITY_DEPTH_REFERENCE_USDT * 0.08, 0.08)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
