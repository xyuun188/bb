"""Fee-aware price limits for discretionary profit-taking orders."""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from math import isfinite
from typing import Any


def profit_lock_execution_required(raw: dict[str, Any]) -> bool:
    policy = raw.get("dynamic_exit_policy") or {}
    if not isinstance(policy, dict) or raw.get("manual_close"):
        return False
    if policy.get("hard_risk") or any(
        policy.get(key)
        for key in (
            "planned_stop_crossed", "funding_loss_budget_crossed",
            "projected_funding_budget_crossed",
        )
    ):
        return False
    try:
        gross_pnl = float(policy.get("gross_unrealized_pnl_usdt", float("nan")))
        price_loss_risk = bool(
            max(
                float(policy.get("stop_risk_usage") or 0.0),
                float(policy.get("continuation_deterioration") or 0.0),
            ) > 0.0
            and (not isfinite(gross_pnl) or gross_pnl < 0.0)
        )
        return bool(
            float(policy.get("lifecycle_net_pnl_usdt") or 0.0) > 0.0
            and max(
                float(policy.get("profit_lock_pressure") or 0.0),
                float(policy.get("profit_retrace_ratio") or 0.0),
            ) > 0.0
            # Settled funding alone is not urgent price-loss evidence when
            # both the remaining price PnL and fee-after total are positive.
            and not price_loss_risk
            and not any(
                float(policy.get(key) or 0.0) > 0.0
                for key in ("opposite_pressure", "model_exit_pressure")
            )
        )
    except (TypeError, ValueError):
        return False


def profit_lock_limit(
    *, side: str, entry_price: float, quantity: float,
    entry_fee_usdt: float, exit_fee_rate: float, tick_size: float,
    adverse_funding_usdt: float = 0.0,
    price_limit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One tick beyond fee-after break-even; quantities are base units."""
    values = (
        entry_price, quantity, entry_fee_usdt, exit_fee_rate,
        tick_size, adverse_funding_usdt,
    )
    if (
        side not in {"buy", "sell"}
        or not all(isfinite(value) for value in values)
        or entry_price <= 0 or quantity <= 0 or tick_size <= 0
        or entry_fee_usdt < 0 or not 0 <= exit_fee_rate < 1
        or adverse_funding_usdt < 0
    ):
        return {"eligible": False, "reason": "profit_lock_execution_economics_missing"}
    try:
        entry, qty, fee, rate, tick, funding = (Decimal(str(v)) for v in values)
        cost = fee + funding
        boundary = (
            (entry * qty - cost) / (qty * (1 + rate))
            if side == "buy"
            else (entry * qty + cost) / (qty * (1 - rate))
        )
        rounding = ROUND_FLOOR if side == "buy" else ROUND_CEILING
        units = (boundary / tick).to_integral_value(rounding=rounding)
        limit = (units - 1 if side == "buy" else units + 1) * tick
        fee_after_limit = limit
        if price_limit is not None:
            bound_key = "buy_price_limit" if side == "buy" else "sell_price_limit"
            bound = Decimal(str(price_limit.get(bound_key, "NaN")))
            if not bound.is_finite() or bound <= 0:
                return {"eligible": False, "reason": "profit_lock_price_limit_missing"}
            # A more favorable price preserves the fee-after floor while
            # keeping an IOC inside the execution venue's current price band.
            bound_units = (bound / tick).to_integral_value(rounding=rounding)
            venue_limit = bound_units * tick
            limit = min(limit, venue_limit) if side == "buy" else max(limit, venue_limit)
    except (InvalidOperation, ZeroDivisionError):
        return {"eligible": False, "reason": "profit_lock_execution_economics_missing"}
    return {
        "eligible": limit > 0,
        "reason": "fee_after_profit_limit" if limit > 0 else "profit_limit_not_positive",
        "limit_price": float(limit),
        "fee_after_limit_price": float(fee_after_limit),
        "execution_price_limit": price_limit,
        "break_even_price": float(boundary),
        "entry_fee_usdt": entry_fee_usdt,
        "exit_fee_rate": exit_fee_rate,
        "adverse_funding_usdt": adverse_funding_usdt,
        "order_type": "ioc",
        "market_fallback_allowed": False,
        "version": "2026-11-18.executable-profit-lock.v2",
    }
