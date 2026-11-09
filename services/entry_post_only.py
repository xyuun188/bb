"""Post-only entry planning, fill classification and honest cost pricing.

A post-only entry rests at the touch and falls back to a market order for the
unfilled remainder.  This module holds the pure decision rules so the executor
and the cost model cannot drift apart, and so each rule is unit-testable
without an exchange connection.
"""

from __future__ import annotations

from typing import Any

POST_ONLY_VERSION = "2026-11-09.entry-post-only.v1"

MIN_POST_ONLY_TIMEOUT_SECONDS = 1.0
MAX_POST_ONLY_TIMEOUT_SECONDS = 90.0

ENTRY_STYLE_POST_ONLY = "post_only"
ENTRY_STYLE_MARKET = "market"
ENTRY_STYLE_POST_ONLY_WITH_FALLBACK = "post_only_with_market_fallback"

FILL_FULL = "full"
FILL_PARTIAL = "partial"
FILL_NONE = "none"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp_timeout_seconds(value: Any) -> float:
    """Bound the passive wait so an entry cannot stall the decision loop."""

    timeout = _number(value, MIN_POST_ONLY_TIMEOUT_SECONDS)
    if timeout <= 0.0:
        return MIN_POST_ONLY_TIMEOUT_SECONDS
    return min(max(timeout, MIN_POST_ONLY_TIMEOUT_SECONDS), MAX_POST_ONLY_TIMEOUT_SECONDS)


def passive_entry_price(*, side: str, bid: Any, ask: Any) -> float:
    """Return the touch price a maker entry should rest at, or ``0.0``.

    Buying joins the bid and selling joins the ask. Joining the touch is the
    only price that can fill without crossing and without paying taker fees.
    """

    normalized = str(side or "").strip().lower()
    best_bid = max(_number(bid), 0.0)
    best_ask = max(_number(ask), 0.0)
    if best_bid <= 0.0 or best_ask <= 0.0 or best_bid > best_ask:
        return 0.0
    if normalized in {"buy", "long"}:
        return best_bid
    if normalized in {"sell", "short"}:
        return best_ask
    return 0.0


def plan_entry_post_only(
    *,
    enabled: Any,
    side: str,
    bid: Any,
    ask: Any,
    timeout_seconds: Any,
) -> dict[str, Any]:
    """Decide whether this entry may rest passively, and at what price."""

    timeout = clamp_timeout_seconds(timeout_seconds)
    plan: dict[str, Any] = {
        "version": POST_ONLY_VERSION,
        "side": str(side or "").strip().lower(),
        "enabled": bool(enabled),
        "timeout_seconds": timeout,
        "price": 0.0,
        "bid": max(_number(bid), 0.0),
        "ask": max(_number(ask), 0.0),
        "eligible": False,
        "reason": "",
    }
    if not plan["enabled"]:
        plan["reason"] = "post_only_disabled"
        return plan
    price = passive_entry_price(side=side, bid=bid, ask=ask)
    if price <= 0.0:
        plan["reason"] = "post_only_touch_unavailable"
        return plan
    plan["price"] = price
    plan["eligible"] = True
    plan["reason"] = "post_only_touch_ready"
    return plan


def classify_passive_fill(
    *,
    filled_contracts: Any,
    requested_contracts: Any,
    tolerance: float = 1e-9,
) -> str:
    """Classify how much of a resting entry ended up filled."""

    filled = max(_number(filled_contracts), 0.0)
    requested = max(_number(requested_contracts), 0.0)
    if requested <= 0.0 or filled <= 0.0:
        return FILL_NONE
    if filled + tolerance >= requested:
        return FILL_FULL
    return FILL_PARTIAL


def is_terminal_unfilled_state(state: Any) -> bool:
    """True when the exchange closed the order without any fill."""

    text = str(state or "").strip().lower()
    return text in {"canceled", "cancelled", "rejected", "expired"}


def expected_post_only_round_trip_cost_pct(
    *,
    maker_fee_rate: Any,
    taker_fee_rate: Any,
    passive_fill_probability: Any,
    fallback_entry_slippage_pct: Any = 0.0,
) -> dict[str, Any]:
    """Expected round-trip cost of a post-only entry with a market fallback.

    The entry leg pays the maker rate when the passive order fills and the
    taker rate plus slippage when the wait expires. The exit leg always pays
    the taker rate because exits stay market orders. Nothing here assumes a
    maker fill; the probability is an explicit input that must be measured.
    """

    maker = max(_number(maker_fee_rate), 0.0)
    taker = max(_number(taker_fee_rate), 0.0)
    probability = min(max(_number(passive_fill_probability), 0.0), 1.0)
    fallback_slippage = max(_number(fallback_entry_slippage_pct), 0.0)
    if maker <= 0.0 or taker <= 0.0:
        return {
            "version": POST_ONLY_VERSION,
            "eligible": False,
            "reason": "maker_or_taker_fee_missing",
            "expected_total_pct": None,
        }

    maker_path_pct = (maker + taker) * 100.0
    fallback_path_pct = (taker + taker) * 100.0 + fallback_slippage
    expected = probability * maker_path_pct + (1.0 - probability) * fallback_path_pct
    return {
        "version": POST_ONLY_VERSION,
        "eligible": True,
        "reason": "post_only_cost_estimate_ready",
        "maker_fee_rate": maker,
        "taker_fee_rate": taker,
        "passive_fill_probability": round(probability, 6),
        "maker_path_pct": round(maker_path_pct, 6),
        "fallback_path_pct": round(fallback_path_pct, 6),
        "expected_total_pct": round(expected, 6),
    }


def post_only_execution_fact(
    *,
    plan: dict[str, Any] | None,
    outcome: str,
    fallback_reason: str = "",
    maker_fill_contracts: Any = 0.0,
    taker_fill_contracts: Any = 0.0,
    post_only_order_id: Any = "",
) -> dict[str, Any]:
    """Execution fact attached to the entry result so fills can be measured."""

    payload = plan if isinstance(plan, dict) else {}
    attempted = bool(payload.get("eligible"))
    return {
        "version": POST_ONLY_VERSION,
        "entry_execution_style": (
            ENTRY_STYLE_POST_ONLY_WITH_FALLBACK
            if attempted
            else ENTRY_STYLE_MARKET
        ),
        "post_only_attempted": attempted,
        "post_only_price": payload.get("price"),
        "post_only_timeout_seconds": payload.get("timeout_seconds"),
        "post_only_plan_reason": payload.get("reason"),
        "post_only_outcome": outcome,
        "fallback_reason": fallback_reason,
        "maker_fill_contracts": round(max(_number(maker_fill_contracts), 0.0), 12),
        "taker_fill_contracts": round(max(_number(taker_fill_contracts), 0.0), 12),
        "post_only_order_id": str(post_only_order_id or "").strip() or None,
    }
