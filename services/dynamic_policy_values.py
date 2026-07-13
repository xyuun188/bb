"""Auditable data-derived values used by live trading policies.

Trading policies may use exchange constraints and mathematical boundaries directly.
Every market-dependent decision value must instead be represented by this module so
callers can prove where it came from and avoid hidden constant fallbacks.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import Any, Literal

PolicySelector = Literal["lower_hinge", "median", "upper_hinge"]

DYNAMIC_POLICY_VERSION = "2026-07-12.dynamic-policy.v1"


def _finite_values(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(number):
            result.append(number)
    return sorted(result)


def _median(ordered: list[float]) -> float:
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _hinges(ordered: list[float]) -> tuple[float, float, float]:
    """Return Tukey hinges without a configured percentile threshold."""

    center = _median(ordered)
    middle = len(ordered) // 2
    lower = ordered[: middle + (len(ordered) % 2)]
    upper = ordered[middle:]
    return _median(lower), center, _median(upper)


@dataclass(frozen=True, slots=True)
class DynamicPolicyValue:
    name: str
    value: float | None
    source: str
    observation_window: str
    sample_count: int
    generated_at: str
    strategy_version: str = DYNAMIC_POLICY_VERSION
    fallback_reason: str = ""
    production_eligible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def empirical_policy_value(
    name: str,
    values: Iterable[Any],
    *,
    selector: PolicySelector,
    observation_window: str,
    generated_at: datetime | None = None,
) -> DynamicPolicyValue:
    """Generate a live policy value from empirical order statistics.

    No constant market threshold is used. Empty distributions are explicitly
    ineligible instead of falling back to a configured trading value.
    """

    ordered = _finite_values(values)
    timestamp = (generated_at or datetime.now(UTC)).isoformat()
    if not ordered:
        return DynamicPolicyValue(
            name=name,
            value=None,
            source="empirical_order_statistics",
            observation_window=observation_window,
            sample_count=0,
            generated_at=timestamp,
            fallback_reason="empirical_distribution_unavailable",
            production_eligible=False,
        )
    lower, center, upper = _hinges(ordered)
    selected = {
        "lower_hinge": lower,
        "median": center,
        "upper_hinge": upper,
    }[selector]
    return DynamicPolicyValue(
        name=name,
        value=selected,
        source=f"empirical_order_statistics:{selector}",
        observation_window=observation_window,
        sample_count=len(ordered),
        generated_at=timestamp,
    )


def continuous_budget_fraction(*pressures: Any) -> float:
    """Combine normalized dynamic pressures without tier or fixed-size buckets."""

    values = _finite_values(pressures)
    if not values:
        return 0.0
    bounded = [min(max(value, 0.0), 1.0) for value in values]
    remaining = 1.0
    for value in bounded:
        remaining *= 1.0 - value
    return min(max(1.0 - remaining, 0.0), 1.0)
