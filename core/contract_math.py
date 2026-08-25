"""Numerical helpers for persisted execution-contract fields."""

from __future__ import annotations

from math import isclose, isfinite
from typing import Any


def persisted_product_isclose(
    product: Any,
    left: Any,
    right: Any,
    *,
    decimal_places: int = 8,
    rel_tol: float = 1e-9,
    minimum_abs_tol: float = 1e-8,
) -> bool:
    """Compare a persisted product without rejecting decimal rounding.

    Contract payloads persist the operands and their product independently,
    currently at eight decimal places. The tolerance follows the maximum
    first-order error from rounding both operands, so tiny legitimate drift
    passes while a material contract mutation still fails closed.
    """

    try:
        product_value = float(product)
        left_value = float(left)
        right_value = float(right)
        places = int(decimal_places)
        relative_tolerance = max(float(rel_tol), 0.0)
        floor_tolerance = max(float(minimum_abs_tol), 0.0)
    except (TypeError, ValueError, OverflowError):
        return False
    if not all(isfinite(value) for value in (product_value, left_value, right_value)):
        return False
    if places < 0 or places > 15:
        return False

    unit = 10.0 ** (-places)
    rounding_tolerance = (
        0.5 * unit * (abs(left_value) + abs(right_value))
        + 0.25 * unit * unit
    )
    return isclose(
        product_value,
        left_value * right_value,
        rel_tol=relative_tolerance,
        abs_tol=max(floor_tolerance, rounding_tolerance),
    )
