"""Deterministic horizon-cohort selection for paper model aggregation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Any


def _positive_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) and number > 0.0 else default


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def select_paper_horizon_cohort(
    rows: Iterable[dict[str, Any]],
    *,
    preferred_horizon_minutes: Any = None,
    source_weights: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select one coherent model horizon without mixing return semantics.

    Each source contributes its weight once per horizon even when the caller
    supplies separate long and short evidence rows. The highest-weight cohort
    wins. Because most sources publish the same weight at every horizon, the
    weight total usually ties; the tie-break is then the best *fee-after*
    evidence in the cohort, and only then the shorter horizon.

    Selecting the shortest horizon on a tie used to win by default even when
    that horizon's cost was larger than its gross edge. Holding time does not
    change the round-trip cost, so the short horizon systematically lost the
    cost/edge comparison. This picks the best net evidence instead; it never
    relaxes a return, cost or data-quality gate.
    """

    weights = source_weights or {}
    groups: dict[float, dict[str, float]] = {}
    net_by_source: dict[float, dict[str, float]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        horizon = _positive_float(row.get("horizon_minutes"))
        if horizon is None:
            continue
        source = str(row.get("source") or row.get("key") or f"row:{index}")
        weight = _positive_float(
            weights.get(source),
            _positive_float(row.get("continuous_weight_multiplier"), 1.0),
        )
        group = groups.setdefault(horizon, {})
        group[source] = max(group.get(source, 0.0), float(weight or 1.0))
        net = _finite_float(row.get("expected_net_return_pct"))
        if net is None:
            net = _finite_float(row.get("objective_expected_return_pct"))
        if net is not None:
            # One vote per source, exactly as the weight aggregation above.
            per_source = net_by_source.setdefault(horizon, {})
            if source not in per_source or net > per_source[source]:
                per_source[source] = net

    def _cohort_net(horizon: float) -> float | None:
        per_source = net_by_source.get(horizon) or {}
        if not per_source:
            return None
        source_rows = groups.get(horizon) or {}
        weighted = [
            (value, float(source_rows.get(source, 1.0)))
            for source, value in per_source.items()
        ]
        denominator = sum(weight for _value, weight in weighted)
        if denominator <= 0.0:
            return None
        return sum(value * weight for value, weight in weighted) / denominator

    available_groups = [
        {
            "horizon_minutes": horizon,
            "source_count": len(source_rows),
            "total_weight": round(sum(source_rows.values()), 12),
            "sources": sorted(source_rows),
            "best_fee_after_return_pct": (
                None if _cohort_net(horizon) is None else round(_cohort_net(horizon), 12)
            ),
        }
        for horizon, source_rows in sorted(groups.items())
    ]
    preferred = _positive_float(preferred_horizon_minutes)
    blockers: list[str] = []
    selection_reason = (
        "highest_continuous_weight_then_best_fee_after_return_then_shortest_horizon"
    )
    selected: float | None = None
    if preferred is not None:
        selection_reason = "authorized_prediction_horizon"
        if preferred in groups:
            selected = preferred
        else:
            blockers.append("paper_prediction_horizon_unavailable")
    elif groups:
        totals = {horizon: sum(source_rows.values()) for horizon, source_rows in groups.items()}
        top_weight = max(totals.values())
        # Weights are compared with a tolerance because they are sums of
        # independently rounded continuous multipliers.
        tied = [
            horizon
            for horizon, total in totals.items()
            if abs(total - top_weight) <= max(abs(top_weight), 1.0) * 1e-9
        ]
        def _rank(horizon: float) -> tuple[float, float]:
            net = _cohort_net(horizon)
            return (float("-inf") if net is None else net, -horizon)

        selected = max(tied, key=_rank)
    else:
        blockers.append("paper_prediction_horizon_unavailable")

    selected_sources = sorted(groups.get(selected, {})) if selected is not None else []
    excluded_sources = [
        {
            "source": source,
            "horizon_minutes": horizon,
            "reason": "paper_prediction_horizon_not_selected",
        }
        for horizon, source_rows in sorted(groups.items())
        if selected is None or horizon != selected
        for source in sorted(source_rows)
    ]
    return {
        "selected_horizon_minutes": selected,
        "selection_reason": selection_reason,
        "available_horizon_groups": available_groups,
        "selected_sources": selected_sources,
        "excluded_sources": excluded_sources,
        "blockers": blockers,
    }
