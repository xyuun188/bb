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


def select_paper_horizon_cohort(
    rows: Iterable[dict[str, Any]],
    *,
    preferred_horizon_minutes: Any = None,
    source_weights: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select one coherent model horizon without mixing return semantics.

    Each source contributes its weight once per horizon even when the caller
    supplies separate long and short evidence rows. The highest-weight cohort
    wins; equal weights deterministically prefer the shorter horizon.
    """

    weights = source_weights or {}
    groups: dict[float, dict[str, float]] = {}
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

    available_groups = [
        {
            "horizon_minutes": horizon,
            "source_count": len(source_rows),
            "total_weight": round(sum(source_rows.values()), 12),
            "sources": sorted(source_rows),
        }
        for horizon, source_rows in sorted(groups.items())
    ]
    preferred = _positive_float(preferred_horizon_minutes)
    blockers: list[str] = []
    selection_reason = "highest_continuous_weight_then_shortest_horizon"
    selected: float | None = None
    if preferred is not None:
        selection_reason = "authorized_prediction_horizon"
        if preferred in groups:
            selected = preferred
        else:
            blockers.append("paper_prediction_horizon_unavailable")
    elif groups:
        selected = max(
            groups,
            key=lambda horizon: (sum(groups[horizon].values()), -horizon),
        )
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
