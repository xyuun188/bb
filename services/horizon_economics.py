"""Read-only horizon economics for the profitability gate.

This module answers one question: given the observed gross directional edge
and the observed round-trip execution cost, can a prediction horizon produce a
positive fee-after expectation at all? It never reads live orders, changes
policy, or authorizes execution.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

HORIZON_ECONOMICS_VERSION = "2026-11-08.horizon-economics.v1"


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _view(
    baselines: dict[str, Any] | None,
    horizon: str,
    side: str,
    view: str,
) -> dict[str, Any]:
    horizon_rows = (baselines or {}).get(horizon)
    if not isinstance(horizon_rows, dict):
        return {}
    side_rows = horizon_rows.get(side)
    if not isinstance(side_rows, dict):
        return {}
    candidate = side_rows.get(view)
    return candidate if isinstance(candidate, dict) else {}


def evaluate_horizon_economics(
    baselines_by_horizon: dict[str, Any] | None,
    *,
    view: str = "nonoverlapping",
) -> dict[str, Any]:
    """Summarize whether each horizon's gross edge can clear its own cost.

    ``gross_edge_pct`` is the mean return before execution cost. The observed
    round-trip cost is reported separately. A horizon is only viable when the
    gross edge exceeds that cost with a positive margin, and the fee-after
    lower confidence bound is positive.
    """

    rows: list[dict[str, Any]] = []
    for horizon in sorted(
        (baselines_by_horizon or {}).keys(), key=lambda value: _sort_key(value)
    ):
        for side in ("long", "short"):
            evidence = _view(baselines_by_horizon, horizon, side, view)
            if not evidence:
                continue
            count = _float(evidence.get("count")) or 0.0
            gross = _float(evidence.get("mean_selected_gross_return_pct"))
            cost = _float(evidence.get("mean_selected_execution_cost_pct"))
            net = _float(evidence.get("avg_return_pct"))
            lcb = _float(evidence.get("return_lcb_pct"))
            pf = _float(evidence.get("profit_factor"))
            gross_to_cost = (
                gross / cost if gross is not None and cost and cost > 0.0 else None
            )
            rows.append(
                {
                    "horizon_minutes": horizon,
                    "side": side,
                    "view": view,
                    "sample_count": int(count),
                    "gross_edge_pct": gross,
                    "execution_cost_pct": cost,
                    "net_edge_pct": net,
                    "return_lcb_pct": lcb,
                    "profit_factor": pf,
                    "gross_edge_to_cost_ratio": gross_to_cost,
                    "viable": bool(
                        count > 0
                        and gross is not None
                        and cost is not None
                        and cost > 0.0
                        and gross > cost
                        and lcb is not None
                        and lcb > 0.0
                        and pf is not None
                        and pf > 1.0
                    ),
                }
            )
    viable = [row for row in rows if row["viable"]]
    return {
        "version": HORIZON_ECONOMICS_VERSION,
        "view": view,
        "diagnostic_only": True,
        "production_permission": False,
        "orders_submitted": False,
        "runtime_policy_changed": False,
        "rows": rows,
        "viable_horizons": [
            {
                "horizon_minutes": row["horizon_minutes"],
                "side": row["side"],
                "net_edge_pct": row["net_edge_pct"],
                "return_lcb_pct": row["return_lcb_pct"],
                "profit_factor": row["profit_factor"],
            }
            for row in viable
        ],
        "conclusion": (
            "at_least_one_horizon_clears_cost"
            if viable
            else "no_horizon_clears_cost_in_this_sample"
        ),
    }


def _sort_key(value: Any) -> tuple[int, float]:
    parsed = _float(value)
    return (0, parsed) if parsed is not None else (1, 0.0)


def required_hit_rate(gross_edge_pct: float, execution_cost_pct: float) -> float | None:
    """Directional accuracy needed to break even for a symmetric +/- edge.

    Uses a two-outcome model where a correct call earns ``gross_edge_pct`` and
    a wrong call loses it, then charges ``execution_cost_pct`` on every trade.
    """

    if gross_edge_pct <= 0.0 or execution_cost_pct < 0.0:
        return None
    # 2p*edge - edge - cost > 0  =>  p > (1 + cost/edge) / 2
    return (1.0 + execution_cost_pct / gross_edge_pct) / 2.0


def assess_baseline_stability(
    fold_baselines: list[dict[str, Any]] | None,
    *,
    view: str = "nonoverlapping",
) -> dict[str, Any]:
    """Check whether an aggregate positive baseline survives time splits.

    An aggregate edge can be produced by one directional market regime. This
    reports the same side/view per chronological fold so a single positive
    aggregate cannot be mistaken for a stable strategy.
    """

    per_fold: list[dict[str, Any]] = []
    for fold_index, baselines in enumerate(fold_baselines or []):
        for horizon in sorted((baselines or {}).keys(), key=_sort_key):
            for side in ("long", "short"):
                evidence = _view(baselines, horizon, side, view)
                if not evidence:
                    continue
                count = _float(evidence.get("count")) or 0.0
                net = _float(evidence.get("avg_return_pct"))
                lcb = _float(evidence.get("return_lcb_pct"))
                per_fold.append(
                    {
                        "fold_index": fold_index,
                        "horizon_minutes": horizon,
                        "side": side,
                        "sample_count": int(count),
                        "net_edge_pct": net,
                        "return_lcb_pct": lcb,
                        "profit_factor": _float(evidence.get("profit_factor")),
                        "positive": bool(
                            count > 0
                            and net is not None
                            and net > 0.0
                            and lcb is not None
                            and lcb > 0.0
                        ),
                    }
                )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_fold:
        grouped.setdefault(
            (str(row["horizon_minutes"]), str(row["side"])), []
        ).append(row)
    stability = [
        {
            "horizon_minutes": horizon,
            "side": side,
            "fold_count": len(rows),
            "positive_fold_count": sum(row["positive"] for row in rows),
            "stable": bool(
                len(rows) >= 3 and all(row["positive"] for row in rows)
            ),
            "folds": rows,
        }
        for (horizon, side), rows in sorted(
            grouped.items(), key=lambda item: (_sort_key(item[0][0]), item[0][1])
        )
    ]
    return {
        "version": HORIZON_ECONOMICS_VERSION,
        "view": view,
        "diagnostic_only": True,
        "production_permission": False,
        "stability": stability,
        "stable_horizons": [
            {
                "horizon_minutes": item["horizon_minutes"],
                "side": item["side"],
            }
            for item in stability
            if item["stable"]
        ],
    }


def primary_horizon_verdict(
    baselines_by_horizon: dict[str, Any] | None,
    *,
    primary_horizon_minutes: int | float | str,
    view: str = "nonoverlapping",
) -> dict[str, Any]:
    """Judge the runtime primary horizon against its own observed cost.

    A horizon is economically infeasible when both directional sides have a
    gross edge at or below the observed round-trip cost. In that state no gate
    relaxation can produce a positive fee-after expectation: the prediction
    window is too short for the cost structure.
    """

    target = str(primary_horizon_minutes)
    rows = [
        row
        for row in evaluate_horizon_economics(baselines_by_horizon, view=view)["rows"]
        if str(row["horizon_minutes"]) == target
    ]
    analyzed = [row for row in rows if row["gross_edge_pct"] is not None]
    reasons: list[str] = []
    if not analyzed:
        reasons.append("primary_horizon_economics_evidence_missing")
    else:
        best_gross = max(float(row["gross_edge_pct"]) for row in analyzed)
        cost_values = [
            float(row["execution_cost_pct"])
            for row in analyzed
            if row["execution_cost_pct"] is not None
        ]
        cost = max(cost_values) if cost_values else None
        if cost is None or cost <= 0.0:
            reasons.append("primary_horizon_execution_cost_missing")
        elif best_gross <= cost:
            reasons.append("primary_horizon_gross_edge_below_execution_cost")
        if not any(row["viable"] for row in analyzed):
            reasons.append("primary_horizon_fee_after_lcb_not_positive")
    return {
        "version": HORIZON_ECONOMICS_VERSION,
        "primary_horizon_minutes": target,
        "view": view,
        "diagnostic_only": True,
        "production_permission": False,
        "economically_feasible": not reasons,
        "reason_codes": reasons,
        "rows": rows,
    }
