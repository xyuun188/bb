"""Research-only fast-exit counterfactuals over mature five-minute labels."""

from __future__ import annotations

from math import isfinite
from typing import Any

from services.shadow_development_split import split_shadow_development_rows

FAST_EXIT_TARGETS_PCT = (0.15, 0.20, 0.30, 0.50, 0.75)
FAST_EXIT_MIN_DEVELOPMENT_GROUPS = 16
FAST_EXIT_LABEL_HORIZON_MINUTES = 5
FAST_EXIT_POLICY_VERSION = "2026-11-18.fast-exit-mfe-shadow.v2"


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _metrics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean_net_return_pct": None,
            "positive_count": 0,
            "negative_count": 0,
            "profit_factor": None,
            "max_drawdown_pct": None,
        }
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    equity = high_water = max_drawdown = 0.0
    for value in values:
        equity += value
        high_water = max(high_water, equity)
        max_drawdown = max(max_drawdown, high_water - equity)
    return {
        "count": len(values),
        "mean_net_return_pct": round(sum(values) / len(values), 8),
        "positive_count": sum(value > 0 for value in values),
        "negative_count": sum(value < 0 for value in values),
        "profit_factor": round(gains / losses, 8) if losses > 0 else None,
        "max_drawdown_pct": round(max_drawdown, 8),
    }


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        side = str(row.get("selected_side") or "").strip().lower()
        if side not in {"long", "short"}:
            continue
        mfe = _number(row.get(f"{side}_mfe_pct"))
        gross = _number(row.get(f"{side}_gross_return_pct"))
        net = _number(row.get(f"{side}_net_return_pct"))
        if mfe is None or gross is None or net is None:
            continue
        normalized.append(
            {
                "mfe_pct": max(mfe, 0.0),
                "baseline_gross_pct": gross,
                "cost_pct": gross - net,
                "baseline_net_pct": net,
            }
        )
    return normalized


def _counterfactual_net(row: dict[str, Any], target_pct: float) -> float:
    gross = target_pct if row["mfe_pct"] >= target_pct else row["baseline_gross_pct"]
    return gross - row["cost_pct"]


def _policy_metrics(
    rows: list[dict[str, Any]],
    *,
    target_pct: float | None,
) -> dict[str, Any]:
    returns = [
        row["baseline_net_pct"]
        if target_pct is None
        else _counterfactual_net(row, target_pct)
        for row in rows
    ]
    return _metrics(returns)


def build_fast_exit_shadow_report(
    rows: list[dict[str, Any]],
    *,
    window: dict[str, Any],
) -> dict[str, Any]:
    """Select a fixed target on earlier development rows and report one later fold."""

    development_rows, validation_rows, timing = split_shadow_development_rows(
        [{**row, "horizon_minutes": FAST_EXIT_LABEL_HORIZON_MINUTES} for row in rows]
    )
    development = _normalize_rows(development_rows)
    validation = _normalize_rows(validation_rows)
    candidates = []
    for target in FAST_EXIT_TARGETS_PCT:
        metrics = _policy_metrics(development, target_pct=target)
        if (
            metrics["count"] >= FAST_EXIT_MIN_DEVELOPMENT_GROUPS
            and metrics["mean_net_return_pct"] is not None
            and metrics["mean_net_return_pct"] > 0
            and metrics["profit_factor"] is not None
            and metrics["profit_factor"] > 1
        ):
            candidates.append((metrics["mean_net_return_pct"], target))
    selected_target = max(candidates)[1] if candidates else None
    report = {
        "experiment": "bounded-fast-exit-mfe-shadow-development",
        "policy_version": FAST_EXIT_POLICY_VERSION,
        "status": (
            "development_candidate_requires_fresh_exam"
            if selected_target is not None
            else "no_fast_exit_candidate"
        ),
        "window": window,
        "input": {
            **timing,
            "selected_decision_group_count": timing["decision_group_count"],
            "duplicate_row_count_excluded": timing["excluded_row_counts"].get("duplicate_group", 0),
            "duplicate_policy": "exclude every row belonging to a duplicate decision group",
            "development_group_count": len(development),
            "validation_group_count": len(validation),
            "invalid_outcome_row_count": (
                len(development_rows) + len(validation_rows) - len(development) - len(validation)
            ),
            "minimum_label_horizon_minutes": FAST_EXIT_LABEL_HORIZON_MINUTES,
            "deduplicated_by": "decision_id",
        },
        "baseline_five_minute": {
            "development": _policy_metrics(development, target_pct=None),
            "validation": _policy_metrics(validation, target_pct=None),
        },
        "fixed_target_candidates": {
            f"{target:.2f}": {
                "development": _policy_metrics(development, target_pct=target),
                "validation": _policy_metrics(validation, target_pct=target),
            }
            for target in FAST_EXIT_TARGETS_PCT
        },
        "selection": {
            "selected_target_pct": selected_target,
            "rule": (
                "select highest development mean among fixed targets with "
                f"at least {FAST_EXIT_MIN_DEVELOPMENT_GROUPS} groups, mean > 0 "
                "and profit_factor > 1; otherwise select no-trade"
            ),
            "exam_labels_used_for_selection": False,
        },
        "limitations": [
            "MFE proves that a target was touched, not that a limit order filled.",
            "A target touch is modeled at its exact threshold with no queue or latency loss.",
            "The realized five-minute round-trip cost is reused unchanged; this is an optimistic upper bound.",
            "This per-opportunity result is not portfolio PnL or an independent promotion exam.",
        ],
        "safety": {
            "read_only": True,
            "database_mutated": False,
            "artifact_persisted": False,
            "paper_execution_permission": False,
            "promotion_authorized": False,
            "runtime_policy_changed": False,
            "orders_submitted": False,
        },
    }
    return report
