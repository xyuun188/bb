"""Research-only fast-exit counterfactuals over mature five-minute labels."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any

FAST_EXIT_TARGETS_PCT = (0.15, 0.20, 0.30, 0.50, 0.75)
FAST_EXIT_MIN_DEVELOPMENT_GROUPS = 16
FAST_EXIT_LABEL_HORIZON_MINUTES = 5
FAST_EXIT_POLICY_VERSION = "2026-10-28.fast-exit-mfe-shadow.v1"


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


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


def _normalize_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    normalized = []
    group_counts = Counter(
        str(row.get("decision_id") or row.get("id") or "").strip()
        for row in rows
    )
    duplicate_groups = {
        group for group, count in group_counts.items() if group and count > 1
    }
    duplicate_rows = sum(group_counts[group] for group in duplicate_groups)
    for row in rows:
        side = str(row.get("selected_side") or "").strip().lower()
        group = str(row.get("decision_id") or row.get("id") or "").strip()
        timestamp = _timestamp(row.get("decision_at"))
        mfe = _number(row.get(f"{side}_mfe_pct")) if side in {"long", "short"} else None
        gross = _number(row.get(f"{side}_gross_return_pct")) if side in {"long", "short"} else None
        net = _number(row.get(f"{side}_net_return_pct")) if side in {"long", "short"} else None
        if (
            not group
            or group in duplicate_groups
            or timestamp is None
            or mfe is None
            or gross is None
            or net is None
        ):
            continue
        normalized.append(
            {
                "id": int(row.get("id") or 0),
                "decision_id": int(row.get("decision_id") or 0) or None,
                "symbol": str(row.get("symbol") or ""),
                "side": side,
                "decision_at": timestamp,
                "mfe_pct": max(mfe, 0.0),
                "baseline_gross_pct": gross,
                "cost_pct": gross - net,
                "baseline_net_pct": net,
            }
        )
    return (
        sorted(normalized, key=lambda item: (item["decision_at"], item["id"])),
        len(duplicate_groups),
        duplicate_rows,
    )


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

    observations, duplicate_group_count, duplicate_row_count = _normalize_rows(rows)
    split_index = max(1, int(len(observations) * 0.70)) if observations else 0
    validation_start = (
        observations[split_index]["decision_at"]
        if split_index < len(observations)
        else None
    )
    purge_start = (
        validation_start - timedelta(minutes=FAST_EXIT_LABEL_HORIZON_MINUTES)
        if validation_start is not None
        else None
    )
    development = (
        [row for row in observations if row["decision_at"] < purge_start]
        if purge_start is not None
        else observations
    )
    validation = (
        [row for row in observations if row["decision_at"] >= validation_start]
        if validation_start is not None
        else []
    )
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
    group_count = len(observations)
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
            "selected_decision_group_count": group_count,
            "duplicate_decision_group_count": duplicate_group_count,
            "duplicate_row_count_excluded": duplicate_row_count,
            "duplicate_policy": "exclude every row belonging to a duplicate decision group",
            "development_group_count": len(development),
            "validation_group_count": len(validation),
            "validation_start": (
                validation_start.isoformat() if validation_start is not None else None
            ),
            "purge_minutes": FAST_EXIT_LABEL_HORIZON_MINUTES,
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
