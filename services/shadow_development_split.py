"""Point-in-time isolation for diagnostic shadow outcome comparisons."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def split_shadow_development_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Use a common decision boundary across horizons and purge late outcomes."""
    group_horizons = Counter((row["decision_id"], row["horizon_minutes"]) for row in rows)
    duplicate_groups = {
        group for (group, _), count in group_horizons.items() if group and count > 1
    }
    decision_times: dict[int, set[datetime]] = defaultdict(set)
    observations = []
    exclusions: Counter[str] = Counter()
    for row in rows:
        group = row["decision_id"]
        horizon = row["horizon_minutes"]
        decision_at = _timestamp(row.get("decision_at"))
        label_at = _timestamp(row.get("label_at"))
        updated_at = _timestamp(row.get("updated_at"))
        if not group:
            exclusions["missing_decision_id"] += 1
            continue
        if group in duplicate_groups:
            exclusions["duplicate_group"] += 1
            continue
        if decision_at is not None:
            decision_times[group].add(decision_at)
        if (
            decision_at is None
            or label_at is None
            or updated_at is None
            or horizon <= 0
            or label_at < decision_at + timedelta(minutes=horizon)
            or updated_at < label_at
        ):
            exclusions["invalid_timing"] += 1
            continue
        observations.append({**row, "decision_at": decision_at, "outcome_available_at": updated_at})

    ordered = sorted(observations, key=lambda row: (row["decision_at"], row["id"]))
    eligible = [
        row for row in ordered if len(decision_times[row["decision_id"]]) == 1
    ]
    exclusions["conflicting_decision_time"] = len(ordered) - len(eligible)
    group_times = list({row["decision_id"]: row["decision_at"] for row in eligible}.values())
    split = max(1, len(group_times) * 7 // 10)
    validation_start = group_times[split] if split < len(group_times) else None
    development = []
    validation = []
    for row in eligible:
        if validation_start is None:
            development.append(row)
        elif row["decision_at"] >= validation_start:
            validation.append(row)
        elif row["outcome_available_at"] < validation_start:
            development.append(row)
        else:
            exclusions["outcome_not_available_before_validation"] += 1

    return development, validation, {
        "split_policy": "original_decision_time_and_persisted_outcome_availability.v2",
        "decision_group_count": len(group_times),
        "duplicate_decision_group_count": len(duplicate_groups),
        "excluded_row_counts": dict(exclusions),
        "validation_start": validation_start.isoformat() if validation_start else None,
    }
