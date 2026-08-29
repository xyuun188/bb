"""Shared snapshot and freshness contracts used by Dashboard diagnostics.

The dashboard has several independent readers (data, models, audits and
trading facts).  Keeping the envelope in one small module prevents each
endpoint from inventing a different meaning for ``0``, ``unknown`` or stale
data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping

OBSERVABILITY_STATUSES = {
    "ok",
    "warming",
    "partial",
    "timeout",
    "deferred",
    "blocked",
    "warning",
    "passed",
    "observing",
    "not_started",
    "stale",
    "missing",
    "error",
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_timestamp(value: datetime | None = None) -> str:
    current = value or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(UTC).isoformat()


def normalize_status(value: Any, *, default: str = "missing") -> str:
    status = str(value or "").strip().lower()
    return status if status in OBSERVABILITY_STATUSES else default


def freshness_payload(
    *,
    checked_at: datetime | str | None,
    now: datetime | None = None,
    stale_after_seconds: float | None = None,
) -> dict[str, Any]:
    """Return one explicit freshness shape for API responses."""

    current = now or utc_now()
    if isinstance(checked_at, datetime):
        parsed = checked_at
    else:
        try:
            parsed = datetime.fromisoformat(str(checked_at).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            parsed = None
    if parsed is not None and parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    age = (
        max((current - parsed.astimezone(UTC)).total_seconds(), 0.0)
        if parsed is not None
        else None
    )
    stale = bool(
        age is None
        or (
            stale_after_seconds is not None
            and age > max(float(stale_after_seconds), 0.0)
        )
    )
    return {
        "checked_at": parsed.astimezone(UTC).isoformat() if parsed is not None else None,
        "age_seconds": round(age, 3) if age is not None else None,
        "stale_after_seconds": (
            round(max(float(stale_after_seconds), 0.0), 3)
            if stale_after_seconds is not None
            else None
        ),
        "is_stale": stale,
        "state": "stale" if stale else "fresh",
    }


def build_snapshot(
    payload: Mapping[str, Any] | None = None,
    *,
    status: Any = "ok",
    source: str,
    checked_at: datetime | None = None,
    stale_after_seconds: float | None = None,
    degraded_reason: str | None = None,
    version: str = "2026-08-29.observability.v1",
    **metadata: Any,
) -> dict[str, Any]:
    """Wrap an endpoint payload in a stable, human-auditable envelope."""

    checked = checked_at or utc_now()
    normalized = normalize_status(status)
    result: dict[str, Any] = {
        **dict(payload or {}),
        "status": normalized,
        "checked_at": iso_timestamp(checked),
        "source": str(source or "unknown"),
        "freshness": freshness_payload(
            checked_at=checked,
            now=utc_now(),
            stale_after_seconds=stale_after_seconds,
        ),
        "degraded_reason": str(degraded_reason)[:500] if degraded_reason else None,
        "observability_version": version,
    }
    result.update(metadata)
    return result


def status_from_sections(sections: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Aggregate child statuses without treating unknown values as success."""

    degraded: list[str] = []
    statuses: list[str] = []
    for name, value in sections.items():
        row = value if isinstance(value, Mapping) else {}
        state = normalize_status(row.get("status"), default="missing")
        statuses.append(state)
        if state != "ok":
            degraded.append(str(name))
    if not statuses:
        return "missing", degraded
    if all(state == "ok" for state in statuses):
        return "ok", degraded
    if any(state in {"error", "timeout"} for state in statuses):
        return "partial" if any(state == "ok" for state in statuses) else "error", degraded
    if any(state == "deferred" for state in statuses):
        return "partial" if any(state == "ok" for state in statuses) else "deferred", degraded
    if any(state == "blocked" for state in statuses):
        return "partial" if any(state == "ok" for state in statuses) else "blocked", degraded
    if any(state == "stale" for state in statuses):
        return "stale", degraded
    if any(state == "warming" for state in statuses):
        return "warming", degraded
    return "partial", degraded
