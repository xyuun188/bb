"""Shared parsing helpers for position open timestamps."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def position_open_time(position: Any) -> datetime | None:
    """Return the best available open time for a DB or exchange position row."""

    for value in _position_open_time_candidates(position):
        parsed = parse_position_time(value)
        if parsed is not None:
            return parsed
    return None


def position_hold_hours(position: Any, *, now: datetime | None = None) -> float:
    """Calculate holding hours from the best available position open time."""

    opened = position_open_time(position)
    if opened is None:
        return 0.0
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return max((current.astimezone(UTC) - opened.astimezone(UTC)).total_seconds() / 3600.0, 0.0)


def parse_position_time(value: Any) -> datetime | None:
    """Parse ISO strings, DB datetimes, and exchange second/millisecond timestamps."""

    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        return _timestamp_to_datetime(float(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if _looks_numeric(stripped):
            try:
                return _timestamp_to_datetime(float(stripped))
            except (TypeError, ValueError, OSError, OverflowError):
                return None
        try:
            parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def serialize_position_time(value: Any) -> Any:
    """Return an ISO string for parsed timestamps while preserving unusable values."""

    parsed = parse_position_time(value)
    if parsed is not None:
        return parsed.isoformat().replace("+00:00", "Z")
    return value


def _position_open_time_candidates(position: Any) -> list[Any]:
    info = _read(position, "info")
    if not isinstance(info, dict):
        info = {}
    return [
        _read(position, "created_at"),
        _read(position, "opened_at"),
        _read(position, "open_time"),
        _read(position, "openTime"),
        _read(position, "timestamp"),
        _read(position, "datetime"),
        info.get("cTime"),
        info.get("uTime"),
        info.get("openTime"),
        info.get("posTime"),
        info.get("created_at"),
        info.get("timestamp"),
        info.get("datetime"),
    ]


def _read(source: Any, key: str) -> Any:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _timestamp_to_datetime(value: float) -> datetime | None:
    if value <= 0:
        return None
    timestamp = value / 1000.0 if value > 10_000_000_000 else value
    try:
        return datetime.fromtimestamp(timestamp, tz=UTC)
    except (OSError, OverflowError, ValueError):
        return None


def _looks_numeric(value: str) -> bool:
    candidate = value.removeprefix("-")
    return candidate.replace(".", "", 1).isdigit()
