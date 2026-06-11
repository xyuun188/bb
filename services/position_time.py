from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

BEIJING_TZ = timezone(timedelta(hours=8))


class PositionTimeParser:
    """Parse exchange timestamps and local position ages consistently."""

    def __init__(self, now_provider: Callable[[], datetime] | None = None) -> None:
        self.now_provider = now_provider or (lambda: datetime.now(UTC))

    def datetime_from_ms(self, timestamp_ms: Any) -> datetime:
        try:
            return datetime.fromtimestamp(float(timestamp_ms) / 1000, tz=UTC)
        except (TypeError, ValueError):
            return self.now_provider()

    def position_age_minutes(self, created_at: Any) -> float | None:
        if not created_at:
            return None
        try:
            opened = self._parse_created_at(created_at)
            if not isinstance(opened, datetime):
                return None
            now = self.now_provider()
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=UTC)
                if opened > now + timedelta(minutes=1):
                    opened = opened.replace(tzinfo=None).replace(tzinfo=BEIJING_TZ)
            return max((now - opened.astimezone(UTC)).total_seconds() / 60.0, 0.0)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_created_at(created_at: Any) -> Any:
        opened = created_at
        if isinstance(opened, (int, float)):
            value = float(opened)
            if value > 10_000_000_000:
                value = value / 1000.0
            return datetime.fromtimestamp(value, tz=UTC)
        if isinstance(opened, str):
            stripped = opened.strip()
            if stripped.isdigit():
                value = float(stripped)
                if value > 10_000_000_000:
                    value = value / 1000.0
                return datetime.fromtimestamp(value, tz=UTC)
            return datetime.fromisoformat(stripped.replace("Z", "+00:00"))
        return opened
