from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from services.position_open_time import parse_position_time

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
        if isinstance(created_at, datetime) and created_at.tzinfo is None:
            return created_at
        return parse_position_time(created_at)
