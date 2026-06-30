from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

PHASE3_BEIJING_TZ = timezone(timedelta(hours=8))
PHASE3_FIRST_CLEAN_DAY = "2026-06-28"
PHASE3_CLEAN_START_LOCAL = datetime(2026, 6, 28, 0, 0, tzinfo=PHASE3_BEIJING_TZ)
PHASE3_CLEAN_START_UTC = PHASE3_CLEAN_START_LOCAL.astimezone(UTC)


def phase3_clean_start_utc_naive() -> datetime:
    """Return the Phase 3 clean start instant in DB-naive UTC form."""

    return PHASE3_CLEAN_START_UTC.replace(tzinfo=None)
