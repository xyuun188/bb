"""Shared timing policy for OKX facts becoming authoritative locally."""

from __future__ import annotations

from datetime import UTC, datetime

AUTHORITATIVE_FILL_SYNC_GRACE_SECONDS = 10 * 60
AUTHORITATIVE_SYNC_CLOCK_SKEW_SECONDS = 5.0


def is_within_authoritative_sync_grace(
    timestamp: datetime | None,
    *,
    observed_at: datetime,
) -> bool:
    if timestamp is None:
        return False
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=UTC)
    age_seconds = (observed_at - timestamp).total_seconds()
    return (
        -AUTHORITATIVE_SYNC_CLOCK_SKEW_SECONDS
        <= age_seconds
        <= AUTHORITATIVE_FILL_SYNC_GRACE_SECONDS
    )
