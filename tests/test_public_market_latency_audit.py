from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from services.public_market_latency_audit import (
    LATENCY_EVENT,
    summarize_public_market_latency_logs,
)


def _event(
    timestamp: str,
    *,
    method: str = "publicGetMarketIndexTickers",
    queue: float = 0.0,
    provider: float = 0.0,
    outcome: str = "ok",
    priority: bool = False,
) -> bytes:
    return json.dumps(
        {
            "timestamp": timestamp,
            "event": LATENCY_EVENT,
            "method": method,
            "queue_wait_ms": queue,
            "provider_duration_ms": provider,
            "outcome": outcome,
            "priority": priority,
            "sensitive_payload": "must_not_be_returned",
        }
    ).encode()


def test_summarizes_only_in_window_latency_fields_without_raw_payload() -> None:
    since = datetime(2026, 10, 25, 1, 0, tzinfo=UTC)
    until = since + timedelta(minutes=45)
    logs = [
        (
            "trading.log",
            b"\n".join(
                (
                    _event(
                        "2026-10-25T01:05:00+00:00",
                        queue=900.0,
                        provider=100.0,
                        priority=True,
                    ),
                    _event(
                        "2026-10-25T01:40:00+00:00",
                        queue=100.0,
                        provider=8000.0,
                        outcome="error",
                    ),
                    _event("2026-10-25T00:59:59+00:00", queue=9000.0),
                    b"not-json",
                )
            ),
        )
    ]

    report = summarize_public_market_latency_logs(logs, since=since, until=until)

    route = report["routes"]["publicGetMarketIndexTickers"]
    assert report["matching_latency_event_count"] == 2
    assert route["queue_wait_ms"] == {
        "count": 2,
        "p50": 100.0,
        "p95": 900.0,
        "max": 900.0,
    }
    assert route["provider_duration_ms"]["p95"] == 8000.0
    assert route["outcomes"] == {"ok": 1, "error": 1}
    assert report["window"]["diagnostic_event_only"] is True
    assert "sensitive_payload" not in json.dumps(report)


def test_parses_structlog_event_nested_in_journald_message() -> None:
    since = datetime(2026, 10, 25, 1, 0, tzinfo=UTC)
    until = since + timedelta(minutes=45)
    nested = json.loads(
        _event(
            "2026-10-25T01:05:00+00:00",
            queue=700.0,
            provider=900.0,
        )
    )
    outer = json.dumps({"MESSAGE": json.dumps(nested)}).encode()

    report = summarize_public_market_latency_logs(
        [("journald", outer)],
        since=since,
        until=until,
    )

    assert report["matching_latency_event_count"] == 1
    assert report["routes"]["publicGetMarketIndexTickers"]["queue_wait_ms"]["max"] == 700.0


def test_bounds_log_count_and_rejects_naive_or_reversed_windows() -> None:
    now = datetime(2026, 10, 25, 1, 0, tzinfo=UTC)
    logs = [
        (f"trading.log.{index}", _event(now.isoformat()))
        for index in range(8)
    ]

    report = summarize_public_market_latency_logs(
        logs,
        since=now - timedelta(minutes=1),
        until=now + timedelta(minutes=1),
    )

    assert report["bounded_log_tails"]["max_files"] == 6
    assert len(report["bounded_log_tails"]["files"]) == 6
    with pytest.raises(ValueError):
        summarize_public_market_latency_logs(
            logs,
            since=now.replace(tzinfo=None),
            until=now + timedelta(minutes=1),
        )
    with pytest.raises(ValueError):
        summarize_public_market_latency_logs(
            logs,
            since=now,
            until=now,
        )
