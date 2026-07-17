from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from services.production_source_health import summarize_production_source_health


def _decision(
    created_at: datetime,
    *,
    source_count: int = 0,
    canary: bool = False,
    executed: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        created_at=created_at,
        analysis_type="market",
        was_executed=executed,
        raw_llm_response={
            "authoritative_return_candidate": {
                "side_evidence": {"production_source_count": source_count}
            },
            "paper_bootstrap_canary": {"requested": canary},
        },
    )


def test_continuous_no_production_source_raises_critical_alert() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    rows = [_decision(now - timedelta(hours=2, minutes=index)) for index in range(20)]

    report = summarize_production_source_health(rows, now=now)

    assert report["status"] == "critical"
    assert report["alert_active"] is True
    assert report["reason"] == "continuous_no_production_return_source"


def test_recent_production_source_clears_alert() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    rows = [
        _decision(now - timedelta(minutes=2), source_count=1),
        _decision(now - timedelta(minutes=3)),
    ]

    report = summarize_production_source_health(rows, now=now)

    assert report["status"] == "ok"
    assert report["alert_active"] is False


def test_alert_reports_paper_bootstrap_recovery_progress() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    rows = [
        _decision(now - timedelta(hours=2), canary=True, executed=True),
        _decision(now - timedelta(hours=3)),
    ]

    report = summarize_production_source_health(rows, now=now)

    assert report["status"] == "critical"
    assert report["recovery_state"] == "paper_bootstrap_collecting"
    assert report["paper_bootstrap_executed_count"] == 1
