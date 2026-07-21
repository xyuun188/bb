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
    sampling_plan_alert: bool = False,
    normal_paper: bool = False,
) -> SimpleNamespace:
    decision = SimpleNamespace(
        created_at=created_at,
        analysis_type="market",
        was_executed=executed,
        raw_llm_response={
            "authoritative_return_candidate": {
                "side_evidence": {"production_source_count": source_count}
            },
            "paper_bootstrap_canary": {
                "requested": canary,
                "trade_kind": "normal_strategy_trade" if normal_paper else None,
            },
        },
    )
    if canary:
        decision.raw_llm_response["paper_bootstrap_canary"]["runtime_guard"] = {
            "sampling_plan_alert_active": sampling_plan_alert,
        }
    return decision


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


def test_normal_paper_trading_reports_continuous_training_without_sample_target() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    rows = [
        _decision(
            now - timedelta(minutes=2),
            canary=True,
            executed=True,
            normal_paper=True,
        ),
        _decision(now - timedelta(minutes=3)),
    ]

    report = summarize_production_source_health(rows, now=now)

    assert report["recovery_state"] == "paper_normal_trading"
    assert report["paper_normal_executed_count"] == 1
    assert report["continuous_training_after_settlement"] is True
    assert report["sample_target"] is None
    assert report["sampling_plan_alert_active"] is False


def test_unreachable_sampling_plan_is_promoted_to_health_alert() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    rows = [
        _decision(
            now - timedelta(minutes=2),
            canary=True,
            sampling_plan_alert=True,
        ),
        _decision(now - timedelta(minutes=3)),
    ]

    report = summarize_production_source_health(rows, now=now)

    assert report["status"] == "critical"
    assert report["reason"] == "paper_bootstrap_sampling_plan_unreachable"
    assert report["sampling_plan_alert_active"] is True
    assert report["recovery_state"] == "paper_bootstrap_plan_unreachable"
