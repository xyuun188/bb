from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from services.production_source_health import summarize_production_source_health


def _decision(
    created_at: datetime,
    *,
    source_count: int = 0,
    executed: bool = False,
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
        },
    )
    if normal_paper:
        from services.normal_paper_trade import build_normal_paper_trade_contract

        decision.raw_llm_response["normal_paper_trade"] = build_normal_paper_trade_contract(
            symbol="BTC/USDT",
            side="long",
            selection_reason="strategy_edge_selected",
            direction_support={
                "eligible": True,
                "selected_side": "long",
                "prediction_horizon_minutes": 15,
                "expected_net_return_pct": 0.1,
                "objective_net_return_pct": 0.05,
            },
        )
    return decision


def test_continuous_no_production_source_raises_critical_alert() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    rows = [_decision(now - timedelta(hours=2, minutes=index)) for index in range(20)]

    report = summarize_production_source_health(rows, now=now, production_permission=True)

    assert report["status"] == "critical"
    assert report["alert_active"] is True
    assert report["reason"] == "continuous_no_production_return_source"


def test_recent_production_source_clears_alert() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    rows = [
        _decision(now - timedelta(minutes=2), source_count=1),
        _decision(now - timedelta(minutes=3)),
    ]

    report = summarize_production_source_health(rows, now=now, production_permission=True)

    assert report["status"] == "ok"
    assert report["alert_active"] is False


def test_old_bootstrap_contract_does_not_count_as_normal_paper_activity() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    row = _decision(now - timedelta(hours=2), executed=True)
    row.raw_llm_response["paper_bootstrap_canary"] = {"authorized": True}
    rows = [row, _decision(now - timedelta(hours=3))]

    report = summarize_production_source_health(rows, now=now)

    assert report["status"] == "warning"
    assert report["recovery_state"] == "normal_paper_candidate_waiting"
    assert report["normal_paper_executed_count"] == 0
    assert report["paper_trade_alert_active"] is True


def test_normal_paper_trading_reports_continuous_training_without_sample_target() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    rows = [
        _decision(
            now - timedelta(minutes=2),
            executed=True,
            normal_paper=True,
        ),
        _decision(now - timedelta(minutes=3)),
    ]

    report = summarize_production_source_health(rows, now=now)

    assert report["recovery_state"] == "normal_paper_trading"
    assert report["normal_paper_executed_count"] == 1
    assert report["continuous_training_after_settlement"] is True
    assert report["paper_trade_alert_active"] is False
    assert report["paper_trade_status"] == "active"
    assert report["status"] == "ok"


def test_continuous_no_normal_paper_candidate_is_reported_separately() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    rows = [
        _decision(now - timedelta(hours=2)),
        _decision(now - timedelta(hours=3)),
    ]

    report = summarize_production_source_health(rows, now=now)

    assert report["status"] == "warning"
    assert report["reason"] == "continuous_no_normal_paper_candidate"
    assert report["paper_trade_alert_active"] is True
    assert report["paper_trade_alert_reason"] == "continuous_no_normal_paper_candidate"
    assert report["recovery_state"] == "normal_paper_candidate_waiting"
