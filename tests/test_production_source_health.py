from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from services import production_source_health as production_source_health_module
from services.production_source_health import summarize_production_source_health
from tests.normal_paper_test_fixtures import paper_quality_permissions


def _decision(
    created_at: datetime,
    *,
    source_count: int = 0,
    executed: bool = False,
    normal_paper: bool = False,
    quality_gate_reason: str | None = None,
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
                "quant_quality_permissions": paper_quality_permissions(),
            },
        )
    if quality_gate_reason:
        decision.raw_llm_response["paper_trade_selection"] = {
            "selected": False,
            "selected_side": "neutral",
            "selection_reason": "no_direction",
            "by_side": {
                side: {
                    "eligible": False,
                    "selected_side": side,
                    "reason": quality_gate_reason,
                }
                for side in ("long", "short")
            },
        }
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
    assert report["decision_pipeline_active"] is False
    assert report["hard_failure"] is True


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


def test_active_pipeline_without_a_classified_candidate_remains_unresolved() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    rows = [
        _decision(now - timedelta(minutes=2)),
        _decision(now - timedelta(hours=2)),
    ]

    report = summarize_production_source_health(rows, now=now)

    assert report["status"] == "warning"
    assert report["reason"] == "continuous_no_normal_paper_candidate"
    assert report["paper_trade_alert_active"] is True
    assert report["paper_trade_alert_reason"] == "continuous_no_normal_paper_candidate"
    assert report["recovery_state"] == "normal_paper_candidate_waiting"
    assert report["decision_pipeline_active"] is True
    assert report["observing"] is False


def test_active_pipeline_with_positive_return_gate_rejections_is_observing() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    gate_reason = "direction_support_objective_net_not_positive"
    rows = [
        _decision(
            now - timedelta(minutes=2),
            quality_gate_reason=gate_reason,
        ),
        _decision(
            now - timedelta(hours=2),
            quality_gate_reason=gate_reason,
        ),
    ]
    for row in rows:
        for support in row.raw_llm_response["paper_trade_selection"]["by_side"].values():
            support["blocking_reasons"] = [
                gate_reason,
                "direction_support_quant_evidence_missing",
            ]

    report = summarize_production_source_health(rows, now=now)

    assert report["status"] == "warning"
    assert report["reason"] == "normal_paper_quality_gate_waiting"
    assert report["decision_pipeline_active"] is True
    assert report["quality_gated_decision_count"] == 1
    assert report["quality_gate_reason_counts"] == {
        gate_reason: 1,
        "direction_support_quant_evidence_missing": 1,
    }
    assert report["observing"] is True
    assert report["hard_failure"] is False
    assert report["recovery_state"] == "normal_paper_quality_gate_waiting"


def test_missing_quant_evidence_alone_is_not_a_quality_gate_observation() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    rows = [
        _decision(
            now - timedelta(minutes=2),
            quality_gate_reason="direction_support_quant_evidence_missing",
        ),
        _decision(now - timedelta(hours=2)),
    ]

    report = summarize_production_source_health(rows, now=now)

    assert report["reason"] == "continuous_no_normal_paper_candidate"
    assert report["quality_gated_decision_count"] == 0
    assert report["observing"] is False
    assert report["recovery_state"] == "normal_paper_candidate_waiting"


def test_stale_quality_gate_evidence_does_not_hide_pipeline_failure() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    rows = [
        _decision(
            now - timedelta(hours=2),
            quality_gate_reason="direction_support_objective_net_not_positive",
        )
    ]

    report = summarize_production_source_health(rows, now=now)

    assert report["status"] == "warning"
    assert report["reason"] == "market_decision_pipeline_stale"
    assert report["decision_pipeline_active"] is False
    assert report["observing"] is False
    assert report["hard_failure"] is True


def test_missing_market_decisions_remain_a_hard_failure() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)

    report = summarize_production_source_health([], now=now)

    assert report["status"] == "warning"
    assert report["reason"] == "market_decision_evidence_unavailable"
    assert report["decision_pipeline_active"] is False
    assert report["observing"] is False
    assert report["hard_failure"] is True


@pytest.mark.asyncio
async def test_exhaustive_report_paginates_market_decisions_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    first_page = [
        {
            "id": index + 2,
            "created_at": now - timedelta(minutes=index + 1),
            "analysis_type": "market",
            "was_executed": False,
            "raw_llm_response": {},
        }
        for index in range(100)
    ]
    second_page = [
        {
            "id": 1,
            "created_at": now - timedelta(minutes=3),
            "analysis_type": "market",
            "was_executed": False,
            "raw_llm_response": {},
        }
    ]

    class _Result:
        def __init__(self, rows: list[dict[str, object]]) -> None:
            self._rows = rows

        def mappings(self) -> _Result:
            return self

        def all(self) -> list[dict[str, object]]:
            return self._rows

    class _Session:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, _statement: object) -> _Result:
            self.calls += 1
            return _Result(first_page if self.calls == 1 else second_page if self.calls == 2 else [])

    session = _Session()

    @asynccontextmanager
    async def _session_factory():
        yield session

    monkeypatch.setattr(production_source_health_module, "get_read_session_ctx", _session_factory)
    report = await production_source_health_module.ProductionSourceHealthService().report(
        hours=24,
        limit=100,
        exhaustive=True,
    )

    assert report["coverage_complete"] is True
    assert report["coverage"]["truncated"] is False
    assert report["coverage"]["row_count"] == 101
    assert report["coverage"]["page_count"] == 2
