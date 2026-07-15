from datetime import UTC, datetime
from types import SimpleNamespace

from services.strategy_signal_root_cause_audit import StrategySignalRootCauseAuditService


def _provenance() -> dict[str, object]:
    return {
        "source": "test_distribution",
        "observation_window": "test_window",
        "sample_count": 3,
        "generated_at": "2026-07-12T00:00:00+00:00",
        "strategy_version": "test-v1",
        "fallback_reason": "",
    }


def _decision(*, complete: bool) -> SimpleNamespace:
    provenance = _provenance()
    return SimpleNamespace(
        action="long",
        symbol="BTC/USDT",
        created_at=datetime(2026, 7, 12, tzinfo=UTC),
        raw_llm_response={
            "production_return_policy": {
                "eligible": complete,
                "expected_net_return_pct": 0.8,
                "return_lcb_pct": 0.2,
                "production_source_count": 3,
                "position_size_pct": 0.12,
                "policy_provenance": provenance,
            },
            "opportunity_score": {
                "production_eligible": complete,
                "policy_provenance": provenance,
                "execution_cost": {
                    "production_eligible": complete,
                    "total_pct": 0.08,
                    "policy_provenance": provenance,
                },
            },
            "profit_risk_sizing": {
                "production_eligible": complete,
                "risk_budget_usdt": 3.0,
                "planned_stressed_loss_usdt": 2.4,
                "stressed_loss_fraction": 0.02,
                "target_notional_usdt": 150.0,
                "final_notional_usdt": 120.0,
                "policy_provenance": provenance,
            },
        },
    )


def test_complete_return_contract_has_no_root_cause() -> None:
    report = StrategySignalRootCauseAuditService().summarize(
        decisions=[_decision(complete=True)],
        shadows=[],
        ml_status={"status": "ready", "allow_live_position_influence": True},
    )

    assert report["status"] == "ok"
    assert report["production_return_ready_count"] == 1
    assert report["root_causes"] == []


def test_incomplete_cost_and_risk_budget_are_reported_without_mutation() -> None:
    decision = _decision(complete=True)
    decision.raw_llm_response["opportunity_score"]["execution_cost"] = {}
    decision.raw_llm_response["profit_risk_sizing"] = {}
    report = StrategySignalRootCauseAuditService().summarize(
        decisions=[decision],
        shadows=[SimpleNamespace(missed_opportunity=True)],
        ml_status={"status": "degraded", "allow_live_position_influence": False},
    )

    assert report["status"] == "warning"
    assert report["can_force_open"] is False
    assert report["can_override_thresholds"] is False
    assert report["contract_blocker_counts"]["live_execution_cost_incomplete"] == 1
    assert report["contract_blocker_counts"]["dynamic_risk_budget_ineligible"] == 1
    assert report["shadow_missed_opportunity"]["can_authorize_entry"] is False


def test_positive_expectation_without_provenance_fails_closed() -> None:
    decision = _decision(complete=True)
    decision.raw_llm_response["production_return_policy"]["policy_provenance"] = {}
    report = StrategySignalRootCauseAuditService().summarize(
        decisions=[decision], shadows=[], ml_status={}
    )

    assert report["production_return_ready_count"] == 0
    assert report["contract_blocker_counts"]["production_return_provenance_incomplete"] == 1
