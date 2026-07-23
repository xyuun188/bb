from datetime import UTC, datetime
from types import SimpleNamespace

from services.strategy_signal_root_cause_audit import StrategySignalRootCauseAuditService
from services.trade_execution_contract import build_live_rules_canary_entry_contract
from tests.paper_canary_fixtures import complete_paper_canary_raw


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
            "live_ml_profit_contract": {
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
    assert report["live_ml_ready_count"] == 1
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
    decision.raw_llm_response["live_ml_profit_contract"]["policy_provenance"] = {}
    report = StrategySignalRootCauseAuditService().summarize(
        decisions=[decision], shadows=[], ml_status={}
    )

    assert report["live_ml_ready_count"] == 0
    assert (
        report["contract_blocker_counts"][
            "live_ml_profit_contract_provenance_incomplete"
        ]
        == 1
    )


def _live_rules_canary_decision() -> SimpleNamespace:
    provenance = _provenance()
    raw = {
        "production_trade_gate": {
            "version": "test-gate",
            "mode": "live_rules_canary",
            "can_trade": True,
            "decision_authority": "rules",
            "model_can_influence": False,
            "risk": {"max_notional_usdt": 10.0},
        },
        "live_rules_canary_signal": {
            "version": "test-rules-canary-signal",
            "execution_scope": "live_rules_canary",
            "decision_authority": "rules",
            "model_can_influence": False,
            "production_eligible": True,
            "action": "long",
            "policy_provenance": provenance,
        },
        "model_shadow_decision": {
            "action": "short",
            "observation_only": True,
            "can_authorize_entry": False,
            "can_change_size_or_leverage": False,
        },
        "opportunity_score": {
            "production_eligible": False,
            "execution_cost": {
                "production_eligible": True,
                "order_size_complete": True,
                "order_notional_usdt": 8.0,
                "total_pct": 0.08,
                "policy_provenance": provenance,
            },
        },
        "profit_risk_sizing": {
            "contract_version": "test-rules-canary-sizing",
            "contract_lifecycle": "live_rules_canary",
            "execution_scope": "live_rules_canary",
            "production_permission": True,
            "decision_authority": "rules",
            "model_can_influence": False,
            "production_eligible": True,
            "risk_budget_usdt": 0.5,
            "planned_stressed_loss_usdt": 0.16,
            "stressed_loss_fraction": 0.02,
            "target_notional_usdt": 8.0,
            "target_inst_id": "BTC-USDT-SWAP",
            "target_price": 100.0,
            "selected_contract_spec": {
                "ctVal": "0.01",
                "ctMult": "1",
                "minSz": "1",
                "lotSz": "1",
            },
            "exchange_minimum_order": {
                "production_eligible": True,
                "minimum_notional_usdt": 1.0,
            },
            "exchange_min_notional_usdt": 1.0,
            "final_notional_usdt": 8.0,
            "final_margin_usdt": 8.0,
            "final_leverage": 1.0,
            "leverage_tier_selection": {"production_eligible": True},
            "policy_provenance": provenance,
        },
    }
    contract, reasons = build_live_rules_canary_entry_contract(raw)
    assert reasons == []
    raw["live_rules_canary_contract"] = contract
    return SimpleNamespace(
        action="long",
        symbol="BTC/USDT",
        created_at=datetime(2026, 7, 23, tzinfo=UTC),
        raw_llm_response=raw,
    )


def test_complete_paper_canary_is_a_ready_entry_not_a_production_gap() -> None:
    decision = SimpleNamespace(
        action="long",
        symbol="BTC/USDT",
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
        raw_llm_response=complete_paper_canary_raw(),
    )

    report = StrategySignalRootCauseAuditService().summarize(
        decisions=[decision],
        shadows=[],
        ml_status={"status": "canary"},
    )

    assert report["status"] == "ok"
    assert report["entry_contract_ready_count"] == 1
    assert report["paper_canary_ready_count"] == 1
    assert report["live_ml_ready_count"] == 0
    assert report["contract_blocker_counts"] == {}


def test_complete_live_rules_canary_is_reported_in_its_own_lifecycle() -> None:
    report = StrategySignalRootCauseAuditService().summarize(
        decisions=[_live_rules_canary_decision()],
        shadows=[],
        ml_status={"status": "shadow", "allow_live_position_influence": False},
    )

    assert report["status"] == "ok"
    assert report["entry_contract_ready_count"] == 1
    assert report["live_rules_canary_entry_count"] == 1
    assert report["live_rules_canary_ready_count"] == 1
    assert report["live_rules_canary_blocked_count"] == 0
    assert report["live_ml_ready_count"] == 0
    assert report["contract_blocker_counts"] == {}
