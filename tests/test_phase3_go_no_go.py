from __future__ import annotations

from copy import deepcopy

import pytest

from services.ml_prediction_contract import ENTRY_COHORT_VERSION, PREDICTION_SCORE_POLICY_VERSION
from services.phase3_go_no_go import evaluate_phase3_go_no_go_cards
from services.profit_training_contract import PROFIT_TRAINING_TARGET


def _cards() -> list[dict]:
    return [
        {"key": "okx_trade_fact_integrity", "status": "ok", "details": {}},
        {
            "key": "trade_execution_contract",
            "status": "ok",
            "details": {
                "report_available": True,
                "policy": {
                    "paper_entry_requires_model_promotion": False,
                    "paper_entry_requires_historical_fee_after_quality_permission": True,
                    "paper_normal_entry_requires_positive_return_lcb": True,
                    "paper_validated_entry_requires_positive_return_lcb": True,
                    "paper_downside_return_is_sizing_stress": True,
                    "paper_quality_observation_requires_positive_expected_net_return": True,
                    "paper_quality_observation_requires_positive_return_lcb": False,
                    "paper_quality_observation_allows_non_positive_return_lcb": True,
                    "paper_quality_observation_is_shadow_only": True,
                    "paper_quality_observation_authorizes_normal_entry": False,
                    "paper_training_entry_is_bounded_paper_only": True,
                    "paper_training_entry_authorizes_normal_entry": True,
                    "paper_training_entry_requires_positive_size_aware_net_return": True,
                    "paper_training_entry_requires_positive_return_lcb": True,
                    "paper_training_entry_allows_non_positive_return_lcb": False,
                    "paper_training_entry_requires_finite_downside_stress": True,
                    "paper_training_entry_allows_non_positive_fee_after_return": False,
                    "paper_training_entry_is_shadow_only": False,
                    "paper_training_entry_production_permission": False,
                    "paper_training_entry_single_trade_risk_fraction": 0.0001,
                    "paper_training_entry_portfolio_risk_fraction": 0.0003,
                    "paper_entry_requires_profit_factor": True,
                    "paper_entry_requires_positive_expected_net_return": True,
                    "paper_entry_requires_current_execution_cost": True,
                    "live_entry_requires_production_trade_gate": True,
                    "live_entry_requires_positive_fee_after_return": True,
                    "live_entry_requires_positive_return_lcb": True,
                    "live_entry_requires_current_execution_cost": True,
                    "entry_requires_dynamic_risk_budget": True,
                    "entry_requires_complete_provenance": True,
                    "exit_requires_position_economics": True,
                    "exit_requires_dynamic_close_fraction": True,
                    "filled_order_link_required": True,
                },
                "current_summary": {"contract_violation_count": 0},
            },
        },
        {
            "key": "position_capacity_release",
            "status": "ok",
            "details": {
                "policy": {"strategy_learning_cannot_expand_capacity": True},
                "position_economics_incomplete_count": 0,
                "executed_dynamic_exit_contract_gap_count": 0,
            },
        },
        {
            "key": "model_training",
            "status": "ok",
            "details": {
                "optimization_target": PROFIT_TRAINING_TARGET,
                "live_ml_ready": True,
                "model_critical_items": [],
                "local_ai_tools": {
                    "profit_factor": 1.25,
                    "return_lcb_pct": 0.04,
                    "mean_fee_after_return_pct": 0.08,
                },
            },
        },
        {"key": "phase3_model_server_readiness", "status": "ok", "details": {}},
        {
            "key": "phase3_paper_resume_preflight",
            "status": "ok",
            "details": {
                "status": "ready",
                "can_resume_paper": True,
                "blockers": [],
            },
        },
    ]


def test_phase3_go_no_go_accepts_complete_dynamic_return_architecture() -> None:
    report = evaluate_phase3_go_no_go_cards(_cards())

    assert report["ready"] is True
    assert report["status"] == "go"
    assert report["blockers"] == []
    assert report["policy"]["optimization_target"] == PROFIT_TRAINING_TARGET
    assert report["policy"]["win_rate_is_diagnostic_only"] is True
    assert report["policy"]["expert_memory_strategy_learning_are_observation_only"] is True


def test_phase3_go_no_go_rejects_drifted_training_risk_caps() -> None:
    cards = deepcopy(_cards())
    trade = next(card for card in cards if card["key"] == "trade_execution_contract")
    trade["details"]["policy"]["paper_training_entry_single_trade_risk_fraction"] = (
        0.005
    )

    report = evaluate_phase3_go_no_go_cards(cards)

    blocker = next(
        item
        for item in report["blockers"]
        if item["code"] == "dynamic_return_contract_policy_incomplete"
    )
    assert blocker["evidence"]["invalid_numeric"] == [
        "paper_training_entry_single_trade_risk_fraction"
    ]


def test_phase3_go_no_go_fails_closed_when_required_audit_is_missing() -> None:
    cards = [card for card in _cards() if card["key"] != "trade_execution_contract"]
    report = evaluate_phase3_go_no_go_cards(cards)

    assert report["status"] == "no_go"
    assert "trade_execution_contract_missing" in report["blocker_codes"]


def test_phase3_go_no_go_does_not_turn_deferred_audit_into_missing_architecture() -> None:
    cards = [
        card
        for card in _cards()
        if card["key"] not in {
            "okx_trade_fact_integrity",
            "trade_execution_contract",
            "position_capacity_release",
        }
    ]
    cards.append(
        {
            "key": "system_audit_deferred:test",
            "status": "warning",
            "details": {
                "primary_section_key": "okx_trade_fact_integrity",
                "affected_section_keys": [
                    "okx_trade_fact_integrity",
                    "trade_execution_contract",
                    "position_capacity_release",
                ],
            },
        }
    )

    report = evaluate_phase3_go_no_go_cards(cards)

    assert report["blocker_codes"] == ["required_audits_deferred"]
    assert report["blockers"][0]["evidence"] == {
        "sections": [
            "okx_trade_fact_integrity",
            "position_capacity_release",
            "trade_execution_contract",
        ]
    }


def test_phase3_go_no_go_rejects_incomplete_return_contract_policy() -> None:
    cards = deepcopy(_cards())
    trade = next(card for card in cards if card["key"] == "trade_execution_contract")
    del trade["details"]["policy"]["live_entry_requires_current_execution_cost"]

    report = evaluate_phase3_go_no_go_cards(cards)

    assert "dynamic_return_contract_policy_incomplete" in report["blocker_codes"]


def test_phase3_go_no_go_requires_positive_lcb_for_all_new_paper_entries() -> None:
    for key in (
        "paper_normal_entry_requires_positive_return_lcb",
        "paper_validated_entry_requires_positive_return_lcb",
        "paper_training_entry_requires_positive_return_lcb",
    ):
        cards = deepcopy(_cards())
        trade = next(card for card in cards if card["key"] == "trade_execution_contract")
        trade["details"]["policy"][key] = False

        report = evaluate_phase3_go_no_go_cards(cards)

        blocker = next(
            item
            for item in report["blockers"]
            if item["code"] == "dynamic_return_contract_policy_incomplete"
        )
        assert key in blocker["evidence"]["missing"]


def test_phase3_go_no_go_requires_bounded_training_downside_policy() -> None:
    for key, value in (
        ("paper_training_entry_allows_non_positive_return_lcb", True),
        ("paper_training_entry_requires_finite_downside_stress", False),
    ):
        cards = deepcopy(_cards())
        trade = next(card for card in cards if card["key"] == "trade_execution_contract")
        trade["details"]["policy"][key] = value

        report = evaluate_phase3_go_no_go_cards(cards)

        blocker = next(
            item
            for item in report["blockers"]
            if item["code"] == "dynamic_return_contract_policy_incomplete"
        )
        assert key in blocker["evidence"]["missing"]


def test_phase3_go_no_go_rejects_executed_contract_violations() -> None:
    cards = deepcopy(_cards())
    trade = next(card for card in cards if card["key"] == "trade_execution_contract")
    trade["details"]["current_summary"]["contract_violation_count"] = 1

    report = evaluate_phase3_go_no_go_cards(cards)

    assert "dynamic_return_contract_current_violations" in report["blocker_codes"]


@pytest.mark.parametrize("net_pnl", [-0.7046, 0.0])
def test_shadow_profit_cannot_override_current_authoritative_losses(net_pnl: float) -> None:
    cards = deepcopy(_cards())
    trade = next(card for card in cards if card["key"] == "trade_execution_contract")
    trade["details"]["current_summary"].update(
        realized_pnl_source="okx_position_history_realized_pnl",
        closed_position_count=12,
        realized_net_pnl_usdt=net_pnl,
    )

    report = evaluate_phase3_go_no_go_cards(cards)

    assert report["ready"] is False
    assert "current_authoritative_return_not_positive" in report["blocker_codes"]


def test_local_pnl_does_not_override_authoritative_profitability_audit() -> None:
    cards = deepcopy(_cards())
    trade = next(card for card in cards if card["key"] == "trade_execution_contract")
    trade["details"]["current_summary"].update(
        realized_pnl_source="local_position_realized_pnl_test_fallback",
        closed_position_count=12,
        realized_net_pnl_usdt=-100.0,
    )

    report = evaluate_phase3_go_no_go_cards(cards)

    assert "current_authoritative_return_not_positive" not in report["blocker_codes"]


def test_positive_current_pnl_does_not_override_model_profitability_failure() -> None:
    cards = deepcopy(_cards())
    trade = next(card for card in cards if card["key"] == "trade_execution_contract")
    trade["details"]["current_summary"].update(
        realized_pnl_source="okx_position_history_realized_pnl",
        closed_position_count=12,
        realized_net_pnl_usdt=1.0,
    )
    training = next(card for card in cards if card["key"] == "model_training")
    training["details"]["live_ml_ready"] = False

    report = evaluate_phase3_go_no_go_cards(cards)

    assert report["ready"] is False
    assert "local_ml_not_live_ready" in report["blocker_codes"]


def test_phase3_go_no_go_keeps_unavailable_contract_audit_as_one_blocker() -> None:
    cards = deepcopy(_cards())
    trade = next(card for card in cards if card["key"] == "trade_execution_contract")
    trade["status"] = "warning"
    trade["details"] = {
        "report_available": False,
        "error": "isolated report exceeded 20 seconds",
        "error_type": "TimeoutError",
        "timeout": True,
    }

    report = evaluate_phase3_go_no_go_cards(cards)

    assert report["ready"] is False
    assert report["blocker_codes"] == ["dynamic_return_contract_unavailable"]
    assert "dynamic_return_contract_policy_incomplete" not in report["blocker_codes"]
    assert "dynamic_return_contract_current_violations" not in report["blocker_codes"]


def test_phase3_go_no_go_rejects_position_economics_or_exit_gaps() -> None:
    cards = deepcopy(_cards())
    capacity = next(card for card in cards if card["key"] == "position_capacity_release")
    capacity["details"]["position_economics_incomplete_count"] = 2
    capacity["details"]["executed_dynamic_exit_contract_gap_count"] = 1

    report = evaluate_phase3_go_no_go_cards(cards)

    assert "position_economics_or_exit_contract_gap" in report["blocker_codes"]


def test_phase3_go_no_go_rejects_non_return_training_objective() -> None:
    cards = deepcopy(_cards())
    training = next(card for card in cards if card["key"] == "model_training")
    training["details"]["optimization_target"] = "classification_accuracy"

    report = evaluate_phase3_go_no_go_cards(cards)

    assert "model_training_objective_mismatch" in report["blocker_codes"]


def test_phase3_go_no_go_rejects_local_ml_without_production_readiness() -> None:
    cards = deepcopy(_cards())
    training = next(card for card in cards if card["key"] == "model_training")
    training["details"]["live_ml_ready"] = False

    report = evaluate_phase3_go_no_go_cards(cards)

    assert report["status"] == "no_go"
    assert "local_ml_not_live_ready" in report["blocker_codes"]


def test_phase3_go_no_go_rejects_training_resource_failures() -> None:
    cards = deepcopy(_cards())
    training = next(card for card in cards if card["key"] == "model_training")
    training["details"]["model_critical_items"] = [
        {"model_id": "profit", "error": "MemoryError"}
    ]

    report = evaluate_phase3_go_no_go_cards(cards)

    assert "model_training_resource_failure" in report["blocker_codes"]


def test_phase3_go_no_go_rejects_unprofitable_model_metrics() -> None:
    cards = deepcopy(_cards())
    training = next(card for card in cards if card["key"] == "model_training")
    training["details"]["local_ai_tools"] = {
        "profit_factor": 0.32,
        "return_lcb_pct": -0.13,
        "mean_fee_after_return_pct": -0.54,
    }

    report = evaluate_phase3_go_no_go_cards(cards)

    assert {
        "model_profit_factor_below_unity",
        "model_return_lcb_not_positive",
        "model_fee_after_return_not_positive",
    }.issubset(report["blocker_codes"])


def _local_ml_evidence(*, long_pf: float, short_pf: float) -> dict:
    return {
        "model_id": "local_ml_profit_quality",
        "available": True,
        "live_ml_ready": False,
        "walk_forward_status": "complete",
        "walk_forward_stable": False,
        "fallback_used": True,
        "entry_cohort_version": ENTRY_COHORT_VERSION,
        "prediction_score_policy_version": PREDICTION_SCORE_POLICY_VERSION,
        "evidence_scope": "holdout_positive_objective_entry_cohort",
        "portfolio": {
            "count": 8, "profit_factor": min(long_pf, short_pf),
            "return_lcb_pct": -3.9919, "mean_return_pct": -1.3916,
        },
        "sides": {
            "long": {
                "profit_factor": long_pf,
                "return_lcb_pct": -2.0421,
                "mean_return_pct": -1.0188,
            },
            "short": {
                "profit_factor": short_pf,
                "return_lcb_pct": -3.9919,
                "mean_return_pct": -1.3916,
            },
        },
    }


def test_phase3_go_no_go_uses_current_local_ml_evidence_over_legacy_bundle() -> None:
    cards = deepcopy(_cards())
    training = next(card for card in cards if card["key"] == "model_training")
    training["details"]["live_ml_ready"] = True
    training["details"]["local_ai_tools"] = {
        "live_ml_ready": True,
        "profit_factor": 1.42,
        "return_lcb_pct": 0.31,
        "mean_fee_after_return_pct": 0.26,
    }
    training["details"]["local_ml_profit_quality"] = _local_ml_evidence(
        long_pf=0.2939,
        short_pf=0.6222,
    )

    report = evaluate_phase3_go_no_go_cards(cards)

    assert report["status"] == "no_go"
    assert report["summary"]["model_profitability_source"] == "local_ml_profit_quality"
    assert report["summary"]["model_profit_factor"] == 0.2939
    blocker = next(
        item
        for item in report["blockers"]
        if item["code"] == "model_profit_factor_below_unity"
    )
    assert blocker["evidence"]["source"] == "local_ml_profit_quality"
    assert report["summary"]["model_profitability_sides"]["short"]["profit_factor"] == 0.6222


@pytest.mark.parametrize("policy", [None, "legacy-tree-mean-only"])
def test_phase3_go_no_go_rejects_profitable_metrics_from_other_scoring_policy(policy) -> None:
    from services.phase3_go_no_go import _model_profitability_evidence

    local_ml = _local_ml_evidence(long_pf=2.0, short_pf=2.0)
    local_ml["prediction_score_policy_version"] = policy
    local_ml["portfolio"].update({"return_lcb_pct": 0.2, "mean_return_pct": 0.4})
    evidence, source = _model_profitability_evidence({
        "local_ml_profit_quality": local_ml,
        "local_ai_tools": {"profit_factor": 99.0, "return_lcb_pct": 10.0},
    })
    assert source == "local_ml_profit_quality"
    assert evidence["count"] == 0
    assert evidence["profit_factor"] is None
    assert evidence["mean_return_pct"] is None
    assert evidence["return_lcb_pct"] is None


def test_phase3_go_no_go_local_ml_live_flag_overrides_legacy_promotion() -> None:
    cards = deepcopy(_cards())
    training = next(card for card in cards if card["key"] == "model_training")
    training["details"]["live_ml_ready"] = True
    training["details"]["local_ai_tools"] = {"live_ml_ready": True}
    training["details"]["local_ml_profit_quality"] = _local_ml_evidence(
        long_pf=0.9,
        short_pf=0.95,
    )

    report = evaluate_phase3_go_no_go_cards(cards)

    assert "local_ml_not_live_ready" in report["blocker_codes"]


def test_phase3_go_no_go_blocks_when_local_ml_metrics_are_missing() -> None:
    cards = deepcopy(_cards())
    training = next(card for card in cards if card["key"] == "model_training")
    training["details"]["local_ml_profit_quality"] = {
        "model_id": "local_ml_profit_quality",
        "available": True,
        "live_ml_ready": False,
        "sides": {},
    }

    report = evaluate_phase3_go_no_go_cards(cards)

    assert {
        "model_profit_factor_evidence_missing",
        "model_return_lcb_evidence_missing",
        "model_fee_after_return_evidence_missing",
    }.issubset(report["blocker_codes"])


@pytest.mark.parametrize("mode", ["missing_version", "empty_portfolio", "losing_portfolio"])
def test_go_no_go_cannot_replace_entry_portfolio_with_profitable_side_or_rankings(mode) -> None:
    cards = deepcopy(_cards())
    training = next(card for card in cards if card["key"] == "model_training")
    evidence = _local_ml_evidence(long_pf=10.0, short_pf=20.0)
    evidence["live_ml_ready"] = True
    evidence["portfolio"].update({
        "profit_factor": 0.5, "return_lcb_pct": -0.2, "mean_return_pct": -0.1,
    })
    if mode == "missing_version":
        evidence.pop("entry_cohort_version")
    elif mode == "empty_portfolio":
        evidence["portfolio"]["count"] = 0
    training["details"]["local_ml_profit_quality"] = evidence
    report = evaluate_phase3_go_no_go_cards(cards)
    assert report["status"] == "no_go"
    assert report["summary"]["model_profit_factor"] == (0.5 if mode == "losing_portfolio" else None)


def test_phase3_go_no_go_rejects_paper_resume_without_operator_safe_preflight() -> None:
    cards = deepcopy(_cards())
    preflight = next(
        card for card in cards if card["key"] == "phase3_paper_resume_preflight"
    )
    preflight["status"] = "warning"
    preflight["details"] = {
        "status": "blocked",
        "can_resume_paper": False,
        "blockers": [{"code": "okx_authoritative_sync_not_clean"}],
    }

    report = evaluate_phase3_go_no_go_cards(cards)

    assert report["ready"] is False
    assert report["status"] == "no_go"
    assert "paper_resume_preflight_not_ready" in report["blocker_codes"]


def test_phase3_go_no_go_accepts_consumed_paper_resume_preflight() -> None:
    cards = deepcopy(_cards())
    preflight = next(
        card for card in cards if card["key"] == "phase3_paper_resume_preflight"
    )
    preflight["status"] = "ok"
    preflight["details"] = {
        "status": "running",
        "can_resume_paper": False,
        "consumed_after_resume": True,
        "blockers": [{"code": "paper_trading_already_active"}],
        "effective_blockers": [],
    }

    report = evaluate_phase3_go_no_go_cards(cards)

    assert report["ready"] is True
    assert report["status"] == "go"
    assert "paper_resume_preflight_not_ready" not in report["blocker_codes"]


def test_phase3_go_no_go_rejects_consumed_preflight_with_effective_blocker() -> None:
    cards = deepcopy(_cards())
    preflight = next(
        card for card in cards if card["key"] == "phase3_paper_resume_preflight"
    )
    preflight["status"] = "warning"
    preflight["details"] = {
        "status": "running",
        "can_resume_paper": False,
        "consumed_after_resume": True,
        "blockers": [{"code": "okx_authoritative_sync_not_clean"}],
        "effective_blockers": [{"code": "okx_authoritative_sync_not_clean"}],
    }

    report = evaluate_phase3_go_no_go_cards(cards)

    assert report["ready"] is False
    assert report["status"] == "no_go"
    assert "paper_resume_preflight_not_ready" in report["blocker_codes"]


def test_phase3_go_no_go_keeps_warning_observable_without_hard_threshold() -> None:
    cards = deepcopy(_cards())
    training = next(card for card in cards if card["key"] == "model_training")
    training["status"] = "warning"

    report = evaluate_phase3_go_no_go_cards(cards)

    assert report["ready"] is True
    assert report["status"] == "go"
    assert "model_training_warning" in report["warning_codes"]
