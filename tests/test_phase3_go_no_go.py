from __future__ import annotations

from copy import deepcopy

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
                    "entry_requires_positive_fee_after_return": True,
                    "entry_requires_positive_return_lcb": True,
                    "entry_requires_live_execution_cost": True,
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
            "details": {"optimization_target": PROFIT_TRAINING_TARGET},
        },
        {"key": "phase3_model_server_readiness", "status": "ok", "details": {}},
    ]


def test_phase3_go_no_go_accepts_complete_dynamic_return_architecture() -> None:
    report = evaluate_phase3_go_no_go_cards(_cards())

    assert report["ready"] is True
    assert report["status"] == "go"
    assert report["blockers"] == []
    assert report["policy"]["optimization_target"] == PROFIT_TRAINING_TARGET
    assert report["policy"]["win_rate_is_diagnostic_only"] is True
    assert report["policy"]["expert_memory_strategy_learning_are_observation_only"] is True


def test_phase3_go_no_go_fails_closed_when_required_audit_is_missing() -> None:
    cards = [card for card in _cards() if card["key"] != "trade_execution_contract"]
    report = evaluate_phase3_go_no_go_cards(cards)

    assert report["status"] == "no_go"
    assert "trade_execution_contract_missing" in report["blocker_codes"]


def test_phase3_go_no_go_rejects_incomplete_return_contract_policy() -> None:
    cards = deepcopy(_cards())
    trade = next(card for card in cards if card["key"] == "trade_execution_contract")
    del trade["details"]["policy"]["entry_requires_live_execution_cost"]

    report = evaluate_phase3_go_no_go_cards(cards)

    assert "dynamic_return_contract_policy_incomplete" in report["blocker_codes"]


def test_phase3_go_no_go_rejects_executed_contract_violations() -> None:
    cards = deepcopy(_cards())
    trade = next(card for card in cards if card["key"] == "trade_execution_contract")
    trade["details"]["current_summary"]["contract_violation_count"] = 1

    report = evaluate_phase3_go_no_go_cards(cards)

    assert "dynamic_return_contract_current_violations" in report["blocker_codes"]


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


def test_phase3_go_no_go_keeps_warning_observable_without_hard_threshold() -> None:
    cards = deepcopy(_cards())
    training = next(card for card in cards if card["key"] == "model_training")
    training["status"] = "warning"

    report = evaluate_phase3_go_no_go_cards(cards)

    assert report["ready"] is True
    assert report["status"] == "go"
    assert "model_training_warning" in report["warning_codes"]
