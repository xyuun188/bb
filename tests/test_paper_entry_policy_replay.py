from copy import deepcopy

from scripts.audit_online_paper_entry_policy import replay_policy
from tests.normal_paper_test_fixtures import paper_quality_permissions


def _row() -> dict:
    return {
        "id": 804049, "symbol": "ALGO/USDT", "was_executed": False,
        "selection": {
            "selected": False,
            "by_side": {
                "short": {
                    "eligible": True, "selected_side": "short",
                    "current_edge_validated": True,
                    "execution_cost_complete": True,
                    "expected_net_return_pct": 0.2665481857,
                    "raw_expected_return_pct": 0.3839841857,
                    "objective_net_return_pct": -0.8251118651,
                    "prediction_horizon_minutes": 30,
                    "loss_probability": 0.549535456,
                    "quant_evidence_families": ["local_ml"],
                    "quant_quality_permissions": paper_quality_permissions(),
                    "blocking_reasons": ["direction_support_risk_adjusted_return_not_positive"],
                },
            },
        },
    }


def test_replay_uses_new_training_policy_without_mutating_historical_evidence() -> None:
    row = _row()
    original = deepcopy(row)
    report = replay_policy([row])
    assert report["counts"]["newly_selection_contract_ready"] == 1
    assert report["counts"].get("contract_rejected", 0) == 0
    assert report["ready_sides"] == {"short": 1}
    assert report["examples"][0]["selection_reason"] == "paper_training_entry"
    assert report["examples"][0]["production_permission"] is False
    assert report["submits_orders"] is False
    assert report["counts"]["side_loss_guard_context_missing"] == 1
    assert row == original


def test_replay_preserves_upstream_blockers() -> None:
    row = _row()
    row["selection"]["by_side"]["short"]["blocking_reasons"].append(
        "direction_support_execution_cost_incomplete",
    )
    report = replay_policy([row])
    assert report["counts"]["still_blocked"] == 1
    assert report["counts"].get("newly_selection_contract_ready", 0) == 0
    assert report["blocking_reasons"]["direction_support_execution_cost_incomplete"] == 1


def test_replay_uses_settled_side_loss_guard_without_changing_source_evidence() -> None:
    row = _row()
    row["strategy_mode"] = {
        "strategy_context_performance": {
            "available": True, "status": "fresh",
            "source": "background_performance_snapshot", "failed_labels": [],
        },
        "symbol_side_performance": {
            "ALGO/USDT|short": {
                "count": 8, "profit": 1.0, "loss": 5.0, "pnl": -4.0,
                "profile_scope": "symbol_side",
                "policy_provenance": {
                    "source": "trusted_closed_position_fee_after_pnl_distribution",
                    "sample_count": 8,
                },
            },
        },
    }
    original = deepcopy(row)
    report = replay_policy([row])
    assert report["counts"]["still_blocked"] == 1
    assert report["counts"]["side_loss_guard_snapshots_losing"] == 1
    assert report["blocking_reasons"]["direction_support_recent_side_loss"] == 1
    assert report["ready_sides"] == {}
    assert row == original


def test_replay_reports_guard_saved_with_support_when_full_context_was_not_persisted() -> None:
    row = _row()
    row["selection"]["by_side"]["short"]["recent_side_loss_guard"] = {
        "enabled": True, "blocked": True, "symbol": "ALGO/USDT", "side": "short",
    }
    report = replay_policy([row])
    assert report["counts"]["side_loss_guard_snapshots_available"] == 1
    assert report["counts"]["side_loss_guard_snapshots_losing"] == 1
    assert report["counts"].get("side_loss_guard_context_missing", 0) == 0
    assert report["counts"]["still_blocked"] == 1
