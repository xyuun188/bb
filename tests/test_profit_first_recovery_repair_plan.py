from __future__ import annotations

from services.profit_first_recovery_repair_plan import build_profit_first_recovery_repair_plan


def test_recovery_repair_plan_maps_blockers_to_operator_actions() -> None:
    report = build_profit_first_recovery_repair_plan(
        {
            "status": "blocked",
            "resume_clear": False,
            "blocking_item_count": 4,
            "warning_item_count": 1,
            "items": [
                {
                    "category": "trade_contract",
                    "severity": "blocking",
                    "code": "missing_profit_first_trade_plan",
                    "count": 2,
                    "samples": [
                        {
                            "decision_id": 9549,
                            "symbol": "AVAX/USDT",
                            "action": "short",
                        },
                        {
                            "decision_id": 9300,
                            "symbol": "MSFT/USDT",
                            "action": "short",
                        },
                    ],
                },
                {
                    "category": "trade_contract",
                    "severity": "blocking",
                    "code": "missing_profit_first_position_ladder",
                    "count": 2,
                    "samples": [{"decision_id": 9549}, {"decision_id": 9300}],
                },
                {
                    "category": "trade_contract",
                    "severity": "blocking",
                    "code": "missing_profit_first_exit_plan_reference",
                    "count": 1,
                    "samples": [{"decision_id": 10391, "symbol": "ARB/USDT"}],
                },
                {
                    "category": "ranking",
                    "severity": "blocking",
                    "code": "strategy_disable",
                    "model_name": "ensemble_trader",
                    "strategy_profile_id": "candidate_2",
                    "symbol": "AVAX/USDT",
                    "side": "short",
                    "decision_lane": "unknown",
                    "ranking_reasons": ["consecutive_losses"],
                },
                {
                    "category": "okx_reconciliation",
                    "severity": "blocking",
                    "code": "local_order_quantity_differs_from_okx_fill",
                    "symbol": "LAB/USDT",
                    "local_order_id": 2678,
                    "exchange_order_id": "3697115296869093376",
                    "classification": "repairable",
                },
            ],
        }
    )

    assert report["dry_run"] is True
    assert report["read_only"] is True
    assert report["mutates_database"] is False
    assert report["submits_orders"] is False
    assert report["changes_model_routing"] is False
    assert report["changes_live_sizing"] is False
    assert report["resume_allowed_by_this_plan"] is False
    assert report["status"] == "blocked"
    assert report["summary"]["blocking_action_count"] == 5
    assert report["summary"]["operator_approval_required_count"] == 5
    assert len(report["blocking_actions"]) == 5

    actions = {action["code"]: action for action in report["actions"]}
    assert actions["missing_profit_first_trade_plan"]["action_type"] == (
        "historical_trade_plan_backfill_or_quarantine"
    )
    assert actions["missing_profit_first_trade_plan"]["target"]["decision_ids"] == [9549, 9300]
    assert actions["missing_profit_first_position_ladder"]["action_type"] == (
        "historical_position_ladder_backfill_or_quarantine"
    )
    assert actions["missing_profit_first_exit_plan_reference"]["action_type"] == (
        "exit_reference_repair_or_legacy_failure_marker"
    )
    assert actions["strategy_disable"]["action_type"] == "ranking_shadow_disable_review"
    assert actions["local_order_quantity_differs_from_okx_fill"]["action_type"] == (
        "okx_exact_order_quantity_repair_review"
    )
    assert actions["local_order_quantity_differs_from_okx_fill"]["target"]["order_ids"] == [2678]
    assert actions["local_order_quantity_differs_from_okx_fill"]["target"][
        "exchange_order_ids"
    ] == ["3697115296869093376"]
    assert any(
        "repair_okx_history_position_reconciliation.py" in command
        for command in actions["local_order_quantity_differs_from_okx_fill"][
            "suggested_commands"
        ]
    )


def test_recovery_repair_plan_clear_when_no_blockers() -> None:
    report = build_profit_first_recovery_repair_plan(
        {
            "status": "ready",
            "resume_clear": True,
            "blocking_item_count": 0,
            "items": [],
        }
    )

    assert report["status"] == "clear"
    assert report["summary"]["action_count"] == 0
    assert report["summary"]["blocking_action_count"] == 0
    assert report["resume_allowed_by_this_plan"] is False
    assert report["starts_trading_service"] is False
