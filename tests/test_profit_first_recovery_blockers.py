from __future__ import annotations

from services.profit_first_recovery_blockers import build_profit_first_recovery_blockers


def test_recovery_blockers_summarize_contract_ranking_and_okx_items() -> None:
    report = build_profit_first_recovery_blockers(
        trade_contract={
            "current_summary": {
                "profit_first_plan_missing_count": 2,
                "profit_first_position_ladder_missing_count": 2,
                "exit_plan_reference_missing_count": 1,
            },
            "current_violations": [
                {
                    "decision_id": 9300,
                    "symbol": "MSFT/USDT",
                    "action": "short",
                    "reason": "missing_profit_first_trade_plan",
                },
                {
                    "decision_id": 9626,
                    "symbol": "MSFT/USDT",
                    "action": "close_short",
                    "reason": "missing_profit_first_exit_plan_reference",
                },
            ],
        },
        ranking={
            "blockers": [
                {
                    "code": "strategy_disable",
                    "severity": "blocking",
                    "message": "disable this lane",
                    "evidence": {
                        "model_name": "ensemble_trader",
                        "strategy_profile_id": "candidate_2",
                        "symbol": "AVAX/USDT",
                        "side": "short",
                        "decision_lane": "unknown",
                        "realized_net_pnl": -1.23,
                        "ranking_reasons": ["consecutive_losses"],
                    },
                },
                {
                    "code": "strategy_demote",
                    "severity": "warning",
                    "evidence": {"symbol": "ALGO/USDT", "ranking_reasons": ["negative_pnl"]},
                },
            ]
        },
        observation={
            "blockers": [
                {
                    "code": "okx_authoritative_sync_has_post_resume_differences",
                    "message": "OKX/local differences appeared",
                    "evidence": {
                        "issues": [
                            {
                                "kind": "local_order_quantity_differs_from_okx_fill",
                                "symbol": "LAB/USDT",
                                "local_order_id": 2678,
                                "exchange_order_id": "3697115296869093376",
                                "classification": "repairable",
                                "reason": "contract conversion mismatch",
                            }
                        ]
                    },
                }
            ]
        },
    )

    assert report["read_only"] is True
    assert report["resume_clear"] is False
    assert report["blocking_item_count"] == 4
    assert report["category_counts"]["trade_contract"] == 3
    assert report["category_counts"]["ranking"] == 2
    assert report["category_counts"]["okx_reconciliation"] == 1
    assert report["items"][0]["samples"][0]["decision_id"] == 9300
    assert any(item["code"] == "strategy_disable" for item in report["items"])
    disabled_lane = next(item for item in report["items"] if item["code"] == "strategy_disable")
    assert disabled_lane["severity"] == "warning"
    assert disabled_lane["lane_scoped_containment"] is True
    assert any(item.get("symbol") == "LAB/USDT" for item in report["items"])


def test_recovery_blockers_ready_when_no_items() -> None:
    report = build_profit_first_recovery_blockers(
        trade_contract={"current_summary": {}},
        ranking={"blockers": []},
        observation={"blockers": []},
    )

    assert report["status"] == "ready"
    assert report["resume_clear"] is True
    assert report["blocking_item_count"] == 0
    assert report["starts_trading_service"] is False
    assert report["submits_orders"] is False


def test_recovery_blockers_warn_for_already_quarantined_historical_contract_rows() -> None:
    report = build_profit_first_recovery_blockers(
        trade_contract={
            "current_summary": {
                "profit_first_plan_missing_count": 0,
                "profit_first_plan_missing_count_unresolved": 0,
                "historical_recovery_quarantined_profit_first_plan_missing_count": 3,
                "historical_recovery_quarantined_violation_count": 3,
            },
            "historical_recovery_quarantined_violations": [
                {
                    "decision_id": 11444,
                    "symbol": "XPL/USDT",
                    "action": "short",
                    "reason": "missing_profit_first_trade_plan",
                }
            ],
        },
        ranking={"blockers": []},
        observation={"blockers": []},
    )

    assert report["status"] == "ready"
    assert report["resume_clear"] is True
    assert report["blocking_item_count"] == 0
    assert report["warning_item_count"] == 1
    item = report["items"][0]
    assert item["code"] == "missing_profit_first_trade_plan_historical_quarantined"
    assert item["severity"] == "warning"
    assert item["required_resolution"] == "keep_quarantined_until_manual_trust_or_window_rolls"


def test_recovery_blockers_preserve_ranking_summary_disable_when_details_truncated() -> None:
    report = build_profit_first_recovery_blockers(
        trade_contract={"current_summary": {}},
        ranking={
            "summary": {
                "disable_count": 1,
                "demote_count": 41,
            },
            "blockers": [],
        },
        observation={"blockers": []},
    )

    assert report["status"] == "ready"
    assert report["resume_clear"] is True
    assert report["blocking_item_count"] == 0
    disable_item = next(
        item for item in report["items"] if item["code"] == "strategy_disable_summary"
    )
    assert disable_item["severity"] == "warning"
    assert disable_item["lane_scoped_containment"] is True
    assert disable_item["count"] == 1
    demote_item = next(
        item for item in report["items"] if item["code"] == "strategy_demote_summary"
    )
    assert demote_item["severity"] == "warning"
    assert demote_item["count"] == 41
