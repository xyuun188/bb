from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from services.profit_first_historical_recovery_package import (
    HistoricalRecoveryInput,
    build_historical_recovery_package,
    target_ids_from_blocking_actions,
)


def _entry_decision(decision_id: int = 9549) -> SimpleNamespace:
    return SimpleNamespace(
        id=decision_id,
        model_name="ensemble_trader",
        symbol="AVAX/USDT",
        action="short",
        confidence=0.82,
        position_size_pct=0.015,
        suggested_leverage=3.0,
        stop_loss_pct=0.012,
        take_profit_pct=0.024,
        feature_snapshot={"close": 21.0},
        raw_llm_response={
            "analysis_type": "market",
            "strategy_profile_id": "candidate_2",
            "opportunity_score": {
                "expected_net_return_pct": 0.45,
                "profit_quality_ratio": 0.6,
                "loss_probability": 0.48,
                "tail_loss_probability": 0.3,
                "fee_pct": 0.08,
                "slippage_pct": 0.04,
                "evidence_score": {
                    "tier": "small",
                    "components": [{"source": "sentiment", "status": "aligned"}],
                },
            },
            "profit_risk_sizing": {
                "position_size_pct": 0.015,
                "final_notional_usdt": 20.0,
                "expected_profit_usdt": 0.09,
                "expected_loss_usdt": 0.2,
                "tail_loss_usdt": 0.4,
            },
        },
        analysis_type="market",
        was_executed=True,
    )


def _exit_decision() -> SimpleNamespace:
    return SimpleNamespace(
        id=10391,
        model_name="ensemble_trader",
        symbol="ARB/USDT",
        action="close_short",
        confidence=0.7,
        raw_llm_response={"close_evidence": {"hard_risk": True}},
        analysis_type="position",
        was_executed=True,
    )


def _order() -> SimpleNamespace:
    return SimpleNamespace(
        id=2678,
        decision_id=9549,
        exchange_order_id="3697115296869093376",
        symbol="LAB/USDT",
        side="sell",
        quantity=12.0,
        price=0.08,
        status="filled",
        okx_inst_id="LAB-USDT-SWAP",
        okx_fill_contracts=1.2,
        okx_sync_status="okx_confirmed",
        okx_raw_fills={"contract_size": 10.0, "base_quantity": 12.0},
    )


def test_target_ids_from_blocking_actions_extracts_exact_targets() -> None:
    targets = target_ids_from_blocking_actions(
        [
            {
                "code": "missing_profit_first_trade_plan",
                "target": {"decision_ids": [9549, 9300]},
            },
            {
                "code": "missing_profit_first_position_ladder",
                "target": {"decision_ids": [9549, 9300]},
            },
            {
                "code": "missing_profit_first_exit_plan_reference",
                "target": {"decision_ids": [10391, 9904]},
            },
            {
                "code": "local_order_quantity_differs_from_okx_fill",
                "target": {
                    "order_ids": [2678],
                    "exchange_order_ids": ["3697115296869093376"],
                },
            },
        ]
    )

    assert targets["entry_decision_ids"] == [9549, 9300]
    assert targets["exit_decision_ids"] == [10391, 9904]
    assert targets["order_ids"] == [2678]
    assert targets["exchange_order_ids"] == ["3697115296869093376"]


def test_historical_recovery_package_is_dry_run_and_builds_patches() -> None:
    report = build_historical_recovery_package(
        HistoricalRecoveryInput(
            entry_decisions=[_entry_decision()],
            exit_decisions=[_exit_decision()],
            orders=[_order()],
            blocking_actions=[
                {
                    "category": "ranking",
                    "code": "strategy_disable_summary",
                    "target": {},
                }
            ],
        ),
        now=datetime(2026, 6, 29, 6, 58, 30, tzinfo=UTC),
    )

    assert report["dry_run"] is True
    assert report["read_only"] is True
    assert report["mutates_database"] is False
    assert report["starts_trading_service"] is False
    assert report["submits_orders"] is False
    assert report["changes_model_routing"] is False
    assert report["changes_live_sizing"] is False
    assert report["resume_allowed_by_this_package"] is False
    assert report["summary"]["item_count"] == 4
    assert report["summary"]["proposed_raw_patch_count"] == 2
    assert report["apply_policy"]["apply_supported_by_this_script"] is False

    entry = next(item for item in report["items"] if item["item_type"] == "entry_decision_recovery")
    assert entry["decision_id"] == 9549
    assert "profit_first_trade_plan" in entry["proposed_raw_patch"]
    assert "profit_risk_sizing" in entry["proposed_raw_patch"]
    assert entry["training_policy"] == "exclude_until_manual_trust"

    exit_item = next(
        item for item in report["items"] if item["item_type"] == "exit_decision_recovery"
    )
    assert exit_item["decision_id"] == 10391
    assert exit_item["proposed_raw_patch"]["profit_first_exit_reference"][
        "missing_original_exit_plan_reference"
    ] is True

    okx = next(item for item in report["items"] if item["item_type"] == "okx_order_quantity_review")
    assert okx["order_id"] == 2678
    assert okx["proposed_raw_patch"] == {}
