from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from services.current_position_management import (
    build_current_position_management_contract,
)
from services.okx_native_facts import OKX_PROTECTION_EXECUTION_VERSION
from services.position_capacity_release_audit import PositionCapacityReleaseAuditService


def _position(**overrides):
    explicit_contract = "current_management_contract" in overrides
    values = {
        "id": 1,
        "model_name": "ensemble_trader",
        "symbol": "BTC/USDT",
        "side": "long",
        "quantity": 2.0,
        "entry_price": 100.0,
        "current_price": 103.0,
        "unrealized_pnl": 6.0,
        "entry_fee": 0.12,
        "stop_loss_price": 98.0,
        "take_profit_price": 110.0,
        "created_at": datetime.now(UTC),
    }
    values.update(overrides)
    position = SimpleNamespace(**values)
    if not explicit_contract:
        position.current_management_contract = build_current_position_management_contract(
            {
                "symbol": position.symbol,
                "side": position.side,
                "quantity": position.quantity,
                "contracts": position.quantity,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "entry_fee_usdt": position.entry_fee,
                "full_entry_fee_usdt": position.entry_fee,
                "full_entry_notional_usdt": position.entry_price * position.quantity,
                "entry_fee_evidence_complete": True,
                "entry_fee_source": "okx_fills_history",
                "stop_loss_price": position.stop_loss_price,
                "take_profit_price": position.take_profit_price,
                "protection_evidence_complete": True,
                "protection_orders": [
                    {
                        "algo_id": "oco-1",
                        "state": "live",
                        "contracts": position.quantity,
                        "reduce_only": True,
                        "stop_loss_price": position.stop_loss_price,
                        "take_profit_price": position.take_profit_price,
                    }
                ],
                "position_stressed_loss_usdt": abs(
                    position.entry_price - position.stop_loss_price
                )
                * position.quantity,
                "portfolio_stressed_loss_usdt": abs(
                    position.entry_price - position.stop_loss_price
                )
                * position.quantity,
                "portfolio_gross_notional_usdt": (
                    position.current_price * position.quantity
                ),
                "account_equity_usdt": 1_000.0,
                "open_position_count": 1,
                "entry_order_ids": ["entry-1"],
                "entry_decision_ids": [1],
                "original_entry_contract_complete": False,
                "original_entry_contract_gaps": ["historical_contract_missing"],
            }
        )
    return position


def _decision(*, decision_id: int, raw: dict, executed: bool = False):
    return SimpleNamespace(
        id=decision_id,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action="close_long",
        was_executed=executed,
        raw_llm_response=raw,
        created_at=datetime.now(UTC),
    )


def _dynamic_exit_policy(*, complete: bool = True) -> dict:
    return {
        "eligible": True,
        "close_fraction": 0.72,
        "hard_risk": False,
        "policy_provenance": {
            "source": "position_fee_after_pnl_peak_stop_continuation_and_portfolio_state",
            "observation_window": "current_position_review",
            "sample_count": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy_version": "2026-07-12.dynamic-exit-return.v1",
            "fallback_reason": "" if complete else "position_economics_missing",
        },
    }


def test_capacity_audit_reports_hard_capacity_and_position_economics_only() -> None:
    service = PositionCapacityReleaseAuditService()
    report = service._summarize(
        [
            _position(),
            _position(
                id=2,
                symbol="ETH/USDT",
                entry_fee=0.0,
                current_management_contract={},
            ),
        ],
        [
            _decision(
                decision_id=10,
                raw={
                    "position_release_policy": {
                        "forced": True,
                        "source": "position_quality_capacity_release",
                    }
                },
            )
        ],
        [],
    )

    assert report["read_only"] is True
    assert report["audit_only"] is True
    assert report["can_force_close"] is False
    assert report["open_position_count"] == 2
    assert report["position_economics_complete_count"] == 1
    assert report["position_economics_incomplete_count"] == 1
    assert report["dynamic_exit_decision_count"] == 1
    assert report["dynamic_exit_decisions"][0]["dynamic_exit_contract_complete"] is False
    assert not any("release_decision" in key or "crowded" in key for key in report)


def test_capacity_audit_requires_dynamic_exit_provenance_and_filled_order_link() -> None:
    service = PositionCapacityReleaseAuditService()
    decision = _decision(
        decision_id=20,
        raw={"dynamic_exit_policy": _dynamic_exit_policy()},
        executed=True,
    )
    filled = SimpleNamespace(decision_id=20, status="filled")

    report = service._summarize([_position()], [decision], [filled])

    assert report["dynamic_exit_decision_count"] == 1
    assert report["executed_dynamic_exit_count"] == 1
    assert report["executed_dynamic_exit_contract_gap_count"] == 0
    row = report["dynamic_exit_decisions"][0]
    assert row["close_fraction"] == 0.72
    assert row["filled_order_count"] == 1
    assert row["dynamic_exit_contract_complete"] is True


def test_capacity_audit_flags_executed_exit_without_complete_dynamic_contract() -> None:
    service = PositionCapacityReleaseAuditService()
    decision = _decision(
        decision_id=30,
        raw={"dynamic_exit_policy": _dynamic_exit_policy(complete=False)},
        executed=True,
    )

    report = service._summarize([_position()], [decision], [])

    assert report["executed_dynamic_exit_count"] == 1
    assert report["executed_dynamic_exit_contract_gap_count"] == 1
    gap = report["executed_dynamic_exit_contract_gaps"][0]
    assert gap["filled_order_count"] == 0
    assert gap["dynamic_exit_contract_complete"] is False


def test_capacity_audit_classifies_okx_protection_exit_without_dynamic_gap() -> None:
    service = PositionCapacityReleaseAuditService()
    decision = _decision(
        decision_id=40,
        raw={
            "system_sync": True,
            "source": "okx_position_reconcile",
            "reconcile_origin": "system_protection",
            "close_fill": {"reconcile_origin": "system_protection"},
        },
        executed=True,
    )
    filled = SimpleNamespace(
        decision_id=40,
        status="filled",
        exchange_order_id="generated-close-1",
        okx_raw_fills={
            "protection_execution": {
                "version": OKX_PROTECTION_EXECUTION_VERSION,
                "source_authority": "okx_algo_history_plus_fills_history",
                "lifecycle_complete": True,
                "actual_side": "tp",
                "contracts": 2.0,
                "generated_order_id": "generated-close-1",
                "algo_id": "algo-tp-1",
            }
        },
    )

    report = service._summarize([_position()], [decision], [filled])

    assert report["executed_dynamic_exit_count"] == 0
    assert report["executed_exchange_protection_exit_count"] == 1
    assert report["executed_dynamic_exit_contract_gap_count"] == 0
    row = report["dynamic_exit_decisions"][0]
    assert row["exit_contract_kind"] == "okx_exchange_protection"
    assert row["exit_contract_complete"] is True
