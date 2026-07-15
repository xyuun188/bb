from __future__ import annotations

from types import SimpleNamespace

from services.trade_execution_contract import summarize_trade_execution_contract


def _provenance(*, samples: int = 8, fallback_reason: str = "") -> dict[str, object]:
    return {
        "source": "live_return_distribution",
        "observation_window": "rolling_market_window",
        "sample_count": samples,
        "generated_at": "2026-07-12T08:00:00+00:00",
        "strategy_version": "return-contract-v1",
        "fallback_reason": fallback_reason,
    }


def _entry_raw() -> dict[str, object]:
    provenance = _provenance()
    return {
        "production_return_policy": {
            "eligible": True,
            "expected_net_return_pct": 0.8,
            "return_lcb_pct": 0.2,
            "execution_cost_pct": 0.08,
            "position_size_pct": 0.12,
            "production_source_count": 3,
            "policy_provenance": provenance,
        },
        "opportunity_score": {
            "production_eligible": True,
            "policy_provenance": provenance,
            "execution_cost": {
                "production_eligible": True,
                "total_pct": 0.08,
                "policy_provenance": provenance,
            },
        },
        "profit_risk_sizing": {
            "production_eligible": True,
            "risk_budget_usdt": 3.0,
            "planned_stressed_loss_usdt": 2.4,
            "stressed_loss_fraction": 0.02,
            "target_notional_usdt": 150.0,
            "final_notional_usdt": 120.0,
            "policy_provenance": provenance,
        },
    }


def _decision(decision_id: int, action: str, raw: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        id=decision_id,
        symbol="BTC/USDT",
        action=action,
        was_executed=False,
        raw_llm_response=raw,
    )


def _filled_order(decision_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        decision_id=decision_id,
        status="filled",
        quantity=1.2,
        price=100.0,
    )


def test_complete_dynamic_return_entry_contract_is_clean() -> None:
    report = summarize_trade_execution_contract(
        [_decision(1, "long", _entry_raw())],
        orders=[_filled_order(1)],
    )

    assert report["summary"]["executed_entry_count"] == 1
    assert report["summary"]["entry_contract_ready_count"] == 1
    assert report["summary"]["contract_violation_count"] == 0


def test_non_positive_fee_after_return_cannot_execute() -> None:
    raw = _entry_raw()
    raw["production_return_policy"]["expected_net_return_pct"] = -0.01
    raw["production_return_policy"]["return_lcb_pct"] = -0.2

    report = summarize_trade_execution_contract(
        [_decision(2, "short", raw)],
        orders=[_filled_order(2)],
    )

    reasons = report["violation_reason_counts"]
    assert reasons["fee_after_expected_return_not_positive"] == 1
    assert reasons["fee_after_return_lcb_not_positive"] == 1


def test_missing_cost_or_provenance_fails_closed() -> None:
    raw = _entry_raw()
    raw["opportunity_score"]["execution_cost"] = {}
    raw["profit_risk_sizing"]["policy_provenance"] = {}

    report = summarize_trade_execution_contract(
        [_decision(3, "long", raw)],
        orders=[_filled_order(3)],
    )

    reasons = report["violation_reason_counts"]
    assert reasons["live_execution_cost_incomplete"] == 1
    assert reasons["execution_cost_provenance_incomplete"] == 1
    assert reasons["dynamic_risk_budget_provenance_incomplete"] == 1


def test_obsolete_policy_payload_is_rejected_even_when_nested() -> None:
    raw = _entry_raw()
    raw["diagnostics"] = {"profit_first_trade_plan": {"decision_lane": "old"}}

    report = summarize_trade_execution_contract(
        [_decision(4, "long", raw)],
        orders=[_filled_order(4)],
    )

    assert report["summary"]["obsolete_policy_payload_count"] == 1
    assert report["violations"][0]["details"]["fields"] == ["profit_first_trade_plan"]


def test_executed_entry_requires_real_filled_order_link() -> None:
    decision = _decision(5, "long", _entry_raw())
    decision.was_executed = True

    report = summarize_trade_execution_contract([decision], orders=[])

    assert report["violation_reason_counts"]["executed_entry_without_filled_order"] == 1


def test_dynamic_exit_requires_position_economics_and_filled_order() -> None:
    raw = {
        "dynamic_exit_policy": {
            "eligible": True,
            "close_fraction": 0.35,
            "hard_risk": False,
            "fee_after_unrealized_pnl_usdt": 2.4,
            "policy_provenance": _provenance(samples=1),
        }
    }
    report = summarize_trade_execution_contract(
        [_decision(6, "close_long", raw)],
        orders=[_filled_order(6)],
    )

    assert report["summary"]["exit_contract_ready_count"] == 1
    assert report["summary"]["contract_violation_count"] == 0


def test_system_protection_exit_uses_okx_algo_fill_lifecycle() -> None:
    raw = {
        "system_sync": True,
        "source": "okx_position_reconcile",
        "reconcile_origin": "system_protection",
        "close_fill": {"reconcile_origin": "system_protection"},
    }
    order = _filled_order(7)
    order.exchange_order_id = "okx-close-7"
    order.okx_raw_fills = {
        "protection_execution": {
            "version": "2026-07-15.okx-protection-execution.v1",
            "source_authority": "okx_algo_history_plus_fills_history",
            "lifecycle_complete": True,
            "algo_id": "algo-7",
            "generated_order_id": "okx-close-7",
            "actual_side": "tp",
            "contracts": 1.2,
            "trigger_to_first_fill_ms": 1.0,
        }
    }

    report = summarize_trade_execution_contract(
        [_decision(7, "close_long", raw)],
        orders=[order],
    )

    assert report["summary"]["exit_contract_ready_count"] == 1
    assert report["summary"]["contract_violation_count"] == 0
    assert report["exit_contracts"][0]["contract_kind"] == "okx_exchange_protection"


def test_system_protection_exit_without_authoritative_lifecycle_fails_closed() -> None:
    raw = {
        "system_sync": True,
        "source": "okx_position_reconcile",
        "reconcile_origin": "system_protection",
    }

    report = summarize_trade_execution_contract(
        [_decision(8, "close_short", raw)],
        orders=[_filled_order(8)],
    )

    assert report["summary"]["exit_contract_ready_count"] == 0
    assert report["violation_reason_counts"][
        "exchange_protection_lifecycle_not_unique"
    ] == 1
    assert report["violation_reason_counts"][
        "exchange_protection_lifecycle_incomplete"
    ] == 1


def test_realized_pnl_summary_uses_closed_positions_only() -> None:
    report = summarize_trade_execution_contract(
        [],
        positions=[
            SimpleNamespace(realized_pnl=3.0, closed_at="2026-07-12T08:00:00Z"),
            SimpleNamespace(realized_pnl=-1.2, closed_at="2026-07-12T09:00:00Z"),
            SimpleNamespace(realized_pnl=99.0, closed_at=None),
        ],
    )

    assert report["summary"]["realized_net_pnl_usdt"] == 1.8
    assert report["summary"]["negative_realized_position_count"] == 1
