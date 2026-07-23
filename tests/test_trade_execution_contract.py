from __future__ import annotations

from types import SimpleNamespace

from models.decision import _compact_decision_learning_snapshot
from services.trade_execution_contract import summarize_trade_execution_contract
from tests.paper_canary_fixtures import (
    bounded_legacy_fill_drift_raw,
    complete_paper_canary_raw,
)


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
        "live_ml_profit_contract": {
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


def _filled_order(
    decision_id: int,
    *,
    quantity: float = 1.2,
    price: float = 100.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        decision_id=decision_id,
        status="filled",
        quantity=quantity,
        price=price,
    )


def _rules_canary_raw(
    *,
    final_notional: float = 8.0,
    max_notional: float = 10.0,
    signal_action: str = "long",
) -> dict[str, object]:
    provenance = _provenance()
    stress_fraction = 0.025
    planned_loss = final_notional * stress_fraction
    exchange_minimum = {
        "production_eligible": True,
        "minimum_notional_usdt": 1.0,
    }
    return {
        "production_trade_gate": {
            "can_trade": True,
            "mode": "live_rules_canary",
            "decision_authority": "rules",
            "model_can_influence": False,
            "reason": "collecting_authoritative_profit_samples",
            "version": "test-gate",
            "risk": {
                "max_notional_usdt": max_notional,
                "max_open_positions": 1,
                "max_daily_loss_usdt": 3.0,
            },
        },
        "live_rules_canary_signal": {
            "version": "test-rules-canary-signal",
            "execution_scope": "live_rules_canary",
            "decision_authority": "rules",
            "model_can_influence": False,
            "production_eligible": True,
            "action": signal_action,
            "policy_provenance": provenance,
        },
        "model_shadow_decision": {
            "action": "short",
            "observation_only": True,
            "can_authorize_entry": False,
            "can_change_size_or_leverage": False,
        },
        "live_rules_canary_contract": {
            "execution_scope": "live_rules_canary",
            "production_permission": True,
            "decision_authority": "rules",
            "model_can_influence": False,
            "order_notional_usdt": final_notional,
            "max_notional_usdt": max_notional,
            "exchange_min_notional_usdt": 1.0,
            "gate_version": "test-gate",
            "gate_reason": "collecting_authoritative_profit_samples",
            "blockers": [],
        },
        "opportunity_score": {
            "production_eligible": False,
            "execution_cost": {
                "production_eligible": True,
                "total_pct": 0.08,
                "order_notional_usdt": final_notional,
                "order_size_complete": True,
                "policy_provenance": provenance,
            },
        },
        "profit_risk_sizing": {
            "contract_version": "test-rules-canary-sizing",
            "contract_lifecycle": "live_rules_canary",
            "production_eligible": True,
            "execution_scope": "live_rules_canary",
            "production_permission": True,
            "decision_authority": "rules",
            "model_can_influence": False,
            "risk_budget_usdt": max(planned_loss, 0.5),
            "planned_stressed_loss_usdt": planned_loss,
            "stressed_loss_fraction": stress_fraction,
            "target_notional_usdt": final_notional,
            "target_inst_id": "BTC-USDT-SWAP",
            "target_price": 100.0,
            "selected_contract_spec": {
                "ctVal": "0.01",
                "ctMult": "1",
                "minSz": "1",
                "lotSz": "1",
            },
            "exchange_minimum_order": exchange_minimum,
            "exchange_min_notional_usdt": 1.0,
            "final_notional_usdt": final_notional,
            "final_margin_usdt": final_notional,
            "final_leverage": 1.0,
            "leverage_tier_selection": {
                "production_eligible": True,
                "max_leverage": 20.0,
            },
            "policy_provenance": provenance,
        },
    }


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
    raw["live_ml_profit_contract"]["expected_net_return_pct"] = -0.01
    raw["live_ml_profit_contract"]["return_lcb_pct"] = -0.2

    report = summarize_trade_execution_contract(
        [_decision(2, "short", raw)],
        orders=[_filled_order(2)],
    )

    reasons = report["violation_reason_counts"]
    assert reasons["fee_after_expected_return_not_positive"] == 1
    assert reasons["fee_after_return_lcb_not_positive"] == 1


def test_live_rules_canary_contract_does_not_require_live_ml_profit_contract() -> None:
    report = summarize_trade_execution_contract(
        [_decision(20, "long", _rules_canary_raw())],
        orders=[_filled_order(20, quantity=0.08)],
    )

    assert report["summary"]["executed_entry_count"] == 1
    assert report["summary"]["entry_contract_ready_count"] == 1
    assert report["summary"]["contract_violation_count"] == 0
    assert report["entry_contracts"][0]["contract_lifecycle"] == "live_rules_canary"


def test_live_rules_canary_contract_enforces_gate_notional_limit() -> None:
    report = summarize_trade_execution_contract(
        [
            _decision(
                21,
                "short",
                _rules_canary_raw(
                    final_notional=12.0,
                    max_notional=10.0,
                    signal_action="short",
                ),
            )
        ],
        orders=[_filled_order(21, quantity=0.12)],
    )

    reasons = report["violation_reason_counts"]
    assert reasons == {
        "filled_order_notional_above_rules_canary_limit": 1,
        "rules_canary_order_notional_above_gate_limit": 1,
    }


def test_live_rules_canary_contract_rejects_notional_below_exchange_minimum() -> None:
    raw = _rules_canary_raw(final_notional=0.5)

    report = summarize_trade_execution_contract(
        [_decision(23, "long", raw)],
        orders=[_filled_order(23, quantity=0.005)],
    )

    reasons = report["violation_reason_counts"]
    assert reasons["rules_canary_notional_below_exchange_minimum"] == 1


def test_live_rules_canary_contract_requires_authoritative_rule_signal() -> None:
    raw = _rules_canary_raw()
    raw.pop("live_rules_canary_signal")

    report = summarize_trade_execution_contract(
        [_decision(22, "long", raw)],
        orders=[_filled_order(22, quantity=0.08)],
    )

    reasons = report["violation_reason_counts"]
    assert reasons["rules_canary_signal_missing_or_ineligible"] == 1
    assert reasons["rules_canary_signal_authority_invalid"] == 1
    assert reasons["rules_canary_signal_model_influence_invalid"] == 1
    assert reasons["rules_canary_signal_provenance_incomplete"] == 1


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


def test_complete_executed_paper_canary_uses_its_own_entry_contract() -> None:
    report = summarize_trade_execution_contract(
        [_decision(11, "long", complete_paper_canary_raw())],
        orders=[
            SimpleNamespace(
                decision_id=11,
                status="filled",
                quantity=0.5,
                price=100.0,
            )
        ],
    )

    assert report["summary"]["entry_contract_ready_count"] == 1
    assert report["summary"]["contract_violation_count"] == 0
    assert report["entry_contracts"][0]["contract_lifecycle"] == (
        "paper_bootstrap_canary"
    )
    assert report["entry_contracts"][0]["production_permission"] is False


def test_compact_decision_projection_preserves_paper_canary_contract() -> None:
    compact = _compact_decision_learning_snapshot(complete_paper_canary_raw())

    report = summarize_trade_execution_contract(
        [_decision(13, "long", compact)],
        orders=[
            SimpleNamespace(
                decision_id=13,
                status="filled",
                quantity=0.5,
                price=100.0,
            )
        ],
    )

    assert report["summary"]["entry_contract_ready_count"] == 1
    assert report["summary"]["contract_violation_count"] == 0


def test_malformed_executed_paper_canary_still_fails_closed() -> None:
    raw = complete_paper_canary_raw()
    raw["paper_bootstrap_canary"]["production_permission"] = True
    raw["profit_risk_sizing"]["planned_stressed_loss_usdt"] = 4.0

    report = summarize_trade_execution_contract(
        [_decision(12, "long", raw)],
        orders=[
            SimpleNamespace(
                decision_id=12,
                status="filled",
                quantity=0.5,
                price=100.0,
            )
        ],
    )

    assert report["summary"]["entry_contract_ready_count"] == 0
    assert report["violation_reason_counts"][
        "paper_canary_production_permission_invalid"
    ] == 1
    assert report["violation_reason_counts"]["paper_canary_risk_budget_invalid"] == 1


def test_bounded_legacy_canary_fill_drift_uses_persisted_cost_evidence() -> None:
    raw = bounded_legacy_fill_drift_raw(excess_fraction=0.001)
    final_notional = raw["profit_risk_sizing"]["final_notional_usdt"]
    report = summarize_trade_execution_contract(
        [_decision(13, "long", raw)],
        orders=[
            SimpleNamespace(
                decision_id=13,
                status="filled",
                quantity=final_notional / 100.0,
                price=100.0,
            )
        ],
    )

    assert report["summary"]["contract_violation_count"] == 0
    assert report["entry_contracts"][0]["bounded_fill_drift_accepted"] is True


def test_confirmed_canary_fill_within_reserved_ceiling_keeps_contract_complete() -> None:
    raw = complete_paper_canary_raw()
    sizing = raw["profit_risk_sizing"]
    reserve_fraction = 0.0025
    fill_ceiling = 50.0
    target_notional = fill_ceiling / (1.0 + reserve_fraction)
    settled_notional = target_notional * 1.001
    sizing.update(
        {
            "target_notional_usdt": target_notional,
            "fill_notional_ceiling_usdt": fill_ceiling,
            "estimated_fill_drift_reserve_fraction": reserve_fraction,
            "final_notional_usdt": settled_notional,
            "final_margin_usdt": settled_notional,
            "planned_stressed_loss_usdt": settled_notional
            * float(sizing["stressed_loss_fraction"]),
            "position_size_pct": settled_notional
            / float(sizing["available_margin_usdt"]),
            "execution_reconciliations": [
                {
                    "source": "okx_pre_submit_order_shape",
                    "final_notional_usdt": target_notional,
                    "eligible": True,
                    "reasons": [],
                },
                {
                    "source": "okx_confirmed_entry_fill",
                    "final_notional_usdt": settled_notional,
                    "eligible": True,
                    "reasons": [],
                },
            ],
        }
    )

    report = summarize_trade_execution_contract(
        [_decision(15, "long", raw)],
        orders=[
            SimpleNamespace(
                decision_id=15,
                status="filled",
                quantity=settled_notional / 100.0,
                price=100.0,
            )
        ],
    )

    assert report["summary"]["contract_violation_count"] == 0
    contract = report["entry_contracts"][0]
    assert contract["bounded_fill_drift_accepted"] is True
    assert contract["fill_drift_evidence"]["explicit_reserve_contract"] is True


def test_confirmed_canary_fill_beyond_reserved_ceiling_fails_closed() -> None:
    raw = complete_paper_canary_raw()
    sizing = raw["profit_risk_sizing"]
    reserve_fraction = 0.0025
    fill_ceiling = 50.0
    target_notional = fill_ceiling / (1.0 + reserve_fraction)
    settled_notional = fill_ceiling + 0.01
    drift_reasons = [
        "execution_notional_exceeds_authoritative_target",
        "execution_stressed_loss_exceeds_risk_budget",
    ]
    sizing.update(
        {
            "production_eligible": False,
            "target_notional_usdt": target_notional,
            "fill_notional_ceiling_usdt": fill_ceiling,
            "estimated_fill_drift_reserve_fraction": reserve_fraction,
            "final_notional_usdt": settled_notional,
            "final_margin_usdt": settled_notional,
            "planned_stressed_loss_usdt": settled_notional
            * float(sizing["stressed_loss_fraction"]),
            "position_size_pct": 0.0,
            "execution_reconciliations": [
                {
                    "source": "okx_pre_submit_order_shape",
                    "final_notional_usdt": target_notional,
                    "eligible": True,
                    "reasons": [],
                },
                {
                    "source": "okx_confirmed_entry_fill",
                    "final_notional_usdt": settled_notional,
                    "eligible": False,
                    "reasons": drift_reasons,
                },
            ],
        }
    )
    sizing["policy_provenance"]["fallback_reason"] = ",".join(drift_reasons)

    report = summarize_trade_execution_contract(
        [_decision(16, "long", raw)],
        orders=[
            SimpleNamespace(
                decision_id=16,
                status="filled",
                quantity=settled_notional / 100.0,
                price=100.0,
            )
        ],
    )

    assert report["summary"]["entry_contract_ready_count"] == 0
    assert report["violation_reason_counts"]["paper_canary_notional_invalid"] == 1


def test_canary_fill_drift_beyond_persisted_cost_evidence_fails_closed() -> None:
    raw = bounded_legacy_fill_drift_raw(excess_fraction=0.003)
    final_notional = raw["profit_risk_sizing"]["final_notional_usdt"]
    report = summarize_trade_execution_contract(
        [_decision(14, "long", raw)],
        orders=[
            SimpleNamespace(
                decision_id=14,
                status="filled",
                quantity=final_notional / 100.0,
                price=100.0,
            )
        ],
    )

    assert report["summary"]["entry_contract_ready_count"] == 0
    assert report["violation_reason_counts"]["paper_canary_risk_contract_ineligible"] == 1


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


def test_external_okx_reconciliation_uses_exact_fills_history_lifecycle() -> None:
    raw = {
        "system_sync": True,
        "source": "okx_position_reconcile",
        "reconcile_origin": "external_okx_sync",
        "close_fill": {"reconcile_origin": "external_okx_sync"},
    }
    order = _filled_order(9)
    order.exchange_order_id = "okx-external-close-9"
    order.okx_raw_fills = {
        "source": "okx_reconcile_close_fill",
        "fills_history_confirmed": True,
        "order_id": "okx-external-close-9",
        "inst_id": "BTC-USDT-SWAP",
        "contracts": 2.0,
        "contract_size_verified": True,
        "base_quantity": 0.02,
        "avg_price": 100.0,
        "fee_abs": 0.01,
    }

    report = summarize_trade_execution_contract(
        [_decision(9, "close_short", raw)],
        orders=[order],
    )

    assert report["summary"]["exit_contract_ready_count"] == 1
    assert report["summary"]["contract_violation_count"] == 0
    assert report["exit_contracts"][0]["contract_kind"] == "okx_external_reconciliation"


def test_external_okx_reconciliation_without_exact_fill_fact_fails_closed() -> None:
    raw = {
        "system_sync": True,
        "source": "okx_position_reconcile",
        "reconcile_origin": "external_okx_sync",
    }

    report = summarize_trade_execution_contract(
        [_decision(10, "close_short", raw)],
        orders=[_filled_order(10)],
    )

    assert report["violation_reason_counts"][
        "external_okx_close_fill_lifecycle_not_unique"
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
