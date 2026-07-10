from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from services.profit_first_trade_plan import build_profit_first_trade_plan
from services.trade_execution_contract import (
    TradeExecutionContractService,
    summarize_trade_execution_contract,
)


def _anchor_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _entry_decision(
    *,
    decision_id: int = 1,
    symbol: str = "BTC/USDT",
    action: str = "long",
    expected_net: float = 1.2,
    evidence_tier: str = "normal",
    position_size_pct: float = 0.05,
    execution_reason: str = "entered because positive EV and aligned evidence",
    raw_extra: dict | None = None,
    include_profit_first_plan: bool = True,
    created_at: datetime | None = None,
) -> SimpleNamespace:
    now = created_at or (_anchor_now() - timedelta(minutes=5))
    raw = {
        "opportunity_score": {
            "score": 3.2,
            "min_score_required": 0.95,
            "side": action,
            "expected_return_pct": expected_net + 0.11,
            "expected_net_return_pct": expected_net,
            "fee_pct": 0.05,
            "slippage_pct": 0.06,
            "expected_loss_pct": 0.18,
            "profit_quality_ratio": 1.35,
            "reward_risk_ratio": 4.0,
            "server_profit_loss_probability": 0.32,
            "tail_risk_score": 0.54,
            "side_realized_pnl_usdt": 1.0,
            "ml_aligned": True,
            "local_profit_aligned": True,
            "timeseries_aligned": True,
            "evidence_score": {
                "tier": evidence_tier,
                "effective_score": 84.0,
                "components": [
                    {"source": "ml", "status": "aligned"},
                    {"source": "timeseries", "status": "aligned"},
                ],
            },
        },
        "strategy_learning_context": {"strategy_profile_id": "balanced_probe"},
        "current_price": 100.0,
        "profit_risk_sizing": {
            "applied": False,
            "quality_tier": "high_profit",
            "position_size_pct": position_size_pct,
            "final_notional_usdt": 100.0,
            "expected_profit_usdt": max(expected_net, 0.0),
            "planned_stop_loss_usdt": 3.5,
            "max_stop_loss_usdt": 9.0,
            "meaningful_size_reason": "positive EV with low loss probability",
            "notional_floor_reason": "high quality evidence supports meaningful size",
            "profit_first_position_ladder": {
                "version": "profit-first-position-ladder-v1",
                "lane": "meaningful_entry",
                "target_min_pct": 0.05,
                "target_max_pct": 0.08,
                "original_size_pct": position_size_pct,
                "adjusted_size_pct": position_size_pct,
                "post_stop_budget_size_pct": position_size_pct,
                "capped_by_stop_loss_budget": False,
                "capped_by_low_payoff": False,
                "reasons": [],
            },
        },
    }
    if raw_extra:
        for key, value in raw_extra.items():
            if isinstance(value, dict) and isinstance(raw.get(key), dict):
                raw[key] = {**raw[key], **value}
            else:
                raw[key] = value
    row = SimpleNamespace(
        id=decision_id,
        model_name="ensemble_trader",
        symbol=symbol,
        action=action,
        confidence=0.82,
        reasoning="positive EV entry",
        position_size_pct=position_size_pct,
        suggested_leverage=4.0,
        stop_loss_pct=0.015,
        take_profit_pct=0.06,
        raw_llm_response=raw,
        execution_reason=execution_reason,
        was_executed=True,
        created_at=created_at or now,
    )
    if include_profit_first_plan:
        raw["profit_first_trade_plan"] = build_profit_first_trade_plan(
            row,
            analysis_type="market",
            now=now,
        ).to_dict()
    return row


def _order(
    decision_id: int,
    *,
    status: str = "filled",
    created_at: datetime | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=10 + decision_id,
        decision_id=decision_id,
        symbol="BTC/USDT",
        side="buy",
        status=status,
        created_at=created_at or (_anchor_now() - timedelta(minutes=4)),
    )


def _position(
    *,
    position_id: int = 1,
    symbol: str = "BTC/USDT",
    side: str = "long",
    hold_minutes: float = 6.0,
    realized_pnl: float = -2.0,
) -> SimpleNamespace:
    closed_at = _anchor_now()
    return SimpleNamespace(
        id=position_id,
        symbol=symbol,
        side=side,
        quantity=1.0,
        entry_price=100.0,
        realized_pnl=realized_pnl,
        is_open=False,
        created_at=closed_at - timedelta(minutes=hold_minutes),
        closed_at=closed_at,
    )


def _exit_decision(
    *,
    strong: bool,
    created_at: datetime | None = None,
    raw_override: dict | None = None,
) -> SimpleNamespace:
    raw = raw_override or (
        {
            "exit_intent": "hard_risk",
            "fast_risk_exit": True,
            "fast_risk_trigger": "stop_loss",
            "close_evidence": {"hard_risk": True},
            "exit_arbitration": {"intent": "hard_risk", "priority": 100},
        }
        if strong
        else {"exit_intent": "ordinary", "close_evidence": {}}
    )
    if raw_override is None:
        raw = {
            **raw,
            "profit_first_exit_reference": {
                "exit_plan_id": "pfep-test-exit-plan",
                "source": "matched_open_position",
                "missing_original_exit_plan_reference": False,
            },
            "close_evidence": {
                **(raw.get("close_evidence") if isinstance(raw.get("close_evidence"), dict) else {}),
                "profit_first_exit_plan_id": "pfep-test-exit-plan",
            },
        }
    return SimpleNamespace(
        id=99,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action="close_long",
        confidence=0.7,
        reasoning="exit",
        position_size_pct=1.0,
        raw_llm_response=raw,
        execution_reason="exit submitted",
        was_executed=True,
        created_at=created_at or _anchor_now(),
    )


def test_executed_entry_with_positive_ev_sizing_and_reason_is_clean() -> None:
    report = summarize_trade_execution_contract(
        decisions=[_entry_decision()],
        orders=[_order(1)],
        positions=[],
    )

    assert report["audit_only"] is True
    assert report["live_entry_mutation"] is False
    assert report["live_exit_mutation"] is False
    assert report["summary"]["executed_entry_count"] == 1
    assert report["summary"]["missing_entry_explanation_count"] == 0
    assert report["summary"]["missing_sizing_explanation_count"] == 0
    assert report["summary"]["weak_evidence_executed_count"] == 0
    assert report["summary"]["contract_violation_count"] == 0
    assert report["entry_explanations"][0]["decision_id"] == 1
    assert report["entry_explanations"][0]["expected_net_return_pct"] == 1.2
    assert report["entry_explanations"][0]["profit_first_position_ladder"]["lane"] == (
        "meaningful_entry"
    )


def test_executed_entry_requires_profit_first_position_ladder() -> None:
    report = summarize_trade_execution_contract(
        decisions=[
            _entry_decision(
                raw_extra={"profit_risk_sizing": {"profit_first_position_ladder": None}},
            )
        ],
        orders=[_order(1)],
        positions=[],
    )

    assert report["summary"]["profit_first_position_ladder_missing_count"] == 1
    assert report["violation_reason_counts"]["missing_profit_first_position_ladder"] == 1


def test_historical_recovery_quarantine_keeps_legacy_shadow_out_of_blocking_counts() -> None:
    report = summarize_trade_execution_contract(
        decisions=[
            _entry_decision(
                include_profit_first_plan=False,
                raw_extra={
                    "profit_first_historical_recovery": {
                        "training_policy": "exclude_until_manual_trust",
                        "operator_approval_required": True,
                    },
                    "profit_first_trade_plan": {
                        "decision_lane": "shadow_only",
                        "is_complete_for_real_trade": False,
                        "training_policy": "exclude_until_manual_trust",
                    },
                    "profit_risk_sizing": {
                        "profit_first_position_ladder": {
                            "lane": "shadow_only",
                            "target_min_pct": 0.0,
                            "target_max_pct": 0.0,
                        }
                    },
                },
            )
        ],
        orders=[_order(1)],
        positions=[],
    )

    assert report["summary"]["contract_violation_count"] == 0
    assert report["summary"]["historical_recovery_quarantined_violation_count"] == 2
    assert report["summary"]["profit_first_plan_incomplete_count_unresolved"] == 0
    assert report["summary"]["shadow_lane_executed_count_unresolved"] == 0
    assert report["summary"]["historical_recovery_quarantined_profit_first_plan_incomplete_count"] == 1
    assert report["summary"]["historical_recovery_quarantined_shadow_lane_executed_count"] == 1
    assert report["violation_reason_counts"] == {}
    assert {
        item["reason"] for item in report["historical_recovery_quarantined_violations"]
    } == {"incomplete_profit_first_trade_plan", "shadow_lane_executed"}


def test_low_payoff_entry_cannot_receive_meaningful_size() -> None:
    report = summarize_trade_execution_contract(
        decisions=[
            _entry_decision(
                position_size_pct=0.06,
                raw_extra={
                    "profit_risk_sizing": {
                        "low_payoff_quality": True,
                        "profit_first_position_ladder": {
                            "version": "profit-first-position-ladder-v1",
                            "lane": "meaningful_entry",
                            "target_min_pct": 0.05,
                            "target_max_pct": 0.08,
                            "post_stop_budget_size_pct": 0.06,
                        },
                    }
                },
            )
        ],
        orders=[_order(1)],
        positions=[],
    )

    assert report["summary"]["low_payoff_meaningful_size_count"] == 1
    assert report["violation_reason_counts"]["low_payoff_meaningful_size"] == 1


def test_executed_entry_can_use_structured_raw_reason_when_final_reason_missing() -> None:
    report = summarize_trade_execution_contract(
        decisions=[
            _entry_decision(
                execution_reason="",
                raw_extra={
                    "opportunity_score": {
                        "expected_net_return_pct": 1.2,
                        "profit_quality_ratio": 1.35,
                        "server_profit_loss_probability": 0.32,
                        "tail_risk_score": 0.54,
                        "evidence_score": {"tier": "normal", "effective_score": 84.0},
                        "selection_reason": "positive net return and aligned evidence selected this entry",
                    }
                },
            )
        ],
        orders=[_order(1)],
        positions=[],
    )

    assert report["summary"]["executed_entry_count"] == 1
    assert report["summary"]["missing_entry_explanation_count"] == 0
    assert report["entry_explanations"][0]["has_execution_reason"] is True
    assert report["entry_explanations"][0]["execution_reason_source"] == "selection_reason"
    assert report["summary"]["contract_violation_count"] == 0


def test_executed_entry_flags_weak_negative_and_unexplained_small_size() -> None:
    report = summarize_trade_execution_contract(
        decisions=[
            _entry_decision(
                decision_id=2,
                expected_net=-0.15,
                evidence_tier="weak_conflict_probe",
                position_size_pct=0.012,
                execution_reason="",
                raw_extra={"profit_risk_sizing": None},
            )
        ],
        orders=[_order(2)],
        positions=[],
    )

    summary = report["summary"]
    assert summary["executed_entry_count"] == 1
    assert summary["weak_evidence_executed_count"] == 1
    assert summary["negative_expected_executed_count"] == 1
    assert summary["missing_entry_explanation_count"] == 1
    assert summary["missing_sizing_explanation_count"] == 1
    assert summary["small_size_without_reason_count"] == 1
    assert report["violation_reason_counts"] == {
        "weak_evidence_executed": 1,
        "non_positive_expected_net_executed": 1,
        "missing_entry_execution_reason": 1,
        "missing_profit_risk_sizing": 1,
        "incomplete_profit_first_trade_plan": 1,
        "shadow_lane_executed": 1,
        "small_size_without_reason": 1,
    }
    assert summary["profit_first_plan_incomplete_count"] == 1
    assert summary["shadow_lane_executed_count"] == 1


def test_executed_entry_uses_final_profit_first_expected_net_for_contract() -> None:
    report = summarize_trade_execution_contract(
        decisions=[
            _entry_decision(
                decision_id=21,
                expected_net=-0.02,
                evidence_tier="exploration",
                position_size_pct=0.012,
                include_profit_first_plan=False,
                raw_extra={
                    "profit_first_trade_plan": {
                        "decision_lane": "tiny_probe",
                        "is_complete_for_real_trade": True,
                        "expected_net_return_pct": 0.34,
                        "expected_profit_usdt": 0.11,
                        "exit_plan_id": "pfep-final-positive",
                        "model_sources": ["decision_llm", "timeseries"],
                    },
                    "profit_risk_sizing": {
                        "quality_tier": "base",
                        "position_size_pct": 0.012,
                        "final_notional_usdt": 32.0,
                        "expected_profit_usdt": 0.11,
                        "reason": "final profit-first adjudication is positive",
                        "profit_first_position_ladder": {
                            "version": "profit-first-position-ladder-v1",
                            "lane": "tiny_probe",
                            "target_min_pct": 0.01,
                            "target_max_pct": 0.02,
                            "adjusted_size_pct": 0.012,
                            "post_stop_budget_size_pct": 0.012,
                            "capped_by_stop_loss_budget": True,
                        },
                    },
                },
            )
        ],
        orders=[_order(21)],
        positions=[],
    )

    assert report["summary"]["negative_expected_executed_count"] == 0
    assert report["summary"]["contract_violation_count"] == 0
    explanation = report["entry_explanations"][0]
    assert explanation["expected_net_return_pct"] == -0.02
    assert explanation["adjudicated_expected_net_return_pct"] == 0.34
    assert explanation["profit_first_real_trade_upgrade"] is True


def test_weak_evidence_without_shadow_only_can_execute_after_profit_first_upgrade() -> None:
    report = summarize_trade_execution_contract(
        decisions=[
            _entry_decision(
                decision_id=22,
                expected_net=1.1,
                evidence_tier="weak_conflict_probe",
                raw_extra={
                    "opportunity_score": {
                        "evidence_score": {
                            "tier": "weak_conflict_probe",
                            "effective_score": 42.0,
                            "shadow_only": False,
                            "tradeable_probe": False,
                        }
                    }
                },
            )
        ],
        orders=[_order(22)],
        positions=[],
    )

    assert report["summary"]["weak_evidence_executed_count"] == 0
    assert report["summary"]["contract_violation_count"] == 0
    assert report["entry_explanations"][0]["profit_first_real_trade_upgrade"] is True


def test_shadow_only_weak_evidence_stays_contract_violation_even_if_profit_first_upgraded() -> None:
    report = summarize_trade_execution_contract(
        decisions=[
            _entry_decision(
                decision_id=23,
                expected_net=2.0,
                evidence_tier="weak_conflict_probe",
                raw_extra={
                    "opportunity_score": {
                        "evidence_score": {
                            "tier": "weak_conflict_probe",
                            "effective_score": 35.0,
                            "shadow_only": True,
                            "tradeable_probe": False,
                        }
                    }
                },
            )
        ],
        orders=[_order(23)],
        positions=[],
    )

    assert report["summary"]["weak_evidence_executed_count"] == 1
    assert report["summary"]["contract_violation_count"] == 1
    explanation = report["entry_explanations"][0]
    assert explanation["evidence_shadow_only"] is True
    assert explanation["profit_first_real_trade_upgrade"] is True


def test_fast_loss_exit_requires_strong_structured_exit_evidence() -> None:
    weak_report = summarize_trade_execution_contract(
        decisions=[_exit_decision(strong=False)],
        orders=[],
        positions=[_position()],
    )
    strong_report = summarize_trade_execution_contract(
        decisions=[_exit_decision(strong=True)],
        orders=[],
        positions=[_position()],
    )

    assert weak_report["summary"]["fast_loss_count"] == 1
    assert weak_report["summary"]["fast_loss_without_strong_exit_count"] == 1
    assert weak_report["violation_reason_counts"]["fast_loss_without_strong_exit"] == 1
    assert strong_report["summary"]["fast_loss_count"] == 1
    assert strong_report["summary"]["fast_loss_without_strong_exit_count"] == 0


def test_executed_exit_requires_profit_first_exit_plan_reference() -> None:
    report = summarize_trade_execution_contract(
        decisions=[
            _exit_decision(
                strong=True,
                raw_override={
                    "exit_intent": "hard_risk",
                    "fast_risk_exit": True,
                    "close_evidence": {"hard_risk": True},
                },
            )
        ],
        orders=[],
        positions=[],
    )

    assert report["summary"]["executed_exit_count"] == 1
    assert report["summary"]["exit_plan_reference_missing_count"] == 1
    assert (
        report["violation_reason_counts"]["missing_profit_first_exit_plan_reference"] == 1
    )


def test_legacy_recovered_exit_failure_marker_satisfies_exit_attribution() -> None:
    report = summarize_trade_execution_contract(
        decisions=[
            _exit_decision(
                strong=True,
                raw_override={
                    "exit_intent": "capital_rotation",
                    "close_evidence": {
                        "profit_first_plan_failure_reason": (
                            "legacy_exit_missing_original_profit_first_plan_reference_before_profit_first_v3"
                        )
                    },
                    "profit_first_exit_reference": {
                        "exit_plan_id": "",
                        "missing_original_exit_plan_reference": True,
                        "plan_failure_reason": (
                            "legacy_exit_missing_original_profit_first_plan_reference_before_profit_first_v3"
                        ),
                        "training_policy": "exclude_until_manual_trust",
                    },
                },
            )
        ],
        orders=[],
        positions=[],
    )

    assert report["summary"]["executed_exit_count"] == 1
    assert report["summary"]["exit_plan_reference_missing_count"] == 0
    assert report["summary"]["exit_plan_failure_reason_missing_count"] == 0
    assert report["summary"]["contract_violation_count"] == 0
    assert report["exit_explanations"][0]["profit_first_exit_plan_id"] == ""
    assert report["exit_explanations"][0]["profit_first_plan_failure_reason"] == (
        "legacy_exit_missing_original_profit_first_plan_reference_before_profit_first_v3"
    )


def test_dust_fast_loss_is_observability_not_hard_contract_violation() -> None:
    closed_at = datetime(2026, 6, 23, 8, 5, tzinfo=UTC)
    report = summarize_trade_execution_contract(
        decisions=[],
        orders=[],
        positions=[
            SimpleNamespace(
                id=23,
                symbol="PEPE/USDT",
                side="short",
                quantity=1.0,
                entry_price=0.000001,
                realized_pnl=-0.00000001,
                is_open=False,
                created_at=closed_at - timedelta(minutes=6),
                closed_at=closed_at,
            )
        ],
    )

    assert report["summary"]["fast_loss_count"] == 0
    assert report["summary"]["dust_or_rounding_fast_loss_count"] == 1
    assert report["summary"]["fast_loss_without_strong_exit_count"] == 0
    assert report["summary"]["contract_violation_count"] == 0
    assert report["dust_or_rounding_fast_loss_samples"][0]["symbol"] == "PEPE/USDT"


def test_real_fast_loss_still_requires_strong_exit_evidence() -> None:
    report = summarize_trade_execution_contract(
        decisions=[],
        orders=[],
        positions=[
            _position(
                position_id=24,
                symbol="LAB/USDT",
                side="short",
                realized_pnl=-0.63,
            )
        ],
    )

    assert report["summary"]["fast_loss_count"] == 1
    assert report["summary"]["dust_or_rounding_fast_loss_count"] == 0
    assert report["summary"]["fast_loss_without_strong_exit_count"] == 1
    assert report["summary"]["contract_violation_count"] == 1


def test_exchange_confirmed_close_fill_is_strong_external_exit_evidence() -> None:
    report = summarize_trade_execution_contract(
        decisions=[
            _exit_decision(
                strong=False,
                raw_override={
                    "system_sync": True,
                    "source": "okx_position_reconcile",
                    "close_fill": {
                        "order_id": "okx-close-1",
                        "price": 99.0,
                        "quantity": 1.0,
                        "pnl": -1.0,
                        "source": "okx_fills_history",
                    },
                },
            )
        ],
        orders=[],
        positions=[_position()],
    )

    assert report["summary"]["fast_loss_count"] == 1
    assert report["summary"]["fast_loss_without_strong_exit_count"] == 0
    assert report["summary"]["contract_violation_count"] == 0


def test_estimated_exchange_quantity_reduction_is_observability_not_fast_loss_violation() -> None:
    report = summarize_trade_execution_contract(
        decisions=[
            _exit_decision(
                strong=False,
                raw_override={
                    "system_sync": True,
                    "source": "okx_position_reconcile",
                    "close_fill": {
                        "estimated": True,
                        "partial_reduction": True,
                        "price": 99.0,
                        "quantity": 1.0,
                        "remaining_quantity": 0.1,
                        "pnl": -1.0,
                    },
                },
            )
        ],
        orders=[],
        positions=[_position()],
    )

    assert report["summary"]["fast_loss_count"] == 0
    assert report["summary"]["exchange_sync_estimated_reduction_count"] == 1
    assert report["summary"]["fast_loss_without_strong_exit_count"] == 0
    assert report["summary"]["contract_violation_count"] == 0


def test_recent_loss_reentry_requires_allowed_high_quality_unlock() -> None:
    loss_profile = {
        "symbol_side_profile": {
            "losses": 1,
            "pnl": -8.0,
            "today_pnl": -8.0,
            "last_loss_age_hours": 0.4,
        }
    }
    missing_unlock = _entry_decision(
        decision_id=3, raw_extra={"opportunity_score": {**loss_profile}}
    )
    allowed_unlock = _entry_decision(
        decision_id=4,
        raw_extra={
            "opportunity_score": {
                **loss_profile,
                "expected_net_return_pct": 1.6,
                "profit_quality_ratio": 1.4,
                "server_profit_loss_probability": 0.34,
                "tail_risk_score": 0.62,
                "loss_cooldown_override": {"allowed": True},
            },
            "loss_cooldown_override": {
                "allowed": True,
                "metrics": {
                    "fresh_loss": True,
                    "expected_net_return_pct": 1.6,
                    "profit_quality_ratio": 1.4,
                    "server_profit_loss_probability": 0.34,
                    "aligned_sources": ["ml_aligned", "local_profit_aligned", "timeseries_aligned"],
                },
            },
        },
    )

    blocked_report = summarize_trade_execution_contract(
        decisions=[missing_unlock],
        orders=[_order(3)],
        positions=[],
    )
    allowed_report = summarize_trade_execution_contract(
        decisions=[allowed_unlock],
        orders=[_order(4)],
        positions=[],
    )

    assert blocked_report["summary"]["reentry_without_strong_unlock_count"] == 1
    assert blocked_report["violation_reason_counts"]["reentry_without_strong_unlock"] == 1
    assert allowed_report["summary"]["reentry_without_strong_unlock_count"] == 0


def test_executed_entry_cannot_bypass_profit_first_probe_loss_brake() -> None:
    report = summarize_trade_execution_contract(
        decisions=[
            _entry_decision(
                decision_id=5,
                position_size_pct=0.05,
                raw_extra={
                    "skip_kind": "profit_first_probe_loss_brake",
                    "shadow_only": True,
                    "probe_loop_health": {
                        "all_recent_probes_losing": True,
                        "probe_closed_count": 3,
                    },
                    "profit_risk_sizing": {
                        "quality_tier": "validated_probe",
                        "position_size_pct": 0.05,
                        "final_notional_usdt": 20.0,
                        "expected_profit_usdt": 0.3,
                        "planned_stop_loss_usdt": 1.0,
                        "max_stop_loss_usdt": 3.0,
                        "reason": "tiny probe only",
                        "profit_first_position_ladder": {
                            "version": "profit-first-position-ladder-v1",
                            "lane": "validated_probe",
                            "target_min_pct": 0.02,
                            "target_max_pct": 0.05,
                            "adjusted_size_pct": 0.05,
                            "post_stop_budget_size_pct": 0.05,
                        },
                    },
                },
            )
        ],
        orders=[_order(5)],
        positions=[],
    )

    assert report["summary"]["probe_loss_brake_bypassed_count"] == 1
    assert report["summary"]["contract_violation_count"] == 1
    assert (
        report["violation_reason_counts"]["profit_first_probe_loss_brake_bypassed"] == 1
    )
    assert report["entry_explanations"][0]["profit_first_probe_loss_brake"]["active"] is True
    assert (
        report["policy"]["profit_first_probe_loss_brake_must_block_execution"] is True
    )


@pytest.mark.asyncio
async def test_report_uses_recent_primary_key_window_for_online_read_only_path() -> None:
    class FakeScalarResult:
        def __init__(self, rows: list[SimpleNamespace]) -> None:
            self._rows = rows

        def all(self) -> list[SimpleNamespace]:
            return self._rows

    class FakeResult:
        def __init__(self, rows: list[SimpleNamespace]) -> None:
            self._rows = rows

        def scalars(self) -> FakeScalarResult:
            return FakeScalarResult(self._rows)

    class FakeSession:
        def __init__(self) -> None:
            self.statements: list[str] = []

        async def execute(self, statement: object) -> FakeResult:
            compiled = str(statement)
            self.statements.append(compiled)
            if "ai_decisions" in compiled:
                return FakeResult([_entry_decision()])
            if "orders" in compiled:
                return FakeResult([_order(1)])
            if "positions" in compiled:
                return FakeResult([])
            return FakeResult([])

    class FakeSessionContext:
        def __init__(self, session: FakeSession) -> None:
            self._session = session

        async def __aenter__(self) -> FakeSession:
            return self._session

        async def __aexit__(self, *_exc: object) -> None:
            return None

    fake_session = FakeSession()

    def session_context_factory() -> FakeSessionContext:
        return FakeSessionContext(fake_session)

    report = await TradeExecutionContractService(
        session_context_factory=session_context_factory
    ).report()

    assert report["window_hours"] == 24
    assert report["query_policy"] == {
        "online_safe": True,
        "ordered_by_primary_key": True,
        "db_time_filter": False,
        "row_limit": 500,
        "supplemental_order_decision_lookup": False,
        "supplemental_order_decision_count": 0,
        "supplemental_exit_lookup": False,
        "supplemental_exit_lookup_minutes": 30,
        "supplemental_exit_decision_count": 0,
        "supplemental_fast_loss_position_count": 0,
    }
    assert any("ORDER BY ai_decisions.id DESC" in item for item in fake_session.statements)
    assert any("ORDER BY orders.id DESC" in item for item in fake_session.statements)
    assert any("ORDER BY positions.id DESC" in item for item in fake_session.statements)


@pytest.mark.asyncio
async def test_report_since_filters_legacy_decisions_and_orders_from_current_window() -> None:
    since = datetime(2026, 6, 23, 8, 0, tzinfo=UTC)
    old_at = since - timedelta(minutes=30)
    current_at = since + timedelta(minutes=5)

    class FakeScalarResult:
        def __init__(self, rows: list[SimpleNamespace]) -> None:
            self._rows = rows

        def all(self) -> list[SimpleNamespace]:
            return self._rows

    class FakeResult:
        def __init__(self, rows: list[SimpleNamespace]) -> None:
            self._rows = rows

        def scalars(self) -> FakeScalarResult:
            return FakeScalarResult(self._rows)

    class FakeSession:
        def __init__(self) -> None:
            self.statements: list[str] = []
            self.ai_decision_calls = 0

        async def execute(self, statement: object) -> FakeResult:
            compiled = str(statement)
            self.statements.append(compiled)
            if "FROM ai_decisions" in compiled:
                self.ai_decision_calls += 1
                if self.ai_decision_calls > 1:
                    return FakeResult([])
                return FakeResult(
                    [
                        _entry_decision(
                            decision_id=1,
                            evidence_tier="weak_conflict_probe",
                            created_at=old_at,
                        ),
                        _entry_decision(decision_id=2, created_at=current_at),
                    ]
                )
            if "FROM orders" in compiled:
                return FakeResult(
                    [
                        _order(1, created_at=old_at),
                        _order(2, created_at=current_at),
                    ]
                )
            if "FROM positions" in compiled:
                return FakeResult([])
            return FakeResult([])

    class FakeSessionContext:
        def __init__(self, session: FakeSession) -> None:
            self._session = session

        async def __aenter__(self) -> FakeSession:
            return self._session

        async def __aexit__(self, *_exc: object) -> None:
            return None

    fake_session = FakeSession()

    def session_context_factory() -> FakeSessionContext:
        return FakeSessionContext(fake_session)

    report = await TradeExecutionContractService(
        session_context_factory=session_context_factory
    ).report(since=since)

    assert report["summary"]["executed_entry_count"] == 1
    assert report["summary"]["weak_evidence_executed_count"] == 0
    assert report["summary"]["contract_violation_count"] == 0
    assert report["query_policy"]["db_time_filter"] is True
    assert report["query_policy"]["since_utc"] == "2026-06-23T08:00:00+00:00"


@pytest.mark.asyncio
async def test_report_supplements_exit_decisions_for_fast_loss_positions() -> None:
    class FakeScalarResult:
        def __init__(self, rows: list[SimpleNamespace]) -> None:
            self._rows = rows

        def all(self) -> list[SimpleNamespace]:
            return self._rows

    class FakeResult:
        def __init__(self, rows: list[SimpleNamespace]) -> None:
            self._rows = rows

        def scalars(self) -> FakeScalarResult:
            return FakeScalarResult(self._rows)

    class FakeSession:
        def __init__(self) -> None:
            self.statements: list[str] = []
            self.ai_decision_calls = 0

        async def execute(self, statement: object) -> FakeResult:
            compiled = str(statement)
            self.statements.append(compiled)
            if "FROM positions" in compiled:
                return FakeResult([_position()])
            if "FROM orders" in compiled:
                return FakeResult([])
            if "FROM ai_decisions" in compiled:
                self.ai_decision_calls += 1
                if self.ai_decision_calls > 1:
                    return FakeResult([_exit_decision(strong=True)])
                return FakeResult([])
            return FakeResult([])

    class FakeSessionContext:
        def __init__(self, session: FakeSession) -> None:
            self._session = session

        async def __aenter__(self) -> FakeSession:
            return self._session

        async def __aexit__(self, *_exc: object) -> None:
            return None

    fake_session = FakeSession()

    def session_context_factory() -> FakeSessionContext:
        return FakeSessionContext(fake_session)

    report = await TradeExecutionContractService(
        session_context_factory=session_context_factory
    ).report(hours=168)

    assert report["summary"]["fast_loss_count"] == 1
    assert report["summary"]["fast_loss_without_strong_exit_count"] == 0
    assert report["summary"]["contract_violation_count"] == 0
    assert report["query_policy"]["supplemental_exit_lookup"] is True
    assert sum("FROM ai_decisions" in item for item in fake_session.statements) == 2


@pytest.mark.asyncio
async def test_report_supplements_order_decisions_for_executed_entries() -> None:
    class FakeScalarResult:
        def __init__(self, rows: list[SimpleNamespace]) -> None:
            self._rows = rows

        def all(self) -> list[SimpleNamespace]:
            return self._rows

    class FakeResult:
        def __init__(self, rows: list[SimpleNamespace]) -> None:
            self._rows = rows

        def scalars(self) -> FakeScalarResult:
            return FakeScalarResult(self._rows)

    class FakeSession:
        def __init__(self) -> None:
            self.statements: list[str] = []
            self.ai_decision_calls = 0

        async def execute(self, statement: object) -> FakeResult:
            compiled = str(statement)
            self.statements.append(compiled)
            if "FROM orders" in compiled:
                return FakeResult([_order(1)])
            if "FROM positions" in compiled:
                return FakeResult([])
            if "FROM ai_decisions" in compiled:
                self.ai_decision_calls += 1
                if self.ai_decision_calls > 1:
                    return FakeResult([_entry_decision()])
                return FakeResult([])
            return FakeResult([])

    class FakeSessionContext:
        def __init__(self, session: FakeSession) -> None:
            self._session = session

        async def __aenter__(self) -> FakeSession:
            return self._session

        async def __aexit__(self, *_exc: object) -> None:
            return None

    fake_session = FakeSession()

    def session_context_factory() -> FakeSessionContext:
        return FakeSessionContext(fake_session)

    report = await TradeExecutionContractService(
        session_context_factory=session_context_factory
    ).report(hours=168)

    assert report["summary"]["executed_entry_count"] == 1
    assert report["summary"]["contract_violation_count"] == 0
    assert report["query_policy"]["supplemental_order_decision_lookup"] is True
    assert report["query_policy"]["supplemental_order_decision_count"] == 1
    assert sum("FROM ai_decisions" in item for item in fake_session.statements) == 2
