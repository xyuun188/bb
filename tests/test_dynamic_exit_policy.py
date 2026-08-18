from datetime import UTC, datetime, timedelta

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.current_position_management import build_current_position_management_contract
from services.dynamic_exit_policy import apply_dynamic_exit
from services.paper_bootstrap_canary import PAPER_BOOTSTRAP_POSITION_LIFECYCLE_VERSION
from services.paper_training import PAPER_TRAINING_POSITION_LIFECYCLE_VERSION


def _decision(raw: dict | None = None) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.8,
        reasoning="exit test",
        position_size_pct=0.7,
        suggested_leverage=1.0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        raw_response=raw or {},
    )


def _position(**overrides: object) -> dict:
    explicit_contract = "current_management_contract" in overrides
    position = {
        "symbol": "BTC/USDT",
        "side": "long",
        "quantity": 10.0,
        "entry_price": 100.0,
        "current_price": 101.0,
        "notional_usdt": 1010.0,
        "unrealized_pnl": 10.0,
        "peak_unrealized_pnl": 20.0,
        "stop_loss_pct": 0.02,
        "stop_loss": 98.0,
        "take_profit": 110.0,
        "entry_fee_usdt": 0.5,
        "created_at": datetime.now(UTC) - timedelta(minutes=20),
    }
    position.update(overrides)
    if not explicit_contract:
        quantity = float(position["quantity"])
        entry_price = float(position["entry_price"])
        current_price = float(position["current_price"])
        stop_loss = float(position.get("stop_loss") or entry_price * 0.98)
        take_profit = float(position.get("take_profit") or entry_price * 1.1)
        entry_fee = float(position.get("entry_fee_usdt") or 0.0)
        position["current_management_contract"] = build_current_position_management_contract(
            {
                "symbol": position["symbol"],
                "side": position["side"],
                "quantity": quantity,
                "contracts": quantity,
                "entry_price": entry_price,
                "current_price": current_price,
                "entry_fee_usdt": entry_fee,
                "full_entry_fee_usdt": entry_fee,
                "full_entry_notional_usdt": entry_price * quantity,
                "entry_fee_evidence_complete": True,
                "entry_fee_source": "okx_fills_history",
                "stop_loss_price": stop_loss,
                "take_profit_price": take_profit,
                "protection_evidence_complete": True,
                "protection_orders": [
                    {
                        "algo_id": "oco-1",
                        "state": "live",
                        "contracts": quantity,
                        "reduce_only": True,
                        "stop_loss_price": stop_loss,
                        "take_profit_price": take_profit,
                    }
                ],
                "position_stressed_loss_usdt": abs(entry_price - stop_loss) * quantity,
                "portfolio_stressed_loss_usdt": abs(entry_price - stop_loss) * quantity,
                "portfolio_gross_notional_usdt": current_price * quantity,
                "account_equity_usdt": 10_000.0,
                "open_position_count": 1,
                "entry_order_ids": ["entry-1"],
                "entry_decision_ids": [],
                "original_entry_contract_complete": False,
                "original_entry_contract_gaps": ["historical_contract_missing"],
            }
        )
    return position


def test_profit_retrace_generates_continuous_fraction_and_overrides_legacy_size() -> None:
    decision = _decision()

    result = apply_dynamic_exit(decision, [_position()])

    assert result.eligible is True
    assert result.profit_retrace_ratio == pytest.approx(0.5)
    assert result.close_fraction == pytest.approx(0.5)
    assert decision.position_size_pct == pytest.approx(0.5)
    assert decision.raw_response["close_fraction"] == pytest.approx(0.5)
    assert result.policy_provenance["source"] == (
        "current_position_takeover_lifecycle_pnl_funding_peak_planned_stop_market_portfolio_and_replacement_facts"
    )
    assert result.policy_provenance["fallback_reason"] == ""
    assert result.execution_cost_complete is True
    assert result.current_management_contract_complete is True


def test_complete_paper_replacement_adds_exit_pressure_without_creating_order() -> None:
    decision = _decision(
        {
            "execution_mode": "paper",
            "stronger_opportunity": {
                "available": True,
                "production_eligible": True,
                "execution_scope": "paper_only",
                "production_permission": False,
                "creates_order": False,
                "can_increase_leverage": False,
                "symbol": "ETH/USDT",
                "side": "short",
                "expected_net_return_pct": 0.6,
                "return_lcb_pct": 0.4,
                "expected_loss_pct": 0.2,
                "execution_cost": {"production_eligible": True},
            },
        }
    )
    position = _position(
        current_price=100.0,
        notional_usdt=1000.0,
        unrealized_pnl=0.0,
        peak_unrealized_pnl=0.0,
    )

    baseline = apply_dynamic_exit(_decision({"execution_mode": "paper"}), [position])
    result = apply_dynamic_exit(decision, [position])

    assert result.eligible is True
    assert result.replacement_opportunity_eligible is True
    assert result.replacement_symbol == "ETH/USDT"
    assert result.replacement_pressure > 0.0
    assert result.close_fraction > baseline.close_fraction
    assert decision.raw_response["dynamic_exit_policy"]["replacement_pressure"] > 0.0


@pytest.mark.parametrize(
    "raw_overrides",
    [
        {"execution_mode": "live"},
        {"stronger_opportunity": {"creates_order": True}},
        {"stronger_opportunity": {"execution_cost": {}}},
        {"stronger_opportunity": {"expected_loss_pct": None}},
    ],
)
def test_ineligible_replacement_evidence_cannot_create_exit_pressure(
    raw_overrides: dict,
) -> None:
    replacement = {
        "available": True,
        "production_eligible": True,
        "execution_scope": "paper_only",
        "production_permission": False,
        "creates_order": False,
        "can_increase_leverage": False,
        "symbol": "ETH/USDT",
        "side": "long",
        "expected_net_return_pct": 0.6,
        "return_lcb_pct": 0.4,
        "expected_loss_pct": 0.2,
        "execution_cost": {"production_eligible": True},
    }
    replacement.update(raw_overrides.get("stronger_opportunity", {}))
    raw = {
        "execution_mode": raw_overrides.get("execution_mode", "paper"),
        "stronger_opportunity": replacement,
    }
    position = _position(
        current_price=100.0,
        notional_usdt=1000.0,
        unrealized_pnl=0.0,
        peak_unrealized_pnl=0.0,
    )

    baseline = apply_dynamic_exit(
        _decision({"execution_mode": raw["execution_mode"]}),
        [position],
    )
    result = apply_dynamic_exit(_decision(raw), [position])

    assert result.replacement_opportunity_eligible is False
    assert result.replacement_pressure == 0.0
    assert result.close_fraction == baseline.close_fraction


def test_explicit_model_close_recommendation_enters_risk_comparison() -> None:
    decision = _decision()
    decision.suggested_close_fraction = 0.25
    position = _position(peak_unrealized_pnl=10.0)

    result = apply_dynamic_exit(decision, [position])

    assert result.eligible is True
    assert result.model_requested_close_fraction == pytest.approx(0.25)
    assert result.model_exit_confidence == pytest.approx(0.8)
    assert result.model_exit_pressure == pytest.approx(0.2)
    assert result.close_fraction == pytest.approx(0.2)


def test_profitable_exit_without_execution_cost_fails_closed() -> None:
    result = apply_dynamic_exit(
        _decision(),
        [_position(entry_fee_usdt=0.0, exit_fee_rate=0.0)],
    )

    assert result.eligible is False
    assert result.execution_cost_complete is False
    assert "exit_execution_cost_missing" in result.reason


def test_loss_uses_planned_stop_budget_continuously() -> None:
    decision = _decision()
    position = _position(
        current_price=99.0,
        notional_usdt=990.0,
        unrealized_pnl=-10.0,
        peak_unrealized_pnl=0.0,
    )

    result = apply_dynamic_exit(decision, [position])

    assert result.eligible is True
    assert 0.0 < result.stop_risk_usage < 1.0
    assert result.close_fraction == pytest.approx(result.stop_risk_usage)


def test_fee_only_loss_does_not_consume_planned_stop_budget() -> None:
    position = _position(
        created_at=datetime.now(UTC) - timedelta(minutes=2),
        current_price=100.0,
        notional_usdt=1000.0,
        unrealized_pnl=0.0,
        peak_unrealized_pnl=0.0,
    )

    result = apply_dynamic_exit(_decision(), [position])

    assert result.fee_after_unrealized_pnl_usdt < 0.0
    assert result.stop_risk_usage == 0.0
    assert result.close_fraction == 0.0
    assert result.eligible is False
    assert "dynamic_exit_pressure_zero" in result.reason


def test_trusted_funding_profit_adds_lifecycle_profit_lock_pressure() -> None:
    position = _position(
        current_price=100.0,
        notional_usdt=1000.0,
        unrealized_pnl=0.0,
        peak_unrealized_pnl=0.0,
    )
    position["current_management_contract"] = {
        **position["current_management_contract"],
            "funding_fee_usdt": 10.0,
            "settled_funding_fee": 10.0,
        "funding_bill_count": 2,
        "funding_fee_source": "okx_account_bills",
        "funding_evidence_complete": True,
        "funding_evidence_eligible": True,
        "funding_evidence_gaps": [],
    }

    result = apply_dynamic_exit(_decision(), [position])

    assert result.funding_fee_usdt == pytest.approx(10.0)
    assert result.funding_fee_included is True
    assert result.settled_funding_fee == pytest.approx(10.0)
    assert result.current_lifecycle_net_pnl > 0.0
    assert result.projected_hold_net_pnl == pytest.approx(
        result.current_lifecycle_net_pnl + result.expected_future_funding_cashflow
    )
    assert result.funding_evidence_eligible is True
    assert result.lifecycle_net_pnl_usdt == pytest.approx(9.0)
    assert result.profit_lock_pressure == pytest.approx(0.45)
    assert result.eligible is True
    assert result.close_fraction == pytest.approx(0.45)


def test_untrusted_funding_is_audited_but_cannot_drive_exit() -> None:
    position = _position(
        current_price=100.0,
        notional_usdt=1000.0,
        unrealized_pnl=0.0,
        peak_unrealized_pnl=0.0,
    )
    position["current_management_contract"] = {
        **position["current_management_contract"],
            "funding_fee_usdt": 2030.93,
            "settled_funding_fee": 2030.93,
        "funding_bill_count": 4,
        "funding_fee_source": "okx_account_bills",
        "funding_evidence_complete": True,
        "funding_evidence_eligible": False,
        "funding_evidence_gaps": ["exchange_position_notional_mismatch"],
    }

    result = apply_dynamic_exit(_decision(), [position])

    assert result.funding_fee_usdt == pytest.approx(2030.93)
    assert result.funding_fee_included is False
    assert result.funding_evidence_eligible is False
    assert result.lifecycle_net_pnl_usdt == pytest.approx(-1.0)
    assert result.profit_lock_pressure == 0.0
    assert result.close_fraction == 0.0
    assert result.eligible is False
    assert "dynamic_exit_pressure_zero" in result.reason


def test_trusted_adverse_funding_loss_consumes_planned_risk_and_forces_exit() -> None:
    position = _position(
        current_price=100.0,
        notional_usdt=1000.0,
        unrealized_pnl=0.0,
        peak_unrealized_pnl=0.0,
    )
    position["current_management_contract"] = {
        **position["current_management_contract"],
            "funding_fee_usdt": -25.0,
            "settled_funding_fee": -25.0,
        "funding_bill_count": 1,
        "funding_fee_source": "okx_account_bills",
        "funding_evidence_complete": True,
        "funding_evidence_eligible": True,
        "funding_evidence_gaps": [],
    }

    result = apply_dynamic_exit(_decision(), [position])

    assert result.funding_loss_budget_crossed is True
    assert result.hard_risk is True
    assert result.stop_risk_usage == 1.0
    assert result.close_fraction == 1.0
    assert result.eligible is True


def test_next_adverse_funding_charge_cannot_exceed_remaining_risk_budget() -> None:
    decision = _decision()
    decision.feature_snapshot = {
        "funding_rate": 0.15,
        "funding_data_available": True,
        "funding_interval_minutes": 60.0,
        "funding_rate_observed_at": "2026-08-16T08:00:00+00:00",
    }
    position = _position(
        current_price=100.0,
        notional_usdt=1000.0,
        unrealized_pnl=0.0,
        peak_unrealized_pnl=0.0,
    )

    result = apply_dynamic_exit(decision, [position])

    assert result.funding_cost_projection_eligible is True
    assert result.projected_funding_cost_usdt == pytest.approx(150.0)
    assert result.projected_funding_budget_crossed is True
    assert result.hard_risk is True
    assert result.close_fraction == 1.0
    assert result.eligible is True


def test_early_non_hard_partial_exit_requires_price_move_to_cover_exit_cost() -> None:
    position = _position(
        created_at=datetime.now(UTC) - timedelta(minutes=2),
        current_price=99.99,
        notional_usdt=999.9,
        unrealized_pnl=-0.1,
        peak_unrealized_pnl=0.0,
    )

    result = apply_dynamic_exit(_decision(), [position])

    assert result.hard_risk is False
    assert result.planned_stop_crossed is False
    assert result.early_exit_observation_active is True
    assert result.estimated_exit_cost_usdt > abs(result.gross_unrealized_pnl_usdt)
    assert result.economic_exit_evidence_complete is False
    assert result.close_fraction == 0.0
    assert "minimum_position_observation_not_elapsed" in result.reason


def test_early_loss_magnitude_does_not_count_as_exit_cost_coverage() -> None:
    position = _position(
        created_at=datetime.now(UTC) - timedelta(minutes=2),
        current_price=99.9,
        notional_usdt=99.9,
        quantity=1.0,
        entry_fee_usdt=0.05,
        unrealized_pnl=-0.1,
        peak_unrealized_pnl=0.0,
    )

    result = apply_dynamic_exit(_decision(), [position])

    assert result.hard_risk is False
    assert result.early_exit_observation_active is True
    assert abs(result.gross_unrealized_pnl_usdt) > result.estimated_exit_cost_usdt
    assert result.economic_exit_evidence_complete is False
    assert result.close_fraction == 0.0
    assert "minimum_position_observation_not_elapsed" in result.reason


@pytest.mark.parametrize("requested_fraction", [0.00449424, 0.06770709])
def test_early_small_model_exit_cannot_bypass_economic_debounce(
    requested_fraction: float,
) -> None:
    decision = _decision()
    decision.suggested_close_fraction = requested_fraction
    position = _position(
        created_at=datetime.now(UTC) - timedelta(minutes=2),
        current_price=100.0,
        notional_usdt=1000.0,
        unrealized_pnl=0.0,
        peak_unrealized_pnl=0.0,
    )

    result = apply_dynamic_exit(decision, [position])

    assert result.hard_risk is False
    assert result.early_exit_observation_active is True
    assert result.model_requested_close_fraction == pytest.approx(requested_fraction)
    assert result.model_exit_pressure < 0.1
    assert result.economic_exit_evidence_complete is False
    assert result.close_fraction == 0.0
    assert "minimum_position_observation_not_elapsed" in result.reason


def test_early_significant_model_exit_cannot_bypass_lifecycle_guard() -> None:
    decision = _decision()
    decision.suggested_close_fraction = 0.25
    position = _position(
        created_at=datetime.now(UTC) - timedelta(minutes=2),
        current_price=100.0,
        notional_usdt=1000.0,
        unrealized_pnl=0.0,
        peak_unrealized_pnl=0.0,
    )

    result = apply_dynamic_exit(decision, [position])

    assert result.hard_risk is False
    assert result.early_exit_observation_active is True
    assert result.model_exit_pressure == pytest.approx(0.2)
    assert result.economic_exit_evidence_complete is False
    assert result.eligible is False
    assert result.close_fraction == 0.0
    assert "minimum_position_observation_not_elapsed" in result.reason


def test_missing_position_age_fails_closed_for_non_hard_exit() -> None:
    position = _position(current_price=99.0, unrealized_pnl=-10.0)
    position.pop("created_at")

    result = apply_dynamic_exit(_decision(), [position])

    assert result.hard_risk is False
    assert result.position_age_evidence_complete is False
    assert result.eligible is False
    assert result.close_fraction == 0.0
    assert "position_age_evidence_missing" in result.reason


def test_post_observation_subminimum_exit_fraction_fails_closed() -> None:
    position = _position(
        created_at=datetime.now(UTC) - timedelta(minutes=20),
        current_price=99.98,
        notional_usdt=999.8,
        unrealized_pnl=-0.2,
        peak_unrealized_pnl=0.0,
    )

    result = apply_dynamic_exit(_decision(), [position])

    assert result.hard_risk is False
    assert result.early_exit_observation_active is False
    assert 0.0 < result.stop_risk_usage < 0.05
    assert result.economic_exit_evidence_complete is True
    assert result.close_fraction == 0.0
    assert "dynamic_exit_fraction_below_execution_minimum" in result.reason


def test_early_observation_never_blocks_planned_stop_crossing() -> None:
    position = _position(
        created_at=datetime.now(UTC) - timedelta(minutes=2),
        current_price=97.9,
        notional_usdt=979.0,
        unrealized_pnl=-21.0,
        peak_unrealized_pnl=0.0,
    )

    result = apply_dynamic_exit(_decision(), [position])

    assert result.hard_risk is True
    assert result.planned_stop_crossed is True
    assert result.early_exit_observation_active is False
    assert result.economic_exit_evidence_complete is True
    assert result.close_fraction == 1.0


def test_adverse_return_alignment_is_scaled_by_consumed_stop_budget() -> None:
    decision = _decision()
    decision.feature_snapshot = {
        "returns_1": -0.0002,
        "returns_5": -0.0004,
        "returns_20": -0.0008,
    }
    position = _position(
        current_price=99.9,
        notional_usdt=999.0,
        unrealized_pnl=-1.0,
        peak_unrealized_pnl=0.0,
    )

    result = apply_dynamic_exit(decision, [position])

    assert result.eligible is True
    assert 0.0 < result.stop_risk_usage < 1.0
    assert result.continuation_deterioration == pytest.approx(
        result.stop_risk_usage
    )
    assert result.close_fraction > result.stop_risk_usage
    assert result.close_fraction < 1.0
    assert result.policy_provenance["strategy_version"] == (
        "2026-08-19.dynamic-exit-lifecycle-guard.v15"
    )


def test_legacy_hard_risk_flag_without_position_economics_fails_closed() -> None:
    decision = _decision({"close_evidence": {"hard_risk": True}, "fast_risk_trigger": "stop_loss"})

    result = apply_dynamic_exit(decision, [])

    assert result.eligible is False
    assert result.hard_risk is False
    assert result.close_fraction == 0.0
    assert "position_economics_missing" in result.reason


def test_ordinary_exit_without_position_economics_fails_closed() -> None:
    decision = _decision()

    result = apply_dynamic_exit(decision, [])

    assert result.eligible is False
    assert result.close_fraction == 0.0
    assert "position_economics_missing" in result.reason


def test_low_priced_asset_stop_is_absolute_price_not_legacy_percent_guess() -> None:
    decision = _decision()
    position = _position(
        entry_price=0.5,
        current_price=0.39,
        stop_loss=0.4,
        stop_loss_pct=0.0,
        quantity=1_000.0,
        notional_usdt=390.0,
        unrealized_pnl=-110.0,
        peak_unrealized_pnl=0.0,
    )

    result = apply_dynamic_exit(decision, [position])

    assert result.eligible is True
    assert result.hard_risk is True
    assert result.planned_stop_crossed is True
    assert result.close_fraction == 1.0


def test_portfolio_concentration_continuously_amplifies_existing_exit_pressure() -> None:
    baseline = apply_dynamic_exit(_decision(), [_position()])
    concentrated = apply_dynamic_exit(
        _decision(),
        [
            _position(
                current_management_contract={
                    **_position()["current_management_contract"],
                    "portfolio_concentration_pressure": 0.5,
                }
            )
        ],
    )

    assert concentrated.portfolio_exposure_pressure == pytest.approx(0.25)
    assert concentrated.close_fraction > baseline.close_fraction


def test_non_hard_exit_fails_closed_without_current_management_contract() -> None:
    result = apply_dynamic_exit(
        _decision(),
        [_position(current_management_contract={})],
    )

    assert result.eligible is False
    assert result.current_management_contract_complete is False
    assert "current_position_management_contract_incomplete" in result.reason


def test_non_hard_exit_fails_closed_when_position_changed_after_contract_refresh() -> None:
    position = _position()
    position["quantity"] = 5.0
    position["notional_usdt"] = 505.0

    result = apply_dynamic_exit(_decision(), [position])

    assert result.eligible is False
    assert result.current_management_contract_complete is False
    assert "current_position_management_contract_incomplete" in result.reason


def test_expired_paper_canary_horizon_does_not_force_full_close() -> None:
    position = _position(
        execution_mode="paper",
        peak_unrealized_pnl=10.0,
        paper_canary_lifecycle={
            "version": PAPER_BOOTSTRAP_POSITION_LIFECYCLE_VERSION,
            "kind": "paper_bootstrap_canary_position",
            "authorized": True,
            "execution_scope": "paper_only",
            "production_permission": False,
            "symbol": "BTC/USDT",
            "side": "long",
            "horizon_minutes": 10,
            "expires_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        },
    )

    result = apply_dynamic_exit(_decision(), [position])

    assert result.eligible is False
    assert result.paper_canary_horizon_elapsed is True
    assert result.close_fraction == 0.0
    assert "dynamic_exit_pressure_zero" in result.reason


def test_expired_paper_training_horizon_cannot_bypass_incomplete_takeover_evidence() -> None:
    position = _position(
        execution_mode="paper",
        current_management_contract={
            "kind": "current_position_takeover",
            "management_eligible": False,
            "blockers": ["okx_protection_evidence_incomplete"],
        },
        paper_training_lifecycle={
            "version": PAPER_TRAINING_POSITION_LIFECYCLE_VERSION,
            "kind": "normal_paper_training_position",
            "authorized": True,
            "execution_scope": "paper_only",
            "production_permission": False,
            "symbol": "BTC/USDT",
            "side": "long",
            "horizon_minutes": 10.0,
            "expires_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
            "continuous_training_after_settlement": True,
            "loss_tolerant_for_training": True,
        },
    )

    result = apply_dynamic_exit(_decision(), [position])

    assert result.eligible is False
    assert result.paper_training_horizon_elapsed is True
    assert result.paper_training_horizon_minutes == 10.0
    assert result.current_management_contract_complete is False
    assert result.close_fraction == 0.0
    assert "current_position_management_contract_incomplete" in result.reason


def test_expired_paper_training_horizon_is_observation_only() -> None:
    baseline = apply_dynamic_exit(
        _decision(),
        [
            _position(
                execution_mode="paper",
                current_price=100.0,
                notional_usdt=1000.0,
                unrealized_pnl=0.0,
                peak_unrealized_pnl=0.0,
            )
        ],
    )
    position = _position(
        execution_mode="paper",
        current_price=100.0,
        notional_usdt=1000.0,
        unrealized_pnl=0.0,
        peak_unrealized_pnl=0.0,
        paper_training_lifecycle={
            "version": PAPER_TRAINING_POSITION_LIFECYCLE_VERSION,
            "kind": "normal_paper_training_position",
            "authorized": True,
            "execution_scope": "paper_only",
            "production_permission": False,
            "symbol": "BTC/USDT",
            "side": "long",
            "horizon_minutes": 10.0,
            "expires_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
            "continuous_training_after_settlement": True,
            "loss_tolerant_for_training": True,
        },
    )

    result = apply_dynamic_exit(_decision(), [position])

    assert result.eligible is baseline.eligible
    assert result.current_management_contract_complete is True
    assert result.paper_training_horizon_elapsed is True
    assert result.close_fraction == baseline.close_fraction
    assert result.reason == baseline.reason


def test_expired_paper_canary_horizon_cannot_bypass_incomplete_takeover_contract() -> None:
    position = _position(
        execution_mode="paper",
        current_management_contract={
            "kind": "current_position_takeover",
            "management_eligible": False,
            "blockers": ["okx_protection_evidence_incomplete"],
        },
        paper_canary_lifecycle={
            "version": PAPER_BOOTSTRAP_POSITION_LIFECYCLE_VERSION,
            "kind": "paper_bootstrap_canary_position",
            "authorized": True,
            "execution_scope": "paper_only",
            "production_permission": False,
            "symbol": "BTC/USDT",
            "side": "long",
            "horizon_minutes": 10,
            "expires_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        },
    )

    result = apply_dynamic_exit(_decision(), [position])

    assert result.eligible is False
    assert result.paper_canary_horizon_elapsed is True
    assert result.current_management_contract_complete is False
    assert result.close_fraction == 0.0
    assert "current_position_management_contract_incomplete" in result.reason


def test_ordinary_position_is_not_closed_only_because_it_is_old() -> None:
    result = apply_dynamic_exit(
        _decision(),
        [
            _position(
                execution_mode="paper",
                created_at=datetime.now(UTC) - timedelta(days=1),
                peak_unrealized_pnl=10.0,
            )
        ],
    )

    assert result.paper_canary_horizon_elapsed is False
    assert result.close_fraction == 0.0
