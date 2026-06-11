from __future__ import annotations

import pytest

from services.entry_stop_loss_budget import EntryStopLossBudgetPolicy


@pytest.mark.parametrize(
    ("risk_mode", "expected"),
    [
        ("normal", 16.0),
        ("drawdown_recovery", 8.0),
        ("defensive_recovery", 4.0),
        ("hard_recovery", 16.0),
        ("unknown", 16.0),
    ],
)
def test_default_budget_preserves_existing_risk_mode_mapping(
    risk_mode: str,
    expected: float,
) -> None:
    assert EntryStopLossBudgetPolicy().default_budget_usdt(risk_mode) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("balance", "expected"),
    [
        (100.0, 6.0),
        (2_000.0, 16.0),
        (10_000.0, 36.0),
    ],
)
def test_dynamic_hard_cap_uses_equity_floor_and_ceiling(
    balance: float,
    expected: float,
) -> None:
    assert EntryStopLossBudgetPolicy().dynamic_hard_cap_usdt(balance) == pytest.approx(expected)


def test_normal_budget_is_capped_by_dynamic_account_cap() -> None:
    budget = EntryStopLossBudgetPolicy().resolve(
        risk_mode="normal",
        configured_max_loss_usdt=0.0,
        balance=1_000.0,
        high_quality_entry=False,
        low_payoff_quality=False,
    )

    assert budget.dynamic_hard_cap_usdt == pytest.approx(8.0)
    assert budget.max_loss_usdt == pytest.approx(8.0)
    assert budget.risk_budget_boost is None


@pytest.mark.parametrize(
    ("risk_mode", "expected_from", "expected_to"),
    [
        ("drawdown_recovery", 8.0, 12.0),
        ("defensive_recovery", 4.0, 8.0),
    ],
)
def test_high_quality_recovery_signal_can_boost_stop_loss_budget(
    risk_mode: str,
    expected_from: float,
    expected_to: float,
) -> None:
    budget = EntryStopLossBudgetPolicy().resolve(
        risk_mode=risk_mode,
        configured_max_loss_usdt=0.0,
        balance=10_000.0,
        high_quality_entry=True,
        low_payoff_quality=False,
    )

    assert budget.max_loss_usdt == pytest.approx(expected_to)
    assert budget.risk_budget_boost == {
        "applied": True,
        "from_usdt": expected_from,
        "to_usdt": expected_to,
        "reason": "当前为高质量同向机会，亏损恢复模式下允许更合理的单笔风险预算。",
    }


def test_low_payoff_quality_forces_defensive_budget_without_boost() -> None:
    budget = EntryStopLossBudgetPolicy().resolve(
        risk_mode="normal",
        configured_max_loss_usdt=16.0,
        balance=10_000.0,
        high_quality_entry=True,
        low_payoff_quality=True,
    )

    assert budget.max_loss_usdt == pytest.approx(4.0)
    assert budget.risk_budget_boost is None


def test_explicit_tiny_budget_keeps_final_minimum_loss_floor() -> None:
    budget = EntryStopLossBudgetPolicy().resolve(
        risk_mode="normal",
        configured_max_loss_usdt=0.10,
        balance=10_000.0,
        high_quality_entry=False,
        low_payoff_quality=False,
    )

    assert budget.configured_max_loss_usdt == pytest.approx(0.10)
    assert budget.max_loss_usdt == pytest.approx(1.0)
