import pytest

from services.profit_lock_execution import profit_lock_execution_required, profit_lock_limit


def _raw():
    return {"dynamic_exit_policy": {
        "lifecycle_net_pnl_usdt": 1.0, "profit_lock_pressure": 0.25,
    }}


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_limit_fills_have_positive_pnl_after_fees_and_adverse_funding(side):
    plan = profit_lock_limit(
        side=side, entry_price=100.0, quantity=10.0,
        entry_fee_usdt=0.5, exit_fee_rate=0.0005,
        tick_size=0.01, adverse_funding_usdt=0.2,
    )
    price = plan["limit_price"]
    gross = (100 - price) * 10 if side == "buy" else (price - 100) * 10
    assert gross - 0.5 - price * 10 * 0.0005 - 0.2 > 0
    assert plan["eligible"] is True
    assert plan["market_fallback_allowed"] is False


def test_sand_false_mark_profit_cannot_cross_the_limit():
    plan = profit_lock_limit(
        side="buy", entry_price=0.0358, quantity=14770.0,
        entry_fee_usdt=0.264383, exit_fee_rate=0.0005, tick_size=0.0001,
    )
    assert plan["limit_price"] < 0.0358 < 0.0359


@pytest.mark.parametrize("key", [
    "hard_risk", "planned_stop_crossed", "funding_loss_budget_crossed",
    "projected_funding_budget_crossed", "stop_risk_usage",
    "continuation_deterioration", "opposite_pressure", "model_exit_pressure",
])
def test_risk_exit_is_not_blocked_by_profit_limit(key):
    raw = _raw()
    raw["dynamic_exit_policy"][key] = 1.0
    assert profit_lock_execution_required(raw) is False


def test_profit_limit_is_only_required_for_discretionary_profitable_exit():
    raw = _raw()
    assert profit_lock_execution_required(raw) is True
    raw["manual_close"] = True
    assert profit_lock_execution_required(raw) is False
    assert profit_lock_execution_required({}) is False


def test_small_settled_funding_cost_cannot_bypass_profitable_exit_price_limit():
    raw = _raw()
    raw["dynamic_exit_policy"].update({
        "gross_unrealized_pnl_usdt": 2.37262575,
        "settled_funding_fee": -0.11263427,
        "stop_risk_usage": 0.01834915,
        "continuation_deterioration": 0.0,
        "lifecycle_net_pnl_usdt": 1.98076796,
    })

    assert profit_lock_execution_required(raw) is True


def test_price_loss_remains_a_risk_exit_even_when_funding_makes_total_positive():
    raw = _raw()
    raw["dynamic_exit_policy"].update({
        "gross_unrealized_pnl_usdt": -2.0,
        "settled_funding_fee": 4.0,
        "stop_risk_usage": 0.5,
    })

    assert profit_lock_execution_required(raw) is False


@pytest.mark.parametrize("overrides", [
    {"exit_fee_rate": -1.0}, {"exit_fee_rate": float("nan")},
    {"entry_fee_usdt": -1.0}, {"tick_size": 0.0}, {"quantity": 0.0},
])
def test_missing_cost_or_contract_evidence_rejects_profit_order(overrides):
    kwargs = dict(side="buy", entry_price=100.0, quantity=10.0,
                  entry_fee_usdt=0.5, exit_fee_rate=0.0005, tick_size=0.01)
    kwargs.update(overrides)
    assert profit_lock_limit(**kwargs)["eligible"] is False
