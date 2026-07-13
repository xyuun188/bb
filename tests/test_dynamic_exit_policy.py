import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.dynamic_exit_policy import apply_dynamic_exit


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
        "entry_fee_usdt": 0.5,
    }
    position.update(overrides)
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
        "current_position_fee_after_pnl_peak_planned_stop_and_market_returns"
    )
    assert result.policy_provenance["fallback_reason"] == ""
    assert result.execution_cost_complete is True


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


def test_legacy_hard_risk_flag_without_position_economics_fails_closed() -> None:
    decision = _decision(
        {"close_evidence": {"hard_risk": True}, "fast_risk_trigger": "stop_loss"}
    )

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
