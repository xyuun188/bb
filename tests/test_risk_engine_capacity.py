import pytest

from ai_brain.base_model import Action, DecisionOutput
from risk_manager.engine import RiskEngine


def _decision(symbol: str = "BTC/USDT", action: Action = Action.LONG) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol=symbol,
        action=action,
        confidence=0.8,
        reasoning="entry",
        position_size_pct=0.03,
        suggested_leverage=12.0,
        stop_loss_pct=0.03,
        raw_response={
            "profit_risk_sizing": {
                "production_eligible": True,
                "planned_stop_loss_usdt": 0.9,
                "max_stop_loss_usdt": 1.2,
                "stress_stop_loss_pct": 0.03,
                "policy_provenance": {
                    "source": "fee_after_return_cost_stop_distance_account_and_portfolio_state",
                    "observation_window": "current_decision_and_open_portfolio",
                    "sample_count": 2,
                    "generated_at": "2026-07-12T00:00:00+00:00",
                    "strategy_version": "test.dynamic-risk.v1",
                    "fallback_reason": "",
                },
            }
        },
    )


def test_position_group_count_cannot_block_dynamic_return_entry() -> None:
    result = RiskEngine().assess(
        decision=_decision("SOL/USDT"),
        current_positions=[
            {
                "model_name": "ensemble_trader",
                "symbol": f"ASSET-{index}/USDT",
                "side": "long",
                "quantity": 0.001,
                "entry_price": 1.0,
                "margin": 0.001,
            }
            for index in range(500)
        ],
        account_balance=1000.0,
    )

    assert result.approved is True
    assert result.decision is not None
    assert result.decision.suggested_leverage == 12.0


def test_missing_dynamic_risk_contract_fails_closed_despite_obsolete_capacity_payload() -> None:
    decision = _decision("SOL/USDT")
    decision.raw_response = {
        "dynamic_position_capacity": {
            "entry_limit": 999,
            "available_group_slots": 999,
        }
    }

    result = RiskEngine().assess(decision, [], 1000.0)

    assert result.approved is False
    assert "Dynamic account risk budget" in result.rejection_reason


def test_same_symbol_opposite_entry_is_still_rejected_by_okx_net_position_fact() -> None:
    result = RiskEngine().assess(
        decision=_decision("MASK/USDT", Action.LONG),
        current_positions=[
            {
                "model_name": "ensemble_trader",
                "symbol": "MASK/USDT",
                "side": "short",
                "quantity": 47.0,
                "entry_price": 0.4103,
                "margin": 2.0,
            }
        ],
        account_balance=1000.0,
    )

    assert result.approved is False
    assert "OKX 净持仓模式" in result.rejection_reason


def test_physical_account_margin_boundary_caps_to_remaining_equity() -> None:
    result = RiskEngine().assess(
        decision=_decision("ETH/USDT"),
        current_positions=[
            {
                "model_name": "ensemble_trader",
                "symbol": "BTC/USDT",
                "side": "long",
                "quantity": 1.0,
                "entry_price": 1000.0,
                "margin": 990.0,
            }
        ],
        account_balance=1000.0,
    )

    assert result.approved is True
    assert result.decision is not None
    assert result.decision.position_size_pct == pytest.approx(0.01)
    assert any("current account capacity" in warning for warning in result.warnings)
