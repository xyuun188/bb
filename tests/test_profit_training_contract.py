from __future__ import annotations

from services.profit_training_contract import validate_profit_training_sample


def _closed_trade_sample(**overrides):
    sample = {
        "symbol": "BTC/USDT",
        "side": "long",
        "entry_order_id": "entry-1",
        "close_order_id": "close-1",
        "entry_price": 100.0,
        "close_price": 99.0,
        "quantity": 0.01,
        "notional": 1.0,
        "entry_fee": -0.001,
        "close_fee": -0.001,
        "funding_fee": 0.0,
        "slippage": 0.002,
        "realized_pnl": -0.012,
        "net_return_after_all_cost_pct": -1.2,
        "holding_minutes": 30.0,
        "decision_authority": "rules",
        "model_shadow_prediction": {"side": "short"},
    }
    sample.update(overrides)
    return sample


def test_profit_training_contract_accepts_closed_loss_as_training_label() -> None:
    contract = validate_profit_training_sample(_closed_trade_sample())

    assert contract.eligible is True
    assert contract.outcome == "loss"
    assert contract.target == "net_return_after_all_cost_pct"
    assert contract.target_value == -1.2
    assert contract.model_shadow_alignment == "avoided_losing_side"


def test_profit_training_contract_accepts_okx_lifecycle_aliases() -> None:
    contract = validate_profit_training_sample(
        {
            "symbol": "ETH/USDT",
            "side": "short",
            "entry_order_ids": ["entry-okx"],
            "close_order_ids": ["close-okx"],
            "entry_price": 100.0,
            "exit_price": 102.0,
            "quantity": 3.0,
            "notional_usdt": 100.0,
            "fee": -0.1,
            "funding_fee": -0.05,
            "realized_pnl": -2.2,
            "hold_minutes": 45.0,
            "decision_authority": "system",
        }
    )

    assert contract.eligible is True
    assert contract.outcome == "loss"
    assert contract.target_value == -2.2

def test_profit_training_contract_rejects_missing_close_order() -> None:
    contract = validate_profit_training_sample(
        _closed_trade_sample(close_order_id="")
    )

    assert contract.eligible is False
    assert contract.reason == "close_order_id_missing"
    assert "close_order_id_missing" in contract.blockers


def test_profit_training_contract_marks_model_supporting_losing_side() -> None:
    contract = validate_profit_training_sample(
        _closed_trade_sample(model_shadow_prediction={"side": "long"})
    )

    assert contract.eligible is True
    assert contract.model_shadow_alignment == "supported_losing_side"
