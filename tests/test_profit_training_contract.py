from __future__ import annotations

from services.okx_execution_slippage import OKX_ROUND_TRIP_SLIPPAGE_SOURCE
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
        "notional_source": "okx_entry_fill_base_quantity_and_average_price",
        "entry_fee": 0.001,
        "close_fee": 0.001,
        "entry_fee_source": "okx_fills_history",
        "close_fee_source": "okx_fills_history",
        "funding_fee": 0.0,
        "slippage": 0.002,
        "slippage_source": OKX_ROUND_TRIP_SLIPPAGE_SOURCE,
        "realized_pnl": -0.012,
        "net_return_after_all_cost_pct": -1.2,
        "holding_minutes": 30.0,
        "pnl_source": "okx_position_history_realized_pnl",
        "funding_fee_source": "okx_positions_history.fundingFee",
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


def test_profit_training_contract_rejects_legacy_lifecycle_aliases() -> None:
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

    assert contract.eligible is False
    assert contract.outcome == "invalid"
    assert "entry_order_id_missing" in contract.blockers
    assert "close_order_id_missing" in contract.blockers
    assert "close_price_missing_or_invalid" in contract.blockers
    assert "notional_missing_or_invalid" in contract.blockers
    assert "entry_fee_missing_or_invalid" in contract.blockers
    assert "close_fee_missing_or_invalid" in contract.blockers
    assert "slippage_missing_or_invalid" in contract.blockers
    assert "holding_minutes_missing_or_invalid" in contract.blockers
    assert "net_return_after_all_cost_pct_missing_or_invalid" in contract.blockers


def test_profit_training_contract_rejects_mismatched_profit_target() -> None:
    contract = validate_profit_training_sample(
        _closed_trade_sample(net_return_after_all_cost_pct=-0.5)
    )

    assert contract.eligible is False
    assert "net_return_target_algebra_mismatch" in contract.blockers


def test_profit_training_contract_rejects_missing_close_order() -> None:
    contract = validate_profit_training_sample(
        _closed_trade_sample(close_order_id="")
    )

    assert contract.eligible is False
    assert contract.reason == "close_order_id_missing"
    assert "close_order_id_missing" in contract.blockers


def test_profit_training_contract_rejects_execution_result_fee_source() -> None:
    contract = validate_profit_training_sample(
        _closed_trade_sample(entry_fee_source="okx_execution_result")
    )

    assert contract.eligible is False
    assert "entry_fee_source_not_authoritative" in contract.blockers


def test_profit_training_contract_marks_model_supporting_losing_side() -> None:
    contract = validate_profit_training_sample(
        _closed_trade_sample(model_shadow_prediction={"side": "long"})
    )

    assert contract.eligible is True
    assert contract.model_shadow_alignment == "supported_losing_side"
