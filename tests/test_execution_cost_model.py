import pytest

from services.execution_cost_model import execution_cost_estimate


def test_execution_cost_uses_dynamic_spread() -> None:
    estimate = execution_cost_estimate(
        {
            "spread_pct": 0.015,
            "orderbook_bid_depth": 120_000.0,
            "orderbook_ask_depth": 110_000.0,
            "orderbook_imbalance": 0.10,
            "taker_fee_rate": 0.0004,
        }
    )

    assert estimate.slippage_pct == pytest.approx(0.008576)
    assert estimate.slippage_source == "dynamic_live_spread_depth_imbalance"
    assert estimate.production_eligible is True
    assert estimate.spread_source == "spread_pct"
    assert estimate.total_pct == estimate.fee_pct + estimate.slippage_pct


def test_execution_cost_reports_extreme_microstructure_without_fixed_cap() -> None:
    estimate = execution_cost_estimate(
        {
            "spread_pct": 2.4,
            "orderbook_bid_depth": 0.0,
            "orderbook_ask_depth": 0.0,
            "orderbook_imbalance": 0.95,
            "taker_fee_rate": 0.0004,
        }
    )

    assert estimate.slippage_pct == pytest.approx(2.34)
    assert estimate.slippage_source == "dynamic_live_spread_depth_imbalance"
    assert estimate.production_eligible is True


def test_execution_cost_can_derive_spread_from_bid_ask() -> None:
    estimate = execution_cost_estimate(
        {"bid": 99.95, "ask": 100.05, "taker_fee_rate": 0.0004}
    )

    assert estimate.spread_source == "bid_ask"
    assert estimate.spread_pct > 0
    assert estimate.slippage_pct >= 0.05


def test_execution_cost_missing_live_spread_is_observation_only() -> None:
    estimate = execution_cost_estimate(
        {"orderbook_bid_depth": 10_000.0, "taker_fee_rate": 0.0004}
    )

    assert estimate.production_eligible is False
    assert estimate.slippage_source == "observation_only_missing_live_spread"
    assert estimate.policy_provenance["fallback_reason"] == "live_spread_missing"


def test_execution_cost_missing_exchange_fee_rate_fails_closed() -> None:
    estimate = execution_cost_estimate({"spread_pct": 0.01})
    assert estimate.production_eligible is False
    assert estimate.fee_pct == 0.0
    assert estimate.policy_provenance["fallback_reason"] == "fee_rate_missing"


def test_exchange_fee_rate_contract_does_not_guess_percent_units() -> None:
    estimate = execution_cost_estimate(
        {"spread_pct": 0.01, "taker_fee_rate": 0.2}
    )

    assert estimate.fee_pct == pytest.approx(40.0)
