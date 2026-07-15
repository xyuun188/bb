import pytest

from data_feed.feature_vector import FeatureVector
from services.execution_cost_model import attach_execution_cost_facts, execution_cost_estimate


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
    assert estimate.slippage_source == "pre_sizing_live_spread_depth_imbalance"
    assert estimate.order_size_complete is False
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
    assert estimate.slippage_source == "pre_sizing_live_spread_depth_imbalance"
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


def test_execution_cost_consumes_orderbook_levels_for_planned_order_size() -> None:
    estimate = execution_cost_estimate(
        {
            "bid": 99.0,
            "ask": 101.0,
            "spread_pct": 2.0,
            "orderbook_bids": [[99.0, 2.0]],
            "orderbook_asks": [[101.0, 1.0], [102.0, 1.0]],
            "orderbook_bid_depth": 198.0,
            "orderbook_ask_depth": 203.0,
            "orderbook_imbalance": (198.0 - 203.0) / 401.0,
            "contract_value_base": 1.0,
            "planned_order_notional_usdt": 150.0,
            "planned_order_side": "long",
            "taker_fee_rate": 0.0004,
            "pre_order_execution_facts": {"input_fingerprint": "book-fingerprint"},
        }
    )

    assert estimate.production_eligible is True
    assert estimate.order_size_complete is True
    assert estimate.market_impact_pct > 0
    assert estimate.estimated_vwap == pytest.approx((101.0 + 51.0) / 1.5)
    assert estimate.slippage_pct == pytest.approx(
        (estimate.estimated_vwap - estimate.reference_price)
        / estimate.reference_price
        * 100.0
    )
    assert estimate.book_levels_consumed == 2
    assert estimate.order_notional_usdt == 150.0
    assert estimate.policy_provenance["orderbook_fingerprint"] == "book-fingerprint"


def test_execution_cost_fails_when_order_exceeds_observed_book_depth() -> None:
    estimate = execution_cost_estimate(
        {
            "bid": 99.0,
            "ask": 101.0,
            "orderbook_bids": [[99.0, 1.0]],
            "orderbook_asks": [[101.0, 1.0]],
            "orderbook_bid_depth": 99.0,
            "orderbook_ask_depth": 101.0,
            "contract_value_base": 1.0,
            "planned_order_notional_usdt": 500.0,
            "planned_order_side": "long",
            "taker_fee_rate": 0.0004,
        }
    )

    assert estimate.production_eligible is False
    assert estimate.order_size_complete is False
    assert estimate.reason == "planned_order_exceeds_observed_orderbook_depth"


def test_exchange_fee_rate_contract_does_not_guess_percent_units() -> None:
    estimate = execution_cost_estimate(
        {"spread_pct": 0.01, "taker_fee_rate": 0.2}
    )

    assert estimate.fee_pct == pytest.approx(40.0)


def test_account_fee_facts_are_attached_to_live_feature_snapshot() -> None:
    feature = FeatureVector(
        symbol="BTC/USDT",
        spread_pct=0.01,
        orderbook_bid_depth=10_000.0,
        orderbook_ask_depth=9_000.0,
    )

    attach_execution_cost_facts(
        feature,
        {
            "taker_fee_rate": 0.0004,
            "entry_fee_rate": 0.0004,
            "exit_fee_rate": 0.0004,
            "fee_rate_source": "okx_account_trade_fee.takerU",
            "fee_rate_observed_at": "2026-07-13T12:00:00+00:00",
            "policy_provenance": {"source": "okx_account_trade_fee_swap"},
        },
    )

    snapshot = feature.to_dict()
    assert snapshot["taker_fee_rate"] == pytest.approx(0.0004)
    assert snapshot["fee_policy_provenance"]["source"] == "okx_account_trade_fee_swap"
    assert execution_cost_estimate(snapshot).production_eligible is True
