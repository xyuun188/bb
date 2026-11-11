import pytest

from data_feed.feature_vector import FeatureVector
from services.execution_cost_model import (
    SHADOW_SETTLEMENT_COST_VERSION,
    attach_execution_cost_facts,
    execution_cost_estimate,
    shadow_execution_cost_estimate,
)


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

    assert estimate.slippage_pct == pytest.approx(0.017152)
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

    assert estimate.slippage_pct == pytest.approx(4.68)
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
        + (estimate.reference_price - 99.0) / estimate.reference_price * 100.0
    )
    assert estimate.book_levels_consumed == 2
    assert estimate.order_notional_usdt == 150.0
    assert estimate.policy_provenance["orderbook_fingerprint"] == "book-fingerprint"
    assert estimate.policy_provenance["exit_slippage_pct"] == pytest.approx(1.0)


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


def test_round_trip_cost_requires_close_side_depth() -> None:
    snapshot = {
        "bid": 99.99,
        "ask": 100.01,
        "orderbook_bids": [[99.99, 1.0]],
        "orderbook_asks": [[100.01, 10.0]],
        "contract_value_base": 1.0,
        "planned_order_notional_usdt": 500.0,
        "planned_order_side": "long",
        "taker_fee_rate": 0.0004,
    }
    estimate = execution_cost_estimate(snapshot)
    assert estimate.production_eligible is False
    assert estimate.reason == "planned_close_exceeds_observed_orderbook_depth"


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


def test_maker_rate_is_attached_to_the_feature_snapshot() -> None:
    feature = FeatureVector(
        symbol="BTC/USDT",
        spread_pct=0.01,
        orderbook_bid_depth=10_000.0,
        orderbook_ask_depth=9_000.0,
    )

    attach_execution_cost_facts(
        feature,
        {"taker_fee_rate": 0.0005, "maker_fee_rate": 0.0002, "exit_fee_rate": 0.0005},
    )

    snapshot = feature.to_dict()
    assert snapshot["maker_fee_rate"] == pytest.approx(0.0002)


def test_entry_leg_stays_on_the_taker_basis_without_a_measured_fill_rate() -> None:
    # Post-only is armed, but no verified maker fill rate exists yet: the entry
    # leg must still be priced at the taker rate.
    estimate = execution_cost_estimate(
        {
            "spread_pct": 0.01,
            "taker_fee_rate": 0.0005,
            "maker_fee_rate": 0.0002,
            "entry_execution_style": "post_only_with_market_fallback",
            "post_only_fill_probability": 0.0,
        }
    )

    assert estimate.fee_pct == pytest.approx(0.1)
    assert "taker_entry_no_measured_maker_fill_rate" in estimate.policy_provenance["source"]
    assert estimate.production_eligible is True


def test_entry_leg_credits_the_maker_rate_by_the_measured_fill_rate() -> None:
    estimate = execution_cost_estimate(
        {
            "spread_pct": 0.01,
            "taker_fee_rate": 0.0005,
            "maker_fee_rate": 0.0002,
            "entry_execution_style": "post_only_with_market_fallback",
            "post_only_fill_probability": 0.5,
        }
    )

    # entry: 0.5*0.02 + 0.5*0.05 = 0.035 ; exit: 0.05 -> 0.085
    assert estimate.fee_pct == pytest.approx(0.085)
    assert "post_only_measured_maker_share" in estimate.policy_provenance["source"]


def test_maker_rate_is_ignored_for_a_plain_taker_entry() -> None:
    estimate = execution_cost_estimate(
        {
            "spread_pct": 0.01,
            "taker_fee_rate": 0.0005,
            "maker_fee_rate": 0.0002,
            "entry_execution_style": "market",
            "post_only_fill_probability": 1.0,
        }
    )

    # A market entry never earns the maker rate, whatever the prior says.
    assert estimate.fee_pct == pytest.approx(0.1)
    assert "snapshot_exchange_fee_rates.taker_entry+" in estimate.policy_provenance["source"]


def _settled_snapshot() -> dict:
    return {
        "bid": 99.99, "ask": 100.01,
        "taker_fee_rate": 0.0004,
        "shadow_settlement_cost_observation": {
            "version": SHADOW_SETTLEMENT_COST_VERSION,
            "used_as_decision_time_features": False,
            "market_fact_verified": True,
            "result_fact_id": "exit-fact",
            "source_timestamp_ms": 1794355200000,
            "received_at": "2026-11-11T00:00:01+00:00",
            "scheduled_label_at": "2026-11-11T00:00:00+00:00",
            "entry_price": 100.0,
            "exit_price": 110.0,
            "market_snapshot": {
                "bid": 109.89, "ask": 110.11,
                "taker_fee_rate": 0.0006,
            },
        },
    }


def test_shadow_labels_use_exit_spread_and_exit_fee_on_exit_notional() -> None:
    snapshot = _settled_snapshot()
    estimate = shadow_execution_cost_estimate(snapshot, side="long")

    assert estimate.production_eligible is True
    assert estimate.fee_pct == pytest.approx(0.04 + 1.1 * 0.06)
    assert estimate.slippage_pct == pytest.approx(0.01 + 1.1 * 0.1)
    assert estimate.policy_provenance["label_cost_basis"] == (
        "entry_and_observed_exit_microstructure"
    )
    assert estimate.policy_provenance["actual_execution"] is False
    assert estimate.policy_provenance["execution_replay_complete"] is False
    # The same stored observation must never change the decision-time projection.
    assert execution_cost_estimate(snapshot).total_pct == pytest.approx(0.1)
    assert snapshot["taker_fee_rate"] == 0.0004


@pytest.mark.parametrize("change, reason", [
    ({"version": "unknown"}, "shadow_exit_cost_version_unknown"),
    ({"version": None}, "shadow_exit_cost_version_unknown"),
    ({"market_fact_verified": False}, "shadow_exit_market_fact_unverified"),
    ({"used_as_decision_time_features": True}, "shadow_exit_observation_not_label_only"),
    ({"result_fact_id": ""}, "shadow_exit_fact_identity_missing"),
    ({"source_timestamp_ms": None}, "shadow_exit_clock_missing"),
    ({"scheduled_label_at": "2026-11-11T00:01:00Z"}, "shadow_exit_clock_invalid"),
    ({"scheduled_label_at": "2026-11-10T23:58:00Z"}, "shadow_exit_observation_late"),
    ({"exit_price": float("nan")}, "shadow_exit_reference_price_missing"),
    ({"market_snapshot": {}}, "shadow_exit:fee_rate_missing"),
])
def test_invalid_observed_exit_does_not_fall_back_to_entry_projection(change, reason) -> None:
    snapshot = _settled_snapshot()
    snapshot["shadow_settlement_cost_observation"].update(change)

    estimate = shadow_execution_cost_estimate(snapshot, side="short")

    assert estimate.production_eligible is False
    assert estimate.reason == reason


def test_shadow_observation_does_not_invent_a_maker_fill() -> None:
    snapshot = _settled_snapshot()
    snapshot.update({
        "maker_fee_rate": 0.0001,
        "entry_execution_style": "post_only_with_market_fallback",
        "post_only_fill_probability": 1.0,
    })

    assert shadow_execution_cost_estimate(snapshot, side="long").fee_pct == (
        pytest.approx(0.106)
    )


def test_shadow_exit_observation_must_match_the_market_path_identity() -> None:
    snapshot = _settled_snapshot()
    snapshot["training_market_fact_contract"] = {"result_fact_id": "different-exit"}

    assert shadow_execution_cost_estimate(snapshot, side="long").reason == (
        "shadow_exit_fact_identity_mismatch"
    )


def test_shadow_cost_uses_only_entry_and_eventual_close_depth_with_same_quantity() -> None:
    snapshot = _settled_snapshot()
    snapshot.update({
        "planned_order_notional_usdt": 100.0,
        "contract_value_base": 1.0,
        "orderbook_asks": [[100.01, 2.0]],
    })
    snapshot["shadow_settlement_cost_observation"]["market_snapshot"].update({
        "contract_value_base": 1.0,
        "orderbook_bids": [[109.89, 1.0]],
    })

    estimate = shadow_execution_cost_estimate(snapshot, side="long")

    assert estimate.production_eligible is True
    assert estimate.order_size_complete is True
    assert estimate.slippage_pct == pytest.approx(0.12)
    snapshot["shadow_settlement_cost_observation"]["market_snapshot"]["orderbook_bids"] = (
        [[109.89, 0.99]]
    )
    assert shadow_execution_cost_estimate(snapshot, side="long").production_eligible is False
