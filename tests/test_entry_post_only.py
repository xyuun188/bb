from __future__ import annotations

import pytest

from services.entry_post_only import (
    ENTRY_STYLE_MARKET,
    ENTRY_STYLE_POST_ONLY_WITH_FALLBACK,
    FILL_FULL,
    FILL_NONE,
    FILL_PARTIAL,
    MAX_POST_ONLY_TIMEOUT_SECONDS,
    MIN_POST_ONLY_TIMEOUT_SECONDS,
    clamp_timeout_seconds,
    classify_passive_fill,
    expected_post_only_round_trip_cost_pct,
    is_terminal_unfilled_state,
    passive_entry_price,
    plan_entry_post_only,
    post_only_execution_fact,
)


def test_passive_price_joins_the_touch_for_both_sides() -> None:
    assert passive_entry_price(side="buy", bid=99.95, ask=100.05) == pytest.approx(99.95)
    assert passive_entry_price(side="sell", bid=99.95, ask=100.05) == pytest.approx(100.05)
    assert passive_entry_price(side="long", bid=10.0, ask=10.2) == pytest.approx(10.0)
    assert passive_entry_price(side="short", bid=10.0, ask=10.2) == pytest.approx(10.2)


def test_passive_price_refuses_a_crossed_or_incomplete_book() -> None:
    assert passive_entry_price(side="buy", bid=100.1, ask=100.0) == 0.0
    assert passive_entry_price(side="buy", bid=0.0, ask=100.0) == 0.0
    assert passive_entry_price(side="buy", bid=99.0, ask=0.0) == 0.0
    assert passive_entry_price(side="hold", bid=99.0, ask=100.0) == 0.0


def test_timeout_is_clamped_to_a_bounded_window() -> None:
    assert clamp_timeout_seconds(0) == MIN_POST_ONLY_TIMEOUT_SECONDS
    assert clamp_timeout_seconds(-5) == MIN_POST_ONLY_TIMEOUT_SECONDS
    assert clamp_timeout_seconds(6.0) == pytest.approx(6.0)
    assert clamp_timeout_seconds(10_000) == MAX_POST_ONLY_TIMEOUT_SECONDS


def test_plan_is_ineligible_when_disabled_or_the_touch_is_unavailable() -> None:
    disabled = plan_entry_post_only(
        enabled=False, side="buy", bid=99.0, ask=100.0, timeout_seconds=6.0
    )
    assert disabled["eligible"] is False
    assert disabled["reason"] == "post_only_disabled"

    no_book = plan_entry_post_only(
        enabled=True, side="buy", bid=0.0, ask=0.0, timeout_seconds=6.0
    )
    assert no_book["eligible"] is False
    assert no_book["reason"] == "post_only_touch_unavailable"


def test_plan_returns_the_touch_price_when_enabled() -> None:
    plan = plan_entry_post_only(
        enabled=True, side="sell", bid=99.95, ask=100.05, timeout_seconds=4.0
    )
    assert plan["eligible"] is True
    assert plan["reason"] == "post_only_touch_ready"
    assert plan["price"] == pytest.approx(100.05)
    assert plan["timeout_seconds"] == pytest.approx(4.0)


def test_fill_classification_separates_full_partial_and_none() -> None:
    assert classify_passive_fill(filled_contracts=10, requested_contracts=10) == FILL_FULL
    assert classify_passive_fill(filled_contracts=4, requested_contracts=10) == FILL_PARTIAL
    assert classify_passive_fill(filled_contracts=0, requested_contracts=10) == FILL_NONE
    assert classify_passive_fill(filled_contracts=5, requested_contracts=0) == FILL_NONE
    # Float noise just under the requested size is still a full fill.
    assert (
        classify_passive_fill(filled_contracts=10 - 1e-12, requested_contracts=10)
        == FILL_FULL
    )


def test_terminal_unfilled_state_detection() -> None:
    assert is_terminal_unfilled_state("canceled") is True
    assert is_terminal_unfilled_state("REJECTED") is True
    assert is_terminal_unfilled_state("filled") is False
    assert is_terminal_unfilled_state("live") is False
    assert is_terminal_unfilled_state(None) is False


def test_cost_estimate_is_fail_closed_without_fee_evidence() -> None:
    result = expected_post_only_round_trip_cost_pct(
        maker_fee_rate=0.0,
        taker_fee_rate=0.0005,
        passive_fill_probability=0.5,
    )
    assert result["eligible"] is False
    assert result["expected_total_pct"] is None
    assert result["reason"] == "maker_or_taker_fee_missing"


def test_cost_estimate_stays_on_the_taker_basis_at_zero_fill_probability() -> None:
    result = expected_post_only_round_trip_cost_pct(
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
        passive_fill_probability=0.0,
    )
    assert result["eligible"] is True
    # 0.05% entry + 0.05% exit, no maker credit assumed.
    assert result["expected_total_pct"] == pytest.approx(0.1)
    assert result["maker_path_pct"] == pytest.approx(0.07)


def test_cost_estimate_blends_maker_and_fallback_paths_by_measured_rate() -> None:
    result = expected_post_only_round_trip_cost_pct(
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
        passive_fill_probability=0.5,
    )
    # 0.5 * (0.02 + 0.05) + 0.5 * (0.05 + 0.05) = 0.035 + 0.05
    assert result["expected_total_pct"] == pytest.approx(0.085)


def test_cost_estimate_includes_fallback_entry_slippage_only_on_the_miss_path() -> None:
    result = expected_post_only_round_trip_cost_pct(
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
        passive_fill_probability=1.0,
        fallback_entry_slippage_pct=0.2,
    )
    # A certain maker fill never pays the fallback slippage.
    assert result["expected_total_pct"] == pytest.approx(0.07)

    miss = expected_post_only_round_trip_cost_pct(
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
        passive_fill_probability=0.0,
        fallback_entry_slippage_pct=0.2,
    )
    assert miss["expected_total_pct"] == pytest.approx(0.3)


def test_execution_fact_records_style_fills_and_fallback_reason() -> None:
    plan = plan_entry_post_only(
        enabled=True, side="buy", bid=99.95, ask=100.05, timeout_seconds=6.0
    )
    fact = post_only_execution_fact(
        plan=plan,
        outcome="partial_then_market_fallback",
        fallback_reason="post_only_partial_timeout",
        maker_fill_contracts=4,
        taker_fill_contracts=6,
        post_only_order_id=" 12345 ",
    )
    assert fact["entry_execution_style"] == ENTRY_STYLE_POST_ONLY_WITH_FALLBACK
    assert fact["post_only_attempted"] is True
    assert fact["post_only_order_id"] == "12345"
    assert fact["maker_fill_contracts"] == pytest.approx(4.0)
    assert fact["taker_fill_contracts"] == pytest.approx(6.0)
    assert fact["fallback_reason"] == "post_only_partial_timeout"


def test_execution_fact_marks_a_plain_market_entry_when_not_attempted() -> None:
    fact = post_only_execution_fact(plan=None, outcome="not_attempted")
    assert fact["entry_execution_style"] == ENTRY_STYLE_MARKET
    assert fact["post_only_attempted"] is False
    assert fact["post_only_order_id"] is None
    assert fact["maker_fill_contracts"] == pytest.approx(0.0)
    assert isinstance(fact, dict)
