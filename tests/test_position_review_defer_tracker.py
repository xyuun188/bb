from __future__ import annotations

from services.position_review_defer_tracker import PositionReviewDeferTracker


def test_position_review_defer_tracker_applies_and_clears_plan_counts() -> None:
    tracker = PositionReviewDeferTracker()
    key = ("ensemble_trader", "BTC/USDT")

    assert tracker.count(key) == 0

    tracker.apply_plan_count(key, 2)
    assert tracker.count(key) == 2

    tracker.apply_plan_count(key, 0)
    assert tracker.count(key) == 0
    assert tracker.counts == {}


def test_position_review_defer_tracker_clears_many_selected_keys() -> None:
    tracker = PositionReviewDeferTracker()
    btc = ("ensemble_trader", "BTC/USDT")
    eth = ("ensemble_trader", "ETH/USDT")
    sol = ("ensemble_trader", "SOL/USDT")
    tracker.apply_plan_count(btc, 1)
    tracker.apply_plan_count(eth, 2)
    tracker.apply_plan_count(sol, 3)

    tracker.clear_many({btc, sol})

    assert tracker.count(btc) == 0
    assert tracker.count(eth) == 2
    assert tracker.count(sol) == 0
