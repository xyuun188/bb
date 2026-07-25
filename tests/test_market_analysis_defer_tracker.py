from __future__ import annotations

from services.market_analysis_defer_tracker import MarketAnalysisDeferTracker


def _tracker(*, max_candidates: int = 512) -> MarketAnalysisDeferTracker:
    return MarketAnalysisDeferTracker(
        normalize_symbol=lambda value: str(value or "").strip().upper(),
        max_candidates=max_candidates,
    )


def test_market_analysis_defer_tracker_requeues_failure_at_tail() -> None:
    tracker = _tracker()
    tracker.defer_many(["BTC/USDT", "ETH/USDT", "SOL/USDT"], "round_budget")

    tracker.defer("BTC/USDT", "feature_unavailable")

    assert tracker.ordered_symbols() == ["ETH/USDT", "SOL/USDT", "BTC/USDT"]
    assert tracker.candidates["BTC/USDT"].defer_count == 2
    assert tracker.candidates["BTC/USDT"].reason == "feature_unavailable"


def test_market_analysis_defer_tracker_completes_and_prunes_ineligible_symbols() -> None:
    tracker = _tracker()
    tracker.defer_many(["BTC/USDT", "ETH/USDT", "SOL/USDT"], "shortlist_capacity")

    tracker.complete("ETH/USDT")
    tracker.retain(["SOL/USDT", "XRP/USDT"])

    assert tracker.ordered_symbols() == ["SOL/USDT"]


def test_market_analysis_defer_tracker_reports_current_reasons() -> None:
    tracker = _tracker(max_candidates=2)
    tracker.defer("BTC/USDT", "model_timeout")
    tracker.defer("ETH/USDT", "round_budget")
    tracker.defer("SOL/USDT", "round_budget")

    snapshot = tracker.snapshot()

    assert tracker.ordered_symbols() == ["ETH/USDT", "SOL/USDT"]
    assert snapshot["deferred_count"] == 2
    assert snapshot["reason_counts"] == [{"reason": "round_budget", "count": 2}]
    assert snapshot["is_entry_gate"] is False
