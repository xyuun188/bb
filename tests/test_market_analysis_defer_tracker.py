from __future__ import annotations

from datetime import UTC, datetime, timedelta

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


def test_market_analysis_defer_tracker_scheduling_preserves_existing_priority() -> None:
    tracker = _tracker()
    tracker.defer_many(["BTC/USDT", "ETH/USDT"], "round_budget")

    tracker.begin_many(["SOL/USDT", "BTC/USDT"])

    assert tracker.ordered_symbols() == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    assert tracker.candidates["BTC/USDT"].reason == "round_budget"
    assert tracker.candidates["SOL/USDT"].reason == "scheduled"


def test_market_analysis_defer_tracker_completes_and_prunes_ineligible_symbols() -> None:
    tracker = _tracker()
    tracker.defer_many(["BTC/USDT", "ETH/USDT", "SOL/USDT"], "shortlist_capacity")

    tracker.complete("ETH/USDT")
    tracker.retain(["SOL/USDT", "XRP/USDT"])

    assert tracker.ordered_symbols() == ["SOL/USDT"]


def test_market_analysis_defer_tracker_reports_current_reasons() -> None:
    tracker = _tracker(max_candidates=2)
    started_at = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
    tracker.defer("BTC/USDT", "model_timeout", now=started_at)
    tracker.defer("ETH/USDT", "round_budget", now=started_at)
    tracker.defer("SOL/USDT", "round_budget", now=started_at + timedelta(minutes=10))

    snapshot = tracker.snapshot(
        now=started_at + timedelta(minutes=35),
        coverage_target_seconds=30 * 60,
    )

    assert tracker.ordered_symbols() == ["ETH/USDT", "SOL/USDT"]
    assert snapshot["deferred_count"] == 2
    assert snapshot["reason_counts"] == [{"reason": "round_budget", "count": 2}]
    assert snapshot["oldest_wait_seconds"] == 35 * 60
    assert snapshot["overdue_count"] == 1
    assert snapshot["overdue_symbols"] == ["ETH/USDT"]
    assert snapshot["pending_coverage_window_met"] is False
    assert snapshot["is_entry_gate"] is False
