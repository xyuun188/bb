from datetime import UTC, datetime, timedelta

from services.position_profit_peaks import PositionProfitPeakTracker


def _tracker(tmp_path):
    return PositionProfitPeakTracker(
        path=tmp_path / "position_profit_peaks.json",
        symbol_normalizer=lambda symbol: str(symbol or "").upper(),
    )


def test_position_profit_peak_tracker_persists_and_keeps_best_peak(tmp_path):
    tracker = _tracker(tmp_path)

    first = tracker.update(
        model_name="ensemble_trader",
        symbol="btc/usdt",
        side="long",
        current_price=110.0,
        entry_price=100.0,
        unrealized_pnl=5.0,
        hold_minutes=12.0,
    )
    second = tracker.update(
        model_name="ensemble_trader",
        symbol="btc/usdt",
        side="long",
        current_price=104.0,
        entry_price=100.0,
        unrealized_pnl=2.0,
        hold_minutes=14.0,
    )

    assert first["peak_unrealized_pnl"] == 5.0
    assert second["peak_unrealized_pnl"] == 5.0
    assert second["last_unrealized_pnl"] == 2.0

    reloaded = _tracker(tmp_path)
    key = reloaded.key("ensemble_trader", "btc/usdt", "long")
    assert reloaded.peaks[key]["peak_unrealized_pnl"] == 5.0
    assert reloaded.peaks[key]["last_unrealized_pnl"] == 2.0


def test_position_profit_peak_tracker_handles_short_ratio(tmp_path):
    tracker = _tracker(tmp_path)

    state = tracker.update(
        model_name="ensemble_trader",
        symbol="eth/usdt",
        side="short",
        current_price=90.0,
        entry_price=100.0,
        unrealized_pnl=3.0,
        hold_minutes=None,
    )

    assert state["peak_pnl_ratio"] == 0.1
    assert state["hold_minutes"] == 0.0


def test_position_profit_peak_tracker_marks_recent_profit_exit(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.update(
        model_name="ensemble_trader",
        symbol="btc/usdt",
        side="long",
        current_price=110.0,
        entry_price=100.0,
        unrealized_pnl=5.0,
        hold_minutes=12.0,
    )

    tracker.remember_profit_exit("ensemble_trader", "btc/usdt", "long")
    state = tracker.peaks[tracker.key("ensemble_trader", "btc/usdt", "long")]

    assert state["profit_exit_count"] == 1
    assert 0.0 <= tracker.seconds_since_profit_exit(state) < 5.0


def test_position_profit_peak_tracker_prunes_and_removes_closed_positions(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.update(
        model_name="ensemble_trader",
        symbol="btc/usdt",
        side="long",
        current_price=110.0,
        entry_price=100.0,
        unrealized_pnl=5.0,
        hold_minutes=12.0,
    )
    tracker.update(
        model_name="ensemble_trader",
        symbol="eth/usdt",
        side="short",
        current_price=90.0,
        entry_price=100.0,
        unrealized_pnl=3.0,
        hold_minutes=8.0,
    )

    tracker.prune(
        [
            {
                "model_name": "ensemble_trader",
                "symbol": "btc/usdt",
                "side": "long",
                "is_open": True,
            }
        ]
    )

    assert tracker.key("ensemble_trader", "btc/usdt", "long") in tracker.peaks
    assert tracker.key("ensemble_trader", "eth/usdt", "short") not in tracker.peaks

    tracker.remove("ensemble_trader", "btc/usdt", "long")

    assert tracker.peaks == {}


def test_position_profit_peak_tracker_ignores_bad_exit_timestamp(tmp_path):
    tracker = _tracker(tmp_path)

    assert tracker.seconds_since_profit_exit({"last_profit_exit_at": "bad"}) == 0.0
    assert (
        tracker.seconds_since_profit_exit(
            {"last_profit_exit_at": (datetime.now(UTC) - timedelta(seconds=2)).isoformat()}
        )
        >= 0.0
    )
