from datetime import UTC, datetime, timedelta

import pandas as pd

from backtest.bias_analysis import (
    run_lookahead_analysis,
    run_recursive_warmup_analysis,
    validate_feature_availability,
)
from services.strategy_contract_adapter import context_from_historical_bar


def _frame(rows: int = 40) -> pd.DataFrame:
    close = [100.0 + index * 0.25 for index in range(rows)]
    return pd.DataFrame(
        {
            "open": close,
            "high": [value + 1 for value in close],
            "low": [value - 1 for value in close],
            "close": close,
            "volume": [1000.0 + index for index in range(rows)],
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
    )


def test_lookahead_analysis_accepts_causal_rolling_feature() -> None:
    def builder(frame: pd.DataFrame) -> pd.DataFrame:
        frame["rolling_mean"] = frame["close"].rolling(3).mean()
        return frame

    report = run_lookahead_analysis(
        _frame(),
        builder,
        feature_columns=["rolling_mean"],
        checkpoints=range(4, 40),
    )
    assert report["status"] == "pass"
    assert report["violation_count"] == 0


def test_lookahead_analysis_detects_future_shift() -> None:
    def builder(frame: pd.DataFrame) -> pd.DataFrame:
        frame["future_close"] = frame["close"].shift(-1)
        return frame

    report = run_lookahead_analysis(
        _frame(),
        builder,
        feature_columns=["future_close"],
        checkpoints=range(4, 39),
    )
    assert report["status"] == "fail"
    assert report["violation_count"] == 35
    assert {item["reason"] for item in report["violations"]} == {"future_data_dependent"}


def test_recursive_analysis_confirms_rolling_window_convergence() -> None:
    def builder(frame: pd.DataFrame) -> pd.DataFrame:
        frame["rolling_mean"] = frame["close"].rolling(5).mean()
        return frame

    report = run_recursive_warmup_analysis(
        _frame(),
        builder,
        feature_columns=["rolling_mean"],
        warmup_lengths=(5, 10),
        required_warmup_length=10,
        comparison_points=3,
    )
    assert report["status"] == "pass"
    assert report["warmups"][-1]["difference_count"] == 0


def test_feature_availability_rejects_future_or_naive_evidence() -> None:
    decision_time = datetime(2026, 1, 1, 1, tzinfo=UTC)
    report = validate_feature_availability(
        [
            {
                "feature_available_at": decision_time,
                "decision_time": decision_time,
            },
            {
                "feature_available_at": decision_time + timedelta(seconds=1),
                "decision_time": decision_time,
            },
            {
                "feature_available_at": datetime(2026, 1, 1, 1),
                "decision_time": decision_time,
            },
        ]
    )
    assert report["status"] == "fail"
    assert report["checked_records"] == 2
    assert report["violation_count"] == 2


def test_historical_context_marks_feature_available_at_bar_close() -> None:
    context = context_from_historical_bar(
        symbol="BTC/USDT",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        timeframe="1h",
        bar={"open": 99, "high": 101, "low": 98, "close": 100, "volume": 1000},
        feature_snapshot={"rsi_14": 50},
        strategy={"strategy_id": "s", "strategy_version": "1"},
        parameters={"parameter_set_id": "p", "parameter_version": "1", "values": {}},
    )
    assert context.decision_time == datetime(2026, 1, 1, 1, tzinfo=UTC)
    assert context.market_snapshot["bar_opened_at"] == "2026-01-01T00:00:00+00:00"
    assert context.market_snapshot["bar_closed_at"] == "2026-01-01T01:00:00+00:00"
