from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backtest.reproducibility import (
    dataframe_from_snapshot,
    load_experiment_bundle,
    write_experiment_bundle,
)
from core.experiment_contracts import build_experiment_result
from scripts.run_reproducible_backtest import _build_spec, _run_engine


def _ohlcv() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=240, freq="h", tz="UTC")
    phase = np.linspace(0.0, 16.0 * math.pi, len(timestamps))
    close = 100.0 + np.sin(phase) * 8.0 + np.sin(phase / 3.0) * 2.0
    opening = close + np.cos(phase) * 0.3
    high = np.maximum(opening, close) + 0.8
    low = np.minimum(opening, close) - 0.8
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opening,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000 + np.sin(phase) * 100,
        }
    )


def test_backtest_bundle_replays_identical_metrics(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("scripts.run_reproducible_backtest._source_sha256", lambda: "b" * 64)
    spec, snapshot = _build_spec(
        _ohlcv(),
        source="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        parameters={"rsi_oversold": 35, "rsi_overbought": 65, "macd_threshold": 0},
        initial_cash=10_000,
        commission_rate=0.001,
        slippage_rate=0.0002,
        random_seed=11,
        git_commit="a" * 40,
    )
    metrics = _run_engine(dataframe_from_snapshot(snapshot), spec)
    result = build_experiment_result(spec, status="complete", metrics=metrics)

    path = write_experiment_bundle(
        tmp_path,
        spec=spec,
        result=result,
        dataset_snapshot=snapshot,
    )
    loaded_spec, loaded_result, loaded_snapshot = load_experiment_bundle(path)
    replay_metrics = _run_engine(dataframe_from_snapshot(loaded_snapshot), loaded_spec)

    assert replay_metrics == loaded_result["metrics"]
    assert loaded_spec["dataset"]["row_count"] == 240
    assert loaded_spec["execution_assumptions"]["commission_rate"] == 0.001
    assert loaded_spec["execution_assumptions"]["slippage_rate"] == 0.0002
    assert loaded_spec["dataset"]["source"] == "fixture"
