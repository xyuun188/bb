from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services import liquidity_regime_development as research
from services.ml_development_snapshot import make_development_snapshot, snapshot_frame
from services.ml_signal_service import FEATURE_KEYS

CUTOFF = "2026-12-01T00:00:00Z"


def _frame(count=240):
    rows = []
    for index in range(count):
        decision = pd.Timestamp("2026-10-01", tz="UTC") + pd.Timedelta(hours=index * 4)
        for horizon in research.HORIZONS:
            rows.append({
                **{key: 0.0 for key in FEATURE_KEYS},
                "id": len(rows) + 1,
                "symbol": "BTC/USDT",
                "decision_group": f"group-{index:04}",
                "decision_timestamp": decision,
                "label_timestamp": decision + pd.Timedelta(minutes=horizon),
                "horizon_minutes": horizon,
                "long_return_pct": -0.1 if index % 10 == 0 else 0.5,
                "short_return_pct": -0.5,
                "long_execution_cost_pct": 0.1,
                "short_execution_cost_pct": 0.1,
                "sample_weight": 1.0,
                "spread_pct": 0.01,
                "log_notional_24h_usdt": 9.0,
                "entry_activity_volume_ratio": 1.5,
                "volatility_20": 0.01,
                "rsi_14": 35.0,
                "returns_20": 0.02,
                "price_vs_sma50": 0.01,
                "price_vs_sma20": -0.01,
                "regime_trending": 1.0,
            })
    return pd.DataFrame(rows)


def _report(frame):
    snapshot = make_development_snapshot(frame, cutoff=CUTOFF)
    return research.build_liquidity_regime_development_report(
        snapshot_frame(snapshot, cutoff=CUTOFF), snapshot=snapshot,
    )


def test_signal_is_bidirectional_and_cannot_use_outcomes_or_realized_costs():
    frame = _frame().query("horizon_minutes == 60").head(4).copy()
    frame.loc[frame.index[1], ["returns_20", "price_vs_sma50"]] = -0.01
    frame.loc[frame.index[2], "regime_trending"] = 0
    frame.loc[frame.index[3], "spread_pct"] = 0
    candidate = research.CANDIDATES[0]
    sides = research.candidate_sides(frame, candidate)
    assert sides.tolist() == ["long", "short", "hold", "hold"]
    frame[["long_return_pct", "short_return_pct"]] = -1e9
    frame[["long_execution_cost_pct", "short_execution_cost_pct"]] = 1e9
    np.testing.assert_array_equal(research.candidate_sides(frame, candidate), sides)


def test_liquidity_filters_reject_missing_expensive_and_inactive_observations():
    frame = _frame().query("horizon_minutes == 60").head(5).copy()
    frame.loc[frame.index[0], "spread_pct"] = np.nan
    frame.loc[frame.index[1], "spread_pct"] = 0.03
    frame.loc[frame.index[2], "log_notional_24h_usdt"] = 0
    frame.loc[frame.index[3], "entry_activity_volume_ratio"] = 0.5
    frame.loc[frame.index[4], "volatility_20"] = 0
    loose, tight = research.CANDIDATES[:2]
    assert research.candidate_sides(frame, loose).tolist() == [
        "hold", "long", "hold", "hold", "hold",
    ]
    assert research.candidate_sides(frame, tight).tolist() == ["hold"] * 5


def test_reversion_uses_only_registered_ranging_regime():
    frame = _frame().query("horizon_minutes == 60").head(3).copy()
    frame["regime_ranging"] = [1, 1, 0]
    frame.loc[frame.index[1], ["rsi_14", "price_vs_sma20"]] = [65, 0.01]
    assert research.candidate_sides(frame, research.CANDIDATES[2]).tolist() == [
        "long", "short", "hold",
    ]


def test_capacity_keeps_symbol_and_global_limits_across_fold_boundaries():
    frame = _frame().query("horizon_minutes == 240").head(7).copy()
    start = frame["decision_timestamp"].iloc[0]
    frame["decision_timestamp"] = [
        start + pd.Timedelta(minutes=value) for value in (0, 1, 2, 3, 4, 240, 241)
    ]
    frame["symbol"] = [
        "BTC-USDT-SWAP", "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
        "BTC/USDT", "XRP/USDT",
    ]
    sides = np.full(len(frame), "long")
    selected = research.capacity_sides(frame, sides)
    assert selected.tolist() == ["long", "hold", "long", "long", "hold", "long", "hold"]
    frame["label_timestamp"] = start + pd.Timedelta(days=50)
    frame["long_return_pct"] = -999
    np.testing.assert_array_equal(research.capacity_sides(frame, sides), selected)
    frame.loc[frame.index[0], "symbol"] = ""
    assert research.capacity_sides(frame, sides)[0] == "hold"


def test_positive_development_is_not_execution_or_exam_authorization():
    report = _report(_frame())
    assert report["status"] == "development_candidate_qualified"
    assert len(report["candidates"]) == 8
    assert len(report["qualified_candidate_ids"]) == 4
    assert len(report["folds"]) == 3
    for fold in report["folds"]:
        assert pd.Timestamp(fold["training_label_end"]) < pd.Timestamp(
            fold["validation_decision_start"]
        )
        assert fold["decision_group_overlap_count"] == 0
    for horizon in report["fixed_baselines_by_horizon"].values():
        assert horizon["hold"]["capacity"]["selected_count"] == 0
        assert horizon["hold"]["capacity"]["average_return_per_opportunity_pct"] == 0
    for key in (
        "promotion_authorized", "exam_labels_evaluated", "orders_submitted",
        "runtime_policy_changed", "artifact_persisted",
    ):
        assert report[key] is False


def test_a_losing_fold_cannot_hide_in_a_profitable_aggregate():
    frame = _frame()
    frame.loc[frame["decision_group"] >= "group-0200", "long_return_pct"] = 0.05
    report = _report(frame)
    candidate = report["candidates"][0]
    assert candidate["aggregate_evidence"]["capacity"]["avg_return_pct"] > 0
    assert "fold_2:capacity_nonpositive_net_mean" in candidate["rejection_reasons"]
    assert report["status"] == "rejected"
    assert report["qualified_candidate_ids"] == []


def test_short_coverage_and_incomplete_horizons_are_explicit():
    frame = _frame(2)
    report = _report(frame)
    assert report["status"] == "insufficient_development_data"
    assert report["candidates"] == []
    report = _report(_frame().query("horizon_minutes == 60"))
    assert report["reason"] == "insufficient_common_horizon_groups"
    assert report["excluded_incomplete_group_count"] == 240


def test_frozen_frame_integrity_is_required():
    frame = _frame()
    snapshot = make_development_snapshot(frame, cutoff=CUTOFF)
    frozen = snapshot_frame(snapshot, cutoff=CUTOFF)
    frozen.loc[0, "rsi_14"] = 99
    with pytest.raises(ValueError, match="exact frozen"):
        research.build_liquidity_regime_development_report(frozen, snapshot=snapshot)
