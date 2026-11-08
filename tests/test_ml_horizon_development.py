from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services import ml_horizon_development as research
from services.ml_development_snapshot import (
    load_development_snapshot,
    make_development_snapshot,
    save_development_snapshot,
    snapshot_frame,
)
from services.ml_signal_service import FEATURE_KEYS

CUTOFF = "2026-08-12T00:00:00+00:00"


def _frame(*, step_minutes: int = 30) -> pd.DataFrame:
    rows = []
    for index in range(480):
        decision = pd.Timestamp("2026-08-01", tz="UTC") + pd.Timedelta(
            minutes=index * step_minutes
        )
        for horizon in research.HORIZONS:
            rows.append({
                **{key: float(index % 7) for key in FEATURE_KEYS},
                "id": len(rows) + 1,
                "decision_group": f"group-{index:04}",
                "decision_timestamp": decision,
                "label_timestamp": decision + pd.Timedelta(minutes=horizon),
                "horizon_minutes": horizon,
                "long_return_pct": -0.1 if index % 10 == 0 else 0.5,
                "short_return_pct": -0.5,
                "long_execution_cost_pct": 0.1,
                "short_execution_cost_pct": 0.1,
                "sample_weight": 1.0,
            })
    return pd.DataFrame(rows)


def _predictions(_train, validation, *, alpha, feature_policy):
    return {
        "long": np.full(len(validation), 0.4),
        "short": np.full(len(validation), -0.6),
    }


def _snapshot(frame: pd.DataFrame):
    snapshot = make_development_snapshot(frame, cutoff=CUTOFF)
    return snapshot_frame(snapshot, cutoff=CUTOFF), snapshot


def test_snapshot_reuse_is_immutable_and_cutoff_bound(tmp_path: Path):
    frame = _frame()
    snapshot = make_development_snapshot(frame, cutoff=CUTOFF)
    path = tmp_path / "development.json"
    original_frame, original = save_development_snapshot(snapshot, path=path)
    original_bytes = path.read_bytes()
    frame["long_return_pct"] += 10
    replacement = make_development_snapshot(frame, cutoff=CUTOFF)
    reused_frame, reused = save_development_snapshot(replacement, path=path)
    pd.testing.assert_frame_equal(reused_frame, original_frame)
    assert reused == original
    assert path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [path]
    with pytest.raises(ValueError, match="cutoff cannot change"):
        load_development_snapshot(path, cutoff="2026-08-13T00:00:00+00:00")


def test_snapshot_rejects_tampering_and_purges_entire_boundary_groups(tmp_path: Path):
    frame = _frame()
    cutoff = frame.iloc[200]["decision_timestamp"].isoformat()
    snapshot = make_development_snapshot(frame, cutoff=cutoff)
    development = snapshot_frame(snapshot, cutoff=cutoff)
    assert development["label_timestamp"].max() < pd.Timestamp(cutoff)
    source = frame[frame["decision_group"].isin(development["decision_group"])]
    assert len(source) == len(development)
    changed = copy.deepcopy(snapshot)
    changed["records"][0]["long_return_pct"] = 999
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity failure"):
        load_development_snapshot(path, cutoff=cutoff)


def test_snapshot_preserves_signed_funding_costs_and_zero_weights():
    frame = _frame()
    frame.loc[0, "sample_weight"] = 0.0
    frame.loc[0, "short_execution_cost_pct"] = -0.25
    development, _ = _snapshot(frame)
    row = development[development["id"] == 1].iloc[0]
    assert row["sample_weight"] == 0.0
    assert row["short_execution_cost_pct"] == -0.25
    evidence = research._views(development.head(1), np.array(["short"]))
    assert evidence["all_opportunities"]["avg_return_pct"] == pytest.approx(-0.25)
    frame.loc[0, "sample_weight"] = -0.1
    with pytest.raises(ValueError, match="sample weights"):
        _snapshot(frame)


def test_snapshot_preserves_canonical_symbols_and_uses_them_for_parallel_capacity():
    frame = _frame().query("horizon_minutes == 240").iloc[[0, 1, 8, 9]].copy()
    frame["symbol"] = [
        "BTC-USDT-SWAP", "ETH/USDT:USDT", "BTC/USDT", "ETH-USDT-SWAP"
    ]

    development, snapshot = _snapshot(frame)
    assert development["symbol"].tolist() == [
        "BTC/USDT", "ETH/USDT", "BTC/USDT", "ETH/USDT"
    ]
    assert snapshot["version"] == "2026-11-06.symbol-bound-frozen-development.v2"
    views = research._views(development, np.array(["long", "short", "long", "short"]))
    assert views["nonoverlapping"]["selected_count"] == 4


def test_missing_symbol_forces_conservative_global_nonoverlap_fallback():
    frame = _frame().query("horizon_minutes == 240").iloc[[0, 1, 8, 9]].copy()
    frame["symbol"] = ["BTC-USDT-SWAP", None, "ETH/USDT", "ETH-USDT-SWAP"]
    sides = np.array(["long", "short", "long", "short"])

    assert research._symbol_serial_sides(frame, sides).tolist() == (
        research._serial_sides(frame, sides).tolist()
    )


def test_snapshot_accepts_mixed_iso_timestamp_precision(tmp_path: Path):
    frame = _frame()
    frame.loc[frame.index[::2], "decision_timestamp"] = frame.loc[
        frame.index[::2], "decision_timestamp"
    ].map(lambda value: value + pd.Timedelta(microseconds=387433))
    snapshot = make_development_snapshot(frame, cutoff=CUTOFF)
    development, _ = save_development_snapshot(
        snapshot, path=tmp_path / "mixed-times.json"
    )
    assert development["decision_timestamp"].dt.tz is not None
    assert len(development) > 0


def test_rows_after_cutoff_cannot_affect_research_inputs():
    frame = _frame()
    cutoff = frame.iloc[1000]["decision_timestamp"].isoformat()
    original = make_development_snapshot(frame, cutoff=cutoff)
    excluded = frame["decision_timestamp"] >= pd.Timestamp(cutoff)
    frame.loc[excluded, FEATURE_KEYS] = 1e9
    frame.loc[excluded, ["long_return_pct", "short_return_pct"]] = -1e9
    changed = make_development_snapshot(frame, cutoff=cutoff)
    assert changed["records"] == original["records"]
    assert changed["development_input_sha256"] == original["development_input_sha256"]


def test_common_coverage_and_horizon_specific_purging(monkeypatch):
    monkeypatch.setattr(research, "_predict", _predictions)
    frame = _frame()
    frame = frame[~(
        (frame["decision_group"] == "group-0350") & (frame["horizon_minutes"] == 240)
    )]
    development, snapshot = _snapshot(frame)
    report = research.build_horizon_development_report(development, snapshot=snapshot)
    assert report["status"] == "complete"
    assert report["incomplete_horizon_group_count"] == 1
    assert report["common_horizon_decision_group_count"] == 479
    for fold in report["folds"]:
        for boundary in (fold, *fold["selection"]["inner_folds"]):
            assert pd.Timestamp(boundary["training_label_end"]) < pd.Timestamp(
                boundary["validation_decision_start"]
            )
            assert boundary["chronological_label_disjoint"] is True
        candidates = fold["selection"]["candidates"]
        assert len(candidates) == 72
        for horizon in research.HORIZONS:
            assert any(row["horizon_minutes"] == horizon for row in candidates)
            coverage = fold["selection"]["horizon_coverage"][str(horizon)]
            assert coverage["ready_fold_count"] == 2
            assert coverage["exclusion_reasons"] == []
            assert all(
                boundary["horizon_minutes"] == horizon
                for boundary in coverage["folds"]
            )
    assert report["exam_labels_evaluated"] is False
    assert report["artifact_persisted"] is False
    assert report["submits_orders"] is False
    assert report["promotion_authorized"] is False


def test_long_labels_do_not_erase_short_horizon_inner_folds(monkeypatch):
    monkeypatch.setattr(research, "_predict", _predictions)
    selection = research._select_policy(_frame(step_minutes=1))

    assert selection["policy"]["horizon_minutes"] == 5
    assert len(selection["candidates"]) == 54
    for horizon in (5, 15, 60):
        assert selection["horizon_coverage"][str(horizon)]["ready_fold_count"] == 2
    excluded = selection["horizon_coverage"]["240"]
    assert excluded["ready_fold_count"] < 2
    assert "insufficient_positive_weight_inner_training" in excluded["exclusion_reasons"]
    assert all(row["horizon_minutes"] != 240 for row in selection["candidates"])


def test_outer_training_keeps_only_settled_rows_without_losing_short_labels(monkeypatch):
    checked = []
    monkeypatch.setattr(research, "ALPHAS", (1.0,))
    monkeypatch.setattr(research, "NET_THRESHOLDS_PCT", (0.0,))

    def check_training(train, validation, **kwargs):
        assert train["horizon_minutes"].nunique() == 1
        assert train["label_timestamp"].max() < validation["decision_timestamp"].min()
        assert set(train["decision_group"]).isdisjoint(validation["decision_group"])
        checked.append(len(train))
        return _predictions(train, validation, **kwargs)

    monkeypatch.setattr(research, "_predict", check_training)
    development, snapshot = _snapshot(_frame(step_minutes=1))
    folds = research._folds(development, 3)
    train, validation, boundary = folds[0]
    start = validation["decision_timestamp"].min()
    expected = development[
        (development["decision_timestamp"] < start)
        & (development["label_timestamp"] < start)
    ]
    pd.testing.assert_frame_equal(train, expected)
    assert set(train["horizon_minutes"]) == {5, 15, 60}
    assert train["decision_timestamp"].max() > start - pd.Timedelta(minutes=240)
    assert boundary["label_purged_training_sample_count"] > 0

    report = research.build_horizon_development_report(development, snapshot=snapshot)
    assert checked
    assert report["status"] == "complete"
    assert len(report["folds"]) == 3
    assert any(
        candidate["horizon_minutes"] == 5
        for candidate in report["folds"][0]["selection"]["candidates"]
    )

    unavailable = (
        (development["decision_timestamp"] < start)
        & (development["label_timestamp"] >= start)
    )
    development.loc[unavailable, ["long_return_pct", "short_return_pct"]] = -1000.0
    market_features = [key for key in FEATURE_KEYS if key != "horizon_minutes"]
    development.loc[unavailable, market_features] = 1e9
    changed_frame, changed_snapshot = _snapshot(development)
    changed = research.build_horizon_development_report(
        changed_frame, snapshot=changed_snapshot
    )
    assert changed["folds"][0]["selection"] == report["folds"][0]["selection"]


def test_zero_weight_horizon_does_not_disable_other_horizons(monkeypatch):
    monkeypatch.setattr(research, "_predict", _predictions)
    frame = _frame()
    frame.loc[frame["horizon_minutes"] == 5, "sample_weight"] = 0.0

    selection = research._select_policy(frame)

    assert selection["policy"]["horizon_minutes"] == 15
    assert len(selection["candidates"]) == 54
    assert selection["horizon_coverage"]["5"]["exclusion_reasons"] == [
        "insufficient_positive_weight_inner_training"
    ]
    assert all(row["horizon_minutes"] != 5 for row in selection["candidates"])


def test_insufficient_horizons_report_coverage_and_choose_no_trade(monkeypatch):
    def forbidden_prediction(*_args, **_kwargs):
        raise AssertionError("excluded horizons must not be fitted")

    monkeypatch.setattr(research, "_predict", forbidden_prediction)
    selection = research._select_policy(_frame().iloc[:24])

    assert selection["policy"] is None
    assert selection["candidates"] == []
    assert all(
        coverage["exclusion_reasons"]
        for coverage in selection["horizon_coverage"].values()
    )


def test_negative_lcb_still_rejects_positive_mean_candidates(monkeypatch):
    monkeypatch.setattr(research, "_predict", _predictions)
    monkeypatch.setattr(research, "ALPHAS", (1.0,))
    monkeypatch.setattr(research, "NET_THRESHOLDS_PCT", (0.0,))
    frame = _frame(step_minutes=1)
    frame["long_return_pct"] = 0.1 + np.repeat(
        np.where(np.arange(480) % 2 == 0, 0.5, -0.46), len(research.HORIZONS)
    )

    selection = research._select_policy(frame)

    candidate = selection["candidates"][0]
    assert candidate["evidence"]["all_opportunities"]["avg_return_pct"] > 0.0
    assert candidate["evidence"]["all_opportunities"]["return_lcb_pct"] < 0.0
    assert "all_opportunities_nonpositive_naive_lcb" in candidate["rejection_reasons"]
    assert selection["policy"] is None


def test_positive_aggregate_cannot_hide_a_losing_inner_fold(monkeypatch):
    monkeypatch.setattr(research, "_predict", _predictions)
    monkeypatch.setattr(research, "ALPHAS", (1.0,))
    monkeypatch.setattr(research, "NET_THRESHOLDS_PCT", (0.0,))
    frame = _frame()
    frame.loc[
        frame["decision_group"] >= "group-0360", "long_return_pct"
    ] = 0.05

    selection = research._select_policy(frame)

    candidate = selection["candidates"][0]
    assert candidate["evidence"]["nonoverlapping"]["return_lcb_pct"] > 0.0
    assert candidate["per_fold"][1]["nonoverlapping"]["avg_return_pct"] < 0.0
    assert "inner_fold_fee_after_instability" in candidate["rejection_reasons"]
    assert selection["policy"] is None


def test_outer_labels_cannot_choose_horizon_or_policy(monkeypatch):
    monkeypatch.setattr(research, "_predict", _predictions)
    development, snapshot = _snapshot(_frame())
    report = research.build_horizon_development_report(development, snapshot=snapshot)
    _, validation, _ = research._folds(development, 3)[0]
    development.loc[
        development["id"].isin(validation["id"]), "long_return_pct"
    ] = -100.0
    changed_frame, changed_snapshot = _snapshot(development)
    changed = research.build_horizon_development_report(
        changed_frame, snapshot=changed_snapshot
    )
    assert report["folds"][0]["selection"]["policy"] is not None
    assert changed["folds"][0]["selection"] == report["folds"][0]["selection"]
    assert changed["folds"][0]["outer_evidence"]["nonoverlapping"]["avg_return_pct"] < 0
    with pytest.raises(ValueError, match="exact frozen"):
        research.build_horizon_development_report(development, snapshot=snapshot)


def test_nonoverlap_selection_uses_time_and_prediction_without_labels():
    frame = _frame().query("horizon_minutes == 240").head(30)
    sides = np.full(len(frame), "long")
    selected = research._serial_sides(frame, sides)
    assert np.flatnonzero(selected == "long").tolist() == [0, 8, 16, 24]
    frame["long_return_pct"] = np.arange(len(frame)) * -100
    np.testing.assert_array_equal(research._serial_sides(frame, sides), selected)
    evidence = research._views(frame, sides)
    assert evidence["all_opportunities"]["selected_count"] == 30
    assert evidence["nonoverlapping"]["selected_count"] == 4
    assert evidence["nonoverlapping"]["avg_return_pct"] == pytest.approx(
        evidence["nonoverlapping"]["mean_selected_gross_return_pct"] - 0.1
    )


def test_nonoverlap_carries_position_across_fold_boundary():
    frame = _frame().query("horizon_minutes == 240").iloc[[0, 1, 9]]
    sides = np.array(["long", "short", "long"])
    assert research._serial_sides(frame, sides).tolist() == ["long", "hold", "long"]


def test_symbol_nonoverlap_allows_independent_symbols_in_parallel():
    frame = _frame().query("horizon_minutes == 240").iloc[[0, 1, 8, 9]].copy()
    frame["symbol"] = ["BTC/USDT", "ETH/USDT", "BTC/USDT", "ETH/USDT"]
    sides = np.array(["long", "short", "long", "short"])

    assert research._symbol_serial_sides(frame, sides).tolist() == [
        "long", "short", "long", "short"
    ]


def test_unqualified_signals_still_choose_no_trade(monkeypatch):
    monkeypatch.setattr(
        research, "_predict",
        lambda _train, validation, **_kwargs: {
            side: np.full(len(validation), -1.0) for side in ("long", "short")
        },
    )
    development, snapshot = _snapshot(_frame())
    report = research.build_horizon_development_report(development, snapshot=snapshot)
    assert report["status"] == "complete"
    assert report["selected_policy_fold_count"] == 0
    assert report["aggregate_outer_evidence"]["nonoverlapping"]["selected_count"] == 0
    for baseline in report["fixed_baselines_by_horizon"].values():
        long = baseline["long"]["all_opportunities"]
        assert long["avg_return_pct"] == pytest.approx(
            long["mean_selected_gross_return_pct"] - 0.1
        )


def test_each_fold_reports_its_own_fixed_horizon_baselines():
    development, snapshot = _snapshot(_frame())
    report = research.build_horizon_development_report(development, snapshot=snapshot)

    assert report["folds"]
    for fold in report["folds"]:
        baselines = fold["fixed_baselines_by_horizon"]
        assert set(baselines) == {str(horizon) for horizon in research.HORIZONS}
        for horizon_baselines in baselines.values():
            assert set(horizon_baselines) == {"long", "short"}
            for views in horizon_baselines.values():
                assert set(views) == {"all_opportunities", "nonoverlapping"}
                assert views["nonoverlapping"]["selected_count"] > 0


def test_real_ridge_supports_both_feature_policies():
    development, _ = _snapshot(_frame())
    primary = development.query("horizon_minutes == 5")
    for policy in research.FEATURE_POLICIES:
        predictions = research._predict(
            primary.iloc[:50], primary.iloc[50:60], alpha=10.0,
            feature_policy=policy,
        )
        assert all(np.isfinite(values).all() for values in predictions.values())


def test_real_ridge_can_evaluate_short_horizons_when_long_folds_are_unavailable(monkeypatch):
    monkeypatch.setattr(research, "ALPHAS", (1.0,))
    monkeypatch.setattr(research, "NET_THRESHOLDS_PCT", (0.0,))

    selection = research._select_policy(_frame(step_minutes=1))

    assert selection["policy"]["horizon_minutes"] == 5
    assert len(selection["candidates"]) == 6
    assert selection["horizon_coverage"]["240"]["exclusion_reasons"]
    selected = next(
        candidate for candidate in selection["candidates"]
        if all(candidate[key] == value for key, value in selection["policy"].items())
    )
    assert selected["evidence"]["nonoverlapping"]["return_lcb_pct"] > 0.0
    assert selected["inner_selection_eligible"] is True


def test_zero_weight_rows_do_not_affect_model_or_preprocessing():
    development, _ = _snapshot(_frame())
    primary = development.query("horizon_minutes == 5").copy()
    original = research._predict(
        primary.iloc[1:50], primary.iloc[50:60], alpha=10.0, feature_policy="existing"
    )
    primary.loc[primary.index[0], FEATURE_KEYS] = 1e9
    primary.loc[primary.index[0], "sample_weight"] = 0.0
    changed = research._predict(
        primary.iloc[:50], primary.iloc[50:60], alpha=10.0, feature_policy="existing"
    )
    for side in ("long", "short"):
        np.testing.assert_array_equal(original[side], changed[side])
