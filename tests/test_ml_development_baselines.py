from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services import ml_development_baselines as baselines
from services.ml_signal_service import FEATURE_KEYS, decision_group_partition


def _frame() -> pd.DataFrame:
    rows = []
    for index in range(240):
        decision_at = pd.Timestamp("2026-08-01", tz="UTC") + pd.Timedelta(
            minutes=index * 10
        )
        for horizon in (5, 240):
            rows.append({
                **{key: float(index % 7) for key in FEATURE_KEYS},
                "id": len(rows) + 1,
                "decision_group": f"group-{index:04}",
                "decision_timestamp": decision_at,
                "label_timestamp": decision_at + pd.Timedelta(minutes=horizon),
                "horizon_minutes": horizon,
                "long_return_pct": 0.02 + index % 3 * 0.01,
                "short_return_pct": -0.02 - index % 3 * 0.01,
                "long_execution_cost_pct": 0.1,
                "short_execution_cost_pct": 0.1,
                "sample_weight": 1.0,
            })
    return pd.DataFrame(rows)


class _MeanModel:
    def fit(self, features, target, **_kwargs):
        self.mean = float(target.mean())
        return self

    def predict(self, features):
        return np.full(len(features), self.mean)


@pytest.fixture
def cheap_models(monkeypatch):
    monkeypatch.setattr(
        baselines, "_baseline_model", lambda _name, _target: _MeanModel()
    )


def test_exam_feature_and_label_changes_cannot_affect_baselines(cheap_models):
    frame = _frame()
    original = baselines.build_development_baseline_report(frame)
    exam = decision_group_partition(frame).holdout
    mask = frame["id"].isin(exam["id"])
    frame.loc[mask, FEATURE_KEYS] = 1e9
    frame.loc[mask, ["long_return_pct", "short_return_pct", "sample_weight"]] = 1e9
    changed = baselines.build_development_baseline_report(frame)
    assert changed == original
    assert original["status"] == "complete"
    assert original["exam_labels_evaluated"] is False
    assert original["artifact_persisted"] is False
    assert original["promotion_authorized"] is False
    assert original["submits_orders"] is False


@pytest.mark.parametrize("column", ["sample_weight", "horizon_minutes"])
def test_fingerprint_covers_weighting_and_horizon_inputs(cheap_models, column):
    frame = _frame()
    original = baselines.build_development_baseline_report(frame)
    development = decision_group_partition(frame).train
    row = development[development["horizon_minutes"] == 240].iloc[0]
    frame.loc[frame["id"] == row["id"], column] += 1
    changed = baselines.build_development_baseline_report(frame)
    assert changed["development_input_sha256"] != original["development_input_sha256"]


def test_folds_purge_long_horizon_labels_and_use_primary_rows_only(cheap_models):
    frame = _frame()
    report = baselines.build_development_baseline_report(frame)
    assert len(report["folds"]) == 3
    for fold in report["folds"]:
        assert fold["decision_group_overlap_count"] == 0
        assert pd.Timestamp(fold["training_label_end"]) < pd.Timestamp(
            fold["validation_decision_start"]
        )
    assert report["evaluated_opportunity_count"] == sum(
        fold["validation_sample_count"] for fold in report["folds"]
    )
    assert report["evaluated_opportunity_count"] == (
        report["evaluated_decision_group_count"]
    )


def test_negative_net_predictions_remain_hold_and_costs_are_counted_once(cheap_models):
    report = baselines.build_development_baseline_report(_frame())
    assert report["baselines"]["no_trade"]["selected_count"] == 0
    for name in ("ridge_net", "existing_tree_net"):
        model = report["baselines"][name]
        assert model["selected_count"] == 0
        assert model["prediction_diagnostics"]["long"]["mean_predicted_net_return_pct"] < 0
    long = report["baselines"]["always_long"]
    assert long["selected_count"] == report["evaluated_opportunity_count"]
    assert -0.08 <= long["avg_return_pct"] <= -0.06
    assert long["mean_selected_gross_return_pct"] > 0
    assert long["mean_selected_execution_cost_pct"] == pytest.approx(0.1)
    assert long["avg_return_pct"] == pytest.approx(
        long["mean_selected_gross_return_pct"]
        - long["mean_selected_execution_cost_pct"]
    )
    assert report["baselines"]["ridge_net"]["rank_only_top_20_percent"][
        "selected_count"
    ] > 0


def test_ridge_pipeline_imputes_and_standardizes():
    frame = _frame().head(20)
    model = baselines._baseline_model("ridge_net", frame["long_return_pct"])
    model.fit(frame[FEATURE_KEYS], frame["long_return_pct"])
    assert np.isfinite(model.predict(frame[FEATURE_KEYS])).all()
