from __future__ import annotations

import numpy as np
import pandas as pd

from services import ml_nested_development as nested
from services.ml_signal_service import FEATURE_KEYS, decision_group_partition


def _frame() -> pd.DataFrame:
    rows = []
    for index in range(480):
        decision = pd.Timestamp("2026-08-01", tz="UTC") + pd.Timedelta(
            minutes=index * 30
        )
        for horizon in (5, 240):
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


def _predictions(_train, validation, _alpha):
    return {
        "long": np.full(len(validation), 0.4),
        "short": np.full(len(validation), -0.6),
    }


def test_nested_exam_changes_cannot_affect_report(monkeypatch):
    monkeypatch.setattr(nested, "_predict", _predictions)
    frame = _frame()
    report = nested.build_nested_development_report(frame)
    exam = decision_group_partition(frame).holdout
    frame.loc[frame["id"].isin(exam["id"]), FEATURE_KEYS] = 1e9
    frame.loc[
        frame["id"].isin(exam["id"]), ["long_return_pct", "short_return_pct"]
    ] = -1e9
    assert nested.build_nested_development_report(frame) == report
    assert report["status"] == "complete"
    assert report["exam_labels_evaluated"] is False
    assert report["promotion_authorized"] is False
    assert report["artifact_persisted"] is False
    assert report["submits_orders"] is False


def test_outer_labels_cannot_choose_their_own_policy(monkeypatch):
    monkeypatch.setattr(nested, "_predict", _predictions)
    frame = _frame()
    report = nested.build_nested_development_report(frame)
    development = decision_group_partition(frame).train
    _, validation, _ = nested._folds(development, 3)[0]
    frame.loc[frame["id"].isin(validation["id"]), "long_return_pct"] = -100.0
    changed = nested.build_nested_development_report(frame)
    assert report["folds"][0]["selection"]["policy"] is not None
    assert changed["folds"][0]["selection"] == report["folds"][0]["selection"]
    assert changed["folds"][0]["outer_evidence"]["avg_return_pct"] < 0
    for fold in report["folds"]:
        for boundary in [fold, *fold["selection"]["inner_folds"]]:
            assert pd.Timestamp(boundary["training_label_end"]) < pd.Timestamp(
                boundary["validation_decision_start"]
            )
            assert boundary["decision_group_overlap_count"] == 0


def test_no_qualified_inner_candidate_selects_no_trade(monkeypatch):
    monkeypatch.setattr(
        nested, "_predict",
        lambda _train, validation, _alpha: {
            side: np.full(len(validation), -1.0) for side in ("long", "short")
        },
    )
    report = nested.build_nested_development_report(_frame())
    assert report["status"] == "complete"
    assert report["selected_policy_fold_count"] == 0
    assert report["aggregate_outer_evidence"]["selected_count"] == 0
    assert report["aggregate_outer_evidence"]["average_return_per_opportunity_pct"] == 0
    assert all(
        fold["selection"]["reason"] == "no_inner_policy_qualified"
        for fold in report["folds"]
    )
