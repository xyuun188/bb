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
    exam = decision_group_partition(
        frame[frame["horizon_minutes"] == nested.PRIMARY_PREDICTION_HORIZON_MINUTES]
    ).holdout
    # The horizon is a partition key, not a model feature.  Mutating it would
    # change the research input instead of testing exam-label isolation.
    model_feature_keys = [
        key for key in FEATURE_KEYS if key != "horizon_minutes"
    ]
    frame.loc[frame["id"].isin(exam["id"]), model_feature_keys] = 1e9
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
    primary = frame[
        frame["horizon_minutes"] == nested.PRIMARY_PREDICTION_HORIZON_MINUTES
    ]
    development = decision_group_partition(primary).train
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


def test_fixed_horizon_folds_do_not_purge_on_unfitted_long_horizon_labels() -> None:
    frame = _frame()
    groups, _ = nested._decision_group_availability(frame)
    all_horizon = nested._folds(frame, 3)
    primary = frame[
        frame["horizon_minutes"] == nested.PRIMARY_PREDICTION_HORIZON_MINUTES
    ].copy()
    fixed_horizon = nested._folds(primary, 3)

    assert len(groups) == 480
    assert len(all_horizon) == len(fixed_horizon) == 3
    assert sum(
        fold[0]["horizon_minutes"].eq(5).sum() for fold in all_horizon
    ) < sum(
        fold[0]["horizon_minutes"].eq(5).sum() for fold in fixed_horizon
    )
    for train, validation, boundary in fixed_horizon:
        assert train["horizon_minutes"].eq(5).sum() >= nested.MIN_TRAINING_SAMPLE_COUNT
        assert validation["horizon_minutes"].eq(5).sum() > 0
        assert boundary["chronological_label_disjoint"] is True


def test_nested_partition_uses_primary_horizon_for_outer_purge() -> None:
    frame = _frame()
    generic = decision_group_partition(frame)
    primary_frame = frame[
        frame["horizon_minutes"] == nested.PRIMARY_PREDICTION_HORIZON_MINUTES
    ]
    primary = decision_group_partition(primary_frame)

    assert primary.report["sample_count"] == 480
    assert primary.report["train_sample_count"] < generic.report["train_sample_count"]
    assert primary.report["holdout_decision_group_count"] > 0
    assert primary.report["chronological_label_disjoint"] is True
