from __future__ import annotations

import numpy as np
import pandas as pd

from services import ml_nested_development_tree as nested
from services.ml_signal_service import FEATURE_KEYS, decision_group_partition


def _frame() -> pd.DataFrame:
    rows = []
    for index in range(480):
        decision = pd.Timestamp("2026-08-01", tz="UTC") + pd.Timedelta(
            minutes=index * 30
        )
        for horizon in (5, 240):
            rows.append(
                {
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
                }
            )
    return pd.DataFrame(rows)


def _predictions(_train, validation):
    return {
        "long": np.full(len(validation), 0.4),
        "short": np.full(len(validation), -0.6),
    }


def test_tree_family_is_diagnostic_and_exam_label_independent(monkeypatch):
    monkeypatch.setattr(nested, "_predict", _predictions)
    frame = _frame()
    report = nested.build_nested_tree_development_report(frame)
    exam = decision_group_partition(
        frame[frame["horizon_minutes"] == nested.PRIMARY_PREDICTION_HORIZON_MINUTES]
    ).holdout
    model_feature_keys = [
        key for key in FEATURE_KEYS if key != "horizon_minutes"
    ]
    frame.loc[frame["id"].isin(exam["id"]), model_feature_keys] = 1e9
    frame.loc[
        frame["id"].isin(exam["id"]), ["long_return_pct", "short_return_pct"]
    ] = -1e9
    assert nested.build_nested_tree_development_report(frame) == report
    assert report["status"] == "complete"
    assert report["version"] == nested.NESTED_TREE_POLICY_VERSION
    assert report["search_space"]["model"] == "random_forest_regressor_net"
    assert report["exam_labels_evaluated"] is False
    assert report["promotion_authorized"] is False
    assert report["artifact_persisted"] is False
    assert report["submits_orders"] is False


def test_tree_family_rejects_no_trade_inner_candidates(monkeypatch):
    monkeypatch.setattr(
        nested,
        "_predict",
        lambda _train, validation: {
            side: np.full(len(validation), -1.0)
            for side in ("long", "short")
        },
    )
    report = nested.build_nested_tree_development_report(_frame())
    assert report["status"] == "complete"
    assert report["selected_policy_fold_count"] == 0
    assert report["aggregate_outer_evidence"]["selected_count"] == 0
    assert all(
        fold["selection"]["reason"] == "no_inner_policy_qualified"
        for fold in report["folds"]
    )
