from __future__ import annotations

import pandas as pd

from services.fixed_direction_development import build_fixed_direction_development_report
from services.ml_signal_service import FEATURE_KEYS


def _frame(*, positive: bool = True) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for index in range(144):
        decision = start + pd.Timedelta(minutes=index * 60)
        label = decision + pd.Timedelta(minutes=60)
        if positive:
            net = -0.01 if index % 10 == 0 else 0.2
        else:
            net = -0.2
        rows.append(
            {
                "id": index + 1,
                "decision_group": f"g-{index + 1}",
                "decision_timestamp": decision,
                "label_timestamp": label,
                "horizon_minutes": 60,
                "long_return_pct": -net,
                "short_return_pct": net,
                "long_execution_cost_pct": 0.0,
                "short_execution_cost_pct": 0.0,
                "sample_weight": 1.0,
                "change_24h_pct": 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    for column in FEATURE_KEYS:
        frame[column] = 60.0 if column == "horizon_minutes" else 0.0
    return frame


def test_fixed_direction_report_passes_only_when_all_purged_folds_are_positive() -> None:
    report = build_fixed_direction_development_report(
        _frame(),
        horizon_minutes=60,
        side="short",
    )

    assert report["status"] == "development_candidate_qualified"
    assert report["policy"]["selection_or_tuning"] is False
    assert report["exam_labels_evaluated"] is False
    assert len(report["folds"]) == 3
    assert all(fold["chronological_label_disjoint"] for fold in report["folds"])
    assert all(fold["nonoverlapping"]["selected_count"] >= 16 for fold in report["folds"])


def test_fixed_direction_report_rejects_negative_fee_after_returns() -> None:
    report = build_fixed_direction_development_report(
        _frame(positive=False),
        horizon_minutes=60,
        side="short",
    )

    assert report["status"] == "rejected"
    assert report["reason"] == "fee_after_stability_rules_failed"
    assert report["aggregate_outer_evidence"]["all_opportunities"]["profit_factor"] == 0.0
