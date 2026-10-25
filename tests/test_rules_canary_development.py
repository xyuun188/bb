from __future__ import annotations

import pandas as pd
import pytest

from services.rules_canary_development import (
    RULES_CANARY_DEVELOPMENT_VERSION,
    build_rules_canary_development_report,
    filter_rules_canary_decision_window,
    rules_canary_side,
)


def test_rules_canary_side_requires_fixed_consensus() -> None:
    snapshot = {
        "current_price": 100.0,
        "atr_14": 1.0,
        "returns_1": 1.0,
        "returns_5": 1.0,
        "returns_20": 1.0,
        "macd_diff": 1.0,
        "price_vs_sma20": 1.0,
        "price_vs_sma50": -1.0,
    }
    assert rules_canary_side(snapshot) == "long"
    snapshot["returns_1"] = 0.0
    snapshot["returns_5"] = -1.0
    assert rules_canary_side(snapshot) == "hold"


def test_rules_canary_report_is_read_only_when_coverage_is_insufficient() -> None:
    frame = pd.DataFrame(
        [
            {
                "id": 1,
                "decision_group": "1",
                "decision_timestamp": "2026-01-01T00:00:00Z",
                "label_timestamp": "2026-01-01T00:05:00Z",
                "horizon_minutes": 5,
                "long_return_pct": 0.2,
                "short_return_pct": -0.2,
                "long_execution_cost_pct": 0.1,
                "short_execution_cost_pct": 0.1,
            }
        ]
    )
    report = build_rules_canary_development_report(
        frame,
        snapshots_by_id={
            1: {
                "current_price": 100.0,
                "atr_14": 1.0,
                "returns_1": 1.0,
                "returns_5": 1.0,
                "returns_20": 1.0,
                "macd_diff": 1.0,
                "price_vs_sma20": 1.0,
                "price_vs_sma50": 1.0,
            }
        },
    )
    assert report["version"] == RULES_CANARY_DEVELOPMENT_VERSION
    assert report["status"] == "insufficient_development_data"
    assert report["orders_submitted"] is False
    assert report["promotion_authorized"] is False


def test_rules_canary_window_uses_decision_time_not_label_completion_time() -> None:
    since = pd.Timestamp("2026-10-25T13:06:52Z").to_pydatetime()
    cutoff = pd.Timestamp("2026-10-25T19:06:52Z").to_pydatetime()
    frame = pd.DataFrame(
        [
            {
                "id": 1,
                "decision_timestamp": "2026-10-25T13:00:00Z",
                "label_timestamp": "2026-10-25T13:05:00Z",
            },
            {
                "id": 2,
                "decision_timestamp": "2026-10-25T13:06:52Z",
                "label_timestamp": "2026-10-25T17:06:52Z",
            },
            {
                "id": 3,
                "decision_timestamp": "2026-10-25T19:06:52Z",
                "label_timestamp": "2026-10-25T19:11:52Z",
            },
            {"id": 4, "decision_timestamp": None, "label_timestamp": None},
        ]
    )

    selected, report = filter_rules_canary_decision_window(
        frame,
        since=since,
        cutoff=cutoff,
    )

    assert selected["id"].tolist() == [2]
    assert report == {
        "decision_timestamp_source": "feature_snapshot.original_decision_timestamp",
        "candidate_row_count": 4,
        "valid_decision_timestamp_row_count": 3,
        "selected_row_count": 1,
        "missing_decision_timestamp_row_count": 1,
        "outside_decision_window_row_count": 2,
    }


def test_rules_canary_window_requires_ordered_aware_bounds() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        filter_rules_canary_decision_window(
            pd.DataFrame(),
            since=pd.Timestamp("2026-10-25T13:00:00Z").to_pydatetime(),
            cutoff=pd.Timestamp("2026-10-25T12:00:00Z").to_pydatetime(),
        )
