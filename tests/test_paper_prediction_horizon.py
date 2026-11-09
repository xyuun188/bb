from __future__ import annotations

from services.paper_prediction_horizon import select_paper_horizon_cohort


def _row(source: str, horizon: float, net: float | None) -> dict:
    row = {
        "source": source,
        "horizon_minutes": horizon,
        "continuous_weight_multiplier": 1.0,
    }
    if net is not None:
        row["expected_net_return_pct"] = net
    return row


def test_tied_weight_prefers_positive_fee_after_horizon_over_shortest() -> None:
    rows = [
        _row("server_profit", 5.0, -0.20),
        _row("timeseries", 5.0, -0.30),
        _row("server_profit", 240.0, 0.35),
        _row("timeseries", 240.0, 0.25),
    ]

    selection = select_paper_horizon_cohort(rows)

    assert selection["selected_horizon_minutes"] == 240.0
    groups = {
        item["horizon_minutes"]: item
        for item in selection["available_horizon_groups"]
    }
    # Cohort net is the source-weighted mean, so both sources contribute one
    # vote at each horizon instead of the single best row deciding.
    assert groups[5.0]["best_fee_after_return_pct"] == -0.25
    assert groups[240.0]["best_fee_after_return_pct"] == 0.3


def test_higher_total_weight_wins_before_fee_after_evidence() -> None:
    rows = [
        _row("server_profit", 5.0, 0.01),
        _row("timeseries", 5.0, 0.01),
        _row("server_profit", 240.0, 5.0),
    ]
    weights = {"server_profit": 1.0, "timeseries": 1.0}

    selection = select_paper_horizon_cohort(rows, source_weights=weights)

    assert selection["selected_horizon_minutes"] == 5.0


def test_shortest_horizon_remains_the_last_tiebreak() -> None:
    rows = [
        _row("server_profit", 240.0, -0.10),
        _row("server_profit", 15.0, -0.10),
    ]

    selection = select_paper_horizon_cohort(rows)

    assert selection["selected_horizon_minutes"] == 15.0


def test_rows_without_net_evidence_still_select_a_horizon() -> None:
    rows = [_row("server_profit", 5.0, None), _row("server_profit", 60.0, None)]

    selection = select_paper_horizon_cohort(rows)

    assert selection["selected_horizon_minutes"] == 5.0
    assert selection["blockers"] == []


def test_zero_fee_after_evidence_is_not_treated_as_missing() -> None:
    rows = [_row("server_profit", 5.0, -0.01), _row("server_profit", 60.0, 0.0)]

    selection = select_paper_horizon_cohort(rows)

    assert selection["selected_horizon_minutes"] == 60.0
    groups = {
        item["horizon_minutes"]: item
        for item in selection["available_horizon_groups"]
    }
    assert groups[60.0]["best_fee_after_return_pct"] == 0.0


def test_authorized_horizon_still_overrides_evidence() -> None:
    rows = [_row("server_profit", 5.0, -1.0), _row("server_profit", 240.0, 1.0)]

    selection = select_paper_horizon_cohort(rows, preferred_horizon_minutes=5)

    assert selection["selected_horizon_minutes"] == 5.0
    assert selection["selection_reason"] == "authorized_prediction_horizon"
