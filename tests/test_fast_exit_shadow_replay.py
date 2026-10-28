from datetime import UTC, datetime, timedelta

from services.fast_exit_shadow_replay import (
    FAST_EXIT_TARGETS_PCT,
    build_fast_exit_shadow_report,
)


def _row(index: int, *, net: float, mfe: float, gross: float = 0.1) -> dict:
    return {
        "id": index + 1,
        "decision_id": index + 1,
        "decision_at": datetime(2026, 10, 27, tzinfo=UTC)
        + timedelta(minutes=index),
        "symbol": f"S{index}",
        "selected_side": "long",
        "long_mfe_pct": mfe,
        "long_gross_return_pct": gross,
        "long_net_return_pct": net,
    }


def test_fast_exit_replay_uses_mfe_and_reuses_realized_round_trip_cost() -> None:
    rows = [_row(index, net=-0.05, mfe=0.3, gross=0.1) for index in range(100)]

    report = build_fast_exit_shadow_report(rows, window={"independent_exam": False})

    candidate = report["fixed_target_candidates"]["0.20"]
    assert candidate["development"]["mean_net_return_pct"] == 0.05
    assert candidate["validation"]["mean_net_return_pct"] == 0.05
    assert report["safety"]["paper_execution_permission"] is False
    assert report["safety"]["orders_submitted"] is False


def test_fast_exit_replay_keeps_no_target_rows_at_the_five_minute_close() -> None:
    rows = [_row(index, net=-0.15, mfe=0.1, gross=0.1) for index in range(100)]

    report = build_fast_exit_shadow_report(rows, window={})

    candidate = report["fixed_target_candidates"]["0.20"]
    assert candidate["development"]["mean_net_return_pct"] == -0.15
    assert candidate["validation"]["mean_net_return_pct"] == -0.15
    assert report["selection"]["selected_target_pct"] is None
    assert report["status"] == "no_fast_exit_candidate"


def test_fast_exit_replay_uses_a_five_minute_purged_temporal_split() -> None:
    rows = [_row(index, net=-0.05, mfe=0.3, gross=0.1) for index in range(100)]

    report = build_fast_exit_shadow_report(rows, window={})

    assert report["input"]["development_group_count"] == 65
    assert report["input"]["validation_group_count"] == 30
    assert report["input"]["purge_minutes"] == 5
    assert report["input"]["selected_decision_group_count"] == 100


def test_fast_exit_replay_reports_candidate_grid_as_optimistic_development_only() -> None:
    report = build_fast_exit_shadow_report(
        [_row(index, net=-0.05, mfe=0.3, gross=0.1) for index in range(100)],
        window={"independent_exam": False},
    )

    assert set(report["fixed_target_candidates"]) == {
        f"{target:.2f}" for target in FAST_EXIT_TARGETS_PCT
    }
    assert report["window"]["independent_exam"] is False
    assert any("optimistic upper bound" in item for item in report["limitations"])
    assert report["safety"]["runtime_policy_changed"] is False


def test_fast_exit_replay_excludes_duplicate_decision_groups() -> None:
    rows = [_row(index, net=-0.05, mfe=0.3, gross=0.1) for index in range(100)]
    rows.append({**rows[0], "id": 1001})

    report = build_fast_exit_shadow_report(rows, window={})

    assert report["input"]["duplicate_decision_group_count"] == 1
    assert report["input"]["duplicate_row_count_excluded"] == 2
    assert report["input"]["selected_decision_group_count"] == 99
