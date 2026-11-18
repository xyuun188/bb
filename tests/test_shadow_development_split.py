from datetime import UTC, datetime, timedelta

from services.shadow_development_split import split_shadow_development_rows


def _row(group: int, minute: int, horizon: int = 5) -> dict:
    decision_at = datetime(2026, 11, 18, tzinfo=UTC) + timedelta(minutes=minute)
    label_at = decision_at + timedelta(minutes=horizon)
    return {
        "id": group * 100 + horizon,
        "decision_id": group,
        "horizon_minutes": horizon,
        "decision_at": decision_at.isoformat(),
        "label_at": label_at.isoformat(),
        "updated_at": label_at,
    }


def test_splits_all_horizons_at_same_decision_time_and_purges_late_labels() -> None:
    rows = [_row(group, group * 10) for group in range(1, 11)]
    rows.extend((_row(2, 20, 60), _row(9, 90, 60), _row(11, 80)))
    rows[0]["updated_at"] = rows[9]["updated_at"]
    development, validation, report = split_shadow_development_rows(rows[::-1])

    assert {(row["decision_id"], row["horizon_minutes"]) for row in development} == {
        (group, 5) for group in range(2, 8)
    }
    assert {row["decision_id"] for row in validation} == {8, 9, 10, 11}
    assert report["validation_start"] == rows[7]["decision_at"]
    assert report["excluded_row_counts"]["outcome_not_available_before_validation"] == 2


def test_excludes_duplicates_conflicting_time_and_unverifiable_outcomes() -> None:
    rows = [
        _row(1, 10), {**_row(1, 90), "id": 777}, _row(1, 10, 60),
        _row(2, 20), _row(2, 21, 60),
        {**_row(3, 30), "label_at": None},
        {**_row(4, 40), "label_at": _row(4, 40)["decision_at"]},
        {**_row(5, 50), "decision_at": None},
        {**_row(6, 60), "decision_id": None},
        {**_row(7, 70), "updated_at": datetime(2026, 11, 18, tzinfo=UTC)},
        _row(8, 80),
    ]
    development, validation, report = split_shadow_development_rows(rows)

    assert [row["decision_id"] for row in development] == [8]
    assert not validation
    assert report["duplicate_decision_group_count"] == 1
    assert report["excluded_row_counts"] == {
        "duplicate_group": 3,
        "conflicting_decision_time": 2,
        "invalid_timing": 4,
        "missing_decision_id": 1,
    }
