from __future__ import annotations

from datetime import UTC, datetime

from scripts import audit_online_nested_current_development as audit


def test_remote_nested_research_query_is_bounded_native_and_cost_complete() -> None:
    source = (
        audit.REMOTE_SCRIPT.replace("__GROUP_LIMIT__", "1000")
        .replace("__HOURS__", "6")
        .replace("__CUTOFF__", repr("2026-10-24T02:27:00Z"))
        .replace("__REVISION__", repr("v1"))
    )
    compile(source, "<online-nested-development>", "exec")

    assert "ShadowBacktest.updated_at >= since" in source
    assert "ShadowBacktest.updated_at < CUTOFF" in source
    assert "original_decision_timestamp" in source
    assert "decision_timestamp >= DECISION_SINCE" in source
    assert "decision_timestamp < CUTOFF" in source
    assert "== NATIVE_SOURCE" in source
    assert '.as_boolean().is_(True)' in source
    assert ".limit(GROUP_LIMIT)" in source
    assert ".group_by(group_key)" in source
    assert "group_key.in_(select(selected_groups.c.decision_group_key))" in source
    assert "exam_labels_evaluated" in source
    assert "outer_fold_coverage" in source
    assert "legacy_all_horizon_development_sha256" in source
    assert "build_nested_hgb_development_report" in source
    assert 'FAMILY == "hgb"' in source
    assert "services/ml_nested_development_hgb.py" in source
    assert "build_nested_extra_trees_development_report" in source
    assert 'FAMILY == "extra_trees"' in source
    assert "services/ml_nested_development_extra_trees.py" in source


def test_remote_runner_uses_isolated_bounded_resources_and_unique_unit() -> None:
    command = audit._build_remote_command(
        1000,
        6,
        datetime(2026, 10, 24, 2, 27, tzinfo=UTC),
        "bb-nested-dev-20261024T022700Z",
    )

    assert "--unit=bb-nested-dev-20261024T022700Z" in command
    assert "--property=CPUQuota=25%" in command
    assert "--property=MemoryMax=1G" in command
    assert "--property=RuntimeMaxSec=300s" in command
    assert "--setenv=DATABASE_POOL_SIZE=1" in command
    assert "primary-horizon-label-time-only" in audit.REMOTE_SCRIPT


def test_hgb_family_uses_a_distinct_remote_unit_and_report_path() -> None:
    command = audit._build_remote_command(
        1000,
        48,
        datetime(2026, 10, 24, 23, 30, tzinfo=UTC),
        "bb-nested-dev-hgb-v1-20261024T233000Z",
        revision="v1",
        family="hgb",
    )

    assert "--unit=bb-nested-dev-hgb-v1-20261024T233000Z" in command
    assert "build_nested_hgb_development_report" in audit.REMOTE_SCRIPT


def test_remote_command_can_pin_a_distinct_decision_window() -> None:
    command = audit._build_remote_command(
        79,
        6,
        datetime(2026, 10, 26, 18, 45, tzinfo=UTC),
        "bb-nested-dev-ridge-v3-20261026T184500Z",
        revision="v3",
        family="ridge",
        decision_since=datetime(2026, 10, 26, 12, 45, tzinfo=UTC),
    )

    assert "DECISION_SINCE" in audit.REMOTE_SCRIPT
    assert "--unit=bb-nested-dev-ridge-v3-20261026T184500Z" in command
    assert "__DECISION_SINCE__" not in command


def test_cutoff_requires_timezone_and_normalizes_to_utc() -> None:
    assert audit._parse_cutoff("2026-10-24T10:27:00+08:00") == datetime(
        2026, 10, 24, 2, 27, tzinfo=UTC
    )

    try:
        audit._parse_cutoff("2026-10-24T02:27:00")
    except ValueError as exc:
        assert "timezone" in str(exc)
    else:
        raise AssertionError("naive cutoff must not be accepted")


def test_remote_hash_parser_preserves_relative_dependency_names() -> None:
    output = (
        "a" * 64 + "  services/ml_nested_development.py\n"
        + "b" * 64 + "  services/ml_training_dataset.py\n"
    )

    assert audit._parse_hashes(output) == {
        "services/ml_nested_development.py": "a" * 64,
        "services/ml_training_dataset.py": "b" * 64,
    }


def test_same_window_revision_requires_matching_input_fingerprint() -> None:
    report = {
        "input": {"development_input_sha256": "abc123"},
    }

    audit._validate_expected_input_hash(report, "abc123")
    try:
        audit._validate_expected_input_hash(report, "different")
    except RuntimeError as exc:
        assert "fingerprint changed" in str(exc)
    else:
        raise AssertionError("changed samples must stop the comparison")


def test_same_window_data_selection_matches_legacy_development_fingerprint() -> None:
    report = {
        "input": {"legacy_all_horizon_development_sha256": "prior"},
    }

    audit._validate_expected_legacy_hash(report, "prior")
    try:
        audit._validate_expected_legacy_hash(report, "changed")
    except RuntimeError as exc:
        assert "legacy development fingerprint changed" in str(exc)
    else:
        raise AssertionError("changed source groups must stop the comparison")
