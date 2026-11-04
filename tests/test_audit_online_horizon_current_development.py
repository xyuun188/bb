from __future__ import annotations

from datetime import UTC, datetime

from scripts import audit_online_horizon_current_development as audit


def test_horizon_development_command_is_bounded_and_cutoff_bound() -> None:
    cutoff = datetime(2026, 10, 24, 5, 57, tzinfo=UTC)
    command = audit._build_remote_command(
        limit=1000,
        hours=24,
        cutoff=cutoff,
        unit="bb-horizon-dev-v1-20261024T055700Z",
        revision="v1",
    )
    assert "--property=CPUQuota=25%" in command
    assert "--property=MemoryMax=1G" in command
    assert "--property=RuntimeMaxSec=300s" in command
    assert audit._remote_runtime_limit_seconds(1000) == 300
    assert "--setenv=DATABASE_POOL_SIZE=1" in command
    rendered = audit._render_remote_script(
        limit=1000,
        hours=24,
        cutoff=cutoff,
        revision="v1",
    )
    assert "2026-10-24T05:57:00Z" in rendered
    assert "__CUTOFF__" not in rendered
    assert '"runtime_limit_seconds": 300' in rendered
    assert ".order_by(func.max(ShadowBacktest.updated_at).desc(), group_key.desc())" in rendered
    assert "exam_labels_evaluated" in rendered
    assert "complete_group_keys = set(" in rendered
    assert "complete_group_count =" in rendered
    assert '"fold_coverage_diagnostics": []' in rendered
    assert 'horizon_specific_purge_diagnostics' in rendered
    assert '"inner_folds": selection.get("inner_folds", [])' in rendered
    assert '"horizon_specific_purge": "diagnostic_only_not_used_for_selection"' in rendered


def test_large_horizon_development_batch_gets_a_bounded_runtime_budget() -> None:
    cutoff = datetime(2026, 10, 24, 5, 57, tzinfo=UTC)
    command = audit._build_remote_command(
        limit=2500,
        hours=24,
        cutoff=cutoff,
        unit="bb-horizon-dev-v1-20261024T055700Z",
        revision="v1",
    )

    assert audit._remote_runtime_limit_seconds(2500) == 900
    assert "--property=RuntimeMaxSec=900s" in command
    assert "--property=CPUQuota=25%" in command
    assert "--property=MemoryMax=1G" in command
    rendered = audit._render_remote_script(
        limit=2500,
        hours=24,
        cutoff=cutoff,
        revision="v2",
    )
    assert '"runtime_limit_seconds": 900' in rendered
    assert "__RUNTIME_LIMIT_SECONDS__" not in rendered


def test_same_window_revision_requires_matching_input_fingerprint() -> None:
    report = {"input": {"development_input_sha256": "abc123"}}

    audit._validate_expected_input_hash(report, "abc123")
    try:
        audit._validate_expected_input_hash(report, "different")
    except RuntimeError as exc:
        assert "fingerprint changed" in str(exc)
    else:
        raise AssertionError("changed samples must stop the comparison")
