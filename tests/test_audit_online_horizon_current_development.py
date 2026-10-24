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
    assert "--setenv=DATABASE_POOL_SIZE=1" in command
    rendered = audit._render_remote_script(
        limit=1000,
        hours=24,
        cutoff=cutoff,
        revision="v1",
    )
    assert "2026-10-24T05:57:00Z" in rendered
    assert "__CUTOFF__" not in rendered
    assert "exam_labels_evaluated" in rendered
