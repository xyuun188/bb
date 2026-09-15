from __future__ import annotations

from scripts.start_online_phase3_paper_with_preflight import (
    CONFIRMATION_PHRASE,
    build_remote_command,
)


def test_online_paper_start_defaults_to_remote_preflight_only() -> None:
    command = build_remote_command(start_service=False, confirmation="", json_indent=0)

    assert command.startswith("cd /data/bb/app && exec .venv/bin/python")
    assert "scripts/start_phase3_paper_with_preflight.py" in command
    assert "--stdout-only --json-indent 0" in command
    assert "--start-service" not in command
    assert "systemctl" not in command


def test_online_paper_start_forwards_explicit_confirmation() -> None:
    command = build_remote_command(
        start_service=True,
        confirmation=CONFIRMATION_PHRASE,
        json_indent=2,
    )

    assert "--start-service" in command
    assert f"--confirm-resume-paper {CONFIRMATION_PHRASE}" in command
    assert "--json-indent 2" in command
