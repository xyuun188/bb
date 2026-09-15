from __future__ import annotations

import json

from scripts.run_online_phase3_paper_resume_preflight import (
    _decode_json,
    build_remote_command,
)


def test_online_paper_preflight_uses_read_only_online_runtime() -> None:
    command = build_remote_command(json_indent=0)

    assert "runuser -u bb" in command
    assert "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql" in command
    assert "scripts/run_phase3_paper_resume_preflight.py" in command
    assert "--stdout-only --json-indent 0" in command
    assert "--start-service" not in command
    assert "systemctl" not in command


def test_online_paper_preflight_decodes_last_structured_line() -> None:
    payload = {"status": "blocked", "can_resume_paper": False, "read_only": True}

    assert _decode_json("runtime note\n" + json.dumps(payload)) == payload
