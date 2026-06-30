from __future__ import annotations

from core import remote_ssh


def test_remote_ssh_respects_requested_output_limit_above_default() -> None:
    assert remote_ssh._normalize_output_limit(80_000) == 80_000


def test_remote_ssh_caps_unbounded_output_limit() -> None:
    assert remote_ssh._normalize_output_limit(500_000) == remote_ssh.MAX_REMOTE_OUTPUT_TEXT_LIMIT

