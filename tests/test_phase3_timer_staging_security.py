from __future__ import annotations

import posixpath

import pytest

from scripts import (
    install_phase3_go_no_go_timer,
    install_phase3_market_data_warmup_timer,
    install_phase3_model_server_readiness_timer,
    install_phase3_rebuild_preflight_timer,
    install_phase3_stage_handoff_timer,
)

TIMER_INSTALLERS = (
    install_phase3_go_no_go_timer,
    install_phase3_market_data_warmup_timer,
    install_phase3_model_server_readiness_timer,
    install_phase3_rebuild_preflight_timer,
    install_phase3_stage_handoff_timer,
)


@pytest.mark.parametrize("timer_script", TIMER_INSTALLERS)
def test_timer_units_use_private_random_project_staging(
    timer_script,
    monkeypatch,
) -> None:
    uploaded_paths: list[str] = []
    commands: list[str] = []

    class FakeRemoteFile:
        def write(self, _content: str) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    class FakeSftp:
        def file(self, path: str, _mode: str) -> FakeRemoteFile:
            uploaded_paths.append(path)
            return FakeRemoteFile()

        def chmod(self, *_args) -> None:
            return None

        def close(self) -> None:
            return None

    class FakeSsh:
        def open_sftp(self) -> FakeSftp:
            return FakeSftp()

        def close(self) -> None:
            return None

    monkeypatch.setattr(timer_script, "connect_remote_ssh", lambda *_args, **_kwargs: FakeSsh())

    def fake_run_remote_text(_ssh, command: str, **_kwargs) -> str:
        commands.append(command)
        return "ok"

    monkeypatch.setattr(timer_script, "run_remote_text", fake_run_remote_text)

    timer_script.install_timer()

    assert len(uploaded_paths) == 2
    assert len(set(uploaded_paths)) == 2
    assert all(path.startswith("/data/bb/app/tmp/systemd-unit-stage/") for path in uploaded_paths)
    global_temp_prefix = f"{posixpath.join('/', 'tmp')}/"
    assert all(not path.startswith(global_temp_prefix) for path in uploaded_paths)
    joined = "\n".join(commands)
    assert "-m 0700 '/data/bb/app/tmp/systemd-unit-stage'" in joined
    assert "rm -f --" in joined
