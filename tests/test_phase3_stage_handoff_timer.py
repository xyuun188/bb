from __future__ import annotations

from scripts import install_phase3_stage_handoff_timer as timer_script


def test_phase3_stage_handoff_timer_service_is_read_only() -> None:
    service = timer_script.render_service()

    assert "run_phase3_stage_handoff_report.py --json-indent 0" in service
    assert "systemctl start bb-paper-trading.service" not in service
    assert "EnvironmentFile=/etc/bb/bb-runtime.env" in service


def test_phase3_stage_handoff_timer_dry_run_does_not_connect(monkeypatch, capsys) -> None:
    def fail_connect(*_args, **_kwargs):
        raise AssertionError("dry-run must not connect")

    monkeypatch.setattr(timer_script, "connect_remote_ssh", fail_connect)

    timer_script.install_timer(dry_run=True)
    output = capsys.readouterr().out

    assert "bb-phase3-stage-handoff.timer" in output
    assert "run_phase3_stage_handoff_report.py" in output


def test_phase3_stage_handoff_run_now_only_starts_handoff_service(monkeypatch) -> None:
    commands: list[str] = []

    class FakeSftp:
        def file(self, *_args, **_kwargs):
            class File:
                def __enter__(self):
                    return self

                def __exit__(self, *_exc):
                    return False

                def write(self, _content):
                    return None

            return File()

        def chmod(self, *_args, **_kwargs):
            return None

        def close(self):
            return None

    class FakeSsh:
        def open_sftp(self):
            return FakeSftp()

        def close(self):
            return None

    monkeypatch.setattr(timer_script, "connect_remote_ssh", lambda *_args, **_kwargs: FakeSsh())

    def fake_run_remote_text(_ssh, command, **_kwargs):
        commands.append(command)
        return "ok"

    monkeypatch.setattr(timer_script, "run_remote_text", fake_run_remote_text)

    timer_script.install_timer(run_now=True)

    joined = "\n".join(commands)
    assert "systemctl start bb-phase3-stage-handoff.service" in joined
    assert "systemctl start bb-paper-trading.service" not in joined
    assert "systemctl is-active bb-paper-trading.service" in joined
