from __future__ import annotations

from scripts import install_phase3_paper_resume_observation_timer as timer_script


def test_phase3_paper_resume_observation_timer_service_is_read_only() -> None:
    service = timer_script.render_service(
        observation_hours=4,
        min_created_shadow_samples=9,
        min_completed_shadow_samples=3,
        report_max_age_seconds=3600,
    )
    timer = timer_script.render_timer(on_calendar="*-*-* *:07,37:00")

    assert "User=bb" in service
    assert "Group=bb" in service
    assert "WorkingDirectory=/data/bb/app" in service
    assert "EnvironmentFile=/etc/bb/bb-runtime.env" in service
    assert "run_phase3_paper_resume_observation.py" in service
    assert "--observation-hours 4" in service
    assert "--min-created-shadow-samples 9" in service
    assert "--min-completed-shadow-samples 3" in service
    assert "--report-max-age-seconds 3600" in service
    assert "bb-paper-trading.service" not in service
    assert "systemctl start" not in service
    assert "OnCalendar=*-*-* *:07,37:00" in timer
    assert "Persistent=true" in timer
    assert "Unit=bb-phase3-paper-resume-observation.service" in timer


def test_phase3_paper_resume_observation_timer_dry_run_does_not_connect(
    monkeypatch,
    capsys,
) -> None:
    def fail_connect(*_args, **_kwargs):
        raise AssertionError("dry-run must not connect to remote server")

    monkeypatch.setattr(timer_script, "connect_remote_ssh", fail_connect)

    timer_script.install_timer(dry_run=True)

    output = capsys.readouterr().out
    assert "bb-phase3-paper-resume-observation.service" in output
    assert "bb-phase3-paper-resume-observation.timer" in output
    assert "run_phase3_paper_resume_observation.py" in output
    assert "starts_trading_service" in output
    assert "False" in output


def test_phase3_paper_resume_observation_run_now_only_starts_observation(
    monkeypatch,
) -> None:
    commands: list[str] = []

    class FakeSftp:
        def file(self, *_args, **_kwargs):
            return self

        def write(self, _content: str) -> None:
            return None

        def chmod(self, *_args, **_kwargs) -> None:
            return None

        def close(self) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeSsh:
        def open_sftp(self):
            return FakeSftp()

        def close(self) -> None:
            return None

    monkeypatch.setattr(timer_script, "connect_remote_ssh", lambda *_args, **_kwargs: FakeSsh())

    def fake_run_remote_text(_ssh, command: str, **_kwargs) -> str:
        commands.append(command)
        return "ok"

    monkeypatch.setattr(timer_script, "run_remote_text", fake_run_remote_text)

    timer_script.install_timer(run_now=True)

    joined = "\n".join(commands)
    assert "systemctl start bb-phase3-paper-resume-observation.service" in joined
    assert "systemctl start bb-paper-trading.service" not in joined
    assert "systemctl is-active bb-paper-trading.service" in joined
