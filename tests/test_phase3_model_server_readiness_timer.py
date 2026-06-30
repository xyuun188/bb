from __future__ import annotations

from scripts import install_phase3_model_server_readiness_timer as timer_script


def test_phase3_model_server_readiness_timer_service_runs_as_bb_with_runtime_env() -> None:
    service = timer_script.render_service()
    timer = timer_script.render_timer(on_calendar="*-*-* 00:55:00")

    assert "User=bb" in service
    assert "Group=bb" in service
    assert "WorkingDirectory=/data/bb/app" in service
    assert "EnvironmentFile=/etc/bb/bb-runtime.env" in service
    assert "run_phase3_model_server_readiness_audit.py --json-indent 0" in service
    assert "bb-paper-trading.service" not in service
    assert "systemctl start" not in service
    assert "OnCalendar=*-*-* 00:55:00" in timer
    assert "Persistent=true" in timer
    assert "Unit=bb-phase3-model-server-readiness.service" in timer


def test_phase3_model_server_readiness_timer_dry_run_does_not_connect(
    monkeypatch,
    capsys,
) -> None:
    def fail_connect(*_args, **_kwargs):
        raise AssertionError("dry-run must not connect to remote server")

    monkeypatch.setattr(timer_script, "connect_remote_ssh", fail_connect)

    timer_script.install_timer(dry_run=True)

    output = capsys.readouterr().out
    assert "bb-phase3-model-server-readiness.service" in output
    assert "bb-phase3-model-server-readiness.timer" in output
    assert "run_phase3_model_server_readiness_audit.py" in output
