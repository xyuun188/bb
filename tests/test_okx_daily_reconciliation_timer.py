from __future__ import annotations

from scripts import install_okx_daily_reconciliation_timer as timer_script


def test_okx_daily_reconciliation_timer_service_runs_as_bb_with_runtime_env() -> None:
    service = timer_script.render_service()
    timer = timer_script.render_timer(on_calendar="*-*-* 00:10:00")

    assert "User=bb" in service
    assert "Group=bb" in service
    assert "WorkingDirectory=/data/bb/app" in service
    assert "EnvironmentFile=/etc/bb/bb-runtime.env" in service
    assert "SuccessExitStatus=1" in service
    assert "run_phase3_okx_fact_sync.py --mode paper --apply-order-sync --json-indent 0" in service
    assert "run_okx_daily_reconciliation_report.py" not in service
    assert "bb-paper-trading.service" not in service
    assert "OnCalendar=*-*-* 00:10:00" in timer
    assert "Persistent=true" in timer
    assert "Unit=bb-okx-daily-reconciliation.service" in timer


def test_okx_daily_reconciliation_timer_dry_run_does_not_connect(
    monkeypatch,
    capsys,
) -> None:
    def fail_connect(*_args, **_kwargs):
        raise AssertionError("dry-run must not connect to remote server")

    monkeypatch.setattr(timer_script, "connect_remote_ssh", fail_connect)

    timer_script.install_timer(dry_run=True)

    output = capsys.readouterr().out
    assert "bb-okx-daily-reconciliation.service" in output
    assert "bb-okx-daily-reconciliation.timer" in output
    assert "run_phase3_okx_fact_sync.py" in output
