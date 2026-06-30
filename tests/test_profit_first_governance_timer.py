from __future__ import annotations

from scripts import install_profit_first_governance_timer as timer_script


def test_profit_first_governance_timer_service_is_read_only() -> None:
    service = timer_script.render_service()
    timer = timer_script.render_timer(on_calendar="*-*-* 00:24:00")

    assert "User=bb" in service
    assert "Group=bb" in service
    assert "WorkingDirectory=/data/bb/app" in service
    assert "EnvironmentFile=/etc/bb/bb-runtime.env" in service
    assert "run_profit_first_governance_report.py --json-indent 0" in service
    assert "bb-paper-trading.service" not in service
    assert "systemctl start" not in service
    assert "OnCalendar=*-*-* 00:24:00" in timer
    assert "Persistent=true" in timer
    assert "Unit=bb-profit-first-governance.service" in timer


def test_profit_first_governance_timer_dry_run_does_not_connect(monkeypatch, capsys) -> None:
    def fail_connect(*_args, **_kwargs):
        raise AssertionError("dry-run must not connect to remote server")

    monkeypatch.setattr(timer_script, "connect_remote_ssh", fail_connect)

    timer_script.install_timer(dry_run=True)

    output = capsys.readouterr().out
    assert "bb-profit-first-governance.service" in output
    assert "bb-profit-first-governance.timer" in output
    assert "run_profit_first_governance_report.py" in output
    assert "starts_trading_service" in output
    assert "False" in output


def test_profit_first_governance_run_now_only_starts_governance_service(monkeypatch) -> None:
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
    assert "systemctl start bb-profit-first-governance.service" in joined
    assert "systemctl start bb-paper-trading.service" not in joined
    assert "systemctl is-active bb-paper-trading.service" in joined
