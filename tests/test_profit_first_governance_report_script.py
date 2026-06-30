from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import run_profit_first_governance_report as report_cli


def _report() -> dict:
    return {
        "report_type": "profit_first_governance",
        "status": "ready",
        "generated_at": "2026-06-29T10:00:00+00:00",
        "read_only": True,
        "audit_only": True,
        "live_mutation": False,
        "can_submit_orders": False,
    }


def test_profit_first_governance_report_writes_latest(tmp_path) -> None:
    report = _report()

    artifacts = report_cli.write_report(report, tmp_path, indent=2)

    latest_path = tmp_path / "latest.json"
    assert latest_path.exists()
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    assert payload["report_artifacts"] == artifacts
    assert payload["status"] == "ready"


@pytest.mark.asyncio
async def test_profit_first_governance_cli_keeps_stdout_json_only(monkeypatch, capsys) -> None:
    async def noisy_collect_report(*, hours: int, limit: int) -> dict:
        assert hours == 24
        assert limit == 800
        print("runtime log that belongs on stderr")
        return _report()

    monkeypatch.setattr(report_cli, "collect_profit_first_governance_report", noisy_collect_report)
    monkeypatch.setattr(
        report_cli,
        "parse_args",
        lambda: SimpleNamespace(
            json_indent=0,
            output_dir=None,
            stdout_only=True,
            hours=24,
            limit=800,
            fail_on_incomplete=False,
        ),
    )

    exit_code = await report_cli._main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "ready"
    assert "runtime log" not in captured.out
    assert "runtime log that belongs on stderr" in captured.err
