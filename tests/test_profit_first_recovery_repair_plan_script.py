from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import plan_profit_first_recovery_repairs as report_cli


@pytest.mark.asyncio
async def test_recovery_repair_plan_cli_keeps_stdout_json_only(monkeypatch, capsys) -> None:
    async def noisy_collect_plan() -> dict:
        print("runtime log that must go to stderr")
        return {
            "report_type": "profit_first_recovery_repair_plan",
            "status": "blocked",
            "generated_at": "2026-06-29T10:00:00+00:00",
            "dry_run": True,
            "read_only": True,
            "audit_only": True,
            "mutates_database": False,
            "resume_allowed_by_this_plan": False,
        }

    monkeypatch.setattr(report_cli, "collect_profit_first_recovery_repair_plan", noisy_collect_plan)
    monkeypatch.setattr(
        report_cli,
        "parse_args",
        lambda: SimpleNamespace(
            json_indent=0,
            output_dir=None,
            stdout_only=True,
            fail_on_blocked=False,
        ),
    )

    exit_code = await report_cli._main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "blocked"
    assert payload["dry_run"] is True
    assert payload["mutates_database"] is False
    assert "runtime log" not in captured.out
    assert "runtime log that must go to stderr" in captured.err


def test_recovery_repair_plan_writes_latest(tmp_path) -> None:
    report = {
        "report_type": "profit_first_recovery_repair_plan",
        "status": "blocked",
        "generated_at": "2026-06-29T10:00:00+00:00",
        "dry_run": True,
    }

    artifacts = report_cli.write_report(report, tmp_path, indent=2)

    latest_path = tmp_path / "latest.json"
    assert latest_path.exists()
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    assert payload["report_artifacts"] == artifacts
    assert payload["status"] == "blocked"
