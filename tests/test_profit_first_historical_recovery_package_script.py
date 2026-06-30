from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import plan_profit_first_historical_recovery_package as report_cli


@pytest.mark.asyncio
async def test_historical_recovery_package_cli_keeps_stdout_json_only(monkeypatch, capsys) -> None:
    async def noisy_collect_package(**kwargs):
        assert kwargs["entry_decision_ids"] == [9549]
        assert kwargs["use_current_blockers"] is False
        print("runtime log goes to stderr")
        return {
            "report_type": "profit_first_historical_recovery_package",
            "status": "ready",
            "generated_at": "2026-06-29T10:00:00+00:00",
            "dry_run": True,
            "read_only": True,
            "mutates_database": False,
            "resume_allowed_by_this_package": False,
        }

    monkeypatch.setattr(report_cli, "collect_historical_recovery_package", noisy_collect_package)
    monkeypatch.setattr(
        report_cli,
        "parse_args",
        lambda: SimpleNamespace(
            json_indent=0,
            output_dir=None,
            stdout_only=True,
            skip_current_blockers=True,
            entry_decision_id=[9549],
            exit_decision_id=[],
            order_id=[],
            exchange_order_id=[],
        ),
    )

    exit_code = await report_cli._main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["dry_run"] is True
    assert payload["mutates_database"] is False
    assert "runtime log" not in captured.out
    assert "runtime log goes to stderr" in captured.err


def test_historical_recovery_package_writes_latest(tmp_path) -> None:
    report = {
        "report_type": "profit_first_historical_recovery_package",
        "status": "ready",
        "generated_at": "2026-06-29T10:00:00+00:00",
        "dry_run": True,
    }

    artifacts = report_cli.write_report(report, tmp_path, indent=2)

    latest_path = tmp_path / "latest.json"
    assert latest_path.exists()
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    assert payload["report_artifacts"] == artifacts
    assert payload["status"] == "ready"
