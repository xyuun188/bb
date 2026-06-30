from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import run_phase3_go_no_go_report as report_cli


@pytest.mark.asyncio
async def test_phase3_go_no_go_report_extracts_total_gate(monkeypatch) -> None:
    async def fake_collect_system_audit_status(*, record_history: bool, source: str):
        assert record_history is False
        assert source == "phase3_go_no_go_report"
        return {
            "status": "warning",
            "checked_at": "2026-06-27T10:00:00+00:00",
            "summary": {"cards": 2},
            "cards": [
                {
                    "key": "phase3_go_no_go",
                    "status": "warning",
                    "details": {
                        "status": "paper_resume_ready",
                        "next_step": "resume_paper_pending_operator_approval",
                        "can_start_paper_with_operator_approval": True,
                        "starts_trading_service": False,
                        "submits_orders": False,
                    },
                }
            ],
            "root_causes": [],
            "issue_ledger": {"summary": {"unresolved": 1}},
        }

    monkeypatch.setattr(report_cli, "collect_system_audit_status", fake_collect_system_audit_status)

    report = await report_cli.collect_phase3_go_no_go_report()

    assert report["status"] == "paper_resume_ready"
    assert report["overall_audit_status"] == "warning"
    assert report["go_no_go"]["can_start_paper_with_operator_approval"] is True
    assert report["starts_trading_service"] is False
    assert report["submits_orders"] is False


def test_phase3_go_no_go_report_writes_latest(tmp_path) -> None:
    report = {
        "status": "blocked",
        "checked_at": "2026-06-27T10:00:00+00:00",
        "read_only": True,
    }

    artifacts = report_cli.write_report(report, tmp_path, indent=2)

    latest_path = tmp_path / "latest.json"
    assert latest_path.exists()
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    assert payload["report_artifacts"] == artifacts


@pytest.mark.asyncio
async def test_phase3_go_no_go_cli_keeps_stdout_json_only(monkeypatch, capsys) -> None:
    async def noisy_collect_report():
        print("executor log that must not pollute json stdout")
        return {
            "status": "paper_observation_healthy",
            "checked_at": "2026-06-27T10:00:00+00:00",
            "read_only": True,
        }

    monkeypatch.setattr(report_cli, "collect_phase3_go_no_go_report", noisy_collect_report)
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
    assert payload["status"] == "paper_observation_healthy"
    assert "executor log" not in captured.out
    assert "executor log that must not pollute json stdout" in captured.err
