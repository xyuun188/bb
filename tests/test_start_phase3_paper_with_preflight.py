from __future__ import annotations

from typing import Any

import pytest

from scripts import start_phase3_paper_with_preflight as start_cli
from scripts.start_phase3_paper_with_preflight import (
    CONFIRMATION_PHRASE,
    CommandResult,
    build_phase3_paper_start_report,
    collect_phase3_paper_resume_preflight_via_command,
)


def test_paper_start_entrypoint_imports_runtime_env_bootstrap() -> None:
    source = start_cli.ROOT.joinpath(
        "scripts",
        "start_phase3_paper_with_preflight.py",
    ).read_text(encoding="utf-8")

    assert "from scripts.runtime_env_bootstrap import" in source
    assert "load_runtime_env_files(project_root=ROOT)" in source


def _ready_preflight() -> dict[str, Any]:
    return {
        "status": "ready",
        "can_resume_paper": True,
        "starts_trading_service": False,
        "submits_orders": False,
        "blockers": [],
    }


def _blocked_preflight() -> dict[str, Any]:
    return {
        "status": "blocked",
        "can_resume_paper": False,
        "starts_trading_service": False,
        "submits_orders": False,
        "blockers": [{"code": "okx_authoritative_sync_has_differences"}],
    }


def test_paper_resume_preflight_command_accepts_structured_json_with_logs() -> None:
    calls: list[list[str]] = []

    def runner(args: list[str], _timeout: float) -> CommandResult:
        calls.append(args)
        return CommandResult(
            status=0,
            stdout='loading runtime env\n{"status":"ready","can_resume_paper":true}\n',
        )

    report = collect_phase3_paper_resume_preflight_via_command(command_runner=runner)

    assert report["status"] == "ready"
    assert report["can_resume_paper"] is True
    assert calls
    assert calls[0][1].endswith("run_phase3_paper_resume_preflight.py")
    assert calls[0][-2:] == ["--json-indent", "0"]


def test_paper_resume_preflight_command_failure_is_blocking() -> None:
    def runner(_args: list[str], _timeout: float) -> CommandResult:
        return CommandResult(status=1, stdout="not json", stderr="db denied")

    report = collect_phase3_paper_resume_preflight_via_command(command_runner=runner)

    assert report["status"] == "blocked"
    assert report["can_resume_paper"] is False
    assert "preflight_command_failed" in {item["code"] for item in report["blockers"]}


@pytest.mark.asyncio
async def test_paper_start_entrypoint_defaults_to_report_only() -> None:
    calls: list[list[str]] = []

    def runner(args: list[str], _timeout: float) -> CommandResult:
        calls.append(args)
        return CommandResult(status=0, stdout="active")

    report = await build_phase3_paper_start_report(
        preflight_provider=_ready_preflight,
        command_runner=runner,
    )

    assert report["status"] == "ready_no_start"
    assert report["can_resume_paper"] is True
    assert report["start_requested"] is False
    assert report["starts_trading_service"] is False
    assert report["submits_orders"] is False
    assert calls == []


@pytest.mark.asyncio
async def test_paper_start_entrypoint_blocks_without_ready_preflight() -> None:
    calls: list[list[str]] = []

    def runner(args: list[str], _timeout: float) -> CommandResult:
        calls.append(args)
        return CommandResult(status=0, stdout="active")

    report = await build_phase3_paper_start_report(
        preflight_provider=_blocked_preflight,
        command_runner=runner,
        start_service=True,
        confirm_resume_paper=CONFIRMATION_PHRASE,
    )

    assert report["status"] == "blocked"
    assert report["starts_trading_service"] is False
    assert "preflight_not_ready" in {item["code"] for item in report["blockers"]}
    assert calls == []


@pytest.mark.asyncio
async def test_paper_start_entrypoint_requires_explicit_confirmation() -> None:
    calls: list[list[str]] = []

    def runner(args: list[str], _timeout: float) -> CommandResult:
        calls.append(args)
        return CommandResult(status=0, stdout="active")

    report = await build_phase3_paper_start_report(
        preflight_provider=_ready_preflight,
        command_runner=runner,
        start_service=True,
        confirm_resume_paper="yes",
    )

    assert report["status"] == "blocked"
    assert report["starts_trading_service"] is False
    assert "resume_confirmation_missing" in {item["code"] for item in report["blockers"]}
    assert calls == []


@pytest.mark.asyncio
async def test_paper_start_entrypoint_starts_only_after_ready_and_confirmed() -> None:
    calls: list[list[str]] = []

    def runner(args: list[str], _timeout: float) -> CommandResult:
        calls.append(args)
        if args[:2] == ["systemctl", "start"]:
            return CommandResult(status=0, stdout="")
        return CommandResult(status=0, stdout="active")

    report = await build_phase3_paper_start_report(
        preflight_provider=_ready_preflight,
        command_runner=runner,
        start_service=True,
        confirm_resume_paper=CONFIRMATION_PHRASE,
    )

    assert report["status"] == "started"
    assert report["starts_trading_service"] is True
    assert report["submits_orders"] is False
    assert calls == [
        ["systemctl", "start", "bb-paper-trading.service"],
        ["systemctl", "is-active", "bb-paper-trading.service"],
    ]


@pytest.mark.asyncio
async def test_paper_start_entrypoint_default_path_runs_preflight_command_before_start() -> None:
    calls: list[list[str]] = []

    def runner(args: list[str], _timeout: float) -> CommandResult:
        calls.append(args)
        if args[1].endswith("run_phase3_paper_resume_preflight.py"):
            return CommandResult(
                status=0,
                stdout='{"status":"ready","can_resume_paper":true,"blockers":[]}',
            )
        if args[:2] == ["systemctl", "start"]:
            return CommandResult(status=0, stdout="")
        return CommandResult(status=0, stdout="active")

    report = await build_phase3_paper_start_report(
        command_runner=runner,
        start_service=True,
        confirm_resume_paper=CONFIRMATION_PHRASE,
    )

    assert report["status"] == "started"
    assert [call[:2] for call in calls] == [
        [calls[0][0], calls[0][1]],
        ["systemctl", "start"],
        ["systemctl", "is-active"],
    ]
    assert calls[0][1].endswith("run_phase3_paper_resume_preflight.py")


@pytest.mark.asyncio
async def test_paper_start_entrypoint_default_path_blocks_when_preflight_command_fails() -> None:
    calls: list[list[str]] = []

    def runner(args: list[str], _timeout: float) -> CommandResult:
        calls.append(args)
        return CommandResult(status=1, stdout="", stderr="preflight failed")

    report = await build_phase3_paper_start_report(
        command_runner=runner,
        start_service=True,
        confirm_resume_paper=CONFIRMATION_PHRASE,
    )

    assert report["status"] == "blocked"
    assert report["starts_trading_service"] is False
    assert "preflight_not_ready" in {item["code"] for item in report["blockers"]}
    assert len(calls) == 1
    assert calls[0][1].endswith("run_phase3_paper_resume_preflight.py")
