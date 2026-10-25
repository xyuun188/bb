"""Controlled Phase 3 paper-trading start entrypoint.

This script exists to prevent ad-hoc paper restarts. By default it only runs
the paper-resume preflight and writes a report. It starts the service only when
all hard gates pass and the operator provides the explicit confirmation token.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runtime_env_bootstrap import (  # noqa: E402
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

load_runtime_env_files(project_root=ROOT)

# These imports must follow runtime environment loading because settings are read at import time.
from core.safe_output import safe_error_text  # noqa: E402
from scripts.apply_phase3_paper_resume_control import (  # noqa: E402
    CONFIRMATION_PHRASE,
)
from scripts.run_phase3_paper_resume_preflight import (  # noqa: E402
    write_report as write_preflight_report,
)

DEFAULT_SERVICE_NAME = "bb-paper-trading.service"
OBSERVATION_ONLY_CONFIRMATION_PHRASE = "START_OBSERVATION_ONLY_WITHOUT_ENTRIES"
_OBSERVATION_ONLY_ALLOWED_BLOCKER = "phase3_quant_api_child_endpoints_unavailable"
_OBSERVATION_ONLY_REQUIRED_CHILDREN = frozenset(
    {
        "profit_prediction",
        "time_series_prediction",
        "sentiment_analysis",
        "exit_advice",
    }
)


@dataclass(frozen=True, slots=True)
class CommandResult:
    status: int
    stdout: str = ""
    stderr: str = ""


CommandRunner = Callable[[list[str], float], CommandResult]
PreflightProvider = Callable[[], Any]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _run_command(args: list[str], timeout: float) -> CommandResult:
    try:
        result = subprocess.run(  # noqa: S603 - args are fixed allowlisted systemctl calls.
            args,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return CommandResult(status=124, stderr=safe_error_text(exc, limit=240))
    return CommandResult(
        status=int(result.returncode),
        stdout=str(result.stdout or "").strip(),
        stderr=str(result.stderr or "").strip(),
    )


def _parse_json_stdout(stdout: str) -> dict[str, Any]:
    text = str(stdout or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                payload, end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and not text[index + end :].strip():
                return payload
        return {}
    return payload if isinstance(payload, dict) else {}


def collect_phase3_paper_resume_preflight_via_command(
    *,
    command_runner: CommandRunner = _run_command,
    timeout_seconds: float = 90.0,
) -> dict[str, Any]:
    """Run preflight in a child process so online root callers can drop only the preflight."""

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_phase3_paper_resume_preflight.py"),
        "--json-indent",
        "0",
    ]
    result = command_runner(command, max(float(timeout_seconds or 1.0), 1.0))
    payload = _parse_json_stdout(result.stdout)
    if result.status == 0 and payload:
        return payload
    return {
        "status": "blocked",
        "read_only": True,
        "audit_only": True,
        "can_resume_paper": False,
        "blockers": [
            {
                "code": "preflight_command_failed",
                "severity": "blocking",
                "message": "Paper-resume preflight command did not return a successful structured report.",
                "evidence": {
                    "command": command,
                    "status": result.status,
                    "stdout": str(result.stdout or "")[-1000:],
                    "stderr": str(result.stderr or "")[-1000:],
                },
            }
        ],
    }


def apply_paper_control_via_command(
    *,
    action: str,
    confirmation: str,
    command_runner: CommandRunner = _run_command,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "apply_phase3_paper_resume_control.py"),
        "--action",
        str(action),
        "--confirm-resume-paper",
        str(confirmation or ""),
        "--stdout-only",
        "--json-indent",
        "0",
        "--fail-on-blocked",
    ]
    result = command_runner(command, max(float(timeout_seconds or 1.0), 1.0))
    payload = _parse_json_stdout(result.stdout)
    if result.status == 0 and payload.get("status") == "ok":
        return payload
    return {
        "status": "blocked",
        "action": action,
        "blockers": [
            {
                "code": f"paper_control_{action}_failed",
                "severity": "blocking",
                "message": f"Paper control {action} did not complete successfully.",
                "evidence": {
                    "status": result.status,
                    "stdout": str(result.stdout or "")[-1000:],
                    "stderr": str(result.stderr or "")[-1000:],
                },
            }
        ],
    }


def _artifact_wait_allows_observation_only(preflight: dict[str, Any]) -> bool:
    """Allow a paused, data-collection-only start for a known artifact wait.

    This deliberately accepts exactly one fail-closed condition: the platform
    is reachable and the four quant child contracts are unavailable only
    because the compatible model bundle is absent.  It never permits a
    general preflight failure, exchange/account problem, stale specialist
    report, or active paper process.
    """

    if bool(preflight.get("can_resume_paper")):
        return True
    blockers = preflight.get("blockers")
    if not isinstance(blockers, list) or len(blockers) != 1:
        return False
    blocker = blockers[0] if isinstance(blockers[0], dict) else {}
    if str(blocker.get("code") or "") != _OBSERVATION_ONLY_ALLOWED_BLOCKER:
        return False
    evidence = blocker.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    children = evidence.get("child_endpoints")
    children = children if isinstance(children, dict) else {}
    if set(children) != set(_OBSERVATION_ONLY_REQUIRED_CHILDREN):
        return False
    return all(
        isinstance(item, dict)
        and item.get("status") == "artifact_unavailable"
        and item.get("error") == "model_bundle_unavailable"
        for item in children.values()
    )


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def build_phase3_paper_start_report(
    *,
    preflight_provider: PreflightProvider | None = None,
    command_runner: CommandRunner = _run_command,
    start_service: bool = False,
    confirm_resume_paper: str = "",
    service_name: str = DEFAULT_SERVICE_NAME,
    command_timeout_seconds: float = 30.0,
    preflight_timeout_seconds: float = 90.0,
    observation_only: bool = False,
    confirm_observation_only: str = "",
) -> dict[str, Any]:
    preflight = await _maybe_await(
        (
            preflight_provider
            or (
                lambda: collect_phase3_paper_resume_preflight_via_command(
                    command_runner=command_runner,
                    timeout_seconds=preflight_timeout_seconds,
                )
            )
        )()
    )
    if not isinstance(preflight, dict):
        preflight = {
            "status": "blocked",
            "can_resume_paper": False,
            "error": f"preflight returned {type(preflight).__name__}",
        }

    can_resume = bool(preflight.get("can_resume_paper"))
    observation_only_allowed = bool(
        observation_only and _artifact_wait_allows_observation_only(preflight)
    )
    can_start = can_resume or observation_only_allowed
    confirmation = confirm_observation_only if observation_only else confirm_resume_paper
    expected_confirmation = (
        OBSERVATION_ONLY_CONFIRMATION_PHRASE if observation_only else CONFIRMATION_PHRASE
    )
    confirmed = str(confirmation or "").strip() == expected_confirmation
    action_status = "preflight_only"
    blockers: list[dict[str, Any]] = []
    command_results: list[dict[str, Any]] = []
    control_actions: list[dict[str, Any]] = []
    started = False
    control_state_resumed = False
    control_state_paused = False
    rollback_paused = False

    if not can_start:
        blockers.append(
            {
                "code": (
                    "observation_only_not_safe"
                    if observation_only
                    else "preflight_not_ready"
                ),
                "severity": "blocking",
                "message": (
                    "Observation-only start requires the single verified "
                    "model-bundle-unavailable blocker."
                    if observation_only
                    else "Paper trading cannot be started until preflight can_resume_paper=true."
                ),
            }
        )
    if start_service and not confirmed:
        blockers.append(
            {
                "code": "resume_confirmation_missing",
                "severity": "blocking",
                "message": (
                    "Observation-only start requires --confirm-observation-only "
                    f"{OBSERVATION_ONLY_CONFIRMATION_PHRASE}."
                    if observation_only
                    else "Starting paper trading requires --confirm-resume-paper "
                    f"{CONFIRMATION_PHRASE}."
                ),
            }
        )

    if start_service:
        if blockers:
            action_status = "blocked"
        else:
            control_action = "pause" if observation_only else "resume"
            control_result = apply_paper_control_via_command(
                action=control_action,
                confirmation=CONFIRMATION_PHRASE,
                command_runner=command_runner,
                timeout_seconds=command_timeout_seconds,
            )
            control_actions.append(control_result)
            control_after = control_result.get("after")
            control_after = control_after if isinstance(control_after, dict) else {}
            control_applied = bool(
                control_result.get("status") == "ok"
                and control_result.get("action") == control_action
                and control_after.get("paused") is observation_only
            )
            control_state_resumed = (
                not observation_only and control_applied
            )
            control_state_paused = (
                observation_only and control_applied
            )
            if not (control_state_resumed or control_state_paused):
                action_status = (
                    "control_pause_failed"
                    if observation_only
                    else "control_resume_failed"
                )
            else:
                start_result = command_runner(
                    ["systemctl", "start", service_name],
                    max(float(command_timeout_seconds or 1.0), 1.0),
                )
                command_results.append(
                    {
                        "command": ["systemctl", "start", service_name],
                        "status": start_result.status,
                        "stdout": start_result.stdout,
                        "stderr": start_result.stderr,
                    }
                )
                if start_result.status == 0:
                    active_result = command_runner(
                        ["systemctl", "is-active", service_name],
                        max(float(command_timeout_seconds or 1.0), 1.0),
                    )
                    command_results.append(
                        {
                            "command": ["systemctl", "is-active", service_name],
                            "status": active_result.status,
                            "stdout": active_result.stdout,
                            "stderr": active_result.stderr,
                        }
                    )
                    started = active_result.status == 0 and active_result.stdout.strip() == "active"
                    action_status = "started" if started else "start_verification_failed"
                else:
                    action_status = "start_failed"
            if not started:
                if control_state_resumed:
                    rollback_result = apply_paper_control_via_command(
                        action="pause",
                        confirmation=confirmation,
                        command_runner=command_runner,
                        timeout_seconds=command_timeout_seconds,
                    )
                    control_actions.append(rollback_result)
                    rollback_after = rollback_result.get("after")
                    rollback_after = rollback_after if isinstance(rollback_after, dict) else {}
                    rollback_paused = bool(
                        rollback_result.get("status") == "ok"
                        and rollback_result.get("action") == "pause"
                        and rollback_after.get("paused") is True
                    )
                blockers.append(
                    {
                        "code": action_status,
                        "severity": "blocking",
                        "message": "Paper trading service start did not complete successfully.",
                        "evidence": command_results,
                    }
                )

    status = (
        "started_observation_only"
        if started and observation_only
        else "started"
        if started
        else "blocked"
        if blockers
        else "ready_no_start"
    )
    return {
        "status": status,
        "checked_at": _now_iso(),
        "service_name": service_name,
        "read_only_preflight": True,
        "preflight": preflight,
        "can_resume_paper": can_resume,
        "observation_only_requested": bool(observation_only),
        "observation_only_allowed": observation_only_allowed,
        "start_requested": bool(start_service),
        "confirmation_phrase_required": CONFIRMATION_PHRASE,
        "observation_only_confirmation_phrase_required": (
            OBSERVATION_ONLY_CONFIRMATION_PHRASE
        ),
        "confirmation_present": confirmed,
        "starts_trading_service": started,
        "control_state_resumed": control_state_resumed,
        "control_state_paused": control_state_paused,
        "rollback_paused": rollback_paused,
        "submits_orders": False,
        "changes_model_routing": False,
        "live_trading_enabled": False,
        "operator_controlled": True,
        "blockers": blockers,
        "command_results": command_results,
        "control_actions": control_actions,
        "action_status": action_status,
        "safety_note": (
            "This entrypoint never bypasses the Phase 3 paper-resume preflight. "
            "Observation-only mode may start only with a verified model-bundle wait, "
            "and it persists paused paper control so no new entry can submit. "
            "Without --start-service and the explicit confirmation token it only writes a report. "
            "A failed service start is rolled back to paused paper control state."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-service", action="store_true")
    parser.add_argument("--confirm-resume-paper", default="")
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--command-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--preflight-timeout-seconds", type=float, default=90.0)
    parser.add_argument(
        "--observation-only",
        action="store_true",
        help="Start paper paused for market/exit observation only during a verified artifact wait.",
    )
    parser.add_argument(
        "--confirm-observation-only",
        default="",
        help=f"Required token for observation-only start: {OBSERVATION_ONLY_CONFIRMATION_PHRASE}",
    )
    parser.add_argument("--json-indent", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stdout-only", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    if not bool(args.start_service):
        drop_privileges_to_runtime_user_if_needed(project_root=ROOT)
    report = await build_phase3_paper_start_report(
        start_service=bool(args.start_service),
        confirm_resume_paper=str(args.confirm_resume_paper or ""),
        observation_only=bool(args.observation_only),
        confirm_observation_only=str(args.confirm_observation_only or ""),
        service_name=str(args.service_name or DEFAULT_SERVICE_NAME),
        command_timeout_seconds=max(float(args.command_timeout_seconds or 1.0), 1.0),
        preflight_timeout_seconds=max(float(args.preflight_timeout_seconds or 1.0), 1.0),
    )
    indent = None if int(args.json_indent or 0) <= 0 else int(args.json_indent)
    if not args.stdout_only:
        try:
            write_preflight_report(report, args.output_dir or Path("data/phase3_paper_resume_start_reports"), indent=indent)
        except Exception as exc:
            report["status"] = "blocked"
            report.setdefault("blockers", []).append(
                {
                    "code": "start_report_write_failed",
                    "severity": "blocking",
                    "message": safe_error_text(exc, limit=240),
                }
            )
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    if args.fail_on_blocked and report.get("status") == "blocked":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
