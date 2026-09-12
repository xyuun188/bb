"""Shell contracts for temporarily leasing model-host services during training.

Training may stop inference services on the single-GPU host, but it must not
change their enablement policy or leave them stopped after any success,
failure, or timeout.  This module is pure and side-effect free so its shell
fragments can be reviewed and tested before an SSH connection is opened.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

SERVICE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]*\.service$")


@dataclass(frozen=True, slots=True)
class TrainingServiceState:
    name: str
    active: bool


def _service_names(services: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(str(item or "").strip() for item in services))
    if not names:
        raise ValueError("training service lease requires at least one service")
    invalid = [name for name in names if not SERVICE_NAME_PATTERN.fullmatch(name)]
    if invalid:
        raise ValueError("unsafe training service name: " + ", ".join(invalid))
    return names


def capture_service_states_command(services: tuple[str, ...] | list[str]) -> str:
    """Render a read-only command that records the active state of each service."""

    names = _service_names(services)
    rows = []
    for name in names:
        quoted = shlex.quote(name)
        rows.append(
            f"state=$(systemctl is-active {quoted} 2>/dev/null || true); "
            f"printf '%s\\t%s\\n' {quoted} \"$state\""
        )
    return "set +e; " + "; ".join(rows)


def parse_service_states(
    raw: str,
    services: tuple[str, ...] | list[str],
) -> dict[str, TrainingServiceState]:
    """Parse one complete capture and reject missing or duplicated rows."""

    expected = _service_names(services)
    parsed: dict[str, TrainingServiceState] = {}
    for line in str(raw or "").splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        name, active_state = (part.strip() for part in parts)
        if name not in expected:
            continue
        if name in parsed:
            raise RuntimeError(f"duplicate remote service state: {name}")
        parsed[name] = TrainingServiceState(name=name, active=active_state == "active")
    missing = [name for name in expected if name not in parsed]
    if missing:
        raise RuntimeError("could not capture remote service state: " + ", ".join(missing))
    return parsed


def stop_services_command(states: dict[str, TrainingServiceState]) -> str:
    """Stop every leased service and prove none remains active before training."""

    names = _service_names(list(states))
    actions = [
        f"sudo -n systemctl stop {shlex.quote(name)} || true"
        for name in names
    ]
    checks = [
        (
            f"if systemctl is-active --quiet {shlex.quote(name)}; then "
            f"echo {shlex.quote('training_service_still_active:' + name)} >&2; exit 41; fi"
        )
        for name in names
    ]
    return "; ".join([*actions, *checks])


def restore_services_command(states: dict[str, TrainingServiceState]) -> str:
    """Restore only the prior active/inactive state and verify the result.

    Enablement is intentionally untouched.  A training lease is a temporary
    runtime operation and must not rewrite the host's boot policy.
    """

    names = _service_names(list(states))
    actions = ["restore_ok=1"]
    for name in names:
        quoted = shlex.quote(name)
        if states[name].active:
            actions.extend(
                (
                    f"sudo -n systemctl start {quoted} || restore_ok=0",
                    f"systemctl is-active --quiet {quoted} || restore_ok=0",
                )
            )
        else:
            actions.extend(
                (
                    f"sudo -n systemctl stop {quoted} || true",
                    f"if systemctl is-active --quiet {quoted}; then restore_ok=0; fi",
                )
            )
    actions.extend(
        (
            'if [ "$restore_ok" -ne 1 ]; then echo training_service_restore_failed >&2; fi',
            'test "$restore_ok" -eq 1',
        )
    )
    return "; ".join(actions)


def wrap_command_with_restore(body: str, restore_command: str) -> str:
    """Install an EXIT trap that preserves the body status after restoration."""

    if not str(body or "").strip() or not str(restore_command or "").strip():
        raise ValueError("training body and restore command are required")
    return (
        "set -euo pipefail\n"
        "cleanup() {\n"
        "  original_rc=$?\n"
        "  trap - EXIT\n"
        "  set +e\n"
        f"  {restore_command}\n"
        "  restore_rc=$?\n"
        "  set -e\n"
        "  if [ \"$restore_rc\" -ne 0 ]; then exit 97; fi\n"
        "  exit \"$original_rc\"\n"
        "}\n"
        "trap cleanup EXIT\n"
        f"{body.strip()}\n"
    )


def service_states_match(
    expected: dict[str, TrainingServiceState],
    actual: dict[str, TrainingServiceState],
) -> bool:
    return expected == actual
