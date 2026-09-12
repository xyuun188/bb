from __future__ import annotations

import pytest

from core.model_training_service_lease import (
    TrainingServiceState,
    capture_service_states_command,
    parse_service_states,
    restore_services_command,
    service_states_match,
    stop_services_command,
    wrap_command_with_restore,
)

SERVICES = (
    "bb-phase3-llm-target.service",
    "bb-phase3-llm-expert.service",
    "bb-phase3-quant-api.service",
)


def test_capture_and_parse_service_states_are_complete_and_ordered() -> None:
    command = capture_service_states_command(SERVICES)
    states = parse_service_states(
        "bb-phase3-llm-target.service\tactive\n"
        "bb-phase3-llm-expert.service\tinactive\n"
        "bb-phase3-quant-api.service\tfailed\n",
        SERVICES,
    )

    assert command.index(SERVICES[0]) < command.index(SERVICES[1])
    assert states[SERVICES[0]].active is True
    assert states[SERVICES[1]].active is False
    assert states[SERVICES[2]].active is False


def test_capture_rejects_missing_duplicate_and_unsafe_service_rows() -> None:
    with pytest.raises(RuntimeError, match="could not capture"):
        parse_service_states(f"{SERVICES[0]}\tactive\n", SERVICES)
    with pytest.raises(RuntimeError, match="duplicate"):
        parse_service_states(
            f"{SERVICES[0]}\tactive\n{SERVICES[0]}\tinactive\n",
            (SERVICES[0],),
        )
    with pytest.raises(ValueError, match="unsafe"):
        capture_service_states_command(("bad.service; reboot",))


def test_stop_and_restore_commands_verify_state_without_changing_enablement() -> None:
    states = {
        SERVICES[0]: TrainingServiceState(SERVICES[0], True),
        SERVICES[1]: TrainingServiceState(SERVICES[1], False),
    }

    stop = stop_services_command(states)
    restore = restore_services_command(states)

    assert "training_service_still_active" in stop
    assert f"systemctl start {SERVICES[0]}" in restore
    assert f"systemctl stop {SERVICES[1]}" in restore
    assert "training_service_restore_failed" in restore
    assert "systemctl enable" not in restore
    assert "systemctl disable" not in restore


def test_training_wrapper_restores_on_every_exit_and_preserves_body_status() -> None:
    wrapped = wrap_command_with_restore("run-training", "restore-services")

    assert "trap cleanup EXIT" in wrapped
    assert wrapped.index("trap cleanup EXIT") < wrapped.index("run-training")
    assert wrapped.index("restore-services") < wrapped.index('exit "$original_rc"')
    assert "exit 97" in wrapped


def test_state_match_requires_exact_prior_active_state() -> None:
    expected = {SERVICES[0]: TrainingServiceState(SERVICES[0], True)}
    assert service_states_match(expected, dict(expected)) is True
    assert service_states_match(
        expected,
        {SERVICES[0]: TrainingServiceState(SERVICES[0], False)},
    ) is False
