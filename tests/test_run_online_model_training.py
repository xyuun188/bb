import json
import re

import pytest

from scripts.run_online_model_training import (
    MAX_REMOTE_OUTPUT_TEXT_LIMIT,
    _persisted_training_result,
    _remote_command,
    _target_argv,
)
from services.ml_training_contract import LOCAL_ML_AUTO_TRAIN_RESULT_PREFIX


def test_local_ai_tools_online_training_has_no_stage_authorization_input() -> None:
    script_path, argv = _target_argv("local_ai_tools", persist_artifact=True)

    assert script_path == "scripts/train_local_ai_tools_models.py"
    assert argv == [
        script_path,
        "--training-mode",
        "shadow",
        "--persist-artifact",
        "--confirm-phase3-rebuild",
    ]
    assert "--model-stage" not in argv


def test_persisted_ml_training_uses_full_candidate_lifecycle() -> None:
    script_path, argv = _target_argv("ml_signal", persist_artifact=True)

    assert script_path == "scripts/run_local_ml_auto_train.py"
    assert argv == [script_path, "--force"]


def test_online_training_command_terminates_before_ssh_timeout() -> None:
    command = _remote_command(
        remote_app_dir="/data/bb/app",
        script_path="scripts/train_ml_signal_model.py",
        argv=["scripts/train_ml_signal_model.py", "--persist-artifact"],
        execution_timeout_seconds=7140,
    )

    assert "timeout --signal=INT --kill-after=30s 7140s $PYBIN -" in command
    assert "contextlib.redirect_stdout(captured_stdout)" in command
    assert 'compact_payload["full_result_chars"]' in command


def test_online_training_requires_successful_structured_business_result() -> None:
    output = (
        "training log\n"
        + LOCAL_ML_AUTO_TRAIN_RESULT_PREFIX
        + json.dumps({"trained": True, "reason": "trained_challenger_rejected"})
    )

    assert _persisted_training_result("ml_signal", output)["trained"] is True


def test_online_training_transport_allows_full_structured_result() -> None:
    assert MAX_REMOTE_OUTPUT_TEXT_LIMIT >= 200_000


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("training log only", "result frame missing"),
        (
            LOCAL_ML_AUTO_TRAIN_RESULT_PREFIX + "not-json",
            "result is invalid JSON",
        ),
        (
            LOCAL_ML_AUTO_TRAIN_RESULT_PREFIX
            + json.dumps({"trained": False, "reason": "error", "error": "fit failed"}),
            "persisted training failed (error): fit failed",
        ),
    ],
)
def test_online_training_rejects_missing_malformed_or_failed_result(
    output: str,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=re.escape(message)):
        _persisted_training_result("ml_signal", output)
