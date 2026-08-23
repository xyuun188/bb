#!/usr/bin/env python3
"""Run audited model training inside the online platform runtime environment."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import (  # noqa: E402
    MAX_REMOTE_OUTPUT_TEXT_LIMIT,
    connect_remote_ssh,
    run_remote_text,
)
from core.safe_output import safe_print  # noqa: E402
from services.local_ai_training_contract import (  # noqa: E402
    LOCAL_AI_TOOLS_TRAIN_RESULT_PREFIX,
)
from services.ml_training_contract import (  # noqa: E402
    LOCAL_ML_AUTO_TRAIN_RESULT_PREFIX,
)

_PERSISTED_TRAINING_RESULT_PREFIXES = {
    "ml_signal": LOCAL_ML_AUTO_TRAIN_RESULT_PREFIX,
    "local_ai_tools": LOCAL_AI_TOOLS_TRAIN_RESULT_PREFIX,
}
_PREFLIGHT_RESULT_PREFIX = "ONLINE_TRAINING_PREFLIGHT_RESULT "
_REMOTE_TRAINING_PID_PREFIX = "/tmp/bb-online-training-"


def _remote_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _remote_training_pid_file(run_token: str) -> str:
    token = str(run_token or "").strip()
    if not token or any(char not in "0123456789abcdef" for char in token.lower()):
        raise ValueError("remote training run token must be hexadecimal")
    return f"{_REMOTE_TRAINING_PID_PREFIX}{token}.pid"


def _remote_cleanup_command(*, remote_app_dir: str, run_token: str) -> str:
    """Terminate only the remote process tree owned by one training run."""

    pid_file = _remote_training_pid_file(run_token)
    expected_token = f"BB_ONLINE_TRAINING_TOKEN={run_token}"
    return (
        f"cd {_remote_quote(remote_app_dir)} && "
        f"PID_FILE={_remote_quote(pid_file)}; "
        f"EXPECTED_TOKEN={_remote_quote(expected_token)}; "
        "if [ -s \"$PID_FILE\" ]; then "
        "ROOT_PID=$(cat \"$PID_FILE\" 2>/dev/null || true); "
        "if [ -n \"$ROOT_PID\" ] && [ -r \"/proc/$ROOT_PID/environ\" ] && "
        "tr '\\0' '\\n' <\"/proc/$ROOT_PID/environ\" | /usr/bin/grep -Fxq \"$EXPECTED_TOKEN\"; then "
        "kill_tree() { "
        "for child in $(/usr/bin/pgrep -P \"$1\" 2>/dev/null || true); do kill_tree \"$child\"; done; "
        "/bin/kill -TERM \"$1\" 2>/dev/null || true; "
        "}; kill_tree \"$ROOT_PID\"; sleep 1; "
        "fi; rm -f \"$PID_FILE\"; fi"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-app-dir", default="/data/bb/app")
    parser.add_argument(
        "--target",
        choices=("ml_signal", "local_ai_tools", "all"),
        default="all",
    )
    parser.add_argument("--persist-artifact", action="store_true")
    parser.add_argument("--confirm-phase3-rebuild", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if args.persist_artifact and not args.confirm_phase3_rebuild:
        parser.error("--persist-artifact requires --confirm-phase3-rebuild")
    return args


def _target_argv(target: str, *, persist_artifact: bool) -> tuple[str, list[str]]:
    if target == "ml_signal":
        script_path = (
            "scripts/run_local_ml_auto_train.py"
            if persist_artifact
            else "scripts/train_ml_signal_model.py"
        )
        argv = [script_path, "--force"] if persist_artifact else [script_path]
    else:
        script_path = "scripts/train_local_ai_tools_models.py"
        argv = [script_path, "--training-mode", "shadow"]
    if persist_artifact and target != "ml_signal":
        argv.extend(("--persist-artifact", "--confirm-phase3-rebuild"))
    return script_path, argv


def _remote_command(
    *,
    remote_app_dir: str,
    script_path: str,
    argv: list[str],
    execution_timeout_seconds: int,
    persist_artifact: bool = True,
    run_token: str | None = None,
) -> str:
    run_token = str(run_token or secrets.token_hex(16))
    pid_file = _remote_training_pid_file(run_token)
    result_prefixes = tuple(_PERSISTED_TRAINING_RESULT_PREFIXES.values())
    remote_script = f"""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys

from scripts.runtime_env_bootstrap import load_runtime_env_files, drop_privileges_to_runtime_user_if_needed

root = Path({remote_app_dir!r})
load_runtime_env_files(project_root=root)
drop_privileges_to_runtime_user_if_needed(project_root=root)
sys.argv = {argv!r}
captured_stdout = io.StringIO()
exit_code = 0
with contextlib.redirect_stdout(captured_stdout):
    try:
        runpy.run_path({script_path!r}, run_name="__main__")
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else int(bool(exc.code))

training_output = captured_stdout.getvalue()
if exit_code != 0:
    sys.stdout.write(training_output[-10000:])
    raise SystemExit(exit_code)

expect_persisted_result = {bool(persist_artifact)!r}
if not expect_persisted_result:
    # The preflight script intentionally emits ordinary human-readable JSON
    # and does not persist an artifact.  It must not be mistaken for a failed
    # persisted training run merely because no lifecycle result prefix exists.
    sys.stdout.write(training_output)
    print({_PREFLIGHT_RESULT_PREFIX!r} + json.dumps(
        {{
            "trained": False,
            "reason": "preflight_completed",
            "artifact_persisted": False,
        }},
        ensure_ascii=False,
        sort_keys=True,
    ))
    raise SystemExit(0)

prefixes = {result_prefixes!r}
result_line = next(
    (
        line
        for line in reversed(training_output.splitlines())
        if any(line.startswith(prefix) for prefix in prefixes)
    ),
    None,
)
if result_line is None:
    sys.stdout.write(training_output[-10000:])
    raise RuntimeError("persisted training result frame missing on remote host")
result_prefix = next(prefix for prefix in prefixes if result_line.startswith(prefix))
full_frame = result_line.removeprefix(result_prefix)
full_payload = json.loads(full_frame)
if not isinstance(full_payload, dict):
    raise RuntimeError("persisted training result is not an object on remote host")
compact_payload = {{
    key: full_payload.get(key)
    for key in (
        "trained",
        "reason",
        "error",
        "artifact_version",
        "artifact_persisted",
        "model_stage",
        "challenger_version",
        "challenger_rejected",
        "champion_version",
        "champion_retained",
        "challenger_artifact_version",
        "current_artifact_version",
        "model_path",
        "training_data_sha256",
        "sample_count",
        "trainable_sample_count",
    )
    if key in full_payload
}}
compact_payload["full_result_chars"] = len(full_frame)
compact_payload["full_result_keys"] = sorted(str(key) for key in full_payload)
print(result_prefix + json.dumps(compact_payload, ensure_ascii=False, sort_keys=True))
"""
    return (
        f"cd {_remote_quote(remote_app_dir)} && "
        f"RUN_TOKEN={_remote_quote(run_token)}; export BB_ONLINE_TRAINING_TOKEN=\"$RUN_TOKEN\"; "
        f"PID_FILE={_remote_quote(pid_file)}; echo \"$$\" > \"$PID_FILE\"; "
        "cleanup_training_pid() { rm -f \"$PID_FILE\"; }; "
        "trap cleanup_training_pid EXIT; trap 'exit 130' INT; trap 'exit 143' TERM HUP; "
        "PYBIN=python3; "
        "if [ -x .venv/bin/python ]; then PYBIN=.venv/bin/python; "
        "elif [ -x venv/bin/python ]; then PYBIN=venv/bin/python; fi; "
        "timeout --signal=INT --kill-after=30s "
        f"{max(int(execution_timeout_seconds), 1)}s $PYBIN - <<'PY'\n"
        f"{remote_script}\nPY"
    )


def _cleanup_remote_training(
    ssh: Any,
    *,
    remote_app_dir: str,
    run_token: str,
) -> None:
    """Best-effort cleanup after local timeout, cancellation, or parse failure."""

    cleanup_command = _remote_cleanup_command(
        remote_app_dir=remote_app_dir,
        run_token=run_token,
    )
    try:
        run_remote_text(ssh, cleanup_command, timeout=20, check=False, max_output_chars=2000)
        return
    except BaseException:
        pass
    # The original channel may have died with the long-running command. Reconnect
    # once so a local Ctrl+C cannot leave a remote training process behind.
    replacement = None
    try:
        replacement = connect_remote_ssh(ROOT, timeout=20)
        run_remote_text(
            replacement,
            cleanup_command,
            timeout=20,
            check=False,
            max_output_chars=2000,
        )
    except BaseException:
        pass
    finally:
        if replacement is not None:
            replacement.close()


def _persisted_training_result(target: str, output: str) -> dict[str, Any]:
    prefix = _PERSISTED_TRAINING_RESULT_PREFIXES[target]
    frame = next(
        (
            line.removeprefix(prefix)
            for line in reversed(output.splitlines())
            if line.startswith(prefix)
        ),
        None,
    )
    if frame is None:
        raise RuntimeError(f"{target} persisted training result frame missing")
    try:
        payload = json.loads(frame)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{target} persisted training result is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{target} persisted training result is not an object")
    if payload.get("trained") is not True:
        reason = str(payload.get("reason") or "unknown")
        error = str(payload.get("error") or "").strip()
        detail = f": {error}" if error else ""
        raise RuntimeError(f"{target} persisted training failed ({reason}){detail}")
    return payload


def main() -> None:
    args = parse_args()
    targets = ("ml_signal", "local_ai_tools") if args.target == "all" else (args.target,)
    command_timeout = max(int(args.timeout or 1), 90)
    execution_timeout = max(command_timeout - 60, 1)
    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        for target in targets:
            run_token = secrets.token_hex(16)
            script_path, argv = _target_argv(
                target,
                persist_artifact=bool(args.persist_artifact),
            )
            try:
                output = run_remote_text(
                    ssh,
                    _remote_command(
                        remote_app_dir=args.remote_app_dir,
                        script_path=script_path,
                        argv=argv,
                        execution_timeout_seconds=execution_timeout,
                        persist_artifact=bool(args.persist_artifact),
                        run_token=run_token,
                    ),
                    timeout=command_timeout,
                    check=True,
                    max_output_chars=MAX_REMOTE_OUTPUT_TEXT_LIMIT,
                )
            except BaseException:
                _cleanup_remote_training(
                    ssh,
                    remote_app_dir=args.remote_app_dir,
                    run_token=run_token,
                )
                raise
            if args.persist_artifact:
                _persisted_training_result(target, output)
            safe_print(f"[{target}]")
            safe_print(output)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
