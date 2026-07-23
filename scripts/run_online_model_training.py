#!/usr/bin/env python3
"""Run audited model training inside the online platform runtime environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402


def _remote_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


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
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.persist_artifact and not args.confirm_phase3_rebuild:
        parser.error("--persist-artifact requires --confirm-phase3-rebuild")
    return args


def _target_argv(target: str, *, persist_artifact: bool) -> tuple[str, list[str]]:
    if target == "ml_signal":
        script_path = "scripts/train_ml_signal_model.py"
        argv = [script_path]
    else:
        script_path = "scripts/train_local_ai_tools_models.py"
        argv = [script_path, "--training-mode", "shadow"]
    if persist_artifact:
        argv.extend(("--persist-artifact", "--confirm-phase3-rebuild"))
    return script_path, argv


def _remote_command(
    *,
    remote_app_dir: str,
    script_path: str,
    argv: list[str],
) -> str:
    remote_script = f"""
from pathlib import Path
import runpy
import sys

from scripts.runtime_env_bootstrap import load_runtime_env_files, drop_privileges_to_runtime_user_if_needed

root = Path({remote_app_dir!r})
load_runtime_env_files(project_root=root)
drop_privileges_to_runtime_user_if_needed(project_root=root)
sys.argv = {argv!r}
runpy.run_path({script_path!r}, run_name="__main__")
"""
    return (
        f"cd {_remote_quote(remote_app_dir)} && "
        "PYBIN=python3; "
        "if [ -x .venv/bin/python ]; then PYBIN=.venv/bin/python; "
        "elif [ -x venv/bin/python ]; then PYBIN=venv/bin/python; fi; "
        "$PYBIN - <<'PY'\n"
        f"{remote_script}\nPY"
    )


def main() -> None:
    args = parse_args()
    targets = ("ml_signal", "local_ai_tools") if args.target == "all" else (args.target,)
    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        for target in targets:
            script_path, argv = _target_argv(
                target,
                persist_artifact=bool(args.persist_artifact),
            )
            output = run_remote_text(
                ssh,
                _remote_command(
                    remote_app_dir=args.remote_app_dir,
                    script_path=script_path,
                    argv=argv,
                ),
                timeout=max(int(args.timeout or 1), 1),
                check=True,
            )
            safe_print(f"[{target}]")
            safe_print(output)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
