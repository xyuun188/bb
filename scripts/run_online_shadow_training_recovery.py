#!/usr/bin/env python3
"""Rebuild online historical shadow facts and run one formal training cycle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import (  # noqa: E402
    connect_remote_ssh,
    exec_remote_command,
    run_remote_text,
)
from core.safe_output import safe_print  # noqa: E402
from services.local_ai_training_contract import LOCAL_AI_TOOLS_TRAIN_RESULT_PREFIX  # noqa: E402


def _quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _tail(value: str, limit: int = 6000) -> str:
    return str(value or "")[-limit:]


def _extract_training_result(output: str) -> dict[str, object] | None:
    for line in reversed(str(output or "").splitlines()):
        if not line.startswith(LOCAL_AI_TOOLS_TRAIN_RESULT_PREFIX):
            continue
        try:
            payload = json.loads(line.removeprefix(LOCAL_AI_TOOLS_TRAIN_RESULT_PREFIX))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-app-dir", default="/data/bb/app")
    parser.add_argument("--small-decisions", type=int, default=50)
    parser.add_argument("--rebuild-timeout", type=int, default=1800)
    parser.add_argument("--training-timeout", type=int, default=1800)
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--counts-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remote_root = _quote(args.remote_app_dir)
    python_bin = (
        "PYBIN=python3; "
        "if [ -x .venv/bin/python ]; then PYBIN=.venv/bin/python; "
        "elif [ -x venv/bin/python ]; then PYBIN=venv/bin/python; fi; "
    )
    result: dict[str, object] = {}
    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        result["trading_service"] = _tail(
            run_remote_text(
                ssh,
                (
                    "systemctl stop bb-paper-trading.service 2>/dev/null || true; "
                    "state=$(systemctl is-active bb-paper-trading.service 2>/dev/null || true); "
                    'printf "%s\n" "$state"; '
                    '[ "$state" = inactive ] || [ "$state" = failed ]'
                ),
                timeout=60,
                check=True,
            ),
            500,
        ).strip()
        if not args.training_only and not args.counts_only:
            small_command = (
                f"cd {remote_root} && {python_bin}"
                f"PYTHONPATH=. timeout {max(args.rebuild_timeout, 60)} "
                "$PYBIN scripts/rebuild_historical_shadow_samples.py "
                f"--max-decisions {max(args.small_decisions, 1)}"
            )
            result["small_rebuild"] = _tail(
                run_remote_text(
                    ssh,
                    small_command,
                    timeout=max(args.rebuild_timeout, 60) + 60,
                    check=True,
                )
            )
            full_command = (
                f"cd {remote_root} && {python_bin}"
                f"PYTHONPATH=. timeout {max(args.rebuild_timeout, 60)} "
                "$PYBIN scripts/rebuild_historical_shadow_samples.py"
            )
            result["full_rebuild"] = _tail(
                run_remote_text(
                    ssh,
                    full_command,
                    timeout=max(args.rebuild_timeout, 60) + 60,
                    check=True,
                )
            )
        if not args.counts_only:
            training_command = (
                f"cd {remote_root} && {python_bin}"
                "for i in $(seq 1 30); do "
                "pg_isready -q && break; sleep 2; done; pg_isready -q; "
                "attempt=1; rc=1; while [ $attempt -le 3 ]; do "
                f"PYTHONPATH=. timeout {max(args.training_timeout, 60)} "
                "$PYBIN scripts/train_local_ai_tools_models.py "
                "--training-mode formal --persist-artifact --confirm-phase3-rebuild "
                "&& { rc=0; break; }; rc=$?; attempt=$((attempt + 1)); sleep 10; "
                "done; exit $rc"
            )
            training_result = exec_remote_command(
                ssh,
                training_command,
                timeout=max(args.training_timeout, 60) + 60,
                max_output_chars=100_000,
            )
            result["formal_training"] = {
                "status": training_result.status,
                "stdout": _tail(training_result.stdout, 12000),
                "stderr": _tail(training_result.stderr, 12000),
            }
            training_payload = _extract_training_result(training_result.stdout)
            if training_payload is not None:
                result["formal_training_result"] = training_payload
                reconcile_payload = json.dumps(training_payload, ensure_ascii=False, separators=(",", ":"))
                reconcile_command = (
                    f"cd {remote_root} && {python_bin}"
                    "PYTHONPATH=. $PYBIN - <<'PY'\n"
                    "import json\n"
                    "from datetime import UTC, datetime, timedelta\n"
                    "from pathlib import Path\n"
                    "from services.model_training_state import LOCAL_AI_TOOL_MODEL_IDS, ModelTrainingStateStore\n"
                    f"payload = json.loads({reconcile_payload!r})\n"
                    "store = ModelTrainingStateStore(Path('/data/bb/app/data/model_training_scheduler_state.json'))\n"
                    "store.record_external_result(scheduler_id='local_ai_tools_auto_train', model_ids=LOCAL_AI_TOOL_MODEL_IDS, run_id=str(payload.get('artifact_version') or payload.get('challenger_artifact_version') or 'external-training'), result=payload, next_check_at=datetime.now(UTC) + timedelta(hours=6))\n"
                    "print('training-state-reconciled')\n"
                    "PY"
                )
                result["training_state_reconciliation"] = _tail(
                    run_remote_text(ssh, reconcile_command, timeout=90, check=True),
                    1000,
                )
            if training_result.status != 0:
                safe_print(json.dumps(result, ensure_ascii=False, sort_keys=True))
                raise SystemExit(training_result.status)
        count_command = (
            f"cd {remote_root} && {python_bin}PYTHONPATH=. $PYBIN - <<'PY'\n"  # noqa: S608
            """
import asyncio
import json
from pathlib import Path

from sqlalchemy import text

from scripts.runtime_env_bootstrap import (
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

root = Path("/data/bb/app")
load_runtime_env_files(project_root=root)
drop_privileges_to_runtime_user_if_needed(project_root=root)

from db.session import close_db, get_session_ctx, init_db


async def main():
    await init_db()
    try:
        async with get_session_ctx() as session:
            values = {
                "shadow_backtests": int(
                    (await session.execute(text("SELECT count(*) FROM shadow_backtests"))).scalar_one()
                ),
                "training_tables": [
                    str(row[0])
                    for row in (
                        await session.execute(
                            text(
                                "SELECT tablename FROM pg_tables "
                                "WHERE schemaname = current_schema() "
                                "AND tablename LIKE '%training%' ORDER BY tablename"
                            )
                        )
                    ).all()
                ],
            }
            print(json.dumps(values, sort_keys=True))
    finally:
        await close_db()


asyncio.run(main())
PY
"""
        )
        result["counts"] = _tail(
            run_remote_text(ssh, count_command, timeout=180, check=True),
            2000,
        ).strip()
    finally:
        ssh.close()
    safe_print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
