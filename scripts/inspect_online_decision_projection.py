#!/usr/bin/env python3
"""Read-only check of full and compact AI-decision projections online."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

REMOTE_APP_DIR = "/data/bb/app"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-id", action="append", type=int, required=True)
    args = parser.parse_args()
    script = f"""
import asyncio, json
from pathlib import Path
from scripts.runtime_env_bootstrap import load_runtime_env_files, drop_privileges_to_runtime_user_if_needed
root = Path({REMOTE_APP_DIR!r})
load_runtime_env_files(project_root=root)
drop_privileges_to_runtime_user_if_needed(project_root=root)
from db.session import get_read_session_ctx
from models.decision import AIDecision
async def main():
    result = []
    async with get_read_session_ctx(statement_timeout_ms=30000) as session:
        for decision_id in {sorted(set(args.decision_id))!r}:
            row = await session.get(AIDecision, decision_id)
            if row is None:
                result.append({{"decision_id": decision_id, "found": False}})
                continue
            raw = row.raw_llm_response if isinstance(row.raw_llm_response, dict) else {{}}
            learning = row.decision_learning_snapshot if isinstance(row.decision_learning_snapshot, dict) else {{}}
            result.append({{
                "decision_id": decision_id,
                "found": True,
                "raw_keys": sorted(str(key) for key in raw),
                "raw_retention_marker": raw.get("_retention"),
                "raw_decision_authority": (raw.get("normal_paper_trade") or {{}}).get("decision_authority"),
                "raw_trade_version": (raw.get("normal_paper_trade") or {{}}).get("version"),
                "learning_keys": sorted(str(key) for key in learning),
                "learning_decision_authority": (learning.get("normal_paper_trade") or {{}}).get("decision_authority"),
                "learning_trade_version": (learning.get("normal_paper_trade") or {{}}).get("version"),
                "learning_has_execution_result": isinstance(learning.get("execution_result"), dict),
            }})
    print(json.dumps(result, ensure_ascii=False, default=str))
asyncio.run(main())
"""
    command = (
        f"cd {shlex.quote(REMOTE_APP_DIR)} && runuser -u bb -- /bin/bash -lc "
        + shlex.quote(".venv/bin/python - <<'PY'\n" + script + "\nPY")
    )
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        safe_print(run_remote_text(ssh, command, timeout=180, max_output_chars=30_000))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
