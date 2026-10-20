"""Evaluate paper strategies in an isolated worker; never submit orders."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runtime_env_bootstrap import (  # noqa: E402
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

load_runtime_env_files(project_root=ROOT)
drop_privileges_to_runtime_user_if_needed(project_root=ROOT)

from core.safe_output import safe_error_text, safe_print  # noqa: E402
from db.session import close_db  # noqa: E402
from services.model_training_state import ModelTrainingStateStore  # noqa: E402
from services.strategy_learning import StrategyLearningService  # noqa: E402
from services.strategy_replay_governance import (  # noqa: E402
    REPLAY_TIMEOUT_SECONDS,
    replay_report_path,
    summarize_replay_governance,
    write_replay_governance,
)


async def run_once() -> dict:
    state = ModelTrainingStateStore(replay_report_path().with_name("worker_state.json"))
    attempt = state.try_acquire_lease(
        scheduler_id="paper_strategy_replay", stale_after_seconds=REPLAY_TIMEOUT_SECONDS
    )
    if not attempt.acquired or attempt.lease is None:
        return {"status": "skipped", "reason": attempt.reason}
    try:
        write_replay_governance({
            "status": "running",
            "generated_at": datetime.now(UTC).isoformat(),
            "execution_scope": "paper_only",
            "live_execution_permission": False,
            "submits_orders": False,
        })
        try:
            payload = await asyncio.wait_for(
                StrategyLearningService().evaluate_paper_champion(),
                timeout=REPLAY_TIMEOUT_SECONDS,
            )
            report = summarize_replay_governance(payload)
        except Exception as exc:
            report = {
                "status": "error",
                "generated_at": datetime.now(UTC).isoformat(),
                "execution_scope": "paper_only",
                "live_execution_permission": False,
                "submits_orders": False,
                "error": safe_error_text(exc, limit=300),
                "error_type": type(exc).__name__,
            }
        write_replay_governance(report)
        return report
    finally:
        try:
            await close_db()
        finally:
            attempt.lease.release()


if __name__ == "__main__":
    result = asyncio.run(run_once())
    safe_print(json.dumps(result, ensure_ascii=False, allow_nan=False, default=str))
    raise SystemExit(1 if result.get("status") == "error" else 0)
