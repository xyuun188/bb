#!/usr/bin/env python3
"""Run one independent Local AI Tools shadow-training check.

This scheduler is deliberately separate from the trading process.  It always
publishes a heartbeat, records an OKX training-gate block as a healthy skipped
run, and only invokes the shadow trainer after the authoritative gate allows
the refresh.  It never enables live routing or submits orders.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import runpy
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from core.safe_output import safe_error_text  # noqa: E402
from db.session import close_db  # noqa: E402
from services.local_ai_training_contract import (  # noqa: E402
    LOCAL_AI_TOOLS_TRAIN_RESULT_PREFIX,
)
from services.model_training_state import (  # noqa: E402
    LOCAL_AI_TOOL_MODEL_IDS,
    ModelTrainingStateStore,
)
from services.okx_training_gate import okx_training_refresh_gate  # noqa: E402
from services.trading_params import DEFAULT_TRADING_PARAMS  # noqa: E402

SCHEDULER_ID = "local_ai_tools_auto_train"
CHECK_INTERVAL_SECONDS = float(
    DEFAULT_TRADING_PARAMS.local_ml_training.auto_train_check_interval_seconds
)
TRAINING_TIMEOUT_SECONDS = 2 * 60 * 60
logger = logging.getLogger(__name__)
STATE_STORE = ModelTrainingStateStore(
    settings.data_dir / "model_training_scheduler_state.json"
)


def _result_from_output(output: str) -> dict[str, Any]:
    """Extract the trainer's single structured result frame."""

    for line in reversed(str(output or "").splitlines()):
        if not line.startswith(LOCAL_AI_TOOLS_TRAIN_RESULT_PREFIX):
            continue
        try:
            payload = json.loads(line.removeprefix(LOCAL_AI_TOOLS_TRAIN_RESULT_PREFIX))
        except json.JSONDecodeError as exc:
            return {
                "trained": False,
                "reason": "invalid_training_response",
                "error": safe_error_text(exc, limit=180),
            }
        return payload if isinstance(payload, dict) else {
            "trained": False,
            "reason": "invalid_training_response",
        }
    return {
        "trained": False,
        "reason": "error",
        "error": "local AI tools trainer did not return a structured result",
    }


def _run_shadow_trainer() -> dict[str, Any]:
    """Invoke the existing shadow trainer without spawning a second service."""

    captured = io.StringIO()
    previous_argv = sys.argv
    sys.argv = ["scripts/train_local_ai_tools_models.py", "--training-mode", "shadow"]
    try:
        with contextlib.redirect_stdout(captured):
            try:
                runpy.run_path(
                    str(ROOT / "scripts" / "train_local_ai_tools_models.py"),
                    run_name="__main__",
                )
            except SystemExit as exc:
                if isinstance(exc.code, int) and exc.code == 0:
                    pass
                elif exc.code not in (None, 0):
                    output = captured.getvalue()
                    parsed = _result_from_output(output)
                    if parsed.get("reason") == "error":
                        parsed["error"] = parsed.get("error") or (
                            f"trainer exited with code {exc.code}"
                        )
                    return parsed
    except BaseException as exc:  # pragma: no cover - defensive process boundary
        return {
            "trained": False,
            "reason": "error",
            "error": safe_error_text(exc, limit=500),
        }
    finally:
        sys.argv = previous_argv
    return _result_from_output(captured.getvalue())


async def run_once() -> dict[str, Any]:
    run_id = uuid.uuid4().hex
    now = datetime.now(UTC)
    try:
        STATE_STORE.heartbeat(
            scheduler_id=SCHEDULER_ID,
            model_ids=LOCAL_AI_TOOL_MODEL_IDS,
            interval_seconds=CHECK_INTERVAL_SECONDS,
        )
        STATE_STORE.record_check(
            scheduler_id=SCHEDULER_ID,
            model_ids=LOCAL_AI_TOOL_MODEL_IDS,
            run_id=run_id,
            force=False,
        )
        gate = okx_training_refresh_gate()
        if not bool(gate.get("allowed")):
            result = {
                "trained": False,
                "reason": "okx_training_gate_blocked",
                "training_gate": gate,
                "training_mode": "shadow",
                "live_routing_enabled": False,
            }
        else:
            STATE_STORE.start_run(
                scheduler_id=SCHEDULER_ID,
                model_ids=LOCAL_AI_TOOL_MODEL_IDS,
                run_id=run_id,
                trigger_reason="independent_shadow_scheduler",
                timeout_seconds=TRAINING_TIMEOUT_SECONDS,
            )
            result = await asyncio.to_thread(_run_shadow_trainer)
            result.setdefault("training_mode", "shadow")
            result.setdefault("live_routing_enabled", False)
        STATE_STORE.finish_check(
            scheduler_id=SCHEDULER_ID,
            model_ids=LOCAL_AI_TOOL_MODEL_IDS,
            run_id=run_id,
            result=result,
            next_check_at=now + timedelta(seconds=CHECK_INTERVAL_SECONDS),
        )
        return result
    except asyncio.CancelledError:
        STATE_STORE.record_timeout(
            scheduler_id=SCHEDULER_ID,
            model_ids=LOCAL_AI_TOOL_MODEL_IDS,
            run_id=run_id,
            error="training_cancelled",
            next_check_at=now + timedelta(seconds=300),
        )
        raise
    except Exception as exc:
        result = {
            "trained": False,
            "reason": "error",
            "error": safe_error_text(exc, limit=500),
        }
        try:
            STATE_STORE.record_exception(
                scheduler_id=SCHEDULER_ID,
                model_ids=LOCAL_AI_TOOL_MODEL_IDS,
                run_id=run_id,
                error=str(result["error"]),
                next_check_at=now + timedelta(seconds=300),
            )
        except Exception:
            logger.debug("failed to persist local AI tools training exception", exc_info=True)
        return result
    finally:
        await close_db()


def main() -> int:
    result = asyncio.run(run_once())
    print(
        LOCAL_AI_TOOLS_TRAIN_RESULT_PREFIX
        + json.dumps(result, ensure_ascii=False, sort_keys=True)
    )
    return 2 if str(result.get("reason") or "").lower() in {"error", "invalid_training_response"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
