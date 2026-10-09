#!/usr/bin/env python3
"""Run the independent timer through the shared governed training coordinator."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.safe_output import safe_error_text  # noqa: E402
from db.session import close_db  # noqa: E402
from services.local_ai_tools_client import LocalAIToolsClient  # noqa: E402
from services.local_ai_training_contract import LOCAL_AI_TOOLS_TRAIN_RESULT_PREFIX  # noqa: E402
from services.model_training_coordinator import ModelTrainingCoordinatorMixin  # noqa: E402
from services.model_training_state import training_result_failed  # noqa: E402


class IndependentTrainingCoordinator(ModelTrainingCoordinatorMixin):
    """Timer host without a trading service, order executor or alternate policy."""

    def __init__(self) -> None:
        self.initialize_model_training(None)
        self.local_ai_tools = LocalAIToolsClient()

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default


async def run_once() -> dict[str, Any]:
    coordinator = IndependentTrainingCoordinator()
    try:
        result = await coordinator.train_local_ai_tools(force=False)
        result.setdefault("training_mode", "walk_forward")
        result.setdefault("live_routing_enabled", False)
        return result
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return {
            "trained": False,
            "reason": "error",
            "error": safe_error_text(exc, limit=500),
        }
    finally:
        try:
            await coordinator.local_ai_tools.close()
        finally:
            await close_db()


def main() -> int:
    result = asyncio.run(run_once())
    print(
        LOCAL_AI_TOOLS_TRAIN_RESULT_PREFIX
        + json.dumps(result, ensure_ascii=False, sort_keys=True)
    )
    return 2 if training_result_failed(result) else 0


if __name__ == "__main__":
    raise SystemExit(main())
