"""Execution service boundary.

ExecutionService owns serialized execution entry.  The legacy order submit and
local-sync implementation still lives on TradingService for now, but callers no
longer take the lock or call the locked implementation directly.
"""

from __future__ import annotations

from typing import Any


class ExecutionService:
    def __init__(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator

    async def execute_candidate(
        self,
        symbol: str,
        model_name: str,
        decision: Any,
        assessment: Any,
        decision_db_id: int | None,
        results: dict[str, Any],
        *,
        open_positions: list[dict[str, Any]] | None = None,
    ) -> Any:
        async with self.orchestrator._execution_lock:
            return await self.orchestrator._execute_candidate_locked(
                symbol,
                model_name,
                decision,
                assessment,
                decision_db_id,
                results,
                open_positions=open_positions,
            )
