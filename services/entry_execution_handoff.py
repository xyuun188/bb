"""Entry execution handoff helpers.

Once an entry signal is selected for execution, the analysis round watchdog must
not abandon the execution coroutine before it can write the final OKX/local
state.  The exchange submit path has its own timeout inside ExecutionService.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


async def await_entry_execution_handoff(
    awaitable: Awaitable[Any],
    *,
    symbol: str,
    model_name: str,
    action: str,
    source: str,
) -> Any:
    """Wait for an entry execution to reach a terminal local state.

    The market analysis loop is protected by a hard watchdog.  If that watchdog
    fires after an entry has been handed to the execution pipeline, cancelling the
    same coroutine would leave the decision as "not executed" even though the
    order may already be in or near the OKX submit path.  Run the handoff in an
    independent task and shield it from the parent cancellation; ExecutionService
    still bounds the actual OKX call with its own timeout.
    """

    task = asyncio.create_task(awaitable)
    cancellation_count = 0
    while True:
        try:
            result = await asyncio.shield(task)
            if cancellation_count:
                logger.info(
                    "entry execution completed after outer analysis cancellation",
                    symbol=symbol,
                    model=model_name,
                    action=action,
                    source=source,
                    outer_cancellations=cancellation_count,
                )
            return result
        except asyncio.CancelledError:
            if task.done():
                return task.result()
            cancellation_count += 1
            logger.warning(
                "entry execution handoff is waiting for terminal result after outer cancellation",
                symbol=symbol,
                model=model_name,
                action=action,
                source=source,
                outer_cancellations=cancellation_count,
            )
