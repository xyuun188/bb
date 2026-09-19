"""Entry execution handoff helpers.

Once an entry signal is selected for execution, the analysis round watchdog must
not abandon the execution coroutine before it can write the final OKX/local
state.  The exchange submit path has its own timeout inside ExecutionService.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ExecutionService allows an entry exchange request up to 60 seconds. The
# handoff deadline must cover that complete window plus local persistence;
# cancelling earlier can leave the exchange task alive while stale-entry repair
# permits the same signal to be retried.
DEFAULT_ENTRY_EXECUTION_HANDOFF_TIMEOUT_SECONDS = 75.0
# Keep stale-entry repair later than every detached handoff. Export the value so
# maintenance cannot silently drift below the execution deadline again.
ENTRY_EXECUTION_PENDING_RECOVERY_SECONDS = 120.0
# Short aliases make the deadline explicit at call sites and keep older
# imports that used the generic name source-compatible.
ENTRY_EXECUTION_HANDOFF_TIMEOUT_SECONDS = DEFAULT_ENTRY_EXECUTION_HANDOFF_TIMEOUT_SECONDS
HANDOFF_TIMEOUT_SECONDS = DEFAULT_ENTRY_EXECUTION_HANDOFF_TIMEOUT_SECONDS


class HandoffTimeoutError(TimeoutError):
    """The handoff did not produce a safe result before its hard deadline."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        cancellation_count: int = 0,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.cancellation_count = int(cancellation_count)
        super().__init__(
            "execution handoff exceeded its hard deadline "
            f"({self.timeout_seconds:.3f}s)"
        )


def _consume_detached_task(task: asyncio.Task[Any]) -> None:
    """Consume a detached task result so late failures do not become warnings."""

    with suppress(asyncio.CancelledError, Exception):
        task.exception()


async def await_entry_execution_handoff(
    awaitable: Awaitable[Any],
    *,
    symbol: str,
    model_name: str,
    action: str,
    source: str,
    timeout_seconds: float = DEFAULT_ENTRY_EXECUTION_HANDOFF_TIMEOUT_SECONDS,
    on_outer_cancellation: Callable[[int], Awaitable[None]] | None = None,
) -> Any:
    """Wait for an entry execution to reach a terminal local state.

    The market analysis loop is protected by a hard watchdog.  If that watchdog
    fires after an entry has been handed to the execution pipeline, cancelling the
    same coroutine would leave the decision as "not executed" even though the
    order may already be in or near the OKX submit path.  Run the handoff in an
    independent task and shield it from the parent cancellation; ExecutionService
    still bounds the actual OKX call with its own timeout.
    """

    timeout = max(float(timeout_seconds), 0.001)
    task = asyncio.create_task(awaitable)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    cancellation_count = 0
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            if task.done():
                return task.result()
            if not task.done():
                task.cancel()
                task.add_done_callback(_consume_detached_task)
            raise HandoffTimeoutError(
                timeout_seconds=timeout,
                cancellation_count=cancellation_count,
            )
        try:
            result = await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
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
            if on_outer_cancellation is not None:
                await on_outer_cancellation(cancellation_count)
            logger.warning(
                "entry execution handoff is waiting for terminal result after outer cancellation",
                symbol=symbol,
                model=model_name,
                action=action,
                source=source,
                outer_cancellations=cancellation_count,
            )
        except TimeoutError as exc:
            if task.done():
                return task.result()
            task.cancel()
            task.add_done_callback(_consume_detached_task)
            logger.error(
                "entry execution handoff exceeded hard deadline",
                symbol=symbol,
                model=model_name,
                action=action,
                source=source,
                timeout_seconds=timeout,
                outer_cancellations=cancellation_count,
            )
            raise HandoffTimeoutError(
                timeout_seconds=timeout,
                cancellation_count=cancellation_count,
            ) from exc
