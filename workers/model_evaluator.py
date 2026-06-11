"""
Model evaluator worker — periodically scores model performance.
Computes decision accuracy by tracking whether executed decisions were profitable.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import structlog

from core.safe_output import safe_error_text
from db.repositories.decision_repo import DecisionRepository
from db.session import get_session_ctx

logger = structlog.get_logger(__name__)


class ModelEvaluator:
    """Background worker that evaluates decision outcomes.

    For each executed decision, checks whether the subsequent price movement
    was in the predicted direction, and marks the decision outcome.
    """

    def __init__(self, price_provider=None) -> None:
        self._price_provider = price_provider  # Function to get current price
        self._running = False
        self._evaluation_gap = timedelta(minutes=30)  # Wait 30 min before evaluating

    async def start(self) -> None:
        self._running = True
        logger.info("model evaluator started")

        while self._running:
            try:
                await self._evaluate_pending_decisions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("evaluation error", error=safe_error_text(e))

            await asyncio.sleep(300)  # Every 5 minutes

    async def stop(self) -> None:
        self._running = False
        logger.info("model evaluator stopped")

    async def _evaluate_pending_decisions(self) -> None:
        """Find executed decisions older than evaluation_gap without outcomes, and score them."""
        async with get_session_ctx() as session:
            decision_repo = DecisionRepository(session)

            # Get recent executed decisions without outcomes
            cutoff = datetime.now(UTC) - self._evaluation_gap
            decisions = await decision_repo.find_by(
                was_executed=True,
                outcome=None,
            )

            evaluated = 0
            for decision in decisions:
                if not decision.executed_at or decision.executed_at > cutoff:
                    continue

                # Simple evaluation: if decision was "long" and price went up, it's a profit
                if decision.feature_snapshot and decision.execution_price and self._price_provider:
                    try:
                        current_price = await self._price_provider(decision.symbol)
                    except Exception as exc:
                        logger.debug(
                            "price provider failed during model evaluation",
                            symbol=decision.symbol,
                            error=safe_error_text(exc),
                        )
                        continue

                    if current_price is None:
                        continue

                    entry_price = decision.execution_price
                    if decision.action == "long":
                        pnl_pct = (current_price - entry_price) / entry_price
                        outcome = "profit" if pnl_pct > 0 else "loss" if pnl_pct < 0 else "flat"
                    elif decision.action == "short":
                        pnl_pct = (entry_price - current_price) / entry_price
                        outcome = "profit" if pnl_pct > 0 else "loss" if pnl_pct < 0 else "flat"
                    else:
                        pnl_pct = 0
                        outcome = "flat"

                    await decision_repo.mark_outcome(decision.id, outcome, pnl_pct)
                    evaluated += 1

            if evaluated > 0:
                logger.info("decisions evaluated", count=evaluated)
