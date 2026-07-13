"""
Competition service — manages the multi-model competition in paper trading.
Ranks models by fee-after profitability and downside-adjusted return quality.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import structlog

from core.safe_output import safe_error_text
from core.trading_mode import mode_manager
from db.repositories.account_repo import AccountRepository
from db.repositories.decision_repo import DecisionRepository
from db.repositories.risk_repo import RiskRepository
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx
from services.return_objective import mean_confidence_lower_bound, profit_factor

logger = structlog.get_logger(__name__)


class CompetitionService:
    """Evaluates and ranks AI models based on paper trading performance.

    Win rate and decision accuracy are reported as diagnostics only. They do
    not affect model ranking or selection.
    """

    def __init__(self) -> None:
        self._evaluation_interval = 3600  # Every hour
        self._last_evaluation: datetime | None = None
        self._rankings: list[dict] = []
        self._pnl_history: dict[str, list[float]] = {}  # model -> [daily pnl]
        self._active_model_names: set[str] | None = None

    async def evaluate_all_models(self, force: bool = False) -> list[dict]:
        """Evaluate all models and return rankings."""
        now = datetime.now(UTC)
        if not force and (
            self._last_evaluation
            and (now - self._last_evaluation).total_seconds() < self._evaluation_interval
        ):
            return self._rankings

        rankings: list[dict[str, Any]] = []
        try:
            async with get_session_ctx() as session:
                account_repo = AccountRepository(session)
                trade_repo = TradeRepository(session)
                decision_repo = DecisionRepository(session)
                risk_repo = RiskRepository(session)

                accounts = await account_repo.get_all_accounts()

                # Filter to only currently active models
                if self._active_model_names is not None:
                    accounts = [a for a in accounts if a.model_name in self._active_model_names]

                for account in accounts:
                    model_name = account.model_name

                    # Get trade statistics
                    # Basic metrics — include unrealized PnL from open positions
                    unrealized = getattr(account, "unrealized_pnl", 0.0) or 0.0
                    total_pnl = (account.realized_pnl or 0.0) + unrealized
                    pnl_pct = (
                        total_pnl / account.initial_balance if account.initial_balance > 0 else 0.0
                    )
                    win_rate = account.win_rate or 0.0
                    total_trades = account.total_trades or 0

                    # Decision accuracy (decisions that led to profitable trades)
                    decision_accuracy = await decision_repo.get_decision_accuracy(model_name)

                    # Calculate Sharpe ratio from PnL history
                    sharpe = await self._calculate_sharpe(trade_repo, decision_repo, model_name)

                    # Max drawdown approximation
                    max_dd = await self._calculate_max_drawdown(
                        trade_repo, model_name, account.initial_balance
                    )

                    closed_positions = await trade_repo.get_position_records(
                        model_name=model_name,
                        is_open=False,
                        limit=500,
                    )
                    fee_after_values = [
                        float(getattr(row, "realized_pnl", 0.0) or 0.0)
                        - max(float(getattr(row, "entry_fee", 0.0) or 0.0), 0.0)
                        - max(float(getattr(row, "close_fee", 0.0) or 0.0), 0.0)
                        + float(getattr(row, "funding_fee", 0.0) or 0.0)
                        for row in closed_positions
                        if all(
                            getattr(row, field, None) is not None
                            for field in ("entry_fee", "close_fee", "funding_fee")
                        )
                    ]
                    account_base = float(account.initial_balance or 0.0)
                    fee_after_returns_pct = [
                        value / account_base * 100.0
                        for value in fee_after_values
                        if account_base > 0.0
                    ]
                    return_lcb_pct = mean_confidence_lower_bound(fee_after_returns_pct)
                    fee_after_pnl_pct = (
                        sum(fee_after_values) / account_base if account_base > 0.0 else None
                    )
                    fee_after_profit_factor = profit_factor(fee_after_values)

                    rankings.append(
                        {
                            "model_name": model_name,
                            "total_pnl": round(total_pnl, 2),
                            "pnl_pct": round(pnl_pct * 100, 2),
                            "sharpe_ratio": round(sharpe, 2),
                            "max_drawdown": round(max_dd * 100, 2),
                            "fee_after_realized_pnl_pct": (
                                round(fee_after_pnl_pct * 100.0, 6)
                                if fee_after_pnl_pct is not None
                                else None
                            ),
                            "return_lcb_pct": (
                                round(return_lcb_pct, 8)
                                if return_lcb_pct is not None
                                else None
                            ),
                            "profit_factor": (
                                round(fee_after_profit_factor, 6)
                                if fee_after_profit_factor is not None
                                else None
                            ),
                            "cost_complete_sample_count": len(fee_after_values),
                            "production_evidence_eligible": bool(fee_after_values),
                            "win_rate": round(win_rate * 100, 2),
                            "total_trades": total_trades,
                            "decision_accuracy": round(decision_accuracy * 100, 2),
                            "ranking_objective": "fee_after_return_lcb_lexicographic",
                        }
                    )

                # Lexicographic ordering avoids policy-changing fixed blend weights.
                # Missing cost-complete evidence sorts last instead of receiving a fallback score.
                rankings.sort(
                    key=lambda row: (
                        float(row["return_lcb_pct"])
                        if row.get("return_lcb_pct") is not None
                        else float("-inf"),
                        float(row["fee_after_realized_pnl_pct"])
                        if row.get("fee_after_realized_pnl_pct") is not None
                        else float("-inf"),
                        float(row["profit_factor"])
                        if row.get("profit_factor") is not None
                        else float("-inf"),
                        -float(row["max_drawdown"]),
                    ),
                    reverse=True,
                )
                for i, r in enumerate(rankings):
                    r["rank"] = i + 1

                # Save snapshots to DB
                for r in rankings:
                    await risk_repo.save_performance_snapshot(
                        {
                            "model_name": r["model_name"],
                            "total_pnl": r["total_pnl"],
                            "pnl_pct": (
                                float(r["fee_after_realized_pnl_pct"]) / 100
                                if r.get("fee_after_realized_pnl_pct") is not None
                                else 0.0
                            ),
                            "sharpe_ratio": r["sharpe_ratio"],
                            "max_drawdown": float(r["max_drawdown"]) / 100,
                            "win_rate": float(r["win_rate"]) / 100,
                            "total_trades": r["total_trades"],
                            "decision_accuracy": float(r["decision_accuracy"]) / 100,
                            "rank": r["rank"],
                        }
                    )

        except Exception as exc:
            logger.error("model evaluation failed", error=safe_error_text(exc))
            return self._rankings  # Return last known rankings

        self._rankings = rankings
        self._last_evaluation = now
        logger.info("models evaluated", rankings=[r["model_name"] for r in rankings])

        return rankings

    async def select_best_model(self) -> str | None:
        """Return the name of the best-performing model."""
        rankings = await self.evaluate_all_models()
        if rankings:
            return rankings[0]["model_name"]
        return None

    async def auto_promote_best_model(self) -> str | None:
        """Return the best model recommendation without mutating live routing.

        Phase 3 keeps model competition, shadow evaluation, and live routing
        separated.  Legacy callers may still invoke this method name, but live
        promotion must be performed by an explicit operator-controlled path
        after the promotion policy gates pass.
        """
        best = await self.select_best_model()
        if best and best != mode_manager.live_model_name:
            logger.info(
                "best model promotion recommendation recorded without live switch",
                model=best,
                current_live_model=mode_manager.live_model_name,
                live_mutation=False,
                policy="phase3_observe_only",
            )
        return best

    async def _calculate_sharpe(
        self, trade_repo: TradeRepository, decision_repo: DecisionRepository, model_name: str
    ) -> float:
        """Estimate Sharpe ratio from daily PnL.

        Sharpe = mean(daily_returns) / std(daily_returns) * sqrt(365)
        """
        try:
            decisions = await decision_repo.get_recent_decisions(model_name, limit=500)
            executed = [d for d in decisions if d.was_executed and d.outcome_pnl_pct is not None]

            if len(executed) < 5:
                return 0.0

            returns = [d.outcome_pnl_pct for d in executed if d.outcome_pnl_pct]
            if not returns:
                return 0.0

            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
            std_ret = math.sqrt(variance)

            if std_ret == 0:
                return 0.0

            # Assume decisions are roughly daily; annualize
            return (mean_ret / std_ret) * math.sqrt(365)

        except Exception as exc:
            logger.warning(
                "model sharpe calculation failed",
                model_name=model_name,
                error=safe_error_text(exc),
            )
            return 0.0

    async def _calculate_max_drawdown(
        self, trade_repo: TradeRepository, model_name: str, initial_balance: float
    ) -> float:
        """Estimate max drawdown from trade history."""
        try:
            orders = await trade_repo.get_recent_orders(model_name, limit=1000)
            if not orders:
                return 0.0

            # Build equity curve from orders
            equity = initial_balance
            peak = equity
            max_dd = 0.0

            for order in reversed(orders):
                if order.status != "filled":
                    continue
                # Approximate: each filled order's PnL
                # (More accurate would require actual position PnL from positions table)
                if order.side == "buy":
                    equity -= order.quantity * (order.price or 0)
                else:
                    equity += order.quantity * (order.price or 0)

                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

            return max_dd

        except Exception as exc:
            logger.warning(
                "model max drawdown calculation failed",
                model_name=model_name,
                error=safe_error_text(exc),
            )
            return 0.0

    def set_active_models(self, names: list[str]) -> None:
        """Set the list of currently active model names for filtering."""
        new_set = set(names)
        if self._active_model_names != new_set:
            self._active_model_names = new_set
            self._last_evaluation = None  # force re-evaluation
            self._rankings = []  # clear stale cache

    def get_rankings(self) -> list[dict]:
        if self._active_model_names is not None:
            return [r for r in self._rankings if r["model_name"] in self._active_model_names]
        return self._rankings

    async def get_rankings_live(self) -> list[dict]:
        """Get rankings, auto-refreshing if older than 5 minutes."""
        if (
            self._last_evaluation is None
            or (datetime.now(UTC) - self._last_evaluation).total_seconds() > 300
        ):
            await self.evaluate_all_models(force=True)
        return self.get_rankings()

    def get_model_rank(self, model_name: str) -> int | None:
        for r in self._rankings:
            if r["model_name"] == model_name:
                return r["rank"]
        return None
