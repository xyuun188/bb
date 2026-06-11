"""
Paper trading executor — simulates order matching against virtual accounts.
Each AI model gets its own isolated virtual account for fair competition.

Fills are simulated at the current market price with a configurable fee.
No real orders are sent to the exchange.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog

from ai_brain.base_model import Action, DecisionOutput
from config.settings import settings
from executor.base_executor import AbstractExecutor, ExecutionResult, OrderStatus

logger = structlog.get_logger(__name__)

PAPER_FEE_RATE = 0.001  # 0.1% trading fee
PAPER_SLIPPAGE = 0.0005  # 0.05% slippage


class PaperExecutor(AbstractExecutor):
    """Virtual order execution with isolated accounts per model.

    State is persisted to the database (virtual_accounts, orders tables).
    This executor runs in the same event loop as live trading — the only
    difference is that orders are simulated rather than sent to OKX.
    """

    def __init__(self, db_session_factory, model_names: list[str] | None = None) -> None:
        self._get_session = db_session_factory
        self._positions: dict[str, list[dict]] = {}  # model_name -> positions
        self._balances: dict[str, float] = {}  # model_name -> balance
        self._model_names = model_names or []

    async def initialize(self) -> None:
        """Load virtual accounts from DB or create defaults."""
        from sqlalchemy import func, select

        from db.repositories.account_repo import AccountRepository
        from db.repositories.trade_repo import TradeRepository
        from models.decision import AIDecision
        from models.trade import Order

        names = self._model_names
        async with self._get_session() as session:
            account_repo = AccountRepository(session)
            trade_repo = TradeRepository(session)
            for model_name in names:
                initial_bal = settings.get_initial_balance(model_name)
                account = await account_repo.get_or_create_account(model_name, initial_bal)
                entry_fee_result = await session.execute(
                    select(func.coalesce(func.sum(Order.fee), 0.0))
                    .join(AIDecision, AIDecision.id == Order.decision_id)
                    .where(
                        Order.model_name == model_name,
                        Order.status == "filled",
                        AIDecision.action.in_(["long", "short"]),
                    )
                )
                entry_fees_paid = float(entry_fee_result.scalar() or 0.0)
                db_positions = await trade_repo.get_open_positions(model_name=model_name)
                open_margin = 0.0
                restored_positions = []
                for p in db_positions:
                    leverage = max(float(p.leverage or 1.0), 1.0)
                    margin_used = (
                        float(p.quantity or 0.0) * float(p.entry_price or 0.0)
                    ) / leverage
                    open_margin += margin_used
                    restored_positions.append(
                        {
                            "id": str(p.id),
                            "symbol": p.symbol,
                            "side": p.side,
                            "quantity": p.quantity,
                            "entry_price": p.entry_price,
                            "current_price": p.current_price or p.entry_price,
                            "leverage": p.leverage,
                            "margin_used": margin_used,
                            "stop_loss": p.stop_loss_price,
                            "take_profit": p.take_profit_price,
                            "is_open": p.is_open,
                            "opened_at": p.created_at,
                            "unrealized_pnl": p.unrealized_pnl or 0.0,
                        }
                    )
                # The per-model quota balance represents available margin.
                account.current_balance = max(
                    0.0,
                    float(account.initial_balance or initial_bal)
                    + float(account.realized_pnl or 0.0)
                    - entry_fees_paid
                    - open_margin,
                )
                await session.flush()
                self._balances[model_name] = account.current_balance
                self._positions[model_name] = restored_positions
                logger.info(
                    "paper account ready",
                    model=model_name,
                    balance=account.current_balance,
                    initial=initial_bal,
                    open_positions=len(restored_positions),
                )

    async def get_balance(self, asset: str = "USDT", model_name: str | None = None) -> float:
        if model_name:
            return self._balances.get(model_name, settings.get_initial_balance(model_name))
        return sum(self._balances.values())

    async def get_balance_for_model(self, model_name: str) -> float:
        return self._balances.get(model_name, settings.get_initial_balance(model_name))

    async def place_order(
        self, decision: DecisionOutput, account_id: str | None = None
    ) -> ExecutionResult:
        """Simulate order execution for a paper trading account.

        Args:
            decision: The AI model's decision.
            account_id: The model_name used as the virtual account identifier.
        """
        model_name = account_id or decision.model_name
        balance = self._balances.get(model_name, settings.get_initial_balance(model_name))
        current_price = (
            decision.feature_snapshot.get("current_price", 0) if decision.feature_snapshot else 0
        )

        # Apply slippage
        if decision.action in (Action.LONG, Action.CLOSE_SHORT):
            fill_price = current_price * (1 + PAPER_SLIPPAGE)
        else:
            fill_price = current_price * (1 - PAPER_SLIPPAGE)

        if fill_price <= 0:
            return ExecutionResult(
                order_id=str(uuid.uuid4())[:12],
                symbol=decision.symbol,
                side=decision.action.value,
                order_type="market",
                quantity=0,
                price=0,
                status=OrderStatus.REJECTED,
                raw_response={"error": "No current price available"},
            )

        if decision.is_hold:
            return ExecutionResult(
                order_id=str(uuid.uuid4())[:12],
                symbol=decision.symbol,
                side="hold",
                order_type="market",
                quantity=0,
                price=fill_price,
                status=OrderStatus.FILLED,
                raw_response={"action": "hold"},
            )

        leverage = max(float(decision.suggested_leverage or 1.0), 1.0)

        # Calculate quantity
        position_value = balance * decision.position_size_pct * leverage
        quantity = position_value / fill_price
        order_value = quantity * fill_price
        fee = order_value * PAPER_FEE_RATE
        margin_used = self._margin_required(order_value, leverage)

        # Handle entry orders
        if decision.action == Action.LONG:
            if margin_used + fee > balance:
                max_order_value = self._max_open_notional(balance, leverage)
                quantity = max_order_value / fill_price if fill_price > 0 else 0.0
                order_value = quantity * fill_price
                fee = order_value * PAPER_FEE_RATE
                margin_used = self._margin_required(order_value, leverage)
                if quantity <= 0:
                    return ExecutionResult(
                        order_id=str(uuid.uuid4())[:12],
                        symbol=decision.symbol,
                        side="long",
                        order_type="market",
                        quantity=0,
                        price=fill_price,
                        status=OrderStatus.REJECTED,
                        raw_response={"error": "Insufficient balance"},
                    )

            # Deduct from balance
            self._balances[model_name] = balance - margin_used - fee

            # Record position
            position = {
                "id": str(uuid.uuid4())[:12],
                "symbol": decision.symbol,
                "side": "long",
                "quantity": quantity,
                "entry_price": fill_price,
                "current_price": fill_price,
                "leverage": leverage,
                "margin_used": margin_used,
                "stop_loss": fill_price * (1 - decision.stop_loss_pct),
                "take_profit": fill_price * (1 + decision.take_profit_pct),
                "is_open": True,
                "opened_at": datetime.now(UTC),
                "unrealized_pnl": 0.0,
            }
            self._positions.setdefault(model_name, []).append(position)

            return ExecutionResult(
                order_id=str(uuid.uuid4())[:12],
                symbol=decision.symbol,
                side="long",
                order_type="market",
                quantity=quantity,
                price=fill_price,
                status=OrderStatus.FILLED,
                fee=fee,
                timestamp=datetime.now(UTC),
            )

        elif decision.action == Action.SHORT:
            if margin_used + fee > balance:
                max_order_value = self._max_open_notional(balance, leverage)
                quantity = max_order_value / fill_price if fill_price > 0 else 0.0
                order_value = quantity * fill_price
                fee = order_value * PAPER_FEE_RATE
                margin_used = self._margin_required(order_value, leverage)
                if quantity <= 0:
                    return ExecutionResult(
                        order_id=str(uuid.uuid4())[:12],
                        symbol=decision.symbol,
                        side="short",
                        order_type="market",
                        quantity=0,
                        price=fill_price,
                        status=OrderStatus.REJECTED,
                        raw_response={"error": "Insufficient balance"},
                    )

            self._balances[model_name] = balance - margin_used - fee

            position = {
                "id": str(uuid.uuid4())[:12],
                "symbol": decision.symbol,
                "side": "short",
                "quantity": quantity,
                "entry_price": fill_price,
                "current_price": fill_price,
                "leverage": leverage,
                "margin_used": margin_used,
                "stop_loss": fill_price * (1 + decision.stop_loss_pct),
                "take_profit": fill_price * (1 - decision.take_profit_pct),
                "is_open": True,
                "opened_at": datetime.now(UTC),
                "unrealized_pnl": 0.0,
            }
            self._positions.setdefault(model_name, []).append(position)

            return ExecutionResult(
                order_id=str(uuid.uuid4())[:12],
                symbol=decision.symbol,
                side="short",
                order_type="market",
                quantity=quantity,
                price=fill_price,
                status=OrderStatus.FILLED,
                fee=fee,
                timestamp=datetime.now(UTC),
            )

        elif decision.action in (Action.CLOSE_LONG, Action.CLOSE_SHORT):
            # Close existing position(s) for this symbol+side
            positions = self._positions.get(model_name, [])
            target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
            to_close = [
                p
                for p in positions
                if p["symbol"] == decision.symbol and p["side"] == target_side and p["is_open"]
            ]

            total_pnl = 0.0
            released_margin = 0.0
            total_fee = 0.0
            total_quantity = 0.0

            for pos in to_close:
                pos["is_open"] = False
                pos["current_price"] = fill_price
                if pos["side"] == "long":
                    pnl = (fill_price - pos["entry_price"]) * pos["quantity"]
                else:
                    pnl = (pos["entry_price"] - fill_price) * pos["quantity"]
                pos["unrealized_pnl"] = pnl
                pos["closed_at"] = datetime.now(UTC)

                close_value = pos["quantity"] * fill_price
                close_fee = close_value * PAPER_FEE_RATE
                total_pnl += pnl
                released_margin += float(
                    pos.get(
                        "margin_used",
                        self._margin_required(
                            pos["quantity"] * pos["entry_price"],
                            pos.get("leverage", 1.0),
                        ),
                    )
                )
                total_fee += close_fee
                total_quantity += pos["quantity"]

            # Return margin + realized PnL to remaining balance
            self._balances[model_name] = (
                self._balances.get(model_name, settings.get_initial_balance(model_name))
                + released_margin
                + total_pnl
                - total_fee
            )

            # Remove closed positions
            self._positions[model_name] = [p for p in positions if p["is_open"]]

            return ExecutionResult(
                order_id=str(uuid.uuid4())[:12],
                symbol=decision.symbol,
                side="close_long" if target_side == "long" else "close_short",
                order_type="market",
                quantity=total_quantity,
                price=fill_price,
                status=OrderStatus.FILLED,
                fee=total_fee,
                pnl=total_pnl - total_fee,
                timestamp=datetime.now(UTC),
            )

        return ExecutionResult(
            order_id=str(uuid.uuid4())[:12],
            symbol=decision.symbol,
            side=decision.action.value,
            order_type="market",
            quantity=0,
            price=fill_price,
            status=OrderStatus.REJECTED,
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        # Paper trading fills instantly, so cancellation is always true (no open orders)
        return True

    def _margin_required(self, notional_value: float, leverage: float | None) -> float:
        lev = max(float(leverage or 1.0), 1.0)
        return float(notional_value or 0.0) / lev

    def _max_open_notional(self, balance: float, leverage: float | None) -> float:
        lev = max(float(leverage or 1.0), 1.0)
        denominator = (1.0 / lev) + PAPER_FEE_RATE
        if denominator <= 0:
            return 0.0
        return max(float(balance or 0.0) / denominator, 0.0)

    async def get_positions(self, symbol: str | None = None) -> list[dict]:
        """Aggregate all open positions across all models."""
        all_positions = []
        for model_name, positions in self._positions.items():
            for pos in positions:
                if pos["is_open"]:
                    pos_copy = dict(pos, model_name=model_name)
                    if symbol is None or pos_copy["symbol"] == symbol:
                        all_positions.append(pos_copy)
        return all_positions

    async def get_all_positions(self, symbol: str | None = None) -> list[dict]:
        """Aggregate ALL positions (open + closed) across all models, sorted by time desc."""
        all_positions = []
        for model_name, positions in self._positions.items():
            for pos in positions:
                pos_copy = dict(pos, model_name=model_name)
                if symbol is not None and pos_copy["symbol"] != symbol:
                    continue
                all_positions.append(pos_copy)
        all_positions.sort(
            key=lambda p: (
                p.get("closed_at") or p.get("opened_at") or datetime(2000, 1, 1, tzinfo=UTC)
            ),
            reverse=True,
        )
        return all_positions

    async def get_positions_for_model(self, model_name: str) -> list[dict]:
        return [
            dict(p, model_name=model_name)
            for p in self._positions.get(model_name, [])
            if p["is_open"]
        ]

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        # Paper trades fill instantly — no open orders
        return []

    async def update_market_prices(self, symbol: str, price: float) -> None:
        """Update current prices and recalculate unrealized PnL for all positions."""
        for _model_name, positions in self._positions.items():
            for pos in positions:
                if pos["symbol"] == symbol and pos["is_open"]:
                    pos["current_price"] = price
                    if pos["side"] == "long":
                        pos["unrealized_pnl"] = (price - pos["entry_price"]) * pos["quantity"]
                    else:
                        pos["unrealized_pnl"] = (pos["entry_price"] - price) * pos["quantity"]

    async def get_account_summary(self, model_name: str) -> dict[str, Any]:
        """Get a summary of the virtual account for a model."""
        initial_bal = settings.get_initial_balance(model_name)
        balance = self._balances.get(model_name, initial_bal)
        positions = [p for p in self._positions.get(model_name, []) if p["is_open"]]
        unrealized = sum(p.get("unrealized_pnl", 0) for p in positions)
        used_margin = sum(
            p.get(
                "margin_used",
                (p.get("quantity", 0) * p.get("entry_price", 0))
                / max(p.get("leverage", 1) or 1, 1),
            )
            for p in positions
        )
        wallet_balance = balance + used_margin
        equity = wallet_balance + unrealized
        total_pnl = equity - initial_bal
        total_pnl_pct = (total_pnl / initial_bal * 100) if initial_bal > 0 else 0.0
        return {
            "model_name": model_name,
            "balance": balance,
            "current_balance": balance,
            "available_balance": balance,
            "wallet_balance": wallet_balance,
            "initial_balance": initial_bal,
            "used_margin": used_margin,
            "unrealized_pnl": unrealized,
            "equity": equity,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "open_positions": len(positions),
            "positions": positions,
        }

    async def get_all_summaries(self) -> list[dict[str, Any]]:
        summaries = []
        for model_name in self._balances:
            summaries.append(await self.get_account_summary(model_name))
        return summaries

    async def shutdown(self) -> None:
        """Persist final account states to database."""
        from db.repositories.account_repo import AccountRepository

        async with self._get_session() as session:
            repo = AccountRepository(session)
            for model_name, balance in self._balances.items():
                positions = [p for p in self._positions.get(model_name, []) if p["is_open"]]
                unrealized = sum(p.get("unrealized_pnl", 0) for p in positions)
                account = await repo.get_account(model_name)
                if account:
                    account.current_balance = balance
                    account.unrealized_pnl = unrealized
                    await session.commit()

        logger.info("paper executor state persisted")
