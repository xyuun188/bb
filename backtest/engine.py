"""
Backtesting engine for strategy validation.
Uses Backtrader for event-driven backtesting with historical data.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import UTC
from typing import Any

import backtrader as bt
import pandas as pd
import structlog

from core.strategy_contracts import (
    PositionSide,
    StrategyAction,
    StrategyContext,
    StrategyContractError,
    StrategyDecision,
    assert_decision_matches_context,
)
from services.strategy_contract_adapter import context_from_historical_bar

logger = structlog.get_logger(__name__)


class StandardContractBacktestStrategy(bt.Strategy):
    """Backtrader execution adapter for a pure ``StrategyDecision`` provider.

    The provider is the only strategy logic. This class translates its decision
    into simulated fills and never talks to an exchange or production state.
    """

    params = dict(
        decision_provider=None,
        feature_provider=None,
        symbol="",
        timeframe="1h",
        strategy_identity={},
        parameters={},
        account_constraints={},
        execution_assumptions={},
    )

    def __init__(self):
        if not callable(self.p.decision_provider):
            raise StrategyContractError("decision_provider must be callable")
        self.order = None
        self.decision_records: list[dict[str, Any]] = []

    def next(self):
        if self.order:
            return
        price = float(self.data.close[0])
        if not math.isfinite(price) or price <= 0:
            return
        timestamp = bt.num2date(self.data.datetime[0])
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        bar = {
            "open": float(self.data.open[0]),
            "high": float(self.data.high[0]),
            "low": float(self.data.low[0]),
            "close": price,
            "volume": float(self.data.volume[0]),
        }
        feature_provider = self.p.feature_provider
        features = feature_provider(self, bar) if callable(feature_provider) else {}
        context = context_from_historical_bar(
            symbol=str(self.p.symbol or ""),
            timestamp=timestamp,
            timeframe=str(self.p.timeframe or "1h"),
            bar=bar,
            feature_snapshot=features if isinstance(features, Mapping) else {},
            strategy=dict(self.p.strategy_identity or {}),
            parameters=dict(self.p.parameters or {}),
            position_snapshot=self._position_snapshot(price),
            account_constraints=dict(self.p.account_constraints or {}),
            execution_assumptions=dict(self.p.execution_assumptions or {}),
        )
        decision = self.p.decision_provider(context)
        if not isinstance(decision, StrategyDecision):
            raise StrategyContractError("decision_provider must return StrategyDecision")
        assert_decision_matches_context(context, decision)
        self.decision_records.append(
            {
                "context_sha256": context.context_sha256,
                "strategy_input_sha256": context.strategy_input_sha256,
                "decision_sha256": decision.decision_sha256,
                "decision": decision.to_dict(),
            }
        )
        self._execute_decision(decision, price)

    def _position_snapshot(self, price: float) -> dict[str, Any]:
        size = float(self.position.size)
        side = "long" if size > 0 else "short" if size < 0 else "none"
        account_value = float(self.broker.getvalue())
        exposure = abs(size * price) / account_value if account_value > 0 else 0.0
        return {
            "side": side,
            "size": size,
            "average_price": float(self.position.price or 0.0),
            "exposure": exposure,
        }

    def _execute_decision(self, decision: StrategyDecision, price: float) -> None:
        if decision.action == StrategyAction.ENTER:
            if self.position:
                return
            account_value = max(float(self.broker.getvalue()), 0.0)
            size = account_value * decision.target_exposure / price
            if size <= 0:
                return
            self.order = self.buy(size=size) if decision.side == PositionSide.LONG else self.sell(size=size)
            return
        if decision.action == StrategyAction.EXIT:
            if not self.position or not self._side_matches(decision.side):
                return
            self.order = self.close()
            return
        if decision.action == StrategyAction.REDUCE:
            if not self.position or not self._side_matches(decision.side):
                return
            account_value = max(float(self.broker.getvalue()), 0.0)
            desired_size = account_value * decision.target_exposure / price
            reduction = max(abs(float(self.position.size)) - desired_size, 0.0)
            if reduction <= 0:
                return
            self.order = (
                self.sell(size=reduction)
                if self.position.size > 0
                else self.buy(size=reduction)
            )

    def _side_matches(self, side: PositionSide) -> bool:
        return (self.position.size > 0 and side == PositionSide.LONG) or (
            self.position.size < 0 and side == PositionSide.SHORT
        )

    def notify_order(self, order):
        if order.status in {order.Completed, order.Canceled, order.Margin, order.Rejected}:
            self.order = None


class AITradingStrategy(bt.Strategy):
    """Backtrader strategy that uses AI model decisions.

    In backtest mode, decisions are simulated based on indicator rules
    rather than calling the LLM API (to avoid cost and latency).
    """

    params = dict(
        rsi_oversold=30,
        rsi_overbought=70,
        macd_threshold=0,
    )

    def __init__(self):
        self.order = None
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.macd = bt.indicators.MACD(self.data.close)
        self.bb = bt.indicators.BollingerBands(self.data.close, period=20)
        self.sma20 = bt.indicators.SMA(self.data.close, period=20)
        self.sma50 = bt.indicators.SMA(self.data.close, period=50)
        self.atr = bt.indicators.ATR(self.data, period=14)

    def next(self):
        if self.order:
            return

        price = self.data.close[0]

        # Simple rule-based strategy (placeholder for AI decisions)
        if not self.position:
            # Entry conditions
            if (
                self.rsi[0] < self.p.rsi_oversold
                and self.macd.macd[0] > self.macd.signal[0]
                and price > self.sma20[0]
            ):
                size = self.broker.getcash() * 0.1 / price
                stop_loss = price * 0.95
                self.order = self.buy(size=size)
                self.sell(exectype=bt.Order.Stop, price=stop_loss, size=size)

            elif (
                self.rsi[0] > self.p.rsi_overbought
                and self.macd.macd[0] < self.macd.signal[0]
                and price < self.sma20[0]
            ):
                size = self.broker.getcash() * 0.1 / price
                stop_loss = price * 1.05
                self.order = self.sell(size=size)
                self.buy(exectype=bt.Order.Stop, price=stop_loss, size=size)

        else:
            # Exit conditions
            if (
                self.position.size > 0
                and self.rsi[0] > 70
                and self.macd.macd[0] < self.macd.signal[0]
            ):
                self.order = self.close()
            elif (
                self.position.size < 0
                and self.rsi[0] < 30
                and self.macd.macd[0] > self.macd.signal[0]
            ):
                self.order = self.close()

    def notify_trade(self, trade):
        if trade.isclosed:
            logger.debug(
                "backtest trade closed",
                pnl=trade.pnl,
                net=trade.pnlcomm,
            )

    def notify_order(self, order):
        """Release the entry/exit guard after Backtrader resolves an order."""

        if order.status in {order.Completed, order.Canceled, order.Margin, order.Rejected}:
            self.order = None


class BacktestEngine:
    """Run backtests with historical data."""

    RUNNER_VERSION = "bb-backtrader-runner.v2"

    def __init__(
        self,
        initial_cash: float = 10000.0,
        *,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0,
        random_seed: int = 0,
    ):
        if not math.isfinite(float(initial_cash)) or float(initial_cash) <= 0:
            raise ValueError("initial_cash must be a positive finite number")
        if not math.isfinite(float(commission_rate)) or float(commission_rate) < 0:
            raise ValueError("commission_rate must be a non-negative finite number")
        if not math.isfinite(float(slippage_rate)) or float(slippage_rate) < 0:
            raise ValueError("slippage_rate must be a non-negative finite number")
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(float(initial_cash))
        self.cerebro.broker.setcommission(commission=float(commission_rate))
        if float(slippage_rate) > 0:
            self.cerebro.broker.set_slippage_perc(perc=float(slippage_rate))
        self.initial_cash = float(initial_cash)
        self.commission_rate = float(commission_rate)
        self.slippage_rate = float(slippage_rate)
        self.random_seed = int(random_seed)
        self._analyzers_added = False

    def load_data(self, df: pd.DataFrame) -> None:
        """Load OHLCV DataFrame into backtrader."""
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # Use index
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
        )
        self.cerebro.adddata(data)

    def add_strategy(self, strategy_class=AITradingStrategy, **params):
        self.cerebro.addstrategy(strategy_class, **params)

    def add_contract_strategy(
        self,
        decision_provider: Callable[[StrategyContext], StrategyDecision],
        *,
        symbol: str,
        timeframe: str = "1h",
        strategy: Mapping[str, Any],
        parameters: Mapping[str, Any],
        account_constraints: Mapping[str, Any] | None = None,
        execution_assumptions: Mapping[str, Any] | None = None,
        feature_provider: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        """Install the standard contract adapter without replacing the legacy strategy."""

        assumptions = {
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "fill_model": "backtrader_next_bar",
            "random_seed": self.random_seed,
            **dict(execution_assumptions or {}),
        }
        self.cerebro.addstrategy(
            StandardContractBacktestStrategy,
            decision_provider=decision_provider,
            feature_provider=feature_provider,
            symbol=symbol,
            timeframe=timeframe,
            strategy_identity=dict(strategy),
            parameters=dict(parameters),
            account_constraints=dict(account_constraints or {}),
            execution_assumptions=assumptions,
        )

    def add_analyzer(self):
        if self._analyzers_added:
            return
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        self._analyzers_added = True

    def run(self) -> dict[str, Any]:
        """Run the backtest and return results."""
        self.add_analyzer()
        results = self.cerebro.run()
        strategy = results[0]

        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash

        sharpe = strategy.analyzers.sharpe.get_analysis()
        drawdown = strategy.analyzers.drawdown.get_analysis()
        trades = strategy.analyzers.trades.get_analysis()

        total_trades = int(trades.get("total", {}).get("total", 0) or 0)
        winning_trades = int(trades.get("won", {}).get("total", 0) or 0)
        losing_trades = int(trades.get("lost", {}).get("total", 0) or 0)
        gross_profit = float(trades.get("won", {}).get("pnl", {}).get("total", 0) or 0)
        gross_loss = abs(float(trades.get("lost", {}).get("pnl", {}).get("total", 0) or 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        result = {
            "runner_version": self.RUNNER_VERSION,
            "initial_cash": self.initial_cash,
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return * 100, 2),
            "sharpe_ratio": sharpe.get("sharperatio", 0) or 0,
            "max_drawdown_pct": round(drawdown.get("max", {}).get("drawdown", 0), 2),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "profit_factor": round(profit_factor, 6),
            "gross_profit": round(gross_profit, 8),
            "gross_loss": round(gross_loss, 8),
            "net_profit": round(float(final_value - self.initial_cash), 8),
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "random_seed": self.random_seed,
        }
        records = getattr(strategy, "decision_records", None)
        if isinstance(records, list):
            result["strategy_contract"] = {
                "contract_version": "bb.strategy-decision.v1",
                "decision_count": len(records),
                "decision_sha256": [record["decision_sha256"] for record in records],
            }
        return result
