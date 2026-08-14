"""
Backtesting engine for strategy validation.
Uses Backtrader for event-driven backtesting with historical data.
"""

from __future__ import annotations

import math
from typing import Any

import backtrader as bt
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


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

        return {
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
