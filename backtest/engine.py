"""
Backtesting engine for strategy validation.
Uses Backtrader for event-driven backtesting with historical data.
"""

from __future__ import annotations

import datetime
from typing import Any

import backtrader as bt
import pandas as pd
import structlog

from config.settings import settings
from data_feed.technical_indicators import compute_all_indicators

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
                self.buy(size=size)
                self.sell(exectype=bt.Order.Stop, price=stop_loss, size=size)

            elif (
                self.rsi[0] > self.p.rsi_overbought
                and self.macd.macd[0] < self.macd.signal[0]
                and price < self.sma20[0]
            ):
                size = self.broker.getcash() * 0.1 / price
                stop_loss = price * 1.05
                self.sell(size=size)
                self.buy(exectype=bt.Order.Stop, price=stop_loss, size=size)

        else:
            # Exit conditions
            if (
                self.position.size > 0
                and self.rsi[0] > 70
                and self.macd.macd[0] < self.macd.signal[0]
            ):
                self.close()
            elif (
                self.position.size < 0
                and self.rsi[0] < 30
                and self.macd.macd[0] > self.macd.signal[0]
            ):
                self.close()

    def notify_trade(self, trade):
        if trade.isclosed:
            logger.debug(
                "backtest trade closed",
                pnl=trade.pnl,
                net=trade.pnlcomm,
            )


class BacktestEngine:
    """Run backtests with historical data."""

    def __init__(self, initial_cash: float = 10000.0):
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(initial_cash)
        self.cerebro.broker.setcommission(commission=0.001)  # 0.1%
        self.initial_cash = initial_cash

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
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

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

        return {
            "initial_cash": self.initial_cash,
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return * 100, 2),
            "sharpe_ratio": sharpe.get("sharperatio", 0) or 0,
            "max_drawdown_pct": round(drawdown.get("max", {}).get("drawdown", 0), 2),
            "total_trades": trades.get("total", {}).get("total", 0),
            "winning_trades": trades.get("won", {}).get("total", 0),
            "losing_trades": trades.get("lost", {}).get("total", 0),
        }
