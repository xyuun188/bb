"""
Performance metrics for evaluating trading strategies.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(returns: list[float], risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio from a list of periodic returns."""
    if not returns or len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    mean_ret = np.mean(arr)
    std_ret = np.std(arr, ddof=1)
    if std_ret == 0:
        return 0.0
    # Assume daily returns, annualize
    return (mean_ret - risk_free_rate / 365) / std_ret * math.sqrt(365)


def calculate_max_drawdown(equity_curve: list[float]) -> float:
    """Calculate maximum drawdown from an equity curve."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def calculate_win_rate(trades: list[dict]) -> float:
    """Calculate win rate from a list of trade results."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return wins / len(trades)


def calculate_profit_factor(trades: list[dict]) -> float:
    """Calculate profit factor (gross profit / gross loss)."""
    gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))
    if gross_loss == 0:
        return gross_profit if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
    """Calculate Calmar ratio (return / max drawdown)."""
    if max_drawdown <= 0:
        return 0.0
    return total_return / max_drawdown


def summarize_performance(
    initial_balance: float,
    final_balance: float,
    trades: list[dict],
    equity_curve: list[float],
) -> dict:
    """Generate a comprehensive performance summary."""
    total_return = (final_balance - initial_balance) / initial_balance
    max_dd = calculate_max_drawdown(equity_curve)
    win_rate = calculate_win_rate(trades)
    profit_factor = calculate_profit_factor(trades)

    # Calculate returns from equity curve
    returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i - 1] != 0:
            returns.append((equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1])

    sharpe = calculate_sharpe_ratio(returns)

    return {
        "initial_balance": initial_balance,
        "final_balance": round(final_balance, 2),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar_ratio": round(calculate_calmar_ratio(total_return, max_dd), 4),
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 4),
        "total_trades": len(trades),
        "equity_curve": equity_curve,
    }
