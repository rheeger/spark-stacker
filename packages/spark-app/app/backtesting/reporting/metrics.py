"""
Metrics calculation module for backtesting reporting.

This module provides functions to calculate performance metrics from backtest results.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def calculate_performance_metrics(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate key performance metrics from a list of trades.

    Args:
        trades: List of trade dictionaries containing at least:
            - 'profit_loss': float - P&L of the trade
            - 'entry_time': datetime - Entry time of the trade
            - 'exit_time': datetime - Exit time of the trade
            - 'entry_price': float - Entry price
            - 'exit_price': float - Exit price

    Returns:
        Dictionary containing performance metrics:
            - win_rate: Percentage of winning trades
            - profit_factor: Ratio of gross profit to gross loss
            - max_drawdown: Maximum drawdown as a percentage
            - sharpe_ratio: Annualized Sharpe ratio
            - total_return: Total return percentage
            - avg_trade_return: Average return per trade
            - num_trades: Total number of trades
    """
    if not trades:
        logger.warning("No trades provided to calculate performance metrics")
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "total_return": 0.0,
            "avg_trade_return": 0.0,
            "num_trades": 0
        }

    # Convert trades to DataFrame for easier processing
    try:
        trades_df = pd.DataFrame(trades)
    except Exception as e:
        logger.error(f"Failed to convert trades to DataFrame: {e}")
        raise ValueError(f"Unable to process trades data: {e}")

    # Calculate basic metrics
    num_trades = len(trades)
    profitable_trades = sum(1 for trade in trades if trade.get('profit_loss', 0) > 0)

    # Win rate
    win_rate = profitable_trades / num_trades if num_trades > 0 else 0

    # Profit factor
    gross_profit = sum(trade.get('profit_loss', 0) for trade in trades if trade.get('profit_loss', 0) > 0)
    gross_loss = abs(sum(trade.get('profit_loss', 0) for trade in trades if trade.get('profit_loss', 0) < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Total return
    # Assuming initial capital is 100% for percentage calculation
    total_profit_loss = sum(trade.get('profit_loss', 0) for trade in trades)
    total_return = total_profit_loss * 100  # As percentage

    # Average trade return
    avg_trade_return = total_return / num_trades if num_trades > 0 else 0

    # Calculate equity curve for drawdown and Sharpe ratio
    equity_curve = calculate_equity_curve(trades)

    # Maximum drawdown
    max_drawdown = calculate_max_drawdown(equity_curve)

    # Sharpe ratio (annualized)
    sharpe_ratio = calculate_sharpe_ratio(equity_curve)

    return {
        "win_rate": round(win_rate * 100, 2),  # As percentage
        "profit_factor": round(profit_factor, 2),
        "max_drawdown": round(max_drawdown * 100, 2),  # As percentage
        "sharpe_ratio": round(sharpe_ratio, 2),
        "total_return": round(total_return, 2),  # As percentage
        "avg_trade_return": round(avg_trade_return, 2),  # As percentage
        "num_trades": num_trades
    }

def calculate_equity_curve(trades: List[Dict[str, Any]]) -> pd.Series:
    """
    Calculate equity curve from a list of trades.

    Args:
        trades: List of trade dictionaries

    Returns:
        Pandas Series representing the equity curve
    """
    if not trades:
        return pd.Series()

    # Sort trades by exit time
    sorted_trades = sorted(trades, key=lambda x: x.get('exit_time', 0))

    # Create cumulative equity series
    equity = [0]  # Start with 0% return

    for trade in sorted_trades:
        pnl = trade.get('profit_loss', 0)
        equity.append(equity[-1] + pnl)

    return pd.Series(equity)

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown from an equity curve.

    Args:
        equity_curve: Pandas Series representing the equity curve

    Returns:
        Maximum drawdown as a fraction (not percentage)
    """
    if len(equity_curve) <= 1:
        return 0.0

    # Calculate drawdown series
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / (running_max + 1e-10)  # Avoid division by zero

    return abs(drawdown.min()) if not np.isnan(drawdown.min()) else 0.0

def calculate_sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0, trading_days: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio from an equity curve.

    Args:
        equity_curve: Pandas Series representing the equity curve
        risk_free_rate: Annual risk-free rate (default: 0.0)
        trading_days: Number of trading days in a year (default: 252)

    Returns:
        Annualized Sharpe ratio
    """
    if len(equity_curve) <= 1:
        return 0.0

    # Calculate daily returns
    daily_returns = equity_curve.diff().dropna()

    if len(daily_returns) == 0:
        return 0.0

    # Mean daily return
    mean_return = daily_returns.mean()

    # Daily risk-free rate
    daily_rf = risk_free_rate / trading_days

    # Standard deviation of daily returns
    std_dev = daily_returns.std()

    if std_dev == 0:
        return 0.0

    # Daily Sharpe ratio
    daily_sharpe = (mean_return - daily_rf) / std_dev

    # Annualized Sharpe ratio
    annualized_sharpe = daily_sharpe * np.sqrt(trading_days)

    return annualized_sharpe
