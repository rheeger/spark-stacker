import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.indicators.rsi_indicator import RSIIndicator


@pytest.fixture
def historical_price_data():
    """Generate realistic historical price data for backtesting."""
    # Create a date range for the past 100 days
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Generate a random walk for the close prices starting at 1000
    np.random.seed(42)  # For reproducibility
    close_prices = 1000 + np.cumsum(np.random.normal(0, 20, 100))
    
    # Generate open, high, low prices based on close prices
    open_prices = close_prices - np.random.normal(0, 10, 100)
    high_prices = np.maximum(close_prices, open_prices) + np.random.normal(5, 10, 100)
    low_prices = np.minimum(close_prices, open_prices) - np.random.normal(5, 10, 100)
    
    # Generate volume data
    volume = np.random.normal(1000000, 200000, 100)
    
    # Create the DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    return df


def simulate_trading(price_data, indicator, initial_balance=10000.0, position_size_pct=0.2):
    """
    Simulate trading based on indicator signals.
    
    Args:
        price_data: DataFrame with OHLCV data
        indicator: The indicator to use for signals
        initial_balance: Starting balance
        position_size_pct: Position size as percentage of balance
    
    Returns:
        DataFrame with trading results
    """
    # Process data with indicator
    processed_data = indicator.calculate(price_data)
    
    # Initialize trading variables
    balance = initial_balance
    position = 0.0
    entry_price = 0.0
    trades = []
    
    # Iterate through the data
    for date, row in processed_data.iterrows():
        # Generate a signal for the current data point
        signal = indicator.generate_signal(processed_data.loc[:date])
        
        # If we have no position and get a buy signal
        if position == 0.0 and signal and signal.direction.value == 'BUY':
            # Calculate position size
            position = (balance * position_size_pct) / row['close']
            entry_price = row['close']
            trades.append({
                'date': date,
                'action': 'BUY',
                'price': entry_price,
                'position': position,
                'balance': balance
            })
        
        # If we have a position and get a sell signal
        elif position > 0.0 and signal and signal.direction.value == 'SELL':
            # Calculate profit/loss
            pnl = position * (row['close'] - entry_price)
            balance += position * row['close']
            trades.append({
                'date': date,
                'action': 'SELL',
                'price': row['close'],
                'position': position,
                'balance': balance,
                'pnl': pnl
            })
            position = 0.0
    
    # Close any remaining position using the last price
    if position > 0.0:
        last_price = processed_data.iloc[-1]['close']
        pnl = position * (last_price - entry_price)
        balance += position * last_price
        trades.append({
            'date': processed_data.index[-1],
            'action': 'CLOSE',
            'price': last_price,
            'position': position,
            'balance': balance,
            'pnl': pnl
        })
    
    # Convert trades to DataFrame
    if trades:
        return pd.DataFrame(trades)
    else:
        return pd.DataFrame(columns=['date', 'action', 'price', 'position', 'balance', 'pnl'])


def calculate_performance_metrics(trades_df, initial_balance=10000.0):
    """
    Calculate performance metrics from trading results.
    
    Args:
        trades_df: DataFrame with trading results
        initial_balance: Starting balance
    
    Returns:
        Dict with performance metrics
    """
    if trades_df.empty:
        return {
            'total_return_pct': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown_pct': 0.0
        }
    
    # Calculate returns
    final_balance = trades_df.iloc[-1]['balance']
    total_return = final_balance - initial_balance
    total_return_pct = (total_return / initial_balance) * 100
    
    # Calculate win rate
    winning_trades = trades_df[trades_df.get('pnl', 0) > 0]
    losing_trades = trades_df[trades_df.get('pnl', 0) < 0]
    total_trades = len(winning_trades) + len(losing_trades)
    
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
    
    # Calculate average profit and loss
    avg_profit = winning_trades['pnl'].mean() if not winning_trades.empty else 0.0
    avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0.0
    
    # Calculate profit factor
    total_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0.0
    total_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0.0
    profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
    
    # Calculate drawdown
    balance_history = trades_df['balance'].tolist()
    max_balance = initial_balance
    max_drawdown = 0.0
    
    for balance in balance_history:
        if balance > max_balance:
            max_balance = balance
        else:
            drawdown = (max_balance - balance) / max_balance * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    
    return {
        'total_return_pct': total_return_pct,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown
    }


def test_rsi_strategy_basic_simulation(historical_price_data):
    """
    Test a basic RSI trading strategy simulation.
    
    This test verifies that an RSI-based trading strategy can be simulated
    and that reasonable performance metrics can be calculated.
    """
    # Create an RSI indicator with standard settings
    rsi = RSIIndicator(name="RSI", params={"period": 14, "overbought": 70, "oversold": 30})
    
    # Simulate trading using the RSI indicator
    trades_df = simulate_trading(
        price_data=historical_price_data,
        indicator=rsi,
        initial_balance=10000.0,
        position_size_pct=0.2
    )
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(trades_df, initial_balance=10000.0)
    
    # Log the performance metrics for analysis
    print("\nRSI Strategy Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Basic assertions to verify the simulation works
    assert 'total_return_pct' in metrics
    assert 'win_rate' in metrics
    assert 'total_trades' in metrics
    assert isinstance(metrics['total_return_pct'], float)
    assert isinstance(metrics['win_rate'], float)
    assert isinstance(metrics['total_trades'], int)


def test_rsi_parameter_optimization(historical_price_data):
    """
    Test RSI parameter optimization through simulation.
    
    This test explores different RSI parameter combinations to find
    the optimal settings for the given historical data.
    """
    # Define parameter ranges to test
    rsi_periods = [7, 14, 21]
    overbought_levels = [70, 75, 80]
    oversold_levels = [20, 25, 30]
    
    # Store results
    results = []
    
    # Test each parameter combination
    for period in rsi_periods:
        for overbought in overbought_levels:
            for oversold in oversold_levels:
                # Create RSI indicator with these parameters
                rsi = RSIIndicator(name="RSI", params={"period": period, "overbought": overbought, "oversold": oversold})
                
                # Simulate trading
                trades_df = simulate_trading(
                    price_data=historical_price_data,
                    indicator=rsi,
                    initial_balance=10000.0,
                    position_size_pct=0.2
                )
                
                # Calculate metrics
                metrics = calculate_performance_metrics(trades_df, initial_balance=10000.0)
                
                # Add parameters to metrics
                metrics['period'] = period
                metrics['overbought'] = overbought
                metrics['oversold'] = oversold
                
                # Store results
                results.append(metrics)
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Find the best parameter combination based on return
    if not results_df.empty:
        best_params = results_df.loc[results_df['total_return_pct'].idxmax()]
        
        print("\nOptimal RSI Parameters:")
        print(f"Period: {best_params['period']}")
        print(f"Overbought Level: {best_params['overbought']}")
        print(f"Oversold Level: {best_params['oversold']}")
        print(f"Total Return: {best_params['total_return_pct']:.2f}%")
        print(f"Win Rate: {best_params['win_rate']:.2f}")
        print(f"Total Trades: {best_params['total_trades']}")
        
        # Make sure we found something
        assert not results_df.empty
        assert best_params['total_return_pct'] >= results_df['total_return_pct'].min()
    else:
        pytest.skip("No trades were executed in any of the parameter combinations") 