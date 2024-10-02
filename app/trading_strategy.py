import pandas as pd
import numpy as np
from ta import trend, volatility
import logging

def get_indicators(api, symbol, timeframe='5Sec', limit=100):
    logging.info(f"Getting indicators for {symbol}")
    barset = api.get_crypto_bars(symbol, timeframe, limit=limit).df
    df = barset.reset_index()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    
    df.set_index('timestamp', inplace=True)
    
    logging.info("Calculating technical indicators")
    
    # Calculate ATR
    df['atr'] = volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=10).average_true_range()
    
    # Calculate SuperTrend
    factor = 3
    df['hl2'] = (df['high'] + df['low']) / 2
    df['upperband'] = df['hl2'] + factor * df['atr']
    df['lowerband'] = df['hl2'] - factor * df['atr']
    df['in_uptrend'] = True

    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['upperband'].iloc[i-1]:
            df['in_uptrend'].iloc[i] = True
        elif df['close'].iloc[i] < df['lowerband'].iloc[i-1]:
            df['in_uptrend'].iloc[i] = False
        else:
            df['in_uptrend'].iloc[i] = df['in_uptrend'].iloc[i-1]
            if df['in_uptrend'].iloc[i] and df['lowerband'].iloc[i] < df['lowerband'].iloc[i-1]:
                df['lowerband'].iloc[i] = df['lowerband'].iloc[i-1]
            if not df['in_uptrend'].iloc[i] and df['upperband'].iloc[i] > df['upperband'].iloc[i-1]:
                df['upperband'].iloc[i] = df['upperband'].iloc[i-1]
    
    df['supertrend'] = np.where(df['in_uptrend'], df['lowerband'], df['upperband'])
    
    logging.info("Finished calculating indicators")
    return df

def generate_signals(df):
    logging.info("Generating trading signals")
    
    # Detect trend shifts
    df['trend_shift'] = df['in_uptrend'].diff()
    
    # Generate buy and sell signals
    if df['trend_shift'].iloc[-1] == 1:  # Bullish trend shift
        return 'buy'
    elif df['trend_shift'].iloc[-1] == -1:  # Bearish trend shift
        return 'sell'
    else:
        return None

class TradingState:
    def __init__(self):
        self.last_sell_amount = 0
        self.in_position = False
        self.sold_in_current_run = False

def execute_strategy(api, symbol, current_position, minimum_amount, latest_price, trading_state):
    df = get_indicators(api, symbol)
    signal = generate_signals(df)
    
    if signal == 'buy':
        if not trading_state.in_position:
            # Buy the minimum amount
            qty_to_buy = minimum_amount / latest_price
            logging.info(f"BUY signal: Buying {qty_to_buy} {symbol}")
            trading_state.in_position = True
            return 'buy', qty_to_buy
        elif trading_state.sold_in_current_run:
            # Rebuy the entire amount sold previously plus minimum_amount
            buy_amount = trading_state.last_sell_amount + minimum_amount
            qty_to_buy = buy_amount / latest_price
            logging.info(f"BUY signal: Rebuying {qty_to_buy} {symbol}")
            trading_state.sold_in_current_run = False
            return 'buy', qty_to_buy
        else:
            # Already in a position, buy minimum_amount more
            qty_to_buy = minimum_amount / latest_price
            logging.info(f"BUY signal: Adding {qty_to_buy} {symbol} to position")
            return 'buy', qty_to_buy
    
    elif signal == 'sell' and trading_state.in_position:
        # Sell entire position
        qty_to_sell = current_position
        trading_state.last_sell_amount = qty_to_sell * latest_price
        trading_state.in_position = False
        trading_state.sold_in_current_run = True
        logging.info(f"SELL signal: Selling entire position of {qty_to_sell} {symbol}")
        return 'sell', qty_to_sell
    
    return None, 0