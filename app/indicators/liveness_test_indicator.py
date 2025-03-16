import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd

from app.indicators.base_indicator import BaseIndicator, Signal, SignalDirection

logger = logging.getLogger(__name__)

class LivenessTestIndicator(BaseIndicator):
    """
    A simple indicator for testing the liveness of the trading system.
    
    This indicator buys when the current minute is even and sells when the minute is odd.
    It's designed purely for testing that the system is operational, not as an actual strategy.
    """
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the liveness test indicator.
        
        Args:
            name: Indicator name
            params: Optional parameters:
                symbol: The trading pair to use (default: "PYUSD/USDT")
                confidence: Signal confidence level (default: 0.7)
        """
        super().__init__(name, params)
        self.symbol = self.params.get('symbol', 'PYUSD/USDT')
        self.confidence = self.params.get('confidence', 0.7)
        
        logger.info(f"Initialized LivenessTestIndicator: Using symbol {self.symbol}")
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the even/odd minute condition for the current time.
        
        Since this indicator doesn't use market data, the input data is not modified.
        Instead, we just add a column indicating whether we should buy or sell based on the current minute.
        
        Args:
            data: Price data as pandas DataFrame (not actually used)
            
        Returns:
            DataFrame with added 'is_even_minute' column
        """
        # Create a copy of the dataframe
        df = data.copy()
        
        # Get current time and check if minute is even
        current_minute = datetime.now().minute
        is_even_minute = current_minute % 2 == 0
        
        # Add a column to indicate even/odd minute
        df['is_even_minute'] = is_even_minute
        
        # Get price from data if available
        current_price = df['close'].iloc[-1] if 'close' in df.columns and len(df) > 0 else 0.0
        
        # Log for debugging
        logger.info(f"LivenessTest calculation: Current minute is {current_minute} ({is_even_minute=}), symbol={self.symbol}, price={current_price}")
        
        return df
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Generate a trading signal based on current minute (even/odd).
        
        Buy when minute is even, sell when minute is odd.
        
        Args:
            data: Price data with is_even_minute column from calculate()
            
        Returns:
            Signal object with appropriate direction
        """
        # Check if our column exists (it should, we just added it)
        if 'is_even_minute' not in data.columns:
            logger.warning("No is_even_minute column in data, cannot generate signal")
            return None
        
        # Get the latest value (should be the same for all rows since we added it)
        is_even_minute = data['is_even_minute'].iloc[-1]
        current_minute = datetime.now().minute
        
        # Get current price from data
        current_price = data['close'].iloc[-1] if 'close' in data.columns and len(data) > 0 else 1.0
        
        # Generate appropriate signal based on even/odd minute
        if is_even_minute:
            direction = SignalDirection.BUY
            trigger = "even_minute"
        else:
            direction = SignalDirection.SELL
            trigger = "odd_minute"
        
        # Create the signal
        signal = Signal(
            direction=direction,
            symbol=self.symbol,
            indicator=self.name,
            confidence=self.confidence,
            params={
                'minute': current_minute,
                'is_even': is_even_minute,
                'trigger': trigger,
                'price': current_price  # Include the price in the signal parameters
            }
        )
        
        logger.info(f"LivenessTest generated signal: {signal} with price {current_price}")
        return signal
    
    def __str__(self) -> str:
        """String representation of the indicator."""
        return f"LivenessTest(symbol={self.symbol}, confidence={self.confidence})" 