import os
import sys
import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, Optional

# Add the app directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import project components
from app.utils.config import (
    AppConfig,
    ExchangeConfig,
    TradingStrategyConfig,
    IndicatorConfig,
)
from app.connectors.base_connector import (
    BaseConnector,
    OrderSide,
    OrderType,
    MarketType,
)
from app.indicators.base_indicator import BaseIndicator, Signal, SignalDirection
from app.risk_management.risk_manager import RiskManager
from app.core.trading_engine import TradingEngine


@pytest.fixture
def sample_price_data():
    """Sample price data for indicator testing."""
    # Create a DataFrame with typical OHLCV data
    data = {
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1h"),
        "symbol": "ETH",
        "open": np.random.normal(1500, 50, 100),
        "high": np.random.normal(1550, 50, 100),
        "low": np.random.normal(1450, 50, 100),
        "close": np.random.normal(1500, 50, 100),
        "volume": np.random.normal(1000, 200, 100),
    }

    # Make sure high is actually the highest and low is the lowest
    for i in range(len(data["open"])):
        data["high"][i] = max(data["open"][i], data["close"][i], data["high"][i])
        data["low"][i] = min(data["open"][i], data["close"][i], data["low"][i])

    return pd.DataFrame(data)


@pytest.fixture
def sample_config():
    """Sample application configuration."""
    config = AppConfig(
        log_level="INFO",
        webhook_enabled=False,
        webhook_port=8080,
        webhook_host="0.0.0.0",
        max_parallel_trades=1,
        polling_interval=60,
        dry_run=True,
        exchanges=[
            ExchangeConfig(
                name="mock_exchange",
                wallet_address="0x123",
                private_key="0xabc",
                testnet=True,
                use_as_main=True,
                use_as_hedge=True,
            )
        ],
        strategies=[
            TradingStrategyConfig(
                name="test_strategy",
                market="ETH",
                enabled=True,
                main_leverage=5.0,
                hedge_leverage=2.0,
                hedge_ratio=0.2,
                stop_loss_pct=10.0,
                take_profit_pct=20.0,
                max_position_size=100.0,
            )
        ],
        indicators=[
            IndicatorConfig(
                name="test_rsi",
                type="rsi",
                enabled=True,
                parameters={"period": 14, "overbought": 70, "oversold": 30},
            )
        ],
    )
    return config


@pytest.fixture
def mock_connector():
    """Mock exchange connector."""
    connector = MagicMock(spec=BaseConnector)

    # Set up basic methods
    connector.connect.return_value = True
    connector.get_markets.return_value = [
        {"symbol": "ETH", "base_asset": "ETH", "quote_asset": "USD", "min_size": 0.001}
    ]
    connector.get_ticker.return_value = {
        "symbol": "ETH",
        "last_price": 1500.0,
        "bid": 1499.0,
        "ask": 1501.0,
        "volume": 1000.0,
    }
    connector.get_account_balance.return_value = {"USD": 10000.0}
    connector.get_positions.return_value = []

    # Mock the place_order method to return a successful order
    connector.place_order.return_value = {
        "order_id": "mock_order_123",
        "symbol": "ETH",
        "side": "BUY",
        "size": 1.0,
        "price": 1500.0,
        "status": "FILLED",
    }

    # Add market_types property
    connector.market_types = MarketType.SPOT

    return connector


@pytest.fixture
def mock_risk_manager():
    """Mock risk manager."""
    risk_manager = MagicMock(spec=RiskManager)

    # Set up basic methods
    risk_manager.calculate_position_size.return_value = (100.0, 5.0)  # size, leverage
    risk_manager.calculate_hedge_parameters.return_value = (20.0, 2.0)  # size, leverage
    risk_manager.validate_trade.return_value = (True, "Trade validated")

    return risk_manager


@pytest.fixture
def trading_engine(mock_connector, mock_risk_manager):
    """Trading engine instance with mock components."""
    engine = TradingEngine(
        main_connector=mock_connector,
        hedge_connector=mock_connector,  # Use the same mock for both
        risk_manager=mock_risk_manager,
        dry_run=True,
        polling_interval=1,
        max_parallel_trades=1,
    )
    return engine


@pytest.fixture
def sample_signal():
    """Sample trading signal."""
    return Signal(
        direction=SignalDirection.BUY,
        symbol="ETH",
        indicator="test_indicator",
        confidence=0.8,
        timestamp=1630000000000,
        params={"reason": "test signal"},
    )
