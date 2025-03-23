import pytest
import json
from unittest.mock import MagicMock, patch
import logging

from app.connectors.coinbase_connector import CoinbaseConnector
from app.connectors.base_connector import OrderSide, OrderType, MarketType

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def mock_rest_client():
    """Create a mock REST client for testing."""
    mock = MagicMock()
    
    # Mock account balances for a complete connector
    accounts_response = MagicMock()
    accounts = [
        MagicMock(
            currency="USD",
            available_balance=MagicMock(value="10000.0")
        ),
        MagicMock(
            currency="ETH",
            available_balance=MagicMock(value="5.0")
        )
    ]
    accounts_response.accounts = accounts
    mock.get_accounts.return_value = accounts_response
    
    return mock

@pytest.fixture
def connector(mock_rest_client):
    """Create a Coinbase connector with mocked client."""
    connector = CoinbaseConnector(
        api_key="test_key",
        api_secret="test_secret",
        testnet=True,
        market_types=MarketType.SPOT
    )
    connector.client = mock_rest_client
    return connector

def test_get_optimal_limit_price_buy_order(connector):
    """Test optimal limit price calculation for a BUY order."""
    # Mock the orderbook response with sample data
    mock_orderbook = {
        "bids": [[1900.0, 2.0], [1895.0, 3.0], [1890.0, 5.0]],
        "asks": [[1905.0, 1.0], [1910.0, 2.0], [1915.0, 3.0]]
    }
    
    with patch.object(connector, 'get_orderbook', return_value=mock_orderbook):
        result = connector.get_optimal_limit_price(
            symbol="ETH",
            side=OrderSide.BUY,
            amount=1.5
        )
        
        # For a BUY order, we should be using the ask side
        # With a small buffer added for immediate execution
        expected_price = 1910.0 * 1.0005  # 1910 + 0.05% buffer
        
        assert result["price"] > 1910.0  # Should be slightly higher than the best ask that covers our amount
        assert abs(result["price"] - expected_price) < 0.01  # Should be within a small margin of error
        assert result["enough_liquidity"] == True
        
        # Check the batches - should have batches for each price level needed
        # Order should span first price level (1905) and part of second (1910)
        assert len(result["batches"]) >= 1  # May be simplified to 1 batch with the optimal price
        
        # If multiple batches, check they're correctly calculated
        if len(result["batches"]) > 1:
            assert result["batches"][0]["price"] == 1905.0
            assert result["batches"][0]["amount"] == 1.0
            assert result["batches"][1]["price"] == 1910.0
            assert result["batches"][1]["amount"] == 0.5
        
        assert result["slippage"] < 1.0  # Small order should have minimal slippage

def test_get_optimal_limit_price_sell_order(connector):
    """Test optimal limit price calculation for a SELL order."""
    # Mock the orderbook response with sample data
    mock_orderbook = {
        "bids": [[1900.0, 2.0], [1895.0, 3.0], [1890.0, 5.0]],
        "asks": [[1905.0, 1.0], [1910.0, 2.0], [1915.0, 3.0]]
    }
    
    with patch.object(connector, 'get_orderbook', return_value=mock_orderbook):
        result = connector.get_optimal_limit_price(
            symbol="ETH",
            side=OrderSide.SELL,
            amount=1.5
        )
        
        # For a SELL order, we should be using the bid side
        # With a small buffer subtracted for immediate execution
        expected_price = 1900.0 * 0.9995  # 1900 - 0.05% buffer
        
        assert result["price"] < 1900.0  # Should be slightly lower than the best bid that covers our amount
        assert abs(result["price"] - expected_price) < 0.01  # Should be within a small margin of error
        assert result["enough_liquidity"] == True
        
        # Check the batches - should have batches for each price level needed
        assert len(result["batches"]) >= 1
        
        # If multiple batches, check they're correctly calculated
        if len(result["batches"]) > 1:
            assert result["batches"][0]["price"] == 1900.0
            assert result["batches"][0]["amount"] == 1.5  # Our amount fits in the first price level
        
        assert result["slippage"] < 1.0  # Small order should have minimal slippage

def test_get_optimal_limit_price_large_buy_order(connector):
    """Test optimal limit price calculation for a large BUY order that spans multiple price levels."""
    # Mock the orderbook response with sample data
    mock_orderbook = {
        "bids": [[1900.0, 2.0], [1895.0, 3.0], [1890.0, 5.0]],
        "asks": [[1905.0, 1.0], [1910.0, 2.0], [1915.0, 3.0]]
    }
    
    with patch.object(connector, 'get_orderbook', return_value=mock_orderbook):
        result = connector.get_optimal_limit_price(
            symbol="ETH",
            side=OrderSide.BUY,
            amount=5.0  # This will span all 3 price levels on the ask side
        )
        
        # For a large BUY order, we need to walk the order book
        # With the buffer applied to the worst price (1915.0)
        expected_price = 1915.0 * 1.0005  # Worst price + 0.05% buffer
        
        assert result["price"] > 1915.0  # Should be higher than the highest ask needed
        assert abs(result["price"] - expected_price) < 0.01  # Should be within a small margin of error
        assert result["enough_liquidity"] == True  # We have enough in the orderbook to fill
        assert result["slippage"] > 0.5  # Larger order should have more slippage
        
        # We should have calculated the total cost properly
        expected_total_cost = (1905.0 * 1.0) + (1910.0 * 2.0) + (1915.0 * 2.0)
        assert abs(result["total_cost"] - expected_total_cost) < 0.01

def test_get_optimal_limit_price_insufficient_liquidity(connector):
    """Test optimal limit price calculation for an order that exceeds available liquidity."""
    # Mock the orderbook response with sample data
    mock_orderbook = {
        "bids": [[1900.0, 2.0], [1895.0, 3.0], [1890.0, 5.0]],
        "asks": [[1905.0, 1.0], [1910.0, 2.0], [1915.0, 3.0]]
    }
    
    with patch.object(connector, 'get_orderbook', return_value=mock_orderbook):
        result = connector.get_optimal_limit_price(
            symbol="ETH",
            side=OrderSide.BUY,
            amount=10.0  # This exceeds the 6.0 total available in the ask side
        )
        
        assert result["enough_liquidity"] == False
        assert result["price"] > 1915.0  # Should still provide a price based on worst ask
        
        # We should have calculated the total cost for the available liquidity
        expected_total_cost = (1905.0 * 1.0) + (1910.0 * 2.0) + (1915.0 * 3.0)
        assert abs(result["total_cost"] - expected_total_cost) < 0.01

def test_get_optimal_limit_price_empty_orderbook(connector):
    """Test handling of empty orderbook."""
    # Mock the orderbook response with empty data
    mock_orderbook = {
        "bids": [],
        "asks": []
    }
    
    # Mock the ticker response for fallback
    mock_ticker = {"last_price": 2000.0}
    
    with patch.object(connector, 'get_orderbook', return_value=mock_orderbook), \
         patch.object(connector, 'get_ticker', return_value=mock_ticker):
        result = connector.get_optimal_limit_price(
            symbol="ETH",
            side=OrderSide.BUY,
            amount=1.0
        )
        
        # Should fall back to ticker price
        assert result["price"] == 2000.0
        assert result["enough_liquidity"] == False
        assert len(result["batches"]) == 1
        assert result["batches"][0]["price"] == 2000.0
        assert result["batches"][0]["amount"] == 1.0

def test_get_optimal_limit_price_exception_handling(connector):
    """Test exception handling in the optimal limit price calculation."""
    # Force an exception in get_orderbook
    with patch.object(connector, 'get_orderbook', side_effect=Exception("Test exception")), \
         patch.object(connector, 'get_ticker', return_value={"last_price": 2000.0}):
        result = connector.get_optimal_limit_price(
            symbol="ETH",
            side=OrderSide.BUY,
            amount=1.0
        )
        
        # Should fall back to ticker price
        assert result["price"] == 2000.0
        assert result["enough_liquidity"] == False 