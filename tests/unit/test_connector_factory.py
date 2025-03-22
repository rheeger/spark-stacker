import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Optional, Any

from app.connectors.connector_factory import ConnectorFactory
from app.connectors.base_connector import BaseConnector
from app.models.order import OrderSide, OrderType, OrderStatus


# Create a mock connector class for testing
class MockConnector(BaseConnector):
    """Mock connector for testing."""
    
    def __init__(self,
                 name: str = "mock",
                 exchange_type: str = "mock",
                 wallet_address: Optional[str] = None,
                 private_key: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 api_passphrase: Optional[str] = None,
                 testnet: bool = False,
                 use_sandbox: bool = False):
        """Initialize the mock connector."""
        super().__init__(name=name, exchange_type=exchange_type)
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.testnet = testnet
        self.use_sandbox = use_sandbox
        self.client = None
    
    def connect(self) -> bool:
        """Mock connection."""
        self.client = True
        return True
    
    def disconnect(self) -> bool:
        """Mock disconnection."""
        self.client = None
        return True
    
    def get_markets(self) -> List[Dict[str, any]]:
        """Mock get markets."""
        return [
            {"symbol": "ETH-USD", "base": "ETH", "quote": "USD"},
            {"symbol": "BTC-USD", "base": "BTC", "quote": "USD"}
        ]
    
    def get_ticker(self, symbol: str) -> Dict[str, any]:
        """Mock get ticker."""
        return {
            "symbol": f"{symbol}-USD",
            "last_price": 2500.0,
            "bid": 2499.0,
            "ask": 2501.0,
            "volume": 1000.0
        }
    
    def get_orderbook(self, symbol: str) -> Dict[str, any]:
        """Mock get orderbook."""
        return {
            "bids": [[2000.0, 1.0], [1999.0, 1.0]],
            "asks": [[2001.0, 1.0], [2002.0, 1.0]]
        }
    
    def get_account_balance(self) -> Dict[str, float]:
        """Mock get account balance."""
        return {
            "USD": 10000.0,
            "ETH": 5.0,
            "BTC": 0.5
        }
    
    def get_positions(self) -> List[Dict[str, any]]:
        """Mock get positions."""
        return [
            {"symbol": "ETH-USD", "size": 1.0, "entry_price": 2500.0},
            {"symbol": "BTC-USD", "size": 0.1, "entry_price": 50000.0},
            {"symbol": "SOL-USD", "size": 10.0, "entry_price": 100.0}
        ]
    
    def place_order(self,
                   symbol: str,
                   side: OrderSide,
                   order_type: OrderType,
                   amount: float,
                   leverage: float,
                   price: Optional[float] = None) -> Dict[str, any]:
        """Mock place order."""
        return {
            "order_id": "test-order-id",
            "status": OrderStatus.OPEN.value,
            "symbol": f"{symbol}-USD",
            "side": side.value,
            "type": order_type.value,
            "amount": amount,
            "price": price
        }
    
    def cancel_order(self, order_id: str) -> bool:
        """Mock cancel order."""
        return True
    
    def get_order(self, order_id: str) -> Dict[str, any]:
        """Mock get order."""
        return {
            "order_id": order_id,
            "status": OrderStatus.OPEN.value,
            "symbol": "ETH-USD",
            "side": OrderSide.BUY.value,
            "type": OrderType.LIMIT.value,
            "amount": 1.0,
            "price": 2500.0
        }
    
    def close_position(self, symbol: str) -> Dict[str, any]:
        """Mock close position."""
        return {"success": True, "message": "Position closed"}
    
    def get_historical_candles(self,
                             symbol: str,
                             interval: str,
                             start_time: int,
                             end_time: int,
                             limit: int = 1000) -> List[Dict[str, any]]:
        """Mock get historical candles."""
        return [{
            "timestamp": start_time,
            "open": 2500.0,
            "high": 2550.0,
            "low": 2450.0,
            "close": 2525.0,
            "volume": 1000.0
        }]
    
    def convert_interval_to_granularity(self, interval: str) -> int:
        """Mock convert interval to granularity."""
        return 3600  # 1 hour in seconds
    
    def get_funding_rate(self, symbol: str) -> float:
        """Mock get funding rate."""
        return 0.0
    
    def get_leverage_tiers(self, symbol: str) -> List[Dict[str, any]]:
        """Mock get leverage tiers."""
        return []
    
    def set_leverage(self, symbol: str, leverage: float) -> Dict[str, any]:
        """Mock set leverage."""
        return {
            "success": False,
            "message": "not supported"
        }
    
    def get_order_status(self, order_id: str) -> str:
        """Mock get order status."""
        return OrderStatus.OPEN.value


def test_connector_registry():
    """Test the connector registry contains the expected connectors."""
    # Get the available connectors
    available_connectors = ConnectorFactory.get_available_connectors()
    
    # Verify that we have the HyperliquidConnector registered
    assert "hyperliquid" in available_connectors


def test_register_connector():
    """Test registering a new connector."""
    # Register the mock connector
    ConnectorFactory.register_connector("mock", MockConnector)
    
    # Verify it was added to the registry
    available_connectors = ConnectorFactory.get_available_connectors()
    assert "mock" in available_connectors
    
    # Clean up after the test
    if "mock" in ConnectorFactory._connector_registry:
        del ConnectorFactory._connector_registry["mock"]


def test_register_invalid_connector():
    """Test registering an invalid connector class."""
    class InvalidClass:
        pass
    
    # Attempt to register a class that doesn't inherit from BaseConnector
    with pytest.raises(TypeError):
        ConnectorFactory.register_connector("invalid", InvalidClass)


def test_create_connector():
    """Test creating a connector instance."""
    # Register the mock connector
    ConnectorFactory.register_connector("mock", MockConnector)
    
    # Create a connector instance
    connector = ConnectorFactory.create_connector(
        exchange_type="mock",
        wallet_address="0x123",
        private_key="abc",
        testnet=True,
        use_sandbox=True
    )
    
    # Verify the connector instance
    assert connector is not None
    assert isinstance(connector, MockConnector)
    assert connector.wallet_address == "0x123"
    assert connector.private_key == "abc"
    assert connector.testnet is True
    assert connector.use_sandbox is True
    
    # Test creating a connector with invalid type
    connector = ConnectorFactory.create_connector(
        exchange_type="nonexistent",
        wallet_address="0x123"
    )
    assert connector is None
    
    # Clean up after the test
    if "mock" in ConnectorFactory._connector_registry:
        del ConnectorFactory._connector_registry["mock"]


def test_create_connector_with_api_credentials():
    """Test creating a connector with API credentials."""
    # Register the mock connector
    ConnectorFactory.register_connector("mock", MockConnector)
    
    # Create a connector instance with API credentials
    connector = ConnectorFactory.create_connector(
        exchange_type="mock",
        api_key="api_key_123",
        api_secret="api_secret_456",
        api_passphrase="test_passphrase",
        testnet=False,
        use_sandbox=False
    )
    
    # Verify the connector instance
    assert connector is not None
    assert isinstance(connector, MockConnector)
    assert connector.api_key == "api_key_123"
    assert connector.api_secret == "api_secret_456"
    assert connector.api_passphrase == "test_passphrase"
    assert connector.testnet is False
    assert connector.use_sandbox is False
    
    # Clean up after the test
    if "mock" in ConnectorFactory._connector_registry:
        del ConnectorFactory._connector_registry["mock"]


def test_create_connectors_from_config():
    """Test creating multiple connectors from a configuration."""
    # Register the mock connector
    ConnectorFactory.register_connector("mock", MockConnector)
    
    # Create a configuration list
    config = [
        {
            "name": "main_connector",
            "exchange_type": "mock",
            "wallet_address": "0x123",
            "private_key": "abc",
            "testnet": True,
            "use_sandbox": True,
            "enabled": True
        },
        {
            "name": "hedge_connector",
            "exchange_type": "mock",
            "api_key": "api_key_123",
            "api_secret": "api_secret_456",
            "api_passphrase": "test_passphrase",
            "testnet": False,
            "use_sandbox": False,
            "enabled": True
        },
        {
            "exchange_type": "nonexistent",  # This one should be skipped (no name)
            "wallet_address": "0x456"
        },
        {
            "name": "disabled_connector",  # This one should be skipped (disabled)
            "exchange_type": "mock",
            "enabled": False,
            "wallet_address": "0x789"
        }
    ]
    
    # Create connectors from the configuration
    connectors = ConnectorFactory.create_connectors_from_config(config)
    
    # Verify the connectors
    assert len(connectors) == 2
    assert "main_connector" in connectors
    assert "hedge_connector" in connectors
    
    main_connector = connectors["main_connector"]
    assert isinstance(main_connector, MockConnector)
    assert main_connector.wallet_address == "0x123"
    assert main_connector.private_key == "abc"
    assert main_connector.testnet is True
    assert main_connector.use_sandbox is True
    
    hedge_connector = connectors["hedge_connector"]
    assert isinstance(hedge_connector, MockConnector)
    assert hedge_connector.api_key == "api_key_123"
    assert hedge_connector.api_secret == "api_secret_456"
    assert hedge_connector.api_passphrase == "test_passphrase"
    assert hedge_connector.testnet is False
    assert hedge_connector.use_sandbox is False
    
    # Verify that the disabled connector and the one without a name were skipped
    assert "disabled_connector" not in connectors
    
    # Clean up after the test
    if "mock" in ConnectorFactory._connector_registry:
        del ConnectorFactory._connector_registry["mock"]


def test_get_available_connectors():
    """Test getting the list of available connectors."""
    # Get the initial list of available connectors
    initial_connectors = ConnectorFactory.get_available_connectors()
    
    # Register a temporary connector
    ConnectorFactory.register_connector("temp_mock", MockConnector)
    
    # Get the updated list of available connectors
    updated_connectors = ConnectorFactory.get_available_connectors()
    
    # Verify that the new connector is in the list
    assert "temp_mock" in updated_connectors
    assert len(updated_connectors) == len(initial_connectors) + 1
    
    # Clean up after the test
    if "temp_mock" in ConnectorFactory._connector_registry:
        del ConnectorFactory._connector_registry["temp_mock"] 