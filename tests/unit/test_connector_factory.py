import pytest
from unittest.mock import MagicMock, patch

from app.connectors.connector_factory import ConnectorFactory
from app.connectors.base_connector import BaseConnector


# Create a mock connector class for testing
class MockConnector(BaseConnector):
    def __init__(self, wallet_address=None, private_key=None, api_key=None, 
                 api_secret=None, testnet=False, rpc_url=None, name="mock_connector"):
        super().__init__(name=name, exchange_type="mock")
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.rpc_url = rpc_url
    
    # Implement abstract methods
    def connect(self): return True
    def disconnect(self): return True
    def get_account_balance(self): return {}
    def get_markets(self): return []
    def get_ticker(self, symbol): return {}
    def get_orderbook(self, symbol, depth=10): return {"bids": [], "asks": []}
    def get_positions(self): return []
    def place_order(self, symbol, side, order_type, amount, leverage, price=None): return {}
    def cancel_order(self, order_id): return {}
    def get_order(self, order_id): return {}
    def get_order_status(self, order_id): return {}
    def close_position(self, symbol, position_id=None): return {}
    def set_leverage(self, symbol, leverage): return {}
    def get_historical_candles(self, symbol, interval, start_time=None, end_time=None, limit=100): return []
    def get_funding_rate(self, symbol): return {}
    def get_leverage_tiers(self, symbol): return []


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
        testnet=True
    )
    
    # Verify the connector instance
    assert connector is not None
    assert isinstance(connector, MockConnector)
    assert connector.wallet_address == "0x123"
    assert connector.private_key == "abc"
    assert connector.testnet is True
    
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
        testnet=False
    )
    
    # Verify the connector instance
    assert connector is not None
    assert isinstance(connector, MockConnector)
    assert connector.api_key == "api_key_123"
    assert connector.api_secret == "api_secret_456"
    assert connector.testnet is False
    
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
            "testnet": True
        },
        {
            "name": "hedge_connector",
            "exchange_type": "mock",
            "api_key": "api_key_123",
            "api_secret": "api_secret_456",
            "testnet": False
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
    
    hedge_connector = connectors["hedge_connector"]
    assert isinstance(hedge_connector, MockConnector)
    assert hedge_connector.api_key == "api_key_123"
    assert hedge_connector.api_secret == "api_secret_456"
    assert hedge_connector.testnet is False
    
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