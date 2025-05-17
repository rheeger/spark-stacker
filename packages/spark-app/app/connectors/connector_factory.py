import logging
import os
from typing import Any, Dict, List, Optional, Set, Type, Union

from app.connectors.base_connector import BaseConnector, MarketType
from app.connectors.coinbase_connector import CoinbaseConnector
from app.connectors.hyperliquid_connector import HyperliquidConnector
from app.connectors.kraken_connector import KrakenConnector

# Import additional connectors as they are implemented
# from .synthetix_connector import SynthetixConnector

logger = logging.getLogger(__name__)

# Default market type mappings for each exchange
DEFAULT_MARKET_TYPES = {
    "hyperliquid": [MarketType.PERPETUAL],
    "coinbase": [MarketType.SPOT],
    "kraken": [MarketType.SPOT, MarketType.PERPETUAL],
    # Add other exchanges and their default market types as needed
}


class ConnectorFactory:
    """
    Factory for creating exchange connector instances.

    This class provides methods to create and manage connector instances
    based on configuration parameters and supported market types.
    """

    # Registry of available connector types
    _connector_registry: Dict[str, Type[BaseConnector]] = {
        "hyperliquid": HyperliquidConnector,
        "coinbase": CoinbaseConnector,
        "kraken": KrakenConnector,
        # Add more connectors as they are implemented
        # 'synthetix': SynthetixConnector,
    }

    @classmethod
    def _is_test_environment(cls) -> bool:
        """Check if we're running in a test environment."""
        return os.environ.get('PYTEST_RUNNING', 'False').lower() in ('true', '1', 't')

    @classmethod
    def create_connector(
        cls,
        exchange_type: str,
        name: Optional[str] = None,
        wallet_address: Optional[str] = None,
        private_key: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        api_passphrase: Optional[str] = None,
        testnet: bool = True,
        use_sandbox: bool = True,
        rpc_url: Optional[str] = None,
        market_types: Optional[List[Union[str, MarketType]]] = None,
    ) -> Optional[BaseConnector]:
        """
        Create a new exchange connector instance.

        Args:
            exchange_type: Type of exchange connector to create (e.g., 'hyperliquid', 'coinbase')
            name: Optional custom name for the connector (defaults to exchange_type if not provided)
            wallet_address: Ethereum wallet address for on-chain exchanges
            private_key: Private key for signing on-chain transactions or API requests
            api_key: API key for exchanges that require it
            api_secret: API secret for exchanges that require it
            api_passphrase: API passphrase for exchanges that require it
            testnet: Whether to use testnet instead of mainnet (for blockchain-based exchanges)
            use_sandbox: Whether to use sandbox instead of production (for CEX APIs)
            rpc_url: Custom RPC URL for connecting to blockchain nodes
            market_types: List of market types this connector supports (SPOT, PERPETUAL, FUTURES)

        Returns:
            Exchange connector instance or None if exchange_type is not supported
        """
        # Use a default name if none is provided
        connector_name = name if name else exchange_type

        # Log the input parameters (without sensitive data)
        logger.info(
            f"Creating connector for exchange type: {exchange_type} with name: {connector_name}"
        )

        # Check if the exchange type is registered
        if exchange_type not in cls._connector_registry:
            logger.error(f"Unsupported exchange type: {exchange_type}")
            return None

        # Process market types
        processed_market_types = cls._process_market_types(exchange_type, market_types)
        logger.info(
            f"Connector will support market types: {[mt.value for mt in processed_market_types]}"
        )

        # Create the connector based on exchange type
        try:
            if exchange_type == "hyperliquid":
                if not wallet_address or not private_key:
                    logger.error(
                        "Wallet address and private key are required for Hyperliquid connector"
                    )
                    return None

                connector = cls._connector_registry[exchange_type](
                    name=connector_name,
                    wallet_address=wallet_address,
                    private_key=private_key,
                    testnet=testnet,
                    rpc_url=rpc_url,
                    market_types=processed_market_types,
                )

            elif exchange_type == "coinbase":
                if not api_key or not api_secret:
                    logger.error(
                        "API key and secret are required for Coinbase connector"
                    )
                    return None

                # For Coinbase, strip quotes if they're present in the keys
                if api_key and api_key.startswith('"') and api_key.endswith('"'):
                    api_key = api_key[1:-1]
                if (
                    api_secret
                    and api_secret.startswith('"')
                    and api_secret.endswith('"')
                ):
                    api_secret = api_secret[1:-1]

                connector = cls._connector_registry[exchange_type](
                    name=connector_name,
                    api_key=api_key,
                    api_secret=api_secret,
                    passphrase=api_passphrase,
                    testnet=use_sandbox,
                    market_types=processed_market_types,
                )

            elif exchange_type == "kraken":
                if not api_key or not api_secret:
                    logger.error(
                        "API key and secret are required for Kraken connector"
                    )
                    return None

                # For Kraken, strip quotes if they're present in the keys
                if api_key and api_key.startswith('"') and api_key.endswith('"'):
                    api_key = api_key[1:-1]
                if (
                    api_secret
                    and api_secret.startswith('"')
                    and api_secret.endswith('"')
                ):
                    api_secret = api_secret[1:-1]

                connector = cls._connector_registry[exchange_type](
                    name=connector_name,
                    api_key=api_key,
                    api_secret=api_secret,
                    testnet=use_sandbox,
                    market_types=processed_market_types,
                )

            else:
                # Generic case for other exchange types
                connector = cls._connector_registry[exchange_type](
                    name=connector_name,
                    wallet_address=wallet_address,
                    private_key=private_key,
                    api_key=api_key,
                    api_secret=api_secret,
                    api_passphrase=api_passphrase,
                    testnet=testnet,
                    use_sandbox=use_sandbox,
                    market_types=processed_market_types,
                )

            # Ensure connector has loggers set up
            if connector:
                # Set up dedicated loggers for this connector
                try:
                    # Skip logger setup if we're in a test environment
                    if cls._is_test_environment():
                        logger.debug(f"Skipping logger setup for {connector.name} in test environment")
                    # Check if loggers are already set up to avoid duplication
                    elif (
                        not hasattr(connector, "balance_logger")
                        or not hasattr(connector, "markets_logger")
                        or not hasattr(connector, "orders_logger")
                        or connector.balance_logger is None
                        or connector.markets_logger is None
                        or connector.orders_logger is None
                    ):
                        connector.setup_loggers()
                    else:
                        logger.debug(
                            f"Loggers already set up for {connector.name}, skipping setup"
                        )
                except Exception as e:
                    logger.warning(
                        f"Error setting up loggers for {connector.name}: {e}, continuing anyway"
                    )

            return connector

        except Exception as e:
            logger.error(f"Error creating {exchange_type} connector: {e}")
            return None

    @classmethod
    def create_connectors_from_config(
        cls, configs: List[Any]
    ) -> Dict[str, BaseConnector]:
        """
        Create multiple connectors from configuration.

        Args:
            configs: List of exchange configurations
                Each config must have 'name' field and other required fields for that exchange

        Returns:
            Dictionary of connector instances, keyed by name
        """
        connectors = {}
        logger.info(
            f"Creating connectors from config. Found {len(configs)} configurations."
        )

        for config in configs:
            # Check if config is a dictionary, an ExchangeConfig object, or another type
            logger.debug(f"Processing config: {type(config)}")

            # Check if it's an ExchangeConfig object from utils.config
            if hasattr(config, "name") and hasattr(config, "exchange_type"):
                # Direct attribute access for ExchangeConfig objects
                name = config.name
                exchange_type = (
                    config.exchange_type.lower() if config.exchange_type else ""
                )
                enabled = getattr(config, "enabled", True)
                wallet_address = config.wallet_address
                private_key = config.private_key
                api_key = getattr(config, "api_key", None)
                api_secret = getattr(config, "api_secret", None)
                api_passphrase = getattr(config, "api_passphrase", None)
                testnet_value = config.testnet
                use_sandbox_value = getattr(config, "use_sandbox", "true")
                rpc_url = config.rpc_url
                market_types = getattr(config, "market_types", None)
            elif isinstance(config, dict):
                # Dictionary-style access
                name = config.get("name", "")
                exchange_type = config.get("exchange_type", "").lower()
                enabled = config.get("enabled", True)
                wallet_address = config.get("wallet_address")
                private_key = config.get("private_key")
                api_key = config.get("api_key")
                api_secret = config.get("api_secret")
                api_passphrase = config.get("api_passphrase")
                testnet_value = config.get("testnet", "true")
                use_sandbox_value = config.get("use_sandbox", "true")
                rpc_url = config.get("rpc_url")
                market_types = config.get("market_types")
            else:
                # Object-style access for other objects
                name = getattr(config, "name", "")
                exchange_type = getattr(config, "exchange_type", "").lower()
                enabled = getattr(config, "enabled", True)
                wallet_address = getattr(config, "wallet_address", None)
                private_key = getattr(config, "private_key", None)
                api_key = getattr(config, "api_key", None)
                api_secret = getattr(config, "api_secret", None)
                api_passphrase = getattr(config, "api_passphrase", None)
                testnet_value = getattr(config, "testnet", "true")
                use_sandbox_value = getattr(config, "use_sandbox", "true")
                rpc_url = getattr(config, "rpc_url", None)
                market_types = getattr(config, "market_types", None)

            # Convert testnet and use_sandbox strings to boolean if necessary
            if isinstance(testnet_value, str):
                testnet = testnet_value.lower() in ("true", "yes", "1", "t", "y")
            else:
                testnet = bool(testnet_value)

            if isinstance(use_sandbox_value, str):
                use_sandbox = use_sandbox_value.lower() in (
                    "true",
                    "yes",
                    "1",
                    "t",
                    "y",
                )
            else:
                use_sandbox = bool(use_sandbox_value)

            # Convert enabled to boolean if it's a string
            if isinstance(enabled, str):
                enabled = enabled.lower() in ("true", "yes", "1", "t", "y")

            logger.info(
                f"Connector config: name={name}, exchange_type={exchange_type}, enabled={enabled}, testnet={testnet}"
            )

            if not name:
                logger.warning(f"Skipping invalid exchange config (missing name)")
                continue

            if not enabled:
                logger.info(f"Skipping disabled connector: {name}")
                continue

            connector = cls.create_connector(
                exchange_type=exchange_type,
                name=name,
                wallet_address=wallet_address,
                private_key=private_key,
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
                testnet=testnet,
                use_sandbox=use_sandbox,
                rpc_url=rpc_url,
                market_types=market_types,
            )

            if connector:
                # Attempt to connect
                try:
                    if connector.connect():
                        connectors[name] = connector
                        logger.info(f"Connected to exchange: {name}")
                    else:
                        logger.error(f"Failed to connect to exchange: {name}")
                except Exception as e:
                    logger.error(f"Error connecting to exchange {name}: {e}")
            else:
                logger.error(f"Failed to create connector for exchange: {name}")

        return connectors

    @classmethod
    def register_connector(
        cls, exchange_type: str, connector_class: Type[BaseConnector]
    ) -> None:
        """
        Register a new connector type.

        Args:
            exchange_type: String identifier for the exchange type
            connector_class: Connector class to register
        """
        cls._connector_registry[exchange_type.lower()] = connector_class
        logger.info(f"Registered connector for exchange type: {exchange_type}")

    @classmethod
    def get_available_connectors(cls) -> List[str]:
        """
        Get a list of all available connector types.

        Returns:
            List of registered connector type strings
        """
        return list(cls._connector_registry.keys())

    @classmethod
    def _process_market_types(
        cls, exchange_type: str, market_types: Optional[List[Union[str, MarketType]]]
    ) -> List[MarketType]:
        """
        Process market types list into standardized MarketType enum values.

        Args:
            exchange_type: Type of exchange
            market_types: List of market types (strings or MarketType enums)

        Returns:
            List of MarketType enum values
        """
        if not market_types:
            # Use default market types for this exchange
            return DEFAULT_MARKET_TYPES.get(exchange_type.lower(), [MarketType.SPOT])

        processed_types = []
        for mt in market_types:
            if isinstance(mt, MarketType):
                processed_types.append(mt)
            elif isinstance(mt, str):
                # Try to convert string to MarketType enum
                try:
                    if mt.upper() in [t.value for t in MarketType]:
                        processed_types.append(MarketType(mt.upper()))
                    else:
                        logger.warning(f"Unknown market type: {mt}, ignoring")
                except ValueError:
                    logger.warning(f"Invalid market type string: {mt}, ignoring")

        # If no valid market types were processed, use defaults
        if not processed_types:
            return DEFAULT_MARKET_TYPES.get(exchange_type.lower(), [MarketType.SPOT])

        return processed_types

    @classmethod
    def get_supported_market_types(cls, exchange_type: str) -> List[MarketType]:
        """
        Get default supported market types for an exchange.

        Args:
            exchange_type: Exchange type identifier

        Returns:
            List of supported MarketType enum values
        """
        return DEFAULT_MARKET_TYPES.get(exchange_type.lower(), [MarketType.SPOT])
