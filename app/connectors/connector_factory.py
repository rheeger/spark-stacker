import logging
from typing import Dict, Any, Optional, List, Type

from app.connectors.base_connector import BaseConnector
from app.connectors.hyperliquid_connector import HyperliquidConnector
# Import additional connectors as they are implemented
# from app.connectors.synthetix_connector import SynthetixConnector

logger = logging.getLogger(__name__)

class ConnectorFactory:
    """
    Factory for creating exchange connector instances.
    
    This class provides methods to create and manage connector instances
    based on configuration parameters.
    """
    
    # Registry of available connector types
    _connector_registry: Dict[str, Type[BaseConnector]] = {
        'hyperliquid': HyperliquidConnector,
        # Add more connectors as they are implemented
        # 'synthetix': SynthetixConnector,
    }
    
    @classmethod
    def create_connector(cls, 
                        exchange_type: str, 
                        wallet_address: Optional[str] = None,
                        private_key: Optional[str] = None,
                        api_key: Optional[str] = None,
                        api_secret: Optional[str] = None,
                        testnet: bool = True,
                        rpc_url: Optional[str] = None) -> Optional[BaseConnector]:
        """
        Create a new exchange connector instance.
        
        Args:
            exchange_type: Type of exchange connector to create (e.g., 'hyperliquid', 'synthetix')
            wallet_address: Ethereum wallet address for on-chain exchanges
            private_key: Private key for signing on-chain transactions or API requests
            api_key: API key for exchanges that require it
            api_secret: API secret for exchanges that require it
            testnet: Whether to use testnet instead of mainnet
            rpc_url: Custom RPC URL for connecting to blockchain nodes
            
        Returns:
            Exchange connector instance or None if exchange_type is not supported
        """
        # Log the input parameters (without sensitive data)
        logger.info(f"Creating connector for exchange: '{exchange_type}'")
        logger.debug(f"Parameters: wallet_address={wallet_address is not None}, "
                    f"private_key={private_key is not None}, "
                    f"api_key={api_key is not None}, "
                    f"api_secret={api_secret is not None}, "
                    f"testnet={testnet}, "
                    f"rpc_url={rpc_url is not None}")
        
        # Normalize exchange_type
        if exchange_type is None:
            exchange_type = ""
        
        # Convert to lowercase for case-insensitive comparison
        exchange_type = exchange_type.lower().strip()
        
        if not exchange_type:
            logger.error(f"Empty exchange_type provided. Available types: {list(cls._connector_registry.keys())}")
            return None
        
        if exchange_type not in cls._connector_registry:
            logger.error(f"Exchange type '{exchange_type}' not supported. Available types: {list(cls._connector_registry.keys())}")
            return None
        
        try:
            # Get the appropriate connector class
            connector_class = cls._connector_registry[exchange_type]
            
            # Create connector with appropriate parameters
            if exchange_type == 'hyperliquid':
                logger.debug(f"Creating Hyperliquid connector with wallet={wallet_address}, testnet={testnet}")
                if not wallet_address or not private_key:
                    logger.error("Hyperliquid connector requires wallet_address and private_key")
                    return None
                
                connector = connector_class(
                    wallet_address=wallet_address,
                    private_key=private_key,
                    testnet=testnet,
                    rpc_url=rpc_url
                )
            # Add cases for other exchange types
            # elif exchange_type == 'synthetix':
            #     if not wallet_address or not private_key:
            #         logger.error("Synthetix connector requires wallet_address and private_key")
            #         return None
            #     
            #     connector = connector_class(
            #         wallet_address=wallet_address,
            #         private_key=private_key,
            #         testnet=testnet,
            #         rpc_url=rpc_url
            #     )
            # Handle 'mock' for testing
            elif exchange_type == 'mock':
                connector = connector_class(
                    wallet_address=wallet_address,
                    private_key=private_key,
                    api_key=api_key,
                    api_secret=api_secret,
                    testnet=testnet,
                    rpc_url=rpc_url
                )
            else:
                logger.error(f"Exchange type '{exchange_type}' found in registry but not handled in factory")
                return None
            
            logger.info(f"Successfully created connector for '{exchange_type}'")
            return connector
        
        except Exception as e:
            logger.error(f"Failed to create connector for '{exchange_type}': {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    @classmethod
    def create_connectors_from_config(cls, configs: List[Any]) -> Dict[str, BaseConnector]:
        """
        Create multiple connectors from configuration.
        
        Args:
            configs: List of exchange configurations
                Each config must have 'name' field and other required fields for that exchange
            
        Returns:
            Dictionary of connector instances, keyed by name
        """
        connectors = {}
        logger.info(f"Creating connectors from config. Found {len(configs)} configurations.")
        
        for config in configs:
            # Check if config is a dictionary, an ExchangeConfig object, or another type
            logger.debug(f"Processing config: {type(config)}")
            
            # Check if it's an ExchangeConfig object from app.utils.config
            if hasattr(config, 'name') and hasattr(config, 'exchange_type'):
                # Direct attribute access for ExchangeConfig objects
                name = config.name
                exchange_type = config.exchange_type.lower() if config.exchange_type else ""
                enabled = getattr(config, 'enabled', True)
                wallet_address = config.wallet_address
                private_key = config.private_key
                api_key = getattr(config, 'api_key', None)
                api_secret = getattr(config, 'api_secret', None)
                testnet_value = config.testnet
                rpc_url = config.rpc_url
            elif isinstance(config, dict):
                # Dictionary-style access
                name = config.get('name', '')
                exchange_type = config.get('exchange_type', '').lower()
                enabled = config.get('enabled', True)
                wallet_address = config.get('wallet_address')
                private_key = config.get('private_key')
                api_key = config.get('api_key')
                api_secret = config.get('api_secret')
                testnet_value = config.get('testnet', 'true')
                rpc_url = config.get('rpc_url')
            else:
                # Object-style access for other objects
                name = getattr(config, 'name', '')
                exchange_type = getattr(config, 'exchange_type', '').lower()
                enabled = getattr(config, 'enabled', True)
                wallet_address = getattr(config, 'wallet_address', None)
                private_key = getattr(config, 'private_key', None)
                api_key = getattr(config, 'api_key', None)
                api_secret = getattr(config, 'api_secret', None)
                testnet_value = getattr(config, 'testnet', 'true')
                rpc_url = getattr(config, 'rpc_url', None)
            
            # Convert testnet string to boolean if necessary
            if isinstance(testnet_value, str):
                testnet = testnet_value.lower() in ('true', 'yes', '1', 't', 'y')
            else:
                testnet = bool(testnet_value)
            
            logger.info(f"Connector config: name={name}, exchange_type={exchange_type}, enabled={enabled}, testnet={testnet}")
            
            if not name:
                logger.warning(f"Skipping invalid exchange config (missing name)")
                continue
            
            if not enabled:
                logger.info(f"Skipping disabled connector: {name}")
                continue
            
            connector = cls.create_connector(
                exchange_type=exchange_type,
                wallet_address=wallet_address,
                private_key=private_key,
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
                rpc_url=rpc_url
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
    def register_connector(cls, exchange_type: str, connector_class: Type[BaseConnector]) -> None:
        """
        Register a new connector type.
        
        Args:
            exchange_type: String identifier for the exchange type
            connector_class: Connector class to register
        """
        if not issubclass(connector_class, BaseConnector):
            raise TypeError(f"Class {connector_class.__name__} is not a subclass of BaseConnector")
        
        cls._connector_registry[exchange_type.lower()] = connector_class
        logger.info(f"Registered connector type: {exchange_type}")
    
    @classmethod
    def get_available_connectors(cls) -> List[str]:
        """
        Get list of available connector types.
        
        Returns:
            List of connector type names
        """
        return list(cls._connector_registry.keys()) 