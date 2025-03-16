import logging
from typing import Dict, Any, Optional, List, Type

from app.connectors.base_connector import BaseConnector
from app.connectors.hyperliquid_connector import HyperliquidConnector
from app.connectors.coinbase_connector import CoinbaseConnector
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
        'coinbase': CoinbaseConnector,
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
                        api_passphrase: Optional[str] = None,
                        testnet: bool = True,
                        use_sandbox: bool = True,
                        rpc_url: Optional[str] = None) -> Optional[BaseConnector]:
        """
        Create a new exchange connector instance.
        
        Args:
            exchange_type: Type of exchange connector to create (e.g., 'hyperliquid', 'coinbase')
            wallet_address: Ethereum wallet address for on-chain exchanges
            private_key: Private key for signing on-chain transactions or API requests
            api_key: API key for exchanges that require it
            api_secret: API secret for exchanges that require it
            api_passphrase: API passphrase for exchanges that require it
            testnet: Whether to use testnet instead of mainnet (for blockchain-based exchanges)
            use_sandbox: Whether to use sandbox instead of production (for CEX APIs)
            rpc_url: Custom RPC URL for connecting to blockchain nodes
            
        Returns:
            Exchange connector instance or None if exchange_type is not supported
        """
        # Log the input parameters (without sensitive data)
        logger.info(f"Creating connector for exchange type: {exchange_type}")
        
        # Check if the exchange type is registered
        if exchange_type not in cls._connector_registry:
            logger.error(f"Unsupported exchange type: {exchange_type}")
            return None
        
        # Create the connector based on exchange type
        try:
            if exchange_type == 'hyperliquid':
                if not wallet_address or not private_key:
                    logger.error("Wallet address and private key are required for Hyperliquid connector")
                    return None
                    
                connector = cls._connector_registry[exchange_type](
                    wallet_address=wallet_address,
                    private_key=private_key,
                    testnet=testnet,
                    rpc_url=rpc_url
                )
                
            elif exchange_type == 'coinbase':
                if not api_key or not api_secret:
                    logger.error("API key and secret are required for Coinbase connector")
                    return None
                
                # For Coinbase, strip quotes if they're present in the keys
                if api_key and api_key.startswith('"') and api_key.endswith('"'):
                    api_key = api_key[1:-1]
                if api_secret and api_secret.startswith('"') and api_secret.endswith('"'):
                    api_secret = api_secret[1:-1]
                
                connector = cls._connector_registry[exchange_type](
                    api_key=api_key,
                    api_secret=api_secret,
                    use_sandbox=use_sandbox
                )
                
            else:
                # Generic case for other exchange types
                connector = cls._connector_registry[exchange_type](
                    api_key=api_key,
                    api_secret=api_secret,
                    api_passphrase=api_passphrase,
                    testnet=testnet
                )
                
            return connector
            
        except Exception as e:
            logger.error(f"Error creating {exchange_type} connector: {e}")
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
                api_passphrase = getattr(config, 'api_passphrase', None)
                testnet_value = config.testnet
                use_sandbox_value = getattr(config, 'use_sandbox', 'true')
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
                api_passphrase = config.get('api_passphrase')
                testnet_value = config.get('testnet', 'true')
                use_sandbox_value = config.get('use_sandbox', 'true')
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
                api_passphrase = getattr(config, 'api_passphrase', None)
                testnet_value = getattr(config, 'testnet', 'true')
                use_sandbox_value = getattr(config, 'use_sandbox', 'true')
                rpc_url = getattr(config, 'rpc_url', None)
            
            # Convert testnet and use_sandbox strings to boolean if necessary
            if isinstance(testnet_value, str):
                testnet = testnet_value.lower() in ('true', 'yes', '1', 't', 'y')
            else:
                testnet = bool(testnet_value)
                
            if isinstance(use_sandbox_value, str):
                use_sandbox = use_sandbox_value.lower() in ('true', 'yes', '1', 't', 'y')
            else:
                use_sandbox = bool(use_sandbox_value)
            
            # Convert enabled to boolean if it's a string
            if isinstance(enabled, str):
                enabled = enabled.lower() in ('true', 'yes', '1', 't', 'y')
                
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
                api_passphrase=api_passphrase,
                testnet=testnet,
                use_sandbox=use_sandbox,
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