#!/usr/bin/env python3

import logging
import os
import signal as signal_module
import sys
import time
import json
from typing import Dict, List, Any, Optional
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup basic logging first so we can see errors during startup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import core components
from app.utils.config import ConfigManager, AppConfig
from app.utils.logging_setup import setup_logging
from app.connectors.connector_factory import ConnectorFactory
from app.indicators.indicator_factory import IndicatorFactory
from app.risk_management.risk_manager import RiskManager
from app.core.trading_engine import TradingEngine
from app.core.strategy_manager import StrategyManager
from app.webhook.webhook_server import WebhookServer

# Global variables
engine: Optional[TradingEngine] = None
webhook_server: Optional[WebhookServer] = None
is_running = True

# Global logging flags
SHOW_MARKET_DETAILS = False
SHOW_ZERO_BALANCES = False

def signal_handler(sig, frame):
    """Handle system signals like CTRL+C"""
    global is_running
    logger.info("Shutdown signal received, stopping services...")
    is_running = False

def initialize_logging(config: Dict[str, Any]) -> None:
    """Set up application logging based on configuration"""
    try:
        log_level = config.get("log_level", "INFO")
        logger.info(f"Setting up logging with level: {log_level}")
        
        # Get logging config
        logging_config = config.get("logging", {})
        
        # Set up logging with detailed configuration
        setup_logging(
            log_level=log_level,
            log_to_file=logging_config.get("log_to_file", True),
            structured=True,
            enable_console=logging_config.get("enable_console", True)
        )
        
        # Set custom log levels for specific components
        connector_log_level = logging_config.get("connector_log_level", "WARNING")
        connector_log_level_int = getattr(logging, connector_log_level.upper(), logging.WARNING)
        logging.getLogger("app.connectors").setLevel(connector_log_level_int)
        
        # Store logging preferences in global state for use elsewhere
        global SHOW_MARKET_DETAILS, SHOW_ZERO_BALANCES
        SHOW_MARKET_DETAILS = logging_config.get("show_market_details", False)
        SHOW_ZERO_BALANCES = logging_config.get("show_zero_balances", False)
        
        logger.info("Logging initialized successfully")
        logger.info("Connector balance debug logs will be redirected to dedicated log files")
    except Exception as e:
        logger.error(f"Failed to initialize logging: {str(e)}", exc_info=True)
        # Continue with basic logging already set up

def create_exchange_connectors(config: AppConfig) -> Dict[str, Any]:
    """Create and initialize exchange connectors from configuration"""
    logger.info("Initializing exchange connectors...")
    
    # Debug-log the exchange configurations
    for idx, ex_config in enumerate(config.exchanges):
        logger.info(f"Exchange config #{idx+1}: name={ex_config.name}, type={ex_config.exchange_type}, enabled={ex_config.enabled}, use_as_main={ex_config.use_as_main}")
    
    # Pass the ExchangeConfig objects directly to the ConnectorFactory
    connectors = ConnectorFactory.create_connectors_from_config(config.exchanges)
    
    if not connectors:
        logger.error("Failed to create any exchange connectors")
        return {}
    
    # Identify main and hedge connectors based on config
    main_connector = None
    hedge_connector = None
    
    for exchange_name, connector in connectors.items():
        for exchange_config in config.exchanges:
            if exchange_config.name.lower() == exchange_name.lower():
                if getattr(exchange_config, 'use_as_main', False):
                    main_connector = connector
                    logger.info(f"Using {exchange_name} as main connector")
                if getattr(exchange_config, 'use_as_hedge', False):
                    hedge_connector = connector
                    logger.info(f"Using {exchange_name} as hedge connector")
    
    # If no main connector is specified, use the first one
    if not main_connector and connectors:
        main_connector = next(iter(connectors.values()))
        logger.info(f"No main connector specified, using first available connector")
    
    # If no hedge connector is specified, use the main connector
    if not hedge_connector and main_connector:
        hedge_connector = main_connector
        logger.info(f"No dedicated hedge connector specified, using main connector for hedging")
    
    return {
        'connectors': connectors,
        'main_connector': main_connector,
        'hedge_connector': hedge_connector
    }

def create_indicators(config: AppConfig) -> Dict[str, Any]:
    """Create technical indicators from configuration"""
    logger.info("Initializing technical indicators...")
    indicators = IndicatorFactory.create_indicators_from_config(config.indicators)
    
    if not indicators:
        logger.warning("No indicators created from configuration")
    
    return indicators

def create_risk_manager(config: AppConfig) -> RiskManager:
    """Create and configure the risk management system"""
    logger.info("Initializing risk management system...")
    
    # Default risk management parameters
    max_account_risk_pct = 2.0
    max_leverage = 25.0
    max_position_size_usd = None
    max_positions = config.max_parallel_trades
    min_margin_buffer_pct = 20.0
    
    # Apply any custom risk parameters from config here if needed
    
    return RiskManager(
        max_account_risk_pct=max_account_risk_pct,
        max_leverage=max_leverage,
        max_position_size_usd=max_position_size_usd,
        max_positions=max_positions,
        min_margin_buffer_pct=min_margin_buffer_pct
    )

def create_trading_engine(
    config: AppConfig,
    main_connector: Any,
    hedge_connector: Any,
    risk_manager: RiskManager
) -> TradingEngine:
    """Create and configure the main trading engine"""
    logger.info("Initializing trading engine...")
    
    if not main_connector:
        raise ValueError("Cannot create trading engine: No main connector available")
    
    engine = TradingEngine(
        main_connector=main_connector,
        hedge_connector=hedge_connector,
        risk_manager=risk_manager,
        dry_run=config.dry_run,
        polling_interval=config.polling_interval,
        max_parallel_trades=config.max_parallel_trades
    )
    
    return engine

def setup_webhook_server(config: AppConfig, trading_engine: TradingEngine) -> Optional[WebhookServer]:
    """Set up webhook server if enabled in config"""
    if not config.webhook_enabled:
        logger.info("Webhook server disabled in configuration")
        return None
    
    logger.info(f"Setting up webhook server on {config.webhook_host}:{config.webhook_port}...")
    
    # Define the signal handler function
    def handle_webhook_signal(signal):
        logger.info(f"Received signal from webhook: {signal}")
        trading_engine.process_signal(signal)
    
    # Create and start the webhook server
    server = WebhookServer(
        host=config.webhook_host,
        port=config.webhook_port,
        signal_handlers=[handle_webhook_signal]
    )
    
    if server.start():
        logger.info(f"Webhook server started at http://{config.webhook_host}:{config.webhook_port}")
        return server
    else:
        logger.error("Failed to start webhook server")
        return None

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json file."""
    try:
        config_file = os.environ.get("CONFIG_FILE", "config.json")
        logger.info(f"Loading configuration from {config_file}")
        
        with open(config_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file {config_file} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        sys.exit(1)

def create_connector(factory: ConnectorFactory, exchange_config: Dict[str, Any]) -> Optional[Any]:
    """
    Create an exchange connector from configuration.
    
    Args:
        factory: The connector factory instance
        exchange_config: Exchange configuration dictionary
    
    Returns:
        Exchange connector instance or None if creation fails
    """
    try:
        # Extract required parameters
        exchange_type = exchange_config.get('exchange_type', '').lower()
        
        # Process environment variables
        def get_env_value(config_value):
            if isinstance(config_value, str) and config_value.startswith('${') and config_value.endswith('}'):
                # Extract environment variable name and get its value
                env_var = config_value.strip('${}')
                return os.getenv(env_var)
            return config_value
            
        api_key = get_env_value(exchange_config.get('api_key', ''))
        api_secret = get_env_value(exchange_config.get('api_secret', ''))
        wallet_address = get_env_value(exchange_config.get('wallet_address', ''))
        private_key = get_env_value(exchange_config.get('private_key', ''))
        
        # Convert boolean configs from strings if needed
        def parse_bool_config(value):
            if isinstance(value, str):
                if value.startswith('${') and value.endswith('}'):
                    env_var = value.strip('${}')
                    env_value = os.getenv(env_var, 'false')
                    return env_value.lower() in ('true', 'yes', '1', 't', 'y')
                return value.lower() in ('true', 'yes', '1', 't', 'y')
            return bool(value)
            
        testnet = parse_bool_config(exchange_config.get('testnet', True))
        use_sandbox = parse_bool_config(exchange_config.get('use_sandbox', True))
        
        logger.debug(f"Creating connector for {exchange_type} with parsed config values, sandbox={use_sandbox}")
        
        # Create the connector with extracted parameters
        connector = factory.create_connector(
            exchange_type=exchange_type,
            name=exchange_config.get('name', None),
            wallet_address=wallet_address,
            private_key=private_key,
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            use_sandbox=use_sandbox
        )
        
        return connector
    except Exception as e:
        logger.error(f"Failed to create connector: {str(e)}", exc_info=True)
        return None

def main():
    """Main application entry point."""
    try:
        # Register signal handlers for graceful shutdown
        signal_module.signal(signal_module.SIGINT, signal_handler)
        signal_module.signal(signal_module.SIGTERM, signal_handler)
        
        # Load configuration
        config = load_config()
        
        # Now initialize proper logging with configuration
        initialize_logging(config)
        
        logger.info("Initializing Spark Stacker...")
        logger.info("Configuration loaded successfully")
        
        # Initialize exchange connectors
        connector_factory = ConnectorFactory()
        exchange_connectors = {}
        main_connector = None
        hedge_connector = None
        
        for exchange_config in config["exchanges"]:
            if not exchange_config.get("enabled", False):
                logger.info(f"Exchange {exchange_config['name']} is disabled, skipping...")
                continue
                
            try:
                connector = create_connector(connector_factory, exchange_config)
                if connector:
                    exchange_connectors[exchange_config["name"]] = connector
                    logger.info(f"Successfully initialized {exchange_config['name']} connector")
                    
                    # Set main and hedge connectors based on configuration
                    if exchange_config.get("use_as_main", False):
                        main_connector = connector
                    if exchange_config.get("use_as_hedge", False):
                        hedge_connector = connector
                        
                    # Get available markets
                    markets = connector.get_markets()
                    logger.info(f"Available markets on {exchange_config['name']}: {len(markets)} markets found")
                    
                    # Always log markets to the dedicated markets logger
                    for market in markets:
                        connector.markets_logger.info(f"Market: {market['symbol']} - {market}")
                    
                    # Only log market details to main log if flag is enabled
                    if SHOW_MARKET_DETAILS:
                        for market in markets:
                            logger.info(f"  {market['symbol']}: {market}")
                    
                    # Check account balances
                    try:
                        balances = connector.get_account_balance()
                        non_zero_balances = {k: v for k, v in balances.items() if v > 0}
                        logger.info(f"Account balances on {exchange_config['name']}: {len(non_zero_balances)} non-zero balances found")
                        
                        # Log balances based on configuration
                        if SHOW_ZERO_BALANCES:
                            # Log all balances
                            for currency, amount in balances.items():
                                connector.balance_logger.info(f"{currency}: {amount}")
                                logger.info(f"  {currency}: {amount}")
                        else:
                            # Log only non-zero balances
                            for currency, amount in non_zero_balances.items():
                                connector.balance_logger.info(f"{currency}: {amount}")
                                logger.info(f"  {currency}: {amount}")
                    except Exception as e:
                        logger.error(f"Failed to get balances: {str(e)}", exc_info=True)
                        
                else:
                    if exchange_config.get("use_as_main", False):
                        logger.error("Failed to initialize main exchange, exiting...")
                        sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to initialize {exchange_config['name']} connector: {str(e)}", exc_info=True)
                if exchange_config.get("use_as_main", False):
                    logger.error("Failed to initialize main exchange, exiting...")
                    sys.exit(1)
        
        # If no main connector is set, use the first available one
        if not main_connector and exchange_connectors:
            main_connector = next(iter(exchange_connectors.values()))
            logger.info("No main connector specified, using first available connector")
        
        # If no hedge connector is set, use the main connector
        if not hedge_connector:
            hedge_connector = main_connector
            logger.info("No hedge connector specified, using main connector for hedging")
        
        # Initialize risk manager with conservative settings for sandbox
        risk_manager = RiskManager(
            max_account_risk_pct=1.0,  # Lower risk percentage
            max_leverage=1.0,  # No leverage in sandbox
            max_position_size_usd=10.0,  # Small position size
            max_positions=1,
            min_margin_buffer_pct=50.0  # Higher margin buffer
        )
        
        # Initialize trading engine with proper parameters
        engine = TradingEngine(
            main_connector=main_connector,
            hedge_connector=hedge_connector,
            risk_manager=risk_manager,
            dry_run=config.get("dry_run", True),
            polling_interval=config.get("polling_interval", 60),
            max_parallel_trades=1  # Limit to 1 trade at a time for testing
        )
        
        # Initialize strategy manager with indicators
        indicators = IndicatorFactory.create_indicators_from_config(config.get("indicators", []))
        
        if not indicators:
            logger.warning("No indicators configured, trading system will only receive signals from webhooks")
        else:
            logger.info(f"Loaded {len(indicators)} indicator(s): {', '.join(indicators.keys())}")
        
        strategy_manager = StrategyManager(
            trading_engine=engine,
            indicators=indicators
        )
        
        # Start the trading engine
        if not engine.start():
            logger.error("Failed to start trading engine")
            sys.exit(1)
            
        # Main trading loop
        try:
            logger.info("Starting main trading loop")
            while is_running:
                try:
                    # Run strategy cycle to check for signals from indicators
                    if indicators:
                        signal_count = strategy_manager.run_cycle()
                        if signal_count > 0:
                            logger.info(f"Generated and processed {signal_count} signals in this cycle")
                except Exception as e:
                    logger.error(f"Error in strategy cycle: {str(e)}", exc_info=True)
                    
                # Wait for next cycle
                time.sleep(config.get("polling_interval", 60))
                
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            raise
        finally:
            # Cleanup
            logger.info("Cleaning up resources")
            engine.stop()
            for connector in exchange_connectors.values():
                connector.cleanup()
    except Exception as e:
        logger.error(f"Fatal error during startup: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 