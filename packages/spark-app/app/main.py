#!/usr/bin/env python3

import argparse
import asyncio
import json
import logging
import os
import re
import signal as signal_module
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

from app.connectors.connector_factory import ConnectorFactory
from app.core.strategy_manager import StrategyManager
from app.core.trading_engine import TradingEngine
from app.indicators.base_indicator import Signal, SignalDirection
from app.indicators.indicator_factory import IndicatorFactory
from app.metrics import start_metrics_server, update_mvp_signal_state
from app.metrics.registry import start_historical_data_api
from app.risk_management.risk_manager import RiskManager
from app.utils.config import AppConfig, ConfigManager
from app.utils.logging_setup import setup_logging
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


def _validate_strategy_indicators(strategies: List[Dict[str, Any]], indicators: Dict[str, Any]) -> None:
    """
    Validate strategy-indicator relationships and configuration requirements.

    Args:
        strategies: List of strategy configurations
        indicators: Dictionary of loaded indicators

    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating strategy-indicator relationships...")

    for strategy in strategies:
        strategy_name = strategy.get("name", "unknown")
        logger.debug(f"Validating strategy: {strategy_name}")

        # Validate market symbol format (must contain "-")
        market = strategy.get("market", "")
        if not market or "-" not in market:
            raise ValueError(f"Strategy '{strategy_name}' has invalid market format: '{market}'. Must contain '-' (e.g., 'ETH-USD')")

        # Validate exchange field is present
        exchange = strategy.get("exchange", "")
        if not exchange:
            raise ValueError(f"Strategy '{strategy_name}' missing required 'exchange' field")

        # Validate indicators list exists and is not empty
        strategy_indicators = strategy.get("indicators", [])
        if not strategy_indicators:
            raise ValueError(f"Strategy '{strategy_name}' has no indicators specified")

        # Validate all strategy indicators exist in loaded indicators
        for indicator_name in strategy_indicators:
            if indicator_name not in indicators:
                raise ValueError(f"Strategy '{strategy_name}' references unknown indicator: '{indicator_name}'")

        logger.debug(f"Strategy '{strategy_name}' validation passed: market={market}, exchange={exchange}, indicators={strategy_indicators}")

    logger.info(f"Successfully validated {len(strategies)} strategies")


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
            enable_console=logging_config.get("enable_console", True),
        )

        # Set custom log levels for specific components
        connector_log_level = logging_config.get("connector_log_level", "WARNING")
        connector_log_level_int = getattr(
            logging, connector_log_level.upper(), logging.WARNING
        )
        logging.getLogger("connectors").setLevel(connector_log_level_int)

        # Store logging preferences in global state for use elsewhere
        global SHOW_MARKET_DETAILS, SHOW_ZERO_BALANCES
        SHOW_MARKET_DETAILS = logging_config.get("show_market_details", False)
        SHOW_ZERO_BALANCES = logging_config.get("show_zero_balances", False)

        logger.info("Logging initialized successfully")
        logger.info(
            "Connector balance debug logs will be redirected to dedicated log files"
        )
    except Exception as e:
        logger.error(f"Failed to initialize logging: {str(e)}", exc_info=True)
        # Continue with basic logging already set up


def create_exchange_connectors(config: AppConfig) -> Dict[str, Any]:
    """Create and initialize exchange connectors from configuration"""
    logger.info("Initializing exchange connectors...")

    # Debug-log the exchange configurations
    for idx, ex_config in enumerate(config.exchanges):
        logger.info(
            f"Exchange config #{idx+1}: name={ex_config.name}, type={ex_config.exchange_type}, enabled={ex_config.enabled}, use_as_main={ex_config.use_as_main}"
        )

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
                if getattr(exchange_config, "use_as_main", False):
                    main_connector = connector
                    logger.info(f"Using {exchange_name} as main connector")
                if getattr(exchange_config, "use_as_hedge", False):
                    hedge_connector = connector
                    logger.info(f"Using {exchange_name} as hedge connector")

    # If no main connector is specified, use the first one
    if not main_connector and connectors:
        main_connector = next(iter(connectors.values()))
        logger.info(f"No main connector specified, using first available connector")

    # If no hedge connector is specified, use the main connector
    if not hedge_connector and main_connector:
        hedge_connector = main_connector
        logger.info(
            f"No dedicated hedge connector specified, using main connector for hedging"
        )

    return {
        "connectors": connectors,
        "main_connector": main_connector,
        "hedge_connector": hedge_connector,
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
        min_margin_buffer_pct=min_margin_buffer_pct,
    )


def create_trading_engine(
    config: AppConfig,
    main_connector: Any,
    hedge_connector: Any,
    risk_manager: RiskManager,
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
        max_parallel_trades=config.max_parallel_trades,
        enable_hedging=config.get("enable_hedging", True),
    )

    return engine


async def handle_webhook_signal(signal):
    """Handle incoming webhook signals."""
    try:
        # Validate signal format
        if not isinstance(signal, dict):
            logger.error("Invalid signal format: not a dictionary")
            return {"error": "Invalid signal format"}

        # Extract required fields
        symbol = signal.get("symbol")
        direction = signal.get("direction")
        confidence = signal.get("confidence", 0.5)  # Default confidence of 0.5

        if not symbol or not direction:
            logger.error("Missing required fields in signal")
            return {"error": "Missing required fields"}

        # Convert direction string to SignalDirection enum
        try:
            signal_direction = SignalDirection[direction.upper()]
        except (KeyError, AttributeError):
            logger.error(f"Invalid signal direction: {direction}")
            return {"error": "Invalid signal direction"}

        # Create Signal object
        signal_obj = Signal(
            direction=signal_direction,
            symbol=symbol,
            source="webhook",
            confidence=confidence,
            timestamp=int(time.time() * 1000)
        )

        # Process signal
        success = await trading_engine.process_signal(signal_obj)

        if success:
            logger.info(f"Successfully processed webhook signal for {symbol}")
            return {"status": "success"}
        else:
            logger.warning(f"Failed to process webhook signal for {symbol}")
            return {"error": "Failed to process signal"}

    except Exception as e:
        logger.error(f"Error processing webhook signal: {e}", exc_info=True)
        return {"error": str(e)}


def setup_webhook_server(
    config: Dict[str, Any], trading_engine: TradingEngine
) -> Optional[WebhookServer]:
    """Set up webhook server if enabled in config"""
    if not config.get("webhook_enabled", False):
        logger.info("Webhook server disabled in configuration")
        return None

    webhook_host = config.get("webhook_host", "0.0.0.0")
    webhook_port = config.get("webhook_port", 8080)

    logger.info(f"Setting up webhook server on {webhook_host}:{webhook_port}...")

    # Create and start the webhook server
    server = WebhookServer(
        host=webhook_host,
        port=webhook_port,
        signal_handlers=[handle_webhook_signal],
    )

    if server.start():
        logger.info(f"Webhook server started at http://{webhook_host}:{webhook_port}")
        return server
    else:
        logger.error("Failed to start webhook server")
        return None


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json file."""
    try:
        config_file = os.environ.get("CONFIG_FILE", "../../shared/config.json")
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


def create_connector(
    factory: ConnectorFactory, exchange_config: Dict[str, Any]
) -> Optional[Any]:
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
        exchange_type = exchange_config.get("exchange_type", "").lower()

        # Process environment variables
        def get_env_value(config_value):
            if (
                isinstance(config_value, str)
                and config_value.startswith("${")
                and config_value.endswith("}")
            ):
                # Extract environment variable name and get its value
                env_var = config_value.strip("${}")
                return os.getenv(env_var)
            return config_value

        api_key = get_env_value(exchange_config.get("api_key", ""))
        api_secret = get_env_value(exchange_config.get("api_secret", ""))
        wallet_address = get_env_value(exchange_config.get("wallet_address", ""))
        private_key = get_env_value(exchange_config.get("private_key", ""))

        # Convert boolean configs from strings if needed
        def parse_bool_config(value):
            if isinstance(value, str):
                if value.startswith("${") and value.endswith("}"):
                    env_var = value.strip("${}")
                    env_value = os.getenv(env_var, "false")
                    return env_value.lower() in ("true", "yes", "1", "t", "y")
                return value.lower() in ("true", "yes", "1", "t", "y")
            return bool(value)

        testnet = parse_bool_config(exchange_config.get("testnet", True))
        use_sandbox = parse_bool_config(exchange_config.get("use_sandbox", True))

        logger.debug(
            f"Creating connector for {exchange_type} with parsed config values, sandbox={use_sandbox}"
        )

        # Create the connector with extracted parameters
        connector = factory.create_connector(
            exchange_type=exchange_type,
            name=exchange_config.get("name", None),
            wallet_address=wallet_address,
            private_key=private_key,
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            use_sandbox=use_sandbox,
        )

        return connector
    except Exception as e:
        logger.error(f"Failed to create connector: {str(e)}", exc_info=True)
        return None


async def run_trading_loop(strategy_manager, config, is_running, indicators):
    """Run the main trading loop."""
    logger.info("Starting main trading loop")
    while is_running:
        try:
            # Run strategy cycle to check for signals from indicators
            if indicators:
                signal_count = await strategy_manager.run_cycle()
                if signal_count > 0:
                    logger.info(
                        f"Generated and processed {signal_count} signals in this cycle"
                    )
        except Exception as e:
            logger.error(f"Error in strategy cycle: {str(e)}", exc_info=True)

        # Wait for next cycle
        await asyncio.sleep(config.get("polling_interval", 60))


async def async_main():
    """Async main application entry point."""
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

        # Start metrics server on a different port than webhook
        metrics_port = config.get(
            "metrics_port", 9000
        )  # Use port 9000 by default for metrics
        metrics_host = config.get(
            "metrics_host", "0.0.0.0"
        )  # Bind to all interfaces by default
        logger.info(f"Starting metrics server on {metrics_host}:{metrics_port}")
        start_metrics_server(port=metrics_port)
        logger.info(
            f"Metrics server started at http://{metrics_host}:{metrics_port}/metrics"
        )

        # Start historical data API server on a different port
        historical_api_port = config.get("historical_api_port", 9001)
        logger.info(f"Starting historical data API server on port {historical_api_port}")
        start_historical_data_api(port=historical_api_port)
        logger.info(f"Historical data API server started at http://{metrics_host}:{historical_api_port}/metrics/list")

        # Initialize exchange connectors
        connector_factory = ConnectorFactory()
        exchange_connectors = {}
        main_connector = None
        hedge_connector = None

        # Find where it initializes the connector and add our custom message
        connector_pattern = re.compile(r'(.*\bConnecting to Hyperliquid API at.*)')

        # Find where it logs the 404 errors and replace them
        failed_spot_pattern = re.compile(r'(.*Failed to fetch spot markets: 404.*)')
        failed_vault_pattern = re.compile(r'(.*Failed to fetch vaults: 404.*)')
        failed_spot_balances_pattern = re.compile(r'(.*Failed to fetch spot balances: 404.*)')
        failed_vault_balances_pattern = re.compile(r'(.*Failed to fetch vault balances: 404.*)')

        # Updated messages
        spot_message = "Hyperliquid spot markets not available: This feature is not supported in the current API"
        vault_message = "Hyperliquid vaults not available: This feature is not supported in the current API"
        spot_balances_message = "Hyperliquid spot balances not available: This feature is not supported in the current API"
        vault_balances_message = "Hyperliquid vault balances not available: This feature is not supported in the current API"

        for exchange_config in config["exchanges"]:
            if not exchange_config.get("enabled", False):
                logger.info(
                    f"Exchange {exchange_config['name']} is disabled, skipping..."
                )
                continue

            try:
                connector = create_connector(connector_factory, exchange_config)
                if connector:
                    exchange_connectors[exchange_config["name"]] = connector
                    logger.info(
                        f"Successfully initialized {exchange_config['name']} connector"
                    )

                    # Set main and hedge connectors based on configuration
                    if exchange_config.get("use_as_main", False):
                        main_connector = connector
                    if exchange_config.get("use_as_hedge", False):
                        hedge_connector = connector

                    # Get available markets
                    markets = connector.get_markets()
                    logger.info(
                        f"Available markets on {exchange_config['name']}: {len(markets)} markets found"
                    )

                    # Always log markets to the dedicated markets logger
                    for market in markets:
                        connector.markets_logger.info(
                            f"Market: {market['symbol']} - {market}"
                        )

                    # Only log market details to main log if flag is enabled
                    if SHOW_MARKET_DETAILS:
                        for market in markets:
                            logger.info(f"  {market['symbol']}: {market}")

                    # Check account balances
                    try:
                        balances = connector.get_account_balance()
                        non_zero_balances = {k: v for k, v in balances.items() if v > 0}
                        logger.info(
                            f"Account balances on {exchange_config['name']}: {len(non_zero_balances)} non-zero balances found"
                        )

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
                logger.error(
                    f"Failed to initialize {exchange_config['name']} connector: {str(e)}",
                    exc_info=True,
                )
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
            logger.info(
                "No hedge connector specified, using main connector for hedging"
            )

        # Initialize indicators first (needed for strategy validation)
        indicators = IndicatorFactory.create_indicators_from_config(
            config.get("indicators", [])
        )

        if not indicators:
            logger.warning(
                "No indicators configured, trading system will only receive signals from webhooks"
            )
        else:
            logger.info(
                f"Loaded {len(indicators)} indicator(s): {', '.join(indicators.keys())}"
            )

        # Load and validate strategies
        strategies = config.get("strategies", [])
        strategy_configs = []
        if strategies:
            try:
                _validate_strategy_indicators(strategies, indicators)
                logger.info(f"Successfully loaded and validated {len(strategies)} strategies")

                # Parse strategies into StrategyConfig objects for risk manager
                from app.core.strategy_config import StrategyConfigLoader
                strategy_configs = StrategyConfigLoader.load_strategies(strategies)
                logger.info(f"Parsed {len(strategy_configs)} strategy configurations for position sizing")

            except ValueError as e:
                logger.error(f"Strategy validation failed: {str(e)}")
                sys.exit(1)
        else:
            logger.warning("No strategies configured in configuration file")

        # Initialize risk manager with position sizing integration and strategy context
        risk_manager = RiskManager.from_config(config, strategies=strategy_configs)

        # Log the risk manager settings
        logger.info(f"Risk Manager initialized with: max_account_risk_pct={risk_manager.max_account_risk_pct}%, "
                    f"max_leverage={risk_manager.max_leverage}x, "
                    f"max_position_size_usd=${risk_manager.max_position_size_usd}, "
                    f"max_positions={risk_manager.max_positions}, "
                    f"min_margin_buffer_pct={risk_manager.min_margin_buffer_pct}%")
        logger.info(f"Default position sizing method: {risk_manager.position_sizer.config.method.value}")
        logger.info(f"Default position sizing config: USD amount=${risk_manager.position_sizer.config.fixed_usd_amount}, "
                    f"Max=${risk_manager.position_sizer.config.max_position_size_usd}, "
                    f"Min=${risk_manager.position_sizer.config.min_position_size_usd}")

        if risk_manager.strategy_position_sizers:
            logger.info(f"Strategy-specific position sizers: {len(risk_manager.strategy_position_sizers)}")
            for strategy_name, sizer in risk_manager.strategy_position_sizers.items():
                logger.info(f"  {strategy_name}: {sizer.config.method.value}")

        # Initialize trading engine with proper parameters
        engine = TradingEngine(
            main_connector=main_connector,
            hedge_connector=hedge_connector,
            risk_manager=risk_manager,
            dry_run=config.get("dry_run", True),
            polling_interval=config.get("polling_interval", 60),
            max_parallel_trades=1,  # Limit to 1 trade at a time for testing
            enable_hedging=config.get("enable_hedging", True),  # Use config setting for hedging
        )

        # Initialize strategy manager with strategies and indicators
        strategy_manager = StrategyManager(
            trading_engine=engine,
            indicators=indicators,
            config=config,
            strategies=strategies
        )

        # Start the trading engine
        if not engine.start():
            logger.error("Failed to start trading engine")
            sys.exit(1)

        # Set up webhook server if configured
        webhook_server = setup_webhook_server(config, engine)

        # Main trading loop
        try:
            await run_trading_loop(strategy_manager, config, is_running, indicators)
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            raise
        finally:
            # Cleanup
            logger.info("Cleaning up resources")
            engine.stop()

            # Stop webhook server if running
            if webhook_server:
                logger.info("Stopping webhook server")
                webhook_server.stop()

            for connector in exchange_connectors.values():
                connector.cleanup()
    except Exception as e:
        logger.error(f"Fatal error during startup: {str(e)}", exc_info=True)
        sys.exit(1)


def main():
    """Main application entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
