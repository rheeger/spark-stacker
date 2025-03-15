#!/usr/bin/env python3
import logging
import os
import signal
import sys
import time
from typing import Dict, List, Any, Optional
import argparse

# Import core components
from app.utils.config import ConfigManager, AppConfig
from app.utils.logging_setup import setup_logging
from app.connectors.connector_factory import ConnectorFactory
from app.indicators.indicator_factory import IndicatorFactory
from app.risk_management.risk_manager import RiskManager
from app.core.trading_engine import TradingEngine
from app.webhook.webhook_server import WebhookServer

# Global variables
engine: Optional[TradingEngine] = None
webhook_server: Optional[WebhookServer] = None
is_running = True

def signal_handler(sig, frame):
    """Handle system signals like CTRL+C"""
    global is_running
    logging.info("Shutdown signal received, stopping services...")
    is_running = False

def initialize_logging(config: AppConfig) -> None:
    """Set up application logging based on configuration"""
    setup_logging(
        log_level=config.log_level,
        log_to_file=True,
        structured=True,
        enable_console=True
    )
    logging.info("Logging initialized")

def create_exchange_connectors(config: AppConfig) -> Dict[str, Any]:
    """Create and initialize exchange connectors from configuration"""
    logging.info("Initializing exchange connectors...")
    
    # Pass the ExchangeConfig objects directly to the ConnectorFactory
    connectors = ConnectorFactory.create_connectors_from_config(config.exchanges)
    
    if not connectors:
        logging.error("Failed to create any exchange connectors")
        return {}
    
    # Identify main and hedge connectors based on config
    main_connector = None
    hedge_connector = None
    
    for exchange_name, connector in connectors.items():
        for exchange_config in config.exchanges:
            if exchange_config.name.lower() == exchange_name.lower():
                if getattr(exchange_config, 'use_as_main', False):
                    main_connector = connector
                    logging.info(f"Using {exchange_name} as main connector")
                if getattr(exchange_config, 'use_as_hedge', False):
                    hedge_connector = connector
                    logging.info(f"Using {exchange_name} as hedge connector")
    
    # If no main connector is specified, use the first one
    if not main_connector and connectors:
        main_connector = next(iter(connectors.values()))
        logging.info(f"No main connector specified, using first available connector")
    
    # If no hedge connector is specified, use the main connector
    if not hedge_connector and main_connector:
        hedge_connector = main_connector
        logging.info(f"No dedicated hedge connector specified, using main connector for hedging")
    
    return {
        'connectors': connectors,
        'main_connector': main_connector,
        'hedge_connector': hedge_connector
    }

def create_indicators(config: AppConfig) -> Dict[str, Any]:
    """Create technical indicators from configuration"""
    logging.info("Initializing technical indicators...")
    indicators = IndicatorFactory.create_indicators_from_config(config.indicators)
    
    if not indicators:
        logging.warning("No indicators created from configuration")
    
    return indicators

def create_risk_manager(config: AppConfig) -> RiskManager:
    """Create and configure the risk management system"""
    logging.info("Initializing risk management system...")
    
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
    logging.info("Initializing trading engine...")
    
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
        logging.info("Webhook server disabled in configuration")
        return None
    
    logging.info(f"Setting up webhook server on {config.webhook_host}:{config.webhook_port}...")
    
    # Define the signal handler function
    def handle_webhook_signal(signal):
        logging.info(f"Received signal from webhook: {signal}")
        trading_engine.process_signal(signal)
    
    # Create and start the webhook server
    server = WebhookServer(
        host=config.webhook_host,
        port=config.webhook_port,
        signal_handlers=[handle_webhook_signal]
    )
    
    if server.start():
        logging.info(f"Webhook server started at http://{config.webhook_host}:{config.webhook_port}")
        return server
    else:
        logging.error("Failed to start webhook server")
        return None

def main():
    """Main application entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Spark Stacker Trading System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Load configuration
        config_manager = ConfigManager(config_path=args.config)
        config = config_manager.load()
        
        # Initialize logging
        initialize_logging(config)
        
        # Create and initialize components
        exchange_info = create_exchange_connectors(config)
        if not exchange_info.get('main_connector'):
            logging.error("No main exchange connector available, cannot continue")
            return 1
        
        indicators = create_indicators(config)
        risk_manager = create_risk_manager(config)
        
        # Create the trading engine
        global engine
        engine = create_trading_engine(
            config=config,
            main_connector=exchange_info.get('main_connector'),
            hedge_connector=exchange_info.get('hedge_connector'),
            risk_manager=risk_manager
        )
        
        # Start the trading engine
        if not engine.start():
            logging.error("Failed to start trading engine")
            return 1
        
        # Set up webhook server if enabled
        global webhook_server
        if config.webhook_enabled:
            webhook_server = setup_webhook_server(config, engine)
        
        # Main application loop
        logging.info(f"Spark Stacker is running (dry_run={config.dry_run})")
        
        # Process any strategies from configuration
        for strategy in config.strategies:
            if not strategy.enabled:
                logging.info(f"Strategy {strategy.name} is disabled, skipping")
                continue
            
            logging.info(f"Configuring strategy: {strategy.name} for {strategy.market}")
            # You could initialize strategy-specific parameters here
        
        # Keep the application running until a termination signal is received
        while is_running:
            try:
                # Check engine status
                if engine.state.value != "RUNNING" and engine.state.value != "PAUSED":
                    logging.warning(f"Engine state: {engine.state.value}, attempting to restart...")
                    engine.start()
                
                # Report active trades status
                active_trades = engine.get_active_trades()
                if active_trades:
                    logging.info(f"Active trades: {len(active_trades)}")
                    for symbol, trade in active_trades.items():
                        logging.info(f"  {symbol}: {trade.get('status', 'unknown')}")
                
                time.sleep(60)  # Status check interval
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(10)  # Shorter sleep on error
        
        # Shutdown sequence
        logging.info("Shutting down Spark Stacker...")
        
        # Stop the trading engine
        if engine:
            engine.stop()
        
        # Stop the webhook server
        if webhook_server:
            webhook_server.stop()
        
        logging.info("Shutdown complete")
        return 0
    
    except Exception as e:
        logging.critical(f"Critical error in main application: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 