import logging
from alpaca_client import initialize_api, get_position, submit_order, get_latest_price, get_api
from trading_strategy import get_indicators, generate_signals
from trade_execution import execute_trade
import time
import argparse
import os
import signal
from datetime import datetime

def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Generate a timestamp for the log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/trading_bot_{timestamp}.log'

    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

def job(symbol, initial_size):
    logging.info(f"Starting job for {symbol}")
    execute_trade(symbol, initial_size)
    logging.info(f"Finished job for {symbol}")

def graceful_shutdown(symbol):
    logging.info("Initiating graceful shutdown...")
    position = get_position(symbol)
    
    if position:
        current_qty = float(position.qty)
        latest_price = get_latest_price(symbol)
        if latest_price is not None:
            position_value = current_qty * latest_price
            print(f"\nCurrent position: {current_qty} {symbol} (Value: ${position_value:.2f})")
        else:
            print(f"\nCurrent position: {current_qty} {symbol} (Unable to determine current value)")
        
        choice = input("Do you want to:\n1. Liquidate entire position\n2. Cash out a specific amount\n3. Keep position and exit\nEnter your choice (1/2/3): ")
        
        if choice == '1':
            logging.info(f"Liquidating entire position of {current_qty} {symbol}")
            order = submit_order(symbol, current_qty, 'sell', 'market', 'gtc')
            logging.info(f"Liquidation order submitted: {order}")
        elif choice == '2':
            if latest_price is None:
                print("Unable to cash out a specific amount due to missing price information.")
            else:
                amount = float(input("Enter the amount to cash out in USD: "))
                qty_to_sell = min(current_qty, amount / latest_price)
                logging.info(f"Cashing out {qty_to_sell} {symbol}")
                order = submit_order(symbol, qty_to_sell, 'sell', 'market', 'gtc')
                logging.info(f"Cash out order submitted: {order}")
        else:
            logging.info("Keeping current position and exiting")
    else:
        logging.info("No position to liquidate")
    
    logging.info("Graceful shutdown complete")

def signal_handler(signum, frame):
    global should_run
    should_run = False
    logging.info("Received shutdown signal. Preparing to exit...")

def main():
    global should_run
    should_run = True
    
    parser = argparse.ArgumentParser(description='Spark Stacker Trading Bot')
    parser.add_argument('symbol', type=str, help='Trading symbol (e.g., ETH/USD)')
    parser.add_argument('--minimum_position', type=float, default=0.05, help='Minimum position size')
    parser.add_argument('--env', type=str, choices=['LIVE', 'PAPER'], default='PAPER', help='Trading environment (LIVE or PAPER)')
    args = parser.parse_args()

    symbol = args.symbol
    minimum_position = args.minimum_position
    env = args.env

    # Set up logging
    setup_logging()

    # Initialize Alpaca API with the selected environment
    initialize_api(env)

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logging.info(f"Welcome to Spark Stacker! Trading {symbol} in {env} environment")
    
    logging.info("Starting main loop")
    try:
        while should_run:
            logging.info("Executing job")
            job(symbol, minimum_position)
            logging.info("Job executed, waiting for 5 seconds")
            time.sleep(5)  # Check every 5 seconds
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        graceful_shutdown(symbol)

if __name__ == "__main__":
    main()