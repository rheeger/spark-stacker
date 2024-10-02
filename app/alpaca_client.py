import alpaca_trade_api as tradeapi
from config import LIVE_ALPACA_API_KEY, LIVE_ALPACA_SECRET_KEY, LIVE_BASE_URL, PAPER_ALPACA_API_KEY, PAPER_ALPACA_SECRET_KEY, PAPER_BASE_URL
import logging

api = None

def initialize_api(env):
    global api
    if env == 'LIVE':
        api = tradeapi.REST(LIVE_ALPACA_API_KEY, LIVE_ALPACA_SECRET_KEY, LIVE_BASE_URL, api_version='v2')
        logging.info("Initialized Alpaca API with LIVE environment")
    else:
        api = tradeapi.REST(PAPER_ALPACA_API_KEY, PAPER_ALPACA_SECRET_KEY, PAPER_BASE_URL, api_version='v2')
        logging.info("Initialized Alpaca API with PAPER environment")

def get_api():
    global api
    if api is None:
        raise ValueError("API not initialized. Call initialize_api() first.")
    return api

def get_account():
    logging.info("Getting account information")
    return api.get_account()

def get_position(symbol):
    logging.info(f"Getting position for {symbol}")
    try:
        # Convert symbol format for crypto (e.g., 'ETH/USD' to 'ETHUSD')
        api_symbol = symbol.replace('/', '')
        position = api.get_position(api_symbol)
        logging.info(f"Current position for {symbol}: {position.qty}")
        return position
    except Exception as e:
        if "position does not exist" in str(e).lower():
            logging.info(f"No current position in {symbol}")
            return None
        else:
            logging.error(f"Error getting position for {symbol}: {str(e)}")
            return None

def submit_order(symbol, qty, side, type, time_in_force):
    logging.info(f"Submitting order: {side} {qty} {symbol}")
    # Convert symbol format for crypto (e.g., 'ETH/USD' to 'ETHUSD')
    api_symbol = symbol.replace('/', '')
    return api.submit_order(
        symbol=api_symbol,
        qty=qty,
        side=side,
        type=type,
        time_in_force=time_in_force
    )

def get_latest_price(symbol):
    logging.info(f"Getting latest price for {symbol}")
    # Convert symbol format for crypto (e.g., 'ETH/USD' to 'ETHUSD')
    api_symbol = symbol.replace('/', '')
    try:
        # Try to get the latest trade first
        trade = api.get_latest_trade(api_symbol)
        logging.info(f"Latest trade price for {symbol}: {trade.price}")
        return float(trade.price)
    except Exception as e:
        logging.warning(f"Error getting latest trade for {symbol}: {str(e)}")
        try:
            # If latest trade fails, try to get the latest quote
            quote = api.get_latest_quote(api_symbol)
            logging.info(f"Latest quote price for {symbol}: {quote.ask_price}")
            return float(quote.ask_price)  # You could also use bid_price or an average
        except Exception as e:
            logging.error(f"Error getting latest quote for {symbol}: {str(e)}")
            return None