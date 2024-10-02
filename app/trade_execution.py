import logging
from alpaca_client import get_position, submit_order, get_latest_price, get_api
from trading_strategy import execute_strategy, TradingState

trading_state = TradingState()

def execute_trade(symbol, minimum_position):
    logging.info(f"Executing trade for {symbol}")
    
    api = get_api()
    position = get_position(symbol)
    current_qty = float(position.qty) if position else 0
    
    # Add a tolerance of 1% of the minimum position
    tolerance = minimum_position * 0.01
    
    if current_qty < minimum_position - tolerance:
        qty_to_buy = minimum_position - current_qty
        logging.info(f"Building minimum position of {qty_to_buy} {symbol}")
        try:
            order = submit_order(
                symbol=symbol,
                qty=qty_to_buy,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            logging.info(f"Minimum position built: Bought {qty_to_buy} {symbol}")
            logging.info(f"Order details: {order}")
            trading_state.in_position = True
        except Exception as e:
            logging.error(f"Error building minimum position: {str(e)}")
        return
    
    latest_price = get_latest_price(symbol)
    if latest_price is None:
        logging.error("Unable to get latest price. Skipping trade.")
        return
    
    action, qty = execute_strategy(api, symbol, current_qty, minimum_position, latest_price, trading_state)
    
    if action == 'buy':
        logging.info(f"Submitting buy order for {qty} {symbol}")
        try:
            order = submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            logging.info(f"Bought {qty} {symbol}")
            logging.info(f"Order details: {order}")
        except Exception as e:
            logging.error(f"Error executing buy order: {str(e)}")
    elif action == 'sell':
        logging.info(f"Submitting sell order for {qty} {symbol}")
        try:
            order = submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            logging.info(f"Sold {qty} {symbol}")
            logging.info(f"Order details: {order}")
        except Exception as e:
            logging.error(f"Error executing sell order: {str(e)}")
    else:
        logging.info("No trade executed")