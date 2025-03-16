import logging
import json
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from flask import Flask, request, jsonify
from waitress import serve
from waitress.server import create_server

from app.indicators.base_indicator import Signal, SignalDirection

logger = logging.getLogger(__name__)

class WebhookServer:
    """
    Webhook server for receiving trading signals from external sources like TradingView.
    
    This server provides HTTP endpoints that can receive webhook POST requests
    containing trading signals, which are then processed by the trading engine.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        signal_handlers: Optional[List[Callable[[Signal], None]]] = None,
        auth_token: Optional[str] = None
    ):
        """
        Initialize the webhook server.
        
        Args:
            host: Host address to bind the server
            port: Port to listen on
            signal_handlers: List of callback functions to handle received signals
            auth_token: Optional authentication token for webhook security
        """
        self.host = host
        self.port = port
        self.signal_handlers = signal_handlers or []
        self.auth_token = auth_token
        
        self.app = Flask(__name__)
        self.server = None
        self.server_thread = None
        self.is_running = False
        self.stop_event = threading.Event()
        
        # Register routes
        self.register_routes()
    
    def register_routes(self) -> None:
        """Register the Flask routes."""
        
        @self.app.route('/', methods=['GET'])
        def index():
            """Health check endpoint."""
            return jsonify({
                "status": "ok",
                "service": "Spark Stacker Webhook Server",
                "timestamp": int(time.time())
            })
        
        @self.app.route('/webhook', methods=['POST'])
        def webhook():
            """
            Main webhook endpoint for receiving trading signals.
            
            Expects a JSON payload with signal information.
            """
            if not request.is_json:
                logger.warning("Received non-JSON request")
                return jsonify({"status": "error", "message": "Content-Type must be application/json"}), 400
            
            # Validate auth token if configured
            if self.auth_token:
                auth_header = request.headers.get('Authorization')
                if not auth_header or auth_header != f"Bearer {self.auth_token}":
                    logger.warning("Unauthorized webhook access attempt")
                    return jsonify({"status": "error", "message": "Unauthorized"}), 401
            
            try:
                payload = request.json
                logger.info(f"Received webhook: {payload}")
                
                # Process the payload
                signal = self.parse_signal(payload)
                
                if signal:
                    # Notify all registered handlers
                    for handler in self.signal_handlers:
                        try:
                            handler(signal)
                        except Exception as e:
                            logger.error(f"Error in signal handler: {e}")
                    
                    return jsonify({
                        "status": "success",
                        "message": f"Signal processed: {signal}"
                    })
                else:
                    return jsonify({
                        "status": "error",
                        "message": "Invalid signal format"
                    }), 400
            
            except Exception as e:
                logger.error(f"Error processing webhook: {e}")
                return jsonify({
                    "status": "error",
                    "message": f"Server error: {str(e)}"
                }), 500
        
        @self.app.route('/webhook/tradingview', methods=['POST'])
        def tradingview_webhook():
            """
            Specialized endpoint for TradingView alerts.
            
            Parses TradingView alert format and converts to our signal format.
            """
            if not request.is_json:
                logger.warning("Received non-JSON request from TradingView")
                return jsonify({"status": "error", "message": "Content-Type must be application/json"}), 400
            
            try:
                payload = request.json
                logger.info(f"Received TradingView alert: {payload}")
                
                # Parse TradingView format
                signal = self.parse_tradingview_alert(payload)
                
                if signal:
                    # Notify all registered handlers
                    for handler in self.signal_handlers:
                        try:
                            handler(signal)
                        except Exception as e:
                            logger.error(f"Error in signal handler: {e}")
                    
                    return jsonify({
                        "status": "success",
                        "message": f"TradingView signal processed: {signal}"
                    })
                else:
                    return jsonify({
                        "status": "error",
                        "message": "Invalid TradingView alert format"
                    }), 400
            
            except Exception as e:
                logger.error(f"Error processing TradingView webhook: {e}")
                return jsonify({
                    "status": "error",
                    "message": f"Server error: {str(e)}"
                }), 500
    
    def parse_signal(self, payload: Dict[str, Any]) -> Optional[Signal]:
        """
        Parse a generic webhook payload into a Signal object.
        
        Args:
            payload: The JSON payload from the webhook
            
        Returns:
            Signal object if valid, None otherwise
        """
        try:
            # Check if we have all required fields
            if 'direction' not in payload or 'symbol' not in payload:
                logger.warning(f"Missing required fields in signal payload: {payload}")
                return None
            
            # Validate direction
            direction_str = payload.get('direction', '').upper()
            try:
                direction = SignalDirection(direction_str)
            except ValueError:
                logger.warning(f"Invalid direction in signal: {direction_str}")
                return None
            
            # Create Signal object
            return Signal(
                direction=direction,
                symbol=payload.get('symbol'),
                indicator=payload.get('indicator', 'webhook'),
                confidence=float(payload.get('confidence', 0.5)),
                timestamp=int(payload.get('timestamp', time.time() * 1000)),
                params=payload.get('params', {})
            )
        
        except Exception as e:
            logger.error(f"Error parsing signal: {e}")
            return None
    
    def parse_tradingview_alert(self, payload: Dict[str, Any]) -> Optional[Signal]:
        """
        Parse a TradingView alert into a Signal object.
        
        TradingView alerts can be formatted in different ways. This method
        attempts to handle common formats used in Pine Script alerts.
        
        Args:
            payload: The JSON payload from TradingView
            
        Returns:
            Signal object if valid, None otherwise
        """
        try:
            # Check if it's already in our format
            if 'direction' in payload and 'symbol' in payload:
                return self.parse_signal(payload)
            
            # TradingView often uses a text message or custom fields
            # Look for action field first (common in Pine Script exports)
            action = payload.get('action', '')
            
            if action:
                direction = SignalDirection.BUY if 'buy' in action.lower() else (
                    SignalDirection.SELL if 'sell' in action.lower() else SignalDirection.NEUTRAL
                )
            elif 'message' in payload:
                # Parse from message text
                message = payload.get('message', '')
                if 'buy' in message.lower():
                    direction = SignalDirection.BUY
                elif 'sell' in message.lower() or 'short' in message.lower():
                    direction = SignalDirection.SELL
                else:
                    direction = SignalDirection.NEUTRAL
            else:
                logger.warning(f"Unsupported TradingView alert format: {payload}")
                return None
            
            # Extract symbol (ticker)
            symbol = payload.get('ticker', '')
            if not symbol:
                # Try to extract from message or chart field
                message = payload.get('message', '')
                chart = payload.get('chart', '')
                
                # Common format: "BUY BTCUSD" or similar
                words = message.split() if message else chart.split()
                for word in words:
                    if any(crypto in word.upper() for crypto in ['BTC', 'ETH', 'SOL', 'AVAX']):
                        symbol = word.upper()
                        break
            
            if not symbol:
                logger.warning("Could not extract symbol from TradingView alert")
                return None
            
            # Clean up symbol (remove suffixes like BINANCE:BTCUSDT)
            if ':' in symbol:
                symbol = symbol.split(':')[1]
            
            # Remove common suffix patterns
            for suffix in ['USDT', 'USD', 'PERP']:
                if symbol.endswith(suffix):
                    symbol = symbol[:-len(suffix)]
                    break
            
            # Extract confidence if available
            confidence = 0.5
            if 'confidence' in payload:
                try:
                    confidence = float(payload.get('confidence', 0.5))
                except (ValueError, TypeError):
                    pass
            
            # Create Signal object
            return Signal(
                direction=direction,
                symbol=symbol,
                indicator='tradingview',
                confidence=confidence,
                timestamp=int(time.time() * 1000),
                params=payload
            )
        
        except Exception as e:
            logger.error(f"Error parsing TradingView alert: {e}")
            return None
    
    def add_signal_handler(self, handler: Callable[[Signal], None]) -> None:
        """
        Add a signal handler callback function.
        
        Args:
            handler: Function to call when a signal is received
        """
        if handler not in self.signal_handlers:
            self.signal_handlers.append(handler)
    
    def start(self) -> bool:
        """
        Start the webhook server in a background thread.
        
        Returns:
            bool: True if server started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("Webhook server is already running")
            return True
        
        def run_server():
            logger.info(f"Starting webhook server on {self.host}:{self.port}")
            try:
                self.server = create_server(self.app, host=self.host, port=self.port)
                self.server.run()
            except Exception as e:
                logger.error(f"Webhook server error: {e}")
                self.is_running = False
                self.stop_event.set()
        
        try:
            self.stop_event.clear()
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            self.is_running = True
            logger.info("Webhook server started")
            return True
        except Exception as e:
            logger.error(f"Failed to start webhook server: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the webhook server.
        
        Returns:
            bool: True if server stopped successfully, False otherwise
        """
        if not self.is_running:
            logger.warning("Webhook server is not running")
            return True
        
        try:
            logger.info("Stopping webhook server...")
            self.is_running = False
            self.stop_event.set()
            
            # Close the server if it exists
            if self.server:
                self.server.close()
            
            # Wait for the server thread to finish
            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=5)
                if self.server_thread.is_alive():
                    logger.warning("Webhook server thread did not stop gracefully")
            
            logger.info("Webhook server stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping webhook server: {e}")
            return False 