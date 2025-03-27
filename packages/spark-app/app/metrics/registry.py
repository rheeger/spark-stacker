import logging
import threading
import time
from typing import Optional

from prometheus_client import REGISTRY, start_http_server

logger = logging.getLogger(__name__)

# Default metrics port
DEFAULT_METRICS_PORT = 8000

# Server instance reference
_server_thread: Optional[threading.Thread] = None
_is_running = False


def start_metrics_server(port: int = DEFAULT_METRICS_PORT) -> None:
    """
    Start the Prometheus metrics server on the specified port.

    Args:
        port: The port to expose metrics on (default: 8000)
    """
    global _server_thread, _is_running

    if _is_running:
        logger.warning("Metrics server is already running")
        return

    def _run_server() -> None:
        global _is_running
        try:
            logger.info(f"Starting metrics server on port {port}")
            start_http_server(port)
            _is_running = True
            logger.info(f"Metrics server started on port {port}")

            # Keep the thread alive
            while _is_running:
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error in metrics server: {str(e)}")
            _is_running = False

    _server_thread = threading.Thread(target=_run_server, daemon=True)
    _server_thread.start()

    # Give the server a moment to start
    time.sleep(0.5)

    if not _is_running:
        logger.error("Failed to start metrics server")


def stop_metrics_server() -> None:
    """Stop the Prometheus metrics server."""
    global _is_running

    if not _is_running:
        logger.warning("Metrics server is not running")
        return

    logger.info("Stopping metrics server")
    _is_running = False

    if _server_thread:
        _server_thread.join(timeout=5)
        logger.info("Metrics server stopped")
