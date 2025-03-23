import logging
import os
import sys
import json
import uuid
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Generate a unique build ID for this container instance
CONTAINER_BUILD_ID = str(uuid.uuid4())[:8]
# Generate a timestamp for this build
BUILD_TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Create a build-specific directory for all logs
BUILD_LOG_DIR = logs_dir / f"{BUILD_TIMESTAMP}_{CONTAINER_BUILD_ID}"
BUILD_LOG_DIR.mkdir(exist_ok=True)

# Dictionary to track connector-specific log directories
CONNECTOR_LOG_DIRS = {}

class StructuredLogRecord(logging.LogRecord):
    """Custom LogRecord that adds structured logging capabilities."""
    
    def getMessage(self) -> str:
        """
        Return the message for this LogRecord.
        
        If the message is a dictionary, convert it to JSON string.
        """
        msg = super().getMessage()
        if isinstance(self.args, dict) and self.args.get('_structured', False):
            # Remove the _structured flag
            data = {k: v for k, v in self.args.items() if k != '_structured'}
            # Add standard log fields
            data.update({
                'timestamp': datetime.now().isoformat(),
                'level': self.levelname,
                'logger': self.name,
                'message': msg
            })
            return json.dumps(data)
        return msg


class StructuredLogger(logging.Logger):
    """Logger that supports structured logging."""
    
    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
        """Create a LogRecord with StructuredLogRecord."""
        return StructuredLogRecord(name, level, fn, lno, msg, args, exc_info, func, sinfo)
    
    def struct(self, msg: str, level: int = logging.INFO, **kwargs) -> None:
        """
        Log a structured message at the specified level.
        
        Args:
            msg: The message to log
            level: The logging level
            **kwargs: Additional key-value pairs to include in the structured log
        """
        kwargs['_structured'] = True
        self.log(level, msg, kwargs)


def setup_connector_log_directory(connector_name: str) -> Path:
    """
    Create a dedicated directory for a connector's logs.
    
    Args:
        connector_name: Name of the connector (e.g., 'coinbase')
        
    Returns:
        Path: Path to the connector's log directory
    """
    # Check if we already created this directory
    if connector_name in CONNECTOR_LOG_DIRS:
        return CONNECTOR_LOG_DIRS[connector_name]
    
    # Create a connector-specific directory in the build directory
    connector_dir = BUILD_LOG_DIR / connector_name
    connector_dir.mkdir(exist_ok=True)
    
    # Store the directory path
    CONNECTOR_LOG_DIRS[connector_name] = connector_dir
    
    logging.info(f"Created connector log directory for {connector_name} at {connector_dir}")
    
    return connector_dir


def setup_connector_balance_logger(connector_name: str, log_level: Union[str, int] = "DEBUG") -> logging.Logger:
    """
    Create a dedicated logger for connector balance operations.
    
    Args:
        connector_name: Name of the connector (e.g., 'coinbase')
        log_level: Logging level for the connector balance logger
        
    Returns:
        logging.Logger: Configured logger for connector balance operations
    """
    # Convert string log level to integer if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.DEBUG)
    
    # Create a unique logger name for this connector's balance operations
    logger_name = f"app.connectors.{connector_name}.balance"
    
    # Get or create the logger
    balance_logger = logging.getLogger(logger_name)
    balance_logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in balance_logger.handlers[:]:
        balance_logger.removeHandler(handler)
    
    # Create connector-specific log directory and file
    connector_dir = setup_connector_log_directory(connector_name)
    connector_log_file = connector_dir / "balance.log"
    
    connector_file_handler = RotatingFileHandler(
        connector_log_file, 
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    connector_file_handler.setFormatter(formatter)
    balance_logger.addHandler(connector_file_handler)
    
    # Make sure this logger doesn't propagate to the root logger
    balance_logger.propagate = False
    
    logging.info(f"Created dedicated balance logger for {connector_name} at {connector_log_file}")
    
    return balance_logger


def setup_connector_markets_logger(connector_name: str, log_level: Union[str, int] = "DEBUG") -> logging.Logger:
    """
    Create a dedicated logger for connector market data operations.
    
    Args:
        connector_name: Name of the connector (e.g., 'coinbase')
        log_level: Logging level for the connector markets logger
        
    Returns:
        logging.Logger: Configured logger for connector market data operations
    """
    # Convert string log level to integer if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.DEBUG)
    
    # Create a unique logger name for this connector's market operations
    logger_name = f"app.connectors.{connector_name}.markets"
    
    # Get or create the logger
    markets_logger = logging.getLogger(logger_name)
    markets_logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in markets_logger.handlers[:]:
        markets_logger.removeHandler(handler)
    
    # Create connector-specific log directory and file
    connector_dir = setup_connector_log_directory(connector_name)
    connector_log_file = connector_dir / "markets.log"
    
    connector_file_handler = RotatingFileHandler(
        connector_log_file, 
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    connector_file_handler.setFormatter(formatter)
    markets_logger.addHandler(connector_file_handler)
    
    # Make sure this logger doesn't propagate to the root logger
    markets_logger.propagate = False
    
    logging.info(f"Created dedicated markets logger for {connector_name} at {connector_log_file}")
    
    return markets_logger


def setup_connector_orders_logger(connector_name: str, log_level: Union[str, int] = "DEBUG") -> logging.Logger:
    """
    Create a dedicated logger for connector order operations.
    
    Args:
        connector_name: Name of the connector (e.g., 'coinbase')
        log_level: Logging level for the connector orders logger
        
    Returns:
        logging.Logger: Configured logger for connector order operations
    """
    # Convert string log level to integer if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.DEBUG)
    
    # Create a unique logger name for this connector's order operations
    logger_name = f"app.connectors.{connector_name}.orders"
    
    # Get or create the logger
    orders_logger = logging.getLogger(logger_name)
    orders_logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in orders_logger.handlers[:]:
        orders_logger.removeHandler(handler)
    
    # Create connector-specific log directory and file
    connector_dir = setup_connector_log_directory(connector_name)
    connector_log_file = connector_dir / "orders.log"
    
    connector_file_handler = RotatingFileHandler(
        connector_log_file, 
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    connector_file_handler.setFormatter(formatter)
    orders_logger.addHandler(connector_file_handler)
    
    # Make sure this logger doesn't propagate to the root logger
    orders_logger.propagate = False
    
    logging.info(f"Created dedicated orders logger for {connector_name} at {connector_log_file}")
    
    return orders_logger


def setup_logging(
    log_level: Union[str, int] = "INFO",
    log_to_file: bool = True,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    enable_console: bool = True,
    structured: bool = False,
    log_format: Optional[str] = None
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (e.g., "DEBUG", "INFO", "WARNING")
        log_to_file: Whether to log to file
        log_file: Path to log file. If None, uses a default path
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        enable_console: Whether to log to console
        structured: Whether to use structured logging (JSON format)
        log_format: Custom log format. If None, uses default format
    """
    # Register the custom logger class
    logging.setLoggerClass(StructuredLogger)
    
    # Convert string log level to integer if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Determine log format
    if log_format is None:
        log_format = DEFAULT_LOG_FORMAT
    
    date_format = DEFAULT_DATE_FORMAT
    
    # Create formatter
    if structured:
        # For structured logging, we don't need a formatter since we format in getMessage
        formatter = logging.Formatter('%(message)s')
    else:
        formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        # Create main log file in the build directory
        build_log_file = BUILD_LOG_DIR / "spark_stacker.log"
        
        # Create file handler for build-specific log
        build_file_handler = RotatingFileHandler(
            build_log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        build_file_handler.setLevel(log_level)
        build_file_handler.setFormatter(formatter)
        root_logger.addHandler(build_file_handler)
        
        # Set log_file to build log file for the message below
        log_file = build_log_file
    
    # Set library loggers to WARNING to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("web3").setLevel(logging.WARNING)
    logging.getLogger("coinbase").setLevel(logging.WARNING)
    logging.getLogger("websocket").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("pytz").setLevel(logging.WARNING)
    # Set internal loggers to a higher level if not in debug mode
    if log_level > logging.DEBUG:
        # Reduce connector logs unless we're in debug mode
        logging.getLogger("app.connectors").setLevel(logging.INFO)
    
    logging.info(f"Logging initialized at level {logging.getLevelName(log_level)}")
    if log_to_file:
        logging.info(f"Logs will be written to directory {BUILD_LOG_DIR}")
        logging.info(f"Container build ID: {CONTAINER_BUILD_ID}")
        logging.info(f"Build timestamp: {BUILD_TIMESTAMP}")


def get_structured_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        StructuredLogger: A logger that supports structured logging
    """
    return logging.getLogger(name) 