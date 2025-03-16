import logging
import os
import sys
import json
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
        if log_file is None:
            # Default log file: logs/spark_stacker_YYYY-MM-DD.log
            timestamp = datetime.now().strftime("%Y-%m-%d")
            log_file = logs_dir / f"spark_stacker_{timestamp}.log"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
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
        logging.info(f"Logs will be written to {log_file}")


def get_structured_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        StructuredLogger: A logger that supports structured logging
    """
    return logging.getLogger(name) 