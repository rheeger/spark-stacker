"""
Symbol conversion utilities for handling exchange-specific symbol formats.

This module provides utilities to convert standard trading symbols (e.g., "ETH-USD")
to exchange-specific formats and vice versa.
"""

import logging
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

# Define supported exchanges and their symbol format requirements
EXCHANGE_SYMBOL_FORMATS = {
    "hyperliquid": {
        "separator": "",
        "base_only": True,  # Hyperliquid uses only base symbol (e.g., "ETH")
        "case": "upper"
    },
    "coinbase": {
        "separator": "-",
        "base_only": False,  # Coinbase uses full pair (e.g., "ETH-USD")
        "case": "upper"
    },
    "kraken": {
        "separator": "",
        "base_only": False,  # Kraken uses concatenated pairs (e.g., "ETHUSD")
        "case": "upper",
    }
}

# Standard symbol separators we recognize
STANDARD_SEPARATORS = ["-", "_", "/"]


def convert_symbol_for_exchange(symbol: str, exchange: str) -> str:
    """
    Convert a standard symbol to exchange-specific format.

    Args:
        symbol: Standard symbol format (e.g., "ETH-USD", "BTC-USD")
        exchange: Target exchange name (e.g., "hyperliquid", "coinbase", "kraken")

    Returns:
        str: Exchange-specific symbol format

    Raises:
        ValueError: If exchange is not supported or symbol format is invalid

    Examples:
        >>> convert_symbol_for_exchange("ETH-USD", "hyperliquid")
        "ETH"
        >>> convert_symbol_for_exchange("ETH-USD", "coinbase")
        "ETH-USD"
        >>> convert_symbol_for_exchange("ETH-USD", "kraken")
        "ETHUSD"
        >>> convert_symbol_for_exchange("BTC-USD", "kraken")
        "BTCUSD"
    """
    exchange_lower = exchange.lower()

    if exchange_lower not in EXCHANGE_SYMBOL_FORMATS:
        raise ValueError(f"Unsupported exchange: {exchange}. Supported exchanges: {list(EXCHANGE_SYMBOL_FORMATS.keys())}")

    if not symbol:
        raise ValueError("Symbol cannot be empty")

    # Get exchange format requirements
    format_config = EXCHANGE_SYMBOL_FORMATS[exchange_lower]

    # Parse the standard symbol
    base_symbol, quote_symbol = _parse_standard_symbol(symbol)

    # Apply exchange-specific formatting
    if format_config["base_only"]:
        result = base_symbol
    else:
        result = f"{base_symbol}{format_config['separator']}{quote_symbol}"

    # Apply case formatting
    if format_config["case"] == "upper":
        result = result.upper()
    elif format_config["case"] == "lower":
        result = result.lower()

    logger.debug(f"Converted symbol '{symbol}' to '{result}' for exchange '{exchange}'")
    return result


def convert_symbol_from_exchange(symbol: str, exchange: str, quote_symbol: str = "USD") -> str:
    """
    Convert an exchange-specific symbol back to standard format.

    Args:
        symbol: Exchange-specific symbol (e.g., "ETH", "ETH-USD", "ETHUSD")
        exchange: Source exchange name (e.g., "hyperliquid", "coinbase", "kraken")
        quote_symbol: Quote symbol to use for base-only exchanges (default: "USD")

    Returns:
        str: Standard symbol format (e.g., "ETH-USD")

    Raises:
        ValueError: If exchange is not supported or symbol format is invalid

    Examples:
        >>> convert_symbol_from_exchange("ETH", "hyperliquid")
        "ETH-USD"
        >>> convert_symbol_from_exchange("ETH-USD", "coinbase")
        "ETH-USD"
        >>> convert_symbol_from_exchange("ETHUSD", "kraken")
        "ETH-USD"
    """
    exchange_lower = exchange.lower()

    if exchange_lower not in EXCHANGE_SYMBOL_FORMATS:
        raise ValueError(f"Unsupported exchange: {exchange}. Supported exchanges: {list(EXCHANGE_SYMBOL_FORMATS.keys())}")

    if not symbol:
        raise ValueError("Symbol cannot be empty")

    format_config = EXCHANGE_SYMBOL_FORMATS[exchange_lower]

    if format_config["base_only"]:
        # For base-only exchanges, we need to add the quote symbol
        base_symbol = symbol.upper()
        result = f"{base_symbol}-{quote_symbol.upper()}"
    elif exchange_lower == "kraken":
        # Handle Kraken's concatenated format and legacy mappings
        if format_config["separator"] == "":
            # Kraken concatenated format - need to split the symbol
            result = _parse_kraken_symbol(symbol, format_config)
        else:
            # Shouldn't happen for Kraken, but handle just in case
            parts = symbol.split(format_config["separator"])
            if len(parts) == 2:
                result = f"{parts[0].upper()}-{parts[1].upper()}"
            else:
                raise ValueError(f"Invalid symbol format for {exchange}: {symbol}")
    else:
        # For full pair exchanges, return as-is (but standardize separator)
        if format_config["separator"] in symbol:
            parts = symbol.split(format_config["separator"])
            if len(parts) == 2:
                result = f"{parts[0].upper()}-{parts[1].upper()}"
            else:
                raise ValueError(f"Invalid symbol format for {exchange}: {symbol}")
        else:
            # Single symbol, assume USD quote
            result = f"{symbol.upper()}-{quote_symbol.upper()}"

    logger.debug(f"Converted exchange symbol '{symbol}' from '{exchange}' to standard format '{result}'")
    return result


def validate_symbol_format(symbol: str) -> bool:
    """
    Validate that a symbol follows standard format.

    Args:
        symbol: Symbol to validate

    Returns:
        bool: True if symbol format is valid

    Examples:
        >>> validate_symbol_format("ETH-USD")
        True
        >>> validate_symbol_format("INVALID")
        False
    """
    if not symbol:
        return False

    try:
        _parse_standard_symbol(symbol)
        return True
    except ValueError:
        return False


def get_supported_exchanges() -> Set[str]:
    """
    Get the set of supported exchange names.

    Returns:
        Set[str]: Set of supported exchange names
    """
    return set(EXCHANGE_SYMBOL_FORMATS.keys())


def _parse_standard_symbol(symbol: str) -> tuple[str, str]:
    """
    Parse a standard symbol into base and quote components.

    Args:
        symbol: Standard symbol (e.g., "ETH-USD", "BTC_USDT")

    Returns:
        tuple[str, str]: (base_symbol, quote_symbol)

    Raises:
        ValueError: If symbol format is invalid
    """
    if not symbol:
        raise ValueError("Symbol cannot be empty")

    # Try different separators
    for separator in STANDARD_SEPARATORS:
        if separator in symbol:
            parts = symbol.split(separator)
            if len(parts) == 2 and all(part.strip() for part in parts):
                return parts[0].strip().upper(), parts[1].strip().upper()

    # If no separator found, this might be an invalid format
    raise ValueError(f"Invalid symbol format: {symbol}. Expected format like 'ETH-USD', 'BTC_USDT', or 'ETH/USD'")


def _parse_kraken_symbol(symbol: str, format_config: Dict[str, any]) -> str:
    """
    Parse a Kraken concatenated symbol back to standard format.

    Kraken uses concatenated symbols like "ETHUSD", "BTCUSD", "ADAEUR".

    Args:
        symbol: Kraken symbol (e.g., "ETHUSD", "BTCUSD")
        format_config: Kraken format configuration

    Returns:
        str: Standard format symbol (e.g., "ETH-USD")

    Raises:
        ValueError: If symbol cannot be parsed
    """
    symbol = symbol.upper()

    # Common quote currencies (in order of length for proper matching)
    quote_currencies = ["USDT", "USDC", "PYUSD", "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF"]

    # Sort by length (longest first) to match USDT before USD, etc.
    quote_currencies.sort(key=len, reverse=True)

    # Try to find a quote currency at the end of the symbol
    for quote in quote_currencies:
        if symbol.endswith(quote):
            base_part = symbol[:-len(quote)]
            quote_part = quote

            return f"{base_part}-{quote_part}"

    # If no quote currency found, this might be an error or unusual symbol
    raise ValueError(f"Cannot parse Kraken symbol: {symbol}. No recognized quote currency found.")


def get_exchange_format_info(exchange: str) -> Optional[Dict[str, any]]:
    """
    Get format information for a specific exchange.

    Args:
        exchange: Exchange name

    Returns:
        Optional[Dict]: Format configuration or None if exchange not supported
    """
    return EXCHANGE_SYMBOL_FORMATS.get(exchange.lower())
