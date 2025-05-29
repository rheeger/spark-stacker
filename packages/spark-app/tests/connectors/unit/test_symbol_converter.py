"""
Unit tests for symbol conversion utilities.

Tests the symbol_converter module functions that handle exchange-specific symbol formats.
"""

import pytest
from app.core.symbol_converter import (EXCHANGE_SYMBOL_FORMATS,
                                       _parse_standard_symbol,
                                       convert_symbol_for_exchange,
                                       convert_symbol_from_exchange,
                                       get_exchange_format_info,
                                       get_supported_exchanges,
                                       validate_symbol_format)


class TestConvertSymbolForExchange:
    """Test the convert_symbol_for_exchange function."""

    def test_convert_to_hyperliquid_format(self):
        """Test symbol conversion to Hyperliquid format (base only)."""
        assert convert_symbol_for_exchange("ETH-USD", "hyperliquid") == "ETH"
        assert convert_symbol_for_exchange("BTC-USD", "hyperliquid") == "BTC"
        assert convert_symbol_for_exchange("MATIC-USD", "hyperliquid") == "MATIC"

    def test_convert_to_coinbase_format(self):
        """Test symbol conversion to Coinbase format (full pair)."""
        assert convert_symbol_for_exchange("ETH-USD", "coinbase") == "ETH-USD"
        assert convert_symbol_for_exchange("BTC-USD", "coinbase") == "BTC-USD"
        assert convert_symbol_for_exchange("MATIC-USD", "coinbase") == "MATIC-USD"

    def test_convert_with_different_separators(self):
        """Test symbol conversion with different input separators."""
        # Test underscore separator
        assert convert_symbol_for_exchange("ETH_USD", "hyperliquid") == "ETH"
        assert convert_symbol_for_exchange("ETH_USD", "coinbase") == "ETH-USD"

        # Test slash separator
        assert convert_symbol_for_exchange("ETH/USD", "hyperliquid") == "ETH"
        assert convert_symbol_for_exchange("ETH/USD", "coinbase") == "ETH-USD"

    def test_convert_case_insensitive_exchange(self):
        """Test that exchange names are case insensitive."""
        assert convert_symbol_for_exchange("ETH-USD", "HYPERLIQUID") == "ETH"
        assert convert_symbol_for_exchange("ETH-USD", "Hyperliquid") == "ETH"
        assert convert_symbol_for_exchange("ETH-USD", "COINBASE") == "ETH-USD"
        assert convert_symbol_for_exchange("ETH-USD", "Coinbase") == "ETH-USD"

    def test_convert_unknown_exchange_raises_error(self):
        """Test that unknown exchange raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported exchange: unknown"):
            convert_symbol_for_exchange("ETH-USD", "unknown")

    def test_convert_empty_symbol_raises_error(self):
        """Test that empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            convert_symbol_for_exchange("", "hyperliquid")

    def test_convert_invalid_symbol_format_raises_error(self):
        """Test that invalid symbol format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            convert_symbol_for_exchange("INVALID", "hyperliquid")


class TestConvertSymbolFromExchange:
    """Test the convert_symbol_from_exchange function."""

    def test_convert_from_hyperliquid_format(self):
        """Test conversion from Hyperliquid format (base only) to standard."""
        assert convert_symbol_from_exchange("ETH", "hyperliquid") == "ETH-USD"
        assert convert_symbol_from_exchange("BTC", "hyperliquid") == "BTC-USD"
        assert convert_symbol_from_exchange("MATIC", "hyperliquid") == "MATIC-USD"

    def test_convert_from_hyperliquid_with_custom_quote(self):
        """Test conversion from Hyperliquid with custom quote symbol."""
        assert convert_symbol_from_exchange("ETH", "hyperliquid", "USDT") == "ETH-USDT"
        assert convert_symbol_from_exchange("BTC", "hyperliquid", "EUR") == "BTC-EUR"

    def test_convert_from_coinbase_format(self):
        """Test conversion from Coinbase format (full pair) to standard."""
        assert convert_symbol_from_exchange("ETH-USD", "coinbase") == "ETH-USD"
        assert convert_symbol_from_exchange("BTC-USD", "coinbase") == "BTC-USD"
        assert convert_symbol_from_exchange("MATIC-USD", "coinbase") == "MATIC-USD"

    def test_convert_from_coinbase_single_symbol(self):
        """Test conversion from Coinbase single symbol (assumes USD)."""
        assert convert_symbol_from_exchange("ETH", "coinbase") == "ETH-USD"
        assert convert_symbol_from_exchange("BTC", "coinbase") == "BTC-USD"

    def test_convert_from_coinbase_with_custom_quote(self):
        """Test conversion from Coinbase single symbol with custom quote."""
        assert convert_symbol_from_exchange("ETH", "coinbase", "USDT") == "ETH-USDT"

    def test_convert_case_insensitive_exchange_reverse(self):
        """Test that exchange names are case insensitive for reverse conversion."""
        assert convert_symbol_from_exchange("ETH", "HYPERLIQUID") == "ETH-USD"
        assert convert_symbol_from_exchange("ETH-USD", "COINBASE") == "ETH-USD"

    def test_convert_from_unknown_exchange_raises_error(self):
        """Test that unknown exchange raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported exchange: unknown"):
            convert_symbol_from_exchange("ETH", "unknown")

    def test_convert_from_empty_symbol_raises_error(self):
        """Test that empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            convert_symbol_from_exchange("", "hyperliquid")

    def test_convert_from_invalid_pair_format_raises_error(self):
        """Test that invalid pair format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            convert_symbol_from_exchange("ETH-USD-EXTRA", "coinbase")


class TestValidateSymbolFormat:
    """Test the validate_symbol_format function."""

    def test_validate_valid_symbols(self):
        """Test validation of valid symbol formats."""
        assert validate_symbol_format("ETH-USD") is True
        assert validate_symbol_format("BTC_USDT") is True
        assert validate_symbol_format("MATIC/USD") is True
        assert validate_symbol_format("AVAX-EUR") is True

    def test_validate_invalid_symbols(self):
        """Test validation of invalid symbol formats."""
        assert validate_symbol_format("INVALID") is False
        assert validate_symbol_format("") is False
        assert validate_symbol_format("ETH-") is False
        assert validate_symbol_format("-USD") is False
        assert validate_symbol_format("ETH--USD") is False

    def test_validate_edge_cases(self):
        """Test validation of edge cases."""
        assert validate_symbol_format("A-B") is True  # Minimum valid case
        assert validate_symbol_format("VERYLONGTOKEN-VERYLONGQUOTE") is True  # Long symbols
        assert validate_symbol_format("ETH USD") is False  # Space instead of separator


class TestGetSupportedExchanges:
    """Test the get_supported_exchanges function."""

    def test_get_supported_exchanges(self):
        """Test that we get the expected set of supported exchanges."""
        supported = get_supported_exchanges()
        assert isinstance(supported, set)
        assert "hyperliquid" in supported
        assert "coinbase" in supported
        assert len(supported) >= 2  # At least the two we know about

    def test_supported_exchanges_match_config(self):
        """Test that supported exchanges match the configuration."""
        supported = get_supported_exchanges()
        config_exchanges = set(EXCHANGE_SYMBOL_FORMATS.keys())
        assert supported == config_exchanges


class TestGetExchangeFormatInfo:
    """Test the get_exchange_format_info function."""

    def test_get_hyperliquid_format_info(self):
        """Test getting Hyperliquid format information."""
        info = get_exchange_format_info("hyperliquid")
        assert info is not None
        assert info["base_only"] is True
        assert info["separator"] == ""
        assert info["case"] == "upper"

    def test_get_coinbase_format_info(self):
        """Test getting Coinbase format information."""
        info = get_exchange_format_info("coinbase")
        assert info is not None
        assert info["base_only"] is False
        assert info["separator"] == "-"
        assert info["case"] == "upper"

    def test_get_unknown_exchange_format_info(self):
        """Test getting format info for unknown exchange returns None."""
        info = get_exchange_format_info("unknown")
        assert info is None

    def test_get_format_info_case_insensitive(self):
        """Test that exchange names are case insensitive."""
        info1 = get_exchange_format_info("HYPERLIQUID")
        info2 = get_exchange_format_info("hyperliquid")
        assert info1 == info2


class TestParseStandardSymbol:
    """Test the _parse_standard_symbol internal function."""

    def test_parse_dash_separated_symbol(self):
        """Test parsing dash-separated symbols."""
        base, quote = _parse_standard_symbol("ETH-USD")
        assert base == "ETH"
        assert quote == "USD"

    def test_parse_underscore_separated_symbol(self):
        """Test parsing underscore-separated symbols."""
        base, quote = _parse_standard_symbol("BTC_USDT")
        assert base == "BTC"
        assert quote == "USDT"

    def test_parse_slash_separated_symbol(self):
        """Test parsing slash-separated symbols."""
        base, quote = _parse_standard_symbol("MATIC/USD")
        assert base == "MATIC"
        assert quote == "USD"

    def test_parse_symbol_with_whitespace(self):
        """Test parsing symbols with whitespace."""
        base, quote = _parse_standard_symbol(" ETH - USD ")
        assert base == "ETH"
        assert quote == "USD"

    def test_parse_lowercase_symbol(self):
        """Test parsing lowercase symbols (should be converted to uppercase)."""
        base, quote = _parse_standard_symbol("eth-usd")
        assert base == "ETH"
        assert quote == "USD"

    def test_parse_empty_symbol_raises_error(self):
        """Test that empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            _parse_standard_symbol("")

    def test_parse_invalid_format_raises_error(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            _parse_standard_symbol("INVALID")

    def test_parse_empty_components_raises_error(self):
        """Test that empty components raise ValueError."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            _parse_standard_symbol("ETH-")
        with pytest.raises(ValueError, match="Invalid symbol format"):
            _parse_standard_symbol("-USD")
        with pytest.raises(ValueError, match="Invalid symbol format"):
            _parse_standard_symbol("ETH--USD")


class TestIntegrationScenarios:
    """Test integration scenarios and round-trip conversions."""

    def test_round_trip_hyperliquid_conversion(self):
        """Test round-trip conversion for Hyperliquid."""
        original = "ETH-USD"

        # Convert to exchange format
        exchange_symbol = convert_symbol_for_exchange(original, "hyperliquid")
        assert exchange_symbol == "ETH"

        # Convert back to standard format
        standard_symbol = convert_symbol_from_exchange(exchange_symbol, "hyperliquid")
        assert standard_symbol == original

    def test_round_trip_coinbase_conversion(self):
        """Test round-trip conversion for Coinbase."""
        original = "BTC-USD"

        # Convert to exchange format
        exchange_symbol = convert_symbol_for_exchange(original, "coinbase")
        assert exchange_symbol == original

        # Convert back to standard format
        standard_symbol = convert_symbol_from_exchange(exchange_symbol, "coinbase")
        assert standard_symbol == original

    def test_multiple_symbol_formats_normalization(self):
        """Test that different input formats normalize to the same output."""
        symbols = ["ETH-USD", "ETH_USD", "ETH/USD"]

        # All should convert to same Hyperliquid format
        for symbol in symbols:
            assert convert_symbol_for_exchange(symbol, "hyperliquid") == "ETH"

        # All should convert to same Coinbase format
        for symbol in symbols:
            assert convert_symbol_for_exchange(symbol, "coinbase") == "ETH-USD"

    def test_real_world_trading_pairs(self):
        """Test with real-world trading pairs."""
        test_pairs = [
            "BTC-USD", "ETH-USD", "MATIC-USD", "AVAX-USD",
            "SOL-USD", "DOT-USD", "LINK-USD", "UNI-USD"
        ]

        for pair in test_pairs:
            # Test Hyperliquid conversion (should extract base symbol)
            base_symbol = pair.split("-")[0]
            assert convert_symbol_for_exchange(pair, "hyperliquid") == base_symbol

            # Test Coinbase conversion (should remain unchanged)
            assert convert_symbol_for_exchange(pair, "coinbase") == pair

            # Test validation
            assert validate_symbol_format(pair) is True


# Pytest fixtures for common test data
@pytest.fixture
def valid_symbols():
    """Fixture providing valid symbol test data."""
    return [
        "ETH-USD", "BTC-USD", "MATIC-USD", "AVAX-USD",
        "SOL_USDT", "DOT/EUR", "LINK-USDC"
    ]


@pytest.fixture
def invalid_symbols():
    """Fixture providing invalid symbol test data."""
    return [
        "", "INVALID", "ETH-", "-USD", "ETH--USD",
        "ETH USD", "ETH-USD-EXTRA", "123", "A"
    ]


@pytest.fixture
def supported_exchanges():
    """Fixture providing supported exchange names."""
    return ["hyperliquid", "coinbase"]


class TestWithFixtures:
    """Test using pytest fixtures."""

    def test_all_valid_symbols_validate(self, valid_symbols):
        """Test that all valid symbols pass validation."""
        for symbol in valid_symbols:
            assert validate_symbol_format(symbol), f"Symbol {symbol} should be valid"

    def test_all_invalid_symbols_fail_validation(self, invalid_symbols):
        """Test that all invalid symbols fail validation."""
        for symbol in invalid_symbols:
            assert not validate_symbol_format(symbol), f"Symbol {symbol} should be invalid"

    def test_all_exchanges_support_valid_symbols(self, valid_symbols, supported_exchanges):
        """Test that all supported exchanges can convert valid symbols."""
        for symbol in valid_symbols:
            for exchange in supported_exchanges:
                try:
                    result = convert_symbol_for_exchange(symbol, exchange)
                    assert result, f"Conversion of {symbol} for {exchange} should not be empty"
                except Exception as e:
                    pytest.fail(f"Failed to convert {symbol} for {exchange}: {e}")
