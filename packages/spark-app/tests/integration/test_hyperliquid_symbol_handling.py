import asyncio
import logging
import os
import time
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from app.connectors.base_connector import MarketType, OrderSide, OrderStatus
from app.connectors.hyperliquid_connector import HyperliquidConnector
from app.core.trading_engine import TradingEngine
from app.indicators.base_indicator import Signal, SignalDirection
from app.risk_management.risk_manager import RiskManager
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from shared .env
load_dotenv(os.path.join(os.path.dirname(__file__), "../../../shared/.env"))

class TestHyperliquidSymbolHandling:
    """
    Integration tests for Hyperliquid symbol handling to ensure proper symbol translation
    between ETH-USD format and ETH format.
    """

    def test_hyperliquid_translate_symbol(self):
        """
        Test the translate_symbol method in the HyperliquidConnector directly.
        """
        # Create a connector instance
        connector = HyperliquidConnector(testnet=True)

        # Test with composite symbol format
        symbol = "ETH-USD"
        translated = connector.translate_symbol(symbol)
        assert translated == "ETH", f"Expected 'ETH', got '{translated}'"

        # Test with single symbol format
        symbol = "ETH"
        translated = connector.translate_symbol(symbol)
        assert translated == "ETH", f"Expected 'ETH', got '{translated}'"

        # Test with different symbol
        symbol = "BTC-USD"
        translated = connector.translate_symbol(symbol)
        assert translated == "BTC", f"Expected 'BTC', got '{translated}'"

    @pytest.mark.asyncio
    async def test_trading_engine_symbol_handling(self, mock_connector, mock_risk_manager):
        """
        Test that the trading engine correctly handles symbol conversion when using Hyperliquid.

        This test verifies that the trading engine properly handles symbols in the format "ETH-USD"
        for Hyperliquid exchanges that expect just "ETH".
        """
        # Create a specialized mock for HyperliquidConnector
        hyperliquid_mock = MagicMock(spec=HyperliquidConnector)

        # Set properties required by trading engine
        hyperliquid_mock.exchange_type = "hyperliquid"
        hyperliquid_mock.supports_derivatives = True
        hyperliquid_mock.market_types = [MarketType.PERPETUAL]
        hyperliquid_mock.is_connected = True

        # Basic necessary mocks
        hyperliquid_mock.get_ticker.return_value = {"symbol": "ETH", "last_price": 1600.0}
        hyperliquid_mock.get_account_balance.return_value = {"PERP_USDC": 100.0}
        hyperliquid_mock.get_leverage_tiers.return_value = [
            {
                "min_notional": 0,
                "max_notional": float("inf"),
                "max_leverage": 25.0,
                "maintenance_margin_rate": 0.02,
                "initial_margin_rate": 0.04,
            }
        ]

        # Create a trading engine with our specialized mock
        trading_engine = TradingEngine(
            main_connector=hyperliquid_mock,
            hedge_connector=hyperliquid_mock,
            risk_manager=mock_risk_manager,
            dry_run=True,  # Use dry_run to avoid place_order issues
            polling_interval=1,
            max_parallel_trades=1,
        )

        # Start the trading engine
        trading_engine.start()

        try:
            # Create a test signal with ETH-USD format
            test_signal = Signal(
                direction=SignalDirection.BUY,
                symbol="ETH-USD",  # Use the format that was causing issues
                indicator="test_indicator",
                confidence=0.8,
                timestamp=int(time.time() * 1000),
                params={"price": 1600.0},
            )

            # Process the signal - our test passes if this doesn't raise an exception
            # and returns True, which means processing was successful
            result = await trading_engine.process_signal(test_signal)

            # Verify signal was processed successfully
            assert result is True, "Signal processing should succeed"

            # Verify we have an active trade
            assert len(trading_engine.active_trades) > 0, "Should have at least one active trade"

        finally:
            # Stop the trading engine
            trading_engine.stop()

    @pytest.mark.asyncio
    async def test_error_handling_for_invalid_symbols(self, mock_risk_manager):
        """
        Test that the system handles invalid symbols gracefully.

        This test verifies that our improved error handling allows the system to
        process signals even for non-existent symbols, by using default values
        instead of throwing exceptions.
        """
        # Create a HyperliquidConnector mock
        connector = MagicMock(spec=HyperliquidConnector)
        connector.exchange_type = "hyperliquid"
        connector.supports_derivatives = True
        connector.market_types = [MarketType.PERPETUAL]
        connector.is_connected = True

        # Basic necessary mocks
        connector.get_ticker.return_value = {"symbol": "XYZ", "last_price": 100.0}
        connector.get_account_balance.return_value = {"PERP_USDC": 1000.0}
        connector.get_leverage_tiers.return_value = [
            {
                "min_notional": 0,
                "max_notional": float("inf"),
                "max_leverage": 10.0,
                "maintenance_margin_rate": 0.02,
                "initial_margin_rate": 0.04,
            }
        ]

        # Create a trading engine with dry_run=True
        engine = TradingEngine(
            main_connector=connector,
            hedge_connector=connector,
            risk_manager=mock_risk_manager,
            dry_run=True,
        )

        # Start the engine
        engine.start()

        try:
            # Create a signal with an invalid symbol
            signal = Signal(
                direction=SignalDirection.BUY,
                symbol="XYZ-USD",  # Non-existent symbol
                indicator="test_indicator",
                confidence=0.8,
                timestamp=int(time.time() * 1000),
            )

            # Process the signal - our test passes if this doesn't raise an exception
            # and returns True, which means processing was successful
            result = await engine.process_signal(signal)

            # Verify signal was processed successfully
            assert result is True, "Signal processing should succeed even with invalid symbol"

            # Verify we have an active trade
            assert len(engine.active_trades) > 0, "Should have at least one active trade"

        finally:
            engine.stop()

    @pytest.mark.asyncio
    async def test_get_leverage_tiers_with_signal(self, mock_risk_manager):
        """
        Test that get_leverage_tiers correctly handles symbol translation when processing a signal.
        This test verifies the entire flow from signal receipt to leverage calculation.
        """
        # Create a real HyperliquidConnector instance
        connector = HyperliquidConnector(testnet=True)
        connector.connect()

        # Create a mock risk manager that logs the symbol format
        symbol_format_used = None
        original_calc_size = mock_risk_manager.calculate_position_size

        def mock_calc_size(*args, **kwargs):
            nonlocal symbol_format_used
            symbol_format_used = kwargs.get('symbol')
            return original_calc_size(*args, **kwargs)

        mock_risk_manager.calculate_position_size = mock_calc_size

        # Create a trading engine with the real connector
        engine = TradingEngine(
            main_connector=connector,
            hedge_connector=connector,
            risk_manager=mock_risk_manager,
            dry_run=True,
            polling_interval=1,
            max_parallel_trades=1,
        )

        # Start the engine
        engine.start()

        try:
            # Create a test signal
            test_signal = Signal(
                direction=SignalDirection.BUY,
                symbol="ETH-USD",
                indicator="test_indicator",
                confidence=0.8,
                timestamp=int(time.time() * 1000),
                params={"price": 1600.0},
            )

            # Process the signal
            result = await engine.process_signal(test_signal)

            # Verify signal was processed
            assert result is True, "Signal processing should succeed"

            # Verify the symbol format used in get_leverage_tiers
            assert symbol_format_used == "ETH-USD", f"Expected symbol format 'ETH-USD', got '{symbol_format_used}'"

            # Get leverage tiers directly to verify format
            tiers = connector.get_leverage_tiers("ETH-USD")
            assert len(tiers) > 0, "Should get valid leverage tiers"
            assert isinstance(tiers[0]["max_leverage"], (int, float)), "Should have valid max_leverage"
            assert tiers[0]["max_leverage"] > 0, "Max leverage should be positive"

        finally:
            engine.stop()
            connector.disconnect()

    @pytest.mark.slow
    def test_leverage_tiers_format_consistency(self):
        """Test that leverage tiers are consistently formatted across different symbols."""
        # Create a connector instance
        connector = HyperliquidConnector(testnet=True)
        connector.connect()

        symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

        for symbol in symbols:
            leverage_tiers = connector.get_leverage_tiers(symbol)

            # Check that we get a list of tiers
            assert isinstance(leverage_tiers, list)
            assert len(leverage_tiers) > 0

            # Check the format of each tier based on the actual API response structure
            for tier in leverage_tiers:
                # Verify the expected fields are present
                assert "initial_margin_fraction" in tier
                assert "maintenance_margin_fraction" in tier
                assert "max_leverage" in tier

                # Type checking
                assert isinstance(tier["initial_margin_fraction"], (int, float))
                assert isinstance(tier["maintenance_margin_fraction"], (int, float))
                assert isinstance(tier["max_leverage"], (int, float))

                # Value checking
                assert tier["initial_margin_fraction"] > 0
                assert tier["maintenance_margin_fraction"] > 0
                assert tier["max_leverage"] > 0
                assert tier["maintenance_margin_fraction"] <= tier["initial_margin_fraction"]

        # Clean up
        connector.disconnect()

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_signal_order_preparation_with_real_api(self):
        """Test the entire flow of signal to order preparation with the real API."""
        # Get wallet credentials from environment variables
        wallet_address = os.environ.get("WALLET_ADDRESS")
        private_key = os.environ.get("PRIVATE_KEY")

        if not wallet_address or not private_key:
            pytest.skip("WALLET_ADDRESS and PRIVATE_KEY environment variables are required")

        # Create a real connector instance with production credentials
        connector = HyperliquidConnector(
            testnet=False,  # Use production
            wallet_address=wallet_address,
            private_key=private_key
        )
        connector.connect()

        # Mock the balance check method to return test balance
        original_get_account_balance = connector.get_account_balance
        connector.get_account_balance = MagicMock(return_value={"PERP_USDC": 1000.0})

        # Create a real risk manager with test configuration
        risk_manager = RiskManager(
            max_account_risk_pct=1.0,  # 1% max risk
            max_leverage=10.0,  # Conservative max leverage
            max_position_size_usd=1000.0,  # Small max position size
            max_positions=3,
            min_margin_buffer_pct=20.0
        )

        # Create a trading engine with dry_run=True
        engine = TradingEngine(
            main_connector=connector,
            hedge_connector=connector,
            risk_manager=risk_manager,
            dry_run=True,
            polling_interval=1,
            max_parallel_trades=1,
        )

        # Start the engine
        engine.start()

        try:
            # Create test signals for both BUY and SELL
            test_signals = [
                Signal(
                    direction=SignalDirection.BUY,
                    symbol="ETH-USD",
                    indicator="test_indicator",
                    confidence=0.8,
                    timestamp=int(time.time() * 1000),
                    params={"price": 1600.0},
                ),
                Signal(
                    direction=SignalDirection.SELL,
                    symbol="ETH-USD",
                    indicator="test_indicator",
                    confidence=0.8,
                    timestamp=int(time.time() * 1000),
                    params={"price": 1600.0},
                )
            ]

            for signal in test_signals:
                # Process the signal
                logger.info(f"Processing {signal.direction.value} signal for ETH-USD")
                result = await engine.process_signal(signal)

                # Verify signal was processed
                assert result is True, f"Signal processing should succeed for {signal.direction.value}"

                # Get the active trade record
                active_trades = engine.get_active_trades()
                assert "ETH-USD" in active_trades, f"Should have active trade for ETH-USD after {signal.direction.value} signal"

                trade = active_trades["ETH-USD"]

                # Verify trade record structure
                assert "main_position" in trade, "Trade should have main position"
                assert "symbol" in trade, "Trade should have symbol"
                assert "timestamp" in trade, "Trade should have timestamp"
                assert "status" in trade, "Trade should have status"
                assert "market_type" in trade, "Trade should have market type"

                # Verify main position details
                main_pos = trade["main_position"]
                assert "side" in main_pos, "Main position should have side"
                assert "size" in main_pos, "Main position should have size"
                assert "leverage" in main_pos, "Main position should have leverage"
                assert "entry_price" in main_pos, "Main position should have entry price"
                assert "order_id" in main_pos, "Main position should have order ID"
                assert "status" in main_pos, "Main position should have status"

                # Verify correct side translation
                if signal.direction == SignalDirection.BUY:
                    assert main_pos["side"] == "BUY", "BUY signal should create BUY main position"
                else:
                    assert main_pos["side"] == "SELL", "SELL signal should create SELL main position"

                # Verify hedge position if hedging is enabled
                if "hedge_position" in trade:
                    hedge_pos = trade["hedge_position"]
                    assert "side" in hedge_pos, "Hedge position should have side"
                    assert "size" in hedge_pos, "Hedge position should have size"
                    assert "leverage" in hedge_pos, "Hedge position should have leverage"
                    assert "entry_price" in hedge_pos, "Hedge position should have entry price"
                    assert "order_id" in hedge_pos, "Hedge position should have order ID"
                    assert "status" in hedge_pos, "Hedge position should have status"

                    # Verify hedge side is opposite of main position
                    if signal.direction == SignalDirection.BUY:
                        assert hedge_pos["side"] == "SELL", "BUY signal should create SELL hedge position"
                    else:
                        assert hedge_pos["side"] == "BUY", "SELL signal should create BUY hedge position"

                # Verify position sizes are reasonable
                assert main_pos["size"] > 0, "Main position size should be positive"
                assert main_pos["leverage"] > 0, "Main position leverage should be positive"
                if "hedge_position" in trade:
                    assert trade["hedge_position"]["size"] > 0, "Hedge position size should be positive"
                    assert trade["hedge_position"]["leverage"] > 0, "Hedge position leverage should be positive"
                    # Hedge size should be smaller than main position
                    assert trade["hedge_position"]["size"] < main_pos["size"], "Hedge size should be smaller than main position"

                # Clear active trades before next signal
                engine.active_trades.clear()

            logger.info("Successfully tested both BUY and SELL signals with real API")

        finally:
            # Restore original balance method
            connector.get_account_balance = original_get_account_balance
            # Stop the engine and disconnect
            engine.stop()
            connector.disconnect()
