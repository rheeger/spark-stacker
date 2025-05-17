import os
import shutil
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from app.backtesting.market_dataset_generator import (MARKET_REGIMES,
                                                      MarketDatasetGenerator)


class TestMarketDatasetGenerator(unittest.TestCase):
    """Test cases for the MarketDatasetGenerator class."""

    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a temporary directory for test data
        self.test_data_dir = "tests/test_data/test_market_datasets"
        os.makedirs(self.test_data_dir, exist_ok=True)

        # Initialize the generator with the test directory
        self.generator = MarketDatasetGenerator(data_dir=self.test_data_dir)

        # Create sample test data
        self.sample_data = pd.DataFrame({
            'timestamp': list(range(1000000, 1000000 + 10 * 3600000, 3600000)),
            'open': [100.0 + i for i in range(10)],
            'high': [105.0 + i for i in range(10)],
            'low': [95.0 + i for i in range(10)],
            'close': [101.0 + i for i in range(10)],
            'volume': [1000.0 + i * 100 for i in range(10)]
        })

    def tearDown(self):
        """Clean up after each test."""
        # Remove the test directory
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

    @patch('app.backtesting.market_dataset_generator.load_dotenv')
    def test_load_environment(self, mock_load_dotenv):
        """Test loading environment variables."""
        # Mock environment variables
        with patch.dict(os.environ, {
            'WALLET_ADDRESS': 'test_wallet',
            'PRIVATE_KEY': 'test_private_key',
            'KRAKEN_API_KEY': 'test_key',
            'KRAKEN_API_SECRET': 'test_secret'
        }):
            result = self.generator.load_environment()
            assert result is True
            mock_load_dotenv.assert_called_once()

    @patch('app.backtesting.market_dataset_generator.load_dotenv')
    def test_load_environment_no_credentials(self, mock_load_dotenv):
        """Test loading environment without credentials."""
        # Mock empty environment
        with patch.dict(os.environ, {}, clear=True):
            result = self.generator.load_environment()
            assert result is False
            mock_load_dotenv.assert_called_once()

    @patch('app.backtesting.market_dataset_generator.ConnectorFactory')
    @patch('app.backtesting.market_dataset_generator.ExchangeDataSource')
    def test_create_exchange_connector_hyperliquid(self, mock_exchange_data_source, mock_factory):
        """Test creating Hyperliquid connector."""
        # Mock successful connector creation
        mock_connector = MagicMock()
        mock_connector.connect.return_value = True
        mock_factory.create_connector.return_value = mock_connector

        # Mock environment variables
        with patch.dict(os.environ, {
            'WALLET_ADDRESS': 'test_wallet',
            'PRIVATE_KEY': 'test_private_key'
        }):
            result = self.generator.create_exchange_connector('hyperliquid')
            assert result is True
            mock_factory.create_connector.assert_called_once()
            mock_exchange_data_source.assert_called_once_with(mock_connector)

    @patch('app.backtesting.market_dataset_generator.ConnectorFactory')
    def test_create_exchange_connector_failure(self, mock_factory):
        """Test connector creation failure."""
        # Mock failed connector creation
        mock_factory.create_connector.return_value = None

        # Mock environment variables
        with patch.dict(os.environ, {
            'WALLET_ADDRESS': 'test_wallet',
            'PRIVATE_KEY': 'test_private_key'
        }):
            result = self.generator.create_exchange_connector('hyperliquid')
            assert result is False

    def test_generate_standard_datasets(self):
        """Test generating standard datasets."""
        # Mock dependencies
        self.generator.load_environment = MagicMock(return_value=True)
        self.generator.create_exchange_connector = MagicMock(return_value=True)

        # Mock data manager methods
        mock_data_manager = MagicMock()
        mock_data_manager.get_multiple_timeframes.return_value = {
            '1h': self.sample_data,
            '4h': self.sample_data,
            '1d': self.sample_data
        }
        mock_data_manager.clean_data.return_value = self.sample_data

        # Set the mocked data_manager
        self.generator.data_manager = mock_data_manager

        # Call generate with limited symbols list
        self.generator.generate_standard_datasets(symbols=['BTC'], exchange_type='kraken')

        # Check expected calls
        self.generator.load_environment.assert_called_once()
        self.generator.create_exchange_connector.assert_called_once_with('kraken')

        # Verify the number of calls to get_multiple_timeframes equals the number of date ranges across all regimes
        # For BTC, we have 2 ranges for each regime (bull, bear, sideways)
        expected_multiple_timeframes_calls = 6  # 2 date ranges * 3 regimes
        assert mock_data_manager.get_multiple_timeframes.call_count == expected_multiple_timeframes_calls

    def test_list_available_datasets_empty(self):
        """Test listing datasets when none exist."""
        # Should return empty dict when no datasets exist
        result = self.generator.list_available_datasets()
        expected = {'bull': [], 'bear': [], 'sideways': []}
        assert result == expected

    def test_list_available_datasets(self):
        """Test listing datasets after creating some."""
        # Create test dataset files
        for regime in MARKET_REGIMES.keys():
            regime_dir = os.path.join(self.test_data_dir, regime)
            os.makedirs(regime_dir, exist_ok=True)

            # Create a sample file
            file_path = os.path.join(regime_dir, f"BTC_1h_{regime}_1.csv")
            self.sample_data.to_csv(file_path, index=False)

        # Get available datasets
        result = self.generator.list_available_datasets()

        # Check that each regime has one file
        for regime in MARKET_REGIMES.keys():
            assert len(result[regime]) == 1
            assert f"BTC_1h_{regime}_1.csv" in result[regime]

    def test_market_regimes_structure(self):
        """Test the structure of predefined market regimes."""
        # Check that all required regimes exist
        assert set(MARKET_REGIMES.keys()) == {'bull', 'bear', 'sideways'}

        # Check that each regime has data for BTC and ETH
        for regime, symbols_data in MARKET_REGIMES.items():
            assert 'BTC' in symbols_data
            assert 'ETH' in symbols_data

            # Check date ranges
            for symbol, date_ranges in symbols_data.items():
                assert len(date_ranges) > 0
                for start_date, end_date in date_ranges:
                    assert isinstance(start_date, datetime)
                    assert isinstance(end_date, datetime)
                    assert start_date < end_date  # Start date should be before end date


if __name__ == '__main__':
    unittest.main()
