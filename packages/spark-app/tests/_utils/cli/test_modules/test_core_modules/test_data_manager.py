"""
Unit tests for CLI DataManager module.
"""
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from core.data_manager import DataManager


class TestDataManager:
    """Test suite for DataManager functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def data_manager(self, temp_cache_dir):
        """Create a DataManager instance with test cache directory."""
        return DataManager(cache_dir=temp_cache_dir)

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': [100 + i * 0.1 for i in range(100)],
            'high': [101 + i * 0.1 for i in range(100)],
            'low': [99 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        })
        return data

    def test_initialization(self, temp_cache_dir):
        """Test DataManager initialization."""
        manager = DataManager(cache_dir=temp_cache_dir)
        assert manager.cache_dir == Path(temp_cache_dir)
        assert manager.cache_enabled is True
        assert manager.cache_timeout == 3600  # Default 1 hour

    def test_initialization_with_custom_settings(self, temp_cache_dir):
        """Test DataManager initialization with custom settings."""
        manager = DataManager(
            cache_dir=temp_cache_dir,
            cache_enabled=False,
            cache_timeout=7200
        )
        assert manager.cache_enabled is False
        assert manager.cache_timeout == 7200

    @patch('core.data_manager.HyperliquidConnector')
    def test_fetch_real_data_success(self, mock_connector, data_manager, sample_market_data):
        """Test successful real data fetching."""
        # Mock the connector
        mock_instance = MagicMock()
        mock_instance.get_historical_data.return_value = sample_market_data
        mock_connector.return_value = mock_instance

        # Fetch data
        data = data_manager.fetch_real_data(
            market="ETH-USD",
            exchange="hyperliquid",
            timeframe="1h",
            days=7
        )

        # Verify
        assert data is not None
        assert len(data) == 100
        assert 'timestamp' in data.columns
        assert 'close' in data.columns
        mock_instance.get_historical_data.assert_called_once()

    @patch('core.data_manager.HyperliquidConnector')
    def test_fetch_real_data_with_caching(self, mock_connector, data_manager, sample_market_data):
        """Test real data fetching with caching."""
        # Mock the connector
        mock_instance = MagicMock()
        mock_instance.get_historical_data.return_value = sample_market_data
        mock_connector.return_value = mock_instance

        # First fetch
        data1 = data_manager.fetch_real_data(
            market="ETH-USD",
            exchange="hyperliquid",
            timeframe="1h",
            days=7
        )

        # Second fetch (should use cache)
        data2 = data_manager.fetch_real_data(
            market="ETH-USD",
            exchange="hyperliquid",
            timeframe="1h",
            days=7
        )

        # Verify connector was called only once
        assert mock_instance.get_historical_data.call_count == 1
        assert len(data1) == len(data2)

    def test_generate_synthetic_data_bull_market(self, data_manager):
        """Test synthetic bull market data generation."""
        data = data_manager.generate_synthetic_data(
            scenario="bull",
            duration_days=30,
            timeframe="1h"
        )

        assert data is not None
        assert len(data) == 30 * 24  # 30 days * 24 hours
        assert 'timestamp' in data.columns
        assert 'close' in data.columns

        # Bull market should have mostly upward trend
        first_close = data['close'].iloc[0]
        last_close = data['close'].iloc[-1]
        assert last_close > first_close

    def test_generate_synthetic_data_bear_market(self, data_manager):
        """Test synthetic bear market data generation."""
        data = data_manager.generate_synthetic_data(
            scenario="bear",
            duration_days=30,
            timeframe="1h"
        )

        assert data is not None
        assert len(data) == 30 * 24

        # Bear market should have mostly downward trend
        first_close = data['close'].iloc[0]
        last_close = data['close'].iloc[-1]
        assert last_close < first_close

    def test_generate_synthetic_data_sideways_market(self, data_manager):
        """Test synthetic sideways market data generation."""
        data = data_manager.generate_synthetic_data(
            scenario="sideways",
            duration_days=30,
            timeframe="1h"
        )

        assert data is not None
        assert len(data) == 30 * 24

        # Sideways market should stay within a range
        first_close = data['close'].iloc[0]
        last_close = data['close'].iloc[-1]
        price_change_pct = abs((last_close - first_close) / first_close)
        assert price_change_pct < 0.15  # Should stay within 15% range

    def test_generate_synthetic_data_high_volatility(self, data_manager):
        """Test synthetic high volatility data generation."""
        data = data_manager.generate_synthetic_data(
            scenario="high_volatility",
            duration_days=30,
            timeframe="1h"
        )

        assert data is not None
        assert len(data) == 30 * 24

        # High volatility should have large price swings
        price_changes = data['close'].pct_change().abs()
        avg_volatility = price_changes.mean()
        assert avg_volatility > 0.01  # Should have significant hourly changes

    def test_generate_synthetic_data_low_volatility(self, data_manager):
        """Test synthetic low volatility data generation."""
        data = data_manager.generate_synthetic_data(
            scenario="low_volatility",
            duration_days=30,
            timeframe="1h"
        )

        assert data is not None
        assert len(data) == 30 * 24

        # Low volatility should have small price swings
        price_changes = data['close'].pct_change().abs()
        avg_volatility = price_changes.mean()
        assert avg_volatility < 0.005  # Should have minimal hourly changes

    def test_generate_synthetic_data_choppy_market(self, data_manager):
        """Test synthetic choppy market data generation."""
        data = data_manager.generate_synthetic_data(
            scenario="choppy",
            duration_days=30,
            timeframe="1h"
        )

        assert data is not None
        assert len(data) == 30 * 24

        # Choppy market should have frequent direction changes
        price_changes = data['close'].diff()
        direction_changes = (price_changes.shift(1) * price_changes < 0).sum()
        change_frequency = direction_changes / len(price_changes)
        assert change_frequency > 0.3  # Should have many direction changes

    def test_generate_synthetic_data_gap_heavy(self, data_manager):
        """Test synthetic gap-heavy market data generation."""
        data = data_manager.generate_synthetic_data(
            scenario="gap_heavy",
            duration_days=30,
            timeframe="1h"
        )

        assert data is not None
        assert len(data) == 30 * 24

        # Gap-heavy market should have some significant gaps
        gaps = abs(data['open'] - data['close'].shift(1))
        gap_threshold = data['close'].std() * 2
        large_gaps = (gaps > gap_threshold).sum()
        assert large_gaps > 5  # Should have several large gaps

    def test_cache_data_and_retrieval(self, data_manager, sample_market_data):
        """Test data caching and retrieval functionality."""
        cache_key = "test_market_1h_7days"

        # Cache the data
        data_manager.cache_data(cache_key, sample_market_data)

        # Retrieve from cache
        cached_data = data_manager.get_cached_data(cache_key)

        assert cached_data is not None
        assert len(cached_data) == len(sample_market_data)
        pd.testing.assert_frame_equal(cached_data, sample_market_data)

    def test_cache_expiration(self, data_manager, sample_market_data):
        """Test cache expiration functionality."""
        # Create manager with very short cache timeout
        manager = DataManager(
            cache_dir=data_manager.cache_dir,
            cache_timeout=0.1  # 0.1 seconds
        )

        cache_key = "test_expiration"
        manager.cache_data(cache_key, sample_market_data)

        # Immediately retrieve (should work)
        cached_data = manager.get_cached_data(cache_key)
        assert cached_data is not None

        # Wait for cache to expire
        import time
        time.sleep(0.2)

        # Try to retrieve again (should be None)
        expired_data = manager.get_cached_data(cache_key)
        assert expired_data is None

    def test_clear_cache(self, data_manager, sample_market_data):
        """Test cache clearing functionality."""
        cache_key = "test_clear"
        data_manager.cache_data(cache_key, sample_market_data)

        # Verify data is cached
        cached_data = data_manager.get_cached_data(cache_key)
        assert cached_data is not None

        # Clear cache
        data_manager.clear_cache()

        # Verify data is no longer cached
        cleared_data = data_manager.get_cached_data(cache_key)
        assert cleared_data is None

    def test_validate_data_quality_valid(self, data_manager, sample_market_data):
        """Test data quality validation with valid data."""
        is_valid, errors = data_manager.validate_data_quality(sample_market_data)
        assert is_valid
        assert len(errors) == 0

    def test_validate_data_quality_missing_columns(self, data_manager):
        """Test data quality validation with missing columns."""
        invalid_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1H'),
            'close': range(10)
            # Missing required columns: open, high, low, volume
        })

        is_valid, errors = data_manager.validate_data_quality(invalid_data)
        assert not is_valid
        assert len(errors) > 0

    def test_validate_data_quality_insufficient_data(self, data_manager):
        """Test data quality validation with insufficient data."""
        small_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='1H'),
            'open': range(5),
            'high': range(1, 6),
            'low': range(5),
            'close': range(5),
            'volume': range(5)
        })

        is_valid, errors = data_manager.validate_data_quality(small_data, min_rows=10)
        assert not is_valid
        assert "insufficient data" in str(errors).lower()

    def test_get_timeframe_minutes(self, data_manager):
        """Test timeframe to minutes conversion."""
        assert data_manager.get_timeframe_minutes("1m") == 1
        assert data_manager.get_timeframe_minutes("5m") == 5
        assert data_manager.get_timeframe_minutes("1h") == 60
        assert data_manager.get_timeframe_minutes("4h") == 240
        assert data_manager.get_timeframe_minutes("1d") == 1440

    def test_get_timeframe_minutes_invalid(self, data_manager):
        """Test timeframe to minutes conversion with invalid input."""
        with pytest.raises(ValueError):
            data_manager.get_timeframe_minutes("invalid")

    def test_estimate_data_size(self, data_manager):
        """Test data size estimation."""
        # Test for 1 hour timeframe, 7 days
        size = data_manager.estimate_data_size(timeframe="1h", days=7)
        expected_rows = 7 * 24  # 7 days * 24 hours
        assert size == expected_rows

        # Test for 1 day timeframe, 30 days
        size = data_manager.estimate_data_size(timeframe="1d", days=30)
        expected_rows = 30  # 30 days
        assert size == expected_rows

    def test_error_handling_invalid_exchange(self, data_manager):
        """Test error handling for invalid exchange."""
        with pytest.raises(ValueError):
            data_manager.fetch_real_data(
                market="ETH-USD",
                exchange="invalid_exchange",
                timeframe="1h",
                days=7
            )

    def test_error_handling_invalid_scenario(self, data_manager):
        """Test error handling for invalid scenario."""
        with pytest.raises(ValueError):
            data_manager.generate_synthetic_data(
                scenario="invalid_scenario",
                duration_days=30,
                timeframe="1h"
            )

    def test_multi_timeframe_data_handling(self, data_manager, sample_market_data):
        """Test handling of multi-timeframe data requirements."""
        # Cache data for different timeframes
        data_1h = sample_market_data.copy()
        data_4h = sample_market_data.iloc[::4].copy()  # Every 4th row for 4h data

        data_manager.cache_data("ETH-USD_1h_7days", data_1h)
        data_manager.cache_data("ETH-USD_4h_7days", data_4h)

        # Retrieve both timeframes
        timeframes = ["1h", "4h"]
        data_dict = {}
        for tf in timeframes:
            cache_key = f"ETH-USD_{tf}_7days"
            data_dict[tf] = data_manager.get_cached_data(cache_key)

        assert "1h" in data_dict
        assert "4h" in data_dict
        assert len(data_dict["1h"]) > len(data_dict["4h"])

    def test_concurrent_access(self, data_manager, sample_market_data):
        """Test concurrent access to data manager."""
        import threading
        import time

        results = {}
        errors = []

        def fetch_and_cache(thread_id):
            try:
                cache_key = f"test_concurrent_{thread_id}"
                data_manager.cache_data(cache_key, sample_market_data)
                time.sleep(0.1)  # Small delay
                results[thread_id] = data_manager.get_cached_data(cache_key)
            except Exception as e:
                errors.append((thread_id, e))

        # Create and start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=fetch_and_cache, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors and all threads succeeded
        assert len(errors) == 0
        assert len(results) == 5
        for i in range(5):
            assert results[i] is not None
            assert len(results[i]) == len(sample_market_data)
