import numpy as np
import pandas as pd
import pytest


def test_synthetic_data_availability(sample_price_data):
    """Test that sample_price_data fixture provides usable data."""
    # Verify that we have data (either real or synthetic)
    assert sample_price_data is not None
    assert isinstance(sample_price_data, pd.DataFrame)
    assert len(sample_price_data) > 0

    # Check required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        assert col in sample_price_data.columns

    # Verify data integrity
    assert (sample_price_data['high'] >= sample_price_data['low']).all(), "High should always be >= Low"

    # Print data source information for debugging
    print(f"\nSample price data info:")
    print(f"- Shape: {sample_price_data.shape}")
    print(f"- Date range: {sample_price_data.index.min()} to {sample_price_data.index.max()}")
    print(f"- Average price: {sample_price_data['close'].mean():.2f}")

    # Test passes if we get here
    assert True
