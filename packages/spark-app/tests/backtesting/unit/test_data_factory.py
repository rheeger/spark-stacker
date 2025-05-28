"""
Tests for the synthetic data generation factory.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from tests._helpers.data_factory import make_price_dataframe


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Create a deterministic hash of a DataFrame for testing purposes.
    This allows us to check if the output is consistent with the same seed.
    """
    # Convert DataFrame to a consistent string representation
    # Round floating point values to reduce sensitivity to minor platform differences
    df_rounded = df.copy()
    for col in df_rounded.columns:
        if df_rounded[col].dtype in (np.float64, np.float32):
            df_rounded[col] = df_rounded[col].round(6)

    # Convert to JSON for consistent serialization
    df_json = json.dumps(df_rounded.to_dict(orient="records"), sort_keys=True)

    # Create hash
    return hashlib.md5(df_json.encode()).hexdigest()


def test_make_price_dataframe_deterministic():
    """Test that the price data factory produces consistent output with the same seed."""
    # Generate data with a fixed seed
    test_seed = 42
    df1 = make_price_dataframe(rows=50, pattern="trend", noise=0.5, seed=test_seed)
    df2 = make_price_dataframe(rows=50, pattern="trend", noise=0.5, seed=test_seed)

    # Verify that both DataFrames have identical hash values
    hash1 = hash_dataframe(df1)
    hash2 = hash_dataframe(df2)

    assert hash1 == hash2, "Data factory should produce identical output with the same seed"

    # The exact hash value is included in the assertion message to help with debugging
    # if this test fails in the future
    assert hash1 == hash_dataframe(df2), f"Expected hash: {hash1}"


def test_different_patterns_produce_different_data():
    """Test that different pattern types produce different data."""
    # Generate data with same seed but different patterns
    test_seed = 42
    trend_df = make_price_dataframe(rows=50, pattern="trend", noise=0.5, seed=test_seed)
    mean_revert_df = make_price_dataframe(rows=50, pattern="mean_revert", noise=0.5, seed=test_seed)
    sideways_df = make_price_dataframe(rows=50, pattern="sideways", noise=0.5, seed=test_seed)

    # Hash each DataFrame
    trend_hash = hash_dataframe(trend_df)
    mean_revert_hash = hash_dataframe(mean_revert_df)
    sideways_hash = hash_dataframe(sideways_df)

    # Check that each pattern produces different data
    assert trend_hash != mean_revert_hash, "Trend and mean reversion patterns should differ"
    assert trend_hash != sideways_hash, "Trend and sideways patterns should differ"
    assert mean_revert_hash != sideways_hash, "Mean reversion and sideways patterns should differ"


def test_price_dataframe_ohlc_relationships():
    """Test that generated OHLC data maintains proper relationships."""
    # Generate test data
    df = make_price_dataframe(rows=100, pattern="trend", noise=1.0, seed=42)

    # Check that High is always >= Open, Close and Low
    assert (df["high"] >= df["open"]).all(), "High should be >= Open"
    assert (df["high"] >= df["close"]).all(), "High should be >= Close"
    assert (df["high"] >= df["low"]).all(), "High should be >= Low"

    # Check that Low is always <= Open, Close and High
    assert (df["low"] <= df["open"]).all(), "Low should be <= Open"
    assert (df["low"] <= df["close"]).all(), "Low should be <= Close"
    assert (df["low"] <= df["high"]).all(), "Low should be <= High"

    # All values should be positive
    assert (df["open"] > 0).all(), "Open prices should be positive"
    assert (df["high"] > 0).all(), "High prices should be positive"
    assert (df["low"] > 0).all(), "Low prices should be positive"
    assert (df["close"] > 0).all(), "Close prices should be positive"
    assert (df["volume"] > 0).all(), "Volume should be positive"


def test_noise_parameter_affects_volatility():
    """Test that the noise parameter affects price volatility."""
    # Generate data with different noise levels
    low_noise_df = make_price_dataframe(rows=100, pattern="trend", noise=0.1, seed=42)
    high_noise_df = make_price_dataframe(rows=100, pattern="trend", noise=1.0, seed=42)

    # Calculate volatility as standard deviation of returns
    low_noise_returns = low_noise_df["close"].pct_change().dropna()
    high_noise_returns = high_noise_df["close"].pct_change().dropna()

    low_noise_volatility = low_noise_returns.std()
    high_noise_volatility = high_noise_returns.std()

    # Higher noise should result in higher volatility
    assert high_noise_volatility > low_noise_volatility, "Higher noise parameter should produce higher volatility"
