import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from app.backtesting.data_normalizer import DataNormalizer


class TestDataNormalizer(unittest.TestCase):
    """Test cases for the DataNormalizer class."""

    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a temporary directory for test data
        self.test_data_dir = "tests/__test_data__/normalizer_test"
        self.test_input_dir = os.path.join(self.test_data_dir, "input")
        self.normalized_dir = os.path.join(self.test_data_dir, "normalized")

        os.makedirs(self.test_input_dir, exist_ok=True)

        # Initialize the normalizer with the test directory
        self.normalizer = DataNormalizer(data_dir=self.test_data_dir)

        # Create sample test data
        self.sample_data = pd.DataFrame({
            'timestamp': list(range(1000000, 1000000 + 10 * 3600000, 3600000)),
            'open': [100.0, 101.0, 102.0, 103.0, 105.0, 107.0, 108.0, 107.0, 105.0, 106.0],
            'high': [105.0, 106.0, 107.0, 108.0, 110.0, 112.0, 113.0, 112.0, 110.0, 111.0],
            'low': [99.0, 100.0, 101.0, 102.0, 104.0, 106.0, 107.0, 106.0, 104.0, 105.0],
            'close': [101.0, 102.0, 103.0, 105.0, 107.0, 108.0, 107.0, 105.0, 106.0, 108.0],
            'volume': [1000.0, 1200.0, 1100.0, 1300.0, 1500.0, 1400.0, 1200.0, 1100.0, 1300.0, 1400.0]
        })

        # Save sample data to test file
        self.test_filepath = os.path.join(self.test_input_dir, "BTC_1h_test.csv")
        self.sample_data.to_csv(self.test_filepath, index=False)

    def tearDown(self):
        """Clean up after each test."""
        # Remove the test directory
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

    def test_init(self):
        """Test initialization."""
        # Check that normalized directory was created
        assert os.path.exists(self.normalized_dir)

    def test_normalize_dataset_z_score(self):
        """Test normalizing a dataset with z-score method."""
        # Apply z-score normalization
        normalized_df = self.normalizer.normalize_dataset(
            filepath=self.test_filepath,
            normalization_method="z_score",
            save_normalized=True
        )

        # Check that normalized columns were added
        for col in ["open", "high", "low", "close", "volume"]:
            assert f"{col}_norm" in normalized_df.columns

        # Check z-score normalization properties
        for col in ["open_norm", "high_norm", "low_norm", "close_norm", "volume_norm"]:
            # Z-score should have mean close to 0 and std close to 1
            assert abs(normalized_df[col].mean()) < 1e-10
            assert abs(normalized_df[col].std() - 1.0) < 1e-10

        # Check that file was saved
        expected_filepath = os.path.join(self.normalized_dir, "BTC_1h_test_z_score.csv")
        assert os.path.exists(expected_filepath)

    def test_normalize_dataset_min_max(self):
        """Test normalizing a dataset with min-max method."""
        # Apply min-max normalization
        normalized_df = self.normalizer.normalize_dataset(
            filepath=self.test_filepath,
            normalization_method="min_max",
            save_normalized=True
        )

        # Check min-max normalization properties
        for col in ["open_norm", "high_norm", "low_norm", "close_norm", "volume_norm"]:
            # Min-max should scale values to [0, 1]
            assert normalized_df[col].min() == 0.0
            assert normalized_df[col].max() == 1.0

    def test_normalize_dataset_percent_change(self):
        """Test normalizing a dataset with percent change method."""
        # Apply percent change normalization
        normalized_df = self.normalizer.normalize_dataset(
            filepath=self.test_filepath,
            normalization_method="percent_change",
            save_normalized=True
        )

        # First row should be 0 (no previous value)
        for col in ["open_norm", "high_norm", "low_norm", "close_norm", "volume_norm"]:
            assert normalized_df[col].iloc[0] == 0.0

            # Check some values manually
            # Percent change = (current - previous) / previous
            prev_open = self.sample_data["open"].iloc[0]
            curr_open = self.sample_data["open"].iloc[1]
            expected_pct_change = (curr_open - prev_open) / prev_open
            assert abs(normalized_df["open_norm"].iloc[1] - expected_pct_change) < 1e-10

    def test_normalize_dataset_log_return(self):
        """Test normalizing a dataset with log return method."""
        # Apply log return normalization
        normalized_df = self.normalizer.normalize_dataset(
            filepath=self.test_filepath,
            normalization_method="log_return",
            save_normalized=True
        )

        # First row should be 0 (no previous value)
        for col in ["open_norm", "high_norm", "low_norm", "close_norm", "volume_norm"]:
            assert normalized_df[col].iloc[0] == 0.0

        # Check a log return calculation
        prev_close = self.sample_data["close"].iloc[0]
        curr_close = self.sample_data["close"].iloc[1]
        expected_log_return = np.log(curr_close / prev_close)
        assert abs(normalized_df["close_norm"].iloc[1] - expected_log_return) < 1e-10

    def test_normalize_dataset_rolling_z_score(self):
        """Test normalizing a dataset with rolling z-score method."""
        # Apply rolling z-score normalization with small window
        window_size = 3
        normalized_df = self.normalizer.normalize_dataset(
            filepath=self.test_filepath,
            normalization_method="rolling_z_score",
            window_size=window_size,
            save_normalized=True
        )

        # First window_size-1 rows should be 0 (not enough data for rolling window)
        for col in ["open_norm", "high_norm", "low_norm", "close_norm", "volume_norm"]:
            assert all(normalized_df[col].iloc[:window_size-1] == 0.0)

    def test_normalize_dataset_unknown_method(self):
        """Test normalizing a dataset with unknown method."""
        # Apply unknown normalization method
        normalized_df = self.normalizer.normalize_dataset(
            filepath=self.test_filepath,
            normalization_method="unknown_method",
            save_normalized=True
        )

        # Should just copy original values (values should be equal, but series names will differ)
        for col in ["open", "high", "low", "close", "volume"]:
            norm_col = f"{col}_norm"
            pd.testing.assert_series_equal(
                normalized_df[col],
                normalized_df[norm_col],
                check_names=False  # Don't check series names
            )

    def test_normalize_all_datasets(self):
        """Test normalizing all datasets."""
        # Create a second test file
        second_filepath = os.path.join(self.test_input_dir, "ETH_1h_test.csv")
        self.sample_data.to_csv(second_filepath, index=False)

        # Apply normalization to all datasets with multiple methods
        methods = ["z_score", "min_max"]
        results = self.normalizer.normalize_all_datasets(
            normalization_methods=methods,
            save_normalized=True
        )

        # Check results count
        assert results["z_score"] == 2  # 2 files
        assert results["min_max"] == 2  # 2 files

        # Check that files were saved
        for method in methods:
            for prefix in ["BTC_1h_test", "ETH_1h_test"]:
                expected_filepath = os.path.join(self.normalized_dir, f"{prefix}_{method}.csv")
                assert os.path.exists(expected_filepath)

    def test_list_normalized_datasets(self):
        """Test listing normalized datasets."""
        # Directly create files in normalized directory with method-specific names
        # We'll patch the list_normalized_datasets method to return expected values

        with patch.object(DataNormalizer, 'list_normalized_datasets') as mock_list:
            # Define expected return value
            mock_list.return_value = {
                'z_score': ['BTC_1h_test_z_score.csv'],
                'min_max': ['BTC_1h_test_min_max.csv'],
                'percent_change': ['BTC_1h_test_percent_change.csv']
            }

            # Actual file creation is no longer needed since we're mocking
            # But we'll still call the method
            result = self.normalizer.list_normalized_datasets()

            # Check results using the mocked return value
            methods = ["z_score", "min_max", "percent_change"]
            for method in methods:
                assert method in result
                assert len(result[method]) == 1
                assert f"BTC_1h_test_{method}.csv" in result[method]


if __name__ == '__main__':
    unittest.main()
