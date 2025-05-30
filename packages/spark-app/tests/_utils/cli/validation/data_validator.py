"""
Data Validator

This module provides comprehensive data validation:
- Add data quality validation
- Add data completeness checking
- Add data consistency validation
- Add data format validation
- Add data source reliability assessment
- Add data preprocessing validation
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from core.data_manager import DataManager

logger = logging.getLogger(__name__)


@dataclass
class DataIssue:
    """Represents a data quality issue with severity and recommendations."""
    severity: str  # 'critical', 'warning', 'suggestion'
    category: str  # 'completeness', 'consistency', 'quality', 'format', 'reliability'
    message: str
    location: str  # Data location (timeframe, column, etc.)
    count: int  # Number of affected records
    percentage: float  # Percentage of data affected
    recommendation: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class DataValidationResult:
    """Result of data validation with detailed quality assessment."""
    is_valid: bool
    quality_score: float  # 0-100 data quality score
    issues: List[DataIssue]
    completeness_score: float
    consistency_score: float
    reliability_score: float
    recommendations: List[str]

    total_records: int = 0
    valid_records: int = 0
    time_range: Optional[Tuple[datetime, datetime]] = None

    def add_issue(self, issue: DataIssue) -> None:
        """Add a data quality issue."""
        self.issues.append(issue)
        if issue.severity == 'critical':
            self.is_valid = False

    def get_critical_issues(self) -> List[DataIssue]:
        """Get all critical data issues."""
        return [issue for issue in self.issues if issue.severity == 'critical']

    def get_warnings(self) -> List[DataIssue]:
        """Get all warning-level issues."""
        return [issue for issue in self.issues if issue.severity == 'warning']

    def get_suggestions(self) -> List[DataIssue]:
        """Get all suggestion-level issues."""
        return [issue for issue in self.issues if issue.severity == 'suggestion']


class DataValidator:
    """
    Validates market data for quality, completeness, consistency,
    and provides recommendations for data preprocessing.
    """

    # Expected OHLCV columns
    REQUIRED_COLUMNS = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}

    # Optional columns that may be present
    OPTIONAL_COLUMNS = {'vwap', 'trades', 'bid', 'ask', 'spread'}

    # Data quality thresholds
    QUALITY_THRESHOLDS = {
        'min_completeness': 95.0,  # Minimum % of non-null values
        'max_gap_minutes': 60,     # Maximum acceptable gap in minutes
        'min_volume_threshold': 0.01,  # Minimum volume for valid candle
        'max_spread_percentage': 10.0,  # Maximum bid-ask spread %
        'max_price_change': 50.0,      # Maximum single-candle price change %
        'min_records_per_day': 10,     # Minimum candles per day
    }

    # Timeframe intervals in minutes
    TIMEFRAME_INTERVALS = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '4h': 240, '1d': 1440, '1w': 10080
    }

    def __init__(self, data_manager: DataManager):
        """
        Initialize the data validator.

        Args:
            data_manager: DataManager instance for data access
        """
        self.data_manager = data_manager

    def validate_market_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        expected_start: Optional[datetime] = None,
        expected_end: Optional[datetime] = None
    ) -> DataValidationResult:
        """
        Validate market data for quality and completeness.

        Args:
            data: Market data DataFrame
            symbol: Trading symbol (e.g., 'BTC-USD')
            timeframe: Data timeframe (e.g., '1h')
            expected_start: Expected start time for data range
            expected_end: Expected end time for data range

        Returns:
            DataValidationResult with comprehensive validation
        """
        result = DataValidationResult(
            is_valid=True,
            quality_score=0.0,
            issues=[],
            completeness_score=0.0,
            consistency_score=0.0,
            reliability_score=0.0,
            recommendations=[]
        )

        if data is None or data.empty:
            result.add_issue(DataIssue(
                severity='critical',
                category='completeness',
                message='No data available for validation',
                location=f'{symbol} {timeframe}',
                count=0,
                percentage=100.0,
                recommendation='Check data source connectivity and symbol availability',
                auto_fixable=False
            ))
            return result

        result.total_records = len(data)

        # Validate data structure and format
        self._validate_data_format(data, symbol, timeframe, result)

        # Validate data completeness
        self._validate_data_completeness(data, symbol, timeframe, result)

        # Validate time consistency
        self._validate_time_consistency(data, symbol, timeframe, result)

        # Validate price consistency and quality
        self._validate_price_quality(data, symbol, timeframe, result)

        # Validate volume data
        self._validate_volume_data(data, symbol, timeframe, result)

        # Validate expected time range
        if expected_start or expected_end:
            self._validate_time_range(data, expected_start, expected_end, symbol, timeframe, result)

        # Detect anomalies and outliers
        self._detect_data_anomalies(data, symbol, timeframe, result)

        # Validate data source reliability
        self._assess_data_reliability(data, symbol, timeframe, result)

        # Generate recommendations
        self._generate_data_recommendations(data, result)

        # Calculate quality scores
        self._calculate_quality_scores(data, result)

        # Set time range
        if not data.empty and 'timestamp' in data.columns:
            result.time_range = (
                pd.to_datetime(data['timestamp'].min()),
                pd.to_datetime(data['timestamp'].max())
            )

        result.valid_records = self._count_valid_records(data)

        logger.debug(f"Data validation completed for {symbol} {timeframe}. "
                    f"Valid: {result.is_valid}, Score: {result.quality_score:.1f}, "
                    f"Records: {result.valid_records}/{result.total_records}")

        return result

    def _validate_data_format(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        result: DataValidationResult
    ) -> None:
        """Validate data format and structure."""
        # Check required columns
        missing_columns = self.REQUIRED_COLUMNS - set(data.columns)
        if missing_columns:
            result.add_issue(DataIssue(
                severity='critical',
                category='format',
                message=f'Missing required columns: {", ".join(missing_columns)}',
                location=f'{symbol} {timeframe}',
                count=len(missing_columns),
                percentage=(len(missing_columns) / len(self.REQUIRED_COLUMNS)) * 100,
                recommendation='Ensure data source provides all required OHLCV columns',
                auto_fixable=False
            ))

        # Check data types
        if 'timestamp' in data.columns:
            try:
                pd.to_datetime(data['timestamp'])
            except (ValueError, TypeError):
                result.add_issue(DataIssue(
                    severity='critical',
                    category='format',
                    message='Invalid timestamp format',
                    location=f'{symbol} {timeframe} timestamp',
                    count=len(data),
                    percentage=100.0,
                    recommendation='Convert timestamps to valid datetime format',
                    auto_fixable=True
                ))

        # Check numeric columns
        numeric_columns = {'open', 'high', 'low', 'close', 'volume'}
        for col in numeric_columns.intersection(data.columns):
            if not pd.api.types.is_numeric_dtype(data[col]):
                non_numeric_count = len(data) - pd.to_numeric(data[col], errors='coerce').notna().sum()
                if non_numeric_count > 0:
                    result.add_issue(DataIssue(
                        severity='warning',
                        category='format',
                        message=f'Non-numeric values in {col} column',
                        location=f'{symbol} {timeframe} {col}',
                        count=non_numeric_count,
                        percentage=(non_numeric_count / len(data)) * 100,
                        recommendation=f'Convert {col} column to numeric type',
                        auto_fixable=True
                    ))

    def _validate_data_completeness(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        result: DataValidationResult
    ) -> None:
        """Validate data completeness and missing values."""
        for col in self.REQUIRED_COLUMNS.intersection(data.columns):
            null_count = data[col].isnull().sum()
            if null_count > 0:
                percentage = (null_count / len(data)) * 100
                severity = 'critical' if percentage > 5.0 else 'warning'

                result.add_issue(DataIssue(
                    severity=severity,
                    category='completeness',
                    message=f'Missing values in {col} column',
                    location=f'{symbol} {timeframe} {col}',
                    count=null_count,
                    percentage=percentage,
                    recommendation=f'Fill missing {col} values using interpolation or remove incomplete records',
                    auto_fixable=True
                ))

        # Check for empty string or zero values where inappropriate
        if 'close' in data.columns:
            zero_prices = (data['close'] <= 0).sum()
            if zero_prices > 0:
                result.add_issue(DataIssue(
                    severity='critical',
                    category='completeness',
                    message='Zero or negative close prices found',
                    location=f'{symbol} {timeframe} close',
                    count=zero_prices,
                    percentage=(zero_prices / len(data)) * 100,
                    recommendation='Remove or interpolate zero/negative price records',
                    auto_fixable=True
                ))

    def _validate_time_consistency(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        result: DataValidationResult
    ) -> None:
        """Validate time consistency and gaps."""
        if 'timestamp' not in data.columns or len(data) < 2:
            return

        timestamps = pd.to_datetime(data['timestamp']).sort_values()
        expected_interval = self.TIMEFRAME_INTERVALS.get(timeframe, 60)

        # Check for duplicates
        duplicate_count = timestamps.duplicated().sum()
        if duplicate_count > 0:
            result.add_issue(DataIssue(
                severity='warning',
                category='consistency',
                message='Duplicate timestamps found',
                location=f'{symbol} {timeframe} timestamp',
                count=duplicate_count,
                percentage=(duplicate_count / len(data)) * 100,
                recommendation='Remove duplicate timestamp records',
                auto_fixable=True
            ))

        # Check for gaps
        time_diffs = timestamps.diff().dt.total_seconds() / 60  # Convert to minutes
        expected_diff = expected_interval

        # Allow for some tolerance (e.g., weekends for daily data)
        tolerance_multiplier = 3 if timeframe in ['1d', '1w'] else 1.5
        max_expected_gap = expected_diff * tolerance_multiplier

        large_gaps = time_diffs[time_diffs > max_expected_gap]
        if len(large_gaps) > 0:
            avg_gap = large_gaps.mean()
            max_gap = large_gaps.max()

            severity = 'critical' if max_gap > self.QUALITY_THRESHOLDS['max_gap_minutes'] else 'warning'

            result.add_issue(DataIssue(
                severity=severity,
                category='consistency',
                message=f'Large time gaps detected (avg: {avg_gap:.1f}min, max: {max_gap:.1f}min)',
                location=f'{symbol} {timeframe} timestamp',
                count=len(large_gaps),
                percentage=(len(large_gaps) / len(time_diffs)) * 100,
                recommendation='Fill time gaps with interpolated data or mark as missing periods',
                auto_fixable=True
            ))

        # Check for irregular intervals
        regular_intervals = time_diffs[(time_diffs >= expected_diff * 0.9) &
                                     (time_diffs <= expected_diff * 1.1)]
        regularity_percentage = (len(regular_intervals) / len(time_diffs)) * 100

        if regularity_percentage < 80:
            result.add_issue(DataIssue(
                severity='warning',
                category='consistency',
                message=f'Irregular time intervals ({regularity_percentage:.1f}% regular)',
                location=f'{symbol} {timeframe} timestamp',
                count=len(time_diffs) - len(regular_intervals),
                percentage=100 - regularity_percentage,
                recommendation='Standardize time intervals or resample data',
                auto_fixable=True
            ))

    def _validate_price_quality(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        result: DataValidationResult
    ) -> None:
        """Validate price data quality and consistency."""
        price_columns = {'open', 'high', 'low', 'close'}
        available_price_cols = price_columns.intersection(data.columns)

        if len(available_price_cols) < 4:
            return

        # Validate OHLC relationships
        ohlc_violations = 0

        # High should be >= Open, Low, Close
        if all(col in data.columns for col in ['high', 'open', 'low', 'close']):
            high_violations = (
                (data['high'] < data['open']) |
                (data['high'] < data['low']) |
                (data['high'] < data['close'])
            ).sum()
            ohlc_violations += high_violations

            # Low should be <= Open, High, Close
            low_violations = (
                (data['low'] > data['open']) |
                (data['low'] > data['high']) |
                (data['low'] > data['close'])
            ).sum()
            ohlc_violations += low_violations

            if ohlc_violations > 0:
                result.add_issue(DataIssue(
                    severity='critical',
                    category='quality',
                    message='OHLC relationship violations (High < Low, etc.)',
                    location=f'{symbol} {timeframe} OHLC',
                    count=ohlc_violations,
                    percentage=(ohlc_violations / len(data)) * 100,
                    recommendation='Correct OHLC data or remove invalid candles',
                    auto_fixable=True
                ))

        # Check for extreme price movements
        if 'close' in data.columns and len(data) > 1:
            price_changes = data['close'].pct_change().abs() * 100
            extreme_moves = price_changes[price_changes > self.QUALITY_THRESHOLDS['max_price_change']]

            if len(extreme_moves) > 0:
                result.add_issue(DataIssue(
                    severity='warning',
                    category='quality',
                    message=f'Extreme price movements detected (>{self.QUALITY_THRESHOLDS["max_price_change"]}%)',
                    location=f'{symbol} {timeframe} close',
                    count=len(extreme_moves),
                    percentage=(len(extreme_moves) / len(price_changes)) * 100,
                    recommendation='Verify extreme price movements or consider outlier filtering',
                    auto_fixable=False
                ))

        # Check for price spikes (isolated extreme values)
        if 'high' in data.columns and 'low' in data.columns and len(data) > 2:
            # Calculate rolling median for spike detection
            window_size = min(5, len(data) // 4)
            if window_size >= 3:
                median_high = data['high'].rolling(window=window_size, center=True).median()
                median_low = data['low'].rolling(window=window_size, center=True).median()

                # Detect spikes as values significantly above/below median
                high_spikes = (data['high'] > median_high * 1.5).sum()
                low_spikes = (data['low'] < median_low * 0.5).sum()

                total_spikes = high_spikes + low_spikes
                if total_spikes > 0:
                    result.add_issue(DataIssue(
                        severity='suggestion',
                        category='quality',
                        message=f'Potential price spikes detected',
                        location=f'{symbol} {timeframe} prices',
                        count=total_spikes,
                        percentage=(total_spikes / len(data)) * 100,
                        recommendation='Review price spikes for data quality issues',
                        auto_fixable=False
                    ))

    def _validate_volume_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        result: DataValidationResult
    ) -> None:
        """Validate volume data quality."""
        if 'volume' not in data.columns:
            return

        # Check for negative volumes
        negative_volumes = (data['volume'] < 0).sum()
        if negative_volumes > 0:
            result.add_issue(DataIssue(
                severity='critical',
                category='quality',
                message='Negative volume values found',
                location=f'{symbol} {timeframe} volume',
                count=negative_volumes,
                percentage=(negative_volumes / len(data)) * 100,
                recommendation='Correct negative volume values',
                auto_fixable=True
            ))

        # Check for zero volumes
        zero_volumes = (data['volume'] == 0).sum()
        if zero_volumes > 0:
            percentage = (zero_volumes / len(data)) * 100
            severity = 'warning' if percentage < 10 else 'critical'

            result.add_issue(DataIssue(
                severity=severity,
                category='quality',
                message='Zero volume candles found',
                location=f'{symbol} {timeframe} volume',
                count=zero_volumes,
                percentage=percentage,
                recommendation='Consider filtering out zero-volume candles or investigate data source',
                auto_fixable=True
            ))

        # Check for suspiciously low volumes
        if len(data) > 0:
            median_volume = data['volume'].median()
            very_low_volumes = (data['volume'] < median_volume * 0.01).sum()

            if very_low_volumes > 0:
                result.add_issue(DataIssue(
                    severity='suggestion',
                    category='quality',
                    message='Very low volume candles detected',
                    location=f'{symbol} {timeframe} volume',
                    count=very_low_volumes,
                    percentage=(very_low_volumes / len(data)) * 100,
                    recommendation='Review low-volume periods for market conditions',
                    auto_fixable=False
                ))

    def _validate_time_range(
        self,
        data: pd.DataFrame,
        expected_start: Optional[datetime],
        expected_end: Optional[datetime],
        symbol: str,
        timeframe: str,
        result: DataValidationResult
    ) -> None:
        """Validate that data covers expected time range."""
        if 'timestamp' not in data.columns or data.empty:
            return

        actual_start = pd.to_datetime(data['timestamp'].min())
        actual_end = pd.to_datetime(data['timestamp'].max())

        if expected_start:
            if actual_start > expected_start:
                missing_duration = actual_start - expected_start
                result.add_issue(DataIssue(
                    severity='warning',
                    category='completeness',
                    message=f'Data starts later than expected (missing {missing_duration})',
                    location=f'{symbol} {timeframe} start_time',
                    count=0,
                    percentage=0.0,
                    recommendation='Extend data collection period to include missing historical data',
                    auto_fixable=False
                ))

        if expected_end:
            if actual_end < expected_end:
                missing_duration = expected_end - actual_end
                result.add_issue(DataIssue(
                    severity='warning',
                    category='completeness',
                    message=f'Data ends earlier than expected (missing {missing_duration})',
                    location=f'{symbol} {timeframe} end_time',
                    count=0,
                    percentage=0.0,
                    recommendation='Update data collection to include recent data',
                    auto_fixable=False
                ))

    def _detect_data_anomalies(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        result: DataValidationResult
    ) -> None:
        """Detect anomalies and outliers in the data."""
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in numeric_columns if col in data.columns]

        for col in available_cols:
            if data[col].dtype in [np.float64, np.int64]:
                # Use IQR method for outlier detection
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1

                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()

                if outliers > 0:
                    outlier_percentage = (outliers / len(data)) * 100
                    severity = 'warning' if outlier_percentage < 5 else 'critical'

                    result.add_issue(DataIssue(
                        severity=severity,
                        category='quality',
                        message=f'Statistical outliers detected in {col}',
                        location=f'{symbol} {timeframe} {col}',
                        count=outliers,
                        percentage=outlier_percentage,
                        recommendation=f'Review {col} outliers for data quality or market events',
                        auto_fixable=False
                    ))

    def _assess_data_reliability(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        result: DataValidationResult
    ) -> None:
        """Assess overall data source reliability."""
        if data.empty:
            return

        # Check data recency
        if 'timestamp' in data.columns:
            latest_timestamp = pd.to_datetime(data['timestamp'].max())
            time_since_last = datetime.now() - latest_timestamp

            expected_delay = timedelta(minutes=self.TIMEFRAME_INTERVALS.get(timeframe, 60) * 2)

            if time_since_last > expected_delay:
                result.add_issue(DataIssue(
                    severity='warning',
                    category='reliability',
                    message=f'Data appears stale (last update: {time_since_last} ago)',
                    location=f'{symbol} {timeframe} timestamp',
                    count=0,
                    percentage=0.0,
                    recommendation='Verify data source is actively updating',
                    auto_fixable=False
                ))

        # Check data density (records per expected time period)
        if 'timestamp' in data.columns and len(data) > 1:
            time_span = pd.to_datetime(data['timestamp'].max()) - pd.to_datetime(data['timestamp'].min())
            expected_records = time_span.total_seconds() / (self.TIMEFRAME_INTERVALS.get(timeframe, 60) * 60)
            actual_records = len(data)

            density = (actual_records / expected_records) * 100 if expected_records > 0 else 0

            if density < 80:
                result.add_issue(DataIssue(
                    severity='warning',
                    category='reliability',
                    message=f'Low data density ({density:.1f}% of expected records)',
                    location=f'{symbol} {timeframe}',
                    count=int(expected_records - actual_records),
                    percentage=100 - density,
                    recommendation='Investigate missing data periods or improve data collection',
                    auto_fixable=False
                ))

    def _generate_data_recommendations(self, data: pd.DataFrame, result: DataValidationResult) -> None:
        """Generate data preprocessing and improvement recommendations."""
        if result.get_critical_issues():
            result.recommendations.append(
                "Address critical data issues before using for backtesting or live trading"
            )

        # Recommend filtering based on data quality
        if len(result.get_warnings()) > 5:
            result.recommendations.append(
                "Consider implementing data quality filters to automatically handle common issues"
            )

        # Recommend data preprocessing
        if any(issue.category == 'consistency' for issue in result.issues):
            result.recommendations.append(
                "Implement data preprocessing pipeline to handle time gaps and irregularities"
            )

        # Recommend volume filtering
        if 'volume' in data.columns and (data['volume'] == 0).any():
            result.recommendations.append(
                "Consider filtering out zero-volume candles for more accurate analysis"
            )

        # Recommend outlier handling
        if any('outliers' in issue.message.lower() for issue in result.issues):
            result.recommendations.append(
                "Implement outlier detection and handling in data preprocessing"
            )

    def _calculate_quality_scores(self, data: pd.DataFrame, result: DataValidationResult) -> None:
        """Calculate comprehensive data quality scores."""
        if data.empty:
            result.completeness_score = 0.0
            result.consistency_score = 0.0
            result.reliability_score = 0.0
            result.quality_score = 0.0
            return

        # Completeness score
        completeness_issues = [issue for issue in result.issues if issue.category == 'completeness']
        completeness_penalty = sum(issue.percentage for issue in completeness_issues)
        result.completeness_score = max(0.0, 100.0 - completeness_penalty)

        # Consistency score
        consistency_issues = [issue for issue in result.issues if issue.category == 'consistency']
        consistency_penalty = sum(min(issue.percentage, 10.0) for issue in consistency_issues)
        result.consistency_score = max(0.0, 100.0 - consistency_penalty)

        # Reliability score
        reliability_issues = [issue for issue in result.issues if issue.category == 'reliability']
        reliability_penalty = len(reliability_issues) * 15.0  # Fixed penalty per issue
        result.reliability_score = max(0.0, 100.0 - reliability_penalty)

        # Overall quality score (weighted average)
        result.quality_score = (
            result.completeness_score * 0.4 +
            result.consistency_score * 0.3 +
            result.reliability_score * 0.3
        )

        # Apply penalties for critical issues
        critical_issues = result.get_critical_issues()
        if critical_issues:
            critical_penalty = len(critical_issues) * 20.0
            result.quality_score = max(0.0, result.quality_score - critical_penalty)

    def _count_valid_records(self, data: pd.DataFrame) -> int:
        """Count records that are considered valid for analysis."""
        if data.empty:
            return 0

        valid_mask = pd.Series(True, index=data.index)

        # Exclude records with null required values
        for col in self.REQUIRED_COLUMNS.intersection(data.columns):
            valid_mask &= data[col].notna()

        # Exclude records with zero/negative prices
        if 'close' in data.columns:
            valid_mask &= (data['close'] > 0)

        # Exclude records with negative volumes
        if 'volume' in data.columns:
            valid_mask &= (data['volume'] >= 0)

        return valid_mask.sum()

    def validate_data_preprocessing_quality(
        self,
        original_data: pd.DataFrame,
        processed_data: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> DataValidationResult:
        """
        Validate the quality of data preprocessing operations.

        Args:
            original_data: Original raw data
            processed_data: Data after preprocessing
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            DataValidationResult for preprocessing quality
        """
        result = DataValidationResult(
            is_valid=True,
            quality_score=0.0,
            issues=[],
            completeness_score=0.0,
            consistency_score=0.0,
            reliability_score=0.0,
            recommendations=[]
        )

        if original_data.empty and processed_data.empty:
            result.add_issue(DataIssue(
                severity='critical',
                category='quality',
                message='Both original and processed data are empty',
                location=f'{symbol} {timeframe}',
                count=0,
                percentage=100.0,
                recommendation='Check data source and preprocessing pipeline',
                auto_fixable=False
            ))
            return result

        # Check data retention after preprocessing
        original_count = len(original_data)
        processed_count = len(processed_data)
        retention_rate = (processed_count / original_count * 100) if original_count > 0 else 0

        if retention_rate < 50:
            result.add_issue(DataIssue(
                severity='critical',
                category='quality',
                message=f'Excessive data loss during preprocessing ({retention_rate:.1f}% retained)',
                location=f'{symbol} {timeframe}',
                count=original_count - processed_count,
                percentage=100 - retention_rate,
                recommendation='Review preprocessing filters to reduce data loss',
                auto_fixable=False
            ))
        elif retention_rate < 80:
            result.add_issue(DataIssue(
                severity='warning',
                category='quality',
                message=f'Significant data loss during preprocessing ({retention_rate:.1f}% retained)',
                location=f'{symbol} {timeframe}',
                count=original_count - processed_count,
                percentage=100 - retention_rate,
                recommendation='Consider relaxing preprocessing criteria',
                auto_fixable=False
            ))

        # Validate preprocessing improvements
        original_validation = self.validate_market_data(original_data, symbol, timeframe)
        processed_validation = self.validate_market_data(processed_data, symbol, timeframe)

        quality_improvement = processed_validation.quality_score - original_validation.quality_score

        if quality_improvement < 0:
            result.add_issue(DataIssue(
                severity='warning',
                category='quality',
                message=f'Data quality decreased after preprocessing ({quality_improvement:.1f} points)',
                location=f'{symbol} {timeframe}',
                count=0,
                percentage=0.0,
                recommendation='Review preprocessing pipeline for unintended quality degradation',
                auto_fixable=False
            ))
        elif quality_improvement > 10:
            result.recommendations.append(
                f"Preprocessing improved data quality by {quality_improvement:.1f} points"
            )

        result.total_records = processed_count
        result.valid_records = self._count_valid_records(processed_data)
        result.quality_score = min(100.0, max(0.0, 80.0 + quality_improvement))

        return result

    def validate_strategy_data_requirements(self, strategy_config) -> 'ValidationResult':
        """
        Validate that strategy data requirements can be met.

        Args:
            strategy_config: Strategy configuration to validate

        Returns:
            ValidationResult indicating whether requirements can be met
        """
        from .strategy_validator import ValidationResult

        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[]
        )
        result.component = "data_requirements"

        try:
            # Check if strategy has valid market and exchange
            if not strategy_config.market:
                result.add_error("Strategy missing market specification")
                return result

            if not strategy_config.exchange:
                result.add_error("Strategy missing exchange specification")
                return result

            # Check if timeframe is valid
            if not strategy_config.timeframe:
                result.add_error("Strategy missing timeframe specification")
                return result

            # Validate timeframe format
            valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
            if strategy_config.timeframe not in valid_timeframes:
                result.add_warning(f"Unusual timeframe '{strategy_config.timeframe}', may not be supported by all exchanges")

            # Check data availability for each indicator
            for indicator_name in strategy_config.indicators:
                # Since indicators is just a list of strings, we can only do basic validation
                self._validate_indicator_data_requirements_by_name(indicator_name, result)

            # Add success messages
            if result.is_valid:
                # Since ValidationResult doesn't have add_info, we'll use suggestions for informational messages
                result.add_suggestion(f"Strategy '{strategy_config.name}' data requirements validated successfully")
                result.add_suggestion(f"Market: {strategy_config.market}, Exchange: {strategy_config.exchange}")
                result.add_suggestion(f"Primary timeframe: {strategy_config.timeframe}")

                indicator_count = len(strategy_config.indicators)
                result.add_suggestion(f"Validated data requirements for {indicator_count} indicators")

        except Exception as e:
            result.add_error(f"Error validating strategy data requirements: {str(e)}")

        return result

    def _validate_indicator_data_requirements_by_name(self, indicator_name: str, result) -> None:
        """
        Validate data requirements for an indicator by name.

        Args:
            indicator_name: Name of the indicator
            result: ValidationResult to populate
        """
        # Check for indicators that require specific data based on name
        if 'volume' in indicator_name.lower():
            result.add_suggestion(f"Indicator '{indicator_name}' requires volume data")

        if 'sma' in indicator_name.lower() or 'ema' in indicator_name.lower():
            result.add_suggestion(f"Indicator '{indicator_name}' is a moving average - ensure sufficient historical data")

        if 'rsi' in indicator_name.lower():
            result.add_suggestion(f"Indicator '{indicator_name}' (RSI) requires minimum 14+ candles for reliable signals")

        if 'bollinger' in indicator_name.lower() or 'bb' in indicator_name.lower():
            result.add_suggestion(f"Indicator '{indicator_name}' (Bollinger Bands) requires substantial historical data")

        if 'macd' in indicator_name.lower():
            result.add_suggestion(f"Indicator '{indicator_name}' (MACD) requires minimum 100+ candles for reliable signals")

    def _validate_indicator_data_requirements(self, indicator_config, result) -> None:
        """
        Validate data requirements for a specific indicator.

        Args:
            indicator_config: Indicator configuration
            result: ValidationResult to populate
        """
        indicator_name = getattr(indicator_config, 'name', 'unknown')

        # Check for indicators that require specific data
        if 'volume' in indicator_name.lower():
            result.add_suggestion(f"Indicator '{indicator_name}' requires volume data")

        if 'sma' in indicator_name.lower() or 'ema' in indicator_name.lower():
            period = getattr(indicator_config, 'period', getattr(indicator_config, 'length', None))
            if period and period > 200:
                result.add_warning(f"Indicator '{indicator_name}' requires {period} periods - ensure sufficient historical data")

        if 'rsi' in indicator_name.lower():
            period = getattr(indicator_config, 'period', 14)
            if period > 50:
                result.add_warning(f"RSI with period {period} may require significant historical data")

        if 'bollinger' in indicator_name.lower() or 'bb' in indicator_name.lower():
            period = getattr(indicator_config, 'period', 20)
            if period > 100:
                result.add_warning(f"Bollinger Bands with period {period} requires substantial historical data")

        if 'macd' in indicator_name.lower():
            # MACD typically requires at least 100+ periods for reliable signals
            result.add_suggestion(f"Indicator '{indicator_name}' requires minimum 100+ candles for reliable signals")
