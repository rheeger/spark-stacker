#!/bin/bash
set -e

# Script to scaffold a new indicator
# Usage: ./scripts/new_indicator.sh <IndicatorName>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <IndicatorName>"
    echo "Example: $0 StochasticRSI"
    exit 1
fi

INDICATOR_NAME="$1"

# Convert to snake_case for file names
INDICATOR_FILE=$(echo "$INDICATOR_NAME" | sed 's/\([A-Z]\)/_\1/g' | sed 's/^_//' | tr '[:upper:]' '[:lower:]')
INDICATOR_TYPE=$(echo "$INDICATOR_FILE" | tr '_' '-')

# Get script directory to determine paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Define paths
INDICATOR_DIR="$APP_DIR/app/indicators"
TEST_DIR="$APP_DIR/tests/indicators/unit"
FACTORY_FILE="$APP_DIR/app/indicators/indicator_factory.py"

echo "Creating new indicator: $INDICATOR_NAME"
echo "File name: ${INDICATOR_FILE}_indicator.py"
echo "Indicator type: $INDICATOR_TYPE"

# Create the indicator Python file
cat >"$INDICATOR_DIR/${INDICATOR_FILE}_indicator.py" <<EOF
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from app.indicators.base_indicator import BaseIndicator, Signal, SignalDirection

logger = logging.getLogger(__name__)


class ${INDICATOR_NAME}Indicator(BaseIndicator):
    """
    ${INDICATOR_NAME} indicator implementation.

    TODO: Add description of what this indicator measures and how it works.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the ${INDICATOR_NAME} indicator.

        Args:
            name: Indicator name
            params: Parameters for the ${INDICATOR_NAME} indicator
                # TODO: Document specific parameters
                period: The period for calculation (default: 14)
        """
        super().__init__(name, params)
        self.period = self.params.get("period", 14)

        # Update params with actual values being used (including defaults)
        self.params.update({
            "period": self.period,
        })

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ${INDICATOR_NAME} values for the provided price data.

        Args:
            data: Price data as pandas DataFrame with required columns

        Returns:
            DataFrame with ${INDICATOR_NAME} values added as new columns
        """
        # Validate required columns
        required_columns = ["close"]  # TODO: Update based on actual requirements
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"DataFrame must contain a '{col}' column")

        if len(data) < self.period:
            logger.warning(
                f"Not enough data points for ${INDICATOR_NAME} calculation. "
                f"Need at least {self.period}, got {len(data)}"
            )
            # Add empty columns
            data["${INDICATOR_FILE}"] = np.nan
            return data

        # Create a copy of the dataframe to avoid modifying the original
        df = data.copy()

        # TODO: Implement indicator calculation logic
        # This is a placeholder implementation
        df["${INDICATOR_FILE}"] = df["close"].rolling(window=self.period).mean()

        # TODO: Add any additional columns needed for signal generation
        # df["signal_condition"] = ...

        return df

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Generate a trading signal based on ${INDICATOR_NAME} values.

        Args:
            data: Price data with ${INDICATOR_NAME} values

        Returns:
            Signal object if conditions are met, None otherwise
        """
        if "${INDICATOR_FILE}" not in data.columns:
            logger.warning("No ${INDICATOR_FILE} column in data, cannot generate signal")
            return None

        if len(data) < 2:
            logger.warning("Not enough data points to generate a signal")
            return None

        # Get the latest data point
        latest = data.iloc[-1]
        symbol = latest.get("symbol", "UNKNOWN")

        # TODO: Implement signal generation logic
        # This is a placeholder implementation

        # Example: Generate buy signal if indicator is above threshold
        # if latest["${INDICATOR_FILE}"] > some_threshold:
        #     return Signal(
        #         direction=SignalDirection.BUY,
        #         symbol=symbol,
        #         indicator=self.name,
        #         confidence=0.7,
        #         params={
        #             "${INDICATOR_FILE}": latest["${INDICATOR_FILE}"],
        #             "trigger": "above_threshold",
        #         },
        #     )

        # No signal by default
        return None

    def __str__(self) -> str:
        """String representation of the indicator."""
        return f"${INDICATOR_NAME}(period={self.period})"
EOF

# Create the test file
cat >"$TEST_DIR/test_${INDICATOR_FILE}_indicator.py" <<EOF
import numpy as np
import pandas as pd
import pytest
from app.indicators.base_indicator import SignalDirection
from app.indicators.${INDICATOR_FILE}_indicator import ${INDICATOR_NAME}Indicator


def test_${INDICATOR_FILE}_initialization():
    """Test ${INDICATOR_NAME} indicator initialization with default and custom parameters."""
    # Test with default parameters
    indicator = ${INDICATOR_NAME}Indicator(name="test_${INDICATOR_FILE}")
    assert indicator.name == "test_${INDICATOR_FILE}"
    assert indicator.period == 14

    # Test with custom parameters
    custom_params = {"period": 20}
    indicator_custom = ${INDICATOR_NAME}Indicator(name="custom_${INDICATOR_FILE}", params=custom_params)
    assert indicator_custom.name == "custom_${INDICATOR_FILE}"
    assert indicator_custom.period == 20


def test_${INDICATOR_FILE}_calculation(sample_price_data):
    """Test ${INDICATOR_NAME} calculation using sample price data."""
    indicator = ${INDICATOR_NAME}Indicator(name="test_${INDICATOR_FILE}")
    result = indicator.calculate(sample_price_data)

    # Verify indicator column was added
    assert "${INDICATOR_FILE}" in result.columns

    # TODO: Add specific assertions based on indicator behavior
    # For example:
    # - Check value ranges
    # - Verify calculation logic
    # - Test edge cases

    # Basic validation
    assert len(result) == len(sample_price_data)


def test_${INDICATOR_FILE}_signal_generation():
    """Test signal generation based on ${INDICATOR_NAME} values."""
    # TODO: Create test data that will trigger signals
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1550, 1600, 1650, 1700],
            "${INDICATOR_FILE}": [1520, 1540, 1580, 1620, 1680],
        }
    )

    indicator = ${INDICATOR_NAME}Indicator(name="test_${INDICATOR_FILE}")

    # Test with the sample data
    signal = indicator.generate_signal(data)

    # TODO: Update based on actual signal logic
    # For now, expecting no signal since logic is not implemented
    assert signal is None


def test_${INDICATOR_FILE}_process_method(sample_price_data):
    """Test the combined process method."""
    indicator = ${INDICATOR_NAME}Indicator(name="test_${INDICATOR_FILE}", params={"period": 5})

    processed_data, signal = indicator.process(sample_price_data)

    # Verify the data was processed
    assert "${INDICATOR_FILE}" in processed_data.columns

    # TODO: Add tests for signal generation with manipulated data


def test_${INDICATOR_FILE}_error_handling():
    """Test error handling in ${INDICATOR_NAME} indicator."""
    indicator = ${INDICATOR_NAME}Indicator(name="test_${INDICATOR_FILE}")

    # Test with invalid DataFrame (missing required columns)
    invalid_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "invalid_column": [1, 2, 3, 4, 5],
        }
    )

    with pytest.raises(ValueError, match="must contain a 'close' column"):
        indicator.calculate(invalid_df)

    # Test with insufficient data points
    insufficient_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=2, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1550],
        }
    )

    # This should not raise an error but log a warning and return data with NaN values
    result = indicator.calculate(insufficient_df)
    assert "${INDICATOR_FILE}" in result.columns
    assert pd.isna(result["${INDICATOR_FILE}"]).all()


def test_${INDICATOR_FILE}_string_representation():
    """Test string representation of the indicator."""
    indicator = ${INDICATOR_NAME}Indicator(name="test_${INDICATOR_FILE}", params={"period": 21})
    expected = "${INDICATOR_NAME}(period=21)"
    assert str(indicator) == expected
EOF

echo "Created indicator file: $INDICATOR_DIR/${INDICATOR_FILE}_indicator.py"
echo "Created test file: $TEST_DIR/test_${INDICATOR_FILE}_indicator.py"

# Add import and registration to factory
echo "Adding import and registration to indicator factory..."

# Add import line after the existing imports
IMPORT_LINE="from app.indicators.${INDICATOR_FILE}_indicator import ${INDICATOR_NAME}Indicator"

# Check if import already exists
if grep -q "$IMPORT_LINE" "$FACTORY_FILE"; then
    echo "Import already exists in factory file"
else
    # Add import after the existing imports but before the comment section
    # Find the line with "# Import additional indicators" and add before it with proper newline
    sed -i.bak "/# Import additional indicators/i\\
$IMPORT_LINE\\
" "$FACTORY_FILE"

    # Remove backup file
    rm -f "${FACTORY_FILE}.bak"

    echo "Added import to factory file"
fi

# Add to register_defaults method
REGISTRY_ENTRY="            \"$INDICATOR_TYPE\": ${INDICATOR_NAME}Indicator,"

# Check if registry entry already exists
if grep -q "\"$INDICATOR_TYPE\":" "$FACTORY_FILE"; then
    echo "Registry entry already exists in factory file"
else
    # Add to the registry dictionary inside register_defaults method, before the closing brace
    # Look for the closing brace that ends the cls._indicator_registry.update() call
    sed -i.bak "/cls._indicator_registry.update({/,/^        })/s/^        })$/        $REGISTRY_ENTRY\n        })/" "$FACTORY_FILE"

    # Remove backup file
    rm -f "${FACTORY_FILE}.bak"

    echo "Added registry entry to register_defaults method"
fi

echo ""
echo "✅ Successfully created ${INDICATOR_NAME} indicator scaffold!"
echo ""
echo "Next steps:"
echo "1. Implement the calculation logic in ${INDICATOR_FILE}_indicator.py"
echo "2. Implement the signal generation logic"
echo "3. Update the test file with proper test cases"
echo "4. Run tests: cd packages/spark-app && .venv/bin/python -m pytest tests/indicators/unit/test_${INDICATOR_FILE}_indicator.py -v"
echo ""
echo "The indicator is now registered in the factory with type: '$INDICATOR_TYPE'"
echo "You can use it in configurations like:"
echo "{"
echo "  \"name\": \"my_${INDICATOR_FILE}\","
echo "  \"type\": \"$INDICATOR_TYPE\","
echo "  \"parameters\": {\"period\": 14}"
echo "}"
