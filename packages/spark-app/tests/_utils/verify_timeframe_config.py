#!/usr/bin/env python3
"""
Verification script for the new timeframe configuration system.

This script tests that:
1. Indicators are created with correct timeframes from config
2. TimeFrame methods work as expected
3. Backward compatibility is maintained
4. Multi-timeframe setup works correctly
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Add the parent directory to the path so we can import from app
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.indicators.base_indicator import BaseIndicator
from app.indicators.indicator_factory import IndicatorFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_timeframe_configuration():
    """Test the timeframe configuration system."""

    print("🕒 Testing Unified Timeframe Configuration System")
    print("=" * 60)

    # Test 1: Basic timeframe configuration
    print("\n1. Testing basic timeframe configuration...")

    test_configs = [
        {
            "name": "test_rsi_4h",
            "type": "rsi",
            "enabled": True,
            "timeframe": "4h",
            "parameters": {
                "period": 14,
                "overbought": 70,
                "oversold": 30
            }
        },
        {
            "name": "test_macd_1h",
            "type": "macd",
            "enabled": True,
            "timeframe": "1h",
            "parameters": {
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9
            }
        },
        {
            "name": "test_rsi_no_timeframe",
            "type": "rsi",
            "enabled": True,
            "parameters": {
                "period": 14
            }
        }
    ]

    # Create indicators from config
    indicators = IndicatorFactory.create_indicators_from_config(test_configs)

    # Verify indicators were created
    assert len(indicators) == 3, f"Expected 3 indicators, got {len(indicators)}"
    print(f"✅ Created {len(indicators)} indicators successfully")

    # Test 2: Verify timeframes are set correctly
    print("\n2. Testing timeframe assignment...")

    # Check 4h RSI
    rsi_4h = indicators["test_rsi_4h"]
    assert rsi_4h.get_effective_timeframe() == "4h", f"Expected 4h, got {rsi_4h.get_effective_timeframe()}"
    assert hasattr(rsi_4h, 'timeframe'), "Indicator should have timeframe attribute"
    assert hasattr(rsi_4h, 'interval'), "Indicator should have interval attribute for backward compatibility"
    assert rsi_4h.timeframe == "4h", f"Expected timeframe 4h, got {rsi_4h.timeframe}"
    assert rsi_4h.interval == "4h", f"Expected interval 4h, got {rsi_4h.interval}"
    print(f"✅ RSI 4h: timeframe={rsi_4h.timeframe}, interval={rsi_4h.interval}")

    # Check 1h MACD
    macd_1h = indicators["test_macd_1h"]
    assert macd_1h.get_effective_timeframe() == "1h", f"Expected 1h, got {macd_1h.get_effective_timeframe()}"
    assert macd_1h.timeframe == "1h", f"Expected timeframe 1h, got {macd_1h.timeframe}"
    print(f"✅ MACD 1h: timeframe={macd_1h.timeframe}, interval={macd_1h.interval}")

    # Check default timeframe (should be 1h)
    rsi_default = indicators["test_rsi_no_timeframe"]
    assert rsi_default.get_effective_timeframe() == "1h", f"Expected default 1h, got {rsi_default.get_effective_timeframe()}"
    print(f"✅ RSI default: timeframe={rsi_default.timeframe}, interval={rsi_default.interval}")

    # Test 3: Test set_timeframe method
    print("\n3. Testing timeframe modification...")

    original_timeframe = rsi_4h.get_effective_timeframe()
    rsi_4h.set_timeframe("1d")
    assert rsi_4h.get_effective_timeframe() == "1d", f"Expected 1d after set, got {rsi_4h.get_effective_timeframe()}"
    assert rsi_4h.timeframe == "1d", "Timeframe attribute should be updated"
    assert rsi_4h.interval == "1d", "Interval attribute should be updated for backward compatibility"

    # Reset for further tests
    rsi_4h.set_timeframe(original_timeframe)
    print(f"✅ Timeframe modification works correctly")

    # Test 4: Test string representation
    print("\n4. Testing string representation...")

    rsi_str = str(rsi_4h)
    assert "timeframe=4h" in rsi_str, f"String representation should include timeframe: {rsi_str}"
    print(f"✅ String representation: {rsi_str}")

    # Test 5: Load from actual config file
    print("\n5. Testing with actual configuration files...")

    # Test with the main config
    main_config_path = Path(__file__).parent.parent.parent.parent / "shared" / "config.json"
    if main_config_path.exists():
        with open(main_config_path, 'r') as f:
            config = json.load(f)

        if "indicators" in config:
            main_indicators = IndicatorFactory.create_indicators_from_config(config["indicators"])
            print(f"✅ Loaded {len(main_indicators)} indicators from main config")

            for name, indicator in main_indicators.items():
                timeframe = indicator.get_effective_timeframe()
                print(f"   - {name}: {timeframe}")

    # Test with the multi-timeframe example
    example_config_path = Path(__file__).parent.parent.parent.parent / "shared" / "examples" / "multi-timeframe-config.json"
    if example_config_path.exists():
        with open(example_config_path, 'r') as f:
            config = json.load(f)

        if "indicators" in config:
            example_indicators = IndicatorFactory.create_indicators_from_config(config["indicators"])
            print(f"✅ Loaded {len(example_indicators)} indicators from multi-timeframe example")

            # Verify different timeframes are present
            timeframes = set()
            for name, indicator in example_indicators.items():
                timeframe = indicator.get_effective_timeframe()
                timeframes.add(timeframe)
                print(f"   - {name}: {timeframe}")

            print(f"✅ Found {len(timeframes)} different timeframes: {sorted(timeframes)}")
            assert len(timeframes) > 1, "Multi-timeframe example should have multiple timeframes"

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! Timeframe configuration system is working correctly.")
    print("\nKey features verified:")
    print("✅ Indicators created with correct timeframes from config")
    print("✅ Default timeframe (1h) applied when not specified")
    print("✅ Timeframe modification methods work")
    print("✅ Backward compatibility with interval attribute")
    print("✅ String representation includes timeframe")
    print("✅ Multi-timeframe configurations load correctly")
    print("\nThe unified timeframe configuration system is ready for use!")


def main():
    """Main function to run the verification tests."""
    try:
        test_timeframe_configuration()
        return 0
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
