#!/usr/bin/env python3
import json
import os
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests/_utils/cli'))

from core.config_manager import ConfigManager

config_data = {
    'exchange_configs': {
        'hyperliquid': {
            'name': 'Hyperliquid',
            'markets': ['ETH-USD', 'BTC-USD']
        }
    },
    'strategy_configs': {
        'test_strategy': {
            'name': 'Test Strategy',
            'market': 'ETH-USD',
            'exchange': 'hyperliquid',
            'timeframe': '1h',
            'indicators': {
                'rsi': {
                    'class': 'RSIIndicator',
                    'timeframe': '1h',
                    'window': 14
                }
            },
            'position_sizing': {
                'method': 'fixed_usd',
                'amount': 100
            },
            'enabled': True
        }
    },
    'global_settings': {
        'default_position_sizing': {
            'method': 'fixed_usd',
            'amount': 100
        }
    }
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(config_data, f)
    temp_file = f.name

try:
    manager = ConfigManager(config_path=temp_file)
    strategies = manager.list_strategies()
    print(f'Number of strategies: {len(strategies)}')
    if strategies:
        print(f'First strategy keys: {list(strategies[0].keys())}')
        print(f'First strategy name: {strategies[0].get("name")}')
    print(f'Strategy names: {[s.get("name") for s in strategies]}')

    # Test what the assertion is checking
    print(f'"test_strategy" in strategies: {"test_strategy" in strategies}')
    print(f'Any strategy with name "test_strategy": {any(s.get("name") == "test_strategy" for s in strategies)}')
finally:
    os.unlink(temp_file)
