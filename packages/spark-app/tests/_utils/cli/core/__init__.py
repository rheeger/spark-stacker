"""
Core Business Logic Modules

This package contains core business logic modules for the CLI:
- config_manager: Configuration loading, validation, and management
- data_manager: Data fetching, caching, and validation
- backtest_orchestrator: Coordinates overall backtesting workflow
- scenario_manager: Multi-scenario testing coordination and execution
"""

# Import implemented core classes
from .config_manager import (ConfigManager, ConfigurationError,
                             ConfigValidationError)
from .data_manager import (DataFetchError, DataManager, DataQuality,
                           DataQualityError, DataQualityReport, DataRequest,
                           DataSourceType)

# TODO: Import remaining core classes when modules are implemented
# from .backtest_orchestrator import BacktestOrchestrator
# from .scenario_manager import ScenarioManager

__all__ = [
    "ConfigManager",
    "ConfigurationError",
    "ConfigValidationError",
    "DataManager",
    "DataRequest",
    "DataQuality",
    "DataQualityReport",
    "DataSourceType",
    "DataFetchError",
    "DataQualityError",
    # "BacktestOrchestrator",
    # "ScenarioManager"
]
