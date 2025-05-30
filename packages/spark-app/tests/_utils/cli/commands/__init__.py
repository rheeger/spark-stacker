"""
Command Handler Modules

This package contains command handler modules for different types of CLI commands:
- strategy_commands: Strategy backtesting and management commands
- indicator_commands: Legacy indicator backtesting commands
- comparison_commands: Strategy and indicator comparison commands
- list_commands: Commands for listing strategies and indicators
- utility_commands: Configuration validation and utility commands
"""

from .comparison_commands import register_comparison_commands
from .indicator_commands import register_indicator_commands
from .list_commands import register_list_commands
# Import command registration functions from implemented modules
from .strategy_commands import register_strategy_commands
from .utility_commands import register_utility_commands

__all__ = [
    "register_strategy_commands",
    "register_indicator_commands",
    "register_comparison_commands",
    "register_list_commands",
    "register_utility_commands"
]
