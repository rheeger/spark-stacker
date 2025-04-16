from .decorators import (track_api_latency, track_order_execution,
                         update_margin_ratio, update_rate_limit)

# Exchange constants
EXCHANGE_HYPERLIQUID = "hyperliquid"
EXCHANGE_COINBASE = "coinbase"
EXCHANGE_KRAKEN = "kraken"

# Market constants
MARKET_ETH_USD = "ETH-USD"
MARKET_BTC_USD = "BTC-USD"

# Strategy constants
STRATEGY_MVP = "mvp"

# Indicator constants
INDICATOR_MACD = "macd"
INDICATOR_RSI = "rsi"
INDICATOR_BB = "bollinger_bands"
INDICATOR_MA = "moving_average"

# Timeframe constants
TIMEFRAME_1M = "1m"
TIMEFRAME_5M = "5m"
TIMEFRAME_15M = "15m"
TIMEFRAME_1H = "1h"
TIMEFRAME_4H = "4h"
TIMEFRAME_1D = "1d"

from .metrics import (active_positions, api_latency_seconds,
                      api_requests_total, capital_utilization_percent)
from .metrics import custom_registry as REGISTRY
from .metrics import (liquidation_price, margin_ratio, max_drawdown_percent,
                      observe_api_latency, order_execution_seconds,
                      pnl_percent, rate_limit_remaining, record_api_request,
                      record_mvp_signal_latency, record_mvp_trade,
                      record_signal, record_trade, signal_count, trades_total,
                      update_candle_data, update_macd_indicator,
                      update_mvp_pnl, update_mvp_position_size,
                      update_mvp_signal_state, update_position)
from .prometheus_exporter import exporter, initialize_metrics
from .registry import start_metrics_server

__all__ = [
    "start_metrics_server",
    "active_positions",
    "api_latency_seconds",
    "api_requests_total",
    "capital_utilization_percent",
    "liquidation_price",
    "margin_ratio",
    "max_drawdown_percent",
    "observe_api_latency",
    "order_execution_seconds",
    "pnl_percent",
    "rate_limit_remaining",
    "record_api_request",
    "record_signal",
    "record_trade",
    "signal_count",
    "trades_total",
    "update_position",
    "initialize_metrics",
    "exporter",
    "track_api_latency",
    "track_order_execution",
    "update_rate_limit",
    "update_margin_ratio",
    # MVP strategy metrics
    "EXCHANGE_HYPERLIQUID",
    "MARKET_ETH_USD",
    "MARKET_BTC_USD",
    "STRATEGY_MVP",
    "record_mvp_signal_latency",
    "record_mvp_trade",
    "update_mvp_position_size",
    "update_mvp_pnl",
    # Indicator constants
    "INDICATOR_MACD",
    "INDICATOR_RSI",
    "INDICATOR_BB",
    "INDICATOR_MA",
    # Timeframe constants
    "TIMEFRAME_1M",
    "TIMEFRAME_5M",
    "TIMEFRAME_15M",
    "TIMEFRAME_1H",
    "TIMEFRAME_4H",
    "TIMEFRAME_1D",
    # Candle and indicator metrics
    "update_candle_data",
    "update_macd_indicator",
    "update_mvp_signal_state",
]
