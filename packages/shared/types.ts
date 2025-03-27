// Common types used across packages

// Trade result types
export type TradeResult = 'success' | 'failure';

// Exchange types
export type ExchangeName = 'hyperliquid' | 'coinbase';

// Trade side types
export type TradeSide = 'buy' | 'sell';

// Position side types
export type PositionSide = 'long' | 'short';

// Order types
export type OrderType = 'market' | 'limit';

// Signal types
export type SignalType = 'buy' | 'sell' | 'neutral';

// API HTTP method types
export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE';

// Timeframe types
export type Timeframe = '1h' | '1d' | '1w';
