# Product Requirements Document (PRD)

## Overview

The **On-Chain Perpetual Trading System** is designed to execute high-leverage trades on decentralized perpetual futures exchanges, with sophisticated hedging mechanics to protect principal while maximizing returns. The system follows technical indicators to initiate positions and implements risk management techniques to control drawdowns.

## Key Features

### 1. **Exchange Integration**

- Support for **DEXs**:
  - **Primary Focus:** Synthetix Perps (Optimism/Base), Hyperliquid DEX (Layer-1 chain)
  - **CEX Support:** Coinbase Exchange
  - **Future Expansion:** dYdX, GMX
- Integration Methods:
  - **Synthetix:** Web3 for contract interaction with Perps V2/V3, Synthetix SDK
  - **Hyperliquid:** HTTP API authenticated with ECDSA signatures, Python SDK, WebSocket API
  - **Coinbase:** REST and WebSocket APIs with HMAC authentication, Advanced Trade API for futures
- Key Features by Exchange:
  - **Synthetix:** Deep liquidity pool, oracle-based pricing, up to 50× leverage
  - **Hyperliquid:** On-chain order book, ~100k orders/sec throughput, ~30+ crypto assets
  - **Coinbase:** High liquidity, advanced order types, reliable infrastructure, futures markets

### 2. **Indicator Support & Trade Execution**

- **Built-in Technical Indicators**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Exponential Moving Averages (EMA, SMA)
- **TradingView Pine Script Integration**:
  - Webhook alert reception from TradingView
  - Custom indicator strategy support
- **Direct Technical Analysis Computation**:
  - Use of TA-Lib / Pandas-TA libraries for internal indicator calculation
  - Real-time processing of market data from exchange APIs
- **Order Flow Pipeline**:
  1. Signal Reception (Pine Script alert or internal computation)
  2. Order Preparation (determine position size, leverage, direction)
  3. Primary Order Execution (main position in signal direction)
  4. Hedge Order Execution (counter-position for protection)
  5. Confirmation & Position Monitoring

### 3. **Hedging Mechanics for Principal Protection**

- **Primary-Hedge Position Structure**:
  - Main Position: Larger size with high leverage in signal direction
  - Hedge Position: Smaller counter-position (20-50% of main position size)
  - Net Exposure: Remains in favor of signal direction
- **Position Sizing & Leverage**:
  - Configurable hedge ratio (e.g., hedge = 20% of main position)
  - Use of leverage on both positions (e.g., 10× main, 5× hedge)
  - Example: $80 margin at 10× for $800 long exposure, $20 margin at 5× for $100 short hedge
- **Execution Options**:
  - Same exchange (if supporting isolated positions)
  - Split across exchanges (e.g., main on Synthetix, hedge on Hyperliquid)
- **Dynamic Hedge Management**:
  - Reduce hedge when trade becomes profitable
  - Hold hedge as protection against adverse moves
  - Close main position and rely on hedge if trend reverses

### 4. **Risk Management Configuration**

- **Position Control Parameters**:
  - **Max Leverage:** Configurable limits (e.g., 10×, 20×, up to 50×)
  - **Hedge Ratio:** Percentage of main position size (configurable: 20-50%)
  - **Max Position Size:** Percentage of total capital per trade
  - **Stop-Loss Levels:** Automatic closure on % drawdown (e.g., -10%)
- **Account Protection Mechanisms**:
  - **Max Daily Loss / Drawdown Limit:** Auto-shutdown if breached
  - **Margin Monitoring:** Track margin ratio and prevent liquidation
  - **Liquidation Prevention:** Auto-deleveraging when approaching limits
- **Order Execution Safety**:
  - **Slippage Control:** Limit orders where appropriate to minimize cost
  - **Confirmation Logic:** Verify transaction success before proceeding
  - **Error Handling:** Retry logic for failed orders with adaptive parameters

### 5. **Monitoring & Performance Analysis**

- **Live Dashboard**
  - Open positions overview with combined P&L
  - Margin health indicators and liquidation warnings
  - Hedge effectiveness metrics
- **Historical Performance Reports**
  - Strategy success rate analysis
  - Risk-adjusted return calculation
  - Hedge contribution analysis (protection provided)

## User Stories

### 1. **Strategy Configuration**

- **As a trader,** I want to configure my preferred indicators and risk settings, so the system can execute trades based on my strategy.
- **Acceptance Criteria:**
  - User can select built-in indicators or connect TradingView alerts
  - User can define risk parameters (max leverage, hedge ratio, stop-loss)
  - User can test the strategy via backtesting before deploying live capital

### 2. **Hedged Trade Execution**

- **As a trader,** I want the system to automatically hedge my position to reduce risk while maximizing potential returns.
- **Acceptance Criteria:**
  - System executes a primary high-leverage trade based on the indicator signal
  - A smaller hedge trade is placed in the opposite direction
  - Both positions are monitored and adjusted according to market movement
  - Overall account equity fluctuates less than with a single unhedged position

### 3. **Risk Management & Monitoring**

- **As a trader,** I want to protect my principal while still capturing significant upside on correct signals.
- **Acceptance Criteria:**
  - System never risks entire principal on a single trade
  - Hedge positions provide meaningful protection during adverse moves
  - Position sizing and leverage follow configurable risk rules
  - Automatic position adjustments prevent liquidation scenarios

## Example Hedged Trade Scenario

**Starting Conditions:**

- Available Capital: $100
- Indicator: RSI and MACD both signal Long on ETH

**Trade Setup:**

- Main Position: Long ETH with 10× leverage using $80 margin = $800 notional exposure
- Hedge Position: Short ETH with 5× leverage using $20 margin = $100 notional exposure
- Net Exposure: $700 long ($800 - $100)

**Outcome Scenarios:**

1. **Market rises 5%**: 
   - Long position: +$40 profit
   - Short hedge: -$5 loss
   - Net result: +$35 profit (35% return on $100 principal)

2. **Market falls 5%**:
   - Long position: -$40 loss
   - Short hedge: +$5 profit
   - Net result: -$35 loss (35% loss on principal, but 12.5% less than unhedged)

The hedging strategy provides significant upside capture while reducing drawdowns and preventing catastrophic losses.

## System Constraints

- API connectivity must be robust with automatic reconnection
- Execution time must be optimized to avoid slippage
- System should handle network interruptions gracefully
- Security measures must be in place for API keys and private keys 
- Backtesting and paper trading environments must accurately simulate real conditions
