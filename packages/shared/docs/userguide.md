# User Guide: On-Chain Perpetual Trading System

## System Overview

The **On-Chain Perpetual Trading System** is designed to execute high-leverage trades on
decentralized perpetual futures exchanges while implementing sophisticated hedging strategies to
protect your capital. The system follows technical indicators to enter trades and automatically
manages risk through position sizing, stop-losses, and strategic hedge positions.

## 1. **Setup & Installation**

### 1.1 System Requirements

- Python 3.9 or higher
- Node.js 16+ (if using TradingView webhook integration)
- Access to Ethereum RPC endpoints (Optimism/Base networks for Synthetix)
- Exchange accounts (Synthetix wallet, Hyperliquid account)

### 1.2 Installation Process

1. Clone the repository:

   ```bash
   git clone https://github.com/user/on-chain-perp-trading.git
   cd on-chain-perp-trading
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:

   - Create a `.env` file based on the provided `.env.example`
   - Add your exchange API keys, wallet details, and network endpoints

   ```env
   # Synthetix (Optimism) Configuration
   OPTIMISM_RPC_URL=https://optimism-mainnet.infura.io/v3/YOUR_KEY
   SYNTHETIX_WALLET_PRIVATE_KEY=your_private_key_here

   # Hyperliquid Configuration
   HYPERLIQUID_API_KEY=your_api_key
   HYPERLIQUID_PRIVATE_KEY=your_signed_key

   # TradingView Webhook (Optional)
   WEBHOOK_SECRET=your_webhook_secret
   ```

## 2. **Exchange Connections**

### 2.1 Connecting to Synthetix

Synthetix Perps is a decentralized perpetual futures platform running on Optimism and Base networks.
The trading system interacts with Synthetix through Web3 calls to its smart contracts.

1. Ensure you have:

   - An Optimism L2 wallet with ETH for gas
   - sUSD (Synthetix USD) for margin/collateral
   - Properly configured RPC endpoint

2. Verify connection:

   ```bash
   python test_connection.py --exchange synthetix
   ```

3. Key Synthetix features to understand:
   - Oracle-based pricing (updates every ~8 seconds)
   - Up to 50× leverage on major assets
   - Deep liquidity pool (no orderbook)
   - Funding rates applied to positions

### 2.2 Connecting to Hyperliquid

Hyperliquid is a high-performance on-chain orderbook exchange with its own L1 chain.

1. Ensure you have:

   - Generated API keys from the Hyperliquid interface
   - USDC for margin/collateral (transferred to Hyperliquid)
   - Properly signed your API credentials

2. Verify connection:

   ```bash
   python test_connection.py --exchange hyperliquid
   ```

3. Key Hyperliquid features to understand:
   - On-chain orderbook (~100,000 orders/sec)
   - ~30+ crypto assets available
   - Up to 50× leverage on major pairs
   - Market and limit order support

## 3. **Configuring Trading Strategies**

### 3.1 Strategy Types

The system supports two main ways to generate trading signals:

#### a) Built-in Technical Indicators

Configure the system to use internal calculations based on:

- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (EMA, SMA)

Example configuration in `config.yml`:

```yaml
strategy:
  type: "indicator"
  primary_indicator: "RSI"
  parameters:
    period: 14
    overbought: 70
    oversold: 30
  confirmation_indicator: "MACD"
  parameters:
    fast: 12
    slow: 26
    signal: 9
```

#### b) TradingView Pine Script Integration

Use your custom TradingView strategies by setting up webhook alerts:

1. Create a Pine Script strategy in TradingView
2. Set up an alert with webhook delivery to your system's endpoint
3. Configure the message format in TradingView:

   ```json
   {
     "passphrase": "YOUR_SECRET",
     "asset": "{{ticker}}",
     "signal": "{{strategy.order.action}}",
     "price": {{close}},
     "confidence": 0.8
   }
   ```

4. Start the webhook receiver:

   ```bash
   python webhook_server.py
   ```

### 3.2 Risk Parameters

Configure these critical risk management settings:

```yaml
risk_management:
  # Position Sizing & Leverage
  max_leverage: 10
  main_position_ratio: 0.8 # Percentage of capital for main position
  hedge_position_ratio: 0.2 # Percentage of capital for hedge position
  hedge_leverage_ratio: 0.5 # Hedge leverage as fraction of main leverage

  # Protection Mechanisms
  stop_loss_percent: -10.0 # Close positions if drawdown exceeds 10%
  take_profit_percent: 20.0 # Take profit at 20% gain
  max_daily_loss_percent: -15.0 # Stop trading if daily loss exceeds 15%

  # Liquidation Protection
  liquidation_warning_threshold: 1.2 # 1.2× maintenance margin
  hedge_profit_threshold: 5.0 # Close hedge when 5% profit to protect main
```

## 4. **Understanding the Hedging Strategy**

### 4.1 How Hedging Works

The system protects your principal through a strategic hedging approach:

1. **Main Position**: When a signal is received, the system opens a larger position in the direction
   of the signal (long or short).
2. **Hedge Position**: Simultaneously, it opens a smaller position in the opposite direction.

For example:

- $100 available capital
- $80 (80%) used for main position at 10× leverage = $800 exposure
- $20 (20%) used for hedge position at 5× leverage = $100 exposure
- Net exposure: $700 in signal direction ($800 - $100)

This approach:

- Retains significant upside if the signal is correct
- Provides protection if the signal is wrong
- Reduces overall volatility of your account equity

### 4.2 Dynamic Hedge Management

The hedge isn't static - the system actively manages it:

- If the main position becomes strongly profitable, the system may reduce the hedge to maximize
  returns
- If the trade moves against the main position, the hedge generates profit to offset some losses
- In extreme adverse moves, the hedge profit helps prevent liquidation of the main position

## 5. **Running the Trading System**

### 5.1 Backtesting Mode

Before committing real capital, test your strategy on historical data:

```bash
python run_bot.py --mode backtest --strategy my_strategy --start-date 2023-01-01 --end-date 2023-12-31
```

The backtesting engine will:

- Load historical price data for your chosen assets
- Apply the selected indicators and generate signals
- Simulate trades including main and hedge positions
- Calculate performance metrics (win rate, profit factor, max drawdown)
- Generate a detailed report of strategy performance

Review these metrics carefully to assess strategy viability before live trading.

### 5.2 Paper Trading Mode

Once your strategy passes backtesting, test it with real-time data but no actual trades:

```bash
python run_bot.py --mode paper --strategy my_strategy
```

This will:

- Connect to live exchange APIs and receive current market data
- Generate signals and simulate order execution
- Track hypothetical positions and P&L
- Log all actions just like in live mode

Monitor paper trading for at least several days to ensure the strategy behaves as expected.

### 5.3 Live Trading Mode

When ready for live deployment, start with minimal capital:

```bash
python run_bot.py --mode live --strategy my_strategy --capital 100
```

The system will:

1. Monitor markets and indicators in real-time
2. Execute main and hedge positions when signals are triggered
3. Actively manage risk through stop-losses and hedge adjustments
4. Log all activities and send alerts for critical events

Monitor system performance closely during initial deployment.

## 6. **Monitoring Your Trades**

### 6.1 Live Dashboard

Access the live dashboard to monitor performance:

```bash
python dashboard.py
```

The dashboard shows:

- Current open positions (main and hedge)
- Real-time P&L and margin utilization
- Recent trade history and performance stats
- Risk metrics and liquidation warnings

### 6.2 Performance Analysis

Generate detailed reports on your trading performance:

```bash
python analyze_performance.py --start-date 2023-01-01 --end-date 2023-12-31
```

This analysis includes:

- Win/loss ratio and average trade P&L
- Strategy performance by market conditions
- Hedge effectiveness metrics
- Risk-adjusted return calculations (Sharpe ratio, etc.)

### 6.3 Logging and Alerts

The system maintains comprehensive logs:

- Trade execution logs: `logs/trades.log`
- System event logs: `logs/system.log`
- Error logs: `logs/error.log`

Configure alerts via email or messaging services in `config.yml`:

```yaml
alerts:
  enabled: true
  methods:
    email: 'your@email.com'
    telegram: true
  events:
    - 'new_position_opened'
    - 'position_closed'
    - 'stop_loss_triggered'
    - 'error_occurred'
```

## 7. **Best Practices & Tips**

### 7.1 Capital Management

- Start with a small portion of your capital (e.g., 5-10%)
- Gradually increase as you gain confidence in the system
- Never commit funds you can't afford to lose
- Consider dividing capital across multiple strategies

### 7.2 Risk Optimization

- Run multiple backtests to optimize risk parameters
- Start with conservative leverage (e.g., 5× instead of 20×)
- Use wider stop-losses in volatile markets
- Consider adjusting hedge ratio based on market volatility

### 7.3 System Maintenance

- Regularly check for system updates
- Monitor exchange API changes that may affect connectivity
- Keep backup copies of your strategy configurations
- Periodically review and optimize your strategies

## 8. **Troubleshooting**

### Common Issues

- **No Trades Executing:** Check indicator thresholds, they may be too restrictive
- **Connection Errors:** Verify API keys and network connectivity
- **Unexpected Losses:** Review hedge ratio and leverage settings
- **Excessive Fees:** Consider using limit orders instead of market orders
- **Rapid Capital Depletion:** Immediately reduce position size and leverage

### Getting Support

- Check the FAQ in the repository documentation
- Review existing issues on GitHub
- Join the community Discord/Telegram
- File a detailed bug report for persistent issues

This user guide covers the essential aspects of operating the On-Chain Perpetual Trading System. For
more detailed information, refer to the Technical Specification document.
