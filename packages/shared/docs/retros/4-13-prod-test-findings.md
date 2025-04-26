# 4/13 Production Test Findings

## Executive Summary

A production test of the Spark Stacker trading system on April 13, 2025 revealed several critical issues that prevented the system from executing trades as intended. Despite the system appearing to function in the logs, the actual trades executed with incorrect parameters and logging was incomplete.

## Critical Issues

### 1. Leverage Misconfiguration

**Issue:** The system was configured to trade with 1.0x leverage but actually executed trades at 25x leverage.

**Evidence:**
- In logs, the system repeatedly states: "Adjusted leverage to 1.0 based on limits"
- However, manual position closing revealed actual leverage was set to 25x
- At 1744548945135, log shows: "Found max exchange leverage: 25.0" before stating "Adjusted leverage to 1.0"

**Root Cause:** The leverage setting is being calculated correctly but not properly applied to the order parameters sent to the Hyperliquid API.

### 2. Missing Order Logs

**Issue:** Despite orders being executed, the orders.log file is completely empty.

**Evidence:**
- At initialization (1744548942862), system creates "dedicated orders logger" for Hyperliquid
- Multiple orders were placed and confirmed in the main log
- At 1744548945484 and 1744548949747, orders were executed
- The orders.log file remains empty

**Root Cause:** The logging mechanism for orders is not properly writing to the designated file.

### 3. Order Size Discrepancies

**Issue:** Planned order sizes significantly differ from actually filled amounts.

**Evidence:**
- Main order calculation: "Final position calculation for ETH-USD: size=7.37, leverage=1.0x"
- Actual filled amount: "filled_amount': '1.2068'"
- Similar issue with hedge order: planned 1.4749 ETH but filled 1.475 ETH

**Root Cause:** Likely related to liquidity limitations or incorrect order sizing implementation.

### 4. Orderbook Retrieval Failures

**Issue:** System consistently fails to retrieve orderbook data.

**Evidence:**
- Multiple occurrences of "Failed to get orderbook after 3 attempts"
- Error: "Invalid orderbook format"
- Messages like "Empty price levels for ETH-USD, cannot calculate optimal price"
- At 1744548947478, detailed invalid orderbook format error

**Root Cause:** The system cannot parse the orderbook format returned by the Hyperliquid API.

### 5. Position Monitoring Failures

**Issue:** System cannot retrieve current positions.

**Evidence:**
- Repeated "Failed to get positions: 'coin'" errors throughout the log
- At 1744549256104, system fetches balance but not position data

**Root Cause:** Incorrect parsing of the API response for positions or API inconsistency.

### 6. Order Placement Issues

**Issue:** Some orders are rejected or only partially filled.

**Evidence:**
- Hedge order rejection: "Order could not immediately match against any resting orders. asset=1"
- Limited fill amounts for main orders
- "Limited liquidity for ETH-USD order" warnings

**Root Cause:** Combination of market liquidity issues and order type/execution strategy.

### 7. Signal Processing Problems

**Issue:** Signals may not be correctly processed due to market data issues.

**Evidence:**
- MACD indicator generates signals but corresponding trades don't execute as expected
- At 1744549256030, system generates a BUY signal that results in partial execution

**Root Cause:** Combination of order execution issues and possibly signal thresholds.

## Technical Details

### Order Execution Flow

1. System calculates position size based on account balance (typically 1% of balance)
2. Adjusts for confidence level from signal (0.50-0.53 in samples)
3. Attempts to fetch orderbook for optimal pricing
4. Falls back to market price when orderbook retrieval fails
5. Submits order with IOC (Immediate-or-Cancel) time-in-force
6. Order is partially filled or rejected
7. System considers trade "processed" regardless of outcome

### API Response Issues

The system struggles with parsing several API responses:

1. **Orderbook Format**: The returned format has nested arrays and objects that don't match expected format
   ```json
   {'coin': 'ETH', 'time': 1744548947478, 'levels': [[{'px': '1603.3', 'sz': '508.0683', 'n': 20}, ...], [...]]}
   ```

2. **Position Data**: Missing expected 'coin' field in position data

### Hedge Ratio Issues

The system uses a fixed hedge ratio (appears to be 0.2 or 20%) but doesn't verify both sides of the trade executed properly before considering the trade complete.

## Revised Recommendations

1. **API Format Verification**: Capture and document exact request/response formats from the Hyperliquid API
   - Establish a robust integration test suite to validate all API interactions
   - Document the actual data structures returned by each endpoint
   - Create schema validators for all API responses

2. **Fix Orderbook Parsing**: Update parser to handle the nested format returned by the API
   - Implement adapters that can transform API responses into expected formats
   - Add comprehensive error handling for malformed responses

3. **Fix Position Data Retrieval**: Update position data parsing to match actual API format
   - Document the exact structure of position data objects
   - Implement proper object mapping for nested position data

4. **Fix Leverage Application**: Ensure the calculated leverage is properly applied to orders
   - Add verification steps to confirm leverage settings are applied
   - Implement end-to-end tests for leverage application

5. **Repair Order Logging**: Fix logger implementation to properly write to orders.log
   - Add comprehensive logging for order lifecycle
   - Implement validation for log file creation and writing

## Revised Next Steps

1. **Immediate API Integration Test Suite**:
   - Create a comprehensive test suite that verifies each API request/response format
   - Capture real production API formats for all endpoints used in the connector
   - Validate that our connector properly handles these formats

2. **API Documentation and Schema Validation**:
   - Document exact request/response formats for all Hyperliquid API endpoints
   - Create schema validators for each response type
   - Update connector implementation based on documented schemas

3. **Connector Hardening**:
   - Fix orderbook parsing to handle nested format
   - Update position data retrieval to handle actual response format
   - Ensure leverage parameter is correctly applied in order requests
   - Fix logger initialization for order logs

4. **Integration Testing Framework**:
   - Implement record/replay capability for API responses
   - Create test data generators for various market conditions
   - Add comprehensive error case testing

5. **Progressive Validation**:
   - Test connector with minimal API interactions on testnet
   - Validate each API endpoint individually before testing integrated flows
   - Perform thorough code review after fixes are implemented

No further production tests will be conducted until the integration test suite has been fully implemented and all tests pass with actual API response formats.
