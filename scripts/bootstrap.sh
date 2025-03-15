#!/bin/bash
set -e

# Create logs directory if it doesn't exist
mkdir -p /app/logs

# Process the config.json file and replace environment variables
CONFIG_FILE="/app/config/config.json"
echo "Processing config file: $CONFIG_FILE"

# Print config file content for debugging
echo "Config file content before substitution:"
cat "$CONFIG_FILE"

# Use envsubst to replace environment variables in the config file
# Create a temporary file
TEMP_FILE=$(mktemp)
envsubst < "$CONFIG_FILE" > "$TEMP_FILE"
mv "$TEMP_FILE" "$CONFIG_FILE"

echo "Config file content after substitution:"
cat "$CONFIG_FILE"

echo "Environment variables have been injected into the config file"

# Print important env vars for debugging (without sensitive data)
echo "Environment variables:"
echo "WEBHOOK_HOST: $WEBHOOK_HOST"
echo "WEBHOOK_PORT: $WEBHOOK_PORT"
echo "HYPERLIQUID_TESTNET: $HYPERLIQUID_TESTNET"
echo "HYPERLIQUID_RPC_URL: $HYPERLIQUID_RPC_URL"
echo "WALLET_ADDRESS set: $(if [ -n "$WALLET_ADDRESS" ]; then echo "yes"; else echo "no"; fi)"
echo "PRIVATE_KEY set: $(if [ -n "$PRIVATE_KEY" ]; then echo "yes"; else echo "no"; fi)"

# Execute the command passed to this script
echo "Starting Spark Stacker..."
exec "$@" 