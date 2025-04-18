#!/bin/bash
set -e

# Create logs directory if it doesn't exist
mkdir -p /app/logs

# Process the config.json file and replace environment variables
CONFIG_FILE=${CONFIG_FILE:-"/app/config.json"}
PROCESSED_CONFIG="/app/processed_config.json"
echo "Processing config file: $CONFIG_FILE"

# Load environment variables if .env file exists
if [ -f /app/.env ]; then
  echo "Loading environment variables from /app/.env"
  set -o allexport
  source /app/.env
  set +o allexport
fi

# Print config file content for debugging
echo "Config file content before substitution:"
cat "$CONFIG_FILE"

# Copy the original config file to the processed location
cp "$CONFIG_FILE" "$PROCESSED_CONFIG"

# Replace environment variables in the config file using a safer method
for var in $(env | cut -d= -f1); do
  # Skip some common env vars that we don't want to replace
  if [[ "$var" =~ ^(PWD|HOME|PATH|SHELL|TERM|USER|HOSTNAME)$ ]]; then
    continue
  fi

  # Get the value directly without using eval
  val="${!var}"

  # Escape special characters for sed
  escaped_val=$(printf '%s\n' "$val" | sed -e 's/[\/&]/\\&/g')

  # Replace the variable in the config file
  sed -i "s|\${$var}|$escaped_val|g" "$PROCESSED_CONFIG"
done

echo "Config file processed successfully to $PROCESSED_CONFIG"

# Print important env vars for debugging (without sensitive data)
echo "Environment variables:"
echo "WEBHOOK_HOST: $WEBHOOK_HOST"
echo "WEBHOOK_PORT: $WEBHOOK_PORT"
echo "HYPERLIQUID_TESTNET: $HYPERLIQUID_TESTNET"
echo "HYPERLIQUID_RPC_URL: $HYPERLIQUID_RPC_URL"
echo "WALLET_ADDRESS set: $(if [ -n "$WALLET_ADDRESS" ]; then echo "yes"; else echo "no"; fi)"
echo "PRIVATE_KEY set: $(if [ -n "$PRIVATE_KEY" ]; then echo "yes"; else echo "no"; fi)"

# Modify the command to use the processed config
if [[ "$@" == *"--config"* ]]; then
  # Replace the config path in the command
  args=()
  skip_next=false
  for arg in "$@"; do
    if $skip_next; then
      skip_next=false
      continue
    elif [ "$arg" == "--config" ]; then
      args+=("$arg")
      args+=("$PROCESSED_CONFIG")
      skip_next=true
    else
      args+=("$arg")
    fi
  done
  # Execute with the modified arguments
  echo "Starting Spark Stacker with processed config..."
  exec "${args[@]}"
else
  # Append the config argument if not present
  echo "Starting Spark Stacker..."
  exec "$@" --config "$PROCESSED_CONFIG"
fi
