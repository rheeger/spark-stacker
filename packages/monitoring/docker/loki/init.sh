#!/bin/bash
set -e

# Create necessary directories with proper permissions
mkdir -p /loki/chunks
mkdir -p /loki/boltdb-shipper-active
mkdir -p /loki/boltdb-shipper-cache
mkdir -p /loki/compactor
mkdir -p /wal

# Set proper ownership for Loki user (10001)
chown -R 10001:10001 /loki
chown -R 10001:10001 /wal

# Set proper permissions
chmod -R 755 /loki
chmod -R 755 /wal

echo "Loki storage directories initialized with proper permissions."
