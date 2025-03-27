# Running Spark Stacker in Docker

This document provides instructions for running the Spark Stacker trading system in Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Setup Instructions

### 1. Environment Configuration

1. Copy the example environment file to create your own:

   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and fill in your own values:

   ```bash
   nano .env
   ```

   The most important settings to change are:
   - `WALLET_ADDRESS`: Your Ethereum wallet address
   - `PRIVATE_KEY`: Your Ethereum private key
   - `WEBHOOK_AUTH_TOKEN`: A secure random token for webhook authentication
   - `TRADINGVIEW_PASSPHRASE`: A passphrase for TradingView alerts

### 2. Configuration

The configuration file is located at `config/config.json`. This file uses environment variable substitution, so you generally don't need to edit it directly. Any values defined in your `.env` file will be automatically inserted into the config.

If you need to customize the configuration beyond what's in the `.env` file, you can edit the config file directly.

### 3. Building and Running

Build and start the container:

```bash
docker-compose up -d
```

To view the logs:

```bash
docker-compose logs -f
```

To stop the container:

```bash
docker-compose down
```

### 4. Testing the Setup

Once the container is running, you can test the health check endpoint:

```bash
curl http://localhost:8080/
```

You should see a response indicating the service is running.

## Docker Environment Variables

The full list of environment variables is documented in the `.env.example` file. Here are the key categories:

- **Exchange Credentials**: API keys and wallet information
- **Server Configuration**: Webhook server settings
- **Trading Parameters**: Risk management and trading settings
- **Logging Configuration**: Log levels and file paths

## Customizing the Docker Setup

### Volume Mounts

The docker-compose.yml file mounts two directories:

- `./logs`: Container logs are stored here
- `./config`: Configuration files

### Ports

By default, the container exposes port 8080. You can change this by modifying the `WEBHOOK_PORT` in your `.env` file and the corresponding port mapping in the docker-compose.yml file.

## Troubleshooting

### Container Not Starting

Check the logs for error messages:

```bash
docker-compose logs
```

### Configuration Issues

If the container starts but the application isn't working correctly:

1. Verify your environment variables in the `.env` file
2. Check that the config.json file has been properly processed with your environment variables:

   ```bash
   docker-compose exec app cat /app/config/config.json
   ```

### Connection Issues

If you're having trouble connecting to exchanges:

1. Verify your API keys and wallet information
2. Check that the exchange is available and responds to your credentials
3. Set `DRY_RUN=true` to test without making real trades

## Security Recommendations

1. **Never commit your `.env` file with real credentials to version control**
2. Regularly rotate your API keys and authentication tokens
3. Use strong, unique passwords for all credentials
4. Consider using Docker secrets for production deployments
