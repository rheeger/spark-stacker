# Use an official Python runtime as a parent image
FROM python:3.11.8-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies
RUN set -ex; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends \
    build-essential=12.9 \
    curl \
    gettext-base \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bootstrap script
COPY scripts/bootstrap.sh /usr/local/bin/bootstrap.sh
RUN chmod +x /usr/local/bin/bootstrap.sh

# Copy application code to the app directory (the root directory)
COPY app/ /app/app/
COPY __init__.py /app/

# Create log directory
RUN mkdir -p /app/logs && \
    chmod -R 777 /app/logs

# Create data directory
RUN mkdir -p /app/data && \
    chmod -R 777 /app/data

# Set environment variables
ENV PYTHONPATH=/app
ENV CONFIG_FILE=/app/config.json
ENV PATH="/usr/bin:${PATH}"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Create a non-root user and switch to it
RUN useradd -m appuser && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /usr/local/bin/bootstrap.sh && \
    touch /app/.env && \
    chmod 644 /app/.env && \
    chown appuser:appuser /app/.env

USER appuser

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Use bootstrap script as entrypoint
ENTRYPOINT ["/usr/local/bin/bootstrap.sh"]

# Run the application
CMD ["python", "-m", "app.main", "--config", "/app/config.json"]
