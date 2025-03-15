# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    gettext-base \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the bootstrap script
COPY scripts/bootstrap.sh /usr/local/bin/bootstrap.sh
RUN chmod +x /usr/local/bin/bootstrap.sh

# Copy the application
COPY . .

# Create logs directory
RUN mkdir -p logs

# Create a non-root user and switch to it
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Use bootstrap script as entrypoint
ENTRYPOINT ["/usr/local/bin/bootstrap.sh"]

# Run main.py when the container launches
CMD ["python", "app/main.py", "--config", "/app/config/config.json"]