FROM python:3.11-slim

WORKDIR /app

# Copy the log metrics exporter script
COPY monitoring/exporters/log-metrics.py /app/log-metrics.py

# Install dependencies
RUN pip install --no-cache-dir prometheus-client

# Make the script executable
RUN chmod +x /app/log-metrics.py

# Set environment variables
ENV LOG_DIR=/logs
ENV PORT=9001
ENV PYTHONUNBUFFERED=1

# Expose the metrics port
EXPOSE ${PORT}

# Create the logs directory if needed
RUN mkdir -p ${LOG_DIR}

# Start with a command to ensure we see the logs
CMD ["sh", "-c", "ls -la ${LOG_DIR} && python -u /app/log-metrics.py --log-dir ${LOG_DIR} --port ${PORT}"]
