.PHONY: help monitoring-start monitoring-stop monitoring-clean install-deps build start test clean lint

# Default target
help:
	@echo "Available commands for monitoring package:"
	@echo "  make help            : Show this help message"
	@echo "  make monitoring-start: Start monitoring services with Docker"
	@echo "  make monitoring-stop : Stop monitoring services"
	@echo "  make monitoring-clean: Stop and remove Docker volumes"
	@echo "  make install-deps    : Install all dependencies"
	@echo "  make build           : Build the monitoring application"
	@echo "  make start           : Start the monitoring application"
	@echo "  make test            : Run tests for monitoring"
	@echo "  make lint            : Run linters for monitoring code"
	@echo "  make clean           : Clean up temporary files and build artifacts"

# Monitoring services
monitoring-start:
	@echo "Starting monitoring services..."
	cd docker && docker-compose up -d

monitoring-stop:
	@echo "Stopping monitoring services..."
	cd docker && docker-compose down

monitoring-clean:
	@echo "Cleaning up monitoring services..."
	cd docker && docker-compose down -v

# Dependencies
install-deps:
	@echo "Installing dependencies..."
	yarn install
	@echo "Dependencies installed"

# Build and development
build:
	@echo "Building monitoring application..."
	yarn nx build monitoring
	@echo "Build complete"

start:
	@echo "Starting monitoring application..."
	yarn nx serve monitoring
	@echo "Application started"

# Testing and linting
test:
	@echo "Running tests for monitoring..."
	yarn nx test monitoring
	@echo "Tests complete"

lint:
	@echo "Linting monitoring code..."
	yarn nx lint monitoring
	@echo "Linting complete"

# Cleanup
clean:
	@echo "Cleaning up monitoring package..."
	rm -rf node_modules/.cache
	@echo "Clean complete"
