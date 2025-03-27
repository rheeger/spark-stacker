.PHONY: help install-all build-all start-all stop-all test-all lint-all clean-all monitoring spark-app shared

# Default target
help:
	@echo "Root-level Makefile for the Spark-Stacker project"
	@echo ""
	@echo "Available commands:"
	@echo "  make help            : Show this help message"
	@echo ""
	@echo "Project-wide commands:"
	@echo "  make install-all     : Install dependencies for all packages"
	@echo "  make build-all       : Build all packages"
	@echo "  make start-all       : Start all services (monitoring and spark-app)"
	@echo "  make stop-all        : Stop all services"
	@echo "  make test-all        : Run tests for all packages"
	@echo "  make lint-all        : Run linters for all packages"
	@echo "  make clean-all       : Clean up all packages"
	@echo ""
	@echo "Package-specific commands:"
	@echo "  make monitoring      : Show monitoring package commands"
	@echo "  make spark-app       : Show spark-app package commands"
	@echo "  make shared          : Show shared package commands"

# Package help commands
monitoring:
	@echo "Monitoring package commands:"
	@cd packages/monitoring && make help

spark-app:
	@echo "Spark-app package commands:"
	@cd packages/spark-app && make help

shared:
	@echo "Shared package commands:"
	@echo "No specific commands available yet"

# Project-wide commands
install-all:
	@echo "Installing dependencies for all packages..."
	@echo "Installing root dependencies..."
	yarn install
	@echo "Installing spark-app dependencies..."
	@cd packages/spark-app && make install-deps
	@echo "Installing monitoring dependencies..."
	@cd packages/monitoring && make install-deps
	@echo "All dependencies installed"

build-all:
	@echo "Building all packages..."
	@echo "Building spark-app..."
	@cd packages/spark-app && make build
	@echo "Building monitoring..."
	@cd packages/monitoring && make build
	@echo "All packages built"

start-all:
	@echo "Starting all services..."
	@echo "Starting monitoring services..."
	@cd packages/monitoring && make monitoring-start
	@echo "Starting spark-app services..."
	@cd packages/spark-app && make spark-app-start
	@echo "All services started"

stop-all:
	@echo "Stopping all services..."
	@echo "Stopping monitoring services..."
	@cd packages/monitoring && make monitoring-stop
	@echo "Stopping spark-app services..."
	@cd packages/spark-app && make spark-app-stop
	@echo "All services stopped"

test-all:
	@echo "Running tests for all packages..."
	@echo "Testing spark-app..."
	-@cd packages/spark-app && make test
	@echo "Testing monitoring..."
	-@cd packages/monitoring && make test
	@echo "All tests complete"

lint-all:
	@echo "Linting all packages..."
	@echo "Linting spark-app..."
	-@cd packages/spark-app && make lint
	@echo "Linting monitoring..."
	-@cd packages/monitoring && make lint
	@echo "All linting complete"

clean-all:
	@echo "Cleaning all packages..."
	@echo "Stopping all services first..."
	@$(MAKE) stop-all
	@echo "Cleaning spark-app..."
	@cd packages/spark-app && make clean
	@echo "Cleaning monitoring..."
	@cd packages/monitoring && make clean
	@echo "All packages cleaned"
