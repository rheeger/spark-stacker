.PHONY: help lint lint-python lint-markdown install-deps start stop clean test spark-app-start spark-app-stop spark-app-clean test-hyperliquid clean-logs restart reset build rebuild force-rebuild

# Default target
help:
	@echo "Available commands for spark-app package:"
	@echo "  make help            : Show this help message"
	@echo "  make build           : Set up virtual environment and install dependencies"
	@echo "  make lint            : Run all linters (Python and Markdown)"
	@echo "  make lint-python     : Run Python linters only (black, pylint, mypy)"
	@echo "  make lint-markdown   : Run Markdown linter only"
	@echo "  make install-deps    : Install all dependencies"
	@echo "  make start           : Start the spark-app services"
	@echo "  make stop            : Stop the spark-app services"
	@echo "  make restart         : Restart the spark-app services"
	@echo "  make reset           : Reset everything (stop, clean, reinstall deps, start)"
	@echo "  make rebuild         : Rebuild Docker image and restart services"
	@echo "  make force-rebuild   : Force rebuild Docker image without cache and restart"
	@echo "  make clean           : Clean up temporary files and build artifacts"
	@echo "  make clean-logs      : Clean up all log files"
	@echo "  make test            : Run tests for spark-app"
	@echo "  make spark-app-clean : Clean up spark-app services and volumes"
	@echo "  make test-hyperliquid: Run Hyperliquid connector tests"

# Build target
build: install-deps
	@echo "Building spark-app package..."
	@echo "Build complete"

# Linting
lint: lint-python lint-markdown

lint-python:
	@echo "Running Python linters..."
	black .
	pylint app/ tests/ scripts/ || true
	mypy app/ tests/ scripts/ || true
	@echo "Python linting complete"

lint-markdown:
	@echo "Running Markdown linter..."
	@if command -v markdownlint > /dev/null; then \
		markdownlint --fix **/*.md; \
	else \
		echo "markdownlint-cli not installed. Install with: npm install -g markdownlint-cli"; \
	fi
	@echo "Markdown linting complete"

# Dependencies
install-deps:
	@echo "Setting up Python virtual environment..."
	python3.11 -m venv .venv
	@echo "Installing Python linting dependencies..."
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install black pylint mypy
	@echo "Installing Markdown linting dependencies..."
	@if command -v npm > /dev/null; then \
		npm install -g markdownlint-cli; \
	else \
		echo "npm not found. Please install Node.js and npm first."; \
	fi
	@echo "Installing application dependencies..."
	.venv/bin/pip install -r requirements.txt
	@echo "Dependencies installed"

# Application commands
start: spark-app-start

stop: spark-app-stop

spark-app-start:
	@echo "Starting spark-app services..."
	cd docker && docker-compose up -d
	@echo "Services started. Check logs with: docker logs -f spark-stacker"

spark-app-stop:
	@echo "Stopping spark-app services..."
	cd docker && docker-compose down
	@echo "Services stopped"

spark-app-clean:
	@echo "Cleaning up spark-app services..."
	cd docker && docker-compose down -v
	@echo "Services cleaned up"

# Run tests
test:
	@echo "Running tests for spark-app..."
	@cd $(CURDIR) && .venv/bin/python -m pytest -v
	@echo "Tests complete"

# Run Hyperliquid connector tests
test-hyperliquid:
	@echo "Running Hyperliquid connector tests..."
	@cd $(CURDIR) && .venv/bin/python -m pytest tests/integration/test_hyperliquid_*.py tests/unit/test_hyperliquid_*.py -v
	@echo "Hyperliquid tests complete"

clean:
	@echo "Cleaning up spark-app package..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .venv
	@echo "Clean complete"

clean-logs:
	@echo "Cleaning up all logs..."
	@cd $(CURDIR) && .venv/bin/python scripts/cleanup_logs.py
	@echo "Logs cleaned up. New logs will be stored in app/logs directory."

# Restart command
restart: stop start
	@echo "Spark app services restarted"

# Reset command - full reset of the environment
reset: spark-app-clean clean
	@echo "Resetting spark-app environment..."
	$(MAKE) rebuild
	@echo "Spark app environment reset complete"

# Rebuild Docker image and restart services
rebuild:
	@echo "Rebuilding Docker image and restarting services..."
	cd docker && docker-compose down
	cd docker && docker-compose build
	cd docker && docker-compose up -d
	@echo "Docker image rebuilt and services restarted"

# Force rebuild Docker image without cache and restart services
force-rebuild:
	@echo "Force rebuilding Docker image without cache and restarting services..."
	cd docker && docker-compose down
	cd docker && docker-compose build --no-cache
	cd docker && docker-compose up -d
	@echo "Docker image force rebuilt and services restarted"
