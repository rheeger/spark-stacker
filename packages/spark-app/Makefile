.PHONY: help lint lint-python lint-markdown install-deps start stop clean test spark-app-start spark-app-stop spark-app-clean test-hyperliquid clean-logs restart reset build rebuild force-rebuild clean-results clean-old-results test-cli check-monitoring ensure-monitoring status

# Default target
help:
	@echo "Available commands for spark-app package:"
	@echo "  make help            : Show this help message"
	@echo "  make status          : Show status of monitoring and spark-app services"
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
	@echo "  make clean-results   : Clean up all test result artifacts"
	@echo "  make clean-old-results: Clean up old test results but keep the 5 most recent runs"
	@echo "  make test            : Run tests for spark-app"
	@echo "  make test-quick      : Run quick tests (excluding slow tests) with coverage"
	@echo "  make test-slow       : Run only slow tests"
	@echo "  make test-extended   : Run extended test suite (slow tests) with full coverage report"
	@echo "  make spark-app-clean : Clean up spark-app services and volumes"

# Status command to show service states
status:
	@echo "=== Spark Stacker Service Status ==="
	@echo "All services run under the 'spark-stacker' project"
	@echo ""
	@echo "Monitoring Services (should stay running):"
	@docker ps --format "  {{.Names}}: {{.Status}}" | grep -E "(grafana|prometheus|loki|promtail|log-metrics|cadvisor|node-exporter)" || echo "  No monitoring services running"
	@echo ""
	@echo "Spark-App Service (affected by start/stop/restart):"
	@docker ps --format "  {{.Names}}: {{.Status}}" | grep "spark-app" || echo "  spark-app: Not running"
	@echo ""
	@if docker network inspect spark-stacker_monitoring >/dev/null 2>&1; then \
		echo "Monitoring Network: ✅ Available"; \
	else \
		echo "Monitoring Network: ❌ Not available"; \
	fi

# Build target
build: install-deps
	@echo "Building spark-app package..."
	@echo "Build complete"

# Check if monitoring services are running
check-monitoring:
	@if ! docker network inspect spark-stacker_monitoring >/dev/null 2>&1; then \
		echo "Monitoring network not found. Monitoring services are not running."; \
		exit 1; \
	fi

# Ensure monitoring services are running, start them if not
ensure-monitoring:
	@echo "Checking monitoring services..."
	@if ! docker network inspect spark-stacker_monitoring >/dev/null 2>&1; then \
		echo "Monitoring services not running. Starting them first..."; \
		cd ../monitoring && make monitoring-start; \
		echo "Waiting for monitoring network to be ready..."; \
		sleep 5; \
	else \
		echo "Monitoring services are already running."; \
	fi

# Linting
lint: lint-python lint-markdown

lint-python:
	@echo "Running Python linters..."
	black .
	pylint app/ tests/ || true
	mypy app/ tests/ || true
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

spark-app-start: ensure-monitoring
	@echo "Starting spark-app service..."
	cd docker && docker-compose --project-name spark-stacker up -d app
	@echo "Spark-app service started. Check logs with: docker logs -f spark-app"

spark-app-stop:
	@echo "Stopping spark-app service..."
	cd docker && docker-compose --project-name spark-stacker stop app
	@echo "Spark-app service stopped"

spark-app-clean:
	@echo "Cleaning up spark-app service..."
	cd docker && docker-compose --project-name spark-stacker rm -f app
	@echo "Spark-app service cleaned up"

# Run tests
test:
	@echo "Running tests for spark-app..."
	@cd $(CURDIR) && .venv/bin/python -m pytest -v
	@echo "Tests complete"

# Run quick tests (skip slow tests)
test-quick:
	@echo "Running quick tests (excluding slow tests) with coverage..."
	@cd $(CURDIR) && .venv/bin/python -m pytest -m "not slow" --cov=app -v --durations=10
	@echo "Quick tests complete. For full coverage report, run: 'coverage html'"

# Run only slow tests
test-slow:
	@echo "Running only slow tests..."
	@cd $(CURDIR) && .venv/bin/python -m pytest -m "slow" -v
	@echo "Slow tests complete"

# Run extended test suite with full coverage report
test-extended:
	@echo "Running extended test suite (slow tests) with coverage..."
	@cd $(CURDIR) && .venv/bin/python -m pytest -m "slow" --cov=app --cov-report=html --cov-report=term-missing -v
	@echo "Extended tests complete. Coverage report generated in _htmlcov/"

# Clean test results
clean-results:
	@echo "Cleaning up test result artifacts..."
	rm -rf tests/__test_results__
	rm -rf tests/**/__test_results__
	rm -rf tests/**/test_results
	rm -rf test_results
	rm -rf _htmlcov
	@echo "Test results cleaned up"

# Clean old test results but keep the 5 most recent runs
clean-old-results:
	@echo "Cleaning up old test result runs (keeping the 5 most recent)..."
	@if [ -d "test_results" ]; then \
		cd test_results && ls -tp | grep run_ | tail -n +6 | xargs -I {} rm -rf -- {}; \
	fi
	@echo "Old test results cleaned up"

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
	@cd $(CURDIR) && .venv/bin/python -m tests.utils.cleanup_logs
	@echo "Logs cleaned up. New logs will be stored in app/_logs directory."

# Restart command
restart: ensure-monitoring
	@echo "Restarting spark-app service..."
	cd docker && docker-compose --project-name spark-stacker restart app
	@echo "Spark app service restarted"

# Reset command - full reset of the environment
reset: spark-app-clean clean
	@echo "Resetting spark-app environment..."
	$(MAKE) rebuild
	@echo "Spark app environment reset complete"

# Rebuild Docker image and restart services
rebuild: ensure-monitoring
	@echo "Rebuilding spark-app Docker image and restarting service..."
	cd docker && docker-compose --project-name spark-stacker stop app
	cd docker && docker-compose --project-name spark-stacker build app
	cd docker && docker-compose --project-name spark-stacker up -d app
	@echo "Spark-app Docker image rebuilt and service restarted"

# Force rebuild Docker image without cache and restart services
force-rebuild: ensure-monitoring
	@echo "Force rebuilding spark-app Docker image without cache and restarting service..."
	cd docker && docker-compose --project-name spark-stacker stop app
	cd docker && docker-compose --project-name spark-stacker build --no-cache app
	cd docker && docker-compose --project-name spark-stacker up -d app
	@echo "Spark-app Docker image force rebuilt and service restarted"
