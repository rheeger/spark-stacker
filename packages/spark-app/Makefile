.PHONY: help lint lint-python lint-markdown install-deps start build test clean

# Default target
help:
	@echo "Available commands for spark-app package:"
	@echo "  make help            : Show this help message"
	@echo "  make lint            : Run all linters (Python and Markdown)"
	@echo "  make lint-python     : Run Python linters only (black, pylint, mypy)"
	@echo "  make lint-markdown   : Run Markdown linter only"
	@echo "  make install-deps    : Install all dependencies"
	@echo "  make start           : Start the spark-app application"
	@echo "  make build           : Build the spark-app application"
	@echo "  make test            : Run tests for spark-app"
	@echo "  make clean           : Clean up temporary files and build artifacts"

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
	@echo "Installing Python linting dependencies..."
	pip install black pylint mypy
	@echo "Installing Markdown linting dependencies..."
	@if command -v npm > /dev/null; then \
		npm install -g markdownlint-cli; \
	else \
		echo "npm not found. Please install Node.js and npm first."; \
	fi
	@echo "Installing application dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed"

# Application commands
start:
	@echo "Starting spark-app application..."
	python app/main.py

build:
	@echo "Building spark-app application..."
	# Add build steps specific to the spark-app package
	@echo "Build complete"

test:
	@echo "Running tests for spark-app..."
	pytest tests/
	@echo "Tests complete"

clean:
	@echo "Cleaning up spark-app package..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete"
