.PHONY: help lint lint-python lint-markdown install-deps

help:
	@echo "Available commands:"
	@echo "  make lint            : Run all linters (Python and Markdown)"
	@echo "  make lint-python     : Run Python linters only (black, pylint, mypy)"
	@echo "  make lint-markdown   : Run Markdown linter only"
	@echo "  make install-deps    : Install linting dependencies"

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

install-deps:
	@echo "Installing Python linting dependencies..."
	pip install black pylint mypy
	@echo "Installing Markdown linting dependencies..."
	@if command -v npm > /dev/null; then \
		npm install -g markdownlint-cli; \
	else \
		echo "npm not found. Please install Node.js and npm first."; \
	fi
	@echo "Dependencies installed"
