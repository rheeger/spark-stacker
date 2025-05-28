# Contributing to Spark Stacker

Welcome to the Spark Stacker project! This guide will help you contribute effectively to our on-chain perpetual trading system.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Yarn package manager
- Git

### Setting Up the Development Environment

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-org/spark-stacker.git
   cd spark-stacker
   ```

2. **Install dependencies:**

   ```bash
   # Install root dependencies
   yarn install

   # Set up Python environment
   cd packages/spark-app
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run initial tests:**
   ```bash
   # Quick test run (recommended before any development)
   cd packages/spark-app
   make test-quick
   ```

## 📝 Development Workflow

### Commit Message Convention

All commit messages must follow this format:

```
phase<X.Y.Z>: <type>(<scope>): <short description>
```

**Examples:**

- `phase3.5.1: feat(indicators): Add Stochastic RSI indicator`
- `phase4: fix(monitoring): Resolve Grafana dashboard loading issue`
- `phase2: test(backtesting): Add comprehensive API response validation`

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks
- `style`: Code style changes
- `perf`: Performance improvements

### Branch Naming

- `feature/phase-X.Y.Z-descriptive-name`
- `fix/phase-X.Y.Z-bug-description`
- `docs/phase-X.Y.Z-documentation-update`

### Testing Requirements

Before submitting any pull request:

1. **Run quick tests:** `make test-quick` (must complete in < 3 minutes)
2. **Run full test suite:** `make test-full` (includes coverage report)
3. **Ensure code quality:** `make lint`

## 🔧 Adding New Indicators

### Using the Indicator Scaffolder

The fastest way to create a new indicator is using our automated scaffolder:

```bash
cd packages/spark-app
./scripts/new_indicator.sh StochasticRSI
```

This script will:

- Create the indicator class file: `app/indicators/stochastic_rsi_indicator.py`
- Create the test file: `tests/indicators/unit/test_stochastic_rsi_indicator.py`
- Automatically register the indicator in `IndicatorFactory`
- Provide you with a fully structured template

### Indicator Development Steps

#### 1. Implement Core Logic

Edit the generated indicator file to implement:

```python
def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate indicator values for the provided price data.

    Args:
        data: Price data with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    Returns:
        DataFrame with indicator values added as new columns
    """
    # Your calculation logic here
```

#### 2. Implement Signal Generation

```python
def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
    """
    Generate trading signals based on indicator values.

    Returns:
        Signal object with direction, confidence, and parameters
    """
    # Your signal logic here
```

#### 3. Write Comprehensive Tests

Update the generated test file with:

- Parameter validation tests
- Calculation accuracy tests
- Signal generation tests
- Edge case handling tests
- Error condition tests

#### 4. Run Tests

```bash
# Test your specific indicator
.venv/bin/python -m pytest tests/indicators/unit/test_your_indicator.py -v

# Run all indicator tests
.venv/bin/python -m pytest tests/indicators/ -v
```

### Indicator Best Practices

#### Code Quality

- Use type hints for all function parameters and return values
- Add comprehensive docstrings
- Follow PEP 8 style guidelines
- Handle edge cases gracefully

#### Performance

- Use vectorized operations (pandas/numpy) when possible
- Avoid loops for large datasets
- Cache expensive calculations
- Consider memory usage for large datasets

#### Testing

- Test with various market conditions (trending, sideways, volatile)
- Include boundary condition tests
- Test with insufficient data scenarios
- Validate mathematical accuracy against known implementations

#### Documentation

- Document the mathematical formula
- Explain parameter meanings and ranges
- Provide usage examples
- Reference academic papers or sources if applicable

### Example Indicator Structure

```python
import logging
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from app.indicators.base_indicator import BaseIndicator, Signal, SignalDirection

logger = logging.getLogger(__name__)

class ExampleIndicator(BaseIndicator):
    """
    Example indicator that demonstrates best practices.

    This indicator calculates a simple moving average and generates
    signals based on price crossovers.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)
        self.period = self.params.get("period", 20)
        self.threshold = self.params.get("threshold", 0.02)

        # Validate parameters
        if self.period < 1:
            raise ValueError("Period must be positive")
        if not 0 < self.threshold < 1:
            raise ValueError("Threshold must be between 0 and 1")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        # Implement calculation logic
        pass

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        # Implement signal logic
        pass
```

## 📚 Documentation Guidelines

### Code Documentation

- **Docstrings:** Use Google-style docstrings for all classes and functions
- **Type Hints:** Always include type hints for function signatures
- **Comments:** Explain complex business logic, not obvious code

### Project Documentation

When updating documentation in `packages/shared/docs/`:

1. **User-facing docs:** Keep language clear and example-heavy
2. **Technical specs:** Include diagrams and detailed explanations
3. **API docs:** Auto-generate when possible, supplement with examples

### Examples

When adding examples to `packages/shared/examples/`:

1. **Complete examples:** Include all necessary imports and setup
2. **Real-world scenarios:** Use realistic data and parameters
3. **Commented code:** Explain each step clearly
4. **Error handling:** Show proper error handling patterns

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── _helpers/          # Test utilities and fixtures
├── fixtures/          # Shared test data
├── backtesting/       # Backtesting engine tests
│   ├── unit/
│   ├── integration/
│   └── simulation/
├── indicators/        # Indicator tests
│   └── unit/
└── connectors/        # Exchange connector tests
```

### Writing Tests

1. **Use fixtures:** Leverage shared fixtures from `tests/conftest.py`
2. **Test isolation:** Each test should be independent
3. **Meaningful names:** Test names should describe what they test
4. **Multiple assertions:** Group related assertions in single tests

### Test Categories

- **Unit tests:** Test individual components in isolation
- **Integration tests:** Test component interactions
- **Simulation tests:** Test with realistic market scenarios
- **Regression tests:** Prevent bugs from reoccurring

Mark slow tests appropriately:

```python
@pytest.mark.slow
def test_extensive_backtesting():
    # Long-running test
    pass
```

## 🔍 Code Review Process

### Submitting Pull Requests

1. **Create feature branch:** `git checkout -b feature/phase-3.5.1-new-indicator`
2. **Make changes:** Follow development guidelines
3. **Run tests:** Ensure all tests pass
4. **Update docs:** Add/update relevant documentation
5. **Submit PR:** Include clear description and testing evidence

### PR Requirements

- [ ] All tests pass (`make test-quick`)
- [ ] Code coverage maintained or improved
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] No hardcoded secrets or credentials
- [ ] Performance implications considered

### Review Checklist

**Functionality:**

- [ ] Code works as intended
- [ ] Edge cases handled
- [ ] Error conditions managed

**Code Quality:**

- [ ] Follows project conventions
- [ ] Proper type hints
- [ ] Adequate test coverage
- [ ] Clear documentation

**Security:**

- [ ] No exposed secrets
- [ ] Input validation
- [ ] Secure API usage

## 🚨 Common Issues & Solutions

### Development Environment Issues

**Virtual Environment Problems:**

```bash
# Recreate virtual environment
cd packages/spark-app
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Dependency Conflicts:**

```bash
# Update requirements
pip freeze > requirements.txt
# Check for conflicts
pip check
```

### Testing Issues

**Import Errors:**

- Ensure you're in the correct directory: `cd packages/spark-app`
- Activate virtual environment: `source .venv/bin/activate`
- Use full Python path: `.venv/bin/python -m pytest`

**Fixture Not Found:**

- Check `tests/conftest.py` for fixture definitions
- Ensure test files are in correct directory structure

### Git Workflow Issues

**Commit Message Rejected:**

```bash
# Fix commit message format
git commit --amend -m "phase3.5.1: feat(indicators): Add new indicator"
```

**Merge Conflicts:**

```bash
# Rebase against main
git fetch origin
git rebase origin/main
# Resolve conflicts, then
git rebase --continue
```

## 🏗️ Architecture Guidelines

### Project Structure

The monorepo uses NX for workspace management:

- `packages/spark-app/`: Core trading application
- `packages/monitoring/`: Grafana dashboards and monitoring
- `packages/shared/`: Shared utilities and documentation

### Design Principles

1. **Separation of Concerns:** Clear boundaries between components
2. **Dependency Injection:** Use factory patterns for flexibility
3. **Error Handling:** Graceful degradation and comprehensive logging
4. **Performance:** Optimize for real-time trading requirements
5. **Testability:** Design components to be easily testable

### Coding Standards

**Python:**

- Follow PEP 8
- Use type hints extensively
- Prefer composition over inheritance
- Use dataclasses for simple data structures

**TypeScript:**

- Enable strict mode
- Use explicit types
- Prefer functional programming patterns
- Use modern ES6+ features

## 🎯 Getting Help

### Documentation Resources

- **User Guide:** `packages/shared/docs/userguide.md`
- **Technical Spec:** `packages/shared/docs/tech-spec.md`
- **Connector Docs:** `packages/shared/docs/connectors.md`
- **Project Roadmap:** `packages/shared/docs/roadmap.md`

### Community & Support

- **Issues:** Create GitHub issues for bugs and feature requests
- **Discussions:** Use GitHub Discussions for questions and ideas
- **Code Review:** Request reviews on pull requests

### Development Tools

- **NX Console:** VS Code extension for NX workspace management
- **Python Extensions:** Python, Pylint, mypy for VS Code
- **Git Integration:** Use VS Code Git integration or command line

## 📋 Checklist Workflow

When working on project checklists (like phase 3.5.1):

1. **Focus on one item:** Complete one checkbox at a time
2. **Update checklist:** Mark `[x]` when finished
3. **Commit changes:** Include both implementation and updated checklist
4. **Start fresh:** Begin new session for next item

This creates clear progress tracking and logical commit history.

## 🤝 Code of Conduct

- **Be respectful:** Treat all contributors with respect
- **Be constructive:** Provide helpful feedback and suggestions
- **Be collaborative:** Work together towards common goals
- **Be patient:** Help newcomers learn the codebase

## 📄 License

By contributing to Spark Stacker, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to Spark Stacker! Your efforts help make algorithmic trading more accessible and robust. 🚀
