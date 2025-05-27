# Spark Stacker Examples

This directory contains example scripts demonstrating how to use various features of the Spark
Stacker trading system.

## Available Examples

### Indicator Development Tutorial

**File:** `indicator_development_tutorial.py`

A comprehensive tutorial that demonstrates how to develop custom technical indicators from scratch.
This example walks through creating a complete Bollinger Bands indicator including:

- Implementing the `BaseIndicator` interface
- Calculation logic with proper error handling
- Signal generation based on indicator values
- Parameter validation and optimization
- Testing and visualization
- Integration with the trading system

**Features:**

- Complete indicator implementation example
- Synthetic data generation for testing
- Signal analysis and visualization
- Parameter optimization testing
- Best practices for indicator development

**Usage:**

```bash
cd packages/shared/examples
python indicator_development_tutorial.py
```

This tutorial is perfect for developers who want to:

- Understand the indicator development workflow
- Learn best practices for technical indicator implementation
- See how indicators integrate with the trading system
- Practice with a complete, working example

### Backtesting Tutorial

**File:** `backtesting_tutorial.py`

Demonstrates how to use the backtesting framework to test trading strategies with custom indicators.
This tutorial covers:

- Setting up a backtesting environment
- Implementing a simple RSI-based strategy
- Running backtests with different parameters
- Calculating comprehensive performance metrics
- Visualizing backtest results
- Parameter optimization workflow

**Features:**

- Complete backtesting workflow example
- Multiple market regime simulation
- Trade execution and position management
- Performance metrics calculation
- Visualization of results
- Parameter optimization

**Usage:**

```bash
cd packages/shared/examples
python backtesting_tutorial.py
```

Perfect for understanding:

- How backtesting engines work
- Strategy development and testing
- Performance evaluation metrics
- Risk assessment techniques

### Genetic Algorithm Optimization

**File:** `genetic_optimization_example.py`

This example demonstrates how to use genetic algorithm optimization to find optimal parameters for a
trading strategy. The genetic algorithm works by:

1. Creating an initial population of random parameter combinations
2. Evaluating each parameter set by running a backtest and calculating performance metrics
3. Selecting the best performing parameter sets for reproduction
4. Creating a new generation through crossover (combining parameters from parent sets) and mutation
   (randomly changing parameters)
5. Repeating the process over multiple generations to evolve towards optimal parameters

The example uses a multi-indicator strategy combining RSI and Bollinger Bands indicators, and
demonstrates:

- Setting up the backtesting environment
- Defining the parameter space (both discrete options and continuous ranges)
- Configuring the genetic algorithm
- Running the optimization process
- Validating the results on out-of-sample data

**Usage:**

```bash
# From the project root directory
python packages/shared/examples/genetic_optimization_example.py
```

or

```bash
# Make the file executable first
chmod +x packages/shared/examples/genetic_optimization_example.py
./packages/shared/examples/genetic_optimization_example.py
```

The script will:

1. Generate sample data if it doesn't exist
2. Run the genetic optimization to find the best parameters
3. Display the performance metrics of the best parameter set
4. Save an equity curve chart
5. Run validation on a different time period to test for robustness
6. Compare key metrics between optimization and validation periods

**Key Parameters:**

- `population_size`: Number of parameter combinations in each generation
- `generations`: Number of evolutionary generations
- `crossover_rate`: Probability of parameter crossover
- `mutation_rate`: Probability of parameter mutation
- `tournament_size`: Number of candidates in tournament selection
- `metric_to_optimize`: Performance metric to maximize

## Example Categories

### 📊 **Learning Examples**

- `indicator_development_tutorial.py` - Learn indicator development
- `backtesting_tutorial.py` - Learn backtesting fundamentals

### 🧬 **Advanced Techniques**

- `genetic_optimization_example.py` - Parameter optimization with genetic algorithms

### 🚀 **Coming Soon**

- Live trading integration example
- Multi-strategy portfolio example
- Risk management techniques
- Exchange connector examples

## Running Examples

All examples are designed to be self-contained and can be run independently. They include:

- Comprehensive documentation and comments
- Error handling and logging
- Visualization capabilities
- Realistic test data generation

### Prerequisites

- Python 3.11+
- Required packages (install with `pip install -r requirements.txt` from `packages/spark-app/`)
- For visualization examples: matplotlib

### General Usage Pattern

```bash
# Navigate to examples directory
cd packages/shared/examples

# Run any example
python example_name.py

# For detailed output, enable debug logging
PYTHONPATH=../../spark-app python example_name.py
```

## Adding New Examples

If you create a new example, please:

1. **File Structure**: Place it in this directory with a descriptive name
2. **Documentation**: Add comprehensive docstrings and comments
3. **Self-Contained**: Include all necessary imports and data generation
4. **Error Handling**: Implement proper error handling and logging
5. **Visualization**: Add helpful plots and charts where appropriate
6. **README Update**: Update this README with a description of your example
7. **Testing**: Ensure it runs without external dependencies

### Example Template

```python
#!/usr/bin/env python3
"""
Brief description of what this example demonstrates.

Author: Your Name
Usage: python example_name.py
"""

import logging
# ... other imports

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main example function"""
    print("Example Title")
    print("=" * 50)

    # Your example code here

    print("Example complete!")

if __name__ == "__main__":
    main()
```

## Getting Help

- **Documentation**: Check `packages/shared/docs/` for comprehensive guides
- **Code Examples**: These examples provide practical implementations
- **Developer Guide**: See `packages/shared/docs/developer-guide.md`
- **Contributing**: See `CONTRIBUTING.md` in the project root

## License

All examples are part of the Spark Stacker project and follow the same license terms.
