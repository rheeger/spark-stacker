# Spark Stacker Examples

This directory contains example scripts demonstrating how to use various features of the Spark
Stacker trading system.

## Available Examples

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
# Make sure you're in the project root directory
python examples/genetic_optimization_example.py
```

or

```bash
# Make sure the file is executable (chmod +x examples/genetic_optimization_example.py)
./examples/genetic_optimization_example.py
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

## Adding New Examples

If you create a new example, please:

1. Add it to this directory
2. Make it executable with `chmod +x your_example.py`
3. Update this README with a description of your example
4. Ensure it includes comments explaining the code
