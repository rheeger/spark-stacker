# Back-Testing Suite Audit & Refactor Road-Map

## Executive Summary

The back-testing and indicator–validation layer now contains more than two hundred passing tests, a
dozen data-generation scripts, multiple report generators and a growing pile of artefacts.
Everything works, but discoverability, maintenance and extension costs are high. This audit
enumerates current assets, pin-points duplication, and lays out an actionable, sequenced plan to
streamline the entire pipeline—​from synthetic-data creation to HTML reports—​with an emphasis on
_systematised onboarding of new indicators_.

---

## 1 Current Landscape (high-level inventory)

| Area              | Representative Paths                      | Purpose                                                                                      |
| ----------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------- |
| Runtime code      | `app/backtesting/**`                      | Engine, SimulationEngine, strategies, indicator helpers, HTML report generator               |
| Manual scripts    | `packages/spark-app/scripts/*.py`         | Ad-hoc demo runs, data generators, clean-up helpers                                          |
| Unit tests        | `tests/unit/**`                           | Fast, isolated tests (engine maths, dataset normaliser, optimisation logic)                  |
| Integration tests | `tests/integration/**`                    | End-to-end workflows (run_backtest, GA optimisation, WFA)                                    |
| Simulation tests  | `tests/simulation/**`                     | Feed synthetic price series into SimulationEngine                                            |
| Indicator harness | `tests/indicator_testing/test_harness.py` | Batch-tests every registered indicator across five market datasets, writes markdown + images |
| Artefacts         | `tests/test_results/**`                   | JSON, HTML, PNG and MD files committed to VCS                                                |

Detailed observations:

1. **Fixture Divergence** – at least six code snippets create "temp CSV + BacktestEngine".
2. **Synthetic Data Generators** – five distinct algorithms exist, with different statistical
   properties.
3. **Scripts vs Tests** – demo scripts replicate integration test logic but live outside pytest,
   risking drift.
4. **Artefacts in Git** – test output directories are checked-in; they pollute diffs and bloat clone
   size.
5. **Folder Nomenclature** – `simulation/` vs `indicator_testing/` vs `integration/` leads to grep
   fatigue.
6. **Reporting Layer Untested** – HTML/Plotly generation is only exercised manually.
7. **CI Duration Risks** – GA optimisation stress tests take minutes; no marker to exclude on quick
   runs.
8. **Indicator On-Ramp** – writing a new indicator requires edits in 4+ places: class, factory, test
   harness, demo script.

---

## 2 Coverage Matrix (today)

| Module                     | Unit | Integration | Simulation | Automated HTML Report | Notes                 |
| -------------------------- | ---- | ----------- | ---------- | --------------------- | --------------------- |
| BacktestEngine             | ✅   | ✅          | –          | –                     | Core logic solid      |
| SimulationEngine           | –    | –           | ✅         | –                     | Only RSI sim test     |
| Indicator Backtest Manager | –    | ✅          | –          | –                     | Via harness           |
| Strategies (MA-X, MACD, …) | –    | ✅          | ✅         | –                     | Tested indirectly     |
| Genetic/Grid Optimisers    | ✅   | ✅          | –          | –                     | Stress tests slow     |
| Report Generator           | –    | –           | –          | ❌                    | Manual scripts only   |
| CLI / Demo scripts         | ❌   | ❌          | ❌         | ❌                    | No automated coverage |

---

## 3 Pain-Points & Root Causes

### 3.1 Duplication

- _Six_ nearly identical fixtures copy-pasted across tests.
- Multiple data generators mean indicator performance numbers are not comparable.

### 3.2 Undefined Ownership of Artefacts

- `tests/test_results/**` is committed, causing merge conflicts and large diffs.
- No canonical "results" vs "examples" split.

### 3.3 Unclear Test Taxonomy

- Some files under `integration/` are actually unit-style.
- `simulation/` folder overlaps with integration tests that also use SimulationEngine.

### 3.4 Indicator Onboarding Friction

Process today:

1. Create indicator class
2. Register in factory
3. Update YAML config
4. Manually edit harness list or script
5. Decide where tests live
6. Possibly craft a demo HTML script

Developers lose time figuring out where each piece goes.

### 3.5 Reporting Blind Spot

HTML generator may break silently—​no CI guard.

---

## 4 Prescriptive Refactor Plan (step-by-step, with checkpoints)

The overarching goal is **clarity**, **repeatability**, and **speed**. The refactor is split into
eight logical phases. Each phase is expressed as bite-sized _checkpoints_ that map one-to-one with
commits / checklist items. You should never tackle more than one checkpoint per PR.

### 4.0 Repository Hygiene & Baseline Clean-up

_Purpose: create a clean slate so later diffs are readable._

| Checkpoint | Action                                                                                                                 | Acceptance Criteria                                               |
| ---------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 0-A        | Add `tests/test_results/` and `*.png` to `.gitignore`                                                                  | `git status` shows zero changes after running the harness locally |
| 0-B        | Remove all committed artefacts (`git rm -r tests/test_results && git commit`)                                          | Repo size shrinks, `pytest` still passes                          |
| 0-C        | Add `make clean-results` that deletes any stray artefacts in **developer** machines                                    | `make clean-results` removes the folder and exits with code 0     |
| 0-D        | Generate a **baseline branch coverage report** (pytest-cov) and store it in `htmlcov/` (ignored) for future comparison | `coverage html` page renders locally                              |
| 0-E        | **Bootstrap CI:** add `.github/workflows/ci-quick.yml` running `pytest -m \"not slow\" --cov=app` on Python 3.11       | Workflow completes successfully; badge renders in README          |

---

### 4.1 Foundational Fixtures

_Purpose: eliminate duplicated bootstrap code._

| Checkpoint | Action                                                                                                                                                               | Acceptance Criteria                                                      |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| 1-A        | Create `tests/conftest.py` containing the four shared fixtures (`price_dataframe`, `temp_csv_dir`, `backtest_env`, `results_dir`) using a deterministic random seed. | Fixtures importable from any test; `pytest -q` still passes.             |
| 1-B        | Add typing stubs `tests/conftest.pyi` so IDE autocompletion works.                                                                                                   | MyPy (if enabled) shows no missing-import errors for the fixtures.       |
| 1-C        | Refactor `tests/unit/test_backtest_engine.py` to consume `backtest_env`, deleting its temporary-directory boilerplate.                                               | File diff shows >90 % reduction in local fixture code; test still green. |
| 1-D        | Run `pytest -q`; ensure all tests pass and `git grep tempfile.TemporaryDirectory tests/` count decreases.                                                            | Command exits 0 and count drops by at least one.                         |

---

### 4.2 Single Synthetic-Data Generator

_Purpose: guarantee comparable indicator metrics across tests._

| Checkpoint | Action                                                                                                                                                        | Acceptance Criteria                                                           |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| 2-A        | Implement `tests/_helpers/data_factory.py::make_price_dataframe(rows:int, pattern:str, noise:float, seed:int)` supporting `trend`, `mean_revert`, `sideways`. | Factory returns a `pd.DataFrame` with 6 OHLCV columns and given row count.    |
| 2-B        | Wire the `price_dataframe` fixture to call the factory (`pattern="trend"`).                                                                                   | All tests depending on the fixture still pass.                                |
| 2-C        | Add unit test `tests/backtesting/unit/test_data_factory.py` asserting deterministic output for a fixed seed.                                                  | Running the test twice with same seed yields identical hash of the DataFrame. |

---

### 4.3 Directory Realignment

_Purpose: make the test tree self-describing._

| Checkpoint | Action                                                                                                              | Acceptance Criteria                                                             |
| ---------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 3-A        | Create target directories `tests/backtesting/{unit,integration,simulation,regression}` and `tests/indicators/unit`. | Folders exist in repo.                                                          |
| 3-B        | Move `tests/unit/test_backtest_engine.py` to `tests/backtesting/unit/` and fix imports if necessary.                | CI green.                                                                       |
| 3-C        | Add `__init__.py` files so IDEs treat directories as packages.                                                      | `python -m pytest -q` passes, `import tests.backtesting.unit` succeeds in REPL. |
| 3-D        | Update `pytest.ini`: set `testpaths = tests` and declare `slow` marker.                                             | `pytest -q` discovers same test count as before.                                |
| 3-E        | After CI green, migrate remaining files folder-by-folder, one PR per folder to keep diffs small.                    | No import-errors and path structure finalised.                                  |

---

### 4.4 Artefact Stewardship

_Purpose: keep blobs out of git while preserving local visibility of results._

| Checkpoint | Action                                                                                                                                             | Acceptance Criteria                                                   |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 4-A        | Extend `IndicatorBacktestManager` and integration tests to accept an `output_path` parameter (defaults to tmp fixture) and return generated paths. | Paths are returned and used in assertions without touching repo tree. |
| 4-B        | Refactor `tests/indicator_testing/test_harness.py` to consume `results_dir` fixture; delete hard-coded `tests/test_results` constants.             | Harness writes only into the tmp path.                                |
| 4-C        | Add README section "Viewing local artefacts" describing how to open the tmp directory after a run.                                                 | Documentation merged; team acknowledges.                              |

---

### 4.5 Script ➜ CLI Consolidation

_Purpose: unify entry-points and prevent script drift._

| Checkpoint | Action                                                                                                       | Acceptance Criteria                                                |
| ---------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| 5-A        | Create `app/cli.py` using `click`; first command `backtest` mirrors `BacktestEngine.run_backtest` arguments. | `python -m app.cli backtest --help` prints usage.                  |
| 5-B        | Add integration smoke test invoking the CLI via `subprocess.run` and completing < 10 s.                      | Test passes on CI with `not slow` marker.                          |
| 5-C        | Replace `run_eth_macd_backtest.py` by a `cli demo-macd` sub-command housed in `examples/`.                   | Functionality parity confirmed; script deleted.                    |
| 5-D        | Remove any other orphan scripts; update docs to point to CLI.                                                | `scripts/` directory contains only maintained helpers or is empty. |

---

### 4.6 Automated Report Verification

_Purpose: break the build when the HTML template breaks._

| Checkpoint | Action                                                                                             | Acceptance Criteria                                           |
| ---------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 6-A        | Use `backtest_env` to run a one-trade backtest and capture a `BacktestResult`.                     | Result object non-empty.                                      |
| 6-B        | Generate report with `generate_indicator_report`, saving to `results_dir`.                         | HTML file exists.                                             |
| 6-C        | Parse HTML via `BeautifulSoup`; assert presence of `total_trades`, `max_drawdown`, `sharpe_ratio`. | Assertions pass.                                              |
| 6-D        | Verify linked chart files referenced in HTML exist on disk.                                        | Every `<img>`/iframe src resolves to a file in `results_dir`. |

---

### 4.7 Indicator Onboarding Template

_Purpose: scaffold everything needed for a new indicator with one command._

| Checkpoint | Action                                                                                                      | Acceptance Criteria                                                     |
| ---------- | ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 7-A        | Implement `scripts/new_indicator.sh <Name>` that creates: indicator class, unit-test skeleton, YAML config. | Running script with `FooBar` produces three new files in correct paths. |
| 7-B        | Script auto-inserts import in `IndicatorFactory.register_defaults()`.                                       | Factory import section contains new line; `python -m pytest -q` passes. |
| 7-C        | Update CONTRIBUTING.md with onboarding steps.                                                               | Markdown renders with new instructions.                                 |

---

### 4.8 CI Strategy

_Purpose: build upon the quick workflow introduced in **0-E** and expand CI for full coverage._

| Checkpoint | Action                                                                                        | Acceptance Criteria                    |
| ---------- | --------------------------------------------------------------------------------------------- | -------------------------------------- |
| 8-A        | Modify GitHub Actions quick workflow to run `pytest -m "not slow" --cov=app` on Python 3.11.  | Workflow passes in < 3 minutes.        |
| 8-B        | Add scheduled `ci-nightly.yml` that runs `pytest -m slow` and uploads `htmlcov/` as artefact. | Artefact downloadable in workflow run. |
| 8-C        | Enable branch protection requiring the quick workflow on `main`.                              | GitHub shows required check status.    |

---

## 5 Long-Term Benefits Anticipated

- **Consistency** – Every test uses identical data generation, making performance metrics
  comparable.
- **Speed** – Shared fixtures eliminate redundant CSV writes; "quick" test target remains under
  seconds.
- **Discoverability** – Single test tree; developers can grep "backtest" and see all relevant files.
- **Scalability** – New indicators gain tests, configs, docs, and CLI hooks in minutes.
- **Reliability** – Report generator covered by CI; failures surface immediately.
- **Clean Git History** – No more binary artefacts in diffs; smaller repo size.

### Scope & Terminology: Runtime vs Test-Suite

Throughout this document the **runtime back-testing engine** (`app/backtesting/**`) and the
**test-suite infrastructure** (`tests/**`) are treated as _distinct_ layers:

- Runtime code is what ships to production and must never import from the test tree.
- Test code is consumer-side scaffolding (fixtures, data factories, harnesses, CLI smoke tests)
  whose only job is to validate the engine and related utilities.

Many of the pain-points called out here sit at the _interface_ between the two layers (think
duplicated fixtures that stand up `BacktestEngine`). Consequently, the roadmap bundles fixes into a
single narrative even though each checkpoint will state whether it touches runtime code or test
infrastructure.

> **Separation of Concerns** Checkpoints are listed together for readability, but each action should
> affect **only one layer at a time**. When a checkpoint modifies production code (e.g. 5-A adding
> `app/cli.py`) this is called out explicitly. The majority of 4.x checkpoints operate purely within
> the `tests/**` tree. PRs must avoid introducing bidirectional imports between `app/backtesting`
> and `tests/`.
