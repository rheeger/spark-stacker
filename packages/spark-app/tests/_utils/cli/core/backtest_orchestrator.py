"""
Backtest Orchestrator

This module provides centralized orchestration for backtesting operations:
- Coordinate overall backtesting workflow
- Handle resource allocation and cleanup
- Manage parallel execution of multiple backtests
- Add progress tracking and user updates
- Handle interruption and graceful shutdown
- Coordinate between different manager types (strategy, indicator, scenario, comparison)
"""

import asyncio
import logging
import signal
import threading
import time
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import psutil
from tqdm import tqdm

from .config_manager import ConfigManager
from .data_manager import DataManager

logger = logging.getLogger(__name__)


class BacktestType(Enum):
    """Types of backtests that can be orchestrated."""
    SINGLE_STRATEGY = "single_strategy"
    MULTI_STRATEGY = "multi_strategy"
    SINGLE_INDICATOR = "single_indicator"
    MULTI_INDICATOR = "multi_indicator"
    SCENARIO_TESTING = "scenario_testing"
    COMPARISON = "comparison"
    BATCH_TESTING = "batch_testing"


class ExecutionMode(Enum):
    """Execution modes for backtest orchestration."""
    SEQUENTIAL = "sequential"
    PARALLEL_THREADS = "parallel_threads"
    PARALLEL_PROCESSES = "parallel_processes"
    ADAPTIVE = "adaptive"  # Automatically choose based on workload


class ResourceLimitType(Enum):
    """Types of resource limits that can be enforced."""
    MAX_MEMORY_MB = "max_memory_mb"
    MAX_CPU_PERCENT = "max_cpu_percent"
    MAX_CONCURRENT_JOBS = "max_concurrent_jobs"
    MAX_EXECUTION_TIME_SECONDS = "max_execution_time_seconds"


@dataclass
class BacktestJob:
    """Represents a single backtest job."""
    job_id: str
    job_type: BacktestType
    parameters: Dict[str, Any]
    priority: int = 50  # 0=lowest, 100=highest
    estimated_duration_seconds: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)  # Job IDs this depends on
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    """Result of a backtest job execution."""
    job_id: str
    success: bool
    result_data: Optional[Any] = None
    error: Optional[Exception] = None
    execution_time_seconds: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationState:
    """Current state of the orchestration process."""
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    running_jobs: int = 0
    queued_jobs: int = 0
    start_time: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None
    current_resource_usage: Dict[str, Any] = field(default_factory=dict)


class BacktestOrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    pass


class ResourceLimitExceededError(BacktestOrchestratorError):
    """Raised when resource limits are exceeded."""
    pass


class JobExecutionError(BacktestOrchestratorError):
    """Raised when job execution fails."""
    pass


class BacktestOrchestrator:
    """
    Central orchestrator for backtesting operations.

    Features:
    - Coordinate multiple types of backtests (strategy, indicator, scenario, comparison)
    - Manage parallel execution with resource limits and monitoring
    - Provide real-time progress tracking and user updates
    - Handle graceful interruption and cleanup
    - Resource allocation and cleanup management
    - Job dependency management and execution scheduling
    """

    def __init__(self,
                 config_manager: Optional[ConfigManager] = None,
                 data_manager: Optional[DataManager] = None,
                 max_workers: Optional[int] = None,
                 default_execution_mode: ExecutionMode = ExecutionMode.ADAPTIVE,
                 resource_limits: Optional[Dict[ResourceLimitType, Any]] = None,
                 progress_callback: Optional[Callable[[OrchestrationState], None]] = None):
        """
        Initialize the backtest orchestrator.

        Args:
            config_manager: Configuration manager instance
            data_manager: Data manager instance
            max_workers: Maximum number of concurrent workers
            default_execution_mode: Default execution mode for jobs
            resource_limits: Resource limits for job execution
            progress_callback: Optional callback for progress updates
        """
        self.config_manager = config_manager or ConfigManager()
        self.data_manager = data_manager or DataManager()

        # Worker configuration
        if max_workers is None:
            self.max_workers = min(psutil.cpu_count(), 8)  # Reasonable default
        else:
            self.max_workers = max_workers

        self.default_execution_mode = default_execution_mode
        self.progress_callback = progress_callback

        # Resource limits
        self.resource_limits = resource_limits or {
            ResourceLimitType.MAX_MEMORY_MB: 8192,  # 8GB default
            ResourceLimitType.MAX_CPU_PERCENT: 80,
            ResourceLimitType.MAX_CONCURRENT_JOBS: self.max_workers,
            ResourceLimitType.MAX_EXECUTION_TIME_SECONDS: 3600  # 1 hour
        }

        # Internal state
        self.state = OrchestrationState()
        self.job_queue: List[BacktestJob] = []
        self.running_jobs: Dict[str, BacktestJob] = {}
        self.completed_jobs: Dict[str, JobResult] = {}
        self.failed_jobs: Dict[str, JobResult] = {}

        # Execution control
        self._shutdown_requested = False
        self._cleanup_functions: List[Callable[[], None]] = []
        self._resource_monitor_thread: Optional[threading.Thread] = None
        self._monitor_lock = threading.Lock()

        # Manager instances (lazy-loaded)
        self._manager_cache: Dict[str, Any] = {}

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        logger.info(f"BacktestOrchestrator initialized with {self.max_workers} max workers")

    def add_job(self, job: BacktestJob) -> str:
        """
        Add a backtest job to the orchestration queue.

        Args:
            job: BacktestJob to add to the queue

        Returns:
            Job ID for tracking
        """
        # Validate job parameters
        self._validate_job(job)

        # Add to queue
        self.job_queue.append(job)
        self.state.total_jobs += 1
        self.state.queued_jobs += 1

        logger.info(f"Added job {job.job_id} of type {job.job_type} to queue")
        logger.debug(f"Job parameters: {job.parameters}")

        # Update progress
        self._update_progress()

        return job.job_id

    def add_strategy_backtest(self,
                            strategy_name: str,
                            days: int = 30,
                            scenarios: Optional[List[str]] = None,
                            **kwargs) -> str:
        """
        Add a strategy backtesting job.

        Args:
            strategy_name: Name of strategy from configuration
            days: Number of days to backtest
            scenarios: Optional list of scenarios to test
            **kwargs: Additional parameters for the backtest

        Returns:
            Job ID for tracking
        """
        job_id = f"strategy_{strategy_name}_{int(time.time())}"

        parameters = {
            'strategy_name': strategy_name,
            'days': days,
            'scenarios': scenarios or ['real'],
            **kwargs
        }

        job = BacktestJob(
            job_id=job_id,
            job_type=BacktestType.SINGLE_STRATEGY,
            parameters=parameters,
            priority=75,  # Higher priority for strategy tests
            estimated_duration_seconds=self._estimate_strategy_duration(strategy_name, days, scenarios)
        )

        return self.add_job(job)

    def add_scenario_testing(self,
                           strategy_name: str,
                           scenarios: List[str],
                           days: int = 30,
                           **kwargs) -> str:
        """
        Add a multi-scenario testing job.

        Args:
            strategy_name: Name of strategy to test
            scenarios: List of scenarios to test against
            days: Number of days per scenario
            **kwargs: Additional parameters

        Returns:
            Job ID for tracking
        """
        job_id = f"scenario_{strategy_name}_{int(time.time())}"

        parameters = {
            'strategy_name': strategy_name,
            'scenarios': scenarios,
            'days': days,
            **kwargs
        }

        job = BacktestJob(
            job_id=job_id,
            job_type=BacktestType.SCENARIO_TESTING,
            parameters=parameters,
            priority=60,
            estimated_duration_seconds=self._estimate_scenario_duration(scenarios, days)
        )

        return self.add_job(job)

    def add_strategy_comparison(self,
                              strategy_names: List[str],
                              days: int = 30,
                              scenarios: Optional[List[str]] = None,
                              **kwargs) -> str:
        """
        Add a strategy comparison job.

        Args:
            strategy_names: List of strategy names to compare
            days: Number of days to backtest each strategy
            scenarios: Optional list of scenarios for comparison
            **kwargs: Additional parameters

        Returns:
            Job ID for tracking
        """
        job_id = f"comparison_{len(strategy_names)}strategies_{int(time.time())}"

        parameters = {
            'strategy_names': strategy_names,
            'days': days,
            'scenarios': scenarios or ['real'],
            **kwargs
        }

        # Create dependency jobs for individual strategy testing
        dependency_jobs = []
        for strategy_name in strategy_names:
            dep_job_id = self.add_strategy_backtest(
                strategy_name=strategy_name,
                days=days,
                scenarios=scenarios,
                **kwargs
            )
            dependency_jobs.append(dep_job_id)

        job = BacktestJob(
            job_id=job_id,
            job_type=BacktestType.COMPARISON,
            parameters=parameters,
            dependencies=dependency_jobs,
            priority=40,  # Lower priority, runs after dependencies
            estimated_duration_seconds=len(strategy_names) * 60  # Comparison is relatively fast
        )

        return self.add_job(job)

    def execute_all(self,
                   execution_mode: Optional[ExecutionMode] = None,
                   timeout_seconds: Optional[float] = None) -> Dict[str, JobResult]:
        """
        Execute all queued jobs with the specified execution mode.

        Args:
            execution_mode: How to execute jobs (sequential, parallel, etc.)
            timeout_seconds: Optional timeout for the entire execution

        Returns:
            Dictionary mapping job_id to JobResult
        """
        if not self.job_queue:
            logger.warning("No jobs in queue to execute")
            return {}

        execution_mode = execution_mode or self.default_execution_mode

        logger.info(f"Starting execution of {len(self.job_queue)} jobs using {execution_mode}")
        self.state.start_time = datetime.now()

        # Start resource monitoring
        self._start_resource_monitoring()

        try:
            # Choose execution strategy based on mode
            if execution_mode == ExecutionMode.SEQUENTIAL:
                results = self._execute_sequential()
            elif execution_mode == ExecutionMode.PARALLEL_THREADS:
                results = self._execute_parallel_threads()
            elif execution_mode == ExecutionMode.PARALLEL_PROCESSES:
                results = self._execute_parallel_processes()
            elif execution_mode == ExecutionMode.ADAPTIVE:
                results = self._execute_adaptive()
            else:
                raise ValueError(f"Unsupported execution mode: {execution_mode}")

            # Wait for all jobs to complete or timeout
            if timeout_seconds:
                self._wait_for_completion(timeout_seconds)

            logger.info(f"Execution completed. Success: {len(self.completed_jobs)}, "
                       f"Failed: {len(self.failed_jobs)}")

            return {**self.completed_jobs, **self.failed_jobs}

        except KeyboardInterrupt:
            logger.info("Execution interrupted by user")
            self.shutdown(graceful=True)
            raise

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            self.shutdown(graceful=False)
            raise

        finally:
            # Stop resource monitoring
            self._stop_resource_monitoring()

            # Run cleanup functions
            self._run_cleanup()

    def shutdown(self, graceful: bool = True) -> None:
        """
        Shutdown the orchestrator and cleanup resources.

        Args:
            graceful: Whether to wait for running jobs to complete
        """
        logger.info(f"Initiating {'graceful' if graceful else 'immediate'} shutdown")
        self._shutdown_requested = True

        if graceful:
            # Wait for running jobs to complete
            timeout = 30  # seconds
            start_time = time.time()

            while self.running_jobs and (time.time() - start_time) < timeout:
                logger.info(f"Waiting for {len(self.running_jobs)} running jobs to complete...")
                time.sleep(1)

            if self.running_jobs:
                logger.warning(f"Timeout reached, {len(self.running_jobs)} jobs still running")

        # Run cleanup
        self._run_cleanup()

        logger.info("Orchestrator shutdown complete")

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a specific job.

        Args:
            job_id: ID of the job to check

        Returns:
            Dictionary with job status information, or None if not found
        """
        # Check completed jobs
        if job_id in self.completed_jobs:
            result = self.completed_jobs[job_id]
            return {
                'status': 'completed',
                'success': result.success,
                'execution_time': result.execution_time_seconds,
                'error': str(result.error) if result.error else None
            }

        # Check failed jobs
        if job_id in self.failed_jobs:
            result = self.failed_jobs[job_id]
            return {
                'status': 'failed',
                'success': False,
                'execution_time': result.execution_time_seconds,
                'error': str(result.error) if result.error else None
            }

        # Check running jobs
        if job_id in self.running_jobs:
            return {
                'status': 'running',
                'started_at': datetime.now().isoformat()  # Approximate
            }

        # Check queued jobs
        for job in self.job_queue:
            if job.job_id == job_id:
                return {
                    'status': 'queued',
                    'job_type': job.job_type.value,
                    'priority': job.priority,
                    'dependencies': job.dependencies
                }

        return None

    def get_overall_progress(self) -> Dict[str, Any]:
        """
        Get overall progress information.

        Returns:
            Dictionary with progress information
        """
        total_time = None
        estimated_remaining = None

        if self.state.start_time:
            elapsed = (datetime.now() - self.state.start_time).total_seconds()

            if self.state.completed_jobs > 0:
                avg_time_per_job = elapsed / self.state.completed_jobs
                remaining_jobs = self.state.total_jobs - self.state.completed_jobs - self.state.failed_jobs
                estimated_remaining = remaining_jobs * avg_time_per_job

        return {
            'total_jobs': self.state.total_jobs,
            'completed_jobs': self.state.completed_jobs,
            'failed_jobs': self.state.failed_jobs,
            'running_jobs': self.state.running_jobs,
            'queued_jobs': self.state.queued_jobs,
            'success_rate': (self.state.completed_jobs / max(1, self.state.completed_jobs + self.state.failed_jobs)) * 100,
            'elapsed_time_seconds': elapsed if self.state.start_time else 0,
            'estimated_remaining_seconds': estimated_remaining,
            'current_resource_usage': self.state.current_resource_usage
        }

    def add_cleanup_function(self, cleanup_func: Callable[[], None]) -> None:
        """
        Add a function to be called during cleanup.

        Args:
            cleanup_func: Function to call during cleanup
        """
        self._cleanup_functions.append(cleanup_func)

    # Private methods for internal functionality

    def _validate_job(self, job: BacktestJob) -> None:
        """Validate job parameters and configuration."""
        if not job.job_id:
            raise ValueError("Job ID cannot be empty")

        if job.job_id in [j.job_id for j in self.job_queue]:
            raise ValueError(f"Job ID {job.job_id} already exists in queue")

        if not job.parameters:
            raise ValueError("Job parameters cannot be empty")

        # Type-specific validation
        if job.job_type == BacktestType.SINGLE_STRATEGY:
            if 'strategy_name' not in job.parameters:
                raise ValueError("Strategy name required for strategy backtest")

            # Validate strategy exists in configuration
            try:
                strategy_config = self.config_manager.get_strategy_config(job.parameters['strategy_name'])
                if not strategy_config:
                    raise ValueError(f"Strategy '{job.parameters['strategy_name']}' not found in configuration")
            except Exception as e:
                raise ValueError(f"Failed to validate strategy configuration: {e}")

    def _estimate_strategy_duration(self, strategy_name: str, days: int, scenarios: Optional[List[str]]) -> float:
        """Estimate duration for a strategy backtest."""
        base_time = 60  # 1 minute base time

        # Adjust for days (more data = more time)
        day_factor = max(1, days / 30)  # Scale based on 30-day baseline

        # Adjust for scenarios
        scenario_count = len(scenarios) if scenarios else 1
        scenario_factor = scenario_count * 0.8  # Each scenario adds 80% of base time

        # Adjust for strategy complexity (number of indicators)
        complexity_factor = 1.0
        try:
            strategy_config = self.config_manager.get_strategy_config(strategy_name)
            if strategy_config:
                indicator_count = len(strategy_config.get('indicators', []))
                complexity_factor = 1 + (indicator_count * 0.2)  # Each indicator adds 20%
        except Exception:
            pass  # Use default if strategy config can't be loaded

        return base_time * day_factor * scenario_factor * complexity_factor

    def _estimate_scenario_duration(self, scenarios: List[str], days: int) -> float:
        """Estimate duration for scenario testing."""
        base_time_per_scenario = 45  # 45 seconds per scenario
        day_factor = max(1, days / 30)

        return len(scenarios) * base_time_per_scenario * day_factor

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown(graceful=True)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _start_resource_monitoring(self) -> None:
        """Start monitoring system resources."""
        def monitor_resources():
            while not self._shutdown_requested:
                try:
                    # Get current resource usage
                    memory_mb = psutil.virtual_memory().used / (1024 * 1024)
                    cpu_percent = psutil.cpu_percent(interval=1)

                    with self._monitor_lock:
                        self.state.current_resource_usage = {
                            'memory_mb': memory_mb,
                            'cpu_percent': cpu_percent,
                            'concurrent_jobs': len(self.running_jobs)
                        }

                    # Check resource limits
                    self._check_resource_limits()

                    time.sleep(5)  # Monitor every 5 seconds

                except Exception as e:
                    logger.warning(f"Resource monitoring error: {e}")
                    time.sleep(10)  # Back off on error

        self._resource_monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        self._resource_monitor_thread.start()

    def _stop_resource_monitoring(self) -> None:
        """Stop resource monitoring."""
        if self._resource_monitor_thread and self._resource_monitor_thread.is_alive():
            # Thread will stop when _shutdown_requested is True
            self._resource_monitor_thread.join(timeout=10)

    def _check_resource_limits(self) -> None:
        """Check if resource limits are exceeded."""
        current_usage = self.state.current_resource_usage

        # Check memory limit
        memory_limit = self.resource_limits.get(ResourceLimitType.MAX_MEMORY_MB)
        if memory_limit and current_usage.get('memory_mb', 0) > memory_limit:
            logger.warning(f"Memory usage ({current_usage['memory_mb']:.1f} MB) exceeds limit ({memory_limit} MB)")

        # Check CPU limit
        cpu_limit = self.resource_limits.get(ResourceLimitType.MAX_CPU_PERCENT)
        if cpu_limit and current_usage.get('cpu_percent', 0) > cpu_limit:
            logger.warning(f"CPU usage ({current_usage['cpu_percent']:.1f}%) exceeds limit ({cpu_limit}%)")

        # Check concurrent jobs limit
        jobs_limit = self.resource_limits.get(ResourceLimitType.MAX_CONCURRENT_JOBS)
        if jobs_limit and current_usage.get('concurrent_jobs', 0) > jobs_limit:
            logger.warning(f"Concurrent jobs ({current_usage['concurrent_jobs']}) exceeds limit ({jobs_limit})")

    def _execute_sequential(self) -> Dict[str, JobResult]:
        """Execute jobs sequentially."""
        results = {}

        # Sort jobs by priority and dependencies
        sorted_jobs = self._sort_jobs_for_execution()

        with tqdm(total=len(sorted_jobs), desc="Executing jobs sequentially") as pbar:
            for job in sorted_jobs:
                if self._shutdown_requested:
                    break

                # Check dependencies
                if not self._dependencies_satisfied(job):
                    logger.warning(f"Skipping job {job.job_id} - dependencies not satisfied")
                    continue

                # Execute job
                result = self._execute_single_job(job)
                results[job.job_id] = result

                # Update progress
                self._update_job_completion(job, result)
                pbar.update(1)

        return results

    def _execute_parallel_threads(self) -> Dict[str, JobResult]:
        """Execute jobs using thread pool."""
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs that can run immediately
            future_to_job = {}
            ready_jobs = [job for job in self.job_queue if self._dependencies_satisfied(job)]

            for job in ready_jobs:
                if self._shutdown_requested:
                    break
                future = executor.submit(self._execute_single_job, job)
                future_to_job[future] = job
                self.running_jobs[job.job_id] = job

            # Process completed jobs and submit new ones
            with tqdm(total=len(self.job_queue), desc="Executing jobs in parallel") as pbar:
                for future in as_completed(future_to_job):
                    if self._shutdown_requested:
                        break

                    job = future_to_job[future]
                    result = future.result()
                    results[job.job_id] = result

                    # Update state
                    self._update_job_completion(job, result)
                    pbar.update(1)

                    # Check for newly ready jobs
                    newly_ready = [j for j in self.job_queue
                                 if j.job_id not in results
                                 and j.job_id not in self.running_jobs
                                 and self._dependencies_satisfied(j)]

                    # Submit newly ready jobs
                    for ready_job in newly_ready:
                        if len(self.running_jobs) < self.max_workers:
                            future = executor.submit(self._execute_single_job, ready_job)
                            future_to_job[future] = ready_job
                            self.running_jobs[ready_job.job_id] = ready_job

        return results

    def _execute_parallel_processes(self) -> Dict[str, JobResult]:
        """Execute jobs using process pool (for CPU-intensive tasks)."""
        # Note: Process pool execution would require careful serialization
        # For now, fall back to thread execution
        logger.info("Process pool execution not fully implemented, falling back to thread execution")
        return self._execute_parallel_threads()

    def _execute_adaptive(self) -> Dict[str, JobResult]:
        """Adaptively choose execution mode based on workload."""
        total_jobs = len(self.job_queue)

        # Simple heuristic for choosing execution mode
        if total_jobs == 1:
            return self._execute_sequential()
        elif total_jobs <= 4:
            return self._execute_parallel_threads()
        else:
            # For larger workloads, use threading with dynamic adjustment
            return self._execute_parallel_threads()

    def _execute_single_job(self, job: BacktestJob) -> JobResult:
        """Execute a single backtest job."""
        start_time = time.time()

        try:
            logger.info(f"Starting execution of job {job.job_id} ({job.job_type})")

            # Get appropriate manager for job type
            manager = self._get_manager_for_job(job)

            # Execute based on job type
            if job.job_type == BacktestType.SINGLE_STRATEGY:
                result_data = self._execute_strategy_job(job, manager)
            elif job.job_type == BacktestType.SCENARIO_TESTING:
                result_data = self._execute_scenario_job(job, manager)
            elif job.job_type == BacktestType.COMPARISON:
                result_data = self._execute_comparison_job(job, manager)
            else:
                raise ValueError(f"Unsupported job type: {job.job_type}")

            execution_time = time.time() - start_time

            result = JobResult(
                job_id=job.job_id,
                success=True,
                result_data=result_data,
                execution_time_seconds=execution_time
            )

            logger.info(f"Job {job.job_id} completed successfully in {execution_time:.1f}s")
            return result

        except Exception as e:
            execution_time = time.time() - start_time

            result = JobResult(
                job_id=job.job_id,
                success=False,
                error=e,
                execution_time_seconds=execution_time
            )

            logger.error(f"Job {job.job_id} failed after {execution_time:.1f}s: {e}")
            return result

    def _get_manager_for_job(self, job: BacktestJob) -> Any:
        """Get the appropriate manager instance for the job type."""
        # For now, return a placeholder since managers aren't fully implemented
        # This will be updated when the actual manager classes are created

        manager_key = f"{job.job_type.value}_manager"

        if manager_key not in self._manager_cache:
            # Create manager instances as needed
            # TODO: Import and instantiate actual manager classes
            logger.debug(f"Creating placeholder manager for {job.job_type}")
            self._manager_cache[manager_key] = {
                'type': job.job_type,
                'config_manager': self.config_manager,
                'data_manager': self.data_manager
            }

        return self._manager_cache[manager_key]

    def _execute_strategy_job(self, job: BacktestJob, manager: Any) -> Dict[str, Any]:
        """Execute a strategy backtest job."""
        # Placeholder implementation
        # TODO: Use actual StrategyBacktestManager when implemented

        strategy_name = job.parameters['strategy_name']
        days = job.parameters.get('days', 30)
        scenarios = job.parameters.get('scenarios', ['real'])

        # Simulate strategy execution
        logger.info(f"Executing strategy '{strategy_name}' for {days} days across {len(scenarios)} scenarios")

        # This would use the actual strategy manager to run the backtest
        return {
            'strategy_name': strategy_name,
            'days': days,
            'scenarios': scenarios,
            'status': 'completed',
            'placeholder': True
        }

    def _execute_scenario_job(self, job: BacktestJob, manager: Any) -> Dict[str, Any]:
        """Execute a scenario testing job."""
        # Placeholder implementation
        # TODO: Use actual ScenarioBacktestManager when implemented

        strategy_name = job.parameters['strategy_name']
        scenarios = job.parameters['scenarios']
        days = job.parameters.get('days', 30)

        logger.info(f"Executing scenario testing for '{strategy_name}' across {len(scenarios)} scenarios")

        return {
            'strategy_name': strategy_name,
            'scenarios': scenarios,
            'days': days,
            'status': 'completed',
            'placeholder': True
        }

    def _execute_comparison_job(self, job: BacktestJob, manager: Any) -> Dict[str, Any]:
        """Execute a comparison job."""
        # Placeholder implementation
        # TODO: Use actual ComparisonManager when implemented

        strategy_names = job.parameters['strategy_names']
        days = job.parameters.get('days', 30)

        logger.info(f"Executing comparison of {len(strategy_names)} strategies")

        return {
            'strategy_names': strategy_names,
            'days': days,
            'status': 'completed',
            'placeholder': True
        }

    def _sort_jobs_for_execution(self) -> List[BacktestJob]:
        """Sort jobs by priority and dependencies."""
        # Simple topological sort by dependencies and priority
        sorted_jobs = []
        remaining_jobs = self.job_queue.copy()

        while remaining_jobs:
            # Find jobs with no unsatisfied dependencies
            ready_jobs = [job for job in remaining_jobs if self._dependencies_satisfied(job)]

            if not ready_jobs:
                # No jobs ready - either circular dependency or missing dependency
                logger.warning("No jobs ready for execution - possible dependency issue")
                break

            # Sort ready jobs by priority (highest first)
            ready_jobs.sort(key=lambda x: x.priority, reverse=True)

            # Take the highest priority job
            next_job = ready_jobs[0]
            sorted_jobs.append(next_job)
            remaining_jobs.remove(next_job)

        return sorted_jobs

    def _dependencies_satisfied(self, job: BacktestJob) -> bool:
        """Check if all dependencies for a job are satisfied."""
        for dep_id in job.dependencies:
            if dep_id not in self.completed_jobs:
                return False
        return True

    def _update_job_completion(self, job: BacktestJob, result: JobResult) -> None:
        """Update state when a job completes."""
        # Remove from running jobs
        if job.job_id in self.running_jobs:
            del self.running_jobs[job.job_id]

        # Add to appropriate completion list
        if result.success:
            self.completed_jobs[job.job_id] = result
            self.state.completed_jobs += 1
        else:
            self.failed_jobs[job.job_id] = result
            self.state.failed_jobs += 1

        # Update counters
        self.state.running_jobs = len(self.running_jobs)
        self.state.queued_jobs = len(self.job_queue) - self.state.completed_jobs - self.state.failed_jobs - self.state.running_jobs

        # Update progress
        self._update_progress()

    def _update_progress(self) -> None:
        """Update progress and call progress callback if set."""
        if self.progress_callback:
            try:
                self.progress_callback(self.state)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def _wait_for_completion(self, timeout_seconds: float) -> None:
        """Wait for all jobs to complete or timeout."""
        start_time = time.time()

        while self.running_jobs and (time.time() - start_time) < timeout_seconds:
            time.sleep(1)

        if self.running_jobs:
            raise TimeoutError(f"Jobs did not complete within {timeout_seconds} seconds")

    def _run_cleanup(self) -> None:
        """Run all registered cleanup functions."""
        for cleanup_func in self._cleanup_functions:
            try:
                cleanup_func()
            except Exception as e:
                logger.error(f"Cleanup function failed: {e}")

        self._cleanup_functions.clear()


# Convenience functions for common orchestration patterns

def create_strategy_orchestrator(config_path: Optional[str] = None,
                               max_workers: Optional[int] = None) -> BacktestOrchestrator:
    """
    Create a preconfigured orchestrator for strategy backtesting.

    Args:
        config_path: Optional path to configuration file
        max_workers: Optional maximum number of workers

    Returns:
        Configured BacktestOrchestrator instance
    """
    config_manager = ConfigManager(config_path=config_path)
    data_manager = DataManager()

    return BacktestOrchestrator(
        config_manager=config_manager,
        data_manager=data_manager,
        max_workers=max_workers,
        default_execution_mode=ExecutionMode.PARALLEL_THREADS
    )


@contextmanager
def orchestrated_backtest(orchestrator: BacktestOrchestrator):
    """
    Context manager for orchestrated backtesting with automatic cleanup.

    Args:
        orchestrator: BacktestOrchestrator instance

    Yields:
        The orchestrator instance
    """
    try:
        yield orchestrator
    finally:
        orchestrator.shutdown(graceful=True)
