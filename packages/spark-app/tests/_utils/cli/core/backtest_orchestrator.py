"""
Backtest Orchestrator

This module provides centralized orchestration for backtesting operations:
- Coordinate overall backtesting workflow
- Handle resource allocation and cleanup
- Manage parallel execution of multiple backtests
- Add progress tracking and user updates
- Handle interruption and graceful shutdown
- Coordinate between different manager types (strategy, indicator, scenario, comparison)
- Enhanced resource management with cleanup, timeouts, and disk space monitoring
"""

import asyncio
import gc
import logging
import os
import shutil
import signal
# Import validation modules for pre-flight checks
import sys
import tempfile
import threading
import time
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                TimeoutError, as_completed)
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import psutil
from tqdm import tqdm

from .config_manager import ConfigManager
from .data_manager import DataManager

cli_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, cli_dir)
from validation.config_validator import ConfigValidator
from validation.data_validator import DataValidator
from validation.strategy_validator import StrategyValidator

from ..utils.progress_trackers import (PerformanceMetricType,
                                       get_performance_tracker)

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
    MAX_DISK_USAGE_MB = "max_disk_usage_mb"
    MIN_FREE_DISK_MB = "min_free_disk_mb"


class ResourceState(Enum):
    """Resource usage states."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EXCEEDED = "exceeded"


@dataclass
class ResourceUsage:
    """Current resource usage snapshot."""
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    disk_usage_mb: float = 0.0
    disk_free_mb: float = 0.0
    active_jobs: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def get_memory_state(self, limit_mb: float) -> ResourceState:
        """Get memory usage state relative to limit."""
        if self.memory_mb > limit_mb:
            return ResourceState.EXCEEDED
        elif self.memory_mb > limit_mb * 0.9:
            return ResourceState.CRITICAL
        elif self.memory_mb > limit_mb * 0.7:
            return ResourceState.WARNING
        else:
            return ResourceState.NORMAL

    def get_disk_state(self, min_free_mb: float) -> ResourceState:
        """Get disk usage state relative to minimum free space."""
        if self.disk_free_mb < min_free_mb:
            return ResourceState.EXCEEDED
        elif self.disk_free_mb < min_free_mb * 1.5:
            return ResourceState.CRITICAL
        elif self.disk_free_mb < min_free_mb * 2.0:
            return ResourceState.WARNING
        else:
            return ResourceState.NORMAL


class ResourceManager:
    """Manages system resources for backtest operations."""

    def __init__(self,
                 resource_limits: Dict[ResourceLimitType, Any],
                 monitoring_interval: float = 2.0,
                 cleanup_interval: float = 30.0):
        """
        Initialize resource manager.

        Args:
            resource_limits: Dictionary of resource limits
            monitoring_interval: How often to check resource usage (seconds)
            cleanup_interval: How often to run cleanup (seconds)
        """
        self.resource_limits = resource_limits
        self.monitoring_interval = monitoring_interval
        self.cleanup_interval = cleanup_interval

        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        self._resource_lock = threading.Lock()

        self._current_usage = ResourceUsage()
        self._usage_history: List[ResourceUsage] = []
        self._temp_directories: Set[Path] = set()
        self._active_processes: Set[int] = set()

        # Performance tracking
        self._performance_tracker = get_performance_tracker()

    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        logger.info("Resource monitoring started")

    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)

        logger.info("Resource monitoring stopped")

    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        with self._resource_lock:
            return self._current_usage

    def check_resource_limits(self) -> List[str]:
        """Check if any resource limits are exceeded."""
        warnings = []
        usage = self.get_current_usage()

        # Check memory limit
        if ResourceLimitType.MAX_MEMORY_MB in self.resource_limits:
            limit = self.resource_limits[ResourceLimitType.MAX_MEMORY_MB]
            state = usage.get_memory_state(limit)
            if state == ResourceState.EXCEEDED:
                warnings.append(f"Memory usage ({usage.memory_mb:.1f}MB) exceeds limit ({limit}MB)")
            elif state == ResourceState.CRITICAL:
                warnings.append(f"Memory usage ({usage.memory_mb:.1f}MB) approaching limit ({limit}MB)")

        # Check disk space
        if ResourceLimitType.MIN_FREE_DISK_MB in self.resource_limits:
            limit = self.resource_limits[ResourceLimitType.MIN_FREE_DISK_MB]
            state = usage.get_disk_state(limit)
            if state == ResourceState.EXCEEDED:
                warnings.append(f"Free disk space ({usage.disk_free_mb:.1f}MB) below minimum ({limit}MB)")
            elif state == ResourceState.CRITICAL:
                warnings.append(f"Free disk space ({usage.disk_free_mb:.1f}MB) approaching minimum ({limit}MB)")

        # Check CPU usage
        if ResourceLimitType.MAX_CPU_PERCENT in self.resource_limits:
            limit = self.resource_limits[ResourceLimitType.MAX_CPU_PERCENT]
            if usage.cpu_percent > limit:
                warnings.append(f"CPU usage ({usage.cpu_percent:.1f}%) exceeds limit ({limit}%)")

        return warnings

    def can_start_job(self, estimated_memory_mb: float = 0) -> Tuple[bool, List[str]]:
        """Check if a new job can be started given current resource usage."""
        warnings = self.check_resource_limits()

        # Check if we have room for the estimated additional memory
        if ResourceLimitType.MAX_MEMORY_MB in self.resource_limits:
            limit = self.resource_limits[ResourceLimitType.MAX_MEMORY_MB]
            projected_memory = self._current_usage.memory_mb + estimated_memory_mb
            if projected_memory > limit:
                warnings.append(f"Cannot start job: projected memory usage ({projected_memory:.1f}MB) would exceed limit ({limit}MB)")

        # Check concurrent job limit
        if ResourceLimitType.MAX_CONCURRENT_JOBS in self.resource_limits:
            limit = self.resource_limits[ResourceLimitType.MAX_CONCURRENT_JOBS]
            if self._current_usage.active_jobs >= limit:
                warnings.append(f"Cannot start job: already at concurrent job limit ({limit})")

        can_start = len(warnings) == 0
        return can_start, warnings

    def register_temp_directory(self, temp_dir: Path) -> None:
        """Register a temporary directory for cleanup."""
        with self._resource_lock:
            self._temp_directories.add(temp_dir)

    def register_process(self, pid: int) -> None:
        """Register a process for monitoring."""
        with self._resource_lock:
            self._active_processes.add(pid)

    def cleanup_temp_files(self) -> int:
        """Clean up temporary files and directories."""
        cleaned_count = 0

        with self._resource_lock:
            temp_dirs_to_remove = []

            for temp_dir in self._temp_directories:
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                        cleaned_count += 1
                        logger.debug(f"Cleaned temp directory: {temp_dir}")
                    temp_dirs_to_remove.append(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean temp directory {temp_dir}: {e}")

            for temp_dir in temp_dirs_to_remove:
                self._temp_directories.discard(temp_dir)

        if cleaned_count > 0:
            logger.info(f"Cleaned {cleaned_count} temporary directories")

        return cleaned_count

    def _monitor_loop(self) -> None:
        """Main resource monitoring loop."""
        while self._monitoring:
            try:
                # Get current system usage
                process = psutil.Process()
                memory_info = process.memory_info()
                disk_usage = shutil.disk_usage("/")

                usage = ResourceUsage(
                    memory_mb=memory_info.rss / 1024 / 1024,
                    memory_percent=process.memory_percent(),
                    cpu_percent=process.cpu_percent(),
                    disk_usage_mb=(disk_usage.total - disk_usage.free) / 1024 / 1024,
                    disk_free_mb=disk_usage.free / 1024 / 1024,
                    active_jobs=len(self._active_processes)
                )

                with self._resource_lock:
                    self._current_usage = usage
                    self._usage_history.append(usage)

                    # Keep only recent history (last hour)
                    cutoff_time = datetime.now() - timedelta(hours=1)
                    self._usage_history = [
                        u for u in self._usage_history
                        if u.timestamp > cutoff_time
                    ]

                # Track performance metrics
                if self._performance_tracker:
                    self._performance_tracker.add_metric(
                        PerformanceMetricType.MEMORY_USAGE,
                        usage.memory_mb,
                        "MB"
                    )
                    self._performance_tracker.add_metric(
                        PerformanceMetricType.CPU_USAGE,
                        usage.cpu_percent,
                        "percent"
                    )

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.warning(f"Error in resource monitoring: {e}")
                time.sleep(self.monitoring_interval)

    def _cleanup_loop(self) -> None:
        """Periodic cleanup loop."""
        while self._monitoring:
            try:
                # Clean up temporary files
                self.cleanup_temp_files()

                # Force garbage collection if memory usage is high
                usage = self.get_current_usage()
                if ResourceLimitType.MAX_MEMORY_MB in self.resource_limits:
                    limit = self.resource_limits[ResourceLimitType.MAX_MEMORY_MB]
                    if usage.memory_mb > limit * 0.8:  # > 80% of limit
                        gc.collect()
                        logger.debug("Forced garbage collection due to high memory usage")

                time.sleep(self.cleanup_interval)

            except Exception as e:
                logger.warning(f"Error in cleanup loop: {e}")
                time.sleep(self.cleanup_interval)


class BacktestOrchestrator:
    """
    Central orchestrator for backtesting operations.

    Features:
    - Coordinate multiple types of backtests (strategy, indicator, scenario, comparison)
    - Manage parallel execution with resource limits and monitoring
    - Provide real-time progress tracking and user updates
    - Handle graceful interruption and cleanup
    - Resource allocation and cleanup management with enhanced monitoring
    - Job dependency management and execution scheduling
    - Timeout handling for long-running operations
    - Disk space management for large report generation
    """

    def __init__(self,
                 config_manager: Optional[ConfigManager] = None,
                 data_manager: Optional[DataManager] = None,
                 max_workers: Optional[int] = None,
                 default_execution_mode: ExecutionMode = ExecutionMode.ADAPTIVE,
                 resource_limits: Optional[Dict[ResourceLimitType, Any]] = None,
                 progress_callback: Optional[Callable[[OrchestrationState], None]] = None,
                 enable_timeout_handling: bool = True,
                 default_job_timeout_seconds: float = 3600.0,
                 enable_resource_monitoring: bool = True):
        """
        Initialize the backtest orchestrator.

        Args:
            config_manager: Configuration manager instance
            data_manager: Data manager instance
            max_workers: Maximum number of concurrent workers
            default_execution_mode: Default execution mode for jobs
            resource_limits: Resource limits for job execution
            progress_callback: Optional callback for progress updates
            enable_timeout_handling: Enable timeout handling for jobs
            default_job_timeout_seconds: Default timeout for jobs
            enable_resource_monitoring: Enable resource monitoring
        """
        self.config_manager = config_manager or ConfigManager()

        # Create DataManager with configuration if not provided
        if data_manager is None:
            config = self.config_manager.load_config()
            self.data_manager = DataManager(config=config)
        else:
            self.data_manager = data_manager
        self.enable_timeout_handling = enable_timeout_handling
        self.default_job_timeout_seconds = default_job_timeout_seconds
        self.enable_resource_monitoring = enable_resource_monitoring

        # Worker configuration
        if max_workers is None:
            self.max_workers = min(psutil.cpu_count(), 8)  # Reasonable default
        else:
            self.max_workers = max_workers

        self.default_execution_mode = default_execution_mode
        self.progress_callback = progress_callback

        # Enhanced resource limits with disk space management
        self.resource_limits = resource_limits or {
            ResourceLimitType.MAX_MEMORY_MB: 8192,  # 8GB default
            ResourceLimitType.MAX_CPU_PERCENT: 80,
            ResourceLimitType.MAX_CONCURRENT_JOBS: self.max_workers,
            ResourceLimitType.MAX_EXECUTION_TIME_SECONDS: 3600,  # 1 hour
            ResourceLimitType.MIN_FREE_DISK_MB: 1024,  # 1GB minimum free
            ResourceLimitType.MAX_DISK_USAGE_MB: 10240  # 10GB max usage
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

        # Enhanced resource management
        self._resource_manager: Optional[ResourceManager] = None
        if enable_resource_monitoring:
            self._resource_manager = ResourceManager(self.resource_limits)

        # Temporary file management
        self._temp_directories: Set[Path] = set()
        self._active_futures: Set[Any] = set()

        # Performance tracking
        self._performance_tracker = get_performance_tracker()

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        logger.info(f"BacktestOrchestrator initialized with {self.max_workers} max workers")
        logger.info(f"Resource monitoring enabled: {enable_resource_monitoring}")
        logger.info(f"Timeout handling enabled: {enable_timeout_handling}")

    def __enter__(self):
        """Context manager entry."""
        if self._resource_manager:
            self._resource_manager.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown(cleanup=True)

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

        # Set default timeout if not specified
        if job.timeout_seconds is None and self.enable_timeout_handling:
            job.timeout_seconds = self.default_job_timeout_seconds

        # Add to queue
        self.job_queue.append(job)
        self.state.total_jobs += 1
        self.state.queued_jobs += 1

        logger.info(f"Added job {job.job_id} to queue (type: {job.job_type.value})")
        return job.job_id

    def execute_jobs(self,
                    execution_mode: Optional[ExecutionMode] = None,
                    max_retries: int = 1) -> Dict[str, JobResult]:
        """
        Execute all queued jobs with enhanced resource management and timeout handling.

        Args:
            execution_mode: Execution mode override
            max_retries: Maximum number of retries for failed jobs

        Returns:
            Dictionary mapping job IDs to their results
        """
        if not self.job_queue:
            logger.warning("No jobs queued for execution")
            return {}

        execution_mode = execution_mode or self.default_execution_mode
        operation_name = f"execute_{len(self.job_queue)}_jobs_{execution_mode.value}"

        if self._performance_tracker:
            with self._performance_tracker.track_operation(operation_name):
                return self._execute_jobs_internal(execution_mode, max_retries)
        else:
            return self._execute_jobs_internal(execution_mode, max_retries)

    def _execute_jobs_internal(self,
                              execution_mode: ExecutionMode,
                              max_retries: int) -> Dict[str, JobResult]:
        """Internal job execution logic."""
        logger.info(f"Starting execution of {len(self.job_queue)} jobs in {execution_mode.value} mode")

        self.state.start_time = datetime.now()

        try:
            if execution_mode == ExecutionMode.SEQUENTIAL:
                return self._execute_sequential(max_retries)
            elif execution_mode == ExecutionMode.PARALLEL_THREADS:
                return self._execute_parallel_threads(max_retries)
            elif execution_mode == ExecutionMode.PARALLEL_PROCESSES:
                return self._execute_parallel_processes(max_retries)
            elif execution_mode == ExecutionMode.ADAPTIVE:
                return self._execute_adaptive(max_retries)
            else:
                raise ValueError(f"Unsupported execution mode: {execution_mode}")

        except KeyboardInterrupt:
            logger.warning("Execution interrupted by user")
            self._handle_interruption()
            raise
        except Exception as e:
            logger.error(f"Error during job execution: {e}")
            self._handle_execution_error(e)
            raise
        finally:
            self._finalize_execution()

    def _execute_with_timeout(self,
                             job: BacktestJob,
                             executor: Union[ThreadPoolExecutor, ProcessPoolExecutor]) -> JobResult:
        """Execute a job with timeout handling."""
        start_time = time.time()

        try:
            # Check resource limits before starting
            if self._resource_manager:
                can_start, warnings = self._resource_manager.can_start_job()
                if not can_start:
                    raise ResourceLimitExceededError(f"Cannot start job {job.job_id}: {'; '.join(warnings)}")

                if warnings:
                    logger.warning(f"Resource warnings for job {job.job_id}: {'; '.join(warnings)}")

            # Create temporary directory for job
            temp_dir = self._create_job_temp_directory(job.job_id)
            job.temp_files.append(temp_dir)

            if self._resource_manager:
                self._resource_manager.register_temp_directory(temp_dir)

            # Submit job with timeout
            future = executor.submit(self._execute_single_job, job)
            self._active_futures.add(future)

            try:
                if self.enable_timeout_handling and job.timeout_seconds:
                    result_data = future.result(timeout=job.timeout_seconds)
                else:
                    result_data = future.result()

                execution_time = time.time() - start_time

                # Get resource usage during execution
                resource_usage = {}
                if self._resource_manager:
                    current_usage = self._resource_manager.get_current_usage()
                    resource_usage = {
                        'peak_memory_mb': current_usage.memory_mb,
                        'avg_cpu_percent': current_usage.cpu_percent,
                        'execution_time': execution_time
                    }

                result = JobResult(
                    job_id=job.job_id,
                    success=True,
                    result_data=result_data,
                    execution_time_seconds=execution_time,
                    resource_usage=resource_usage
                )

                # Cleanup job resources
                self._cleanup_job_resources(job)

                return result

            except TimeoutError:
                logger.error(f"Job {job.job_id} timed out after {job.timeout_seconds}s")
                future.cancel()

                result = JobResult(
                    job_id=job.job_id,
                    success=False,
                    error=TimeoutError(f"Job timed out after {job.timeout_seconds}s"),
                    execution_time_seconds=time.time() - start_time,
                    was_timeout=True
                )

                # Force cleanup after timeout
                self._cleanup_job_resources(job, force=True)

                return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing job {job.job_id}: {e}")

            result = JobResult(
                job_id=job.job_id,
                success=False,
                error=e,
                execution_time_seconds=execution_time
            )

            # Cleanup after error
            self._cleanup_job_resources(job, force=True)

            return result
        finally:
            self._active_futures.discard(future)

    def _cleanup_job_resources(self, job: BacktestJob, force: bool = False) -> None:
        """Clean up resources used by a job."""
        cleanup_successful = True

        try:
            # Run custom cleanup functions
            for cleanup_func in job.cleanup_functions:
                try:
                    cleanup_func()
                except Exception as e:
                    logger.warning(f"Job {job.job_id} cleanup function failed: {e}")
                    cleanup_successful = False

            # Clean up temporary files
            for temp_path in job.temp_files:
                try:
                    if temp_path.exists():
                        if temp_path.is_file():
                            temp_path.unlink()
                        else:
                            shutil.rmtree(temp_path)
                        logger.debug(f"Cleaned up temp path: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {temp_path}: {e}")
                    cleanup_successful = False

            # Force garbage collection if requested
            if force:
                gc.collect()

        except Exception as e:
            logger.error(f"Error during job cleanup for {job.job_id}: {e}")
            cleanup_successful = False

        if not cleanup_successful:
            logger.warning(f"Some cleanup operations failed for job {job.job_id}")

    def _create_job_temp_directory(self, job_id: str) -> Path:
        """Create a temporary directory for job execution."""
        temp_dir = Path(tempfile.mkdtemp(prefix=f"backtest_job_{job_id}_"))
        self._temp_directories.add(temp_dir)
        return temp_dir

    def cleanup_all_resources(self) -> None:
        """Clean up all orchestrator resources."""
        logger.info("Starting comprehensive resource cleanup")

        cleanup_count = 0

        # Cancel any active futures
        for future in list(self._active_futures):
            try:
                if not future.done():
                    future.cancel()
                    cleanup_count += 1
            except Exception as e:
                logger.warning(f"Failed to cancel future: {e}")

        self._active_futures.clear()

        # Clean up temporary directories
        for temp_dir in list(self._temp_directories):
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    cleanup_count += 1
                    logger.debug(f"Cleaned temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean temp directory {temp_dir}: {e}")

        self._temp_directories.clear()

        # Run registered cleanup functions
        for cleanup_func in self._cleanup_functions:
            try:
                cleanup_func()
                cleanup_count += 1
            except Exception as e:
                logger.warning(f"Cleanup function failed: {e}")

        # Clean up data manager caches
        if hasattr(self.data_manager, 'cleanup_temp_files'):
            try:
                cache_cleaned = self.data_manager.cleanup_temp_files()
                cleanup_count += cache_cleaned
            except Exception as e:
                logger.warning(f"Failed to clean data manager cache: {e}")

        # Stop resource monitoring
        if self._resource_manager:
            try:
                self._resource_manager.stop_monitoring()
                cleanup_count += self._resource_manager.cleanup_temp_files()
            except Exception as e:
                logger.warning(f"Failed to stop resource manager: {e}")

        # Force garbage collection
        gc.collect()

        logger.info(f"Resource cleanup completed. Cleaned {cleanup_count} items")

    def shutdown(self, cleanup: bool = True, timeout: float = 30.0) -> None:
        """
        Gracefully shutdown the orchestrator.

        Args:
            cleanup: Whether to perform resource cleanup
            timeout: Maximum time to wait for shutdown
        """
        logger.info("Shutting down BacktestOrchestrator")

        self._shutdown_requested = True

        # Wait for running jobs to complete or timeout
        start_time = time.time()
        while self.state.running_jobs > 0 and (time.time() - start_time) < timeout:
            time.sleep(0.5)

        if self.state.running_jobs > 0:
            logger.warning(f"{self.state.running_jobs} jobs still running after timeout")

        if cleanup:
            self.cleanup_all_resources()

        logger.info("BacktestOrchestrator shutdown complete")

    def get_resource_usage_report(self) -> Dict[str, Any]:
        """Get comprehensive resource usage report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'orchestrator_state': {
                'total_jobs': self.state.total_jobs,
                'completed_jobs': self.state.completed_jobs,
                'failed_jobs': self.state.failed_jobs,
                'running_jobs': self.state.running_jobs,
                'queued_jobs': self.state.queued_jobs
            }
        }

        if self._resource_manager:
            current_usage = self._resource_manager.get_current_usage()
            warnings = self._resource_manager.check_resource_limits()

            report['resource_usage'] = {
                'memory_mb': current_usage.memory_mb,
                'memory_percent': current_usage.memory_percent,
                'cpu_percent': current_usage.cpu_percent,
                'disk_free_mb': current_usage.disk_free_mb,
                'active_jobs': current_usage.active_jobs,
                'warnings': warnings
            }

            report['resource_limits'] = self.resource_limits

        # Add data manager cache stats if available
        if hasattr(self.data_manager, 'get_enhanced_cache_stats'):
            try:
                report['cache_stats'] = self.data_manager.get_enhanced_cache_stats()
            except Exception as e:
                logger.warning(f"Failed to get cache stats: {e}")

        return report

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown(cleanup=True)

        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except ValueError:
            # Signal handlers can only be set from main thread
            logger.debug("Could not set signal handlers (not in main thread)")

    def _handle_interruption(self) -> None:
        """Handle user interruption."""
        logger.info("Handling interruption - cleaning up resources")
        self._shutdown_requested = True
        self.cleanup_all_resources()

    def _handle_execution_error(self, error: Exception) -> None:
        """Handle execution errors."""
        logger.error(f"Execution error occurred: {error}")
        # Perform any error-specific cleanup here

    def _finalize_execution(self) -> None:
        """Finalize execution and update state."""
        end_time = datetime.now()
        if self.state.start_time:
            total_duration = end_time - self.state.start_time
            logger.info(f"Job execution completed in {total_duration.total_seconds():.2f}s")

        # Update final state
        self.state.estimated_completion_time = end_time

        # Log final statistics
        self._log_execution_summary()

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

    def _execute_sequential(self, max_retries: int) -> Dict[str, JobResult]:
        """Execute jobs sequentially."""
        results = {}

        with ThreadPoolExecutor(max_workers=1) as executor:
            for job in self.job_queue:
                if self._shutdown_requested:
                    break

                self.state.running_jobs += 1
                self.state.queued_jobs -= 1

                try:
                    result = self._execute_with_timeout(job, executor)
                    results[job.job_id] = result

                    if result.success:
                        self.completed_jobs[job.job_id] = result
                        self.state.completed_jobs += 1
                    else:
                        self.failed_jobs[job.job_id] = result
                        self.state.failed_jobs += 1

                finally:
                    self.state.running_jobs -= 1

        return results

    def _execute_parallel_threads(self, max_retries: int) -> Dict[str, JobResult]:
        """Execute jobs using thread pool."""
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {}
            for job in self.job_queue:
                if self._shutdown_requested:
                    break

                future = executor.submit(self._execute_with_timeout, job, executor)
                future_to_job[future] = job
                self.state.running_jobs += 1
                self.state.queued_jobs -= 1

            # Collect results as they complete
            for future in as_completed(future_to_job):
                job = future_to_job[future]

                try:
                    result = future.result()
                    results[job.job_id] = result

                    if result.success:
                        self.completed_jobs[job.job_id] = result
                        self.state.completed_jobs += 1
                    else:
                        self.failed_jobs[job.job_id] = result
                        self.state.failed_jobs += 1

                except Exception as e:
                    error_result = JobResult(
                        job_id=job.job_id,
                        success=False,
                        error=e,
                        execution_time_seconds=0.0
                    )
                    results[job.job_id] = error_result
                    self.failed_jobs[job.job_id] = error_result
                    self.state.failed_jobs += 1

                finally:
                    self.state.running_jobs -= 1

        return results

    def _execute_parallel_processes(self, max_retries: int) -> Dict[str, JobResult]:
        """Execute jobs using process pool (for CPU-intensive tasks)."""
        # Note: Process pool execution would require careful serialization
        # For now, fall back to thread execution
        logger.info("Process pool execution not fully implemented, falling back to thread execution")
        return self._execute_parallel_threads(max_retries)

    def _execute_adaptive(self, max_retries: int) -> Dict[str, JobResult]:
        """Adaptively choose execution mode based on workload."""
        total_jobs = len(self.job_queue)

        # Simple heuristic for choosing execution mode
        if total_jobs == 1:
            return self._execute_sequential(max_retries)
        elif total_jobs <= 4:
            return self._execute_parallel_threads(max_retries)
        else:
            # For larger workloads, use threading with dynamic adjustment
            return self._execute_parallel_threads(max_retries)

    def _execute_single_job(self, job: BacktestJob) -> JobResult:
        """Execute a single backtest job."""
        logger.info(f"Executing job {job.job_id} of type {job.job_type.value}")

        # This is a placeholder - actual job execution would depend on job type
        # and would involve calling appropriate manager classes

        try:
            if job.job_type == BacktestType.SINGLE_STRATEGY:
                return self._execute_strategy_job(job)
            elif job.job_type == BacktestType.SCENARIO_TESTING:
                return self._execute_scenario_job(job)
            elif job.job_type == BacktestType.COMPARISON:
                return self._execute_comparison_job(job)
            else:
                raise ValueError(f"Unsupported job type: {job.job_type}")

        except Exception as e:
            logger.error(f"Job {job.job_id} execution failed: {e}")
            raise

    def _execute_strategy_job(self, job: BacktestJob) -> Dict[str, Any]:
        """Execute a strategy backtest job."""
        # Placeholder implementation
        strategy_name = job.parameters['strategy_name']
        days = job.parameters.get('days', 30)

        # Simulate work
        import time
        time.sleep(2)

        return {
            'strategy_name': strategy_name,
            'days': days,
            'total_return': 0.15,  # Placeholder result
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08
        }

    def _execute_scenario_job(self, job: BacktestJob) -> Dict[str, Any]:
        """Execute a scenario testing job."""
        # Placeholder implementation
        scenarios = job.parameters.get('scenarios', [])

        # Simulate work
        import time
        time.sleep(len(scenarios) * 0.5)

        return {
            'scenarios_tested': len(scenarios),
            'scenario_results': {scenario: {'return': 0.1} for scenario in scenarios}
        }

    def _execute_comparison_job(self, job: BacktestJob) -> Dict[str, Any]:
        """Execute a strategy comparison job."""
        # Placeholder implementation
        strategies = job.parameters.get('strategy_names', [])

        # Simulate work
        import time
        time.sleep(len(strategies) * 0.3)

        return {
            'strategies_compared': len(strategies),
            'comparison_results': {strategy: {'rank': i+1} for i, strategy in enumerate(strategies)}
        }

    def _update_progress(self) -> None:
        """Update progress and call progress callback if provided."""
        if self.progress_callback:
            try:
                self.progress_callback(self.state)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def _log_execution_summary(self) -> None:
        """Log execution summary and performance metrics."""
        end_time = datetime.now()
        if self.state.start_time:
            total_duration = end_time - self.state.start_time
            logger.info(f"Total execution time: {total_duration.total_seconds():.2f}s")

            if self._performance_tracker:
                self._performance_tracker.add_metric(
                    PerformanceMetricType.EXECUTION_TIME,
                    total_duration.total_seconds(),
                    "seconds"
                )

        # Log final state
        logger.info(f"Execution summary: {self.state.completed_jobs} completed, {self.state.failed_jobs} failed, {self.state.running_jobs} still running")


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
    timeout_seconds: Optional[float] = None
    temp_files: List[Path] = field(default_factory=list)
    cleanup_functions: List[Callable[[], None]] = field(default_factory=list)


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
    was_timeout: bool = False
    cleanup_successful: bool = True


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
    current_resource_usage: ResourceUsage = field(default_factory=ResourceUsage)
    resource_warnings: List[str] = field(default_factory=list)


class BacktestOrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    pass


class ResourceLimitExceededError(BacktestOrchestratorError):
    """Raised when resource limits are exceeded."""
    pass


class JobExecutionError(BacktestOrchestratorError):
    """Raised when job execution fails."""
    pass
