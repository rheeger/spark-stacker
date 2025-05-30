"""
Progress Trackers

Performance monitoring and progress tracking utilities for CLI operations:
- Measure and report backtest execution time
- Track memory usage for large strategy comparisons
- Add performance benchmarks for strategy vs indicator testing
- Include performance metrics in CLI output
- Add performance optimization suggestions
- Integrate performance tracking across all modules
"""

import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import psutil
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PerformanceMetricType(Enum):
    """Types of performance metrics that can be tracked."""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DISK_IO = "disk_io"
    CACHE_HITS = "cache_hits"
    CACHE_MISSES = "cache_misses"
    DATA_FETCH_TIME = "data_fetch_time"
    INDICATOR_CALC_TIME = "indicator_calc_time"
    STRATEGY_EXEC_TIME = "strategy_exec_time"


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""
    metric_type: PerformanceMetricType
    value: Union[float, int]
    unit: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Comprehensive performance report for an operation."""
    operation_name: str
    start_time: datetime
    end_time: datetime
    total_duration: timedelta
    metrics: List[PerformanceMetric] = field(default_factory=list)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    optimization_suggestions: List[str] = field(default_factory=list)
    comparison_baseline: Optional['PerformanceReport'] = None


class ResourceMonitor:
    """Real-time resource usage monitoring."""

    def __init__(self,
                 sampling_interval_seconds: float = 1.0,
                 track_memory: bool = True,
                 track_cpu: bool = True,
                 track_disk_io: bool = True):
        """
        Initialize resource monitor.

        Args:
            sampling_interval_seconds: How often to sample resource usage
            track_memory: Whether to track memory usage
            track_cpu: Whether to track CPU usage
            track_disk_io: Whether to track disk I/O
        """
        self.sampling_interval = sampling_interval_seconds
        self.track_memory = track_memory
        self.track_cpu = track_cpu
        self.track_disk_io = track_disk_io

        self._process = psutil.Process()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._metrics_lock = threading.Lock()
        self._metrics: List[Dict[str, Any]] = []

    def start_monitoring(self) -> None:
        """Start resource monitoring in background thread."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.debug("Resource monitoring started")

    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stop resource monitoring and return summary.

        Returns:
            Dictionary with resource usage summary
        """
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        with self._metrics_lock:
            if not self._metrics:
                return {}

            # Calculate summary statistics
            summary = self._calculate_resource_summary()
            self._metrics.clear()

        logger.debug("Resource monitoring stopped")
        return summary

    def _monitor_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        while self._monitoring:
            try:
                timestamp = datetime.now()
                sample = {'timestamp': timestamp}

                if self.track_memory:
                    memory_info = self._process.memory_info()
                    sample.update({
                        'memory_rss_mb': memory_info.rss / 1024 / 1024,
                        'memory_vms_mb': memory_info.vms / 1024 / 1024,
                        'memory_percent': self._process.memory_percent()
                    })

                if self.track_cpu:
                    sample['cpu_percent'] = self._process.cpu_percent()

                if self.track_disk_io:
                    try:
                        io_counters = self._process.io_counters()
                        sample.update({
                            'disk_read_mb': io_counters.read_bytes / 1024 / 1024,
                            'disk_write_mb': io_counters.write_bytes / 1024 / 1024
                        })
                    except (psutil.AccessDenied, AttributeError):
                        # Some systems don't support I/O counters
                        pass

                with self._metrics_lock:
                    self._metrics.append(sample)

                time.sleep(self.sampling_interval)

            except Exception as e:
                logger.warning(f"Error in resource monitoring: {e}")
                time.sleep(self.sampling_interval)

    def _calculate_resource_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics from collected metrics."""
        if not self._metrics:
            return {}

        summary = {
            'samples_count': len(self._metrics),
            'duration_seconds': (self._metrics[-1]['timestamp'] -
                               self._metrics[0]['timestamp']).total_seconds()
        }

        # Calculate stats for each numeric metric
        numeric_keys = [k for k in self._metrics[0].keys()
                       if k != 'timestamp' and isinstance(self._metrics[0].get(k), (int, float))]

        for key in numeric_keys:
            values = [m[key] for m in self._metrics if key in m]
            if values:
                summary[f'{key}_avg'] = sum(values) / len(values)
                summary[f'{key}_max'] = max(values)
                summary[f'{key}_min'] = min(values)

        return summary


class PerformanceTracker:
    """Main performance tracking and analysis class."""

    def __init__(self,
                 enable_resource_monitoring: bool = True,
                 enable_cache_tracking: bool = True,
                 baseline_comparison: bool = True):
        """
        Initialize performance tracker.

        Args:
            enable_resource_monitoring: Enable real-time resource monitoring
            enable_cache_tracking: Track cache hit/miss rates
            baseline_comparison: Enable baseline performance comparisons
        """
        self.enable_resource_monitoring = enable_resource_monitoring
        self.enable_cache_tracking = enable_cache_tracking
        self.baseline_comparison = baseline_comparison

        self._operation_stack: List[Dict[str, Any]] = []
        self._completed_operations: List[PerformanceReport] = []
        self._baseline_reports: Dict[str, PerformanceReport] = {}
        self._cache_stats = defaultdict(int)

        # Resource monitor
        self._resource_monitor: Optional[ResourceMonitor] = None
        if enable_resource_monitoring:
            self._resource_monitor = ResourceMonitor()

    @contextmanager
    def track_operation(self,
                       operation_name: str,
                       context: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracking operation performance.

        Args:
            operation_name: Name of the operation being tracked
            context: Additional context information

        Yields:
            PerformanceTracker instance for adding custom metrics
        """
        start_time = datetime.now()
        context = context or {}

        operation_info = {
            'name': operation_name,
            'start_time': start_time,
            'context': context,
            'metrics': []
        }

        self._operation_stack.append(operation_info)

        # Start resource monitoring
        if self._resource_monitor:
            self._resource_monitor.start_monitoring()

        try:
            yield self
        finally:
            # Stop resource monitoring
            resource_usage = {}
            if self._resource_monitor:
                resource_usage = self._resource_monitor.stop_monitoring()

            # Complete operation tracking
            end_time = datetime.now()
            operation_info = self._operation_stack.pop()

            report = PerformanceReport(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                total_duration=end_time - start_time,
                metrics=operation_info['metrics'],
                resource_usage=resource_usage
            )

            # Add optimization suggestions
            report.optimization_suggestions = self._generate_optimization_suggestions(report)

            # Add baseline comparison if available
            if self.baseline_comparison and operation_name in self._baseline_reports:
                report.comparison_baseline = self._baseline_reports[operation_name]

            self._completed_operations.append(report)

            logger.info(f"Operation '{operation_name}' completed in {report.total_duration.total_seconds():.2f}s")

    def add_metric(self,
                   metric_type: PerformanceMetricType,
                   value: Union[float, int],
                   unit: str,
                   context: Optional[Dict[str, Any]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a custom performance metric to current operation.

        Args:
            metric_type: Type of metric being recorded
            value: Metric value
            unit: Unit of measurement
            context: Additional context
            metadata: Additional metadata
        """
        if not self._operation_stack:
            logger.warning("No active operation for metric recording")
            return

        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            context=context or {},
            metadata=metadata or {}
        )

        self._operation_stack[-1]['metrics'].append(metric)

    def record_cache_hit(self, cache_type: str = "default") -> None:
        """Record a cache hit."""
        if self.enable_cache_tracking:
            self._cache_stats[f"{cache_type}_hits"] += 1

    def record_cache_miss(self, cache_type: str = "default") -> None:
        """Record a cache miss."""
        if self.enable_cache_tracking:
            self._cache_stats[f"{cache_type}_misses"] += 1

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics."""
        if not self.enable_cache_tracking:
            return {}

        stats = dict(self._cache_stats)

        # Calculate hit rates
        for cache_type in set(key.split('_')[0] for key in stats.keys()):
            hits = stats.get(f"{cache_type}_hits", 0)
            misses = stats.get(f"{cache_type}_misses", 0)
            total = hits + misses

            if total > 0:
                stats[f"{cache_type}_hit_rate"] = hits / total
            else:
                stats[f"{cache_type}_hit_rate"] = 0.0

        return stats

    def save_baseline(self, operation_name: str) -> None:
        """Save current performance as baseline for future comparisons."""
        if not self._completed_operations:
            logger.warning("No completed operations to save as baseline")
            return

        # Find most recent operation with matching name
        for report in reversed(self._completed_operations):
            if report.operation_name == operation_name:
                self._baseline_reports[operation_name] = report
                logger.info(f"Saved baseline for operation '{operation_name}'")
                return

        logger.warning(f"No completed operation found with name '{operation_name}'")

    def generate_performance_report(self,
                                  operation_filter: Optional[str] = None) -> str:
        """
        Generate comprehensive performance report.

        Args:
            operation_filter: Optional filter for operation names

        Returns:
            Formatted performance report string
        """
        if operation_filter:
            reports = [r for r in self._completed_operations
                      if operation_filter in r.operation_name]
        else:
            reports = self._completed_operations

        if not reports:
            return "No performance data available."

        lines = ["📊 Performance Report", "=" * 50, ""]

        # Summary statistics
        total_time = sum(r.total_duration.total_seconds() for r in reports)
        lines.extend([
            f"Total Operations: {len(reports)}",
            f"Total Time: {total_time:.2f}s",
            f"Average Time: {total_time / len(reports):.2f}s",
            ""
        ])

        # Individual operation details
        for report in reports:
            lines.extend(self._format_operation_report(report))
            lines.append("")

        # Cache statistics
        cache_stats = self.get_cache_stats()
        if cache_stats:
            lines.extend(["📈 Cache Statistics", "-" * 30])
            for key, value in cache_stats.items():
                if isinstance(value, float):
                    lines.append(f"{key}: {value:.2%}")
                else:
                    lines.append(f"{key}: {value}")
            lines.append("")

        return "\n".join(lines)

    def _format_operation_report(self, report: PerformanceReport) -> List[str]:
        """Format individual operation report."""
        lines = [
            f"🔍 {report.operation_name}",
            f"Duration: {report.total_duration.total_seconds():.2f}s",
            f"Started: {report.start_time.strftime('%H:%M:%S')}"
        ]

        # Resource usage
        if report.resource_usage:
            lines.append("Resource Usage:")
            for key, value in report.resource_usage.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.2f}")
                else:
                    lines.append(f"  {key}: {value}")

        # Performance metrics
        if report.metrics:
            lines.append("Metrics:")
            for metric in report.metrics:
                lines.append(f"  {metric.metric_type.value}: {metric.value} {metric.unit}")

        # Optimization suggestions
        if report.optimization_suggestions:
            lines.append("💡 Optimization Suggestions:")
            for suggestion in report.optimization_suggestions:
                lines.append(f"  • {suggestion}")

        # Baseline comparison
        if report.comparison_baseline:
            baseline = report.comparison_baseline
            current_time = report.total_duration.total_seconds()
            baseline_time = baseline.total_duration.total_seconds()

            if current_time < baseline_time:
                improvement = ((baseline_time - current_time) / baseline_time) * 100
                lines.append(f"📈 {improvement:.1f}% faster than baseline")
            else:
                regression = ((current_time - baseline_time) / baseline_time) * 100
                lines.append(f"📉 {regression:.1f}% slower than baseline")

        return lines

    def _generate_optimization_suggestions(self, report: PerformanceReport) -> List[str]:
        """Generate optimization suggestions based on performance data."""
        suggestions = []

        # Memory usage suggestions
        if 'memory_rss_mb_max' in report.resource_usage:
            max_memory = report.resource_usage['memory_rss_mb_max']
            if max_memory > 1000:  # > 1GB
                suggestions.append("Consider reducing memory usage or implementing data streaming")

        # CPU usage suggestions
        if 'cpu_percent_avg' in report.resource_usage:
            avg_cpu = report.resource_usage['cpu_percent_avg']
            if avg_cpu < 20:
                suggestions.append("Low CPU usage detected - consider parallel execution")
            elif avg_cpu > 90:
                suggestions.append("High CPU usage - consider optimizing algorithms or adding delays")

        # Cache performance suggestions
        cache_stats = self.get_cache_stats()
        for cache_type in set(key.split('_')[0] for key in cache_stats.keys()):
            hit_rate = cache_stats.get(f"{cache_type}_hit_rate", 0)
            if hit_rate < 0.5:
                suggestions.append(f"Low {cache_type} cache hit rate ({hit_rate:.1%}) - consider cache optimization")

        # Duration-based suggestions
        if report.total_duration.total_seconds() > 300:  # > 5 minutes
            suggestions.append("Long execution time - consider breaking into smaller operations or caching")

        return suggestions


# Global performance tracker instance
_global_tracker: Optional[PerformanceTracker] = None


def get_performance_tracker() -> PerformanceTracker:
    """Get or create global performance tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
    return _global_tracker


def track_performance(operation_name: str, context: Optional[Dict[str, Any]] = None):
    """
    Decorator for tracking function performance.

    Args:
        operation_name: Name of the operation
        context: Additional context information
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = get_performance_tracker()
            with tracker.track_operation(operation_name, context):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class ProgressReporter:
    """Enhanced progress reporting with performance integration."""

    def __init__(self,
                 total_items: int,
                 description: str = "Processing",
                 show_performance: bool = True,
                 update_interval: float = 0.5):
        """
        Initialize progress reporter.

        Args:
            total_items: Total number of items to process
            description: Description for progress bar
            show_performance: Whether to show performance metrics
            update_interval: How often to update progress (seconds)
        """
        self.total_items = total_items
        self.description = description
        self.show_performance = show_performance
        self.update_interval = update_interval

        self._progress_bar: Optional[tqdm] = None
        self._start_time: Optional[datetime] = None
        self._last_update_time: Optional[datetime] = None
        self._performance_tracker = get_performance_tracker() if show_performance else None

    def __enter__(self):
        """Start progress tracking."""
        self._start_time = datetime.now()
        self._last_update_time = self._start_time

        self._progress_bar = tqdm(
            total=self.total_items,
            desc=self.description,
            unit="item",
            miniters=1,
            dynamic_ncols=True
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete progress tracking."""
        if self._progress_bar:
            self._progress_bar.close()

        if self._start_time:
            duration = datetime.now() - self._start_time
            logger.info(f"Completed {self.description} in {duration.total_seconds():.2f}s")

    def update(self,
               increment: int = 1,
               custom_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Update progress with optional performance metrics.

        Args:
            increment: How many items to increment
            custom_metrics: Optional custom metrics to display
        """
        if not self._progress_bar:
            return

        self._progress_bar.update(increment)

        # Update performance information periodically
        now = datetime.now()
        if (self._last_update_time and
            (now - self._last_update_time).total_seconds() >= self.update_interval):

            self._update_performance_info(custom_metrics)
            self._last_update_time = now

    def _update_performance_info(self, custom_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Update progress bar with performance information."""
        if not self._progress_bar or not self._start_time:
            return

        elapsed = datetime.now() - self._start_time
        completed = self._progress_bar.n

        if completed > 0:
            rate = completed / elapsed.total_seconds()
            remaining = self.total_items - completed
            eta = remaining / rate if rate > 0 else 0

            postfix = f"rate={rate:.1f}/s"
            if eta > 0:
                eta_str = str(timedelta(seconds=int(eta)))
                postfix += f", ETA={eta_str}"

            if custom_metrics:
                metric_str = ", ".join(f"{k}={v}" for k, v in custom_metrics.items())
                postfix += f", {metric_str}"

            self._progress_bar.set_postfix_str(postfix)


def create_progress_reporter(total_items: int,
                           description: str = "Processing",
                           show_performance: bool = True) -> ProgressReporter:
    """
    Create a progress reporter with performance tracking.

    Args:
        total_items: Total number of items to process
        description: Description for progress bar
        show_performance: Whether to show performance metrics

    Returns:
        ProgressReporter instance
    """
    return ProgressReporter(
        total_items=total_items,
        description=description,
        show_performance=show_performance
    )
