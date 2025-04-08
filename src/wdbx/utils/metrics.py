"""
Metrics utilities for WDBX.

This module provides functionality for collecting and exposing metrics
for monitoring and observability, with built-in Prometheus integration.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

from .logging_utils import get_logger

# Initialize logger
logger = get_logger("wdbx.metrics")

try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary

    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not available, metrics will not be exported to Prometheus")
    PROMETHEUS_AVAILABLE = False

# Type variable for generic function
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class MetricsRegistry:
    """Registry for metrics."""

    counters: Dict[str, Any] = field(default_factory=dict)
    gauges: Dict[str, Any] = field(default_factory=dict)
    histograms: Dict[str, Any] = field(default_factory=dict)
    summaries: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize metrics registry."""
        self.enabled = True

    def register_counter(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Any:
        """
        Register a counter metric.

        Args:
            name: Metric name
            description: Metric description
            labels: Labels for the metric

        Returns:
            Counter metric
        """
        if not self.enabled:
            return None

        if PROMETHEUS_AVAILABLE:
            full_name = f"wdbx_{name}"
            counter = Counter(full_name, description, labels or [])
            self.counters[name] = counter
            return counter
        else:
            # Use a simple dict to track counters if Prometheus is not available
            counter = {
                "name": name,
                "description": description,
                "labels": labels or [],
                "value": 0,
            }
            self.counters[name] = counter
            return counter

    def register_gauge(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Any:
        """
        Register a gauge metric.

        Args:
            name: Metric name
            description: Metric description
            labels: Labels for the metric

        Returns:
            Gauge metric
        """
        if not self.enabled:
            return None

        if PROMETHEUS_AVAILABLE:
            full_name = f"wdbx_{name}"
            gauge = Gauge(full_name, description, labels or [])
            self.gauges[name] = gauge
            return gauge
        else:
            # Use a simple dict to track gauges if Prometheus is not available
            gauge = {
                "name": name,
                "description": description,
                "labels": labels or [],
                "value": 0,
            }
            self.gauges[name] = gauge
            return gauge

    def register_histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Any:
        """
        Register a histogram metric.

        Args:
            name: Metric name
            description: Metric description
            labels: Labels for the metric
            buckets: Histogram buckets

        Returns:
            Histogram metric
        """
        if not self.enabled:
            return None

        if PROMETHEUS_AVAILABLE:
            full_name = f"wdbx_{name}"
            histogram = Histogram(full_name, description, labels or [], buckets=buckets)
            self.histograms[name] = histogram
            return histogram
        else:
            # Use a simple dict to track histograms if Prometheus is not available
            histogram = {
                "name": name,
                "description": description,
                "labels": labels or [],
                "buckets": buckets or [],
                "values": [],
            }
            self.histograms[name] = histogram
            return histogram

    def register_summary(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Any:
        """
        Register a summary metric.

        Args:
            name: Metric name
            description: Metric description
            labels: Labels for the metric

        Returns:
            Summary metric
        """
        if not self.enabled:
            return None

        if PROMETHEUS_AVAILABLE:
            full_name = f"wdbx_{name}"
            summary = Summary(full_name, description, labels or [])
            self.summaries[name] = summary
            return summary
        else:
            # Use a simple dict to track summaries if Prometheus is not available
            summary = {
                "name": name,
                "description": description,
                "labels": labels or [],
                "values": [],
            }
            self.summaries[name] = summary
            return summary

    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Increment a counter.

        Args:
            name: Counter name
            value: Increment value
            labels: Counter labels
        """
        if not self.enabled or name not in self.counters:
            return

        counter = self.counters[name]

        if PROMETHEUS_AVAILABLE:
            if labels:
                counter.labels(**labels).inc(value)
            else:
                counter.inc(value)
        else:
            counter["value"] += value

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Set a gauge value.

        Args:
            name: Gauge name
            value: Gauge value
            labels: Gauge labels
        """
        if not self.enabled or name not in self.gauges:
            return

        gauge = self.gauges[name]

        if PROMETHEUS_AVAILABLE:
            if labels:
                gauge.labels(**labels).set(value)
            else:
                gauge.set(value)
        else:
            gauge["value"] = value

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Observe a histogram value.

        Args:
            name: Histogram name
            value: Observed value
            labels: Histogram labels
        """
        if not self.enabled or name not in self.histograms:
            return

        histogram = self.histograms[name]

        if PROMETHEUS_AVAILABLE:
            if labels:
                histogram.labels(**labels).observe(value)
            else:
                histogram.observe(value)
        else:
            histogram["values"].append(value)

    def observe_summary(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Observe a summary value.

        Args:
            name: Summary name
            value: Observed value
            labels: Summary labels
        """
        if not self.enabled or name not in self.summaries:
            return

        summary = self.summaries[name]

        if PROMETHEUS_AVAILABLE:
            if labels:
                summary.labels(**labels).observe(value)
            else:
                summary.observe(value)
        else:
            summary["values"].append(value)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics as a dictionary.

        Returns:
            Dictionary of metrics
        """
        if PROMETHEUS_AVAILABLE:
            return {}  # Not needed for Prometheus

        return {
            "counters": {name: counter["value"] for name, counter in self.counters.items()},
            "gauges": {name: gauge["value"] for name, gauge in self.gauges.items()},
            "histograms": {
                name: histogram["values"] for name, histogram in self.histograms.items()
            },
            "summaries": {name: summary["values"] for name, summary in self.summaries.items()},
        }

    def start_http_server(self, port: int = 8000, addr: str = "") -> None:
        """
        Start a Prometheus HTTP server to expose metrics.

        Args:
            port: HTTP port
            addr: HTTP address
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus_client not available, cannot start HTTP server")
            return

        prometheus_client.start_http_server(port, addr)
        logger.info(f"Started Prometheus HTTP server on port {port}")


# Global metrics registry
_METRICS_REGISTRY = MetricsRegistry()


def get_metrics_registry() -> MetricsRegistry:
    """
    Get the global metrics registry.

    Returns:
        Global metrics registry
    """
    return _METRICS_REGISTRY


def track_time(metric_name: str, labels: Optional[Dict[str, str]] = None) -> Callable[[F], F]:
    """
    Decorator to track function execution time.

    Args:
        metric_name: Name of the metric to track time
        labels: Labels for the metric

    Returns:
        Decorated function
    """
    registry = get_metrics_registry()
    # Ensure histogram exists
    if metric_name not in registry.histograms:
        registry.register_histogram(
            f"{metric_name}_seconds",
            f"Time spent in {metric_name}",
            list(labels.keys()) if labels else None,
        )

    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_time = time.time() - start_time
                registry.observe_histogram(
                    f"{metric_name}_seconds",
                    elapsed_time,
                    labels,
                )

        return cast(F, wrapper)

    return decorator


def track_calls(metric_name: str, labels: Optional[Dict[str, str]] = None) -> Callable[[F], F]:
    """
    Decorator to track function calls.

    Args:
        metric_name: Name of the metric to track calls
        labels: Labels for the metric

    Returns:
        Decorated function
    """
    registry = get_metrics_registry()
    # Ensure counter exists
    if metric_name not in registry.counters:
        registry.register_counter(
            f"{metric_name}_calls",
            f"Number of calls to {metric_name}",
            list(labels.keys()) if labels else None,
        )

    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            finally:
                registry.inc_counter(
                    f"{metric_name}_calls",
                    1.0,
                    labels,
                )

        return cast(F, wrapper)

    return decorator


def track_errors(metric_name: str, labels: Optional[Dict[str, str]] = None) -> Callable[[F], F]:
    """
    Decorator to track function errors.

    Args:
        metric_name: Name of the metric to track errors
        labels: Labels for the metric

    Returns:
        Decorated function
    """
    registry = get_metrics_registry()
    # Ensure counter exists
    if metric_name not in registry.counters:
        registry.register_counter(
            f"{metric_name}_errors",
            f"Number of errors in {metric_name}",
            list(labels.keys()) if labels else None,
        )

    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception:
                registry.inc_counter(
                    f"{metric_name}_errors",
                    1.0,
                    labels,
                )
                raise

        return cast(F, wrapper)

    return decorator


def setup_default_metrics() -> None:
    """Set up default metrics for WDBX."""
    registry = get_metrics_registry()

    # System metrics
    registry.register_gauge(
        "memory_usage_bytes",
        "Memory usage in bytes",
    )

    registry.register_gauge(
        "cpu_usage_percent",
        "CPU usage in percent",
    )

    # Vector metrics
    registry.register_counter(
        "vectors_created_total",
        "Total number of vectors created",
    )

    registry.register_gauge(
        "vectors_in_memory",
        "Number of vectors in memory",
    )

    registry.register_counter(
        "vectors_saved_total",
        "Total number of vectors saved",
    )

    # Block metrics
    registry.register_counter(
        "blocks_created_total",
        "Total number of blocks created",
    )

    registry.register_gauge(
        "blocks_in_memory",
        "Number of blocks in memory",
    )

    registry.register_counter(
        "blocks_saved_total",
        "Total number of blocks saved",
    )

    # Search metrics
    registry.register_counter(
        "searches_total",
        "Total number of searches",
        ["type"],
    )

    registry.register_histogram(
        "search_time_seconds",
        "Time spent on searches",
        ["type"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    )

    # API metrics
    registry.register_counter(
        "api_requests_total",
        "Total number of API requests",
        ["method", "endpoint", "status"],
    )

    registry.register_histogram(
        "api_request_duration_seconds",
        "API request duration in seconds",
        ["method", "endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    )

    # Memory optimization metrics
    registry.register_counter(
        "memory_optimizations_total",
        "Total number of memory optimizations",
    )

    registry.register_gauge(
        "memory_before_optimization_bytes",
        "Memory usage before optimization in bytes",
    )

    registry.register_gauge(
        "memory_after_optimization_bytes",
        "Memory usage after optimization in bytes",
    )

    logger.info("Default metrics registered")


def update_memory_metrics() -> None:
    """Update memory usage metrics."""
    registry = get_metrics_registry()

    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        registry.set_gauge("memory_usage_bytes", memory_info.rss)

        cpu_percent = process.cpu_percent(interval=0.1)
        registry.set_gauge("cpu_usage_percent", cpu_percent)
    except ImportError:
        logger.warning("psutil not available, cannot update memory metrics")
        return
    except Exception as e:
        logger.error(f"Error updating memory metrics: {e}")


def track_vector_operations(wdbx_core: Any) -> None:
    """
    Track vector and block operations by monkeypatching the WDBX core.

    Args:
        wdbx_core: WDBX core instance
    """
    registry = get_metrics_registry()

    # Store original methods
    original_create_vector = wdbx_core.create_vector
    original_save_vector = wdbx_core.save_vector
    original_create_block = wdbx_core.create_block
    original_save_block = wdbx_core.save_block
    original_find_similar_vectors = wdbx_core.find_similar_vectors
    original_search_blocks = wdbx_core.search_blocks
    original_optimize_memory = wdbx_core.optimize_memory

    # Patch create_vector
    def patched_create_vector(*args: Any, **kwargs: Any) -> Any:
        result = original_create_vector(*args, **kwargs)
        registry.inc_counter("vectors_created_total")
        registry.set_gauge("vectors_in_memory", len(wdbx_core.vectors))
        return result

    # Patch save_vector
    def patched_save_vector(*args: Any, **kwargs: Any) -> Any:
        result = original_save_vector(*args, **kwargs)
        registry.inc_counter("vectors_saved_total")
        return result

    # Patch create_block
    def patched_create_block(*args: Any, **kwargs: Any) -> Any:
        result = original_create_block(*args, **kwargs)
        registry.inc_counter("blocks_created_total")
        registry.set_gauge("blocks_in_memory", len(wdbx_core.blocks))
        return result

    # Patch save_block
    def patched_save_block(*args: Any, **kwargs: Any) -> Any:
        result = original_save_block(*args, **kwargs)
        registry.inc_counter("blocks_saved_total")
        return result

    # Patch find_similar_vectors
    def patched_find_similar_vectors(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = original_find_similar_vectors(*args, **kwargs)
            registry.inc_counter("searches_total", labels={"type": "vector"})
            return result
        finally:
            elapsed_time = time.time() - start_time
            registry.observe_histogram(
                "search_time_seconds",
                elapsed_time,
                {"type": "vector"},
            )

    # Patch search_blocks
    def patched_search_blocks(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = original_search_blocks(*args, **kwargs)
            registry.inc_counter("searches_total", labels={"type": "block"})
            return result
        finally:
            elapsed_time = time.time() - start_time
            registry.observe_histogram(
                "search_time_seconds",
                elapsed_time,
                {"type": "block"},
            )

    # Patch optimize_memory
    def patched_optimize_memory(*args: Any, **kwargs: Any) -> Any:
        # Get memory usage before optimization
        try:
            import psutil

            process = psutil.Process()
            memory_before = process.memory_info().rss
            registry.set_gauge("memory_before_optimization_bytes", memory_before)
        except ImportError:
            memory_before = 0

        # Call original method
        result = original_optimize_memory(*args, **kwargs)
        registry.inc_counter("memory_optimizations_total")

        # Get memory usage after optimization
        try:
            import psutil

            process = psutil.Process()
            memory_after = process.memory_info().rss
            registry.set_gauge("memory_after_optimization_bytes", memory_after)

            # Log memory change
            memory_diff = memory_before - memory_after
            logger.info(
                f"Memory optimization reduced memory usage by {memory_diff / 1024 / 1024:.2f} MB"
            )
        except ImportError:
            pass

        return result

    # Apply patches
    wdbx_core.create_vector = patched_create_vector
    wdbx_core.save_vector = patched_save_vector
    wdbx_core.create_block = patched_create_block
    wdbx_core.save_block = patched_save_block
    wdbx_core.find_similar_vectors = patched_find_similar_vectors
    wdbx_core.search_blocks = patched_search_blocks
    wdbx_core.optimize_memory = patched_optimize_memory

    logger.info("Vector and block operations tracking enabled")


def enable_metrics_server(port: int = 8000) -> None:
    """
    Enable Prometheus metrics server.

    Args:
        port: HTTP port for Prometheus metrics
    """
    registry = get_metrics_registry()

    # Set up default metrics
    setup_default_metrics()

    # Start HTTP server
    registry.start_http_server(port)

    logger.info(f"Metrics server enabled on port {port}")


def disable_metrics() -> None:
    """Disable metrics collection."""
    registry = get_metrics_registry()
    registry.enabled = False
    logger.info("Metrics collection disabled")
