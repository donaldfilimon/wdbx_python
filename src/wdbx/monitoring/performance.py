# wdbx/performance.py
import gc
import statistics
import threading
import time
import tracemalloc
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.constants import logger

# Try to import ML components for backend monitoring
try:
    from ..ml import JAX_AVAILABLE, TORCH_AVAILABLE, get_ml_backend

    ML_MONITORING_AVAILABLE = True
except ImportError:
    ML_MONITORING_AVAILABLE = False
    JAX_AVAILABLE = False
    TORCH_AVAILABLE = False
    logger.warning("ML module not available for performance monitoring")


class PerformanceAnalyzer:
    """
    Analyzes and reports on system performance metrics including latency,
    throughput, and resource utilization.
    """

    def __init__(self, wdbx: Any) -> None:
        """Initialize the PerformanceAnalyzer."""
        self.wdbx = wdbx
        self.latency_samples: Dict[str, List[float]] = {}
        self.throughput_samples: Dict[str, List[Tuple[float, int]]] = {}
        self.resource_samples: Dict[str, List[float]] = {}
        self.start_time = time.time()
        self.component_latencies = {"api": 0.01, "model": 0.1, "db": 0.03, "moderation": 0.02}
        self.error_counts: Dict[str, int] = {}

        # Initialize ML monitoring if available
        self.ml_backend = get_ml_backend() if ML_MONITORING_AVAILABLE else None
        if self.ml_backend:
            self.component_latencies["ml_backend"] = 0.005
            self.ml_latencies: Dict[str, List[float]] = {}

    def measure_latency(self, operation: str, *args, **kwargs) -> float:
        """
        Measures the latency of a single operation and records it.

        Args:
            operation: The name of the operation to measure
            *args: Arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation

        Returns:
            float: The measured latency in seconds

        Raises:
            ValueError: If the operation doesn't exist
        """
        start = time.time()
        try:
            if hasattr(self.wdbx, operation):
                getattr(self.wdbx, operation)(*args, **kwargs)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        except Exception as e:
            self.error_counts[operation] = self.error_counts.get(operation, 0) + 1
            logger.error(f"Error in operation {operation}: {str(e)}")
            raise
        finally:
            latency = time.time() - start
            if operation not in self.latency_samples:
                self.latency_samples[operation] = []
            self.latency_samples[operation].append(latency)

        return latency

    def measure_throughput(self, operation: str, count: int, *args, **kwargs) -> float:
        """
        Measures the throughput of an operation over multiple calls.

        Args:
            operation: The name of the operation to measure
            count: Number of times to execute the operation
            *args: Arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation

        Returns:
            float: The measured throughput in operations per second

        Raises:
            ValueError: If the operation doesn't exist
        """
        if count <= 0:
            return 0.0

        start = time.time()
        success_count = 0

        for _ in range(count):
            try:
                if hasattr(self.wdbx, operation):
                    getattr(self.wdbx, operation)(*args, **kwargs)
                    success_count += 1
                else:
                    raise ValueError(f"Unknown operation: {operation}")
            except Exception as e:
                self.error_counts[operation] = self.error_counts.get(operation, 0) + 1
                logger.error(f"Error in throughput test for {operation}: {str(e)}")

        duration = time.time() - start
        throughput = success_count / duration if duration > 0 else 0

        if operation not in self.throughput_samples:
            self.throughput_samples[operation] = []
        self.throughput_samples[operation].append((duration, success_count))

        return throughput

    def performance_monitor(self, operation_name: Optional[str] = None) -> Callable:
        """
        Decorator to monitor performance of functions.

        Args:
            operation_name: Optional custom name for the operation

        Returns:
            Callable: Decorated function that measures performance
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                op_name = operation_name or func.__name__
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    self.error_counts[op_name] = self.error_counts.get(op_name, 0) + 1
                    logger.error(f"Error in {op_name}: {str(e)}")
                    raise
                finally:
                    latency = time.time() - start
                    if op_name not in self.latency_samples:
                        self.latency_samples[op_name] = []
                    self.latency_samples[op_name].append(latency)

            return wrapper

        return decorator

    def update_component_latency(self, component: str, latency: float) -> None:
        """
        Updates the expected latency for a system component.

        Args:
            component: The component name ('api', 'model', 'db', 'moderation', etc.)
            latency: The new latency value in seconds
        """
        if component in self.component_latencies:
            self.component_latencies[component] = latency
        else:
            logger.warning(f"Unknown component: {component}, adding anyway")
            self.component_latencies[component] = latency

    def calculate_system_latency(self) -> float:
        """
        Calculates the expected end-to-end system latency based on component latencies.

        Returns:
            float: The total system latency in seconds
        """
        return sum(self.component_latencies.values())

    def calculate_throughput(self, num_requests: int) -> float:
        """
        Calculates the theoretical system throughput based on latency.

        Args:
            num_requests: Number of requests to process

        Returns:
            float: Estimated throughput in requests per second
        """
        l_total = self.calculate_system_latency()
        return num_requests / l_total if l_total > 0 else 0

    def calculate_scaling_throughput(
        self, base_throughput: float, base_gpus: int, scaled_gpus: int
    ) -> float:
        """
        Projects throughput when scaling GPU resources.

        Args:
            base_throughput: Current throughput with base_gpus
            base_gpus: Current number of GPUs
            scaled_gpus: Target number of GPUs

        Returns:
            float: Projected throughput with scaled resources
        """
        if base_gpus <= 0:
            logger.warning("Base GPU count must be positive, using 1")
            base_gpus = 1

        # Apply Amdahl's Law with a parallelization factor
        parallelizable_fraction = 0.8  # Assume 80% of work can be parallelized
        serial_fraction = 1 - parallelizable_fraction

        speedup = 1 / (serial_fraction + parallelizable_fraction / scaled_gpus * base_gpus)
        return base_throughput * speedup

    def get_percentile_latency(self, operation: str, percentile: float = 95.0) -> Optional[float]:
        """
        Calculates the specified percentile latency for an operation.

        Args:
            operation: The operation name
            percentile: The percentile to calculate (0-100)

        Returns:
            Optional[float]: The percentile latency or None if no samples
        """
        if operation in self.latency_samples and self.latency_samples[operation]:
            return statistics.quantiles(self.latency_samples[operation], n=100, method="inclusive")[
                int(percentile) - 1
            ]
        return None

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Generates a comprehensive performance metrics report.

        Returns:
            Dict[str, Any]: Performance metrics including latency, throughput, and errors
        """
        # Calculate statistics for latency
        latency_stats = {}
        for op, samples in self.latency_samples.items():
            if samples:
                latency_stats[op] = {
                    "mean": statistics.mean(samples),
                    "median": statistics.median(samples),
                    "min": min(samples),
                    "max": max(samples),
                    "p95": (
                        statistics.quantiles(samples, n=100, method="inclusive")[94]
                        if len(samples) >= 5
                        else None
                    ),
                    "p99": (
                        statistics.quantiles(samples, n=100, method="inclusive")[98]
                        if len(samples) >= 5
                        else None
                    ),
                    "samples": len(samples),
                }

        # Calculate statistics for throughput
        throughput_stats = {}
        for op, samples in self.throughput_samples.items():
            if samples:
                total_duration = sum(dur for dur, _ in samples)
                total_reqs = sum(req for _, req in samples)
                if total_duration > 0:
                    throughput_stats[op] = {
                        "mean": total_reqs / total_duration,
                        "median": statistics.median(samples) if len(samples) > 1 else samples[0],
                        "min": min(samples),
                        "max": max(samples),
                        "samples": len(samples),
                    }

        return {
            "latency": latency_stats,
            "throughput": throughput_stats,
            "error_counts": self.error_counts,
            "system_latency": self.calculate_system_latency(),
            "system_throughput": self.calculate_throughput(100),
            "component_latencies": self.component_latencies,
            "uptime": time.time() - self.start_time,
        }

    def record_latency(self, operation: str, latency: float) -> None:
        """
        Record latency for a specific operation.

        Args:
            operation: Name of the operation
            latency: Latency value in seconds
        """
        if operation not in self.latency_samples:
            self.latency_samples[operation] = []
        self.latency_samples[operation].append(latency)

    def record_throughput(self, operation: str, num_requests: int, duration: float) -> None:
        """
        Record throughput for a specific operation.

        Args:
            operation: Name of the operation
            num_requests: Number of requests processed
            duration: Time duration in seconds
        """
        if operation not in self.throughput_samples:
            self.throughput_samples[operation] = []
        self.throughput_samples[operation].append((duration, num_requests))

    def record_resource_usage(self, resource: str, usage: float) -> None:
        """
        Record resource usage.

        Args:
            resource: Name of the resource (e.g., "cpu", "memory")
            usage: Usage value (e.g., percentage)
        """
        if resource not in self.resource_samples:
            self.resource_samples[resource] = []
        self.resource_samples[resource].append(usage)

    def calculate_average_latency(self, operation: str) -> float:
        """
        Calculate average latency for a specific operation.

        Args:
            operation: Name of the operation

        Returns:
            float: Average latency in seconds
        """
        if operation not in self.latency_samples or not self.latency_samples[operation]:
            return 0.0
        return sum(self.latency_samples[operation]) / len(self.latency_samples[operation])

    def calculate_average_throughput(self, operation: str) -> float:
        """
        Calculate average throughput for a specific operation.

        Args:
            operation: Name of the operation

        Returns:
            float: Average throughput in requests per second
        """
        if operation not in self.throughput_samples or not self.throughput_samples[operation]:
            return 0.0

        total_duration = sum(dur for dur, _ in self.throughput_samples[operation])
        total_requests = sum(req for _, req in self.throughput_samples[operation])

        if total_duration == 0:
            return 0.0

        return total_requests / total_duration

    def calculate_system_latency(self) -> float:
        """
        Calculate overall system latency (average across all operations).

        Returns:
            float: Average system latency in seconds
        """
        all_latencies = []
        for samples in self.latency_samples.values():
            all_latencies.extend(samples)

        if not all_latencies:
            return 0.0

        return sum(all_latencies) / len(all_latencies)

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a summary performance report.

        Returns:
            Dict[str, Any]: Dictionary containing performance metrics
        """
        report = {}
        for op, samples in self.latency_samples.items():
            if samples:
                report[f"{op}_latency_avg"] = sum(samples) / len(samples)
                report[f"{op}_latency_p95"] = statistics.quantiles(
                    samples, n=100, method="inclusive"
                )[94]
                report[f"{op}_latency_max"] = max(samples)

        for op, samples in self.throughput_samples.items():
            if samples:
                total_duration = sum(dur for dur, _ in samples)
                total_reqs = sum(req for _, req in samples)
                if total_duration > 0:
                    report[f"{op}_throughput_avg"] = total_reqs / total_duration

        for res, samples in self.resource_samples.items():
            if samples:
                report[f"{res}_usage_avg"] = sum(samples) / len(samples)
                report[f"{res}_usage_max"] = max(samples)

        report["system_latency_avg"] = self.calculate_system_latency()
        report["uptime"] = time.time() - self.start_time
        return report

    def identify_bottlenecks(self, top_k: int = 5) -> Dict[str, float]:
        """
        Identify the top K operations contributing most to latency.

        Args:
            top_k: Number of top operations to identify

        Returns:
            Dict[str, float]: Dictionary of operation names and their latencies
        """
        # Implementation of identify_bottlenecks method
        # This is a placeholder and should be implemented based on your specific requirements
        return {}

    def measure_ml_operation(
        self, operation_type: str, input_size: int, iterations: int = 1
    ) -> float:
        """
        Measure performance of ML operations.

        Args:
            operation_type: Type of ML operation ('similarity', 'normalize', etc.)
            input_size: Size of input data (e.g., number of vectors)
            iterations: Number of iterations to run

        Returns:
            Average latency per operation in seconds
        """
        if not ML_MONITORING_AVAILABLE or not self.ml_backend:
            logger.warning("ML backend not available for performance measurement")
            return 0.0

        try:
            # Generate test data based on operation type
            if operation_type in ["similarity", "normalize", "batch_similarity"]:
                # For vector operations, generate random vectors
                test_data = np.random.randn(input_size, 128).astype(np.float32)

                # Measure different operations
                if operation_type == "similarity":
                    # Measure cosine similarity between pairs of vectors
                    start = time.time()
                    for _ in range(iterations):
                        for i in range(min(100, input_size)):
                            idx1 = i
                            idx2 = (i + 1) % input_size
                            _ = self.ml_backend.cosine_similarity(test_data[idx1], test_data[idx2])
                    latency = (time.time() - start) / (min(100, input_size) * iterations)

                elif operation_type == "normalize":
                    # Measure vector normalization
                    start = time.time()
                    for _ in range(iterations):
                        for i in range(min(100, input_size)):
                            _ = self.ml_backend.normalize(test_data[i])
                    latency = (time.time() - start) / (min(100, input_size) * iterations)

                elif operation_type == "batch_similarity":
                    # Measure batch similarity computation
                    start = time.time()
                    for _ in range(iterations):
                        query = test_data[0]
                        _ = self.ml_backend.batch_cosine_similarity(query, test_data)
                    latency = (time.time() - start) / iterations
            else:
                logger.warning(f"Unknown ML operation type: {operation_type}")
                return 0.0

            # Store latency sample
            if operation_type not in self.ml_latencies:
                self.ml_latencies[operation_type] = []
            self.ml_latencies[operation_type].append(latency)

            # Update component latency for this operation type
            self.component_latencies[f"ml_{operation_type}"] = latency

            return latency

        except Exception as e:
            logger.error(f"Error measuring ML operation {operation_type}: {e}")
            return 0.0


class Profiler:
    """Context manager for profiling code execution."""

    def __init__(self, name: str, monitor: Optional["PerformanceMonitor"] = None):
        """Initialize the profiler."""
        self.name = name
        self.monitor = monitor
        self.start_time = None
        self.start_memory = None

    def __enter__(self):
        """Start profiling."""
        self.start_time = time.time()

        # Record memory usage if tracemalloc is enabled
        if tracemalloc.is_tracing():
            self.start_memory = tracemalloc.get_traced_memory()[0]

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End profiling and record stats."""
        duration = time.time() - self.start_time
        memory_change = None

        # Calculate memory change if tracemalloc is enabled
        if tracemalloc.is_tracing() and self.start_memory is not None:
            current_memory = tracemalloc.get_traced_memory()[0]
            memory_change = current_memory - self.start_memory

        # Record event if monitor is available
        if self.monitor:
            self.monitor.record_event(self.name, duration, memory_change)

            # If we're profiling an ML operation, add extra attributes
            if self.name.startswith("ml_") and ML_MONITORING_AVAILABLE:
                self.monitor.ml_operation_count += 1

                # Extract operation type from name (assuming format: ml_operation_type)
                parts = self.name.split("_", 2)
                if len(parts) >= 2:
                    op_type = parts[1]
                    self.monitor.ml_operation_times.setdefault(op_type, []).append(duration)

        return False  # Don't suppress exceptions


class PerformanceMonitor:
    """
    Monitors and records performance statistics for the application.
    This expanded version includes ML operation monitoring.
    """

    def __init__(self, max_history: int = 1000, enable_ml_monitoring: bool = True):
        """
        Initialize the performance monitor.

        Args:
            max_history: Maximum number of events to keep in history
            enable_ml_monitoring: Whether to enable ML-specific monitoring
        """
        self.max_history = max_history
        self.events: Dict[str, List[Dict[str, Any]]] = {}
        self.stats: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()

        # Enable ML monitoring if ML module is available
        self.enable_ml_monitoring = enable_ml_monitoring and ML_MONITORING_AVAILABLE
        self.ml_backend = get_ml_backend() if self.enable_ml_monitoring else None
        self.ml_operation_count = 0
        self.ml_operation_times: Dict[str, List[float]] = {}
        self.ml_memory_usage: Dict[str, List[int]] = {}

        # Track peak memory usage
        self.peak_memory = 0

        # Initialize memory tracing if enabled in environment
        self.memory_tracing_enabled = False

    def start_memory_tracing(self) -> None:
        """Start tracking memory allocations."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.memory_tracing_enabled = True
            logger.info("Memory tracing started")

    def stop_memory_tracing(self) -> None:
        """Stop tracking memory allocations."""
        if tracemalloc.is_tracing() and self.memory_tracing_enabled:
            tracemalloc.stop()
            self.memory_tracing_enabled = False
            logger.info("Memory tracing stopped")

    def profile(self, name: str) -> Profiler:
        """Get a profiler for the given operation."""
        return Profiler(name, self)

    def record_event(
        self, event_name: str, duration: float, memory_change: Optional[int] = None
    ) -> None:
        """
        Record a performance event.

        Args:
            event_name: Name of the event/operation
            duration: Duration of the event in seconds
            memory_change: Optional memory change during the event in bytes
        """
        with self.lock:
            if event_name not in self.events:
                self.events[event_name] = []

            # Create event record
            event = {
                "timestamp": time.time(),
                "duration": duration,
                "memory_change": memory_change,
            }

            # Add ML backend info if applicable
            if self.enable_ml_monitoring and event_name.startswith("ml_"):
                event["ml_backend"] = (
                    self.ml_backend.selected_backend if self.ml_backend else "unknown"
                )

            # Add to event history, maintaining max_history limit
            self.events[event_name].append(event)
            if len(self.events[event_name]) > self.max_history:
                self.events[event_name] = self.events[event_name][-self.max_history :]

            # Update stats
            self._update_stats(event_name)

    def record_ml_memory_usage(self, backend_type: str, device: str, memory_bytes: int) -> None:
        """
        Record ML memory usage.

        Args:
            backend_type: ML backend type (jax, torch, etc.)
            device: Device identifier (cpu, gpu:0, etc.)
            memory_bytes: Memory usage in bytes
        """
        if not self.enable_ml_monitoring:
            return

        with self.lock:
            key = f"{backend_type}_{device}"
            if key not in self.ml_memory_usage:
                self.ml_memory_usage[key] = []

            self.ml_memory_usage[key].append(memory_bytes)

            # Keep only recent history
            if len(self.ml_memory_usage[key]) > self.max_history:
                self.ml_memory_usage[key] = self.ml_memory_usage[key][-self.max_history :]

            # Update peak memory
            self.peak_memory = max(self.peak_memory, memory_bytes)

    def _update_stats(self, event_name: str) -> None:
        """Update statistics for the given event."""
        events = self.events[event_name]
        if not events:
            return

        durations = [e["duration"] for e in events]
        memory_changes = [e["memory_change"] for e in events if e["memory_change"] is not None]

        # Calculate statistics
        stats = {
            "count": len(events),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "avg_duration": sum(durations) / len(durations),
            "p95_duration": (
                sorted(durations)[int(len(durations) * 0.95)] if len(durations) >= 20 else None
            ),
            "last_duration": durations[-1],
            "total_duration": sum(durations),
        }

        # Add memory stats if available
        if memory_changes:
            stats["avg_memory_change"] = sum(memory_changes) / len(memory_changes)
            stats["total_memory_change"] = sum(memory_changes)
            stats["max_memory_change"] = max(memory_changes)

        # Add ML backend info if applicable
        if self.enable_ml_monitoring and event_name.startswith("ml_"):
            stats["ml_backend"] = self.ml_backend.selected_backend if self.ml_backend else "unknown"

        self.stats[event_name] = stats

    def get_stats(self, event_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get performance statistics.

        Args:
            event_name: Optional name of event to get stats for

        Returns:
            Dictionary of statistics by event name
        """
        with self.lock:
            if event_name:
                if event_name in self.stats:
                    return {event_name: self.stats[event_name]}
                return {}

            # Return all stats with a copy to avoid threading issues
            stats_copy = {k: v.copy() for k, v in self.stats.items()}

            # Add ML-specific statistics
            if self.enable_ml_monitoring:
                # Add general ML stats
                ml_stats = {
                    "ml_operations": {
                        "count": self.ml_operation_count,
                        "backend": (
                            self.ml_backend.selected_backend if self.ml_backend else "unknown"
                        ),
                    }
                }

                # Add per-operation type stats
                for op_type, times in self.ml_operation_times.items():
                    if times:
                        ml_stats[f"ml_{op_type}"] = {
                            "count": len(times),
                            "avg_duration": sum(times) / len(times),
                            "min_duration": min(times),
                            "max_duration": max(times),
                            "p95_duration": (
                                sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else None
                            ),
                        }

                # Add memory usage stats
                for key, memory_samples in self.ml_memory_usage.items():
                    if memory_samples:
                        ml_stats[f"memory_{key}"] = {
                            "current": memory_samples[-1],
                            "avg": sum(memory_samples) / len(memory_samples),
                            "max": max(memory_samples),
                            "min": min(memory_samples),
                        }

                # Update stats dictionary
                stats_copy.update(ml_stats)

            return stats_copy

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get detailed memory statistics.

        Returns:
            Dictionary of memory statistics
        """
        stats = {
            "process": {
                "peak_bytes": self.peak_memory,
            }
        }

        # Add tracemalloc stats if enabled
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            stats["tracemalloc"] = {
                "current_bytes": current,
                "peak_bytes": peak,
            }

            # Get top memory consumers
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")

            # Add top 5 memory consumers
            stats["top_consumers"] = [
                {
                    "file": str(stat.traceback.frame.filename),
                    "line": stat.traceback.frame.lineno,
                    "size_bytes": stat.size,
                }
                for stat in top_stats[:5]
            ]

        # Add ML-specific memory stats
        if self.enable_ml_monitoring:
            # Add backend memory usage
            stats["ml"] = {}

            for key, memory_samples in self.ml_memory_usage.items():
                if memory_samples:
                    stats["ml"][key] = {
                        "current_bytes": memory_samples[-1],
                        "peak_bytes": max(memory_samples),
                        "samples": len(memory_samples),
                    }

            # Try to get backend-specific memory stats
            if self.ml_backend:
                if self.ml_backend.selected_backend == "torch" and TORCH_AVAILABLE:
                    try:
                        import torch

                        if torch.cuda.is_available():
                            for i in range(torch.cuda.device_count()):
                                stats["ml"][f"torch_cuda_{i}"] = {
                                    "allocated_bytes": torch.cuda.memory_allocated(i),
                                    "reserved_bytes": torch.cuda.memory_reserved(i),
                                    "max_allocated_bytes": torch.cuda.max_memory_allocated(i),
                                }
                    except Exception as e:
                        logger.debug(f"Error getting PyTorch memory stats: {e}")

        return stats

    def clear_stats(self, event_name: Optional[str] = None) -> None:
        """
        Clear performance statistics.

        Args:
            event_name: Optional name of event to clear stats for
        """
        with self.lock:
            if event_name:
                if event_name in self.events:
                    del self.events[event_name]
                if event_name in self.stats:
                    del self.stats[event_name]
            else:
                self.events.clear()
                self.stats.clear()

                # Reset ML stats too
                if self.enable_ml_monitoring:
                    self.ml_operation_count = 0
                    self.ml_operation_times.clear()
                    self.ml_memory_usage.clear()
                    self.peak_memory = 0

    def perform_gc(self) -> Dict[str, Any]:
        """
        Perform garbage collection and return statistics.

        Returns:
            Dictionary with garbage collection statistics
        """
        # Record memory before GC
        if tracemalloc.is_tracing():
            memory_before = tracemalloc.get_traced_memory()[0]

        # Perform garbage collection
        gc.collect()

        # Record memory after GC
        if tracemalloc.is_tracing():
            memory_after = tracemalloc.get_traced_memory()[0]
            memory_freed = memory_before - memory_after
        else:
            memory_freed = 0

        return {
            "gc_performed": True,
            "memory_freed_bytes": memory_freed,
        }

    def get_ml_backend_stats(self) -> Dict[str, Any]:
        """
        Get statistics specifically for ML backend operations.

        Returns:
            Dictionary with ML backend statistics
        """
        if not self.enable_ml_monitoring:
            return {"ml_monitoring_enabled": False}

        stats = {
            "backend": self.ml_backend.selected_backend if self.ml_backend else "unknown",
            "jax_available": JAX_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "operation_count": self.ml_operation_count,
        }

        # Add per-operation stats
        operation_stats = {}
        for op_type, times in self.ml_operation_times.items():
            if times:
                operation_stats[op_type] = {
                    "count": len(times),
                    "avg_duration": sum(times) / len(times),
                    "min_duration": min(times),
                    "max_duration": max(times),
                }

        stats["operations"] = operation_stats

        return stats


# Global instance (optional, consider dependency injection)
# performance_monitor = PerformanceMonitor()
