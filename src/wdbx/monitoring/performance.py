# wdbx/performance.py
import statistics
import threading
import time
import tracemalloc
from collections import deque
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.constants import logger


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
        self.component_latencies = {
            "api": 0.01,
            "model": 0.1,
            "db": 0.03,
            "moderation": 0.02
        }

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
                _result = getattr(self.wdbx, operation)(*args, **kwargs)
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
            self,
            base_throughput: float,
            base_gpus: int,
            scaled_gpus: int) -> float:
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
            return statistics.quantiles(self.latency_samples[operation],
                                        n=100,
                                        method="inclusive")[int(percentile) - 1]
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
                    "p95": statistics.quantiles(
                        samples,
                        n=100,
                        method="inclusive")[94] if len(samples) >= 5 else None,
                    "p99": statistics.quantiles(
                        samples,
                        n=100,
                        method="inclusive")[98] if len(samples) >= 5 else None,
                    "samples": len(samples)}

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
                        "samples": len(samples)
                    }

        return {
            "latency": latency_stats,
            "throughput": throughput_stats,
            "error_counts": self.error_counts,
            "system_latency": self.calculate_system_latency(),
            "system_throughput": self.calculate_throughput(100),
            "component_latencies": self.component_latencies,
            "uptime": time.time() - self.start_time
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
                report[f"{op}_latency_p95"] = statistics.quantiles(samples, n=100, method="inclusive")[94]
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


class Profiler:
    """Context manager for profiling code blocks."""
    def __init__(self, name: str, monitor: Optional["PerformanceMonitor"] = None):
        self.name = name
        self.monitor = monitor
        self.start_time = 0.0
        self.start_mem = 0

    def __enter__(self):
        if tracemalloc.is_tracing():
            self.start_mem = tracemalloc.get_traced_memory()[0]
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        mem_diff = 0
        if tracemalloc.is_tracing():
            current_mem = tracemalloc.get_traced_memory()[0]
            mem_diff = current_mem - self.start_mem

        if self.monitor:
            self.monitor.record_event(self.name, duration, mem_diff)
        else:
            # Basic logging if no monitor is attached
            log_msg = f"Profiled '{self.name}': Duration={duration*1000:.2f}ms"
            if mem_diff != 0:
                log_msg += f", Memory Diff={mem_diff / 1024:.2f}KB"
            logger.debug(log_msg)


class PerformanceMonitor:
    """
    Tracks performance metrics like execution time and memory usage.

    Provides methods to record events and retrieve aggregated statistics.
    """
    def __init__(self, max_history: int = 1000):
        self.events: Dict[str, Deque[Tuple[float, int]]] = {}
        self.max_history = max_history
        self.lock = threading.Lock() # Protect access to events dictionary
        self.is_tracing_memory = False
        logger.info("PerformanceMonitor initialized.")

    def start_memory_tracing(self) -> None:
        """Start tracing memory allocations using tracemalloc."""
        if not self.is_tracing_memory:
            tracemalloc.start()
            self.is_tracing_memory = True
            logger.info("Memory tracing started.")
        else:
            logger.warning("Memory tracing is already active.")

    def stop_memory_tracing(self) -> None:
        """Stop tracing memory allocations."""
        if self.is_tracing_memory:
            tracemalloc.stop()
            self.is_tracing_memory = False
            logger.info("Memory tracing stopped.")
        else:
            logger.warning("Memory tracing is not active.")
            
    def profile(self, name: str) -> Profiler:
        """Return a Profiler context manager associated with this monitor."""
        return Profiler(name, self)

    def record_event(self, event_name: str, duration: float, memory_change: Optional[int] = None) -> None:
        """Record a performance event.

        Args:
            event_name: Name of the event (e.g., 'vector_search', 'block_creation').
            duration: Duration of the event in seconds.
            memory_change: Change in memory usage during the event in bytes (optional).
        """
        with self.lock:
            if event_name not in self.events:
                self.events[event_name] = deque(maxlen=self.max_history)
            
            # Use 0 if memory change is None or not tracing
            mem_change = memory_change if memory_change is not None and self.is_tracing_memory else 0
            self.events[event_name].append((duration, mem_change))
            # logger.debug(f"Recorded event '{event_name}': Duration={duration*1000:.2f}ms, MemChange={mem_change}B")

    def get_stats(self, event_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Calculate and return performance statistics.

        Args:
            event_name: If specified, return stats only for this event.
                         Otherwise, return stats for all recorded events.

        Returns:
            A dictionary where keys are event names and values are dictionaries
            containing statistics (count, avg_duration, total_duration, 
            avg_mem_change, total_mem_change).
        """
        stats = {}
        with self.lock:
            event_names_to_process = [event_name] if event_name else list(self.events.keys())
            
            for name in event_names_to_process:
                if name in self.events:
                    event_data = list(self.events[name])
                    if not event_data:
                        stats[name] = {"count": 0, "avg_duration": 0.0, "total_duration": 0.0,
                                     "avg_mem_change": 0.0, "total_mem_change": 0}
                        continue
                        
                    durations = [e[0] for e in event_data]
                    mem_changes = [e[1] for e in event_data]
                    count = len(durations)
                    total_duration = sum(durations)
                    total_mem_change = sum(mem_changes)
                    
                    stats[name] = {
                        "count": count,
                        "avg_duration": total_duration / count if count else 0,
                        "total_duration": total_duration,
                        "min_duration": min(durations) if durations else 0,
                        "max_duration": max(durations) if durations else 0,
                        "avg_mem_change": total_mem_change / count if count else 0,
                        "total_mem_change": total_mem_change,
                        "min_mem_change": min(mem_changes) if mem_changes else 0,
                        "max_mem_change": max(mem_changes) if mem_changes else 0,
                    }
                elif event_name: # Specific event requested but not found
                     stats[name] = {"count": 0, "avg_duration": 0.0, "total_duration": 0.0,
                                     "avg_mem_change": 0.0, "total_mem_change": 0}
                                     
        return stats

    def clear_stats(self, event_name: Optional[str] = None) -> None:
        """Clear performance data for a specific event or all events."""
        with self.lock:
            if event_name:
                if event_name in self.events:
                    self.events[event_name].clear()
                    logger.info(f"Cleared performance stats for event: {event_name}")
                else:
                    logger.warning(f"Cannot clear stats: Event '{event_name}' not found.")
            else:
                self.events.clear()
                logger.info("Cleared all performance stats.")

# Global instance (optional, consider dependency injection)
# performance_monitor = PerformanceMonitor()
