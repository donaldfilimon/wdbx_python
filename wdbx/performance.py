# wdbx/performance.py
import time
import statistics
from functools import wraps
from typing import Dict, Any, List, Tuple, Callable, Optional, Union
from wdbx.constants import logger

class PerformanceAnalyzer:
    """
    Analyzes and reports on system performance metrics including latency,
    throughput, and resource utilization.
    """
    def __init__(self, wdbx: Any) -> None:
        self.wdbx = wdbx
        self.latency_samples: Dict[str, List[float]] = {}
        self.throughput_samples: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        self.start_time = time.time()
        self.component_latencies = {
            'api': 0.01,
            'model': 0.1,
            'db': 0.03,
            'moderation': 0.02
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
                result = getattr(self.wdbx, operation)(*args, **kwargs)
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
        self.throughput_samples[operation].append(throughput)

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
        L_total = self.calculate_system_latency()
        return num_requests / L_total if L_total > 0 else 0

    def calculate_scaling_throughput(self, base_throughput: float, base_gpus: int, scaled_gpus: int) -> float:
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

        speedup = 1 / (serial_fraction + parallelizable_fraction/scaled_gpus * base_gpus)
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
                                        method='inclusive')[int(percentile)-1]
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
                    "p95": statistics.quantiles(samples, n=100, method='inclusive')[94] if len(samples) >= 5 else None,
                    "p99": statistics.quantiles(samples, n=100, method='inclusive')[98] if len(samples) >= 5 else None,
                    "samples": len(samples)
                }

        # Calculate statistics for throughput
        throughput_stats = {}
        for op, samples in self.throughput_samples.items():
            if samples:
                throughput_stats[op] = {
                    "mean": statistics.mean(samples),
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
