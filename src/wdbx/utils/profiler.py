"""
Performance profiling utilities for WDBX.

This module provides tools for profiling code performance,
including execution time measurement, memory usage tracking,
and function call statistics.
"""

import cProfile
import functools
import io
import os
import pstats
import resource
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TextIO, TypeVar, cast

import numpy as np

from .logging_utils import get_logger

# Initialize logger
logger = get_logger("wdbx.profiler")

# Type variable for generic function
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class ProfilerStats:
    """Statistics collected by the profiler."""

    # Time tracking
    start_time: float = 0.0
    end_time: float = 0.0
    elapsed_time: float = 0.0

    # Memory tracking
    start_memory: int = 0
    peak_memory: int = 0
    end_memory: int = 0
    memory_diff: int = 0

    # Call tracking
    calls: int = 0
    func_name: str = ""

    # Nested stats for child operations
    children: Dict[str, "ProfilerStats"] = field(default_factory=dict)

    # Additional data
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert statistics to a dictionary.

        Returns:
            Dictionary representation of statistics
        """
        result = {
            "time": {
                "start": self.start_time,
                "end": self.end_time,
                "elapsed_ms": self.elapsed_time * 1000,
            },
            "memory": {
                "start_kb": self.start_memory / 1024,
                "peak_kb": self.peak_memory / 1024,
                "end_kb": self.end_memory / 1024,
                "diff_kb": self.memory_diff / 1024,
            },
            "calls": self.calls,
        }

        if self.func_name:
            result["function"] = self.func_name

        if self.children:
            result["children"] = {name: child.to_dict() for name, child in self.children.items()}

        if self.context:
            result["context"] = self.context

        return result

    def log_summary(self, level: str = "info", prefix: str = "") -> None:
        """
        Log a summary of profiling statistics.

        Args:
            level: Logging level to use
            prefix: Prefix to add to log messages
        """
        log_method = getattr(logger, level)

        if self.func_name:
            func_info = f"{self.func_name} "
        else:
            func_info = ""

        log_method(
            f"{prefix}{func_info}executed in {self.elapsed_time * 1000:.2f}ms with "
            f"peak memory {self.peak_memory / 1024:.2f}KB "
            f"(delta: {self.memory_diff / 1024:.2f}KB)"
        )

        for name, child in self.children.items():
            child.log_summary(level, prefix=f"{prefix}  ")


class Profiler:
    """
    Performance profiler for tracking execution time and memory usage.

    This class provides methods for profiling code execution, including
    time measurement, memory usage tracking, and function call statistics.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize the profiler.

        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self.current_stats: Optional[ProfilerStats] = None
        self.stats_stack: List[ProfilerStats] = []
        self.global_stats: Dict[str, ProfilerStats] = {}

        # tracemalloc state
        self.tracemalloc_enabled = False

    def start(self, name: str = "root") -> ProfilerStats:
        """
        Start profiling.

        Args:
            name: Name for this profiling session

        Returns:
            Profiler statistics object
        """
        if not self.enabled:
            return ProfilerStats()

        # Create new stats object
        stats = ProfilerStats()
        stats.start_time = time.time()
        stats.func_name = name

        # Get current memory usage
        stats.start_memory = self._get_memory_usage()
        stats.peak_memory = stats.start_memory

        # Track the stats
        self.current_stats = stats
        self.stats_stack.append(stats)

        # Store in global stats if this is a root measurement
        if len(self.stats_stack) == 1:
            self.global_stats[name] = stats
        # Add as child to parent if this is a nested measurement
        elif len(self.stats_stack) > 1:
            parent = self.stats_stack[-2]
            parent.children[name] = stats

        return stats

    def stop(self) -> ProfilerStats:
        """
        Stop profiling and return statistics.

        Returns:
            Profiler statistics object
        """
        if not self.enabled or not self.stats_stack:
            return ProfilerStats()

        # Get the current stats
        stats = self.stats_stack.pop()
        stats.end_time = time.time()
        stats.elapsed_time = stats.end_time - stats.start_time

        # Get final memory usage
        stats.end_memory = self._get_memory_usage()
        stats.memory_diff = stats.end_memory - stats.start_memory

        # Update current stats pointer
        if self.stats_stack:
            self.current_stats = self.stats_stack[-1]
        else:
            self.current_stats = None

        return stats

    def reset(self) -> None:
        """Reset all profiling data."""
        self.current_stats = None
        self.stats_stack = []
        self.global_stats = {}

        # Stop tracemalloc if it's running
        if self.tracemalloc_enabled:
            tracemalloc.stop()
            self.tracemalloc_enabled = False

    def enable_tracemalloc(self) -> None:
        """Enable detailed memory tracking with tracemalloc."""
        if not self.tracemalloc_enabled:
            tracemalloc.start()
            self.tracemalloc_enabled = True

    def get_tracemalloc_stats(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get detailed memory statistics from tracemalloc.

        Args:
            top_n: Number of top memory consumers to return

        Returns:
            List of memory statistics
        """
        if not self.tracemalloc_enabled:
            return []

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        results = []
        for stat in top_stats[:top_n]:
            frame = stat.traceback[0]
            results.append(
                {
                    "file": frame.filename,
                    "line": frame.lineno,
                    "size": stat.size,
                    "count": stat.count,
                }
            )

        return results

    def _get_memory_usage(self) -> int:
        """
        Get current memory usage.

        Returns:
            Memory usage in bytes
        """
        # Check for tracemalloc first
        if self.tracemalloc_enabled:
            return tracemalloc.get_traced_memory()[0]

        # Fall back to resource module
        try:
            # For Unix-based systems
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except (AttributeError, ImportError):
            # For Windows or if resource module is not available
            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss

    def print_stats(
        self,
        stream: Optional[TextIO] = None,
        sort_by: str = "cumulative",
    ) -> None:
        """
        Print profiling statistics.

        Args:
            stream: Output stream (defaults to stdout)
            sort_by: Sorting criteria for statistics
        """
        if not self.enabled:
            return

        # Print global stats
        for name, stats in self.global_stats.items():
            if stream:
                stream.write(f"=== Profile: {name} ===\n")
                stream.write(f"Time: {stats.elapsed_time * 1000:.2f}ms\n")
                stream.write(f"Memory peak: {stats.peak_memory / 1024:.2f}KB\n")
                stream.write(f"Memory change: {stats.memory_diff / 1024:.2f}KB\n")

                # Print children stats
                if stats.children:
                    stream.write("\nChildren:\n")
                    for child_name, child_stats in stats.children.items():
                        stream.write(
                            f"  {child_name}: {child_stats.elapsed_time * 1000:.2f}ms, "
                            f"{child_stats.memory_diff / 1024:.2f}KB\n"
                        )
            else:
                # Log to the logger
                logger.info(f"=== Profile: {name} ===")
                logger.info(f"Time: {stats.elapsed_time * 1000:.2f}ms")
                logger.info(f"Memory peak: {stats.peak_memory / 1024:.2f}KB")
                logger.info(f"Memory change: {stats.memory_diff / 1024:.2f}KB")

                # Log children stats
                if stats.children:
                    logger.info("Children:")
                    for child_name, child_stats in stats.children.items():
                        logger.info(
                            f"  {child_name}: {child_stats.elapsed_time * 1000:.2f}ms, "
                            f"{child_stats.memory_diff / 1024:.2f}KB"
                        )


# Global profiler instance
_PROFILER = Profiler()


def get_profiler() -> Profiler:
    """
    Get the global profiler instance.

    Returns:
        Global profiler instance
    """
    return _PROFILER


@contextmanager
def profile(name: str = "operation", enabled: bool = True):
    """
    Context manager for profiling a block of code.

    Args:
        name: Name for this profiling session
        enabled: Whether profiling is enabled

    Yields:
        Profiler statistics object
    """
    profiler = get_profiler()

    if not profiler.enabled or not enabled:
        yield ProfilerStats()
        return

    try:
        stats = profiler.start(name)
        yield stats
    finally:
        profiler.stop()


def profile_function(name: Optional[str] = None, enabled: bool = True) -> Callable[[F], F]:
    """
    Decorator for profiling a function.

    Args:
        name: Custom name for the profiling session (defaults to function name)
        enabled: Whether profiling is enabled

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profiler = get_profiler()

            if not profiler.enabled or not enabled:
                return func(*args, **kwargs)

            profile_name = name or func.__name__
            with profile(profile_name):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


@contextmanager
def cprofile_block(
    name: str = "cprofile",
    enabled: bool = True,
    sort_by: str = "cumulative",
    top_n: int = 20,
    print_stats: bool = True,
):
    """
    Context manager for detailed profiling with cProfile.

    Args:
        name: Name for this profiling session
        enabled: Whether profiling is enabled
        sort_by: Sorting criteria for statistics
        top_n: Number of top functions to display
        print_stats: Whether to print statistics after profiling

    Yields:
        cProfile.Profile object
    """
    if not enabled:
        yield None
        return

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        yield profiler
    finally:
        profiler.disable()

        if print_stats:
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
            ps.print_stats(top_n)
            logger.info(f"cProfile results for {name}:\n{s.getvalue()}")


def profile_memory(size_threshold: int = 1024 * 100, enabled: bool = True) -> None:
    """
    Start detailed memory profiling with tracemalloc.

    Args:
        size_threshold: Size threshold for reporting memory allocations (in bytes)
        enabled: Whether profiling is enabled
    """
    if not enabled:
        return

    profiler = get_profiler()
    profiler.enable_tracemalloc()

    # Create a snapshot and filter by size
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    logger.info(f"Top memory allocations (threshold: {size_threshold / 1024:.1f}KB):")
    for stat in top_stats:
        if stat.size < size_threshold:
            continue

        frame = stat.traceback[0]
        logger.info(
            f"{frame.filename}:{frame.lineno}: {stat.size / 1024:.1f}KB, "
            f"{stat.count} allocations"
        )


def profile_numpy_arrays() -> Dict[str, Any]:
    """
    Profile memory usage of NumPy arrays in the current process.

    Returns:
        Dictionary with memory statistics for NumPy arrays
    """
    import gc

    # Collect all NumPy arrays in memory
    arrays = []
    total_size = 0
    type_counts: Dict[str, int] = {}
    type_sizes: Dict[str, int] = {}

    for obj in gc.get_objects():
        if isinstance(obj, np.ndarray):
            array_size = obj.nbytes
            arrays.append((obj, array_size))
            total_size += array_size

            # Count by data type
            dtype_name = str(obj.dtype)
            type_counts[dtype_name] = type_counts.get(dtype_name, 0) + 1
            type_sizes[dtype_name] = type_sizes.get(dtype_name, 0) + array_size

    # Sort arrays by size (largest first)
    arrays.sort(key=lambda x: x[1], reverse=True)

    # Prepare results
    result = {
        "total_arrays": len(arrays),
        "total_size_mb": total_size / (1024 * 1024),
        "type_counts": type_counts,
        "type_sizes_mb": {k: v / (1024 * 1024) for k, v in type_sizes.items()},
        "largest_arrays": [
            {
                "shape": arr[0].shape,
                "dtype": str(arr[0].dtype),
                "size_mb": arr[1] / (1024 * 1024),
                "contiguous": arr[0].flags.c_contiguous or arr[0].flags.f_contiguous,
            }
            for arr, _ in arrays[:10]  # Top 10 largest arrays
        ],
    }

    return result


def print_memory_report(level: str = "info") -> None:
    """
    Print a detailed memory usage report.

    Args:
        level: Logging level to use
    """
    log_method = getattr(logger, level)

    # Get current memory usage
    profiler = get_profiler()
    current_memory = profiler._get_memory_usage()

    log_method(f"Current memory usage: {current_memory / (1024 * 1024):.2f}MB")

    # Get tracemalloc stats if enabled
    if profiler.tracemalloc_enabled:
        top_stats = profiler.get_tracemalloc_stats(top_n=10)
        log_method("Top memory consumers:")
        for stat in top_stats:
            log_method(
                f"  {stat['file']}:{stat['line']}: {stat['size'] / 1024:.1f}KB, "
                f"{stat['count']} allocations"
            )

    # Get NumPy array stats
    try:
        pass

        numpy_stats = profile_numpy_arrays()
        log_method(
            f"NumPy arrays: {numpy_stats['total_arrays']} arrays, "
            f"{numpy_stats['total_size_mb']:.2f}MB total"
        )
        log_method("Top NumPy arrays:")
        for arr in numpy_stats["largest_arrays"][:5]:
            log_method(
                f"  Shape: {arr['shape']}, dtype: {arr['dtype']}, "
                f"size: {arr['size_mb']:.2f}MB, contiguous: {arr['contiguous']}"
            )
    except ImportError:
        pass
