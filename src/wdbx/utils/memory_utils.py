"""
Memory management utilities for WDBX.

This module provides utilities for memory monitoring and optimization.
"""

import gc
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Set

import psutil

from .logging_utils import get_logger

logger = get_logger("WDBX.MemoryUtils")

# Import constants safely - use direct os.environ access to avoid circular imports
MAX_MEMORY_PERCENT = float(os.environ.get("WDBX_MAX_MEMORY_PERCENT", "85.0"))
MEMORY_CHECK_INTERVAL = int(os.environ.get("WDBX_MEMORY_CHECK_INTERVAL", "10"))
MEMORY_OPTIMIZATION_ENABLED = os.environ.get(
    "WDBX_MEMORY_OPTIMIZATION_ENABLED", "true"
).lower() in ("true", "1", "yes", "y")


class MemoryMonitor:
    """
    Monitors memory usage and triggers optimization when thresholds are exceeded.

    Attributes:
        max_memory_percent: Maximum memory usage percentage before optimization
        check_interval: Interval (in refresh cycles) for checking memory usage
        enabled: Whether memory monitoring is enabled
    """

    def __init__(
        self,
        max_memory_percent: float = MAX_MEMORY_PERCENT,
        check_interval: int = MEMORY_CHECK_INTERVAL,
        enabled: bool = MEMORY_OPTIMIZATION_ENABLED,
    ):
        """
        Initialize a MemoryMonitor.

        Args:
            max_memory_percent: Maximum memory usage percentage before optimization
            check_interval: Interval (in refresh cycles) for checking memory usage
            enabled: Whether memory monitoring is enabled
        """
        self.max_memory_percent = max_memory_percent
        self.check_interval = check_interval
        self.enabled = enabled

        self._check_count = 0
        self._monitoring_thread = None
        self._stop_event = threading.Event()
        self._optimization_callbacks: List[Callable[[], None]] = []
        self._registered_objects: Dict[str, Any] = {}

        logger.info(
            f"Initialized memory monitor with max_percent={self.max_memory_percent}, "
            f"interval={self.check_interval}, enabled={self.enabled}"
        )

    def register_optimization_callback(self, callback: Callable[[], None]) -> None:
        """
        Register a callback to be called when memory optimization is triggered.

        Args:
            callback: Function to call for memory optimization
        """
        if callback not in self._optimization_callbacks:
            self._optimization_callbacks.append(callback)
            logger.debug(f"Registered optimization callback: {callback.__name__}")

    def register_object(self, name: str, obj: Any) -> None:
        """
        Register an object for memory optimization.

        The object should have an optimize_memory() method that will be called
        during optimization.

        Args:
            name: Name to identify the object
            obj: Object with optimize_memory method
        """
        self._registered_objects[name] = obj
        logger.debug(f"Registered object for memory optimization: {name}")

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage information.

        Returns:
            Dictionary with memory usage information:
            - percent: Percentage of memory used
            - used: Memory used in bytes
            - available: Memory available in bytes
            - total: Total memory in bytes
        """
        mem = psutil.virtual_memory()
        process = psutil.Process()
        process_mem = process.memory_info()

        return {
            "system": {
                "percent": mem.percent,
                "used": mem.used,
                "available": mem.available,
                "total": mem.total,
            },
            "process": {
                "rss": process_mem.rss,  # Resident Set Size
                "vms": process_mem.vms,  # Virtual Memory Size
                "percent": process_mem.rss / mem.total * 100,
            },
        }

    def should_optimize(self) -> bool:
        """
        Check if memory optimization should be triggered.

        Returns:
            True if memory usage exceeds the threshold, False otherwise
        """
        if not self.enabled:
            return False

        # Only check periodically to avoid overhead
        self._check_count += 1
        if self._check_count < self.check_interval:
            return False

        self._check_count = 0

        # Check memory usage
        mem_usage = self.get_memory_usage()
        system_percent = mem_usage["system"]["percent"]
        process_percent = mem_usage["process"]["percent"]

        # Trigger optimization if either system or process memory is high
        if system_percent > self.max_memory_percent or process_percent > (
            self.max_memory_percent * 0.8
        ):
            logger.warning(
                f"Memory threshold exceeded - System: {system_percent:.1f}%, "
                f"Process: {process_percent:.1f}%"
            )
            return True

        return False

    def optimize_memory(self) -> Dict[str, Any]:
        """
        Optimize memory usage.

        This method:
        1. Records memory usage before optimization
        2. Calls optimize_memory() on all registered objects
        3. Calls all registered optimization callbacks
        4. Runs garbage collection
        5. Records memory usage after optimization

        Returns:
            Dictionary with optimization results:
            - before: Memory usage before optimization
            - after: Memory usage after optimization
            - freed_bytes: Bytes freed by optimization
            - freed_percent: Percentage of memory freed
        """
        if not self.enabled:
            logger.info("Memory optimization is disabled")
            return {"optimized": False, "reason": "disabled"}

        # Get memory usage before optimization
        before = self.get_memory_usage()
        logger.info(
            f"Memory usage before optimization: "
            f"System {before['system']['percent']:.1f}%, "
            f"Process {before['process']['percent']:.1f}% "
            f"({before['process']['rss'] / (1024 * 1024):.1f} MB)"
        )

        # Optimize registered objects
        for name, obj in self._registered_objects.items():
            if hasattr(obj, "optimize_memory") and callable(obj.optimize_memory):
                try:
                    logger.debug(f"Optimizing memory for {name}")
                    obj.optimize_memory()
                except Exception as e:
                    logger.error(f"Error optimizing memory for {name}: {e}")

        # Call all registered optimization callbacks
        for callback in self._optimization_callbacks:
            try:
                logger.debug(f"Calling optimization callback: {callback.__name__}")
                callback()
            except Exception as e:
                logger.error(f"Error in optimization callback {callback.__name__}: {e}")

        # Clear any cached objects
        self._clear_caches()

        # Run garbage collection
        gc.collect()

        # Get memory usage after optimization
        after = self.get_memory_usage()
        freed_bytes = before["process"]["rss"] - after["process"]["rss"]
        freed_percent = before["process"]["percent"] - after["process"]["percent"]

        logger.info(
            f"Memory usage after optimization: "
            f"System {after['system']['percent']:.1f}%, "
            f"Process {after['process']['percent']:.1f}% "
            f"({after['process']['rss'] / (1024 * 1024):.1f} MB), "
            f"freed: {freed_bytes / (1024 * 1024):.1f} MB ({freed_percent:.1f}%)"
        )

        return {
            "before": before,
            "after": after,
            "freed_bytes": freed_bytes,
            "freed_percent": freed_percent,
            "optimized": True,
        }

    def _clear_caches(self) -> None:
        """Clear common Python library caches."""
        # Clear NumPy cache if available
        try:
            import numpy as np

            np.clear_cache()
            logger.debug("Cleared NumPy cache")
        except (ImportError, AttributeError):
            pass

        # Clear PyTorch cache if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared PyTorch CUDA cache")
        except (ImportError, AttributeError):
            pass

        # Clear JAX cache if available
        try:
            import jax

            jax.clear_caches()
            logger.debug("Cleared JAX cache")
        except (ImportError, AttributeError):
            pass

        # Clear memoization caches if available
        try:
            from functools import _lru_cache_wrapper

            # Get all objects with an lru_cache
            for obj in gc.get_objects():
                if isinstance(obj, _lru_cache_wrapper):
                    obj.cache_clear()
            logger.debug("Cleared LRU caches")
        except (ImportError, AttributeError) as e:
            logger.debug(f"Could not clear LRU caches: {e}")

    def start_monitoring(self, interval_seconds: float = 60.0) -> None:
        """
        Start a background thread for periodic memory monitoring.

        Args:
            interval_seconds: Interval in seconds between memory checks
        """
        if not self.enabled:
            logger.info("Memory monitoring is disabled")
            return

        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.info("Memory monitoring is already running")
            return

        # Reset stop event
        self._stop_event.clear()

        # Define monitoring function
        def monitor_memory() -> None:
            logger.info(f"Starting memory monitoring with interval of {interval_seconds} seconds")

            while not self._stop_event.is_set():
                try:
                    mem_usage = self.get_memory_usage()
                    system_percent = mem_usage["system"]["percent"]
                    process_percent = mem_usage["process"]["percent"]

                    if system_percent > self.max_memory_percent or process_percent > (
                        self.max_memory_percent * 0.8
                    ):
                        logger.warning(
                            f"Memory threshold exceeded - System: {system_percent:.1f}%, "
                            f"Process: {process_percent:.1f}%"
                        )
                        self.optimize_memory()
                    else:
                        # Only log at debug level to avoid excessive output
                        logger.debug(
                            f"Memory usage - System: {system_percent:.1f}%, "
                            f"Process: {process_percent:.1f}% "
                            f"({mem_usage['process']['rss'] / (1024 * 1024):.1f} MB)"
                        )
                except Exception as e:
                    logger.error(f"Error in memory monitoring: {e}")

                # Wait for the next check interval or until stopped
                self._stop_event.wait(interval_seconds)

            logger.info("Memory monitoring stopped")

        # Start monitoring thread
        self._monitoring_thread = threading.Thread(
            target=monitor_memory, daemon=True, name="MemoryMonitor"
        )
        self._monitoring_thread.start()
        logger.debug("Memory monitoring thread started")

    def stop_monitoring(self) -> None:
        """Stop the background memory monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.info("Stopping memory monitoring")
            self._stop_event.set()
            self._monitoring_thread.join(timeout=5.0)
            logger.debug("Memory monitoring thread stopped")
        else:
            logger.info("Memory monitoring is not running")


# Singleton instance for reuse
_MEMORY_MONITOR_INSTANCE: Optional[MemoryMonitor] = None


def get_memory_monitor() -> MemoryMonitor:
    """
    Get the memory monitor instance, creating it if necessary.

    Returns:
        MemoryMonitor instance
    """
    global _MEMORY_MONITOR_INSTANCE

    if _MEMORY_MONITOR_INSTANCE is None:
        _MEMORY_MONITOR_INSTANCE = MemoryMonitor()

    return _MEMORY_MONITOR_INSTANCE


def optimize_memory() -> Dict[str, Any]:
    """
    Optimize memory usage across the application.

    This is a convenience function that calls optimize_memory on the
    singleton MemoryMonitor instance.

    Returns:
        Dictionary with optimization results
    """
    monitor = get_memory_monitor()
    return monitor.optimize_memory()


def register_for_memory_optimization(name: str, obj: Any) -> None:
    """
    Register an object for memory optimization.

    This is a convenience function that calls register_object on the
    singleton MemoryMonitor instance.

    Args:
        name: Name to identify the object
        obj: Object with optimize_memory method
    """
    monitor = get_memory_monitor()
    monitor.register_object(name, obj)


def get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage information.

    This is a convenience function that calls get_memory_usage on the
    singleton MemoryMonitor instance.

    Returns:
        Dictionary with memory usage information
    """
    monitor = get_memory_monitor()
    return monitor.get_memory_usage()


def get_object_size(obj: Any) -> int:
    """
    Get the memory size of an object in bytes.

    Args:
        obj: Object to get the size of

    Returns:
        Size of the object in bytes
    """
    import sys

    # For basic types, use sys.getsizeof
    if isinstance(obj, (int, float, str, bool, bytes, type(None))):
        return sys.getsizeof(obj)

    # For NumPy arrays, use nbytes
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.nbytes
    except ImportError:
        pass

    # For PyTorch tensors, use element_size * numel
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.numel()
    except ImportError:
        pass

    # For other objects, try to get a rough estimate using recursion
    seen: Set[int] = set()

    def _get_size(obj: Any) -> int:
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        size = sys.getsizeof(obj)

        if isinstance(obj, dict):
            size += sum(_get_size(k) + _get_size(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            size += sum(_get_size(i) for i in obj)

        return size

    return _get_size(obj)


def list_large_objects(limit: int = 10) -> List[Dict[str, Any]]:
    """
    List the largest objects in memory.

    Args:
        limit: Maximum number of objects to list

    Returns:
        List of dictionaries with object information
    """
    import traceback

    # Get all objects in memory
    gc.collect()
    objects = gc.get_objects()

    # Calculate sizes
    object_info = []
    for obj in objects:
        try:
            size = get_object_size(obj)
            # Only include objects larger than 1MB
            if size > 1024 * 1024:
                # Get object type and where it was created
                obj_type = type(obj).__name__

                # Try to get a reference to where the object was created
                try:
                    if hasattr(obj, "__traceback__"):
                        trace = "".join(traceback.format_tb(obj.__traceback__))
                    else:
                        trace = "Unknown"
                except Exception:
                    trace = "Error getting traceback"

                object_info.append(
                    {
                        "id": id(obj),
                        "type": obj_type,
                        "size": size,
                        "size_mb": size / (1024 * 1024),
                        "trace": trace,
                    }
                )
        except Exception:
            # Skip objects that we can't get size for
            pass

    # Sort by size (largest first) and take top N
    object_info.sort(key=lambda x: x["size"], reverse=True)
    return object_info[:limit]
