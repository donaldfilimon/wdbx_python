"""
WDBX System Diagnostics Module.

Provides utilities for monitoring system resources and WDBX performance.
"""

import functools
import logging
import os
import platform
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configure logging
logger = logging.getLogger("wdbx.diagnostics")


class SystemMonitor:
    """Monitor system resources and WDBX performance."""

    def __init__(
        self,
        check_interval: int = 5,
        max_history_points: int = 1000,
        threshold_memory_percent: float = 85.0,
        auto_start: bool = False,
    ):
        """
        Initialize the system monitor.

        Args:
            check_interval: Time between checks in seconds
            max_history_points: Maximum number of data points to store
            threshold_memory_percent: Memory threshold for warnings
            auto_start: Whether to start monitoring automatically
        """
        self.check_interval = check_interval
        self.max_history_points = max_history_points
        self.threshold_memory_percent = threshold_memory_percent
        self.running = False
        self.monitoring_thread = None

        # Metrics storage
        self.metrics = {
            "start_time": time.time(),
            "system_info": self.get_system_info(),
            "history": {
                "timestamps": [],
                "memory_usage": [],
                "cpu_usage": [],
                "disk_usage": [],
                "query_count": [],
            },
            "events": [],
            "stats": {
                "total_queries": 0,
                "total_operations": 0,
                "peak_memory_percent": 0.0,
                "peak_memory_timestamp": None,
                "query_durations_ms": [],
            },
        }

        # Register operation timing
        self.op_timers = {}

        # Components to monitor
        self.monitored_components = {}

        # Start monitoring thread if requested
        if auto_start:
            self.start_monitoring()

    def start_monitoring(self):
        """
        Start background monitoring thread.

        Returns:
            bool: True if monitoring started, False if already running
        """
        if self.running:
            return False

        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info(f"Started system monitoring (interval: {self.check_interval}s)")
        return True

    def stop_monitoring(self):
        """
        Stop background monitoring thread.

        Returns:
            bool: True if monitoring was stopped
        """
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
            logger.info("Stopped system monitoring")
        return True

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self.collect_metrics()

                # Check thresholds
                memory_percent = self.get_memory_usage().get("percent", 0)
                if memory_percent > self.threshold_memory_percent:
                    self.log_event(
                        "warning",
                        {
                            "type": "memory_threshold_exceeded",
                            "memory_percent": memory_percent,
                            "threshold": self.threshold_memory_percent,
                        },
                    )

                # Update peak memory
                if memory_percent > self.metrics["stats"]["peak_memory_percent"]:
                    self.metrics["stats"]["peak_memory_percent"] = memory_percent
                    self.metrics["stats"]["peak_memory_timestamp"] = datetime.now().isoformat()

                # Cleanup old metrics periodically
                self._cleanup_old_metrics()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            time.sleep(self.check_interval)

    def collect_metrics(self):
        """
        Collect a snapshot of current metrics.

        Returns:
            bool: True if collection was successful
        """
        try:
            # Get system metrics
            memory_info = self.get_memory_usage() if PSUTIL_AVAILABLE else {"percent": 0}
            cpu_percent = psutil.cpu_percent() if PSUTIL_AVAILABLE else 0
            disk_usage = self.get_disk_usage() if PSUTIL_AVAILABLE else {"percent": 0}

            # Get component metrics
            component_metrics = {}
            for name, component in self.monitored_components.items():
                if hasattr(component, "get_metrics"):
                    try:
                        component_metrics[name] = component.get_metrics()
                    except Exception as e:
                        logger.warning(f"Error getting metrics from {name}: {e}")

            # Record metrics
            now = time.time()
            self.metrics["history"]["timestamps"].append(now)
            self.metrics["history"]["memory_usage"].append(memory_info.get("percent", 0))
            self.metrics["history"]["cpu_usage"].append(cpu_percent)
            self.metrics["history"]["disk_usage"].append(disk_usage.get("percent", 0))

            # Calculate query count
            query_count = sum(comp.get("queries", 0) for comp in component_metrics.values())
            self.metrics["history"]["query_count"].append(query_count)

            # Limit history length
            if len(self.metrics["history"]["timestamps"]) > self.max_history_points:
                for key in self.metrics["history"]:
                    self.metrics["history"][key] = self.metrics["history"][key][
                        -self.max_history_points :
                    ]

            # Update component metrics
            self.metrics["components"] = component_metrics

            # Update stats
            self.metrics["current"] = {
                "memory_percent": memory_info.get("percent", 0),
                "cpu_percent": cpu_percent,
                "disk_percent": disk_usage.get("percent", 0),
                "uptime_seconds": now - self.metrics["start_time"],
            }

            return True

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return False

    def register_component(self, name: str, component: Any):
        """
        Register a component for monitoring.

        The component should implement get_metrics() method.

        Args:
            name: Component name
            component: Component object
        """
        self.monitored_components[name] = component
        logger.debug(f"Registered component for monitoring: {name}")

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log a monitoring event.

        Args:
            event_type: Event type (e.g., "warning", "error", "info")
            data: Event data
        """
        event = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "type": event_type,
            "data": data,
        }

        self.metrics["events"].append(event)

        # Limit event history
        if len(self.metrics["events"]) > self.max_history_points:
            self.metrics["events"] = self.metrics["events"][-self.max_history_points :]

        # Log important events
        if event_type in ("error", "warning"):
            level = logging.ERROR if event_type == "error" else logging.WARNING
            logger.log(level, f"{event_type.upper()}: {data}")

    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager for timing an operation."""
        op_id = self.start_operation_timer(operation_name)
        try:
            yield
        finally:
            self.end_operation_timer(op_id)

    def start_operation_timer(self, operation_name: str) -> str:
        """
        Start timing an operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Operation ID
        """
        op_id = f"{operation_name}_{time.time()}"
        self.op_timers[op_id] = {
            "name": operation_name,
            "start_time": time.time(),
            "completed": False,
        }
        return op_id

    def end_operation_timer(self, op_id: str) -> Optional[float]:
        """
        End timing an operation and record the duration.

        Args:
            op_id: Operation ID from start_operation_timer

        Returns:
            Duration in milliseconds or None if operation not found
        """
        if op_id not in self.op_timers:
            return None

        # Calculate duration
        op_info = self.op_timers[op_id]
        duration_sec = time.time() - op_info["start_time"]
        duration_ms = duration_sec * 1000

        # Update operation info
        op_info["end_time"] = time.time()
        op_info["duration_ms"] = duration_ms
        op_info["completed"] = True

        # Record in stats
        if "query" in op_info["name"].lower():
            self.metrics["stats"]["query_durations_ms"].append(duration_ms)
            self.metrics["stats"]["total_queries"] += 1

        self.metrics["stats"]["total_operations"] += 1

        # Cleanup old timers
        self._cleanup_timers()

        return duration_ms

    def _cleanup_timers(self):
        """Remove old operation timers."""
        # Remove completed timers older than 1 hour
        current_time = time.time()
        to_remove = [
            op_id
            for op_id, op_info in self.op_timers.items()
            if op_info["completed"] and (current_time - op_info["end_time"]) > 3600
        ]

        for op_id in to_remove:
            del self.op_timers[op_id]

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage information.

        Returns:
            Dictionary with memory usage information
        """
        if not PSUTIL_AVAILABLE:
            return {"error": "psutil not available"}

        try:
            memory = psutil.virtual_memory()
            return {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "free": memory.free,
                "percent": memory.percent,
                "formatted": {
                    "total": self._format_bytes(memory.total),
                    "available": self._format_bytes(memory.available),
                    "used": self._format_bytes(memory.used),
                    "free": self._format_bytes(memory.free),
                },
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {"error": str(e)}

    def get_disk_usage(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get disk usage information.

        Args:
            path: Path to check disk usage for

        Returns:
            Dictionary with disk usage information
        """
        if not PSUTIL_AVAILABLE:
            return {"error": "psutil not available"}

        try:
            if path is None:
                # Use current directory
                path = os.getcwd()

            disk_usage = psutil.disk_usage(path)
            return {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "percent": disk_usage.percent,
                "path": path,
                "formatted": {
                    "total": self._format_bytes(disk_usage.total),
                    "used": self._format_bytes(disk_usage.used),
                    "free": self._format_bytes(disk_usage.free),
                },
            }
        except OSError as e:  # Specific OS/IO errors first
            logger.error(f"Error accessing disk path '{path}': {e}")
            return {
                "error": f"Disk access error: {e}",
                "path": path,
                "percent": 0,
                "free": 0,
                "total": 0,
            }
        except Exception as e:  # Catch any other unexpected exceptions
            logger.error(f"Error getting disk usage: {e}")
            return {"error": str(e)}

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information.

        Returns:
            Dictionary with system information
        """
        try:
            info = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
                "processor": platform.processor(),
                "os": platform.system(),
                "architecture": platform.architecture()[0],
            }

            if PSUTIL_AVAILABLE:
                info["cpu_count"] = psutil.cpu_count(logical=True)
                info["cpu_physical_count"] = psutil.cpu_count(logical=False)

            return info
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {"error": str(e)}

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics summary.

        Returns:
            Dictionary with metrics summary
        """
        # Make sure metrics are updated
        self.collect_metrics()

        # Calculate average query duration
        query_durations = self.metrics["stats"]["query_durations_ms"]
        avg_query_latency_ms = sum(query_durations) / len(query_durations) if query_durations else 0

        # Create metrics summary
        return {
            "memory_usage": self.metrics["current"]["memory_percent"],
            "cpu_usage": self.metrics["current"]["cpu_percent"],
            "disk_usage": self.metrics["current"]["disk_percent"],
            "uptime_seconds": self.metrics["current"]["uptime_seconds"],
            "total_queries": self.metrics["stats"]["total_queries"],
            "total_operations": self.metrics["stats"]["total_operations"],
            "peak_memory_percent": self.metrics["stats"]["peak_memory_percent"],
            "avg_query_latency_ms": avg_query_latency_ms,
            "component_count": len(self.monitored_components),
        }

    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes into human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_value < 1024 or unit == "TB":
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024
        return f"{bytes_value:.2f} TB"

    def _cleanup_old_metrics(self):
        """Clean up old metrics and events to prevent memory bloat."""
        current_time = time.time()
        cutoff_time_events = current_time - (24 * 60 * 60)  # Keep events for 24 hours
        cutoff_time_history = current_time - (7 * 24 * 60 * 60)  # Keep history for 7 days

        # Clean up old events
        original_event_count = len(self.metrics["events"])
        self.metrics["events"] = [
            event for event in self.metrics["events"] if event["timestamp"] > cutoff_time_events
        ]
        if len(self.metrics["events"]) < original_event_count:
            logger.debug(
                f"Cleaned up {original_event_count - len(self.metrics['events'])} old events."
            )

        # Clean up old history data points (older than 7 days)
        original_history_count = len(self.metrics["history"]["timestamps"])
        indices_to_keep = [
            i
            for i, ts in enumerate(self.metrics["history"]["timestamps"])
            if ts > cutoff_time_history
        ]

        if len(indices_to_keep) < original_history_count:
            for key in self.metrics["history"]:
                self.metrics["history"][key] = [
                    self.metrics["history"][key][i] for i in indices_to_keep
                ]
            logger.debug(
                f"Cleaned up {original_history_count - len(indices_to_keep)} old history data points."
            )


# Global singleton instance
_global_monitor = None


def get_monitor() -> SystemMonitor:
    """Get global system monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SystemMonitor()
    return _global_monitor


def start_monitoring():
    """Start global system monitoring."""
    return get_monitor().start_monitoring()


def stop_monitoring() -> None:
    """
    Stop the system monitoring and clean up resources.

    This function should be called when monitoring is no longer needed.
    """
    global _global_monitor
    if _global_monitor is not None:
        _global_monitor.stop_monitoring()
        _global_monitor = None


def register_component(name: str, component: Any):
    """Register component with global monitor."""
    return get_monitor().register_component(name, component)


def log_event(event_type: str, data: Dict[str, Any]):
    """Log event with global monitor."""
    return get_monitor().log_event(event_type, data)


def get_metrics() -> Dict[str, Any]:
    """Get metrics from global monitor."""
    return get_monitor().get_metrics()


def time_operation(operation_name: str, func, *args, **kwargs):
    """
    Time a function call and record with global monitor.

    Args:
        operation_name: Name of the operation
        func: Function to call
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

        Returns:
        Result of the function call
    """
    monitor = get_monitor()
    op_id = monitor.start_operation_timer(operation_name)

    try:
        return func(*args, **kwargs)
    finally:
        duration_ms = monitor.end_operation_timer(op_id)
        if duration_ms is not None:
            logger.debug(f"{operation_name} completed in {duration_ms:.2f} ms")


class PerformanceProfiler:
    """Performance profiler for timing code blocks and function calls."""

    def __init__(self):
        """Initialize the performance profiler."""
        self.timers = {}
        self.active_timers = {}
        self.stats = {"calls": {}, "total_time": {}, "avg_time": {}, "min_time": {}, "max_time": {}}
        self.lock = threading.Lock()

    @contextmanager
    def profile_block(self, name: str):
        """
        Context manager for profiling a block of code.

        Args:
            name: Name of the block to profile
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            with self.lock:
                if name not in self.stats["calls"]:
                    self.stats["calls"][name] = 0
                    self.stats["total_time"][name] = 0.0
                    self.stats["min_time"][name] = float("inf")
                    self.stats["max_time"][name] = 0.0

                self.stats["calls"][name] += 1
                self.stats["total_time"][name] += elapsed
                self.stats["avg_time"][name] = (
                    self.stats["total_time"][name] / self.stats["calls"][name]
                )
                self.stats["min_time"][name] = min(self.stats["min_time"][name], elapsed)
                self.stats["max_time"][name] = max(self.stats["max_time"][name], elapsed)

    def profile_function(self, func):
        """
        Decorator for profiling a function.

        Args:
            func: Function to profile

        Returns:
            Wrapped function with profiling
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.profile_block(func.__qualname__):
                return func(*args, **kwargs)

        return wrapper

    def get_all_operations(self) -> List[str]:
        """
        Get all operations that have been profiled.

        Returns:
            List of operation names
        """
        with self.lock:
            return list(self.stats["calls"].keys())

    def get_statistics(self, operation: str) -> Dict[str, Any]:
        """
        Get statistics for a specific operation.

        Args:
            operation: Name of the operation

        Returns:
            Dictionary with operation statistics
        """
        with self.lock:
            if operation not in self.stats["calls"]:
                return {"call_count": 0, "error_count": 0}

            return {
                "call_count": self.stats["calls"].get(operation, 0),
                "total_time_ms": self.stats["total_time"].get(operation, 0) * 1000,
                "avg_time_ms": self.stats["avg_time"].get(operation, 0) * 1000,
                "min_time_ms": (
                    self.stats["min_time"].get(operation, float("inf")) * 1000
                    if self.stats["min_time"].get(operation, float("inf")) < float("inf")
                    else 0
                ),
                "max_time_ms": self.stats["max_time"].get(operation, 0) * 1000,
                "error_count": 0,  # Default to 0 errors
            }

    def get_stats(self):
        """
        Get profiling statistics.

        Returns:
            Dictionary of profiling statistics
        """
        with self.lock:
            return {
                "calls": dict(self.stats["calls"]),
                "total_time_ms": {k: v * 1000 for k, v in self.stats["total_time"].items()},
                "avg_time_ms": {k: v * 1000 for k, v in self.stats["avg_time"].items()},
                "min_time_ms": {
                    k: v * 1000 for k, v in self.stats["min_time"].items() if v < float("inf")
                },
                "max_time_ms": {k: v * 1000 for k, v in self.stats["max_time"].items()},
            }

    def reset(self):
        """Reset all profiling statistics."""
        with self.lock:
            self.stats = {
                "calls": {},
                "total_time": {},
                "avg_time": {},
                "min_time": {},
                "max_time": {},
            }


# Singleton profiler instance
_performance_profiler = None


def get_performance_profiler():
    """
    Get or create the global performance profiler instance.

    Returns:
        PerformanceProfiler: The global profiler instance
    """
    global _performance_profiler
    if _performance_profiler is None:
        _performance_profiler = PerformanceProfiler()
    return _performance_profiler


# Standalone function for getting memory usage (used outside of SystemMonitor)
def get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage statistics.

    Returns:
        Dict with memory usage information
    """
    result = {
        "percent": 0.0,
        "available": 0,
        "used": 0,
        "total": 0,
    }

    if PSUTIL_AVAILABLE:
        try:
            memory = psutil.virtual_memory()
            result["percent"] = memory.percent
            result["available"] = memory.available
            result["used"] = memory.used
            result["total"] = memory.total
            result["available_gb"] = memory.available / (1024**3)
            result["used_gb"] = memory.used / (1024**3)
            result["total_gb"] = memory.total / (1024**3)
        except Exception as e:
            logger.warning(f"Error getting memory info: {e}")

    return result


# Function for getting full system diagnostics information
def system_diagnostics() -> Dict[str, Any]:
    """
    Get comprehensive system diagnostics information.

    Returns:
        Dict with system diagnostics information
    """
    diagnostics = {
        "memory": get_memory_usage(),
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    if PSUTIL_AVAILABLE:
        try:
            diagnostics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            diagnostics["cpu_count"] = psutil.cpu_count()
            diagnostics["boot_time"] = datetime.fromtimestamp(psutil.boot_time()).isoformat()

            # Process information
            process = psutil.Process()
            diagnostics["process"] = {
                "pid": process.pid,
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_info": {
                    "rss": process.memory_info().rss,
                    "vms": process.memory_info().vms,
                    "rss_mb": process.memory_info().rss / (1024 * 1024),
                    "vms_mb": process.memory_info().vms / (1024 * 1024),
                },
                "threads": len(process.threads()),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
            }
        except Exception as e:
            logger.warning(f"Error getting detailed system info: {e}")

    return diagnostics
