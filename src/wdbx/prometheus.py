"""
Prometheus metrics exporter for WDBX.

This module provides Prometheus metrics integration for WDBX,
exporting key metrics for monitoring and alerting purposes.
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary
    from prometheus_client.exposition import start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from .utils.logging_utils import get_logger

# Initialize logger
logger = get_logger("wdbx.prometheus")


class WDBXMetrics:
    """WDBX metrics collector and exporter for Prometheus."""
    
    def __init__(self, prefix: str = "wdbx"):
        """
        Initialize metrics collector.
        
        Args:
            prefix: Prefix for all metric names
        """
        self.prefix = prefix
        self.enabled = PROMETHEUS_AVAILABLE
        
        if not self.enabled:
            logger.warning("Prometheus client library not available. Metrics collection disabled.")
            return
        
        # Vector metrics
        self.vector_count = Gauge(
            f"{prefix}_vectors_total",
            "Total number of vectors stored in WDBX"
        )
        
        self.vector_dimension = Gauge(
            f"{prefix}_vector_dimension",
            "Dimension of vectors stored in WDBX"
        )
        
        self.vector_create_count = Counter(
            f"{prefix}_vector_creates_total",
            "Total number of vectors created",
            ["status"]  # success, error
        )
        
        self.vector_batch_create_count = Counter(
            f"{prefix}_vector_batch_creates_total",
            "Total number of vector batch creations",
            ["status"]  # success, error
        )
        
        self.vector_search_count = Counter(
            f"{prefix}_vector_searches_total",
            "Total number of vector searches performed",
            ["status"]  # success, error
        )
        
        self.vector_batch_search_count = Counter(
            f"{prefix}_vector_batch_searches_total",
            "Total number of vector batch searches performed",
            ["status"]  # success, error
        )
        
        # Block metrics
        self.block_count = Gauge(
            f"{prefix}_blocks_total",
            "Total number of blocks stored in WDBX"
        )
        
        self.block_create_count = Counter(
            f"{prefix}_block_creates_total",
            "Total number of blocks created",
            ["status"]  # success, error
        )
        
        self.block_search_count = Counter(
            f"{prefix}_block_searches_total",
            "Total number of block searches performed",
            ["status"]  # success, error
        )
        
        # Memory metrics
        self.memory_usage = Gauge(
            f"{prefix}_memory_usage_bytes",
            "Memory usage of WDBX process in bytes"
        )
        
        self.memory_optimization_count = Counter(
            f"{prefix}_memory_optimizations_total",
            "Total number of memory optimizations performed",
            ["status"]  # success, error
        )
        
        # Performance metrics
        self.search_latency = Histogram(
            f"{prefix}_search_latency_seconds",
            "Latency of search operations in seconds",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
        )
        
        self.create_latency = Histogram(
            f"{prefix}_create_latency_seconds",
            "Latency of create operations in seconds",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
        )
        
        # API metrics
        self.http_requests = Counter(
            f"{prefix}_http_requests_total",
            "Total number of HTTP requests",
            ["method", "endpoint", "status"]
        )
        
        self.http_request_latency = Histogram(
            f"{prefix}_http_request_latency_seconds",
            "Latency of HTTP requests in seconds",
            ["method", "endpoint"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
        )
        
        # Background tasks
        self.background_tasks_active = Gauge(
            f"{prefix}_background_tasks_active",
            "Number of active background tasks"
        )
        
        self.background_task_errors = Counter(
            f"{prefix}_background_task_errors_total",
            "Total number of background task errors",
            ["task_type"]
        )
        
        # I/O metrics
        self.persistence_operations = Counter(
            f"{prefix}_persistence_operations_total",
            "Total number of persistence operations",
            ["operation", "status"]  # load, save, success, error
        )
        
        self.persistence_latency = Histogram(
            f"{prefix}_persistence_latency_seconds",
            "Latency of persistence operations in seconds",
            ["operation"],  # load, save
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 30.0, 60.0)
        )
        
        logger.info("Prometheus metrics initialized")
    
    def start_server(self, port: int = 9090) -> bool:
        """
        Start Prometheus metrics HTTP server.
        
        Args:
            port: Port to listen on
            
        Returns:
            True if server started successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Cannot start Prometheus server: prometheus_client library not available")
            return False
        
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {str(e)}")
            return False
    
    def collect_memory_usage(self, wdbx_instance) -> None:
        """
        Collect memory usage metrics.
        
        Args:
            wdbx_instance: WDBX instance to monitor
        """
        if not self.enabled:
            return
        
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            self.memory_usage.set(memory_info.rss)
        except ImportError:
            logger.warning("psutil not available, cannot collect memory metrics")
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {str(e)}")
    
    def collect_vector_stats(self, wdbx_instance) -> None:
        """
        Collect vector statistics.
        
        Args:
            wdbx_instance: WDBX instance to monitor
        """
        if not self.enabled:
            return
        
        try:
            # Vector count
            if hasattr(wdbx_instance, 'vectors'):
                vector_count = len(wdbx_instance.vectors)
                self.vector_count.set(vector_count)
            
            # Vector dimension
            if hasattr(wdbx_instance, 'vector_dim') and wdbx_instance.vector_dim:
                self.vector_dimension.set(wdbx_instance.vector_dim)
        except Exception as e:
            logger.error(f"Error collecting vector metrics: {str(e)}")
    
    def collect_block_stats(self, wdbx_instance) -> None:
        """
        Collect block statistics.
        
        Args:
            wdbx_instance: WDBX instance to monitor
        """
        if not self.enabled:
            return
        
        try:
            # Block count
            if hasattr(wdbx_instance, 'blocks'):
                block_count = len(wdbx_instance.blocks)
                self.block_count.set(block_count)
        except Exception as e:
            logger.error(f"Error collecting block metrics: {str(e)}")
    
    def collect_all_stats(self, wdbx_instance) -> None:
        """
        Collect all statistics.
        
        Args:
            wdbx_instance: WDBX instance to monitor
        """
        if not self.enabled:
            return
        
        self.collect_memory_usage(wdbx_instance)
        self.collect_vector_stats(wdbx_instance)
        self.collect_block_stats(wdbx_instance)
    
    def record_vector_create(self, success: bool = True) -> None:
        """
        Record vector creation.
        
        Args:
            success: Whether the operation was successful
        """
        if not self.enabled:
            return
        
        status = "success" if success else "error"
        self.vector_create_count.labels(status=status).inc()
    
    def record_vector_batch_create(self, success: bool = True) -> None:
        """
        Record vector batch creation.
        
        Args:
            success: Whether the operation was successful
        """
        if not self.enabled:
            return
        
        status = "success" if success else "error"
        self.vector_batch_create_count.labels(status=status).inc()
    
    def record_vector_search(self, success: bool = True) -> None:
        """
        Record vector search.
        
        Args:
            success: Whether the operation was successful
        """
        if not self.enabled:
            return
        
        status = "success" if success else "error"
        self.vector_search_count.labels(status=status).inc()
    
    def record_vector_batch_search(self, success: bool = True) -> None:
        """
        Record vector batch search.
        
        Args:
            success: Whether the operation was successful
        """
        if not self.enabled:
            return
        
        status = "success" if success else "error"
        self.vector_batch_search_count.labels(status=status).inc()
    
    def record_block_create(self, success: bool = True) -> None:
        """
        Record block creation.
        
        Args:
            success: Whether the operation was successful
        """
        if not self.enabled:
            return
        
        status = "success" if success else "error"
        self.block_create_count.labels(status=status).inc()
    
    def record_block_search(self, success: bool = True) -> None:
        """
        Record block search.
        
        Args:
            success: Whether the operation was successful
        """
        if not self.enabled:
            return
        
        status = "success" if success else "error"
        self.block_search_count.labels(status=status).inc()
    
    def record_memory_optimization(self, success: bool = True) -> None:
        """
        Record memory optimization.
        
        Args:
            success: Whether the operation was successful
        """
        if not self.enabled:
            return
        
        status = "success" if success else "error"
        self.memory_optimization_count.labels(status=status).inc()
    
    def record_persistence_operation(self, operation: str, success: bool = True) -> None:
        """
        Record persistence operation.
        
        Args:
            operation: Type of operation (load, save)
            success: Whether the operation was successful
        """
        if not self.enabled:
            return
        
        status = "success" if success else "error"
        self.persistence_operations.labels(operation=operation, status=status).inc()
    
    def record_http_request(self, method: str, endpoint: str, status: int) -> None:
        """
        Record HTTP request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            status: HTTP status code
        """
        if not self.enabled:
            return
        
        self.http_requests.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    
    def time_search_operation(self) -> Optional[Callable]:
        """
        Time a search operation.
        
        Returns:
            Timer context manager or None if metrics disabled
        """
        if not self.enabled:
            return None
        
        return self.search_latency.time()
    
    def time_create_operation(self) -> Optional[Callable]:
        """
        Time a create operation.
        
        Returns:
            Timer context manager or None if metrics disabled
        """
        if not self.enabled:
            return None
        
        return self.create_latency.time()
    
    def time_persistence_operation(self, operation: str) -> Optional[Callable]:
        """
        Time a persistence operation.
        
        Args:
            operation: Type of operation (load, save)
            
        Returns:
            Timer context manager or None if metrics disabled
        """
        if not self.enabled:
            return None
        
        return self.persistence_latency.labels(operation=operation).time()
    
    def time_http_request(self, method: str, endpoint: str) -> Optional[Callable]:
        """
        Time an HTTP request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            
        Returns:
            Timer context manager or None if metrics disabled
        """
        if not self.enabled:
            return None
        
        return self.http_request_latency.labels(method=method, endpoint=endpoint).time()
    
    def track_background_task(self, task_type: str = None) -> None:
        """
        Track the start of a background task.
        
        Args:
            task_type: Type of background task
        """
        if not self.enabled:
            return
        
        self.background_tasks_active.inc()
    
    def untrack_background_task(self, error: bool = False, task_type: str = None) -> None:
        """
        Track the end of a background task.
        
        Args:
            error: Whether the task ended with an error
            task_type: Type of background task
        """
        if not self.enabled:
            return
        
        self.background_tasks_active.dec()
        
        if error and task_type:
            self.background_task_errors.labels(task_type=task_type).inc()


# Global metrics instance
_global_metrics = None


def get_metrics(prefix: str = "wdbx") -> WDBXMetrics:
    """
    Get or create the global metrics instance.
    
    Args:
        prefix: Prefix for all metric names
        
    Returns:
        WDBXMetrics instance
    """
    global _global_metrics
    
    if _global_metrics is None:
        _global_metrics = WDBXMetrics(prefix=prefix)
    
    return _global_metrics


# Middleware for AIOHTTP

async def prometheus_middleware(app, handler):
    """
    AIOHTTP middleware for recording Prometheus metrics.
    
    Args:
        app: AIOHTTP application
        handler: Request handler
        
    Returns:
        Middleware handler
    """
    metrics = get_metrics()
    
    async def middleware_handler(request):
        if not metrics.enabled:
            return await handler(request)
        
        method = request.method
        endpoint = request.path
        
        # Normalize endpoint by removing IDs
        parts = endpoint.split('/')
        normalized_parts = []
        for part in parts:
            # Replace numeric IDs with {id}
            if part.isdigit():
                normalized_parts.append("{id}")
            else:
                normalized_parts.append(part)
        
        normalized_endpoint = '/'.join(normalized_parts)
        
        # Time the request
        start_time = time.time()
        
        try:
            response = await handler(request)
            status = response.status
            metrics.record_http_request(method, normalized_endpoint, status)
            return response
        except Exception as e:
            metrics.record_http_request(method, normalized_endpoint, 500)
            raise
        finally:
            # Record latency
            latency = time.time() - start_time
            metrics.http_request_latency.labels(method=method, endpoint=normalized_endpoint).observe(latency)
    
    return middleware_handler


# Decorator for instrumenting functions

def instrument(func=None, *, name=None, kind=None):
    """
    Decorator to instrument functions with Prometheus metrics.
    
    Args:
        func: Function to instrument
        name: Name for the metric (defaults to function name)
        kind: Kind of operation (search, create, etc.)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        nonlocal name
        if name is None:
            name = func.__name__
        
        # Get metrics
        metrics = get_metrics()
        
        # Create metrics for this function if they don't exist
        if metrics.enabled and not hasattr(metrics, f"func_{name}_calls"):
            setattr(
                metrics,
                f"func_{name}_calls",
                Counter(
                    f"wdbx_func_{name}_calls_total",
                    f"Total number of calls to {name}",
                    ["status"]
                )
            )
            
            setattr(
                metrics,
                f"func_{name}_latency",
                Histogram(
                    f"wdbx_func_{name}_latency_seconds",
                    f"Latency of {name} in seconds",
                    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
                )
            )
        
        async def async_wrapper(*args, **kwargs):
            if not metrics.enabled:
                return await func(*args, **kwargs)
            
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                getattr(metrics, f"func_{name}_calls").labels(status="success").inc()
                return result
            except Exception as e:
                getattr(metrics, f"func_{name}_calls").labels(status="error").inc()
                raise
            finally:
                latency = time.time() - start_time
                getattr(metrics, f"func_{name}_latency").observe(latency)
                
                # Record specific operation metrics if applicable
                if kind == "search":
                    metrics.search_latency.observe(latency)
                elif kind == "create":
                    metrics.create_latency.observe(latency)
        
        def sync_wrapper(*args, **kwargs):
            if not metrics.enabled:
                return func(*args, **kwargs)
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                getattr(metrics, f"func_{name}_calls").labels(status="success").inc()
                return result
            except Exception as e:
                getattr(metrics, f"func_{name}_calls").labels(status="error").inc()
                raise
            finally:
                latency = time.time() - start_time
                getattr(metrics, f"func_{name}_latency").observe(latency)
                
                # Record specific operation metrics if applicable
                if kind == "search":
                    metrics.search_latency.observe(latency)
                elif kind == "create":
                    metrics.create_latency.observe(latency)
        
        # Choose the right wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    if func is None:
        return decorator
    
    return decorator(func) 