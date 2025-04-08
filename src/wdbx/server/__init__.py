"""
WDBX Server Module

This module provides server functionality for WDBX, including:

- HTTP API server with ML-enhanced monitoring & security
- Distributed cluster management with ML-based failure prediction
- Service integration across operating systems
- Security features with JWT authentication and content filtering
- CORS support for web application integration
- Performance optimization with optional JAX acceleration
- Vector storage and retrieval API
- Batch processing capabilities for high-throughput workloads

The server can be run in standalone mode or as part of a cluster for
high availability and scale-out performance. The ML monitoring features
provide predictive maintenance and anomaly detection to ensure system
reliability.
"""

# Cluster management components
from .cluster import ClusterConfig, ClusterCoordinator, ClusterManager, ClusterNode

# HTTP server components
from .http_server import (
    WDBXHttpServer,
    error_response,
    run_server,
    start_server,
    success_response,
)

# Service management components
from .service import (
    LinuxServiceManager,
    MacServiceManager,
    MLMonitoredWorker,
    ServiceConfig,
    ServiceManager,
    ServiceWorker,
    WindowsServiceManager,
)

__all__ = [
    # Cluster components
    "ClusterManager",
    "ClusterNode",
    "ClusterCoordinator",
    "ClusterConfig",
    # HTTP server components
    "WDBXHttpServer",
    "start_server",
    "run_server",
    "success_response",
    "error_response",
    # Service components
    "ServiceManager",
    "ServiceWorker",
    "MLMonitoredWorker",
    "WindowsServiceManager",
    "LinuxServiceManager",
    "MacServiceManager",
    "ServiceConfig",
]
