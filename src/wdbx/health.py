"""
Health monitoring for WDBX.

This module provides health check and monitoring capabilities for WDBX,
allowing operators to assess system health and detect problems early.
"""

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .utils.logging_utils import get_logger

# Initialize logger
logger = get_logger("wdbx.health")


class HealthStatus(Enum):
    """Health status values."""
    
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    
    name: str
    status: HealthStatus
    description: str
    details: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "description": self.description,
            "details": self.details or {},
            "timestamp": self.timestamp
        }


@dataclass
class HealthReport:
    """Overall health report for the system."""
    
    checks: List[HealthCheck] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    @property
    def status(self) -> HealthStatus:
        """Overall health status based on all checks."""
        if not self.checks:
            return HealthStatus.UNKNOWN
        
        if any(check.status == HealthStatus.ERROR for check in self.checks):
            return HealthStatus.ERROR
        
        if any(check.status == HealthStatus.WARNING for check in self.checks):
            return HealthStatus.WARNING
        
        if all(check.status == HealthStatus.OK for check in self.checks):
            return HealthStatus.OK
        
        return HealthStatus.UNKNOWN
    
    def add_check(self, check: HealthCheck) -> None:
        """Add a health check to the report."""
        self.checks.append(check)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "checks": [check.to_dict() for check in self.checks]
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class HealthMonitor:
    """Health monitor for WDBX system."""
    
    def __init__(self, wdbx_instance=None):
        """
        Initialize health monitor.
        
        Args:
            wdbx_instance: WDBX instance to monitor
        """
        self.wdbx = wdbx_instance
    
    def check_disk_space(self) -> HealthCheck:
        """
        Check available disk space.
        
        Returns:
            Health check result
        """
        try:
            if not self.wdbx or not self.wdbx.data_dir:
                return HealthCheck(
                    name="disk_space",
                    status=HealthStatus.UNKNOWN,
                    description="WDBX instance or data directory not available"
                )
            
            import shutil
            total, used, free = shutil.disk_usage(self.wdbx.data_dir)
            
            # Convert to GB
            total_gb = total / (1024 ** 3)
            used_gb = used / (1024 ** 3)
            free_gb = free / (1024 ** 3)
            
            # Check free space percentage
            free_percent = (free / total) * 100
            
            status = HealthStatus.OK
            description = f"Sufficient disk space available: {free_gb:.2f} GB free ({free_percent:.1f}%)"
            
            # Warning threshold: 20% or 5GB, whichever is smaller
            if free_percent < 20 or free_gb < 5:
                status = HealthStatus.WARNING
                description = f"Low disk space: {free_gb:.2f} GB free ({free_percent:.1f}%)"
            
            # Error threshold: 10% or 1GB, whichever is smaller
            if free_percent < 10 or free_gb < 1:
                status = HealthStatus.ERROR
                description = f"Critical disk space: {free_gb:.2f} GB free ({free_percent:.1f}%)"
            
            return HealthCheck(
                name="disk_space",
                status=status,
                description=description,
                details={
                    "total_gb": round(total_gb, 2),
                    "used_gb": round(used_gb, 2),
                    "free_gb": round(free_gb, 2),
                    "free_percent": round(free_percent, 1)
                }
            )
        except Exception as e:
            logger.error(f"Error checking disk space: {str(e)}")
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.ERROR,
                description=f"Error checking disk space: {str(e)}"
            )
    
    def check_memory_usage(self) -> HealthCheck:
        """
        Check memory usage.
        
        Returns:
            Health check result
        """
        try:
            import psutil
            process = psutil.Process()
            
            # Memory usage in MB
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / (1024 * 1024)
            
            # System memory
            system_memory = psutil.virtual_memory()
            total_memory_mb = system_memory.total / (1024 * 1024)
            available_memory_mb = system_memory.available / (1024 * 1024)
            
            # Calculate percentages
            process_percent = (memory_usage_mb / total_memory_mb) * 100
            system_percent = 100 - (available_memory_mb / total_memory_mb * 100)
            
            status = HealthStatus.OK
            description = f"Normal memory usage: {memory_usage_mb:.1f} MB ({process_percent:.1f}% of system)"
            
            # Warning threshold: Process using more than 30% of system memory or system at 80%
            if process_percent > 30 or system_percent > 80:
                status = HealthStatus.WARNING
                description = f"High memory usage: {memory_usage_mb:.1f} MB ({process_percent:.1f}% of system)"
            
            # Error threshold: Process using more than 50% of system memory or system at 90%
            if process_percent > 50 or system_percent > 90:
                status = HealthStatus.ERROR
                description = f"Critical memory usage: {memory_usage_mb:.1f} MB ({process_percent:.1f}% of system)"
            
            return HealthCheck(
                name="memory_usage",
                status=status,
                description=description,
                details={
                    "process_memory_mb": round(memory_usage_mb, 1),
                    "process_percent": round(process_percent, 1),
                    "system_total_mb": round(total_memory_mb, 1),
                    "system_available_mb": round(available_memory_mb, 1),
                    "system_percent_used": round(system_percent, 1)
                }
            )
        except ImportError:
            logger.warning("psutil not available, cannot check memory usage")
            return HealthCheck(
                name="memory_usage",
                status=HealthStatus.UNKNOWN,
                description="psutil not available, cannot check memory usage"
            )
        except Exception as e:
            logger.error(f"Error checking memory usage: {str(e)}")
            return HealthCheck(
                name="memory_usage",
                status=HealthStatus.ERROR,
                description=f"Error checking memory usage: {str(e)}"
            )
    
    def check_vector_store(self) -> HealthCheck:
        """
        Check vector store health.
        
        Returns:
            Health check result
        """
        if not self.wdbx:
            return HealthCheck(
                name="vector_store",
                status=HealthStatus.UNKNOWN,
                description="WDBX instance not available"
            )
        
        try:
            # Get vector count
            vector_count = len(self.wdbx.vectors) if hasattr(self.wdbx, 'vectors') else 0
            
            status = HealthStatus.OK
            description = f"Vector store is healthy with {vector_count} vectors in memory"
            
            return HealthCheck(
                name="vector_store",
                status=status,
                description=description,
                details={
                    "vector_count": vector_count
                }
            )
        except Exception as e:
            logger.error(f"Error checking vector store: {str(e)}")
            return HealthCheck(
                name="vector_store",
                status=HealthStatus.ERROR,
                description=f"Error checking vector store: {str(e)}"
            )
    
    def check_data_dir(self) -> HealthCheck:
        """
        Check data directory access.
        
        Returns:
            Health check result
        """
        if not self.wdbx or not self.wdbx.data_dir:
            return HealthCheck(
                name="data_dir",
                status=HealthStatus.UNKNOWN,
                description="WDBX instance or data directory not available"
            )
        
        try:
            data_dir = self.wdbx.data_dir
            
            # Check if directory exists
            if not os.path.exists(data_dir):
                return HealthCheck(
                    name="data_dir",
                    status=HealthStatus.ERROR,
                    description=f"Data directory does not exist: {data_dir}"
                )
            
            # Check if it's a directory
            if not os.path.isdir(data_dir):
                return HealthCheck(
                    name="data_dir",
                    status=HealthStatus.ERROR,
                    description=f"Data path is not a directory: {data_dir}"
                )
            
            # Check read/write access
            readable = os.access(data_dir, os.R_OK)
            writable = os.access(data_dir, os.W_OK)
            
            if not readable and not writable:
                return HealthCheck(
                    name="data_dir",
                    status=HealthStatus.ERROR,
                    description=f"No read/write access to data directory: {data_dir}"
                )
            
            if not readable:
                return HealthCheck(
                    name="data_dir",
                    status=HealthStatus.ERROR,
                    description=f"No read access to data directory: {data_dir}"
                )
            
            if not writable:
                return HealthCheck(
                    name="data_dir",
                    status=HealthStatus.WARNING,
                    description=f"No write access to data directory (read-only mode): {data_dir}"
                )
            
            # Get subdirectories
            subdir_names = []
            try:
                subdir_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            except Exception:
                pass
            
            return HealthCheck(
                name="data_dir",
                status=HealthStatus.OK,
                description=f"Data directory is accessible: {data_dir}",
                details={
                    "path": data_dir,
                    "readable": readable,
                    "writable": writable,
                    "subdirectories": subdir_names
                }
            )
        except Exception as e:
            logger.error(f"Error checking data directory: {str(e)}")
            return HealthCheck(
                name="data_dir",
                status=HealthStatus.ERROR,
                description=f"Error checking data directory: {str(e)}"
            )
    
    def check_ml_backend(self) -> HealthCheck:
        """
        Check ML backend health.
        
        Returns:
            Health check result
        """
        if not self.wdbx:
            return HealthCheck(
                name="ml_backend",
                status=HealthStatus.UNKNOWN,
                description="WDBX instance not available"
            )
        
        try:
            # Get ML backend
            if not hasattr(self.wdbx, 'ml_backend'):
                return HealthCheck(
                    name="ml_backend",
                    status=HealthStatus.UNKNOWN,
                    description="ML backend not available"
                )
            
            ml_backend = self.wdbx.ml_backend
            
            # Get backend type
            backend_type = getattr(ml_backend, 'backend_type', 'unknown')
            
            # Simple test: create and normalize a vector
            import numpy as np
            test_vector = np.ones((1, 5), dtype=np.float32)
            normalized = ml_backend.normalize(test_vector)
            
            # Check if normalization worked
            norm = np.linalg.norm(normalized)
            if not np.isclose(norm, 1.0):
                return HealthCheck(
                    name="ml_backend",
                    status=HealthStatus.ERROR,
                    description=f"ML backend ({backend_type}) failed normalization test"
                )
            
            return HealthCheck(
                name="ml_backend",
                status=HealthStatus.OK,
                description=f"ML backend ({backend_type}) is healthy",
                details={
                    "backend_type": backend_type
                }
            )
        except Exception as e:
            logger.error(f"Error checking ML backend: {str(e)}")
            return HealthCheck(
                name="ml_backend",
                status=HealthStatus.ERROR,
                description=f"Error checking ML backend: {str(e)}"
            )
    
    def get_health_report(self) -> HealthReport:
        """
        Get comprehensive health report.
        
        Returns:
            Health report with all checks
        """
        report = HealthReport()
        
        # Add checks
        report.add_check(self.check_disk_space())
        report.add_check(self.check_memory_usage())
        report.add_check(self.check_data_dir())
        report.add_check(self.check_vector_store())
        report.add_check(self.check_ml_backend())
        
        return report


# Utility functions for health checks

def get_health_endpoint_handler(wdbx_instance):
    """
    Create a health endpoint handler for HTTP server.
    
    Args:
        wdbx_instance: WDBX instance to monitor
        
    Returns:
        Health endpoint handler function
    """
    
    async def health_handler(request):
        """
        Handle health check HTTP request.
        
        Args:
            request: HTTP request
            
        Returns:
            HTTP response with health check result
        """
        monitor = HealthMonitor(wdbx_instance)
        report = monitor.get_health_report()
        
        # Convert to response format
        response_data = report.to_dict()
        
        # Set HTTP status code based on health status
        status_code = 200
        if report.status == HealthStatus.WARNING:
            status_code = 200  # Still OK but with warnings
        elif report.status == HealthStatus.ERROR:
            status_code = 500  # Internal Server Error
        elif report.status == HealthStatus.UNKNOWN:
            status_code = 503  # Service Unavailable
        
        # Import web only if needed
        try:
            from aiohttp import web
            return web.json_response(response_data, status=status_code)
        except ImportError:
            # Generic response for testing
            return {
                "status": status_code,
                "body": response_data
            }
    
    return health_handler


def get_health_status(wdbx_instance) -> Dict[str, Any]:
    """
    Get health status as dictionary.
    
    Args:
        wdbx_instance: WDBX instance to monitor
        
    Returns:
        Health status dictionary
    """
    monitor = HealthMonitor(wdbx_instance)
    report = monitor.get_health_report()
    return report.to_dict()


def print_health_report(wdbx_instance) -> None:
    """
    Print health report to console.
    
    Args:
        wdbx_instance: WDBX instance to monitor
    """
    monitor = HealthMonitor(wdbx_instance)
    report = monitor.get_health_report()
    
    print("\n=== WDBX Health Report ===")
    print(f"Overall Status: {report.status.value.upper()}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}")
    print("\nChecks:")
    
    for check in report.checks:
        status_str = check.status.value.upper()
        print(f"  • {check.name}: {status_str}")
        print(f"    {check.description}")
        
        if check.details:
            for key, value in check.details.items():
                print(f"      - {key}: {value}")
        
        print() 