"""
OS-level service integration for WDBX.

This module provides tools to install WDBX as a system service on
various platforms (Linux, Windows, macOS) and container environments.
It also includes service monitoring and management capabilities with
ML-based anomaly detection when available.
"""

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..core.constants import logger

# Try to import ML monitoring components for enhanced service management
try:
    from ..monitoring import PerformanceMonitor
    from ..monitoring.performance import MonitoredProcess

    ML_MONITORING_AVAILABLE = True
except ImportError:
    ML_MONITORING_AVAILABLE = False
    logger.warning("ML monitoring components not available. Using basic monitoring.")

# Try to import security components
try:
    from ..security import SecurityManager

    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    logger.warning("Security components not available. Running with reduced security.")

from ..templates import SYSTEMD_SERVICE_TEMPLATE


@dataclass
class ServiceConfig:
    """Configuration for service installation."""

    host: str = "127.0.0.1"
    port: int = 8080
    vector_dim: int = 1024
    shards: int = 8
    log_level: str = "INFO"
    data_dir: str = "/var/lib/wdbx"
    log_dir: str = "/var/log/wdbx"
    user: str = "wdbx"
    group: str = "wdbx"
    password: Optional[str] = None
    start: bool = False
    no_admin_check: bool = False

    # Security options
    enable_security: bool = True
    security_token_expiration: int = 3600
    security_secret_key: Optional[str] = None

    # ML options
    use_ml_backend: bool = True
    ml_device: str = "auto"  # auto, cpu, cuda, etc.

    # Monitoring options
    enable_monitoring: bool = True
    metrics_port: int = 9100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "host": self.host,
            "port": self.port,
            "vector_dim": self.vector_dim,
            "shards": self.shards,
            "log_level": self.log_level,
            "data_dir": self.data_dir,
            "log_dir": self.log_dir,
            "user": self.user,
            "group": self.group,
            "enable_security": self.enable_security,
            "use_ml_backend": self.use_ml_backend,
            "enable_monitoring": self.enable_monitoring,
            "metrics_port": self.metrics_port,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceConfig":
        """Create from dictionary."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    @classmethod
    def from_file(cls, file_path: str) -> "ServiceConfig":
        """Load from config file."""
        try:
            with open(file_path) as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load config from {file_path}: {e}")
            return cls()

    def save_to_file(self, file_path: str) -> bool:
        """Save to config file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Don't save sensitive data like passwords
            data = self.to_dict()

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {file_path}: {e}")
            return False


class ServiceWorker(ABC):
    """Abstract base class for background service workers."""

    def __init__(self, name: str = "worker", restart_on_failure: bool = True):
        """
        Initialize the worker.

        Args:
            name: Name of the worker for identification
            restart_on_failure: Whether to restart the worker on failure
        """
        self.name = name
        self.restart_on_failure = restart_on_failure
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self.last_error: Optional[Exception] = None
        self.start_time: Optional[float] = None
        self.stop_time: Optional[float] = None
        self.restart_count = 0

        # Performance metrics if monitoring is enabled
        self.metrics: Dict[str, Any] = {
            "processed_items": 0,
            "errors": 0,
            "processing_time": 0.0,
            "avg_processing_time": 0.0,
            "peak_memory_usage": 0,
        }

        # Health history tracking
        self.health_history: List[Dict[str, Any]] = []
        self.max_history_entries = 100

        self._metrics_lock = threading.Lock()

    @abstractmethod
    def start(self) -> None:
        """Start the worker process."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the worker process."""

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the worker is running."""

    def get_metrics(self) -> Dict[str, Any]:
        """Get current worker metrics."""
        with self._metrics_lock:
            metrics_copy = self.metrics.copy()

        # Add general status information
        metrics_copy.update(
            {
                "name": self.name,
                "running": self.is_running(),
                "uptime": (
                    (time.time() - (self.start_time or time.time())) if self.is_running() else 0
                ),
                "restart_count": self.restart_count,
                "last_error": str(self.last_error) if self.last_error else None,
                "health_history": len(self.health_history),
            }
        )

        return metrics_copy

    def update_metric(self, name: str, value: Any) -> None:
        """Update a specific metric."""
        with self._metrics_lock:
            self.metrics[name] = value

    def increment_metric(self, name: str, increment: int = 1) -> None:
        """Increment a numeric metric."""
        with self._metrics_lock:
            if name not in self.metrics:
                self.metrics[name] = 0
            self.metrics[name] += increment

    def record_processing_time(self, duration: float) -> None:
        """Record time taken to process an item."""
        with self._metrics_lock:
            self.metrics["processed_items"] += 1
            self.metrics["processing_time"] += duration

            if self.metrics["processed_items"] > 0:
                self.metrics["avg_processing_time"] = (
                    self.metrics["processing_time"] / self.metrics["processed_items"]
                )

    def record_health_snapshot(self, status: str = "healthy") -> None:
        """
        Record a snapshot of current worker health.

        Args:
            status: Health status (healthy, degraded, failing)
        """
        snapshot = {
            "timestamp": time.time(),
            "status": status,
            "metrics": self.get_metrics(),
            "error": str(self.last_error) if self.last_error else None,
        }

        with self._metrics_lock:
            self.health_history.append(snapshot)

            # Limit history size
            if len(self.health_history) > self.max_history_entries:
                self.health_history = self.health_history[-self.max_history_entries :]

    def get_health_trend(self) -> Dict[str, Any]:
        """
        Analyze health history to identify trends.

        Returns:
            Dictionary with health trend information
        """
        if not self.health_history:
            return {"trend": "unknown", "data_points": 0}

        # Simple trend analysis based on error frequency
        recent_history = self.health_history[-min(20, len(self.health_history)) :]
        error_states = sum(1 for entry in recent_history if entry["status"] != "healthy")
        error_ratio = error_states / len(recent_history)

        if error_ratio > 0.5:
            trend = "degrading"
        elif error_ratio > 0.2:
            trend = "occasional_errors"
        elif error_ratio > 0:
            trend = "occasional_errors"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "data_points": len(recent_history),
            "error_ratio": error_ratio,
            "last_status": recent_history[-1]["status"],
        }


class MLMonitoredWorker(ServiceWorker):
    """
    Worker with ML-enhanced monitoring capabilities.

    This worker can detect anomalies in its operation pattern
    and predict potential failures using ML techniques when available.
    """

    def __init__(self, name: str = "ml_worker", restart_on_failure: bool = True):
        """Initialize the ML-monitored worker."""
        super().__init__(name=name, restart_on_failure=restart_on_failure)

        # ML-specific metrics
        self.anomaly_scores: List[float] = []
        self.anomaly_threshold = 0.7
        self.prediction_horizon = 60  # seconds

        # Initialize ML monitoring if available
        self.ml_monitor = None
        if ML_MONITORING_AVAILABLE:
            try:
                self.ml_monitor = MonitoredProcess(self.name)
                logger.info(f"ML monitoring enabled for worker: {self.name}")
            except Exception as e:
                logger.warning(f"Failed to initialize ML monitoring for {self.name}: {e}")

    def detect_anomalies(self) -> Tuple[bool, float]:
        """
        Detect anomalies in worker operation using ML techniques.

        Returns:
            Tuple of (anomaly_detected, anomaly_score)
        """
        if not self.ml_monitor:
            return False, 0.0

        try:
            metrics = self.get_metrics()
            anomaly_score = self.ml_monitor.detect_anomalies(metrics)

            # Store recent anomaly scores for trend analysis
            self.anomaly_scores.append(anomaly_score)
            if len(self.anomaly_scores) > 100:
                self.anomaly_scores.pop(0)

            # Detect if we're above threshold
            is_anomaly = anomaly_score > self.anomaly_threshold

            if is_anomaly:
                logger.warning(
                    f"Anomaly detected in worker {self.name}: "
                    f"score={anomaly_score:.2f}, threshold={self.anomaly_threshold}"
                )

            return is_anomaly, anomaly_score

        except Exception as e:
            logger.error(f"Error in anomaly detection for {self.name}: {e}")
            return False, 0.0

    def predict_failure(self) -> Tuple[bool, float, float]:
        """
        Predict if the worker will fail within the prediction horizon.

        Returns:
            Tuple of (failure_predicted, confidence, estimated_time_to_failure)
        """
        if not self.ml_monitor or len(self.anomaly_scores) < 10:
            return False, 0.0, 0.0

        try:
            metrics = self.get_metrics()
            metrics["anomaly_scores"] = self.anomaly_scores

            prediction = self.ml_monitor.predict_failure(metrics)

            failure_predicted = prediction.get("failure_predicted", False)
            confidence = prediction.get("confidence", 0.0)
            time_to_failure = prediction.get("time_to_failure", 0.0)

            if failure_predicted and confidence > 0.8:
                logger.warning(
                    f"Failure predicted for worker {self.name}: "
                    f"TTF={time_to_failure:.1f}s, confidence={confidence:.2f}"
                )

            return failure_predicted, confidence, time_to_failure

        except Exception as e:
            logger.error(f"Error in failure prediction for {self.name}: {e}")
            return False, 0.0, 0.0


class ServiceManager:
    """Manages WDBX as a background service."""

    def __init__(self, wdbx_instance=None, config: Optional[ServiceConfig] = None):
        """
        Initialize the service manager.

        Args:
            wdbx_instance: WDBX instance to manage
            config: Service configuration
        """
        self.wdbx = wdbx_instance
        self.config = config or ServiceConfig()
        self.workers: List[ServiceWorker] = []
        self.monitor_thread = None
        self.running = False
        self._stop_event = threading.Event()

        # Process management
        self.managed_processes: Dict[int, str] = {}  # pid -> description

        # Security manager if available and enabled
        self.security_manager = None
        if SECURITY_AVAILABLE and self.config.enable_security:
            try:
                self.security_manager = SecurityManager()
                logger.info("Security manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize security manager: {e}")

        # Performance monitoring if available and enabled
        self.performance_monitor = None
        if ML_MONITORING_AVAILABLE and self.config.enable_monitoring:
            try:
                self.performance_monitor = PerformanceMonitor()
                logger.info("Performance monitoring initialized")
            except Exception as e:
                logger.error(f"Failed to initialize performance monitoring: {e}")

        # Install signal handlers for graceful shutdown
        self._setup_signal_handlers()

        logger.info(f"ServiceManager initialized with {len(self.workers)} workers")

    def check_admin(self) -> bool:
        """Check if the process has administrative privileges."""
        if self.config.no_admin_check:
            return True

        try:
            if platform.system() == "Windows":
                import ctypes

                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            else:
                return os.geteuid() == 0
        except Exception as e:
            logger.warning(f"Failed to check admin privileges: {e}")
            return False

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        try:
            # Handle SIGTERM and SIGINT (Ctrl+C) for graceful shutdown
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, self._signal_handler)

            logger.debug("Signal handlers installed")
        except (AttributeError, ValueError) as e:
            # Some environments don't support signal handling
            logger.warning(f"Could not set up signal handlers: {e}")

    def _signal_handler(self, signum, frame) -> None:
        """Handle termination signals."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received signal {sig_name}, shutting down...")
        self.stop_all()

    def add_worker(self, worker: ServiceWorker) -> None:
        """Add a worker to be managed."""
        self.workers.append(worker)
        logger.info(f"Added worker: {worker.name} ({type(worker).__name__})")

    def start_all(self) -> None:
        """Start all managed workers."""
        if self.running:
            logger.warning("ServiceManager already running")
            return

        logger.info("Starting all service workers...")
        self.running = True
        self._stop_event.clear()

        # Start performance monitoring if available
        if self.performance_monitor:
            self.performance_monitor.start()
            logger.info("Performance monitoring started")

        # Start all workers
        for worker in self.workers:
            try:
                worker.start()
                logger.info(f"Started worker: {worker.name}")
            except Exception as e:
                logger.error(f"Failed to start worker {worker.name}: {e}")

        self._start_monitoring()
        logger.info("All service workers started.")

    def stop_all(self) -> None:
        """Stop all managed workers."""
        if not self.running:
            logger.warning("ServiceManager is not running")
            return

        logger.info("Stopping all service workers...")
        self.running = False
        self._stop_event.set()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            if self.monitor_thread.is_alive():
                logger.warning("Monitor thread did not terminate cleanly")

        # Stop workers in reverse order (dependency handling)
        for worker in reversed(self.workers):
            try:
                if worker.is_running():
                    worker.stop()
                    logger.info(f"Stopped worker: {worker.name}")
            except Exception as e:
                logger.error(f"Failed to stop worker {worker.name}: {e}")

        # Stop performance monitoring if available
        if self.performance_monitor:
            try:
                self.performance_monitor.stop()
                logger.info("Performance monitoring stopped")
            except Exception as e:
                logger.error(f"Error stopping performance monitoring: {e}")

        logger.info("All service workers stopped.")

    def _start_monitoring(self) -> None:
        """Start the monitoring thread."""
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, name="service_monitor", daemon=True
        )
        self.monitor_thread.start()
        logger.debug("Service monitor thread started")

    def _monitor_loop(self) -> None:
        """Monitor worker health and restart if necessary."""
        logger.info("Starting service monitor loop")

        # Record metrics every few seconds
        metrics_interval = 10.0  # seconds
        last_metrics_time = time.time()

        # Health snapshot interval
        health_interval = 60.0  # seconds
        last_health_time = time.time()

        # Failure and recovery tracking
        consecutive_failures = {}  # worker name -> count
        recovery_attempts = {}  # worker name -> count
        backoff_times = {}  # worker name -> next retry time

        # Check for ML-based anomaly detection
        anomaly_detection_enabled = ML_MONITORING_AVAILABLE and self.performance_monitor is not None

        while not self._stop_event.is_set() and self.running:
            try:
                current_time = time.time()

                # Check worker health
                for worker in self.workers:
                    worker_name = worker.name

                    try:
                        # Skip workers in backoff period
                        if (
                            worker_name in backoff_times
                            and current_time < backoff_times[worker_name]
                        ):
                            continue

                        if not worker.is_running():
                            # Record health status as failing
                            worker.record_health_snapshot("failing")

                            # Track consecutive failures
                            consecutive_failures[worker_name] = (
                                consecutive_failures.get(worker_name, 0) + 1
                            )

                            if worker.restart_on_failure:
                                # Calculate exponential backoff for repeated failures
                                attempts = recovery_attempts.get(worker_name, 0)
                                backoff_seconds = min(30, 2**attempts)  # Cap at 30 seconds

                                logger.warning(
                                    f"Worker {worker_name} is not running "
                                    f"(failure #{consecutive_failures[worker_name]}). "
                                    f"Attempting restart with {backoff_seconds}s backoff."
                                )

                                # Ensure clean stop before restart
                                try:
                                    worker.stop()
                                except Exception as e:
                                    logger.error(f"Error stopping worker {worker_name}: {e}")

                                # Record restart attempt
                                recovery_attempts[worker_name] = attempts + 1
                                backoff_times[worker_name] = current_time + backoff_seconds

                                # Attempt restart
                                try:
                                    worker.restart_count += 1
                                    worker.start()
                                    logger.info(f"Restarted worker: {worker_name}")
                                except Exception as e:
                                    logger.error(f"Failed to restart worker {worker_name}: {e}")
                                    worker.last_error = e
                            else:
                                logger.warning(
                                    f"Worker {worker_name} is not running and restart_on_failure=False"
                                )
                        else:
                            # Worker is running, reset failure counters
                            if (
                                worker_name in consecutive_failures
                                and consecutive_failures[worker_name] > 0
                            ):
                                logger.info(
                                    f"Worker {worker_name} has recovered after {consecutive_failures[worker_name]} failures"
                                )

                            consecutive_failures[worker_name] = 0
                            recovery_attempts[worker_name] = 0

                            # Record health status
                            if current_time - last_health_time >= health_interval:
                                worker.record_health_snapshot("healthy")

                        # Check for anomalies if the worker supports it
                        if anomaly_detection_enabled and isinstance(worker, MLMonitoredWorker):
                            is_anomaly, score = worker.detect_anomalies()
                            if is_anomaly:
                                # Record degraded health status
                                worker.record_health_snapshot("degraded")

                                # Predict if failure is imminent
                                failure_predicted, confidence, ttf = worker.predict_failure()
                                if failure_predicted and confidence > 0.7 and ttf < 60:
                                    logger.warning(
                                        f"Preemptive restart of {worker_name} due to "
                                        f"imminent failure prediction (TTF: {ttf:.1f}s)"
                                    )

                                    # Ensure clean stop before restart
                                    try:
                                        worker.stop()
                                    except Exception as e:
                                        logger.error(f"Error stopping worker {worker_name}: {e}")

                                    # Attempt restart
                                    try:
                                        worker.restart_count += 1
                                        worker.start()
                                        logger.info(f"Preemptively restarted worker: {worker_name}")
                                    except Exception as e:
                                        logger.error(f"Failed to restart worker {worker_name}: {e}")
                                        worker.last_error = e
                    except Exception as e:
                        logger.error(f"Error monitoring worker {worker_name}: {e}")

                # Record system metrics periodically
                if current_time - last_metrics_time >= metrics_interval:
                    self._record_metrics()
                    last_metrics_time = current_time

                # Update health snapshot interval
                if current_time - last_health_time >= health_interval:
                    last_health_time = current_time

                # Sleep for a short time before checking again
                time.sleep(2.0)

            except Exception as e:
                logger.error(f"Error in monitor loop: {e}", exc_info=True)
                time.sleep(5.0)  # Sleep longer after an error

        logger.info("Stopping service monitor loop")

    def _record_metrics(self) -> None:
        """Record system and worker metrics."""
        if not self.performance_monitor:
            return

        try:
            # Collect worker metrics
            worker_metrics = {}
            for worker in self.workers:
                worker_metrics[worker.name] = worker.get_metrics()

            # Record system metrics
            system_metrics = {
                "cpu_percent": self.performance_monitor.get_cpu_percent(),
                "memory_percent": self.performance_monitor.get_memory_percent(),
                "disk_usage": self.performance_monitor.get_disk_usage(),
                "worker_count": len(self.workers),
                "workers": worker_metrics,
            }

            # Add WDBX specific metrics if available
            if self.wdbx:
                try:
                    wdbx_stats = self.wdbx.get_system_stats()
                    system_metrics["wdbx"] = wdbx_stats
                except Exception as e:
                    logger.error(f"Error getting WDBX stats: {e}")

            # Record in performance monitor
            self.performance_monitor.record_system_metrics(system_metrics)

        except Exception as e:
            logger.error(f"Error recording metrics: {e}")

    def install_systemd_service(self, service_name="wdbx", user=None, group=None) -> bool:
        """Install WDBX as a systemd service (Linux only)."""
        if platform.system() != "Linux":
            logger.error("Systemd service installation is only supported on Linux.")
            return False

        if not self.check_admin():
            logger.error("Root privileges are required to install systemd service.")
            return False

        python_executable = sys.executable
        script_path = os.path.abspath(sys.argv[0])  # Assuming the entry point script
        working_directory = os.getcwd()

        # Get effective user/group if not specified
        try:
            import grp
            import pwd

            effective_user = pwd.getpwuid(os.geteuid()).pw_name
            effective_group = grp.getgrgid(os.getegid()).gr_name
            user = user or effective_user
            group = group or effective_group
        except ImportError:
            logger.warning(
                "pwd/grp modules not available on this system. Using default user/group."
            )
            user = user or "root"
            group = group or "root"

        # Add ML and security configuration if enabled
        service_args = "--server"

        if self.config.enable_security:
            service_args += " --enable-security"

        if self.config.use_ml_backend:
            service_args += " --use-ml"

        if self.config.enable_monitoring:
            service_args += " --enable-monitoring"

        service_content = SYSTEMD_SERVICE_TEMPLATE.format(
            description="WDBX Database Service",
            user=user,
            group=group,
            working_directory=working_directory,
            python_executable=python_executable,
            script_path=script_path,
            service_args=service_args,
        )

        service_file_path = f"/etc/systemd/system/{service_name}.service"

        try:
            with open(service_file_path, "w") as f:
                f.write(service_content)
            logger.info(f"Created systemd service file: {service_file_path}")

            # Reload systemd daemon
            subprocess.check_call(["systemctl", "daemon-reload"])
            logger.info("Systemd daemon reloaded.")

            # Enable the service
            subprocess.check_call(["systemctl", "enable", service_name])
            logger.info(f"Enabled systemd service: {service_name}")

            print(f"Systemd service '{service_name}' installed successfully.")
            print(f"Start with: sudo systemctl start {service_name}")
            print(f"Check status: sudo systemctl status {service_name}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Systemd command failed: {e}")
            return False
        except OSError as e:
            logger.error(f"Failed to write service file: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during service installation: {e}")
            return False

    def create_directories(self) -> bool:
        """Create necessary directories for service operation."""
        try:
            # Create data directory
            if not os.path.exists(self.config.data_dir):
                os.makedirs(self.config.data_dir, exist_ok=True)
                logger.info(f"Created data directory: {self.config.data_dir}")

            # Create log directory
            if not os.path.exists(self.config.log_dir):
                os.makedirs(self.config.log_dir, exist_ok=True)
                logger.info(f"Created log directory: {self.config.log_dir}")

            # Set permissions if running as root
            if os.geteuid() == 0:
                import grp
                import pwd

                try:
                    uid = pwd.getpwnam(self.config.user).pw_uid
                    gid = grp.getgrnam(self.config.group).gr_gid

                    os.chown(self.config.data_dir, uid, gid)
                    os.chown(self.config.log_dir, uid, gid)
                    logger.info(
                        f"Set ownership of directories to {self.config.user}:{self.config.group}"
                    )
                except (KeyError, OSError) as e:
                    logger.warning(f"Failed to set directory ownership: {e}")

            return True
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False

    def restart_worker(self, worker_name: str) -> bool:
        """
        Restart a specific worker by name.

        Args:
            worker_name: Name of the worker to restart

        Returns:
            True if worker was restarted, False otherwise
        """
        for worker in self.workers:
            if worker.name == worker_name:
                try:
                    logger.info(f"Restarting worker: {worker_name}")
                    worker.stop()
                    worker.restart_count += 1
                    worker.start()
                    logger.info(f"Worker {worker_name} restarted successfully")
                    return True
                except Exception as e:
                    logger.error(f"Failed to restart worker {worker_name}: {e}")
                    return False

        logger.warning(f"Worker not found: {worker_name}")
        return False

    def get_worker_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all workers.

        Returns:
            Dictionary of worker status by worker name
        """
        status = {}
        for worker in self.workers:
            status[worker.name] = {
                "running": worker.is_running(),
                "restart_count": worker.restart_count,
                "last_error": str(worker.last_error) if worker.last_error else None,
                "metrics": worker.get_metrics(),
            }
        return status

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status.

        Returns:
            Dictionary with system status information
        """
        status = {
            "running": self.running,
            "worker_count": len(self.workers),
            "workers": self.get_worker_status(),
            "security_enabled": self.security_manager is not None,
            "monitoring_enabled": self.performance_monitor is not None,
        }

        # Add system metrics if monitoring is available
        if self.performance_monitor:
            status["system"] = {
                "cpu_percent": self.performance_monitor.get_cpu_percent(),
                "memory_percent": self.performance_monitor.get_memory_percent(),
                "disk_usage": self.performance_monitor.get_disk_usage(),
                "uptime": self.performance_monitor.get_uptime(),
            }

        # Add WDBX metrics if available
        if self.wdbx:
            try:
                status["wdbx"] = self.wdbx.get_system_stats()
            except Exception as e:
                logger.error(f"Error getting WDBX stats: {e}")
                status["wdbx"] = {"error": str(e)}

        return status


class LinuxServiceManager(ServiceManager):
    """Linux systemd service manager."""

    def install(self) -> bool:
        """Install systemd service."""
        if not self.check_admin():
            logger.error("Root privileges required")
            return False

        service_content = SYSTEMD_SERVICE_TEMPLATE.format(
            user=self.config.user,
            group=self.config.group,
            working_dir=self.working_dir,
            python_path=self.python_path,
            host=self.config.host,
            port=self.config.port,
            vector_dim=self.config.vector_dim,
            shards=self.config.shards,
            log_level=self.config.log_level,
            data_dir=self.config.data_dir,
        )

        try:
            self.create_directories()

            # Write service file
            service_path = "/etc/systemd/system/wdbx.service"
            with open(service_path, "w") as f:
                f.write(service_content)

            # Reload systemd and start service
            subprocess.run(["systemctl", "daemon-reload"], check=True)

            if self.config.start:
                subprocess.run(["systemctl", "enable", "wdbx"], check=True)
                subprocess.run(["systemctl", "start", "wdbx"], check=True)

            return True

        except Exception as e:
            logger.error(f"Service installation failed: {e}")
            return False

    def uninstall(self) -> bool:
        """Remove systemd service."""
        if not self.check_admin():
            return False

        try:
            subprocess.run(["systemctl", "stop", "wdbx"], check=False)
            subprocess.run(["systemctl", "disable", "wdbx"], check=False)

            service_path = "/etc/systemd/system/wdbx.service"
            if os.path.exists(service_path):
                os.remove(service_path)

            subprocess.run(["systemctl", "daemon-reload"], check=True)
            return True

        except Exception as e:
            logger.error(f"Service uninstallation failed: {e}")
            return False


class MacServiceManager(ServiceManager):
    """macOS launchd service manager."""

    def install(self) -> bool:
        """Install launchd service."""
        if not self.check_admin():
            logger.error("Administrator privileges required")
            return False

        try:
            # Create plist file for launchd
            plist_path = "/Library/LaunchDaemons/com.wdbx.service.plist"

            # Implementation would be expanded here
            logger.info("MacOS service installation not fully implemented")
            return False
        except Exception as e:
            logger.error(f"Service installation failed: {e}")
            return False

    def uninstall(self) -> bool:
        """Remove launchd service."""
        if not self.check_admin():
            return False

        try:
            # Implementation would be expanded here
            logger.info("MacOS service uninstallation not fully implemented")
            return False
        except Exception as e:
            logger.error(f"Service uninstallation failed: {e}")
            return False


class WindowsServiceManager(ServiceManager):
    """Windows service manager."""

    def install(self) -> bool:
        """Install Windows service."""
        if not self.check_admin():
            logger.error("Administrator privileges required")
            return False

        try:
            # Use Windows Service Control Manager API
            import win32service
            import win32serviceutil

            service_name = "WDBXService"
            binary_path = os.path.abspath(sys.argv[0])
            display_name = "WDBX Vector Database Service"

            # Create service configuration
            service_config = {
                "ServiceName": service_name,
                "DisplayName": display_name,
                "BinaryPathName": f'"{binary_path}" --service',
                "StartType": win32service.SERVICE_AUTO_START,
                "Description": "WDBX vector database and embedding service",
            }

            # Check if service already exists
            try:
                service_status = win32serviceutil.QueryServiceStatus(service_name)
                logger.warning(f"Service '{service_name}' already exists")
                return True
            except:
                # Service doesn't exist, continue with installation
                pass

            # Create the service
            handle = win32serviceutil.CreateService(
                service_name,
                display_name,
                win32service.SERVICE_ALL_ACCESS,
                win32service.SERVICE_WIN32_OWN_PROCESS,
                service_config["StartType"],
                win32service.SERVICE_ERROR_NORMAL,
                service_config["BinaryPathName"],
                None,
                None,
                None,
                None,
                None,
            )

            win32service.CloseServiceHandle(handle)
            logger.info(f"Windows service '{service_name}' installed successfully")

            # Start the service if requested
            if self.config.start:
                win32serviceutil.StartService(service_name)
                logger.info(f"Service '{service_name}' started")

            return True

        except ImportError:
            logger.error("Windows service components not available. Install pywin32 package.")
            return False
        except Exception as e:
            logger.error(f"Service installation failed: {e}")
            return False

    def uninstall(self) -> bool:
        """Remove Windows service."""
        if not self.check_admin():
            logger.error("Administrator privileges required")
            return False

        try:
            import win32serviceutil

            service_name = "WDBXService"

            # Check if service exists
            try:
                service_status = win32serviceutil.QueryServiceStatus(service_name)
            except:
                logger.warning(f"Service '{service_name}' does not exist")
                return True

            # Stop the service if it's running
            if service_status[1] != win32serviceutil.SERVICE_STOPPED:
                win32serviceutil.StopService(service_name)
                logger.info(f"Service '{service_name}' stopped")

            # Remove the service
            win32serviceutil.RemoveService(service_name)
            logger.info(f"Service '{service_name}' removed successfully")
            return True

        except ImportError:
            logger.error("Windows service components not available. Install pywin32 package.")
            return False
        except Exception as e:
            logger.error(f"Service uninstallation failed: {e}")
            return False


class ServiceFactory:
    """Factory for creating platform-specific service managers."""

    @staticmethod
    def create(config: ServiceConfig) -> ServiceManager:
        """Create appropriate service manager for current platform."""
        system = platform.system()

        if system == "Linux":
            return LinuxServiceManager(config)
        if system == "Darwin":
            return MacServiceManager(config)
        if system == "Windows":
            return WindowsServiceManager(config)
        raise RuntimeError(f"Unsupported platform: {system}")


def generate_docker_compose(args: argparse.Namespace) -> bool:
    """Generate Docker Compose configuration file."""
    # Implementation would go here
    return False


def generate_kubernetes(args: argparse.Namespace) -> bool:
    """Generate Kubernetes configuration files."""
    # Implementation would go here
    return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WDBX Service Manager")
    # Add arguments...
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    config = ServiceConfig(**vars(args))

    try:
        manager = ServiceFactory.create(config)

        if args.command == "install":
            success = manager.install()
        elif args.command == "uninstall":
            success = manager.uninstall()
        elif args.command == "docker":
            success = generate_docker_compose(args)
        elif args.command == "kubernetes":
            success = generate_kubernetes(args)
        else:
            parser = argparse.ArgumentParser(description="WDBX Service Manager")
            parser.print_help()
            return 1

        return 0 if success else 1

    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


def is_admin() -> bool:
    """Check if the script is running with administrative privileges."""
    # This function seems to be a duplicate of ServiceManager.check_admin
    # Implementation would go here
    return False
