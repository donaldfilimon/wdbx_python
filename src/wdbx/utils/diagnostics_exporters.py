"""
WDBX Diagnostics Exporters.

Adapters for exporting diagnostics metrics to external monitoring systems.
"""

import logging
import time
from abc import ABC, abstractmethod
from threading import Event, Thread
from typing import Any, Callable, Dict, List, Optional

import requests

from wdbx.utils.diagnostics import SystemMonitor, get_monitor

logger = logging.getLogger("wdbx.diagnostics.exporters")


class MetricsExporter(ABC):
    """Base class for metrics exporters."""

    def __init__(
        self,
        export_interval: int = 60,
        monitor: Optional[SystemMonitor] = None,
        auto_start: bool = False,
    ):
        """
        Initialize metrics exporter.

        Args:
            export_interval: Seconds between exports
            monitor: SystemMonitor instance to export metrics from (or use global)
            auto_start: Whether to start exporter immediately
        """
        self.export_interval = export_interval
        self.monitor = monitor or get_monitor()
        self.running = False
        self.exporter_thread = None
        self._stop_event = Event()

        # Track export stats
        self.export_count = 0
        self.last_export_time = None
        self.last_export_success = False
        self.last_export_error = None

        if auto_start:
            self.start()

    def start(self) -> bool:
        """Start the metrics exporter."""
        if self.running:
            return False

        self.running = True
        self._stop_event.clear()
        self.exporter_thread = Thread(target=self._export_loop, daemon=True)
        self.exporter_thread.start()
        logger.info(f"Started {self.__class__.__name__} with interval: {self.export_interval}s")
        return True

    def stop(self) -> bool:
        """Stop the metrics exporter."""
        if not self.running:
            return False

        self.running = False
        self._stop_event.set()

        if self.exporter_thread:
            self.exporter_thread.join(timeout=5.0)
            if self.exporter_thread.is_alive():
                logger.warning("Exporter thread didn't terminate gracefully")

        logger.info(f"Stopped {self.__class__.__name__}")
        return True

    def _export_loop(self) -> None:
        """Main export loop."""
        while self.running and not self._stop_event.is_set():
            try:
                export_success = self.export_metrics()
                self.last_export_time = time.time()
                self.last_export_success = export_success

                if export_success:
                    self.export_count += 1
            except Exception as e:
                logger.error(f"Error in export loop: {e}")
                self.last_export_success = False
                self.last_export_error = str(e)

            # Sleep until next interval, but check for stop periodically
            for _ in range(self.export_interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    @abstractmethod
    def export_metrics(self) -> bool:
        """
        Export metrics to external system.

        Returns:
            True if export was successful
        """

    def get_stats(self) -> Dict[str, Any]:
        """Get exporter statistics."""
        return {
            "exporter_type": self.__class__.__name__,
            "export_count": self.export_count,
            "last_export_time": self.last_export_time,
            "last_export_success": self.last_export_success,
            "last_export_error": self.last_export_error,
            "running": self.running,
            "export_interval": self.export_interval,
        }


class PrometheusExporter(MetricsExporter):
    """Export metrics in Prometheus format."""

    def __init__(
        self,
        export_interval: int = 60,
        monitor: Optional[SystemMonitor] = None,
        pushgateway_url: Optional[str] = None,
        job_name: str = "wdbx",
        instance_name: Optional[str] = None,
        auto_start: bool = False,
    ):
        """
        Initialize Prometheus exporter.

        Args:
            export_interval: Seconds between exports
            monitor: SystemMonitor instance (or use global)
            pushgateway_url: Prometheus Pushgateway URL (e.g., 'http://localhost:9091')
                If None, metrics will only be prepared but not sent
            job_name: Prometheus job name
            instance_name: Prometheus instance name (default: hostname)
            auto_start: Whether to start exporter immediately
        """
        super().__init__(export_interval, monitor, auto_start)
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name

        if instance_name:
            self.instance_name = instance_name
        else:
            import socket

            self.instance_name = socket.gethostname()

    def export_metrics(self) -> bool:
        """Export metrics to Prometheus Pushgateway."""
        metrics = self.monitor.get_metrics()

        # Convert metrics to Prometheus format
        prom_lines = self._convert_to_prometheus(metrics)

        # If no Pushgateway URL, just log metrics
        if not self.pushgateway_url:
            logger.debug("Prometheus metrics prepared but no Pushgateway URL configured")
            return True

        # Send metrics to Pushgateway
        try:
            url = (
                f"{self.pushgateway_url}/metrics/job/{self.job_name}/instance/{self.instance_name}"
            )
            response = requests.post(
                url, data="\n".join(prom_lines), headers={"Content-Type": "text/plain"}
            )

            if response.status_code != 200:
                logger.error(
                    f"Failed to push metrics to Prometheus: {response.status_code} - {response.text}"
                )
                return False

            logger.debug(f"Successfully pushed {len(prom_lines)} metrics to Prometheus")
            return True

        except Exception as e:
            logger.error(f"Error pushing to Prometheus: {e}")
            return False

    def _convert_to_prometheus(self, metrics: Dict[str, Any]) -> List[str]:
        """Convert WDBX metrics to Prometheus format."""
        lines = []

        # Add metric help and type info
        lines.append("# HELP wdbx_memory_usage Memory usage percentage")
        lines.append("# TYPE wdbx_memory_usage gauge")
        lines.append(f"wdbx_memory_usage{{{self._get_labels()}}} {metrics['memory_usage']}")

        lines.append("# HELP wdbx_cpu_usage CPU usage percentage")
        lines.append("# TYPE wdbx_cpu_usage gauge")
        lines.append(f"wdbx_cpu_usage{{{self._get_labels()}}} {metrics['cpu_usage']}")

        lines.append("# HELP wdbx_disk_usage Disk usage percentage")
        lines.append("# TYPE wdbx_disk_usage gauge")
        lines.append(f"wdbx_disk_usage{{{self._get_labels()}}} {metrics['disk_usage']}")

        lines.append("# HELP wdbx_uptime_seconds System uptime in seconds")
        lines.append("# TYPE wdbx_uptime_seconds counter")
        lines.append(f"wdbx_uptime_seconds{{{self._get_labels()}}} {metrics['uptime_seconds']}")

        lines.append("# HELP wdbx_total_queries Total number of queries")
        lines.append("# TYPE wdbx_total_queries counter")
        lines.append(f"wdbx_total_queries{{{self._get_labels()}}} {metrics['total_queries']}")

        lines.append("# HELP wdbx_total_operations Total number of operations")
        lines.append("# TYPE wdbx_total_operations counter")
        lines.append(f"wdbx_total_operations{{{self._get_labels()}}} {metrics['total_operations']}")

        lines.append("# HELP wdbx_peak_memory_percent Peak memory usage percentage")
        lines.append("# TYPE wdbx_peak_memory_percent gauge")
        lines.append(
            f"wdbx_peak_memory_percent{{{self._get_labels()}}} {metrics['peak_memory_percent']}"
        )

        lines.append("# HELP wdbx_avg_query_latency_ms Average query latency in milliseconds")
        lines.append("# TYPE wdbx_avg_query_latency_ms gauge")
        lines.append(
            f"wdbx_avg_query_latency_ms{{{self._get_labels()}}} {metrics['avg_query_latency_ms']}"
        )

        lines.append("# HELP wdbx_component_count Number of monitored components")
        lines.append("# TYPE wdbx_component_count gauge")
        lines.append(f"wdbx_component_count{{{self._get_labels()}}} {metrics['component_count']}")

        # Add component-specific metrics if available
        if hasattr(self.monitor, "metrics") and "components" in self.monitor.metrics:
            for component_name, component_metrics in self.monitor.metrics["components"].items():
                for metric_name, metric_value in component_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        metric_id = f"wdbx_{component_name}_{metric_name}"
                        labels = self._get_labels(component=component_name)
                        lines.append(f"# HELP {metric_id} {component_name} {metric_name}")
                        lines.append(f"# TYPE {metric_id} gauge")
                        lines.append(f"{metric_id}{{{labels}}} {metric_value}")

        return lines

    def _get_labels(self, **extra_labels) -> str:
        """Generate Prometheus labels string."""
        labels = {
            "job": self.job_name,
            "instance": self.instance_name,
        }
        labels.update(extra_labels)

        return ",".join([f'{k}="{v}"' for k, v in labels.items()])


class InfluxDBExporter(MetricsExporter):
    """Export metrics to InfluxDB."""

    def __init__(
        self,
        export_interval: int = 60,
        monitor: Optional[SystemMonitor] = None,
        influxdb_url: str = "http://localhost:8086",
        token: Optional[str] = None,
        org: Optional[str] = None,
        bucket: str = "wdbx_metrics",
        measurement: str = "wdbx_system",
        tags: Optional[Dict[str, str]] = None,
        auto_start: bool = False,
    ):
        """
        Initialize InfluxDB exporter.

        Args:
            export_interval: Seconds between exports
            monitor: SystemMonitor instance (or use global)
            influxdb_url: InfluxDB server URL
            token: InfluxDB API token (for InfluxDB 2.x)
            org: InfluxDB organization (for InfluxDB 2.x)
            bucket: InfluxDB bucket/database
            measurement: InfluxDB measurement name
            tags: Additional tags to add to each point
            auto_start: Whether to start exporter immediately
        """
        super().__init__(export_interval, monitor, auto_start)
        self.influxdb_url = influxdb_url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.measurement = measurement
        self.tags = tags or {}

        # Set defaults for hostname tag
        if "host" not in self.tags:
            import socket

            self.tags["host"] = socket.gethostname()

    def export_metrics(self) -> bool:
        """Export metrics to InfluxDB."""
        metrics = self.monitor.get_metrics()

        # Convert metrics to InfluxDB line protocol
        lines = self._convert_to_influxdb(metrics)

        try:
            # Determine which InfluxDB API to use
            if self.token and self.org:
                # InfluxDB 2.x API
                url = f"{self.influxdb_url}/api/v2/write"
                params = {"org": self.org, "bucket": self.bucket, "precision": "s"}
                headers = {"Authorization": f"Token {self.token}"}
            else:
                # InfluxDB 1.x API
                url = f"{self.influxdb_url}/write"
                params = {"db": self.bucket, "precision": "s"}
                headers = {}

            response = requests.post(url, params=params, data="\n".join(lines), headers=headers)

            if response.status_code not in (200, 204):
                logger.error(
                    f"Failed to push metrics to InfluxDB: {response.status_code} - {response.text}"
                )
                return False

            logger.debug(f"Successfully pushed {len(lines)} metrics to InfluxDB")
            return True

        except Exception as e:
            logger.error(f"Error pushing to InfluxDB: {e}")
            return False

    def _convert_to_influxdb(self, metrics: Dict[str, Any]) -> List[str]:
        """Convert WDBX metrics to InfluxDB line protocol."""
        timestamp = int(time.time())
        lines = []

        # Add base tags
        tags_str = ",".join([f"{k}={v}" for k, v in self.tags.items()])
        if tags_str:
            tags_str = "," + tags_str

        # Add core metrics
        fields = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if isinstance(value, int):
                    fields.append(f"{key}={value}i")
                else:
                    fields.append(f"{key}={value}")

        fields_str = ",".join(fields)
        lines.append(f"{self.measurement}{tags_str} {fields_str} {timestamp}")

        # Add component metrics if available
        if hasattr(self.monitor, "metrics") and "components" in self.monitor.metrics:
            for component_name, component_metrics in self.monitor.metrics["components"].items():
                component_fields = []
                for metric_name, metric_value in component_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        if isinstance(metric_value, int):
                            component_fields.append(f"{metric_name}={metric_value}i")
                        else:
                            component_fields.append(f"{metric_name}={metric_value}")

                if component_fields:
                    component_fields_str = ",".join(component_fields)
                    component_measurement = f"{self.measurement}_component"
                    component_tags = f"component={component_name}{tags_str}"
                    lines.append(
                        f"{component_measurement},{component_tags} {component_fields_str} {timestamp}"
                    )

        return lines


class WebhookExporter(MetricsExporter):
    """Export metrics to a webhook endpoint."""

    def __init__(
        self,
        webhook_url: str,
        export_interval: int = 60,
        monitor: Optional[SystemMonitor] = None,
        auth_token: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
        transform_func: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        auto_start: bool = False,
    ):
        """
        Initialize webhook exporter.

        Args:
            webhook_url: URL to send metrics to
            export_interval: Seconds between exports
            monitor: SystemMonitor instance (or use global)
            auth_token: Authorization token for webhook
            extra_payload: Additional data to include in payload
            transform_func: Optional function to transform metrics before sending
            auto_start: Whether to start exporter immediately
        """
        super().__init__(export_interval, monitor, auto_start)
        self.webhook_url = webhook_url
        self.auth_token = auth_token
        self.extra_payload = extra_payload or {}
        self.transform_func = transform_func

    def export_metrics(self) -> bool:
        """Export metrics to webhook."""
        metrics = self.monitor.get_metrics()

        # Apply transformation if provided
        if self.transform_func:
            metrics = self.transform_func(metrics)

        # Build payload
        payload = {
            "timestamp": time.time(),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
        }

        # Add extra payload data
        payload.update(self.extra_payload)

        # Set headers
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            response = requests.post(self.webhook_url, json=payload, headers=headers)

            if not response.ok:
                logger.error(
                    f"Failed to send metrics to webhook: {response.status_code} - {response.text}"
                )
                return False

            logger.debug(f"Successfully sent metrics to webhook: {self.webhook_url}")
            return True

        except Exception as e:
            logger.error(f"Error sending to webhook: {e}")
            return False


# Factory function to create the appropriate exporter
def create_exporter(exporter_type: str, **kwargs) -> MetricsExporter:
    """
    Create a metrics exporter instance.

    Args:
        exporter_type: Type of exporter ('prometheus', 'influxdb', or 'webhook')
        **kwargs: Additional arguments for the exporter

    Returns:
        Configured MetricsExporter instance

    Raises:
        ValueError: If exporter_type is not recognized
    """
    exporter_type = exporter_type.lower()

    if exporter_type == "prometheus":
        return PrometheusExporter(**kwargs)
    if exporter_type == "influxdb":
        return InfluxDBExporter(**kwargs)
    if exporter_type == "webhook":
        return WebhookExporter(**kwargs)
    raise ValueError(f"Unknown exporter type: {exporter_type}")
