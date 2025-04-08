# wdbx/monitoring.py
"""
Comprehensive monitoring and observability system for WDBX.

This module provides detailed monitoring of WDBX operations, including:
- Performance metrics collection
- Prometheus integration
- Distributed tracing
- Health checks
- Log aggregation
- Alerting
- ML backend monitoring
"""
import functools
import logging
import os
import queue
import random
import socket
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, TypeVar

from ..core.constants import logger

# Import ML module components for backend monitoring
try:
    from ..ml import JAX_AVAILABLE, TORCH_AVAILABLE, get_ml_backend
    from ..ml.backend import BackendType

    ML_MONITORING_AVAILABLE = True
except ImportError:
    ML_MONITORING_AVAILABLE = False
    JAX_AVAILABLE = False
    TORCH_AVAILABLE = False
    logger.warning("ML module not available. ML monitoring will be disabled.")

# Define Self type for return type annotations
T = TypeVar("T")

try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Info, Summary
    from prometheus_client import start_http_server as prometheus_start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes to avoid type errors

    class Counter:
        pass

    class Gauge:
        pass

    class Histogram:
        pass

    class Summary:
        pass

    class Info:
        pass

    def prometheus_start_http_server(*args, **kwargs):
        pass

    logging.warning("Prometheus client not available. Metrics will not be exported to Prometheus.")

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Create dummy modules/classes to avoid type errors

    class TracerProvider:
        pass

    class ConsoleSpanExporter:
        pass

    class SimpleSpanProcessor:
        pass

    class OTLPSpanExporter:
        pass

    trace = type(
        "DummyModule",
        (),
        {
            "get_tracer_provider": lambda: type(
                "DummyProvider", (), {"add_span_processor": lambda x: None}
            )(),
            "get_current_span": lambda: None,
            "set_tracer_provider": lambda x: None,
        },
    )
    logging.warning("OpenTelemetry not available. Distributed tracing will not be available.")


# Monitoring configuration
METRICS_PORT = int(os.environ.get("WDBX_METRICS_PORT", 9090))
METRICS_ENABLED = os.environ.get("WDBX_METRICS_ENABLED", "true").lower() == "true"
TRACING_ENABLED = os.environ.get("WDBX_TRACING_ENABLED", "true").lower() == "true"
LOG_SAMPLING_RATE = float(os.environ.get("WDBX_LOG_SAMPLING_RATE", 0.1))
HEALTH_CHECK_INTERVAL = float(os.environ.get("WDBX_HEALTH_CHECK_INTERVAL", 60.0))
ML_MONITORING_ENABLED = os.environ.get("WDBX_ML_MONITORING_ENABLED", "true").lower() == "true"


class MetricsRegistry:
    """
    Registry for metrics collection.
    Provides a unified interface for metrics, with or without Prometheus.
    """

    def __init__(self, name: str, enable_prometheus: bool = METRICS_ENABLED):
        """
        Initialize the metrics registry.

        Args:
            name: Name/prefix for metrics in this registry
            enable_prometheus: Whether to enable Prometheus metrics
        """
        self.name = name
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.metrics: Dict[str, Any] = {}
        self.values: Dict[str, Any] = {}
        self.last_update: Dict[str, float] = {}
        self.lock = threading.RLock()

        # Start Prometheus HTTP server if enabled
        if self.enable_prometheus:
            self._start_prometheus_server()

    def _start_prometheus_server(self):
        """Start the Prometheus HTTP server."""
        try:
            prometheus_start_http_server(METRICS_PORT)
            logger.info(f"Prometheus metrics server started on port {METRICS_PORT}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {e}")
            self.enable_prometheus = False

    def counter(
        self, name: str, description: str, labels: Optional[List[str]] = None
    ) -> "PrometheusCounter":
        """
        Create or get a counter.

        Args:
            name: Metric name
            description: Metric description
            labels: List of label names

        Returns:
            Counter metric
        """
        with self.lock:
            key = f"counter:{name}"
            if key not in self.metrics:
                if self.enable_prometheus:
                    counter = Counter(f"{self.name}_{name}", description, labels or [])
                else:
                    counter = SimpleCounter(name, description, labels or [])

                self.metrics[key] = counter
                self.values[key] = {}
                self.last_update[key] = time.time()

            return PrometheusCounter(self, name, labels or [])

    def gauge(
        self, name: str, description: str, labels: Optional[List[str]] = None
    ) -> "PrometheusGauge":
        """
        Create or get a gauge.

        Args:
            name: Metric name
            description: Metric description
            labels: List of label names

        Returns:
            Gauge metric
        """
        with self.lock:
            key = f"gauge:{name}"
            if key not in self.metrics:
                if self.enable_prometheus:
                    gauge = Gauge(f"{self.name}_{name}", description, labels or [])
                else:
                    gauge = SimpleGauge(name, description, labels or [])

                self.metrics[key] = gauge
                self.values[key] = {}
                self.last_update[key] = time.time()

            return PrometheusGauge(self, name, labels or [])

    def histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> "PrometheusHistogram":
        """
        Create or get a histogram.

        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
            buckets: List of bucket boundaries

        Returns:
            Histogram metric
        """
        with self.lock:
            key = f"histogram:{name}"
            if key not in self.metrics:
                if self.enable_prometheus:
                    kwargs = {}
                    if buckets:
                        kwargs["buckets"] = buckets

                    histogram = Histogram(
                        f"{self.name}_{name}", description, labels or [], **kwargs
                    )
                else:
                    histogram = SimpleHistogram(name, description, labels or [], buckets)

                self.metrics[key] = histogram
                self.values[key] = {}
                self.last_update[key] = time.time()

            return PrometheusHistogram(self, name, labels or [])

    def summary(
        self, name: str, description: str, labels: Optional[List[str]] = None
    ) -> "PrometheusSummary":
        """
        Create or get a summary.

        Args:
            name: Metric name
            description: Metric description
            labels: List of label names

        Returns:
            Summary metric
        """
        with self.lock:
            key = f"summary:{name}"
            if key not in self.metrics:
                if self.enable_prometheus:
                    summary = Summary(f"{self.name}_{name}", description, labels or [])
                else:
                    summary = SimpleSummary(name, description, labels or [])

                self.metrics[key] = summary
                self.values[key] = {}
                self.last_update[key] = time.time()

            return PrometheusSummary(self, name, labels or [])

    def info(self, name: str, description: str) -> "PrometheusInfo":
        """
        Create or get an info metric.

        Args:
            name: Metric name
            description: Metric description

        Returns:
            Info metric
        """
        with self.lock:
            key = f"info:{name}"
            if key not in self.metrics:
                if self.enable_prometheus:
                    info = Info(f"{self.name}_{name}", description)
                else:
                    info = SimpleInfo(name, description)

                self.metrics[key] = info
                self.values[key] = {}
                self.last_update[key] = time.time()

            return PrometheusInfo(self, name)

    def get_metric(self, metric_type: str, name: str) -> Optional[Any]:
        """
        Get a metric by type and name.

        Args:
            metric_type: Metric type (counter, gauge, histogram, summary, info)
            name: Metric name

        Returns:
            Metric object, or None if not found
        """
        with self.lock:
            key = f"{metric_type}:{name}"
            return self.metrics.get(key)

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all metrics and their values.

        Returns:
            Dictionary of metric values, grouped by type
        """
        with self.lock:
            result = {"counters": {}, "gauges": {}, "histograms": {}, "summaries": {}, "infos": {}}

            for key, value in self.values.items():
                metric_type, name = key.split(":", 1)

                if metric_type == "counter":
                    result["counters"][name] = value
                elif metric_type == "gauge":
                    result["gauges"][name] = value
                elif metric_type == "histogram":
                    result["histograms"][name] = value
                elif metric_type == "summary":
                    result["summaries"][name] = value
                elif metric_type == "info":
                    result["infos"][name] = value

            return result


class PrometheusCounter:
    """Wrapper for Prometheus Counter metric."""

    def __init__(self, registry: MetricsRegistry, name: str, labels: List[str]):
        self.registry = registry
        self.name = name
        self.labels = labels
        self.key = f"counter:{name}"

    def inc(self, value: float = 1, **labels) -> None:
        """
        Increment the counter.

        Args:
            value: Value to increment by
            **labels: Label values
        """
        with self.registry.lock:
            # Update Prometheus metric if enabled
            metric = self.registry.metrics[self.key]
            if self.registry.enable_prometheus:
                if self.labels:
                    label_values = [labels.get(label, "") for label in self.labels]
                    metric.labels(*label_values).inc(value)
                else:
                    metric.inc(value)
            # Update our internal tracking
            elif self.labels:
                label_str = ",".join(f"{k}={v}" for k, v in labels.items())
                if label_str not in self.registry.values[self.key]:
                    self.registry.values[self.key][label_str] = 0
                self.registry.values[self.key][label_str] += value
            else:
                if "value" not in self.registry.values[self.key]:
                    self.registry.values[self.key]["value"] = 0
                self.registry.values[self.key]["value"] += value

            self.registry.last_update[self.key] = time.time()


class PrometheusGauge:
    """Wrapper for Prometheus Gauge metric."""

    def __init__(self, registry: MetricsRegistry, name: str, labels: List[str]):
        self.registry = registry
        self.name = name
        self.labels = labels
        self.key = f"gauge:{name}"

    def set(self, value: float, **labels) -> None:
        """
        Set the gauge value.

        Args:
            value: Value to set
            **labels: Label values
        """
        with self.registry.lock:
            # Update Prometheus metric if enabled
            metric = self.registry.metrics[self.key]
            if self.registry.enable_prometheus:
                if self.labels:
                    label_values = [labels.get(label, "") for label in self.labels]
                    metric.labels(*label_values).set(value)
                else:
                    metric.set(value)
            # Update our internal tracking
            elif self.labels:
                label_str = ",".join(f"{k}={v}" for k, v in labels.items())
                self.registry.values[self.key][label_str] = value
            else:
                self.registry.values[self.key]["value"] = value

            self.registry.last_update[self.key] = time.time()

    def inc(self, value: float = 1, **labels) -> None:
        """
        Increment the gauge.

        Args:
            value: Value to increment by
            **labels: Label values
        """
        with self.registry.lock:
            metric = self.registry.metrics[self.key]
            if self.registry.enable_prometheus:
                if self.labels:
                    label_values = [labels.get(label, "") for label in self.labels]
                    metric.labels(*label_values).inc(value)
                else:
                    metric.inc(value)
            # Update our internal tracking
            elif self.labels:
                label_str = ",".join(f"{k}={v}" for k, v in labels.items())
                if label_str not in self.registry.values[self.key]:
                    self.registry.values[self.key][label_str] = 0
                self.registry.values[self.key][label_str] += value
            else:
                if "value" not in self.registry.values[self.key]:
                    self.registry.values[self.key]["value"] = 0
                self.registry.values[self.key]["value"] += value

            self.registry.last_update[self.key] = time.time()

    def dec(self, value: float = 1, **labels) -> None:
        """
        Decrement the gauge.

        Args:
            value: Value to decrement by
            **labels: Label values
        """
        with self.registry.lock:
            metric = self.registry.metrics[self.key]
            if self.registry.enable_prometheus:
                if self.labels:
                    label_values = [labels.get(label, "") for label in self.labels]
                    metric.labels(*label_values).dec(value)
                else:
                    metric.dec(value)
            # Update our internal tracking
            elif self.labels:
                label_str = ",".join(f"{k}={v}" for k, v in labels.items())
                if label_str not in self.registry.values[self.key]:
                    self.registry.values[self.key][label_str] = 0
                self.registry.values[self.key][label_str] -= value
            else:
                if "value" not in self.registry.values[self.key]:
                    self.registry.values[self.key]["value"] = 0
                self.registry.values[self.key]["value"] -= value

            self.registry.last_update[self.key] = time.time()

    @contextmanager
    def track_inprogress(self, **labels):
        """
        Track in-progress operations.

        Args:
            **labels: Label values
        """
        self.inc(**labels)
        try:
            yield
        finally:
            self.dec(**labels)


class PrometheusHistogram:
    """Wrapper for Prometheus Histogram metric."""

    def __init__(self, registry: MetricsRegistry, name: str, labels: List[str]):
        self.registry = registry
        self.name = name
        self.labels = labels
        self.key = f"histogram:{name}"

    def observe(self, value: float, **labels) -> None:
        """
        Observe a value.

        Args:
            value: Value to observe
            **labels: Label values
        """
        with self.registry.lock:
            metric = self.registry.metrics[self.key]
            if self.registry.enable_prometheus:
                if self.labels:
                    label_values = [labels.get(label, "") for label in self.labels]
                    metric.labels(*label_values).observe(value)
                else:
                    metric.observe(value)
            # Update our internal tracking
            elif self.labels:
                label_str = ",".join(f"{k}={v}" for k, v in labels.items())
                if label_str not in self.registry.values[self.key]:
                    self.registry.values[self.key][label_str] = []
                self.registry.values[self.key][label_str].append(value)
            else:
                if "values" not in self.registry.values[self.key]:
                    self.registry.values[self.key]["values"] = []
                self.registry.values[self.key]["values"].append(value)

            self.registry.last_update[self.key] = time.time()

    @contextmanager
    def time(self, **labels):
        """
        Time a block of code.

        Args:
            **labels: Label values
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.observe(duration, **labels)


class PrometheusSummary:
    """Wrapper for Prometheus Summary metric."""

    def __init__(self, registry: MetricsRegistry, name: str, labels: List[str]):
        self.registry = registry
        self.name = name
        self.labels = labels
        self.key = f"summary:{name}"

    def observe(self, value: float, **labels) -> None:
        """
        Observe a value.

        Args:
            value: Value to observe
            **labels: Label values
        """
        with self.registry.lock:
            metric = self.registry.metrics[self.key]
            if self.registry.enable_prometheus:
                if self.labels:
                    label_values = [labels.get(label, "") for label in self.labels]
                    metric.labels(*label_values).observe(value)
                else:
                    metric.observe(value)
            # Update our internal tracking
            elif self.labels:
                label_str = ",".join(f"{k}={v}" for k, v in labels.items())
                if label_str not in self.registry.values[self.key]:
                    self.registry.values[self.key][label_str] = []
                self.registry.values[self.key][label_str].append(value)
            else:
                if "values" not in self.registry.values[self.key]:
                    self.registry.values[self.key]["values"] = []
                self.registry.values[self.key]["values"].append(value)

            self.registry.last_update[self.key] = time.time()

    @contextmanager
    def time(self, **labels):
        """
        Time a block of code.

        Args:
            **labels: Label values
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.observe(duration, **labels)


class PrometheusInfo:
    """Wrapper for Prometheus Info metric."""

    def __init__(self, registry: MetricsRegistry, name: str):
        self.registry = registry
        self.name = name
        self.key = f"info:{name}"

    def info(self, info: Dict[str, str]) -> None:
        """
        Set info values.

        Args:
            info: Dictionary of info values
        """
        with self.registry.lock:
            metric = self.registry.metrics[self.key]
            if self.registry.enable_prometheus:
                metric.info(info)
            else:
                self.registry.values[self.key] = info

            self.registry.last_update[self.key] = time.time()


# Simple metrics implementations for when Prometheus is not available
class SimpleCounter:
    """Simple implementation of a counter."""

    def __init__(self, name: str, description: str, labels: List[str]):
        self.name = name
        self.description = description
        self._labels = labels
        self.values = {}
        self._label_values = None

    def labels(self, *label_values) -> "SimpleCounter":
        """
        Get a counter with the given label values.

        Args:
            *label_values: Label values

        Returns:
            Self
        """
        self._label_values = label_values
        key = ",".join(str(v) for v in label_values)
        if key not in self.values:
            self.values[key] = 0
        return self

    def inc(self, value: float = 1) -> "SimpleCounter":
        """
        Increment the counter.

        Args:
            value: Value to increment by

        Returns:
            Self
        """
        if self._label_values:
            key = ",".join(str(v) for v in self._label_values)
            self.values[key] += value
        else:
            if "value" not in self.values:
                self.values["value"] = 0
            self.values["value"] += value
        return self


class SimpleGauge:
    """Simple implementation of a gauge."""

    def __init__(self, name: str, description: str, labels: List[str]):
        self.name = name
        self.description = description
        self._labels = labels
        self.values = {}
        self._label_values = None

    def labels(self, *label_values) -> "SimpleGauge":
        """
        Get a gauge with the given label values.

        Args:
            *label_values: Label values

        Returns:
            Self
        """
        self._label_values = label_values
        key = ",".join(str(v) for v in label_values)
        if key not in self.values:
            self.values[key] = 0
        return self

    def set(self, value: float) -> "SimpleGauge":
        """
        Set the gauge value.

        Args:
            value: Value to set

        Returns:
            Self
        """
        if self._label_values:
            key = ",".join(str(v) for v in self._label_values)
            self.values[key] = value
        else:
            self.values["value"] = value
        return self

    def inc(self, value: float = 1) -> "SimpleGauge":
        """
        Increment the gauge.

        Args:
            value: Value to increment by

        Returns:
            Self
        """
        if self._label_values:
            key = ",".join(str(v) for v in self._label_values)
            if key not in self.values:
                self.values[key] = 0
            self.values[key] += value
        else:
            if "value" not in self.values:
                self.values["value"] = 0
            self.values["value"] += value
        return self

    def dec(self, value: float = 1) -> "SimpleGauge":
        """
        Decrement the gauge.

        Args:
            value: Value to decrement by

        Returns:
            Self
        """
        if self._label_values:
            key = ",".join(str(v) for v in self._label_values)
            if key not in self.values:
                self.values[key] = 0
            self.values[key] -= value
        else:
            if "value" not in self.values:
                self.values["value"] = 0
            self.values["value"] -= value
        return self


class SimpleHistogram:
    """Simple implementation of a histogram."""

    def __init__(
        self, name: str, description: str, labels: List[str], buckets: Optional[List[float]] = None
    ):
        self.name = name
        self.description = description
        self._labels = labels
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        self.values = {}
        self._label_values = None

    def labels(self, *label_values) -> "SimpleHistogram":
        """
        Get a histogram with the given label values.

        Args:
            *label_values: Label values

        Returns:
            Self
        """
        self._label_values = label_values
        key = ",".join(str(v) for v in label_values)
        if key not in self.values:
            self.values[key] = []
        return self

    def observe(self, value: float) -> "SimpleHistogram":
        """
        Observe a value.

        Args:
            value: Value to observe

        Returns:
            Self
        """
        if self._label_values:
            key = ",".join(str(v) for v in self._label_values)
            if key not in self.values:
                self.values[key] = []
            self.values[key].append(value)
        else:
            if "values" not in self.values:
                self.values["values"] = []
            self.values["values"].append(value)
        return self


class SimpleSummary:
    """Simple implementation of a summary."""

    def __init__(self, name: str, description: str, labels: List[str]):
        self.name = name
        self.description = description
        self._labels = labels
        self.values = {}
        self._label_values = None

    def labels(self, *label_values) -> "SimpleSummary":
        """
        Get a summary with the given label values.

        Args:
            *label_values: Label values

        Returns:
            Self
        """
        self._label_values = label_values
        key = ",".join(str(v) for v in label_values)
        if key not in self.values:
            self.values[key] = []
        return self

    def observe(self, value: float) -> "SimpleSummary":
        """
        Observe a value.

        Args:
            value: Value to observe

        Returns:
            Self
        """
        if self._label_values:
            key = ",".join(str(v) for v in self._label_values)
            if key not in self.values:
                self.values[key] = []
            self.values[key].append(value)
        else:
            if "values" not in self.values:
                self.values["values"] = []
            self.values["values"].append(value)
        return self


class SimpleInfo:
    """Simple implementation of an info metric."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.values = {}

    def info(self, info: Dict[str, str]) -> "SimpleInfo":
        """
        Set info values.

        Args:
            info: Dictionary of info values

        Returns:
            Self
        """
        self.values = info
        return self


class TracingManager:
    """
    Manager for distributed tracing.
    """

    def __init__(self, service_name: str = "wdbx"):
        """
        Initialize the tracing manager.

        Args:
            service_name: Name of the service for tracing
        """
        self.service_name = service_name
        self.enabled = TRACING_ENABLED and OPENTELEMETRY_AVAILABLE
        self.tracer = None

        if self.enabled:
            self._init_tracer()

    def _init_tracer(self):
        """Initialize the tracer."""
        try:
            # Set up the tracer provider
            trace.set_tracer_provider(TracerProvider())

            # Set up a console exporter
            console_exporter = ConsoleSpanExporter()
            trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(console_exporter))

            # Set up an OTLP exporter if configured
            otlp_endpoint = os.environ.get("WDBX_OTLP_ENDPOINT")
            if otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(otlp_exporter))

            # Get a tracer
            self.tracer = trace.get_tracer(self.service_name)
            logger.info(f"Distributed tracing initialized for service: {self.service_name}")
        except Exception as e:
            logger.error(f"Failed to initialize tracer: {e}")
            self.enabled = False

    def get_current_span(self) -> Optional[Any]:
        """
        Get the current span.

        Returns:
            Current span, or None if tracing is not enabled
        """
        if not self.enabled or not self.tracer:
            return None

        return trace.get_current_span()

    def start_span(
        self,
        name: str,
        context: Optional[Any] = None,
        kind: Optional[Any] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Start a new span.

        Args:
            name: Span name
            context: Parent context, or None for a new trace
            kind: Span kind (e.g., SERVER, CLIENT, etc.)
            attributes: Span attributes

        Returns:
            Span object, or a dummy span if tracing is not enabled
        """
        if not self.enabled or not self.tracer:
            return DummySpan(name)

        span_kwargs = {}
        if context:
            span_kwargs["context"] = context
        if kind:
            span_kwargs["kind"] = kind
        if attributes:
            span_kwargs["attributes"] = attributes

        return self.tracer.start_span(name, **span_kwargs)

    @contextmanager
    def trace(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Context manager for creating a span.

        Args:
            name: Span name
            attributes: Span attributes
        """
        if not self.enabled or not self.tracer:
            yield None
            return

        with self.tracer.start_as_current_span(name, attributes=attributes or {}) as span:
            yield span

    def trace_function(
        self, name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Decorator for tracing a function.

        Args:
            name: Span name, or None to use the function name
            attributes: Span attributes

        Returns:
            Decorator function
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                span_name = name or func.__qualname__
                with self.trace(span_name, attributes or {}):
                    return func(*args, **kwargs)

            return wrapper

        return decorator


class DummySpan:
    """Dummy span for when tracing is not enabled."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""

    def record_exception(self, exception: Exception) -> None:
        """Record an exception."""

    def end(self) -> None:
        """End the span."""


class HealthChecker:
    """
    Health checker for WDBX components.
    """

    def __init__(self, interval: float = HEALTH_CHECK_INTERVAL):
        """
        Initialize the health checker.

        Args:
            interval: Interval between health checks in seconds
        """
        self.interval = interval
        self.components: Dict[str, Callable[[], bool]] = {}
        self.results: Dict[str, bool] = {}
        self.last_check: Dict[str, float] = {}
        self.running = False
        self.thread = None
        self.lock = threading.RLock()

    def register_component(self, name: str, check_func: Callable[[], bool]) -> None:
        """
        Register a component for health checking.

        Args:
            name: Component name
            check_func: Function that returns True if the component is healthy
        """
        with self.lock:
            self.components[name] = check_func
            self.results[name] = False
            self.last_check[name] = 0

    def check_component(self, name: str) -> bool:
        """
        Check the health of a component.

        Args:
            name: Component name

        Returns:
            True if the component is healthy, False otherwise
        """
        with self.lock:
            if name not in self.components:
                logger.warning(f"Unknown component: {name}")
                return False

            try:
                result = self.components[name]()
                self.results[name] = result
                self.last_check[name] = time.time()
                return result
            except Exception as e:
                logger.error(f"Health check failed for component {name}: {e}")
                self.results[name] = False
                self.last_check[name] = time.time()
                return False

    def check_all(self) -> Dict[str, bool]:
        """
        Check the health of all components.

        Returns:
            Dictionary of component name -> health status
        """
        with self.lock:
            for name in self.components:
                self.check_component(name)
            return self.results.copy()

    def is_healthy(self) -> bool:
        """
        Check if all components are healthy.

        Returns:
            True if all components are healthy, False otherwise
        """
        with self.lock:
            return all(self.results.values())

    def start(self) -> None:
        """Start the health checker thread."""
        with self.lock:
            if self.running:
                return

            self.running = True
            self.thread = threading.Thread(target=self._check_loop, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        """Stop the health checker thread."""
        with self.lock:
            self.running = False
            if self.thread:
                self.thread.join(timeout=self.interval)

    def _check_loop(self) -> None:
        """Health check loop."""
        while self.running:
            try:
                self.check_all()
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

            # Sleep interval
            time.sleep(self.interval)


class LogCollector:
    """
    Collector for log messages with sampling.
    """

    def __init__(self, sampling_rate: float = LOG_SAMPLING_RATE, max_logs: int = 1000):
        """
        Initialize the log collector.

        Args:
            sampling_rate: Probability of collecting a log message (0-1)
            max_logs: Maximum number of logs to keep
        """
        self.sampling_rate = sampling_rate
        self.max_logs = max_logs
        self.logs = queue.Queue(maxsize=max_logs)
        self.random = random.Random()

        # Set up log handler
        self.handler = self._setup_log_handler()

    def _setup_log_handler(self) -> logging.Handler:
        """
        Set up the log handler.

        Returns:
            Configured log handler
        """

        class SamplingHandler(logging.Handler):
            def __init__(self, collector):
                super().__init__()
                self.collector = collector

            def emit(self, record):
                self.collector.add_log(record)

        handler = SamplingHandler(self)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

        # Add to root logger
        logging.getLogger().addHandler(handler)

        return handler

    def add_log(self, record: logging.LogRecord) -> None:
        """
        Add a log record to the collector.

        Args:
            record: Log record to add
        """
        # Apply sampling
        if self.random.random() >= self.sampling_rate:
            return

        # Format the record
        message = self.handler.format(record)

        # Add to queue, dropping oldest if full
        try:
            self.logs.put_nowait(
                {
                    "timestamp": record.created,
                    "level": record.levelname,
                    "logger": record.name,
                    "message": message,
                    "exception": (
                        None
                        if not record.exc_info
                        else {
                            "type": record.exc_info[0].__name__,
                            "value": str(record.exc_info[1]),
                            "traceback": traceback.format_exception(*record.exc_info),
                        }
                    ),
                }
            )
        except queue.Full:
            try:
                # Remove oldest log
                self.logs.get_nowait()
                # Add new log
                self.logs.put_nowait(
                    {
                        "timestamp": record.created,
                        "level": record.levelname,
                        "logger": record.name,
                        "message": message,
                        "exception": (
                            None
                            if not record.exc_info
                            else {
                                "type": record.exc_info[0].__name__,
                                "value": str(record.exc_info[1]),
                                "traceback": traceback.format_exception(*record.exc_info),
                            }
                        ),
                    }
                )
            except Exception:
                pass  # Give up if we can't add the log

    def get_logs(
        self, count: int = 100, level: Optional[str] = None, logger_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get collected logs.

        Args:
            count: Maximum number of logs to return
            level: Filter by log level, or None for all levels
            logger_name: Filter by logger name, or None for all loggers

        Returns:
            List of log records
        """
        # Get all logs
        all_logs = list(self.logs.queue)

        # Apply filters
        if level:
            all_logs = [log for log in all_logs if log["level"] == level]
        if logger_name:
            all_logs = [log for log in all_logs if logger_name in log["logger"]]

        # Sort by timestamp (newest first)
        all_logs.sort(key=lambda log: log["timestamp"], reverse=True)

        # Return requested number of logs
        return all_logs[:count]


class MLBackendMonitor:
    """Monitor for ML backend operations and performance metrics."""

    def __init__(self, metrics_registry: MetricsRegistry):
        """
        Initialize the ML backend monitor.

        Args:
            metrics_registry: Metrics registry to use for reporting
        """
        self.metrics = metrics_registry
        self.enabled = ML_MONITORING_AVAILABLE and ML_MONITORING_ENABLED

        if not self.enabled:
            logger.warning(
                "ML backend monitoring is disabled. Enable by installing ML dependencies."
            )
            return

        # Initialize ML backend
        self.ml_backend = get_ml_backend()
        self.backend_type = self.ml_backend.selected_backend

        # Set up metrics
        self.op_counter = self.metrics.counter(
            "ml_operations_total", "Total number of ML operations", ["backend", "operation_type"]
        )

        self.op_latency = self.metrics.histogram(
            "ml_operation_latency_seconds",
            "Latency of ML operations",
            ["backend", "operation_type"],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        )

        self.mem_usage = self.metrics.gauge(
            "ml_memory_usage_bytes", "Memory usage of ML operations", ["backend", "device"]
        )

        self.conversion_counter = self.metrics.counter(
            "ml_tensor_conversions_total",
            "Total number of tensor conversions",
            ["source_backend", "target_backend"],
        )

        # Initialize backend-specific monitoring
        self._init_backend_monitoring()

        logger.info(f"ML backend monitoring initialized for backend: {self.backend_type}")

    def _init_backend_monitoring(self) -> None:
        """Initialize backend-specific monitoring based on available backends."""
        if not self.enabled:
            return

        # Track available accelerators
        if self.backend_type == "jax" and JAX_AVAILABLE:
            try:
                import jax

                device_count = len(jax.devices())
                device_type = str(jax.devices()[0].device_kind) if device_count > 0 else "unknown"

                self.metrics.gauge(
                    "ml_available_devices",
                    "Number of available ML accelerator devices",
                    ["backend", "device_type"],
                ).set(device_count, backend="jax", device_type=device_type)

                logger.info(f"JAX monitoring initialized with {device_count} {device_type} devices")
            except Exception as e:
                logger.warning(f"Failed to initialize JAX monitoring: {e}")

        elif self.backend_type == "pytorch" and TORCH_AVAILABLE:
            try:
                import torch

                gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

                self.metrics.gauge(
                    "ml_available_devices",
                    "Number of available ML accelerator devices",
                    ["backend", "device_type"],
                ).set(gpu_count, backend="pytorch", device_type="GPU")

                # Check for MPS (Apple Silicon)
                has_mps = hasattr(torch, "mps") and torch.backends.mps.is_available()
                if has_mps:
                    self.metrics.gauge(
                        "ml_available_devices",
                        "Number of available ML accelerator devices",
                        ["backend", "device_type"],
                    ).set(1, backend="pytorch", device_type="MPS")

                logger.info(
                    f"PyTorch monitoring initialized with {gpu_count} GPUs"
                    + (", MPS available" if has_mps else "")
                )
            except Exception as e:
                logger.warning(f"Failed to initialize PyTorch monitoring: {e}")

    def record_operation(
        self, operation_type: str, duration: float, backend: Optional[str] = None
    ) -> None:
        """
        Record an ML operation.

        Args:
            operation_type: Type of operation (e.g., 'similarity', 'normalize', 'batch_process')
            duration: Duration of the operation in seconds
            backend: Optional backend override, defaults to the current backend
        """
        if not self.enabled:
            return

        used_backend = backend or self.backend_type

        # Record metrics
        self.op_counter.inc(1, backend=used_backend, operation_type=operation_type)
        self.op_latency.observe(duration, backend=used_backend, operation_type=operation_type)

    def record_tensor_conversion(self, source_backend: str, target_backend: str) -> None:
        """
        Record a tensor conversion operation.

        Args:
            source_backend: Source backend name
            target_backend: Target backend name
        """
        if not self.enabled:
            return

        self.conversion_counter.inc(1, source_backend=source_backend, target_backend=target_backend)

    def update_memory_usage(self) -> None:
        """Update memory usage metrics for the current backend."""
        if not self.enabled:
            return

        # Update backend-specific memory metrics
        if self.backend_type == "jax" and JAX_AVAILABLE:
            try:
                import jax

                for i, device in enumerate(jax.devices()):
                    try:
                        # This is an approximation as JAX doesn't provide direct memory usage
                        memory_info = jax.device_get(jax.device_put(0, device))
                        # Use device bus memory if available
                        if hasattr(device, "memory_stats") and callable(device.memory_stats):
                            stats = device.memory_stats()
                            if "bytes_in_use" in stats:
                                self.mem_usage.set(
                                    stats["bytes_in_use"],
                                    backend="jax",
                                    device=f"{device.device_kind}:{i}",
                                )
                    except Exception as e:
                        logger.debug(f"Failed to get JAX device memory for device {i}: {e}")
            except Exception as e:
                logger.debug(f"Failed to update JAX memory usage: {e}")

        elif self.backend_type == "pytorch" and TORCH_AVAILABLE:
            try:
                import torch

                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        try:
                            # Get CUDA memory statistics
                            allocated = torch.cuda.memory_allocated(i)
                            self.mem_usage.set(allocated, backend="pytorch", device=f"cuda:{i}")
                        except Exception as e:
                            logger.debug(f"Failed to get CUDA memory for device {i}: {e}")

                # Check for MPS (Apple Silicon)
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    try:
                        # MPS doesn't have direct memory stats, use a placeholder or estimated value
                        self.mem_usage.set(
                            0,  # Placeholder, MPS doesn't expose memory stats directly
                            backend="pytorch",
                            device="mps:0",
                        )
                    except Exception as e:
                        logger.debug(f"Failed to update MPS memory usage: {e}")
            except Exception as e:
                logger.debug(f"Failed to update PyTorch memory usage: {e}")

    def check_backend_health(self) -> bool:
        """
        Check if the ML backend is healthy.

        Returns:
            True if the ML backend is healthy, False otherwise
        """
        if not self.enabled:
            return True

        try:
            # Perform a simple operation to verify the backend works
            import numpy as np

            test_vector = np.random.randn(10).astype(np.float32)

            # Test the backend with a simple operation
            _ = self.ml_backend.normalize(test_vector)

            # If using JAX, test JIT compilation
            if self.backend_type == "jax" and JAX_AVAILABLE:
                import jax
                import jax.numpy as jnp

                @jax.jit
                def test_jit(x):
                    return jnp.sum(x)

                _ = test_jit(jnp.array(test_vector))

            # If using PyTorch, test CUDA if available
            elif self.backend_type == "pytorch" and TORCH_AVAILABLE:
                import torch

                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    x = torch.tensor(test_vector, device=device)
                    _ = x + x

            return True
        except Exception as e:
            logger.error(f"ML backend health check failed: {e}")
            return False


class MonitoringSystem:
    """
    Complete monitoring system for WDBX.
    """

    def __init__(self, service_name: str = "wdbx"):
        """
        Initialize the monitoring system.

        Args:
            service_name: Name of the service for monitoring
        """
        self.service_name = service_name
        self.metrics = MetricsRegistry(service_name, METRICS_ENABLED)
        self.tracing = TracingManager(service_name)
        self.health = HealthChecker()
        self.logs = LogCollector()

        # Initialize ML backend monitoring if enabled
        self.ml_monitor = (
            MLBackendMonitor(self.metrics)
            if ML_MONITORING_ENABLED and ML_MONITORING_AVAILABLE
            else None
        )

        # Initialize common metrics
        self._init_metrics()

        # Start health checker
        self.health.start()

        logger.info(f"Monitoring system initialized for service: {service_name}")

    def _init_metrics(self):
        """Initialize common metrics."""
        # System metrics
        self.system_info = self.metrics.info("system", "System information")
        self.system_info.info(
            {
                "service_name": self.service_name,
                "hostname": socket.gethostname(),
                "python_version": ".".join(map(str, sys.version_info[:3])),
                "pid": str(os.getpid()),
            }
        )

        self.uptime = self.metrics.gauge("uptime_seconds", "Service uptime in seconds")
        self.uptime.set(0)  # Will be updated periodically

        # Start time
        self.start_time = time.time()

        # Start a background thread to update metrics
        self.running = True
        self.update_thread = threading.Thread(target=self._update_metrics_loop, daemon=True)
        self.update_thread.start()

    def _update_metrics_loop(self):
        """Background thread for updating metrics."""
        while self.running:
            try:
                # Update uptime
                self.uptime.set(time.time() - self.start_time)

                # Update process metrics
                self._update_process_metrics()

                # Update ML backend metrics if available
                if self.ml_monitor is not None:
                    self.ml_monitor.update_memory_usage()

            except Exception as e:
                logger.error(f"Error updating metrics: {e}")

            # Sleep for a bit
            time.sleep(5)

    def _update_process_metrics(self):
        """Update process metrics (CPU, memory, etc.)."""
        try:
            import psutil

            process = psutil.Process(os.getpid())

            # CPU usage
            cpu_metric = self.metrics.gauge("process_cpu_percent", "Process CPU usage")
            cpu_metric.set(process.cpu_percent())

            # Memory usage
            memory_info = process.memory_info()
            memory_metric = self.metrics.gauge(
                "process_memory_bytes", "Process memory usage in bytes"
            )
            memory_metric.set(memory_info.rss)

            # Open file descriptors
            open_files_metric = self.metrics.gauge(
                "process_open_fds", "Number of open file descriptors"
            )
            open_files_metric.set(len(process.open_files()))

            # Threads
            threads_metric = self.metrics.gauge("process_threads", "Number of threads")
            threads_metric.set(process.num_threads())
        except ImportError:
            # psutil not available, skip process metrics
            pass
        except Exception as e:
            logger.warning(f"Error updating process metrics: {e}")

    def register_health_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """
        Register a component for health checking.

        Args:
            name: Component name
            check_func: Function that returns True if the component is healthy
        """
        self.health.register_component(name, check_func)

    def check_health(self) -> Dict[str, bool]:
        """
        Check the health of all components.

        Returns:
            Dictionary of component name -> health status
        """
        return self.health.check_all()

    def is_healthy(self) -> bool:
        """
        Check if all components are healthy.

        Returns:
            True if all components are healthy, False otherwise
        """
        return self.health.is_healthy()

    def track_operation(
        self, operation_type: str, name: str, attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracking an operation.

        Args:
            operation_type: Type of operation (e.g., "http", "db", etc.)
            name: Operation name
            attributes: Additional attributes for the operation
        """
        return OperationTracker(self, operation_type, name, attributes)

    def track_ml_operation(self, operation_type: str, name: str, backend: Optional[str] = None):
        """
        Track an ML operation with timing and tracing.

        Args:
            operation_type: Type of ML operation (e.g., similarity, vector_search)
            name: Name of the specific operation
            backend: Optional backend name override

        Returns:
            OperationTracker context manager
        """
        return MLOperationTracker(self, operation_type, name, backend=backend)

    def close(self):
        """Close the monitoring system."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1)
        self.health.stop()


class OperationTracker:
    """
    Context manager for tracking operations.
    """

    def __init__(
        self,
        monitoring: MonitoringSystem,
        operation_type: str,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the operation tracker.

        Args:
            monitoring: Monitoring system
            operation_type: Type of operation (e.g., "http", "db", etc.)
            name: Operation name
            attributes: Additional attributes for the operation
        """
        self.monitoring = monitoring
        self.operation_type = operation_type
        self.name = name
        self.attributes = attributes or {}

        # Create metrics if they don't exist
        self.counter = self.monitoring.metrics.counter(
            f"{operation_type}_operations_total",
            f"Total number of {operation_type} operations",
            ["name", "status"],
        )

        self.latency = self.monitoring.metrics.histogram(
            f"{operation_type}_operation_duration_seconds",
            f"Duration of {operation_type} operations in seconds",
            ["name", "status"],
        )

        self.in_progress = self.monitoring.metrics.gauge(
            f"{operation_type}_operations_in_progress",
            f"Number of {operation_type} operations in progress",
            ["name"],
        )

    def __enter__(self):
        # Start span
        self.span = self.monitoring.tracing.start_span(
            f"{self.operation_type}:{self.name}", attributes=self.attributes
        )

        # Update in-progress metric
        self.in_progress.inc(name=self.name)

        # Record start time
        self.start_time = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate duration
        duration = time.time() - self.start_time

        # Determine status
        status = "success" if exc_type is None else "error"

        # Update metrics
        self.counter.inc(name=self.name, status=status)
        self.latency.observe(duration, name=self.name, status=status)
        self.in_progress.dec(name=self.name)

        # Update span
        if self.span:
            if exc_type:
                self.span.set_attribute("error", True)
                self.span.set_attribute("error.type", exc_type.__name__)
                self.span.set_attribute("error.message", str(exc_val) if exc_val else "")
                self.span.record_exception(exc_val)

            self.span.set_attribute("duration", duration)
            self.span.set_attribute("status", status)

            self.span.end()


class MLOperationTracker:
    """Context manager for tracking ML operations with performance metrics."""

    def __init__(
        self,
        monitoring: MonitoringSystem,
        operation_type: str,
        name: str,
        backend: Optional[str] = None,
    ):
        """
        Initialize the ML operation tracker.

        Args:
            monitoring: MonitoringSystem instance
            operation_type: Type of ML operation
            name: Name of the operation
            backend: Optional backend name override
        """
        self.monitoring = monitoring
        self.operation_type = operation_type
        self.name = name
        self.backend = backend

        if self.monitoring.ml_monitor is not None and backend is None:
            self.backend = self.monitoring.ml_monitor.backend_type

        self.start_time = None
        self.span = None

    def __enter__(self):
        """Start tracking the operation."""
        self.start_time = time.time()

        # Start tracing span if tracing is enabled
        if self.monitoring.tracing:
            attributes = {
                "operation.type": self.operation_type,
                "operation.name": self.name,
            }

            if self.backend:
                attributes["ml.backend"] = self.backend

            self.span = self.monitoring.tracing.start_span(
                f"ml.{self.operation_type}.{self.name}", attributes=attributes
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End tracking and record metrics."""
        # Calculate duration
        duration = time.time() - self.start_time if self.start_time else 0

        # Record basic operation metrics
        self.monitoring.operation_counter.inc(1, type=self.operation_type, name=self.name)
        self.monitoring.operation_latency.observe(
            duration, type=self.operation_type, name=self.name
        )

        # Record ML-specific metrics if available
        if self.monitoring.ml_monitor is not None:
            self.monitoring.ml_monitor.record_operation(
                self.operation_type, duration, backend=self.backend
            )

            # Record ML-specific metrics directly on monitoring system
            self.monitoring.ml_operation_counter.inc(
                1, operation_type=self.operation_type, backend=self.backend or "unknown"
            )

            self.monitoring.ml_operation_latency.observe(
                duration, operation_type=self.operation_type, backend=self.backend or "unknown"
            )

        # Record error if applicable
        if exc_type is not None:
            self.monitoring.operation_errors.inc(1, type=self.operation_type, name=self.name)
            if self.span:
                self.span.record_exception(exc_val)

        # End tracing span
        if self.span:
            if hasattr(self.span, "end"):
                self.span.end()


def init_monitoring(app, service_name: str = "wdbx") -> MonitoringSystem:
    """
    Initialize monitoring for a Flask application.

    Args:
        app: Flask application
        service_name: Name of the service for monitoring

    Returns:
        MonitoringSystem instance
    """
    monitoring = MonitoringSystem(service_name)
    app.monitoring = monitoring

    # Register health check for the app
    monitoring.register_health_check("app", lambda: True)

    try:
        # Add monitoring-related routes
        @app.route("/metrics", methods=["GET"])
        def get_metrics():
            """Expose metrics for Prometheus."""
            # If Prometheus is available, use its metrics handler
            if monitoring.metrics.enable_prometheus:
                import io

                from flask import Response

                output = io.StringIO()
                prometheus_client.write_to_textfile(output, prometheus_client.REGISTRY)
                return Response(output.getvalue(), mimetype="text/plain")

            # Otherwise, return our own metrics
            from flask import jsonify

            return jsonify(monitoring.metrics.get_all_metrics())

        @app.route("/health", methods=["GET"])
        def get_health():
            """Check health status."""
            from flask import jsonify

            health_status = monitoring.check_health()
            all_healthy = all(health_status.values())
            return jsonify(
                {
                    "status": "healthy" if all_healthy else "unhealthy",
                    "checks": health_status,
                    "timestamp": time.time(),
                }
            ), (200 if all_healthy else 503)

        @app.route("/health/live", methods=["GET"])
        def get_liveness():
            """Liveness probe for Kubernetes."""
            from flask import jsonify

            return jsonify({"status": "alive", "timestamp": time.time()})

        @app.route("/health/ready", methods=["GET"])
        def get_readiness():
            """Readiness probe for Kubernetes."""
            from flask import jsonify

            health_status = monitoring.check_health()
            all_healthy = all(health_status.values())
            return jsonify(
                {
                    "status": "ready" if all_healthy else "not ready",
                    "checks": health_status,
                    "timestamp": time.time(),
                }
            ), (200 if all_healthy else 503)

        @app.route("/logs", methods=["GET"])
        def get_logs():
            """Get recent logs."""
            from flask import jsonify, request

            count = request.args.get("count", 100, type=int)
            level = request.args.get("level")
            logger_name = request.args.get("logger")

            logs = monitoring.logs.get_logs(count, level, logger_name)
            return jsonify(logs)

        # Add before/after request handlers to track HTTP requests
        @app.before_request
        def before_request():
            """Track HTTP request start."""
            from flask import g, request

            # Store start time and tracker
            g.request_start_time = time.time()
            g.request_tracker = monitoring.track_operation(
                "http",
                f"{request.method} {request.path}",
                {
                    "http.method": request.method,
                    "http.url": request.url,
                    "http.path": request.path,
                    "http.remote_addr": request.remote_addr,
                },
            )
            g.request_tracker.__enter__()

        @app.after_request
        def after_request(response):
            """Track HTTP request end."""
            from flask import g

            if hasattr(g, "request_tracker"):
                # Add response info to the span
                g.request_tracker.span.set_attribute("http.status_code", response.status_code)

                # Exit the tracker
                exc_info = (None, None, None)
                if response.status_code >= 500:
                    exc_info = (Exception, Exception(f"HTTP {response.status_code}"), None)

                g.request_tracker.__exit__(*exc_info)

            return response

    except Exception as e:
        logger.error(f"Error setting up Flask monitoring routes: {e}")

    logger.info("Monitoring initialized")
    return monitoring


# Unit tests
def test_monitoring():
    """Run unit tests for the monitoring module."""
    import unittest

    class MonitoringTest(unittest.TestCase):
        def setUp(self):
            self.monitoring = MonitoringSystem("test")

        def test_metrics(self):
            # Test counter
            counter = self.monitoring.metrics.counter("test_counter", "Test counter")
            counter.inc()
            counter.inc(5)

            # Test gauge
            gauge = self.monitoring.metrics.gauge("test_gauge", "Test gauge")
            gauge.set(42)
            gauge.inc()
            gauge.dec(2)

            # Test histogram
            histogram = self.monitoring.metrics.histogram("test_histogram", "Test histogram")
            histogram.observe(0.1)
            histogram.observe(0.2)

            # Test summary
            summary = self.monitoring.metrics.summary("test_summary", "Test summary")
            summary.observe(0.1)
            summary.observe(0.2)

            # Test info
            info = self.monitoring.metrics.info("test_info", "Test info")
            info.info({"version": "1.0.0"})

            # Get all metrics
            metrics = self.monitoring.metrics.get_all_metrics()
            self.assertIn("counters", metrics)
            self.assertIn("gauges", metrics)
            self.assertIn("histograms", metrics)
            self.assertIn("summaries", metrics)
            self.assertIn("infos", metrics)

        def test_health(self):
            # Register component
            self.monitoring.register_health_check("test", lambda: True)

            # Check health
            health_status = self.monitoring.check_health()
            self.assertIn("test", health_status)
            self.assertTrue(health_status["test"])

            # Register unhealthy component
            self.monitoring.register_health_check("failing", lambda: False)

            # Check overall health
            self.assertFalse(self.monitoring.is_healthy())

        def test_operation_tracking(self):
            with self.monitoring.track_operation("test", "sample_operation"):
                # Do something
                time.sleep(0.01)

            # Test with error
            try:
                with self.monitoring.track_operation("test", "failing_operation"):
                    raise ValueError("Test error")
            except ValueError:
                pass

    unittest.main(argv=["first-arg-is-ignored"], exit=False)


if __name__ == "__main__":
    test_monitoring()
