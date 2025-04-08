"""
WDBX Prometheus + Grafana Integration Example

This example demonstrates how to:
1. Monitor WDBX system resources
2. Export metrics to Prometheus
3. Visualize metrics in Grafana

Requirements:
- Prometheus installed and running (https://prometheus.io/docs/prometheus/latest/installation/)
- Prometheus Pushgateway installed and running (https://github.com/prometheus/pushgateway)
- Grafana installed and running (https://grafana.com/docs/grafana/latest/installation/)
"""

import logging
import random
import time
from typing import Any, Dict

from wdbx.utils import register_component, start_monitoring, stop_monitoring
from wdbx.utils.diagnostics_exporters import PrometheusExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("wdbx_prometheus_example")


class DatabaseSimulator:
    """Simulates a database component for demonstration."""

    def __init__(self):
        self.query_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self.avg_query_time_ms = 0
        self.total_query_time_ms = 0

    def get_metrics(self) -> Dict[str, Any]:
        """Return component metrics for monitoring."""
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_ratio = self.cache_hits / max(cache_total, 1)

        return {
            "queries": self.query_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": cache_hit_ratio,
            "errors": self.errors,
            "avg_query_time_ms": self.avg_query_time_ms,
        }

    def simulate_query(self, use_cache: bool = True):
        """Simulate a database query."""
        self.query_count += 1

        # Simulate query time between 10-200ms
        query_time = random.uniform(10, 200)
        self.total_query_time_ms += query_time
        self.avg_query_time_ms = self.total_query_time_ms / self.query_count

        # Simulate cache behavior (70% hit rate if cache is used)
        if use_cache:
            if random.random() < 0.7:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
        else:
            self.cache_misses += 1

        # Simulate occasional errors (2% chance)
        if random.random() < 0.02:
            self.errors += 1

        return {"result": f"Query result {self.query_count}"}


class APISimulator:
    """Simulates an API component for demonstration."""

    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_response_time_ms = 0
        self.avg_response_time_ms = 0

    def get_metrics(self) -> Dict[str, Any]:
        """Return component metrics for monitoring."""
        error_rate = self.error_count / max(self.request_count, 1)

        return {
            "requests": self.request_count,
            "errors": self.error_count,
            "error_rate": error_rate,
            "avg_response_time_ms": self.avg_response_time_ms,
        }

    def simulate_request(self):
        """Simulate an API request."""
        self.request_count += 1

        # Simulate response time between 50-500ms
        response_time = random.uniform(50, 500)
        self.total_response_time_ms += response_time
        self.avg_response_time_ms = self.total_response_time_ms / self.request_count

        # Simulate occasional errors (5% chance)
        if random.random() < 0.05:
            self.error_count += 1
            return {"error": "Simulated API error"}

        return {"status": "success", "data": f"Response data {self.request_count}"}


def main():
    """Main function to run the example."""
    logger.info("Starting WDBX Prometheus integration example")

    # Start monitoring
    start_monitoring()
    logger.info("Started system monitoring")

    # Create simulated components
    db = DatabaseSimulator()
    api = APISimulator()

    # Register components with monitoring system
    register_component("database", db)
    register_component("api", api)
    logger.info("Registered components for monitoring")

    # Setup Prometheus exporter
    # Note: Replace localhost with your Pushgateway address if different
    exporter = PrometheusExporter(
        export_interval=15,  # Export every 15 seconds
        pushgateway_url="http://localhost:9091",
        job_name="wdbx_example",
        auto_start=True,
    )
    logger.info(f"Started Prometheus exporter (interval: {exporter.export_interval}s)")

    try:
        # Simulate activity for 10 minutes
        logger.info("Simulating system activity for 10 minutes...")
        end_time = time.time() + 600  # 10 minutes

        while time.time() < end_time:
            # Simulate database queries (1-5 queries per second)
            db_queries = random.randint(1, 5)
            for _ in range(db_queries):
                # 20% queries bypass cache
                db.simulate_query(use_cache=random.random() > 0.2)

            # Simulate API requests (0-3 requests per second)
            api_requests = random.randint(0, 3)
            for _ in range(api_requests):
                api.simulate_request()

            # Wait a second before next batch
            time.sleep(1)

            # Log progress every minute
            elapsed = time.time() - (end_time - 600)
            if elapsed % 60 < 1:
                logger.info(
                    f"Running for {int(elapsed)}s. "
                    + f"DB: {db.query_count} queries, "
                    + f"API: {api.request_count} requests"
                )

    except KeyboardInterrupt:
        logger.info("Example interrupted by user")
    finally:
        # Stop exporter and monitoring
        exporter.stop()
        stop_monitoring()
        logger.info("Stopped monitoring and exporters")

        # Summary
        logger.info("Example completed. Summary:")
        logger.info(
            f"Database: {db.query_count} queries, "
            + f"{db.cache_hits} cache hits, {db.errors} errors, "
            + f"{db.avg_query_time_ms:.2f}ms avg query time"
        )
        logger.info(
            f"API: {api.request_count} requests, "
            + f"{api.error_count} errors, "
            + f"{api.avg_response_time_ms:.2f}ms avg response time"
        )
        logger.info(f"Metrics exported {exporter.export_count} times to Prometheus")

        logger.info("\nNow you can:")
        logger.info("1. Open Prometheus (usually http://localhost:9090)")
        logger.info("2. Query metrics with names starting with 'wdbx_'")
        logger.info("3. Open Grafana (usually http://localhost:3000)")
        logger.info("4. Add the Prometheus data source if not already added")
        logger.info("5. Import the WDBX dashboard from docs/dashboards/grafana_wdbx_dashboard.json")
        logger.info("   (or create your own dashboard with the exported metrics)")


if __name__ == "__main__":
    main()
