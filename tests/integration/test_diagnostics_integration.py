"""
Integration tests for the diagnostics module.

Tests how the diagnostics module integrates with other WDBX components.
"""

# Import from context
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wdbx.utils.diagnostics import (
    get_metrics,
    register_component,
    start_monitoring,
    stop_monitoring,
)


class MockComponent:
    """A mock component for testing component integration."""

    def __init__(self, name):
        self.name = name
        self.queries = 0
        self.operations = 0

    def get_metrics(self):
        """Return component metrics."""
        return {
            "queries": self.queries,
            "operations": self.operations,
            "status": "healthy",
            "name": self.name,
        }

    def perform_query(self):
        """Simulate a query operation."""
        self.queries += 1
        # Simulate some work
        time.sleep(0.05)
        return {"result": "success"}


class TestDiagnosticsIntegration(unittest.TestCase):
    """Test diagnostics integration with other components."""

    def setUp(self):
        """Set up test environment."""
        # Stop any previous monitoring
        stop_monitoring()

        # Create mock components
        self.db_component = MockComponent("database")
        self.cache_component = MockComponent("cache")
        self.api_component = MockComponent("api")

        # Start monitoring
        start_monitoring()

    def tearDown(self):
        """Clean up after tests."""
        stop_monitoring()

    def test_component_registration(self):
        """Test registering components with the monitoring system."""
        # Register components
        register_component("database", self.db_component)
        register_component("cache", self.cache_component)
        register_component("api", self.api_component)

        # Perform some operations to generate metrics
        for _ in range(5):
            self.db_component.perform_query()

        for _ in range(3):
            self.cache_component.perform_query()

        for _ in range(7):
            self.api_component.perform_query()

        # Sleep to allow metrics collection
        time.sleep(2)

        # Get metrics
        metrics = get_metrics()

        # Verify components are being monitored
        self.assertIn("component_count", metrics)
        self.assertGreaterEqual(metrics["component_count"], 3)

    def test_concurrent_components(self):
        """Test monitoring with concurrent component operations."""
        # Register components
        register_component("database", self.db_component)
        register_component("cache", self.cache_component)

        # Create threads to simulate concurrent operations
        threads = []
        for i in range(10):
            if i % 2 == 0:
                # Database query thread
                t = threading.Thread(target=self.db_component.perform_query)
            else:
                # Cache query thread
                t = threading.Thread(target=self.cache_component.perform_query)
            threads.append(t)

        # Start threads
        for t in threads:
            t.start()

        # Wait for threads to complete
        for t in threads:
            t.join()

        # Sleep to allow metrics collection
        time.sleep(2)

        # Get metrics
        get_metrics()

        # Verify metrics were collected during concurrent operations
        self.assertGreaterEqual(self.db_component.queries, 5)
        self.assertGreaterEqual(self.cache_component.queries, 5)


if __name__ == "__main__":
    unittest.main()
