"""
Unit tests for the diagnostics module.

Tests the SystemMonitor class and utility functions in wdbx.utils.diagnostics.
"""
# Import from context
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wdbx.utils.diagnostics import (
    SystemMonitor,
    get_monitor,
    log_event,
    start_monitoring,
    stop_monitoring,
    time_operation,
)


class TestSystemMonitor(unittest.TestCase):
    """Tests for the SystemMonitor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.monitor = SystemMonitor(check_interval=1, max_history_points=10)
        
    def tearDown(self):
        """Clean up after tests."""
        self.monitor.stop_monitoring()
        
    def test_init(self):
        """Test SystemMonitor initialization."""
        self.assertEqual(self.monitor.check_interval, 1)
        self.assertEqual(self.monitor.max_history_points, 10)
        self.assertFalse(self.monitor.running)
        self.assertIsNone(self.monitor.monitoring_thread)
        
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        # Test starting
        self.assertTrue(self.monitor.start_monitoring())
        self.assertTrue(self.monitor.running)
        self.assertIsNotNone(self.monitor.monitoring_thread)
        self.assertTrue(self.monitor.monitoring_thread.is_alive())
        
        # Test attempting to start twice
        self.assertFalse(self.monitor.start_monitoring())
        
        # Test stopping
        self.assertTrue(self.monitor.stop_monitoring())
        time.sleep(0.1)  # Allow thread to stop
        self.assertFalse(self.monitor.running)
        
    def test_collect_metrics(self):
        """Test collecting metrics."""
        # Collect initial metrics
        self.assertTrue(self.monitor.collect_metrics())
        
        # Check that history has been populated
        self.assertEqual(len(self.monitor.metrics["history"]["timestamps"]), 1)
        self.assertEqual(len(self.monitor.metrics["history"]["memory_usage"]), 1)
        self.assertEqual(len(self.monitor.metrics["history"]["cpu_usage"]), 1)
        
    def test_register_component(self):
        """Test registering a component."""
        mock_component = MagicMock()
        mock_component.get_metrics.return_value = {"metric1": 42}
        
        self.monitor.register_component("test_component", mock_component)
        self.assertIn("test_component", self.monitor.monitored_components)
        
        # Test that component metrics are collected
        self.monitor.collect_metrics()
        self.monitor.get_metrics()  # This should collect metrics from components
        
        # Verify component's get_metrics was called
        mock_component.get_metrics.assert_called()
    
    def test_log_event(self):
        """Test logging events."""
        initial_events = len(self.monitor.metrics["events"])
        
        # Log test event
        self.monitor.log_event("info", {"message": "test event"})
        
        # Verify event was added
        self.assertEqual(len(self.monitor.metrics["events"]), initial_events + 1)
        self.assertEqual(self.monitor.metrics["events"][-1]["type"], "info")
        self.assertEqual(self.monitor.metrics["events"][-1]["data"]["message"], "test event")
    
    def test_operation_timer(self):
        """Test operation timing."""
        # Start and end timer
        op_id = self.monitor.start_operation_timer("test_operation")
        time.sleep(0.1)  # Small delay
        duration = self.monitor.end_operation_timer(op_id)
        
        # Verify duration is reasonable (greater than 0.09 seconds)
        self.assertGreater(duration, 90)  # at least 90ms
        
        # Test non-existent operation
        self.assertIsNone(self.monitor.end_operation_timer("nonexistent"))
    
    def test_time_operation_context_manager(self):
        """Test the time_operation context manager."""
        # Count operations before
        initial_operations = self.monitor.metrics["stats"]["total_operations"]
        
        # Use context manager
        with self.monitor.time_operation("test_context_manager"):
            time.sleep(0.1)  # Small delay
        
        # Verify operation was recorded
        self.assertEqual(self.monitor.metrics["stats"]["total_operations"], initial_operations + 1)
        
        # Check that the timer has completed
        completed_timers = [timer for timer in self.monitor.op_timers.values() if timer["completed"]]
        self.assertGreaterEqual(len(completed_timers), 1)
    
    def test_cleanup_old_metrics(self):
        """Test cleaning up old metrics."""
        # Add some test events with old timestamps
        current_time = time.time()
        old_time = current_time - (25 * 60 * 60)  # 25 hours ago (older than the 24-hour cutoff)
        
        # Add old event
        self.monitor.metrics["events"].append({
            "timestamp": old_time,
            "datetime": "2023-01-01T00:00:00",
            "type": "test",
            "data": {"message": "old event"}
        })
        
        # Add old history points
        very_old_time = current_time - (8 * 24 * 60 * 60)  # 8 days ago (older than the 7-day cutoff)
        self.monitor.metrics["history"]["timestamps"].append(very_old_time)
        self.monitor.metrics["history"]["memory_usage"].append(50.0)
        self.monitor.metrics["history"]["cpu_usage"].append(30.0)
        self.monitor.metrics["history"]["disk_usage"].append(40.0)
        self.monitor.metrics["history"]["query_count"].append(5)
        
        # Run cleanup
        self.monitor._cleanup_old_metrics()
        
        # Verify old events were removed
        for event in self.monitor.metrics["events"]:
            self.assertGreater(event["timestamp"], current_time - (24 * 60 * 60))
        
        # Verify old history points were removed
        for ts in self.monitor.metrics["history"]["timestamps"]:
            self.assertGreater(ts, current_time - (7 * 24 * 60 * 60))
    
    @patch("psutil.disk_usage")
    def test_get_disk_usage_error_handling(self, mock_disk_usage):
        """Test error handling in get_disk_usage method."""
        # Test OSError exception
        mock_disk_usage.side_effect = OSError("Permission denied")
        
        # Call method
        result = self.monitor.get_disk_usage("/nonexistent")
        
        # Verify error is handled properly
        self.assertIn("error", result)
        self.assertIn("Disk access error", result["error"])
        self.assertEqual(result["percent"], 0)
        self.assertEqual(result["free"], 0)
        self.assertEqual(result["total"], 0)
        
        # Test general Exception
        mock_disk_usage.side_effect = Exception("Other error")
        result = self.monitor.get_disk_usage()
        self.assertIn("error", result)


class TestGlobalFunctions(unittest.TestCase):
    """Tests for the global utility functions in diagnostics module."""
    
    def setUp(self):
        """Set up test environment."""
        # Ensure any previous monitor is stopped
        stop_monitoring()
    
    def tearDown(self):
        """Clean up after tests."""
        stop_monitoring()
    
    def test_get_monitor(self):
        """Test getting global monitor instance."""
        monitor = get_monitor()
        self.assertIsInstance(monitor, SystemMonitor)
        
        # Calling again should return the same instance
        monitor2 = get_monitor()
        self.assertIs(monitor2, monitor)
    
    def test_start_stop_global_monitoring(self):
        """Test starting and stopping global monitoring."""
        # Start monitoring
        self.assertTrue(start_monitoring())
        
        # Get monitor and check it's running
        monitor = get_monitor()
        self.assertTrue(monitor.running)
        
        # Stop monitoring
        stop_monitoring()
        
        # Get a new monitor (old one should be gone)
        new_monitor = get_monitor()
        self.assertFalse(new_monitor.running)
        self.assertIsNot(new_monitor, monitor)
    
    def test_global_log_event(self):
        """Test global log_event function."""
        # Get monitor
        monitor = get_monitor()
        initial_events = len(monitor.metrics["events"])
        
        # Log event
        log_event("info", {"message": "test global event"})
        
        # Verify event was added
        self.assertEqual(len(monitor.metrics["events"]), initial_events + 1)
        self.assertEqual(monitor.metrics["events"][-1]["data"]["message"], "test global event")
    
    def test_global_time_operation(self):
        """Test global time_operation function."""
        # Define test function
        def test_func(x, y):
            time.sleep(0.1)
            return x + y
        
        # Time the function
        result = time_operation("test_global_func", test_func, 5, 7)
        
        # Verify result
        self.assertEqual(result, 12)
        
        # Check monitoring stats
        monitor = get_monitor()
        self.assertGreaterEqual(monitor.metrics["stats"]["total_operations"], 1)


if __name__ == "__main__":
    unittest.main() 