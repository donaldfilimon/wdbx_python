# WDBX Diagnostics Module

The WDBX Diagnostics module provides a comprehensive system for monitoring and collecting metrics about your application's performance and resource usage. This document explains how to use this module effectively.

## Basic Usage

### Starting and Stopping Monitoring

The simplest way to use the diagnostics module is through the global monitoring functions:

```python
from wdbx.utils import start_monitoring, stop_monitoring, get_metrics

# Start monitoring system resources
start_monitoring()

# Your application code here...

# Get current metrics
metrics = get_metrics()
print(f"Memory usage: {metrics['memory_usage']}%")
print(f"CPU usage: {metrics['cpu_usage']}%")

# Stop monitoring when done
stop_monitoring()
```

### Timing Operations

The diagnostics module provides two ways to time operations:

#### Using the Context Manager (Recommended)

```python
from wdbx.utils import time_this

# Time a block of code
with time_this("database_query"):
    # Your database query code here
    results = db.execute_query("SELECT * FROM users")
```

#### Using the Function Wrapper

```python
from wdbx.utils import time_operation

def my_function(x, y):
    # Function code here
    return x + y

# Time the function call
result = time_operation("my_operation", my_function, 5, 7)
```

### Logging Events

You can log custom events to track important moments in your application:

```python
from wdbx.utils import log_event

# Log an informational event
log_event("info", {
    "message": "User logged in",
    "user_id": 12345,
    "ip_address": "192.168.1.1"
})

# Log a warning event
log_event("warning", {
    "message": "High memory usage detected",
    "memory_percent": 87.5
})

# Log an error event
log_event("error", {
    "message": "Database connection failed",
    "error": "Connection timeout"
})
```

## Advanced Usage

### Creating a Custom Monitor

You can create your own instance of `SystemMonitor` with custom settings:

```python
from wdbx.utils.diagnostics import SystemMonitor

# Create a custom monitor with shorter check interval
monitor = SystemMonitor(
    check_interval=1,  # Check every second
    max_history_points=500,  # Store up to 500 data points
    threshold_memory_percent=90.0,  # Higher memory threshold
    auto_start=True  # Start monitoring immediately
)

# Use the custom monitor's methods
monitor.log_event("info", {"message": "Using custom monitor"})

# Use the context manager directly on the monitor
with monitor.time_operation("custom_operation"):
    # Your code here
    pass

# Stop the custom monitor when done
monitor.stop_monitoring()
```

### Monitoring Custom Components

You can register your own components to be monitored:

```python
from wdbx.utils import register_component

class MyComponent:
    def __init__(self):
        self.query_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
    def get_metrics(self):
        # This method is called by the monitor to collect metrics
        return {
            "queries": self.query_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": self.cache_hits / (self.cache_hits + self.cache_misses + 0.001)
        }
        
    def perform_query(self):
        self.query_count += 1
        # Query logic...

# Create your component
my_component = MyComponent()

# Register it with the monitoring system
register_component("my_custom_component", my_component)
```

## Memory Management

The diagnostics module includes automatic memory management to prevent excessive memory usage during long-running operations:

- Event logs older than 24 hours are automatically removed
- Metric history data points older than 7 days are automatically removed
- Operation timers are cleaned up when they are no longer needed

## Error Handling

The diagnostics module includes robust error handling to prevent monitoring issues from affecting your application:

- If `psutil` is not available, the module will still work but with limited functionality
- Disk usage errors are properly handled and won't cause your application to crash
- All monitoring operations are performed in a separate thread to avoid impacting main application performance

## Best Practices

1. **Always Stop Monitoring**: Call `stop_monitoring()` when your application exits to properly clean up resources
2. **Use Context Managers**: Prefer using `time_this()` context manager over function wrappers for better readability
3. **Be Selective with Events**: Log only important events to avoid cluttering the event log
4. **Monitor Key Components**: Register components that are critical to your application's performance
5. **Set Appropriate Thresholds**: Adjust memory and CPU thresholds based on your specific environment

## Integration with Other WDBX Components

The diagnostics module integrates seamlessly with other WDBX components:

- **Storage Engine**: Monitor database operations and query performance
- **Cache**: Track cache hit/miss ratios and cache size
- **Network**: Monitor API requests and response times
- **ML Models**: Track inference times and resource usage

## Advanced Configuration

For detailed control over the monitoring system, you can modify the following settings:

```python
from wdbx.utils.diagnostics import get_monitor

monitor = get_monitor()

# Adjust check interval dynamically
monitor.check_interval = 10  # Check every 10 seconds

# Change memory threshold
monitor.threshold_memory_percent = 80.0  # Alert at 80% memory usage

# Adjust history size
monitor.max_history_points = 2000  # Store more history points
```

## Troubleshooting

If you encounter issues with the diagnostics module:

1. **Check psutil Installation**: Make sure psutil is installed (`pip install psutil`)
2. **Verify Thread Safety**: If you're using custom components, ensure they are thread-safe
3. **Review Logs**: Check the application logs for any diagnostics-related errors
4. **Memory Usage**: If memory usage is high, consider reducing `max_history_points` 