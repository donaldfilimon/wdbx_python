# Contributing to WDBX Python

Thank you for considering contributing to WDBX Python! This document provides guidelines to help maintain code quality and avoid common issues.

## Avoiding Circular Import Issues

Circular imports have been a challenge in this codebase. Please follow these best practices to avoid them:

### 1. Use TYPE_CHECKING for Type Hints

For imports that are only needed for type hints, use the `TYPE_CHECKING` pattern:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .other_module import SomeClass
```

This ensures that the import only happens during static type checking and not at runtime.

### 2. Leverage Module `__init__.py` Files

When appropriate, define placeholder classes in `__init__.py` files that can be imported safely, then use dynamic imports for the actual implementations.

### 3. Use Dynamic Imports

For imports that are only needed at specific times, use dynamic imports within functions:

```python
def get_some_class():
    from .module import SomeClass
    return SomeClass
```

### 4. Use Centralized Constants

Always import constants from the central constants module:

```python
from wdbx.core.constants import VECTOR_DIMENSION, DEFAULT_SIMILARITY_THRESHOLD
```

If you need to add new constants, add them to `src/wdbx/core/constants.py` instead of defining them locally.

### 5. Standardized Logging

Use the logging utilities for consistent logging across the project:

```python
from wdbx.utils.logging_utils import get_logger

logger = get_logger("WDBX.MyComponent")
```

### 6. Import Structure

Follow this import order:
1. Standard library imports
2. Third-party library imports
3. Type-checking imports
4. Utility and constants imports from WDBX
5. Local module imports

## Memory Efficiency Guidelines

To maintain memory efficiency in this codebase, please follow these guidelines:

### 1. Use Efficient Data Structures

- For large collections, consider using NumPy arrays or memory-mapped arrays instead of Python lists
- Use generators and iterators for processing large datasets
- Consider specialized libraries like FAISS for vector operations

### 2. Clean Up Resources

- Implement `close()` methods for classes that own resources
- Use context managers (`with` statements) for resource management
- Release large objects explicitly when no longer needed

### 3. Memory Monitoring

- Add memory checks for long-running operations
- Use the `optimize_memory` pattern for components that might consume large amounts of memory
- Include memory usage logging for debugging

Example:

```python
def process_large_dataset(self, data):
    # Process in chunks to avoid memory issues
    for chunk in self._get_chunks(data):
        self._process_chunk(chunk)
        
        # Periodically check and optimize memory
        if self._should_check_memory():
            self.optimize_memory()
```

### 4. System Diagnostics for Development

When developing or troubleshooting memory issues, use the diagnostics utilities to monitor system behavior:

```python
from wdbx.utils.diagnostics import system_diagnostics

# Register your components
system_diagnostics.register_component("my_component", my_component)

# Start monitoring during development/testing
system_diagnostics.start_monitoring(interval=5)  # Check every 5 seconds

# Your test or development code...
# ...

# Check memory at key points
memory_info = system_diagnostics.get_memory_usage()
print(f"Memory usage: {memory_info['percent']:.1f}%")

# Force optimization when needed during testing
if memory_info['percent'] > 70:
    system_diagnostics.optimize_memory()
    
# Log important operations for later analysis
system_diagnostics.log_event("test_operation", {
    "operation": "insert_batch",
    "batch_size": 1000,
    "duration_ms": 450
})

# Get metrics history for reporting
metrics = system_diagnostics.get_metrics_history()

# Stop monitoring
system_diagnostics.stop_monitoring()
```

When profiling memory usage in tests:
1. Set thresholds lower for earlier detection (`MAX_MEMORY_PERCENT=50`)
2. Monitor shorter intervals (`MEMORY_CHECK_INTERVAL=1`)
3. Check metrics history to identify memory growth patterns
4. Use the system info report for detailed diagnostics
5. Consider using `tracemalloc` for specific object tracking

When profiling performance:
1. Enable monitoring to collect timing data for operations.
2. Check `metrics['operation_timing']` in the metrics history.
3. Analyze average and maximum durations for key operations.
4. Identify bottlenecks by looking for operations with high durations.

## Performance Profiling

WDBX includes a powerful performance profiling system that makes it easy to identify bottlenecks and optimize critical code paths.

### Using the Performance Profiler

```python
from wdbx.utils.diagnostics import get_performance_profiler

profiler = get_performance_profiler()

# Profile a function with decorator
@profiler.profile("vector_search")
def search_vectors(query, top_k=10):
    # Implementation...
    return results

# Profile a code block with context manager
def process_batch(items):
    # Setup...
    
    with profiler.profile_block("data_processing"):
        # Only this block will be profiled
        for item in items:
            process_item(item)
    
    # Cleanup...
```

### Analyzing Performance Data

```python
# Get statistics for a specific operation
stats = profiler.get_statistics("vector_search")

# Key metrics to analyze
print(f"Average duration: {stats['avg_duration_ms']:.2f} ms")
print(f"95th percentile: {stats['p95_duration_ms']:.2f} ms")
print(f"Memory impact: {stats['avg_memory_delta'] / 1024:.2f} KB")
print(f"Success rate: {stats['success_rate']:.1f}%")

# Compare multiple operations
for op in profiler.get_all_operations():
    stats = profiler.get_statistics(op)
    print(f"{op}: {stats['avg_duration_ms']:.2f} ms, {stats['calls_per_second']:.2f} calls/s")
```

### Best Practices for Performance Analysis

1. **Focus on Critical Paths**: Profile operations that are executed frequently or have high impact
2. **Use p95 for Stability**: The 95th percentile is more reliable than averages for performance guarantees
3. **Check Memory Impact**: Operations with high or growing memory usage may indicate leaks
4. **Reset Between Tests**: Use `profiler.reset_statistics()` before each test for clean measurements
5. **Compare Before/After**: Record baseline metrics before making optimizations for valid comparisons

For detailed information, see the [Performance Profiling Guide](docs/performance_profiling.md).

## Code Style

- Use type hints for all function parameters and return values
- Document classes and methods with docstrings
- Follow PEP 8 for code style
- Write unit tests for new functionality

## Pull Request Process

1. Ensure all tests pass with `python test_imports.py`
2. Update documentation if necessary
3. Add tests for new functionality
4. Submit a detailed PR description explaining the changes and their purpose 