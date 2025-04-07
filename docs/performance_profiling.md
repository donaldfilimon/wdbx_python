# WDBX Performance Profiling Guide

This guide explains how to use the WDBX performance profiling system to identify bottlenecks, optimize critical operations, and monitor application performance.

## Overview

The WDBX Performance Profiling system allows you to:

- Measure execution time of operations with high precision
- Track memory usage impact of specific code blocks
- Calculate statistical performance metrics (min, max, average, percentiles)
- Monitor call frequencies and success rates
- Identify performance trends over time
- Generate detailed performance reports

## Getting Started

### Basic Profiling

The simplest way to profile a function is to use the `profile` decorator:

```python
from wdbx.utils.diagnostics import get_performance_profiler

profiler = get_performance_profiler()

@profiler.profile("vector_search")
def search_vectors(query, top_k=10):
    # Your implementation here
    return results
```

This automatically tracks:
- Execution time
- Memory impact
- Success/failure rate
- Call frequency

### Profiling Code Blocks

For more granular profiling, use the context manager:

```python
from wdbx.utils.diagnostics import get_performance_profiler

profiler = get_performance_profiler()

def process_batch(items):
    # Some initial setup
    
    with profiler.profile_block("batch_processing"):
        # Only this block will be profiled
        for item in items:
            process_item(item)
            
    # Cleanup code here
```

### Analyzing Performance Data

To retrieve statistics for a profiled operation:

```python
stats = profiler.get_statistics("vector_search")

print(f"Call count: {stats['call_count']}")
print(f"Average duration: {stats['avg_duration_ms']:.2f} ms")
print(f"95th percentile: {stats['p95_duration_ms']:.2f} ms")
print(f"Success rate: {stats['success_rate']:.1f}%")
print(f"Memory impact: {stats['avg_memory_delta'] / 1024:.2f} KB per call")
print(f"Calls per second: {stats['calls_per_second']:.2f}")
```

### Profiling Multiple Operations

You can profile multiple operations and compare their performance:

```python
# Get a list of all profiled operations
operations = profiler.get_all_operations()

print("Performance Comparison:")
print("-" * 50)
print(f"{'Operation':<30} {'Avg (ms)':<10} {'Max (ms)':<10} {'Calls/s':<8}")
print("-" * 50)

for op in operations:
    stats = profiler.get_statistics(op)
    print(f"{op:<30} {stats['avg_duration_ms']:<10.2f} {stats['max_duration_ms']:<10.2f} {stats['calls_per_second']:<8.2f}")
```

### Resetting Statistics

To reset statistics for a fresh measurement:

```python
# Reset a specific operation
profiler.reset_statistics("vector_search")

# Reset all operations
profiler.reset_statistics()
```

### Generating Performance Reports

You can easily generate comprehensive performance reports in text or markdown format:

```python
from wdbx.utils.diagnostics import generate_performance_report

# Generate a plain text report
text_report = generate_performance_report()
print(text_report)

# Generate a markdown report (for documentation or GitHub)
md_report = generate_performance_report(format_type="markdown")
with open("performance_report.md", "w") as f:
    f.write(md_report)
```

The generated report includes:
- Summary statistics across all operations
- Detailed performance metrics for each operation
- Error analysis for operations with failures
- Recommendations for performance optimization

Example report:

```
WDBX PERFORMANCE REPORT
==================================================

Total Operations Profiled: 8
Total Function Calls: 1247
Operations with Errors: 1

OPERATION DETAILS
--------------------------------------------------------------------------------
Operation                       Calls     Avg (ms)   P95 (ms)   Max (ms)   Mem (KB)   Success %
--------------------------------------------------------------------------------
vector_store.search_similar     342       75.32      124.56     198.76     12.34      100.0    
block_manager.create_block      185       42.18      68.92      87.34      45.67      99.5     
vector_store.add                458       18.45      32.17      45.89      2.45       100.0    
...

ERRORS
--------------------------------------------------
Operation                       Error Count      Error Rate %     
--------------------------------------------------
block_manager.create_block      1                0.5              

PERFORMANCE RECOMMENDATIONS
--------------------------------------------------
* Optimize vector_store.search_similar: Average duration 75.32ms
* Optimize block_manager.create_block: Average duration 42.18ms
* Optimize vector_store.add: Average duration 18.45ms
* Review memory usage in block_manager.create_block: Average impact 45.67KB per call
```

## Advanced Usage

### Integration with Monitoring Systems

The profiling data is automatically integrated with the WDBX diagnostics system, allowing you to:

- Track performance trends over time
- Set up alerts for performance degradation
- Generate performance reports

```python
from wdbx.utils.diagnostics import system_diagnostics

# Get all metrics including profiling data
metrics = system_diagnostics.get_metrics_history()

# Access profiler data for a specific operation
vector_search_metrics = metrics['profiler_data'].get('vector_search', [])

# Example: Calculate moving average of the last 10 calls
if vector_search_metrics:
    recent_durations = [m['duration_ms'] for m in vector_search_metrics[-10:]]
    moving_avg = sum(recent_durations) / len(recent_durations)
    print(f"Recent average execution time: {moving_avg:.2f} ms")
```

### Identifying Memory Leaks

The profiler can help identify potential memory leaks by tracking memory delta over time:

```python
for op in profiler.get_all_operations():
    stats = profiler.get_statistics(op)
    if stats['total_memory_impact'] > 10 * 1024 * 1024:  # 10 MB
        print(f"Warning: Operation '{op}' has accumulated {stats['total_memory_impact'] / (1024*1024):.2f} MB")
```

### Performance Comparison Between Components

Compare performance metrics across different system components:

```python
# Profile vector store operations
vector_store = OptimizedVectorStore()
with profiler.profile_block("vector_store.add"):
    vector_store.add(vector_id, vector)

# Profile block manager operations
block_manager = OptimizedBlockManager()
with profiler.profile_block("block_manager.create_block"):
    block_manager.create_block_batch(blocks)

# Compare performance
vs_stats = profiler.get_statistics("vector_store.add")
bm_stats = profiler.get_statistics("block_manager.create_block")

print(f"Vector Store Add: {vs_stats['avg_duration_ms']:.2f} ms")
print(f"Block Manager Create: {bm_stats['avg_duration_ms']:.2f} ms")
```

## Best Practices

1. **Profile Strategically**: Focus on profiling critical operations and suspected bottlenecks, not everything.

2. **Consider Overhead**: The profiler itself adds a small overhead. For extremely time-sensitive operations, consider enabling profiling only in development/testing.

3. **Reset When Appropriate**: Reset statistics after configuration changes or optimizations to get clean measurements.

4. **Monitor Memory Impact**: Pay close attention to memory impact statistics to identify memory-intensive operations.

5. **Use Context Managers for Granularity**: When profiling complex functions, use context managers to isolate specific sections rather than profiling the entire function.

6. **Review Periodically**: Set up regular performance reviews to track changes over time, especially after updates.

## Troubleshooting

### High Standard Deviation

If you see a high standard deviation in your timing metrics, it may indicate:
- Inconsistent workloads
- Background processes interfering
- Cache effects (cold vs. warm cache)
- Garbage collection pauses

Try profiling with more consistent workloads or increasing the sample size.

### Negative Memory Delta

A negative memory delta doesn't necessarily indicate a memory leak fix. It may be due to:
- Garbage collection timing
- Memory fragmentation
- System memory management

For accurate memory profiling, consider running multiple iterations and looking at the aggregate trend.

### High Error Count

If you notice a high error count for an operation:
1. Check exception handling
2. Verify input validation
3. Ensure resources are properly initialized
4. Look for race conditions in multi-threaded contexts

## Conclusion

Effective performance profiling is a key part of maintaining high-quality code. By regularly monitoring and analyzing performance metrics, you can identify problems early, optimize critical paths, and ensure your application remains responsive and efficient. 