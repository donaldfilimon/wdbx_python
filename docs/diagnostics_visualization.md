# Diagnostics Visualization

<!-- category: Development -->
<!-- priority: 66 -->
<!-- tags: visualization, monitoring, dashboards, metrics -->

This guide explains how to visualize WDBX diagnostic data.

## Overview

WDBX provides visualization tools for:

- Performance metrics
- System health
- Resource usage
- Error patterns

## Visualization Types

### Interactive Dashboards

```python
from wdbx.viz import Dashboard

# Create dashboard
dash = Dashboard()

# Add metrics panel
dash.add_panel("metrics", {
    "cpu_usage": True,
    "memory_usage": True,
    "throughput": True
})

# Add system health panel
dash.add_panel("health", {
    "services": True,
    "plugins": True
})

# Start dashboard server
dash.serve(port=8050)
```

### Static Reports

```python
from wdbx.viz import Report

# Generate PDF report
report = Report()
report.add_metrics_summary()
report.add_health_status()
report.save("diagnostics.pdf")
```

## Performance Visualization

### Time Series Data

```python
from wdbx.viz import TimeSeriesPlot

# Create time series plot
plot = TimeSeriesPlot()

# Add metrics
plot.add_metric("cpu_usage", color="blue")
plot.add_metric("memory_usage", color="red")

# Show plot
plot.show()
```

### Heatmaps

```python
from wdbx.viz import Heatmap

# Create vector operation heatmap
hmap = Heatmap()
hmap.plot_vector_operations()
hmap.show()
```

## System Health Visualization

### Status Dashboard

```python
from wdbx.viz import HealthDashboard

# Create health dashboard
health = HealthDashboard()

# Add components
health.add_component("database")
health.add_component("cache")
health.add_component("api")

# Update status
health.update()
```

### Alert Visualization

```python
from wdbx.viz import AlertVisualizer

# Create alert visualizer
viz = AlertVisualizer()

# Plot alert history
viz.plot_alerts(
    start_time="2024-01-01",
    end_time="2024-04-01"
)
```

## Resource Usage

### Memory Profiling

```python
from wdbx.viz import MemoryProfiler

# Create memory profiler
prof = MemoryProfiler()

# Record memory usage
with prof.record():
    process_large_dataset()

# Show memory profile
prof.show()
```

### CPU Usage

```python
from wdbx.viz import CPUProfiler

# Create CPU profiler
prof = CPUProfiler()

# Profile function
prof.profile(process_vectors)

# Show flame graph
prof.show_flame_graph()
```

## Error Analysis

### Error Patterns

```python
from wdbx.viz import ErrorAnalyzer

# Create error analyzer
analyzer = ErrorAnalyzer()

# Analyze error patterns
patterns = analyzer.find_patterns()

# Visualize patterns
analyzer.plot_patterns()
```

### Error Distribution

```python
from wdbx.viz import ErrorDistribution

# Create distribution plot
dist = ErrorDistribution()

# Plot error types
dist.plot_by_type()
dist.plot_by_time()
```

## Custom Visualizations

### Custom Metrics

```python
from wdbx.viz import MetricsPlot

# Create custom plot
plot = MetricsPlot()

# Add custom metric
@plot.metric
def custom_metric():
    return calculate_custom_value()

# Show plot
plot.show()
```

### Custom Dashboards

```python
from wdbx.viz import CustomDashboard

# Create custom dashboard
dash = CustomDashboard()

# Add custom panel
dash.add_custom_panel(
    name="My Panel",
    data_source=my_data_function,
    update_interval=60
)
```

## Export Options

### Data Export

```python
from wdbx.viz import DataExporter

# Create exporter
exporter = DataExporter()

# Export metrics
exporter.export_metrics("metrics.csv")
exporter.export_health("health.json")
```

### Image Export

```python
from wdbx.viz import PlotExporter

# Create plot
plot = TimeSeriesPlot()
plot.add_metric("throughput")

# Export as image
plot.export("throughput.png", dpi=300)
```

## Best Practices

1. Use appropriate visualization types
2. Set meaningful time ranges
3. Include context in plots
4. Use consistent color schemes
5. Add clear labels and titles

## Configuration

### Theme Configuration

```python
from wdbx.viz import set_theme

# Set visualization theme
set_theme({
    "background": "dark",
    "colors": ["#3366cc", "#dc3912", "#ff9900"],
    "font": "Arial"
})
```

### Layout Settings

```python
from wdbx.viz import configure_layout

# Configure dashboard layout
configure_layout({
    "columns": 2,
    "padding": 20,
    "spacing": 10
})
```

## Integration

### Jupyter Integration

```python
from wdbx.viz.jupyter import init_notebook

# Initialize notebook integration
init_notebook()

# Create interactive plot
plot = TimeSeriesPlot()
plot.show_notebook()
```

### Web Integration

```python
from wdbx.viz.web import WebDashboard

# Create web dashboard
dash = WebDashboard()
dash.serve(host="0.0.0.0", port=8080)
```

## Recent Improvements

### T-SNE Visualization Enhancements

The t-SNE visualization component has been improved to handle input data more robustly:

- Added proper NumPy array conversion for input vectors
- Fixed the "list object has no attribute 'shape'" error that could occur with certain data structures
- Improved perplexity parameter adjustment based on the number of vectors
- Enhanced error handling for better feedback when visualization fails

```python
from wdbx.viz import TSNEVisualizer

# Create a t-SNE visualizer with improved robustness
viz = TSNEVisualizer()

# Visualize vectors with automatic conversion
viz.visualize(vectors, labels=vector_ids)

# Save visualization
viz.save("tsne_visualization.png", dpi=300)
```

### Matplotlib Integration Updates

The visualization module has been updated to use the latest Matplotlib APIs:

- Replaced deprecated `get_cmap` function with the newer `colormaps` approach
- Added fallback mechanism for compatibility with older Matplotlib versions
- Improved color handling across different visualizations for consistency

```python
from wdbx.viz import ColorManager

# Get a color scheme using the new API
color_manager = ColorManager()
colors = color_manager.get_colormap("viridis", num_colors=10)

# Use in any visualization
viz.set_colors(colors)
``` 