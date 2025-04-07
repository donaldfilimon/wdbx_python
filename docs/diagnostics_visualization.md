# WDBX Diagnostics Visualization

The WDBX Diagnostics Visualization module provides tools for creating dynamic, interactive visualizations of system metrics, component performance, and events. This document explains how to use the visualization capabilities effectively.

## Installation Requirements

The visualization module has optional dependencies on popular Python visualization libraries:

```bash
# Install with matplotlib support (static visualizations)
pip install matplotlib

# Install with plotly support (interactive visualizations)
pip install plotly

# Install both for maximum flexibility
pip install matplotlib plotly
```

The module will automatically detect which libraries are available and use the appropriate one. If both are available, you can choose which to use.

## Basic Usage

### Getting a Visualizer

The easiest way to start is by getting a visualizer that uses the global monitor:

```python
from wdbx.utils.diagnostics_viz import get_visualizer

# Get a visualizer using the global monitor
visualizer = get_visualizer()

# Check if visualization is supported
if visualizer.check_visualization_support():
    # Visualization libraries are available
    pass
else:
    # No visualization libraries available
    print("Please install matplotlib or plotly")
```

### Creating a System Overview

The system overview visualization shows key system metrics:

```python
# Create an interactive Plotly visualization and save to file
visualizer.create_system_overview(
    output_file="system_overview.html",  # Save to file (None to display only)
    time_range_minutes=60,               # Show the last 60 minutes
    use_plotly=True,                     # Use Plotly for interactive charts
    fig_size=(12, 8),                    # Figure size in inches
    dark_mode=True                       # Use dark theme
)

# Create a static Matplotlib visualization
fig = visualizer.create_system_overview(
    output_file="system_overview.png",   # Save to PNG file
    time_range_minutes=30,               # Show the last 30 minutes
    use_plotly=False,                    # Use Matplotlib
    dark_mode=False                      # Use light theme
)

# With Matplotlib, you can further customize the figure
if fig:
    # Add custom annotations or modifications
    plt.title("Custom System Overview")
    plt.savefig("custom_overview.png")
    plt.show()
```

### Creating a Component Dashboard

The component dashboard shows metrics for specific components:

```python
# Create a dashboard for all registered components
visualizer.create_component_dashboard(
    output_file="component_dashboard.html",
    time_range_minutes=60,
    use_plotly=True
)

# Create a dashboard for specific components
visualizer.create_component_dashboard(
    component_names=["database", "api", "cache"],
    output_file="selected_components.html",
    fig_size=(14, 10)
)
```

### Creating an Events Timeline

The events timeline visualizes events recorded by the monitoring system:

```python
# Create a timeline of all events in the last 24 hours
visualizer.create_events_timeline(
    output_file="events_timeline.html",
    time_range_hours=24,
    use_plotly=True
)

# Create a timeline of specific event types
visualizer.create_events_timeline(
    event_types=["error", "warning"],  # Only show errors and warnings
    output_file="errors_timeline.html",
    time_range_hours=48,               # Show the last 48 hours
    use_plotly=True
)
```

## Using Custom Monitors

You can create a visualizer for a custom monitor:

```python
from wdbx.utils.diagnostics import SystemMonitor
from wdbx.utils.diagnostics_viz import DiagnosticsVisualizer

# Create a custom monitor
custom_monitor = SystemMonitor(
    check_interval=1,
    max_history_points=1000
)

# Create a visualizer using the custom monitor
visualizer = DiagnosticsVisualizer(custom_monitor)
```

## Plotly vs. Matplotlib

The visualization module supports both Plotly and Matplotlib:

| Feature | Plotly | Matplotlib |
|---------|--------|------------|
| Interactivity | ✅ Interactive (zoom, pan, hover) | ❌ Static images |
| Output | HTML files | PNG, PDF, SVG, etc. |
| Dependencies | Heavier | Lighter |
| Customization | Via Python or JavaScript | Via Python |
| In-browser | ✅ Can be viewed in any browser | ❌ Requires image viewer |

Choose Plotly for:
- Interactive dashboards
- Web integration
- Sharing with non-technical users

Choose Matplotlib for:
- Static reports
- Publication-quality figures
- Lower dependencies

## Advanced Usage

### Integrating with Web Applications

You can integrate Plotly visualizations into web applications:

```python
from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def dashboard():
    visualizer = get_visualizer()
    
    # Create a Plotly figure
    fig = visualizer.create_system_overview(use_plotly=True)
    
    # Convert to HTML
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Render in template
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>WDBX Monitoring Dashboard</title>
        </head>
        <body>
            <h1>System Overview</h1>
            {{ plot_html|safe }}
        </body>
        </html>
    """, plot_html=plot_html)

if __name__ == '__main__':
    app.run(debug=True)
```

### Creating Custom Visualizations

You can access the raw metrics and create custom visualizations:

```python
import matplotlib.pyplot as plt
import numpy as np

visualizer = get_visualizer()
monitor = visualizer.monitor

# Access raw metrics
metrics = monitor.metrics
history = metrics["history"]
timestamps = history["timestamps"]
memory_usage = history["memory_usage"]

# Create custom visualization
plt.figure(figsize=(10, 6))
plt.plot(timestamps, memory_usage, 'b-', label='Memory Usage')

# Add a trend line (e.g., moving average)
window_size = 10
if len(memory_usage) >= window_size:
    moving_avg = np.convolve(memory_usage, np.ones(window_size)/window_size, mode='valid')
    plt.plot(timestamps[window_size-1:], moving_avg, 'r-', label=f'{window_size}-point Moving Average')

plt.title('Custom Memory Usage Visualization')
plt.xlabel('Time')
plt.ylabel('Memory Usage (%)')
plt.legend()
plt.grid(True)
plt.savefig('custom_visualization.png')
plt.show()
```

## Best Practices

1. **Choose the Right Tool**: Use Plotly for interactive dashboards and Matplotlib for static reports.
2. **Set Appropriate Time Ranges**: Show relevant time periods (e.g., last hour for recent issues, last day for trends).
3. **Use Dark Mode for Dashboards**: Dark mode reduces eye strain for monitoring dashboards viewed for extended periods.
4. **Save to Files**: Save visualizations to files for sharing and documentation.
5. **Filter Components**: Focus on specific components when investigating performance issues.
6. **Filter Event Types**: Show only relevant event types (e.g., errors, warnings) when troubleshooting.

## Troubleshooting

If you encounter issues with visualizations:

1. **Check Library Installation**: Ensure matplotlib or plotly is installed.
2. **Verify Data Availability**: Ensure you have collected enough metrics (run your application for a few minutes).
3. **Check Output Directory**: Ensure the output directory exists and is writable.
4. **Adjust Time Range**: If no data appears, try increasing the time range.
5. **Check Component Registration**: Ensure components are registered with the monitoring system.

## Performance Considerations

- Creating visualizations requires processing the metrics data, which can be memory-intensive for large datasets.
- For high-frequency metrics collection, consider filtering data before visualization.
- Use appropriate `time_range_minutes` or `time_range_hours` parameters to limit the amount of data processed.
- For long-running applications, focus on recent data to avoid processing the entire history.

## Integration with Monitoring Systems

The visualization module complements the diagnostics exporters:

- Use **exporters** for long-term storage, alerts, and integration with enterprise monitoring systems.
- Use **visualizations** for interactive analysis, debugging, and ad-hoc investigations.

For the best results, use both approaches:

```python
from wdbx.utils import start_monitoring
from wdbx.utils.diagnostics_exporters import PrometheusExporter
from wdbx.utils.diagnostics_viz import get_visualizer

# Start monitoring
start_monitoring()

# Setup exporter for continuous monitoring
exporter = PrometheusExporter(
    export_interval=60,
    pushgateway_url="http://localhost:9091",
    auto_start=True
)

# Use visualizer for analysis when needed
visualizer = get_visualizer()
``` 