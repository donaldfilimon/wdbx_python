"""
WDBX Diagnostics Visualization Module.

Provides utilities for visualizing diagnostics metrics through charts and dashboards.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Try to import visualization libraries, but make them optional
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import diagnostics
from wdbx.utils.diagnostics import SystemMonitor, get_monitor

# Configure logging
logger = logging.getLogger("wdbx.diagnostics.viz")


class DiagnosticsVisualizer:
    """Visualize WDBX diagnostics metrics."""
    
    def __init__(self, monitor: Optional[SystemMonitor] = None):
        """
        Initialize the visualizer.
        
        Args:
            monitor: SystemMonitor instance (or use global)
        """
        self.monitor = monitor or get_monitor()
        
        # Check visualization libraries availability
        if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
            logger.warning(
                "Neither matplotlib nor plotly are available. "
                "Install at least one: pip install matplotlib plotly"
            )
    
    def check_visualization_support(self) -> bool:
        """Check if visualization is supported."""
        return MATPLOTLIB_AVAILABLE or PLOTLY_AVAILABLE
    
    def create_system_overview(
        self,
        output_file: Optional[str] = None,
        time_range_minutes: int = 60,
        use_plotly: bool = True,
        fig_size: Tuple[int, int] = (12, 8),
        dark_mode: bool = False
    ) -> Optional[Union[Figure, go.Figure]]:
        """
        Create a system overview visualization.
        
        Args:
            output_file: Path to save the visualization (None for display only)
            time_range_minutes: Time range in minutes to include in visualization
            use_plotly: Whether to use plotly instead of matplotlib
            fig_size: Figure size (width, height) in inches
            dark_mode: Whether to use dark theme
            
        Returns:
            Matplotlib Figure or Plotly Figure based on use_plotly
        """
        # Check if visualization is supported
        if use_plotly and not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Falling back to matplotlib.")
            use_plotly = False
            
        if not use_plotly and not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Falling back to plotly.")
            use_plotly = True
            
        if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
            logger.error("No visualization libraries available.")
            return None
        
        # Filter data based on time range
        cutoff_time = time.time() - (time_range_minutes * 60)
        
        # Get data from monitor
        history = self.monitor.metrics["history"]
        timestamps = history["timestamps"]
        memory_usage = history["memory_usage"]
        cpu_usage = history["cpu_usage"]
        disk_usage = history["disk_usage"]
        query_count = history["query_count"]
        
        # Filter data by time range
        indices = [i for i, ts in enumerate(timestamps) if ts >= cutoff_time]
        if not indices:
            logger.warning(f"No data available in the last {time_range_minutes} minutes.")
            return None
            
        filtered_timestamps = [timestamps[i] for i in indices]
        filtered_memory = [memory_usage[i] for i in indices]
        filtered_cpu = [cpu_usage[i] for i in indices]
        filtered_disk = [disk_usage[i] for i in indices]
        filtered_queries = [query_count[i] for i in indices]
        
        # Convert timestamps to datetime
        datetime_labels = [datetime.fromtimestamp(ts) for ts in filtered_timestamps]
        
        if use_plotly and PLOTLY_AVAILABLE:
            return self._create_plotly_overview(
                datetime_labels,
                filtered_memory,
                filtered_cpu,
                filtered_disk,
                filtered_queries,
                output_file,
                fig_size,
                dark_mode
            )
        if MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_overview(
                datetime_labels,
                filtered_memory,
                filtered_cpu,
                filtered_disk,
                filtered_queries,
                output_file,
                fig_size,
                dark_mode
            )
        logger.error("No visualization libraries available.")
        return None
    
    def _create_plotly_overview(
        self,
        timestamps: List[datetime],
        memory_usage: List[float],
        cpu_usage: List[float],
        disk_usage: List[float],
        query_count: List[int],
        output_file: Optional[str],
        fig_size: Tuple[int, int],
        dark_mode: bool
    ) -> go.Figure:
        """Create system overview using Plotly."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Memory Usage (%)",
                "CPU Usage (%)",
                "Disk Usage (%)",
                "Query Count"
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=memory_usage,
                name="Memory",
                line=dict(color="#2E86C1", width=2),
                fill="tozeroy",
                fillcolor="rgba(46, 134, 193, 0.2)"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=cpu_usage,
                name="CPU",
                line=dict(color="#2ECC71", width=2),
                fill="tozeroy",
                fillcolor="rgba(46, 204, 113, 0.2)"
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=disk_usage,
                name="Disk",
                line=dict(color="#E67E22", width=2),
                fill="tozeroy",
                fillcolor="rgba(230, 126, 34, 0.2)"
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=query_count,
                name="Queries",
                line=dict(color="#9B59B6", width=2),
                fill="tozeroy",
                fillcolor="rgba(155, 89, 182, 0.2)"
            ),
            row=2, col=2
        )
        
        # Update layout
        template = "plotly_dark" if dark_mode else "plotly_white"
        
        fig.update_layout(
            title="WDBX System Overview",
            width=fig_size[0] * 100,  # Convert inches to pixels (approximate)
            height=fig_size[1] * 100,
            template=template,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Set y-axis ranges for resource usage
        fig.update_yaxes(range=[0, 100], row=1, col=1)
        fig.update_yaxes(range=[0, 100], row=1, col=2)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        
        # Save if output file provided
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Saved visualization to {output_file}")
        
        return fig
    
    def _create_matplotlib_overview(
        self,
        timestamps: List[datetime],
        memory_usage: List[float],
        cpu_usage: List[float],
        disk_usage: List[float],
        query_count: List[int],
        output_file: Optional[str],
        fig_size: Tuple[int, int],
        dark_mode: bool
    ) -> Figure:
        """Create system overview using Matplotlib."""
        # Set the style
        style = "dark_background" if dark_mode else "seaborn-v0_8-whitegrid"
        plt.style.use(style)
        
        # Create figure and subplots
        fig, axs = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle("WDBX System Overview", fontsize=16)
        
        # Memory Usage
        axs[0, 0].plot(timestamps, memory_usage, "b-", linewidth=2)
        axs[0, 0].fill_between(timestamps, memory_usage, alpha=0.2)
        axs[0, 0].set_title("Memory Usage (%)")
        axs[0, 0].set_ylim(0, 100)
        axs[0, 0].grid(True)
        
        # CPU Usage
        axs[0, 1].plot(timestamps, cpu_usage, "g-", linewidth=2)
        axs[0, 1].fill_between(timestamps, cpu_usage, alpha=0.2)
        axs[0, 1].set_title("CPU Usage (%)")
        axs[0, 1].set_ylim(0, 100)
        axs[0, 1].grid(True)
        
        # Disk Usage
        axs[1, 0].plot(timestamps, disk_usage, "r-", linewidth=2)
        axs[1, 0].fill_between(timestamps, disk_usage, alpha=0.2)
        axs[1, 0].set_title("Disk Usage (%)")
        axs[1, 0].set_ylim(0, 100)
        axs[1, 0].grid(True)
        
        # Query Count
        axs[1, 1].plot(timestamps, query_count, "y-", linewidth=2)
        axs[1, 1].fill_between(timestamps, query_count, alpha=0.2)
        axs[1, 1].set_title("Query Count")
        axs[1, 1].grid(True)
        
        # Format x-axis to display time nicely
        for ax in axs.flat:
            ax.xaxis_date()
            fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save if output file provided
        if output_file:
            plt.savefig(output_file, dpi=100, bbox_inches="tight")
            logger.info(f"Saved visualization to {output_file}")
        
        return fig
    
    def create_component_dashboard(
        self,
        component_names: Optional[List[str]] = None,
        output_file: Optional[str] = None,
        time_range_minutes: int = 60,
        use_plotly: bool = True,
        fig_size: Tuple[int, int] = (14, 10),
        dark_mode: bool = False
    ) -> Optional[Union[Figure, go.Figure]]:
        """
        Create a component metrics dashboard.
        
        Args:
            component_names: List of component names to include (None for all)
            output_file: Path to save the visualization (None for display only)
            time_range_minutes: Time range in minutes to include in visualization
            use_plotly: Whether to use plotly instead of matplotlib
            fig_size: Figure size (width, height) in inches
            dark_mode: Whether to use dark theme
            
        Returns:
            Matplotlib Figure or Plotly Figure based on use_plotly
        """
        # Check if visualization is supported
        if use_plotly and not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Falling back to matplotlib.")
            use_plotly = False
            
        if not use_plotly and not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Falling back to plotly.")
            use_plotly = True
            
        if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
            logger.error("No visualization libraries available.")
            return None
        
        # Get component metrics
        if not hasattr(self.monitor, "metrics") or not hasattr(self.monitor.metrics, "components"):
            logger.warning("No component metrics available")
            return None
            
        components = self.monitor.metrics.get("components", {})
        if component_names:
            components = {name: data for name, data in components.items() if name in component_names}
            
        if not components:
            logger.warning("No component metrics found")
            return None
        
        # Filter history data based on time range
        cutoff_time = time.time() - (time_range_minutes * 60)
        history = self.monitor.metrics["history"]
        timestamps = history["timestamps"]
        
        # Filter timestamps by time range
        indices = [i for i, ts in enumerate(timestamps) if ts >= cutoff_time]
        if not indices:
            logger.warning(f"No data available in the last {time_range_minutes} minutes.")
            return None
            
        filtered_timestamps = [timestamps[i] for i in indices]
        
        # Convert timestamps to datetime
        datetime_labels = [datetime.fromtimestamp(ts) for ts in filtered_timestamps]
        
        if use_plotly and PLOTLY_AVAILABLE:
            return self._create_plotly_component_dashboard(
                components,
                datetime_labels,
                output_file,
                fig_size,
                dark_mode
            )
        if MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_component_dashboard(
                components,
                datetime_labels,
                output_file,
                fig_size,
                dark_mode
            )
        logger.error("No visualization libraries available.")
        return None
    
    def _create_plotly_component_dashboard(
        self,
        components: Dict[str, Dict[str, Any]],
        timestamps: List[datetime],
        output_file: Optional[str],
        fig_size: Tuple[int, int],
        dark_mode: bool
    ) -> go.Figure:
        """Create component dashboard using Plotly."""
        num_components = len(components)
        if num_components == 0:
            return None
            
        template = "plotly_dark" if dark_mode else "plotly_white"
        
        # Create figure with subplots
        fig = make_subplots(
            rows=num_components, cols=1,
            subplot_titles=list(components.keys()),
            vertical_spacing=0.1
        )
        
        # Color map for metrics
        colors = {
            "queries": "#2E86C1",
            "requests": "#2E86C1",
            "cache_hits": "#2ECC71",
            "cache_misses": "#E67E22",
            "errors": "#E74C3C",
            "avg_query_time_ms": "#9B59B6",
            "avg_response_time_ms": "#9B59B6"
        }
        
        # Add traces for each component
        for i, (component_name, metrics) in enumerate(components.items(), 1):
            for metric_name, metric_value in metrics.items():
                # Skip non-numeric metrics and some specific ones we don't want to graph
                if not isinstance(metric_value, (int, float)) or metric_name in ("status", "name"):
                    continue
                    
                color = colors.get(metric_name, "#3498DB")
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=[metric_value] * len(timestamps),  # Use the same value for all points
                        name=f"{component_name}.{metric_name}",
                        line=dict(color=color, width=2),
                        fill="tozeroy",
                        fillcolor=f"rgba({color[1:3]}, {color[3:5]}, {color[5:7]}, 0.2)"
                    ),
                    row=i, col=1
                )
        
        # Update layout
        fig.update_layout(
            title="WDBX Component Metrics",
            width=fig_size[0] * 100,  # Convert inches to pixels (approximate)
            height=fig_size[1] * 100,
            template=template,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Save if output file provided
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Saved component dashboard to {output_file}")
        
        return fig
    
    def _create_matplotlib_component_dashboard(
        self,
        components: Dict[str, Dict[str, Any]],
        timestamps: List[datetime],
        output_file: Optional[str],
        fig_size: Tuple[int, int],
        dark_mode: bool
    ) -> Figure:
        """Create component dashboard using Matplotlib."""
        num_components = len(components)
        if num_components == 0:
            return None
            
        # Set the style
        style = "dark_background" if dark_mode else "seaborn-v0_8-whitegrid"
        plt.style.use(style)
        
        # Create figure and subplots
        fig, axs = plt.subplots(num_components, 1, figsize=fig_size)
        fig.suptitle("WDBX Component Metrics", fontsize=16)
        
        # Make sure axs is iterable even with one component
        if num_components == 1:
            axs = [axs]
        
        # Color map for metrics
        colors = {
            "queries": "blue",
            "requests": "blue",
            "cache_hits": "green",
            "cache_misses": "orange",
            "errors": "red",
            "avg_query_time_ms": "purple",
            "avg_response_time_ms": "purple"
        }
        
        # Plot each component
        for i, (component_name, metrics) in enumerate(components.items()):
            ax = axs[i]
            ax.set_title(component_name)
            
            # Plot each metric
            for metric_name, metric_value in metrics.items():
                # Skip non-numeric metrics and some specific ones we don't want to graph
                if not isinstance(metric_value, (int, float)) or metric_name in ("status", "name"):
                    continue
                
                color = colors.get(metric_name, "blue")
                ax.plot(
                    timestamps,
                    [metric_value] * len(timestamps),  # Use the same value for all points
                    label=metric_name,
                    color=color
                )
                
            ax.legend()
            ax.grid(True)
            ax.xaxis_date()
        
        # Format x-axis to display time nicely
        fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save if output file provided
        if output_file:
            plt.savefig(output_file, dpi=100, bbox_inches="tight")
            logger.info(f"Saved component dashboard to {output_file}")
        
        return fig
    
    def create_events_timeline(
        self,
        output_file: Optional[str] = None,
        time_range_hours: int = 24,
        event_types: Optional[List[str]] = None,
        use_plotly: bool = True,
        fig_size: Tuple[int, int] = (14, 6),
        dark_mode: bool = False
    ) -> Optional[Union[Figure, go.Figure]]:
        """
        Create a timeline of events.
        
        Args:
            output_file: Path to save the visualization (None for display only)
            time_range_hours: Time range in hours to include in visualization
            event_types: List of event types to include (None for all)
            use_plotly: Whether to use plotly instead of matplotlib
            fig_size: Figure size (width, height) in inches
            dark_mode: Whether to use dark theme
            
        Returns:
            Matplotlib Figure or Plotly Figure based on use_plotly
        """
        # Check if visualization is supported
        if use_plotly and not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Falling back to matplotlib.")
            use_plotly = False
            
        if not use_plotly and not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Falling back to plotly.")
            use_plotly = True
            
        if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
            logger.error("No visualization libraries available.")
            return None
        
        # Get events
        events = self.monitor.metrics.get("events", [])
        if not events:
            logger.warning("No events available")
            return None
        
        # Filter events based on time range
        cutoff_time = time.time() - (time_range_hours * 3600)
        events = [e for e in events if e["timestamp"] >= cutoff_time]
        
        if not events:
            logger.warning(f"No events in the last {time_range_hours} hours")
            return None
        
        # Filter by event type if specified
        if event_types:
            events = [e for e in events if e["type"] in event_types]
            
        if not events:
            logger.warning(f"No events of types {event_types} in the time range")
            return None
        
        # Sort events by timestamp
        events.sort(key=lambda e: e["timestamp"])
        
        # Extract timestamps and convert to datetime
        timestamps = [datetime.fromtimestamp(e["timestamp"]) for e in events]
        types = [e["type"] for e in events]
        
        # Map event types to colors
        type_colors = {
            "info": "#3498DB",    # Blue
            "warning": "#F39C12",  # Orange
            "error": "#E74C3C",    # Red
            "critical": "#8E44AD"  # Purple
        }
        
        if use_plotly and PLOTLY_AVAILABLE:
            return self._create_plotly_events_timeline(
                events,
                timestamps,
                types,
                type_colors,
                output_file,
                fig_size,
                dark_mode
            )
        if MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_events_timeline(
                events,
                timestamps,
                types,
                type_colors,
                output_file,
                fig_size,
                dark_mode
            )
        logger.error("No visualization libraries available.")
        return None
    
    def _create_plotly_events_timeline(
        self,
        events: List[Dict[str, Any]],
        timestamps: List[datetime],
        types: List[str],
        type_colors: Dict[str, str],
        output_file: Optional[str],
        fig_size: Tuple[int, int],
        dark_mode: bool
    ) -> go.Figure:
        """Create events timeline using Plotly."""
        template = "plotly_dark" if dark_mode else "plotly_white"
        
        # Prepare hover text
        hover_texts = []
        for event in events:
            data_str = "<br>".join([f"{k}: {v}" for k, v in event["data"].items()])
            hover_texts.append(f"Type: {event['type']}<br>{data_str}")
        
        # Create a scatter plot for the timeline
        fig = go.Figure()
        
        # Add traces for each event type
        for event_type in set(types):
            indices = [i for i, t in enumerate(types) if t == event_type]
            fig.add_trace(
                go.Scatter(
                    x=[timestamps[i] for i in indices],
                    y=[1 for _ in indices],  # All events at same y level
                    mode="markers",
                    name=event_type,
                    marker=dict(
                        color=type_colors.get(event_type, "#3498DB"),
                        size=12,
                        symbol="circle"
                    ),
                    text=[hover_texts[i] for i in indices],
                    hoverinfo="text"
                )
            )
        
        # Update layout
        fig.update_layout(
            title="WDBX Events Timeline",
            width=fig_size[0] * 100,
            height=fig_size[1] * 100,
            template=template,
            showlegend=True,
            xaxis_title="Time",
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False
            )
        )
        
        # Save if output file provided
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Saved events timeline to {output_file}")
        
        return fig
    
    def _create_matplotlib_events_timeline(
        self,
        events: List[Dict[str, Any]],
        timestamps: List[datetime],
        types: List[str],
        type_colors: Dict[str, str],
        output_file: Optional[str],
        fig_size: Tuple[int, int],
        dark_mode: bool
    ) -> Figure:
        """Create events timeline using Matplotlib."""
        # Set the style
        style = "dark_background" if dark_mode else "seaborn-v0_8-whitegrid"
        plt.style.use(style)
        
        # Create figure
        fig, ax = plt.subplots(figsize=fig_size)
        fig.suptitle("WDBX Events Timeline", fontsize=16)
        
        # Plot each event type with different markers
        for event_type in set(types):
            indices = [i for i, t in enumerate(types) if t == event_type]
            ax.scatter(
                [timestamps[i] for i in indices],
                [1 for _ in indices],  # All events at same y level
                label=event_type,
                color=type_colors.get(event_type, "blue"),
                marker="o",
                s=100
            )
        
        # Format the plot
        ax.set_yticks([])  # Hide y-axis ticks
        ax.set_xlabel("Time")
        ax.grid(True, axis="x")
        ax.legend()
        
        # Format x-axis to display time nicely
        fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save if output file provided
        if output_file:
            plt.savefig(output_file, dpi=100, bbox_inches="tight")
            logger.info(f"Saved events timeline to {output_file}")
        
        return fig


# Helper function to get a visualizer instance
def get_visualizer() -> DiagnosticsVisualizer:
    """Get a DiagnosticsVisualizer instance using the global monitor."""
    return DiagnosticsVisualizer(get_monitor()) 