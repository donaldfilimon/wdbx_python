#!/usr/bin/env python3
"""
Terminal UI for the WDBX database.

This module provides a rich terminal interface for monitoring, visualizing,
and interacting with the WDBX database system.
"""
from __future__ import annotations

import os
import platform
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import psutil

# Try to import rich and textual for the terminal UI
try:
    from rich.align import Align
    from rich.console import Console, RenderableType
    from rich.layout import Layout
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.prompt import Confirm, Prompt
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    # Try to import Chart, but provide fallback
    try:
        from rich.chart import Chart  # type: ignore

        CHART_AVAILABLE = True
    except ImportError:
        CHART_AVAILABLE = False

    # Try to import textual components
    try:
        from textual import events
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Container, Grid, Horizontal, Vertical
        from textual.message import Message
        from textual.reactive import reactive
        from textual.screen import Screen
        from textual.widgets import (
            Button,
            DataTable,
            Footer,
            Header,
            Input,
            Label,
            Placeholder,
            ProgressBar,
            RadioButton,
            RadioSet,
            Select,
            Static,
            Switch,
            Tab,
            Tabs,
            TextLog,
        )
        from textual.widgets.text_area import TextArea

        TEXTUAL_AVAILABLE = True
    except ImportError as import_error:
        TEXTUAL_AVAILABLE = False
        print(
            f"Warning: Advanced terminal UI unavailable due to missing dependency: {import_error}"
        )
        print("Install with: pip install textual psutil numpy matplotlib")

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    CHART_AVAILABLE = False
    TEXTUAL_AVAILABLE = False
    print("Rich and Textual libraries not available. Install with: pip install rich textual")

# Fix the import path for logger
from ...core.constants import logger

# Change the import from absolute to relative import
from ...core.wdbx import WDBX

# Set up console for fallback
console = None
if RICH_AVAILABLE:
    console = Console()


class WDBXMonitor:
    """
    Base monitoring class for WDBX that works without the terminal UI.
    """

    def __init__(self, wdbx_instance: WDBX):
        self.wdbx = wdbx_instance
        self.running = False
        self.stats_history: List[Dict[str, Any]] = []
        self.max_history_size = 2000  # Increased history size for better trend analysis
        self.update_interval = 0.25  # Decreased interval for more responsive updates
        self.monitor_thread: Optional[threading.Thread] = None
        self.last_stats: Optional[Dict[str, Any]] = None
        self.system_info = self._get_system_info()
        self.alert_thresholds = {
            "memory_usage": 85.0,  # Percent
            "cpu_usage": 90.0,  # Percent
            "disk_usage": 90.0,  # Percent
            "vector_rate_drop": 50.0,  # Percent drop from peak
        }
        self.peak_values = {
            "vectors_delta": 0.0,
            "blocks_delta": 0.0,
        }
        self.alerts: List[Dict[str, str]] = []
        self.max_alerts = 100
        self.performance_score = 100  # Start with perfect score

    def start_monitoring(self) -> None:
        """Start the monitoring thread."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Monitoring stopped")

    def _get_system_info(self) -> Dict[str, Any]:
        """Gather detailed system information."""
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count() or 0,
            "hostname": platform.node(),
            "memory_total": 0,
            "disk_total": 0,
            "disk_free": 0,
        }

        try:
            # Get memory info
            memory = psutil.virtual_memory()
            info["memory_total"] = memory.total / (1024 * 1024 * 1024)  # GB

            # Get disk info
            disk = psutil.disk_usage("/")
            info["disk_total"] = disk.total / (1024 * 1024 * 1024)  # GB
            info["disk_free"] = disk.free / (1024 * 1024 * 1024)  # GB
        except Exception as e:
            logger.warning("Failed to get system info: %s", str(e))
            # Fallback if psutil fails
            pass

        return info

    def _monitor_loop(self) -> None:
        """Main monitoring loop with enhanced error handling and stats calculation."""
        while self.running:
            try:
                stats = self.wdbx.get_system_stats()
                current_time = time.time()

                # Add timestamp and calculate deltas
                stats["timestamp"] = current_time

                # Add system resource usage
                try:
                    stats["cpu_usage"] = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    stats["memory_usage_percent"] = memory.percent
                    stats["memory_usage_mb"] = memory.used / (1024 * 1024)
                    disk = psutil.disk_usage("/")
                    stats["disk_usage_percent"] = disk.percent
                except Exception as e:
                    logger.warning("Failed to get system metrics: %s", str(e))
                    # Fallback values if psutil fails
                    stats["cpu_usage"] = 0.0
                    stats["memory_usage_percent"] = 0.0
                    stats["memory_usage_mb"] = 0.0
                    stats["disk_usage_percent"] = 0.0

                if self.last_stats:
                    time_delta = current_time - self.last_stats["timestamp"]
                    stats["blocks_delta"] = (
                        stats["blocks_created"] - self.last_stats["blocks_created"]
                    ) / time_delta
                    stats["vectors_delta"] = (
                        stats["vectors_stored"] - self.last_stats["vectors_stored"]
                    ) / time_delta
                    stats["transactions_delta"] = (
                        stats["transactions_processed"] - self.last_stats["transactions_processed"]
                    ) / time_delta

                    # Update peak values
                    for key in self.peak_values:
                        if key in stats and stats[key] > self.peak_values[key]:
                            self.peak_values[key] = stats[key]

                    # Calculate performance score (0-100)
                    self._calculate_performance_score(stats)

                    # Check for alerts
                    self._check_alerts(stats)

                self.last_stats = stats.copy()
                self.stats_history.append(stats)

                # Trim history if needed
                if len(self.stats_history) > self.max_history_size:
                    self.stats_history = self.stats_history[-self.max_history_size :]

                # Print stats if no terminal UI
                if not RICH_AVAILABLE:
                    self._print_stats(stats)

                time.sleep(self.update_interval)
            except Exception as e:
                logger.error("Error in monitor loop: %s", str(e))
                self._add_alert("ERROR", f"Monitoring error: {str(e)}")
                time.sleep(self.update_interval * 2)  # Back off on error

    def _calculate_performance_score(self, stats: Dict[str, Any]) -> None:
        """Calculate a performance score based on various metrics."""
        score = 100.0

        # Penalize for high resource usage
        if stats.get("memory_usage_percent", 0) > 80:
            score -= (stats["memory_usage_percent"] - 80) * 0.5
        if stats.get("cpu_usage", 0) > 80:
            score -= (stats["cpu_usage"] - 80) * 0.5
        if stats.get("disk_usage_percent", 0) > 90:
            score -= (stats["disk_usage_percent"] - 90) * 1.0

        # Penalize for drops from peak performance
        for key in ["vectors_delta", "blocks_delta"]:
            if key in stats and self.peak_values[key] > 0:
                current = stats[key]
                peak = self.peak_values[key]
                if current < peak * 0.5:  # More than 50% drop
                    score -= 10.0 * (1.0 - (current / peak))

        # Ensure score stays in range 0-100
        self.performance_score = max(0.0, min(100.0, score))

    def _check_alerts(self, stats: Dict[str, Any]) -> None:
        """Check for conditions that should trigger alerts."""
        # Check memory usage
        if stats.get("memory_usage_percent", 0) > self.alert_thresholds["memory_usage"]:
            self._add_alert("WARNING", f"High memory usage: {stats['memory_usage_percent']:.1f}%")

        # Check CPU usage
        if stats.get("cpu_usage", 0) > self.alert_thresholds["cpu_usage"]:
            self._add_alert("WARNING", f"High CPU usage: {stats['cpu_usage']:.1f}%")

        # Check disk usage
        if stats.get("disk_usage_percent", 0) > self.alert_thresholds["disk_usage"]:
            self._add_alert("WARNING", f"High disk usage: {stats['disk_usage_percent']:.1f}%")

        # Check for performance drops
        for key in ["vectors_delta", "blocks_delta"]:
            if key in stats and self.peak_values[key] > 0:
                current = stats[key]
                peak = self.peak_values[key]
                drop_percent = (1.0 - (current / peak)) * 100.0
                if drop_percent > self.alert_thresholds["vector_rate_drop"]:
                    self._add_alert(
                        "WARNING", f"Performance drop in {key}: {drop_percent:.1f}% below peak"
                    )

    def _add_alert(self, level: str, message: str) -> None:
        """Add an alert to the alerts list."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.alerts.append({"timestamp": timestamp, "level": level, "message": message})

        # Trim alerts if needed
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts :]

    def _print_stats(self, stats: Dict[str, Any]) -> None:
        """Enhanced stats printing with color and formatting."""
        print("\033[2J\033[H")  # Clear screen
        print("\033[1;34m=== WDBX System Monitor ===\033[0m")
        print(f"\033[1mUptime:\033[0m {stats['uptime']:.2f} seconds")
        print(
            f"\033[1mBlocks:\033[0m {stats['blocks_created']} ({stats.get('blocks_delta', 0):.2f}/s)"
        )
        print(
            f"\033[1mVectors:\033[0m {stats['vectors_stored']} ({stats.get('vectors_delta', 0):.2f}/s)"
        )
        print(f"\033[1mTransactions:\033[0m {stats['transactions_processed']}")
        print(f"\033[1mTraces:\033[0m {stats['traces_created']}")
        print(f"\033[1mVector dimension:\033[0m {stats['vector_dimension']}")
        print(f"\033[1mShard count:\033[0m {stats['shard_count']}")
        print(f"\033[1mPerformance score:\033[0m {self.performance_score:.1f}/100")
        print("\033[1;34m============================\033[0m")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        if not self.stats_history:
            return {}

        latest = self.stats_history[-1]

        # Calculate averages over the last minute
        one_minute_ago = latest["timestamp"] - 60
        minute_stats = [s for s in self.stats_history if s["timestamp"] >= one_minute_ago]

        if not minute_stats:
            return {}

        avg_vectors_delta = sum(s.get("vectors_delta", 0) for s in minute_stats) / len(minute_stats)
        avg_blocks_delta = sum(s.get("blocks_delta", 0) for s in minute_stats) / len(minute_stats)

        return {
            "current_vectors_rate": latest.get("vectors_delta", 0),
            "current_blocks_rate": latest.get("blocks_delta", 0),
            "avg_vectors_rate_1m": avg_vectors_delta,
            "avg_blocks_rate_1m": avg_blocks_delta,
            "peak_vectors_rate": self.peak_values["vectors_delta"],
            "peak_blocks_rate": self.peak_values["blocks_delta"],
            "performance_score": self.performance_score,
            "memory_usage": latest.get("memory_usage_percent", 0),
            "cpu_usage": latest.get("cpu_usage", 0),
            "disk_usage": latest.get("disk_usage_percent", 0),
        }


# Only define the WDBXTerminalUI class if Textual is available
if TEXTUAL_AVAILABLE:

    class WDBXTerminalUI(App):
        """
        Rich terminal UI for WDBX using Textual.
        """

        TITLE = "WDBX Terminal UI"
        SUB_TITLE = "Database Monitoring and Management"
        CSS_PATH = None  # We'll use inline styling for simplicity

        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("r", "refresh", "Refresh"),
            Binding("c", "clear_log", "Clear Log"),
            Binding("t", "toggle_theme", "Toggle Theme"),
            Binding("1", "show_dashboard", "Dashboard"),
            Binding("2", "show_performance", "Performance"),
            Binding("3", "show_logs", "Logs"),
            Binding("4", "show_settings", "Settings"),
            Binding("5", "show_query", "Query"),
            Binding("h", "help", "Help"),
            Binding("f", "fullscreen", "Toggle Fullscreen"),
            Binding("s", "save_state", "Save State"),
            Binding("l", "load_state", "Load State"),
        ]

        def __init__(self, wdbx_instance: WDBX):
            super().__init__()
            self.wdbx = wdbx_instance
            self.monitor = WDBXMonitor(wdbx_instance)
            self.update_interval = 0.25  # seconds
            self.chart_data: List[Dict[str, float]] = []
            self.max_chart_points = 100  # Increased for better visualization
            self.log_counter = 0
            self.current_screen = "dashboard"
            self.theme_mode = "dark"  # or "light"
            self.query_history: List[str] = []
            self.max_query_history = 50
            self.custom_queries = [
                "SELECT * FROM vectors LIMIT 10;",
                "SELECT COUNT(*) FROM blocks;",
                "SELECT AVG(similarity) FROM vector_search WHERE query_id = 'latest';",
            ]
            self.fullscreen = False
            self.saved_states: Dict[str, Any] = {}
            self.help_text = """
            [bold]WDBX Terminal UI Help[/bold]
            
            [cyan]Navigation:[/cyan]
            - [bold]1-5[/bold]: Switch between views
            - [bold]q[/bold]: Quit application
            - [bold]r[/bold]: Refresh current view
            - [bold]c[/bold]: Clear logs
            - [bold]t[/bold]: Toggle theme
            - [bold]f[/bold]: Toggle fullscreen
            - [bold]h[/bold]: Show this help
            
            [cyan]Views:[/cyan]
            - [bold]Dashboard[/bold]: Overview of system status
            - [bold]Performance[/bold]: Performance metrics and charts
            - [bold]Logs[/bold]: System logs and alerts
            - [bold]Settings[/bold]: UI and monitoring settings
            - [bold]Query[/bold]: Run custom queries
            
            [cyan]Data Management:[/cyan]
            - [bold]s[/bold]: Save current state
            - [bold]l[/bold]: Load saved state
            """

        def compose(self) -> ComposeResult:
            """Compose the UI layout."""
            # Main layout with tabs
            yield Header()

            # Tabs for different sections
            yield Tabs(
                Tab("Dashboard", id="tab_dashboard"),
                Tab("Performance", id="tab_performance"),
                Tab("Logs", id="tab_logs"),
                Tab("Settings", id="tab_settings"),
                Tab("Query", id="tab_query"),
                id="main_tabs",
            )

            # Dashboard view
            with Container(id="dashboard_view"):
                with Grid(id="dashboard_grid"):
                    # System stats panel
                    yield Panel(
                        Static(id="stats_panel"), title="System Statistics", border_style="blue"
                    )

                    # Performance score panel
                    yield Panel(
                        Static(id="score_panel"), title="Performance Score", border_style="green"
                    )

                    # Alerts panel
                    yield Panel(
                        TextLog(id="alerts_log", highlight=True, markup=True),
                        title="Recent Alerts",
                        border_style="red",
                    )

                    # Quick actions panel
                    with Panel(title="Quick Actions", border_style="yellow") as actions_panel:
                        yield Button("Refresh Stats", variant="primary", id="refresh_btn")
                        yield Button("Clear Log", variant="warning", id="clear_log_btn")
                        yield Button("Run Query", variant="success", id="run_query_btn")
                        yield Button("System Check", variant="default", id="system_check_btn")

            # Performance view
            with Container(id="performance_view", classes="hidden"):
                # Performance charts
                yield Panel(
                    Static(id="perf_chart"), title="Performance Metrics", border_style="green"
                )

                # Detailed metrics table
                yield Panel(
                    Static(id="metrics_table"), title="Detailed Metrics", border_style="blue"
                )

                # Resource usage
                yield Panel(
                    Static(id="resource_usage"), title="System Resources", border_style="yellow"
                )

                # Historical trends
                yield Panel(
                    Static(id="historical_trends"),
                    title="Historical Trends",
                    border_style="magenta",
                )

            # Logs view
            with Container(id="logs_view", classes="hidden"):
                # System log
                yield Panel(
                    TextLog(id="system_log", highlight=True, markup=True),
                    title="System Log",
                    border_style="cyan",
                )

                # Log filter controls
                with Panel(title="Log Controls", border_style="blue") as log_controls:
                    yield Input(placeholder="Filter logs...", id="log_filter")
                    with Horizontal():
                        yield Button("Clear", variant="warning", id="clear_system_log_btn")
                        yield Button("Export", variant="primary", id="export_log_btn")
                        yield Switch(value=True, id="auto_scroll_switch")
                        yield Label("Auto-scroll")

            # Settings view
            with Container(id="settings_view", classes="hidden"):
                # UI settings
                with Panel(title="UI Settings", border_style="green") as ui_settings:
                    with Horizontal():
                        yield Label("Theme:")
                        yield Button("Light", id="light_theme_btn")
                        yield Button("Dark", id="dark_theme_btn")

                    with Horizontal():
                        yield Label("Update Interval:")
                        yield Input(value="0.25", id="update_interval_input")
                        yield Label("seconds")

                # Alert thresholds
                with Panel(title="Alert Thresholds", border_style="yellow") as alert_settings:
                    with Horizontal():
                        yield Label("Memory Usage:")
                        yield Input(value="85", id="memory_threshold_input")
                        yield Label("%")

                    with Horizontal():
                        yield Label("CPU Usage:")
                        yield Input(value="90", id="cpu_threshold_input")
                        yield Label("%")

                    with Horizontal():
                        yield Label("Performance Drop:")
                        yield Input(value="50", id="perf_drop_threshold_input")
                        yield Label("%")

                # Apply button
                yield Button("Apply Settings", variant="primary", id="apply_settings_btn")

            # Query view
            with Container(id="query_view", classes="hidden"):
                # Query input
                yield Panel(TextArea(id="query_input"), title="Query Input", border_style="blue")

                # Query results
                yield Panel(Static(id="query_results"), title="Query Results", border_style="green")

                # Query controls
                with Horizontal():
                    yield Button("Run Query", variant="primary", id="execute_query_btn")
                    yield Button("Clear", variant="warning", id="clear_query_btn")
                    yield Button("Save Query", variant="success", id="save_query_btn")

                # Saved queries
                yield Panel(
                    Static(id="saved_queries"), title="Saved Queries", border_style="yellow"
                )

            yield Footer()

        async def on_mount(self) -> None:
            """Set up the UI when the app is mounted."""
            # Start monitors with improved update frequency
            self.monitor.start_monitoring()
            self.set_interval(self.update_interval, self.update_stats)
            self.set_interval(0.5, self.update_log)
            self.set_interval(2.0, self.update_visualization)
            self.set_interval(5.0, self.update_alerts)

            # Initialize UI components
            await self.update_stats()
            await self.update_visualization()
            await self.update_saved_queries()

            # Show dashboard by default
            await self.show_dashboard()

        async def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
            """Handle tab activation."""
            tab_id = event.tab.id

            if tab_id == "tab_dashboard":
                await self.show_dashboard()
            elif tab_id == "tab_performance":
                await self.show_performance()
            elif tab_id == "tab_logs":
                await self.show_logs()
            elif tab_id == "tab_settings":
                await self.show_settings()
            elif tab_id == "tab_query":
                await self.show_query()

        async def action_show_dashboard(self) -> None:
            """Show the dashboard view."""
            await self.show_dashboard()
            await self.query_one("#main_tabs").activate_tab("tab_dashboard")

        async def action_show_performance(self) -> None:
            """Show the performance view."""
            await self.show_performance()
            await self.query_one("#main_tabs").activate_tab("tab_performance")

        async def action_show_logs(self) -> None:
            """Show the logs view."""
            await self.show_logs()
            await self.query_one("#main_tabs").activate_tab("tab_logs")

        async def action_show_settings(self) -> None:
            """Show the settings view."""
            await self.show_settings()
            await self.query_one("#main_tabs").activate_tab("tab_settings")

        async def action_show_query(self) -> None:
            """Show the query view."""
            await self.show_query()
            await self.query_one("#main_tabs").activate_tab("tab_query")

        async def show_dashboard(self) -> None:
            """Show the dashboard view."""
            self.current_screen = "dashboard"

            # Hide all views
            for view_id in [
                "dashboard_view",
                "performance_view",
                "logs_view",
                "settings_view",
                "query_view",
            ]:
                view = self.query_one(f"#{view_id}")
                if view_id == "dashboard_view":
                    view.remove_class("hidden")
                else:
                    view.add_class("hidden")

            # Update dashboard components
            await self.update_stats()
            await self.update_alerts()

        async def show_performance(self) -> None:
            """Show the performance view."""
            self.current_screen = "performance"

            # Hide all views
            for view_id in [
                "dashboard_view",
                "performance_view",
                "logs_view",
                "settings_view",
                "query_view",
            ]:
                view = self.query_one(f"#{view_id}")
                if view_id == "performance_view":
                    view.remove_class("hidden")
                else:
                    view.add_class("hidden")

            # Update performance components
            await self.update_visualization()
            await self.update_resource_usage()
            await self.update_historical_trends()

        async def show_logs(self) -> None:
            """Show the logs view."""
            self.current_screen = "logs"

            # Hide all views
            for view_id in [
                "dashboard_view",
                "performance_view",
                "logs_view",
                "settings_view",
                "query_view",
            ]:
                view = self.query_one(f"#{view_id}")
                if view_id == "logs_view":
                    view.remove_class("hidden")
                else:
                    view.add_class("hidden")

        async def show_settings(self) -> None:
            """Show the settings view."""
            self.current_screen = "settings"

            # Hide all views
            for view_id in [
                "dashboard_view",
                "performance_view",
                "logs_view",
                "settings_view",
                "query_view",
            ]:
                view = self.query_one(f"#{view_id}")
                if view_id == "settings_view":
                    view.remove_class("hidden")
                else:
                    view.add_class("hidden")

            # Update settings with current values
            self.query_one("#update_interval_input").value = str(self.update_interval)
            self.query_one("#memory_threshold_input").value = str(
                self.monitor.alert_thresholds["memory_usage"]
            )
            self.query_one("#cpu_threshold_input").value = str(
                self.monitor.alert_thresholds["cpu_usage"]
            )
            self.query_one("#perf_drop_threshold_input").value = str(
                self.monitor.alert_thresholds["vector_rate_drop"]
            )

        async def show_query(self) -> None:
            """Show the query view."""
            self.current_screen = "query"

            # Hide all views
            for view_id in [
                "dashboard_view",
                "performance_view",
                "logs_view",
                "settings_view",
                "query_view",
            ]:
                view = self.query_one(f"#{view_id}")
                if view_id == "query_view":
                    view.remove_class("hidden")
                else:
                    view.add_class("hidden")

            # Update saved queries
            await self.update_saved_queries()

        async def update_stats(self) -> None:
            """Update the stats panel with enhanced metrics."""
            if not self.monitor.stats_history:
                return

            stats = self.monitor.stats_history[-1]

            # Create an enhanced stats table
            table = Table(show_header=True, header_style="bold magenta", expand=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Current", justify="right", style="green")
            table.add_column("Rate", justify="right", style="yellow")
            table.add_column("Peak", justify="right", style="red")

            # Calculate peak rates
            peak_blocks = self.monitor.peak_values["blocks_delta"]
            peak_vectors = self.monitor.peak_values["vectors_delta"]

            # Add rows with more detailed information
            table.add_row("Uptime", f"{stats['uptime']:.2f}s", "", "")
            table.add_row(
                "Blocks",
                str(stats["blocks_created"]),
                f"{stats.get('blocks_delta', 0):.2f}/s",
                f"{peak_blocks:.2f}/s",
            )
            table.add_row(
                "Vectors",
                str(stats["vectors_stored"]),
                f"{stats.get('vectors_delta', 0):.2f}/s",
                f"{peak_vectors:.2f}/s",
            )
            table.add_row(
                "Transactions",
                str(stats["transactions_processed"]),
                f"{stats.get('transactions_delta', 0):.2f}/s",
                "",
            )
            table.add_row("Traces", str(stats["traces_created"]), "", "")

            # System configuration
            table.add_section()
            table.add_row("Vector Dimension", str(stats["vector_dimension"]), "", "")
            table.add_row("Shard Count", str(stats["shard_count"]), "", "")

            # System resources
            table.add_section()
            table.add_row(
                "Memory Usage",
                f"{stats.get('memory_usage_mb', 0):.2f}MB",
                f"{stats.get('memory_usage_percent', 0):.1f}%",
                "",
            )
            table.add_row("CPU Usage", "", f"{stats.get('cpu_usage', 0):.1f}%", "")
            table.add_row("Disk Usage", "", f"{stats.get('disk_usage_percent', 0):.1f}%", "")

            # Update the stats panel
            stats_panel = self.query_one("#stats_panel")
            stats_panel.update(table)

            # Update performance score panel
            score_panel = self.query_one("#score_panel")
            score = self.monitor.performance_score

            # Create a visual representation of the score
            score_color = "green" if score >= 80 else "yellow" if score >= 50 else "red"
            score_bar = "█" * int(score / 2)
            score_text = f"[bold {score_color}]{score:.1f}/100[/bold {score_color}]"

            score_content = f"""
            Performance Score: {score_text}
            
            [{score_color}]{score_bar}[/{score_color}]
            
            Status: [bold {score_color}]{"Excellent" if score >= 90 else "Good" if score >= 70 else "Fair" if score >= 50 else "Poor"}[/bold {score_color}]
            """

            score_panel.update(score_content)

        async def update_log(self) -> None:
            """Update the log panel with enhanced logging."""
            self.log_counter += 1
            system_log = self.query_one("#system_log")

            # Add more detailed log messages
            if self.log_counter % 5 == 0:
                stats = self.monitor.stats_history[-1] if self.monitor.stats_history else None
                if stats:
                    system_log.write(
                        f"[green]INFO[/green] [{datetime.now().strftime('%H:%M:%S')}]: System running normally - "
                        f"Vectors: {stats['vectors_stored']} "
                        f"({stats.get('vectors_delta', 0):.2f}/s)"
                    )

            if self.log_counter % 10 == 0:
                stats = self.monitor.stats_history[-1] if self.monitor.stats_history else None
                if stats:
                    system_log.write(
                        f"[blue]DEBUG[/blue] [{datetime.now().strftime('%H:%M:%S')}]: Performance metrics - "
                        f"Memory usage: {stats.get('memory_usage_mb', 0):.2f}MB"
                    )

        async def update_visualization(self) -> None:
            """Update the visualization panel with enhanced performance data."""
            if not self.monitor.stats_history or len(self.monitor.stats_history) < 2:
                return

            # Get the last stats entries for visualization
            history = self.monitor.stats_history[-self.max_chart_points :]

            if CHART_AVAILABLE:
                # Create a new chart with the updated data
                chart = Chart()

                # Extract data for vectors/s and blocks/s
                vectors_data = [stats.get("vectors_delta", 0) for stats in history]
                blocks_data = [stats.get("blocks_delta", 0) for stats in history]
                transactions_data = [stats.get("transactions_delta", 0) for stats in history]

                # Add the data series with enhanced styling
                chart.add_series(
                    "Vectors/s",
                    vectors_data,
                    color="green",
                    style="bold",
                    marker="█",
                )
                chart.add_series(
                    "Blocks/s",
                    blocks_data,
                    color="blue",
                    style="bold",
                    marker="█",
                )
                chart.add_series(
                    "Transactions/s",
                    transactions_data,
                    color="yellow",
                    style="bold",
                    marker="█",
                )

                # Add trend lines
                if len(vectors_data) > 1:
                    chart.add_trend_line(vectors_data, color="green", style="dashed")
                if len(blocks_data) > 1:
                    chart.add_trend_line(blocks_data, color="blue", style="dashed")
                if len(transactions_data) > 1:
                    chart.add_trend_line(transactions_data, color="yellow", style="dashed")

                # Update the visualization panel
                self.viz_panel.update(
                    Panel(
                        chart,
                        title="Performance Metrics",
                        border_style="yellow",
                        subtitle="Real-time monitoring",
                    )
                )
            else:
                # Create a text-based visualization as fallback
                latest = history[-1]
                text = (
                    f"[bold]Latest Performance Metrics:[/bold]\n\n"
                    f"Vectors/s: [green]{latest.get('vectors_delta', 0):.2f}[/green]\n"
                    f"Blocks/s: [blue]{latest.get('blocks_delta', 0):.2f}[/blue]\n"
                    f"Transactions/s: [yellow]{latest.get('transactions_delta', 0):.2f}[/yellow]\n\n"
                    f"[yellow]Chart visualization not available - install latest rich package[/yellow]"
                )
                self.viz_panel.update(
                    Panel(
                        Static(text),
                        title="Performance Metrics",
                        border_style="yellow",
                        subtitle="Real-time monitoring",
                    )
                )

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            """Handle button press events."""
            button_id = event.button.id

            if button_id == "refresh":
                await self.action_refresh()
            elif button_id == "clear_log":
                await self.action_clear_log()
            elif button_id == "exit":
                await self.action_quit()

        async def action_refresh(self) -> None:
            """Refresh all UI components."""
            await self.update_stats()
            await self.update_log()
            await self.update_visualization()
            self.log_panel.write("[magenta]INFO[/magenta]: Manual refresh triggered")

        async def action_clear_log(self) -> None:
            """Clear the log panel."""
            self.log_panel.clear()
            self.log_panel.write("[magenta]INFO[/magenta]: Log cleared")

        async def action_quit(self) -> None:
            """Gracefully shut down the application."""
            self.monitor.stop_monitoring()
            await self.shutdown()

        def display_status(self, message: str) -> None:
            """Display a status message in the log panel."""
            if hasattr(self, "log_panel"):
                self.log_panel.write(f"[cyan]STATUS[/cyan]: {message}")

        async def action_help(self) -> None:
            """Show help information."""
            help_panel = Panel(
                Markdown(self.help_text),
                title="Help",
                border_style="blue",
                expand=True,
            )
            await self.push_screen(help_panel)

        async def action_fullscreen(self) -> None:
            """Toggle fullscreen mode."""
            self.fullscreen = not self.fullscreen
            if self.fullscreen:
                self.screen.styles.width = "100%"
                self.screen.styles.height = "100%"
            else:
                self.screen.styles.width = "auto"
                self.screen.styles.height = "auto"
            await self.refresh()

        async def action_save_state(self) -> None:
            """Save current UI state."""
            state = {
                "current_screen": self.current_screen,
                "theme_mode": self.theme_mode,
                "update_interval": self.update_interval,
                "query_history": self.query_history,
                "custom_queries": self.custom_queries,
                "monitor_settings": {
                    "alert_thresholds": self.monitor.alert_thresholds,
                    "max_history_size": self.monitor.max_history_size,
                },
            }
            self.saved_states["last_saved"] = state
            self.display_status("State saved successfully")

        async def action_load_state(self) -> None:
            """Load saved UI state."""
            if "last_saved" not in self.saved_states:
                self.display_status("No saved state found")
                return

            state = self.saved_states["last_saved"]
            self.current_screen = state["current_screen"]
            self.theme_mode = state["theme_mode"]
            self.update_interval = state["update_interval"]
            self.query_history = state["query_history"]
            self.custom_queries = state["custom_queries"]
            
            # Update monitor settings
            self.monitor.alert_thresholds = state["monitor_settings"]["alert_thresholds"]
            self.monitor.max_history_size = state["monitor_settings"]["max_history_size"]
            
            # Refresh UI
            await self.refresh()
            self.display_status("State loaded successfully")

else:
    # Fallback class when Textual is not available
    class WDBXTerminalUI:
        """Fallback Terminal UI for WDBX when Textual is not available."""

        def __init__(self, wdbx_instance: WDBX):
            self.wdbx = wdbx_instance
            if console:
                console.print("[yellow]Textual library not available. Using fallback UI.[/yellow]")
            else:
                print("Textual library not available. Using fallback UI.")

        def run(self) -> None:
            """Run a fallback UI."""
            monitor = WDBXMonitor(self.wdbx)
            monitor.start_monitoring()
            try:
                print("Starting basic monitoring mode. Press Ctrl+C to exit.")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                monitor.stop_monitoring()
                print("\nMonitoring stopped")


def run_terminal_ui(wdbx_instance: WDBX) -> None:
    """
    Run the enhanced terminal UI with better error handling.

    Args:
        wdbx_instance: WDBX instance to monitor
    """
    # Try to import the enhanced terminal UI
    try:
        from .terminal_ui_addons import run_enhanced_terminal_ui

        run_enhanced_terminal_ui(wdbx_instance)
        return
    except ImportError as e:
        if console:
            console.print(f"[yellow]Enhanced terminal UI not available: {str(e)}[/yellow]")
            console.print("[yellow]Using fallback UI.[/yellow]")
        else:
            print(f"Enhanced terminal UI not available: {str(e)}")
            print("Using fallback UI.")

    if RICH_AVAILABLE:
        try:
            app = WDBXTerminalUI(wdbx_instance)
            app.run()
        except Exception as e:
            if console:
                console.print(f"[red]Error running terminal UI: {str(e)}[/red]")
                console.print_exception(show_locals=True)
            else:
                print(f"Error running terminal UI: {str(e)}")
            # Fallback to simple monitor
            run_simple_dashboard(wdbx_instance)
    else:
        print("Rich library not available. Install with: pip install rich textual")
        # Fallback to simple monitor
        monitor = WDBXMonitor(wdbx_instance)
        monitor.start_monitoring()
        try:
            print("Starting basic monitoring mode. Press Ctrl+C to exit.")
            while True:
                stats = wdbx_instance.get_system_stats()
                print(
                    f"\rBlocks: {stats['blocks_created']} | Vectors: {stats['vectors_stored']}",
                    end="",
                )
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("\nMonitoring stopped")
        except Exception as e:
            print(f"\nError in monitoring: {str(e)}")
            monitor.stop_monitoring()


def run_simple_dashboard(wdbx_instance: WDBX) -> None:
    """
    Run an enhanced simple dashboard using just Rich.

    Args:
        wdbx_instance: WDBX instance to monitor
    """
    # Try to import the enhanced terminal UI
    try:
        from .terminal_ui_addons import create_quick_dashboard

        create_quick_dashboard(wdbx_instance, compact=True)
        return
    except ImportError as e:
        if console:
            console.print(f"[yellow]Enhanced dashboard not available: {str(e)}[/yellow]")
            console.print("[yellow]Using fallback dashboard.[/yellow]")
        else:
            print(f"Enhanced dashboard not available: {str(e)}")
            print("Using fallback dashboard.")

    if not RICH_AVAILABLE or not console:
        print("Rich library not available. Install with: pip install rich")
        return

    try:
        monitor = WDBXMonitor(wdbx_instance)
        monitor.start_monitoring()

        with Live(auto_refresh=True, refresh_per_second=4) as live:
            while True:
                try:
                    # Get enhanced stats
                    if monitor.stats_history:
                        stats = monitor.stats_history[-1]
                        performance_score = monitor.performance_score
                    else:
                        stats = wdbx_instance.get_system_stats()
                        performance_score = 0

                    # Create layout
                    layout = Layout()
                    layout.split(
                        Layout(name="header", size=3),
                        Layout(name="main"),
                        Layout(name="footer", size=3),
                    )

                    layout["main"].split_row(Layout(name="stats"), Layout(name="performance"))

                    # Header
                    header_text = Text("WDBX System Monitor", style="bold blue")
                    header_text.append(f"\nRunning for {stats['uptime']:.2f} seconds", style="blue")
                    layout["header"].update(Align.center(header_text))

                    # Stats table
                    table = Table(title="System Statistics", expand=True)
                    table.add_column("Metric", style="cyan", justify="right")
                    table.add_column("Current Value", style="green")
                    table.add_column("Rate", style="yellow")

                    blocks_rate = stats.get("blocks_delta", 0)
                    vectors_rate = stats.get("vectors_delta", 0)

                    table.add_row("Blocks", str(stats["blocks_created"]), f"{blocks_rate:.2f}/s")
                    table.add_row("Vectors", str(stats["vectors_stored"]), f"{vectors_rate:.2f}/s")
                    table.add_row("Transactions", str(stats["transactions_processed"]), "-")
                    table.add_row("Traces", str(stats["traces_created"]), "-")
                    table.add_row("Memory", f"{stats.get('memory_usage_mb', 0):.2f}MB", "-")

                    # System info
                    system_info = Panel(
                        f"Vector dimension: {stats['vector_dimension']}\n"
                        f"Shard count: {stats['shard_count']}\n"
                        f"Performance score: {performance_score:.1f}/100",
                        title="System Information",
                        border_style="blue",
                    )

                    layout["stats"].update(table)
                    layout["performance"].update(system_info)

                    # Footer
                    footer = Panel(
                        "[bold]Controls:[/bold] [white]Press Ctrl+C to exit[/white]",
                        border_style="green",
                    )
                    layout["footer"].update(footer)

                    # Update live display
                    live.update(layout)

                    # Sleep
                    time.sleep(0.25)
                except Exception as e:
                    if console:
                        console.print(f"[red]Error updating dashboard: {str(e)}[/red]")
                    else:
                        print(f"Error updating dashboard: {str(e)}")
                    time.sleep(1)  # Back off on error

    except KeyboardInterrupt:
        if console:
            console.print("[bold red]Monitoring stopped[/bold red]")
        else:
            print("\nMonitoring stopped")
    except Exception as e:
        if console:
            console.print(f"[bold red]Error in dashboard: {str(e)}[/bold red]")
        else:
            print(f"\nError in dashboard: {str(e)}")
    finally:
        if hasattr(monitor, "stop_monitoring"):
            monitor.stop_monitoring()


def sample_log_messages() -> List[str]:
    """Generate sample log messages for UI testing."""
    messages = [
        "[red]ERROR[/red]: Failed to connect to database server",
        "[yellow]WARNING[/yellow]: Memory usage approaching 85% threshold",
        "[green]INFO[/green]: Database connection established",
        "[blue]DEBUG[/blue]: Query executed: SELECT * FROM users",
    ]
    return messages


def create_dashboard() -> None:
    """Create a simple dashboard for WDBX monitoring."""
    if not RICH_AVAILABLE or not console:
        print("Rich library not available. Install with: pip install rich")
        return

    console.print("[bold blue]WDBX System Monitor[/bold blue]", justify="center")
    console.print("[yellow]Starting dashboard...[/yellow]")

    # Create a simple static dashboard
    table = Table(title="WDBX Dashboard", expand=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    table.add_row("Database", "Online")
    table.add_row("Vector Index", "Ready")
    table.add_row("Security", "Active")
    table.add_row("API Server", "Running")

    console.print(table)


if __name__ == "__main__":
    # Sample WDBX instance for testing
    class MockWDBX:
        def __init__(self):
            self.stats = {
                "blocks_created": 1234,
                "vectors_stored": 56789,
                "transactions_processed": 987,
                "traces_created": 456,
                "vector_dimension": 768,
                "shard_count": 4,
                "uptime": 3600 * 24 * 7,  # One week
                "timestamp": time.time(),
            }
            self.start_time = time.time()

        def get_system_stats(self) -> Dict[str, Any]:
            # Simulate changing stats
            self.stats["blocks_created"] += np.random.randint(0, 3)
            self.stats["vectors_stored"] += np.random.randint(0, 10)
            self.stats["transactions_processed"] += np.random.randint(0, 5)
            self.stats["traces_created"] += np.random.randint(0, 2)
            self.stats["uptime"] = time.time() - self.start_time
            self.stats["timestamp"] = time.time()
            return self.stats

        def get_memory_usage(self) -> float:
            return 1024.5 + np.random.uniform(-10, 10)

        def get_log_messages(self, limit: int = 20) -> List[str]:
            # Sample log messages
            msgs = [
                f"[{level.lower()}]{datetime.now().strftime('%H:%M:%S')} {msg}"
                for level, msg in [
                    ("INFO", "System initialized"),
                    ("DEBUG", "Vector search started"),
                    ("INFO", "Block added to chain"),
                    ("WARNING", "High latency detected"),
                    ("ERROR", "Transaction failed"),
                ]
            ]
            return msgs[-limit:]

    # Run the TUI with mock database
    mock_db = MockWDBX()
    run_terminal_ui(mock_db) # type: ignore # Suppress type error for mock object
