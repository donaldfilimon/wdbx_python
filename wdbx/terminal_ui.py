# wdbx/terminal_ui.py
"""
Terminal UI for the WDBX database.

This module provides a rich terminal interface for monitoring, visualizing,
and interacting with the WDBX database system.
"""
import os
import sys
import time
import threading
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Try to import rich and textual for the terminal UI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich.layout import Layout
    from rich.text import Text
    from rich.chart import Chart
    from textual.app import App
    from textual.widgets import Header, Footer, Static, Button, DataTable, TextLog
    from textual.reactive import Reactive
    from textual.containers import Container, Horizontal, Vertical
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich and Textual libraries not available. Install with: pip install rich textual")

from wdbx import WDBX
from wdbx.data_structures import EmbeddingVector
from wdbx.constants import logger, VECTOR_DIMENSION, SHARD_COUNT

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
        self.stats_history = []
        self.max_history_size = 1000  # Increased history size
        self.update_interval = 0.5  # Decreased interval for more responsive updates
        self.monitor_thread = None
        self.last_stats = None

    def start_monitoring(self):
        """Start the monitoring thread."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop with enhanced error handling and stats calculation."""
        while self.running:
            try:
                stats = self.wdbx.get_system_stats()
                current_time = time.time()

                # Add timestamp and calculate deltas
                stats['timestamp'] = current_time
                if self.last_stats:
                    time_delta = current_time - self.last_stats['timestamp']
                    stats['blocks_delta'] = (stats['blocks_created'] - self.last_stats['blocks_created']) / time_delta
                    stats['vectors_delta'] = (stats['vectors_stored'] - self.last_stats['vectors_stored']) / time_delta

                self.last_stats = stats.copy()
                self.stats_history.append(stats)

                # Trim history if needed
                if len(self.stats_history) > self.max_history_size:
                    self.stats_history = self.stats_history[-self.max_history_size:]

                # Print stats if no terminal UI
                if not RICH_AVAILABLE:
                    self._print_stats(stats)

                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitor loop: {str(e)}")
                time.sleep(self.update_interval * 2)  # Back off on error

    def _print_stats(self, stats):
        """Enhanced stats printing with color and formatting."""
        print("\033[2J\033[H")  # Clear screen
        print("\033[1;34m=== WDBX System Monitor ===\033[0m")
        print(f"\033[1mUptime:\033[0m {stats['uptime']:.2f} seconds")
        print(f"\033[1mBlocks:\033[0m {stats['blocks_created']} ({stats.get('blocks_delta', 0):.2f}/s)")
        print(f"\033[1mVectors:\033[0m {stats['vectors_stored']} ({stats.get('vectors_delta', 0):.2f}/s)")
        print(f"\033[1mTransactions:\033[0m {stats['transactions_processed']}")
        print(f"\033[1mTraces:\033[0m {stats['traces_created']}")
        print(f"\033[1mVector dimension:\033[0m {stats['vector_dimension']}")
        print(f"\033[1mShard count:\033[0m {stats['shard_count']}")
        print("\033[1;34m============================\033[0m")


class WDBXTerminalUI(App):
    """
    Rich terminal UI for WDBX using Textual.
    """
    TITLE = "WDBX Terminal UI"
    SUB_TITLE = "Database Monitoring and Management"

    def __init__(self, wdbx_instance: WDBX):
        super().__init__()
        self.wdbx = wdbx_instance
        self.monitor = WDBXMonitor(wdbx_instance)
        self.update_interval = 0.5  # seconds
        self.chart_data = []
        self.max_chart_points = 50

    async def on_mount(self) -> None:
        """Set up the UI when the app is mounted."""
        # Set up the layout
        self.header = Header()
        self.footer = Footer()

        # Create the main layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )

        # Split the main area into sections
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2)
        )

        # Split the left panel into stats and controls
        layout["left"].split_column(
            Layout(name="stats", ratio=2),
            Layout(name="controls", ratio=1)
        )

        # Split the right panel into log and visualization
        layout["right"].split_column(
            Layout(name="log", ratio=1),
            Layout(name="visualization", ratio=2)
        )

        # Create enhanced widgets with better styling
        self.stats_panel = Static(Panel("Loading stats...", title="System Statistics", border_style="blue"))
        self.log_panel = TextLog(highlight=True, markup=True)
        self.controls_panel = Static(Panel(self._create_controls(), title="Actions", border_style="green"))
        self.viz_panel = Static(Panel(self._create_visualization(), title="Performance Metrics", border_style="yellow"))

        # Add widgets to layout with improved styling
        layout["header"].update(Header())
        layout["stats"].update(self.stats_panel)
        layout["log"].update(Panel(self.log_panel, title="System Log", border_style="red"))
        layout["controls"].update(self.controls_panel)
        layout["visualization"].update(self.viz_panel)
        layout["footer"].update(Footer())

        # Mount the layout
        await self.view.dock(layout)

        # Start monitors with improved update frequency
        self.monitor.start_monitoring()
        self.set_interval(self.update_interval, self.update_stats)
        self.set_interval(0.5, self.update_log)
        self.set_interval(2.0, self.update_visualization)

    def _create_controls(self) -> Container:
        """Create enhanced control buttons."""
        container = Container()
        container.add_widget(Button("Refresh Stats", variant="primary"))
        container.add_widget(Button("Clear Log", variant="warning"))
        container.add_widget(Button("Exit", variant="error"))
        return container

    def _create_visualization(self) -> Chart:
        """Create performance visualization chart."""
        chart = Chart()
        chart.add_series("Vectors/s", color="green")
        chart.add_series("Blocks/s", color="blue")
        return chart

    async def update_stats(self) -> None:
        """Update the stats panel with enhanced metrics."""
        if not self.monitor.stats_history:
            return

        stats = self.monitor.stats_history[-1]

        # Create an enhanced stats table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Current", justify="right", style="green")
        table.add_column("Rate", justify="right", style="yellow")

        table.add_row(
            "Uptime", 
            f"{stats['uptime']:.2f}s",
            ""
        )
        table.add_row(
            "Blocks", 
            str(stats['blocks_created']),
            f"{stats.get('blocks_delta', 0):.2f}/s"
        )
        table.add_row(
            "Vectors", 
            str(stats['vectors_stored']),
            f"{stats.get('vectors_delta', 0):.2f}/s"
        )
        table.add_row(
            "Transactions",
            str(stats['transactions_processed']),
            ""
        )
        table.add_row(
            "Traces",
            str(stats['traces_created']),
            ""
        )

        # System configuration
        table.add_section()
        table.add_row(
            "Vector Dimension",
            str(stats['vector_dimension']),
            ""
        )
        table.add_row(
            "Shard Count",
            str(stats['shard_count']),
            ""
        )

        # Update the stats panel with improved styling
        self.stats_panel.update(Panel(table, title="System Statistics", border_style="blue"))

    async def update_log(self) -> None:
        """Update the log panel with enhanced logging."""
        if not hasattr(self, 'log_counter'):
            self.log_counter = 0

        self.log_counter += 1

        # Add more detailed log messages
        if self.log_counter % 5 == 0:
            stats = self.monitor.stats_history[-1] if self.monitor.stats_history else None
            if stats:
                self.log_panel.write(
                    f"[green]INFO[/green]: System running normally - "
                    f"Vectors: {stats['vectors_stored']} "
                    f"({stats.get('vectors_delta', 0):.2f}/s)"
                )

        if self.log_counter % 10 == 0:
            self.log_panel.write(
                f"[blue]DEBUG[/blue]: Performance metrics - "
                f"Memory usage: {self.wdbx.get_memory_usage():.2f}MB"
            )

    async def action_refresh(self) -> None:
        """Refresh all UI components."""
        await self.update_stats()
        await self.update_log()
        await self.update_visualization()

    async def action_quit(self) -> None:
        """Gracefully shut down the application."""
        self.monitor.stop_monitoring()
        await self.shutdown()


def run_terminal_ui(wdbx_instance: WDBX):
    """
    Run the enhanced terminal UI with better error handling.

    Args:
        wdbx_instance: WDBX instance to monitor
    """
    if RICH_AVAILABLE:
        try:
            app = WDBXTerminalUI(wdbx_instance)
            app.run()
        except Exception as e:
            console.print(f"[red]Error running terminal UI: {str(e)}[/red]")
            # Fallback to simple monitor
            run_simple_dashboard(wdbx_instance)
    else:
        # Fallback to simple monitor
        monitor = WDBXMonitor(wdbx_instance)
        monitor.start_monitoring()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("\nMonitoring stopped")


def run_simple_dashboard(wdbx_instance: WDBX):
    """
    Run an enhanced simple dashboard using just Rich.

    Args:
        wdbx_instance: WDBX instance to monitor
    """
    if not RICH_AVAILABLE:
        print("Rich library not available. Install with: pip install rich")
        return

    console = Console()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            monitor_task = progress.add_task("[green]Monitoring WDBX...", total=None)
            performance_task = progress.add_task("[cyan]Performance", total=None)

            while True:
                # Get enhanced stats
                stats = wdbx_instance.get_system_stats()

                # Clear screen
                console.clear()

                # Show enhanced header
                console.print(f"[bold blue]WDBX System Monitor[/bold blue]", justify="center")
                console.print(f"[blue]Running for {stats['uptime']:.2f} seconds[/blue]", justify="center")
                console.print()

                # Show enhanced stats in a table
                table = Table(title="System Statistics", expand=True)
                table.add_column("Metric", style="cyan", justify="right")
                table.add_column("Current Value", style="green")
                table.add_column("Rate", style="yellow")

                table.add_row(
                    "Blocks",
                    str(stats['blocks_created']),
                    f"{stats['blocks_per_second']:.2f}/s"
                )
                table.add_row(
                    "Vectors",
                    str(stats['vectors_stored']),
                    f"{stats['vectors_per_second']:.2f}/s"
                )
                table.add_row(
                    "Transactions",
                    str(stats['transactions_processed']),
                    "-"
                )
                table.add_row(
                    "Traces",
                    str(stats['traces_created']),
                    "-"
                )

                console.print(table)
                console.print()

                # Show enhanced system info
                console.print(Panel(
                    f"Vector dimension: {stats['vector_dimension']}\n"
                    f"Shard count: {stats['shard_count']}\n"
                    f"Memory usage: {wdbx_instance.get_memory_usage():.2f}MB",
                    title="System Information",
                    border_style="blue"
                ))

                # Show enhanced commands
                console.print()
                console.print("[bold yellow]Commands:[/bold yellow]")
                console.print("Press [bold]Ctrl+C[/bold] to exit")
                console.print("Press [bold]R[/bold] to refresh")

                # Update progress
                progress.update(monitor_task, advance=1)
                progress.update(performance_task, advance=stats['vectors_per_second'])

                # Sleep
                time.sleep(1)

    except KeyboardInterrupt:
        console.print("[bold red]Monitoring stopped[/bold red]")


if __name__ == "__main__":
    # Example usage
    from wdbx import WDBX

    wdbx = WDBX(vector_dimension=128, num_shards=4)

    # Create some sample data
    for i in range(10):
        vector = np.random.randn(wdbx.vector_dimension).astype(np.float32)
        embedding = EmbeddingVector(
            vector=vector,
            metadata={"description": f"Sample embedding {i}", "timestamp": time.time()}
        )
        wdbx.store_embedding(embedding)

    # Run the appropriate UI based on available libraries
    if RICH_AVAILABLE:
        run_terminal_ui(wdbx)
    else:
        monitor = WDBXMonitor(wdbx)
        monitor.start_monitoring()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("\nMonitoring stopped")
