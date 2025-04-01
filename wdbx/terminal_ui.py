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
        self.max_history_size = 100
        self.update_interval = 1.0  # seconds
        self.monitor_thread = None

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
        """Main monitoring loop."""
        while self.running:
            try:
                stats = self.wdbx.get_system_stats()

                # Add timestamp
                stats['timestamp'] = time.time()

                # Add to history
                self.stats_history.append(stats)

                # Trim history if needed
                if len(self.stats_history) > self.max_history_size:
                    self.stats_history = self.stats_history[-self.max_history_size:]

                # Print stats if no terminal UI
                if not RICH_AVAILABLE:
                    self._print_stats(stats)

                # Sleep
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(self.update_interval)

    def _print_stats(self, stats):
        """Print stats to console."""
        print("\033[2J\033[H")  # Clear screen
        print("=== WDBX System Monitor ===")
        print(f"Uptime: {stats['uptime']:.2f} seconds")
        print(f"Blocks: {stats['blocks_created']} ({stats['blocks_per_second']:.2f}/s)")
        print(f"Vectors: {stats['vectors_stored']} ({stats['vectors_per_second']:.2f}/s)")
        print(f"Transactions: {stats['transactions_processed']}")
        print(f"Traces: {stats['traces_created']}")
        print(f"Vector dimension: {stats['vector_dimension']}")
        print(f"Shard count: {stats['shard_count']}")
        print("============================")


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
            Layout(name="stats"),
            Layout(name="controls")
        )

        # Split the right panel into log and visualization
        layout["right"].split_column(
            Layout(name="log"),
            Layout(name="visualization")
        )

        # Create widgets
        self.stats_panel = Static(Panel("Loading stats...", title="System Statistics"))
        self.log_panel = TextLog(highlight=True, markup=True)
        self.controls_panel = Static(Panel("Controls", title="Actions"))
        self.viz_panel = Static(Panel("No data to visualize", title="Visualization"))

        # Add widgets to layout
        layout["header"].update(Header())
        layout["stats"].update(self.stats_panel)
        layout["log"].update(Panel(self.log_panel, title="System Log"))
        layout["controls"].update(self.controls_panel)
        layout["visualization"].update(self.viz_panel)
        layout["footer"].update(Footer())

        # Mount the layout
        await self.view.dock(layout)

        # Start monitors
        self.monitor.start_monitoring()
        self.set_interval(self.update_interval, self.update_stats)
        self.set_interval(1.0, self.update_log)

    async def update_stats(self) -> None:
        """Update the stats panel."""
        if not self.monitor.stats_history:
            return

        stats = self.monitor.stats_history[-1]

        # Create a rich table for the stats
        table = Table()
        table.add_column("Metric")
        table.add_column("Value")

        table.add_row("Uptime", f"{stats['uptime']:.2f} seconds")
        table.add_row("Blocks", f"{stats['blocks_created']} ({stats['blocks_per_second']:.2f}/s)")
        table.add_row("Vectors", f"{stats['vectors_stored']} ({stats['vectors_per_second']:.2f}/s)")
        table.add_row("Transactions", f"{stats['transactions_processed']}")
        table.add_row("Traces", f"{stats['traces_created']}")
        table.add_row("Vector dimension", f"{stats['vector_dimension']}")
        table.add_row("Shard count", f"{stats['shard_count']}")

        # Update the stats panel
        self.stats_panel.update(Panel(table, title="System Statistics"))

    async def update_log(self) -> None:
        """Update the log panel with recent log messages."""
        # This would normally subscribe to a log handler
        # For now, just add a placeholder message
        if not hasattr(self, 'log_counter'):
            self.log_counter = 0

        self.log_counter += 1
        if self.log_counter % 5 == 0:
            self.log_panel.write(f"[green]INFO[/green]: System running normally at {time.time()}")
        if self.log_counter % 10 == 0:
            self.log_panel.write(f"[blue]DEBUG[/blue]: Vector store contains {self.wdbx.vector_store.get_vector_count()} vectors")

    async def action_refresh(self) -> None:
        """Refresh the UI (bound to F5)."""
        await self.update_stats()
        await self.update_log()

    async def action_quit(self) -> None:
        """Quit the application (bound to q)."""
        self.monitor.stop_monitoring()
        await self.shutdown()


def run_terminal_ui(wdbx_instance: WDBX):
    """
    Run the terminal UI.

    Args:
        wdbx_instance: WDBX instance to monitor
    """
    if RICH_AVAILABLE:
        app = WDBXTerminalUI(wdbx_instance)
        app.run()
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
    Run a simple dashboard using just Rich (no full terminal UI).

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
            monitor_task = progress.add_task("[green]Monitoring...", total=None)

            while True:
                # Get stats
                stats = wdbx_instance.get_system_stats()

                # Clear screen
                console.clear()

                # Show header
                console.print(f"[bold blue]WDBX System Monitor[/bold blue] - Running for {stats['uptime']:.2f} seconds")
                console.print()

                # Show stats in a table
                table = Table(title="System Statistics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Blocks created", str(stats['blocks_created']))
                table.add_row("Blocks per second", f"{stats['blocks_per_second']:.2f}")
                table.add_row("Vectors stored", str(stats['vectors_stored']))
                table.add_row("Vectors per second", f"{stats['vectors_per_second']:.2f}")
                table.add_row("Transactions processed", str(stats['transactions_processed']))
                table.add_row("Traces created", str(stats['traces_created']))

                console.print(table)
                console.print()

                # Show system info
                console.print("[bold]System Information[/bold]")
                console.print(f"Vector dimension: {stats['vector_dimension']}")
                console.print(f"Shard count: {stats['shard_count']}")

                # Show commands
                console.print()
                console.print("[bold yellow]Commands:[/bold yellow]")
                console.print("Press Ctrl+C to exit")

                # Update progress
                progress.update(monitor_task, advance=1)

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
