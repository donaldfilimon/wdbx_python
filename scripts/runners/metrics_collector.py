#!/usr/bin/env python3
"""
WDBX Metrics Collector

This script collects performance metrics from a running WDBX instance
and provides visualization and monitoring capabilities.
"""

import argparse
import csv
import json
import logging
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("metrics_collector")

# Add project root to Python path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Try to import WDBX modules, but continue if not available
    # as we can still monitor using socket API
    from wdbx.utils.diagnostics import SystemMonitor
    HAS_WDBX = True
except ImportError:
    logger.warning("WDBX package not found. Limited functionality available.")
    HAS_WDBX = False

class MetricsCollector:
    """Collector for WDBX performance metrics."""
    
    def __init__(self, output_dir=None, interval=5):
        """
        Initialize the metrics collector.
        
        Args:
            output_dir: Directory for storing metrics output
            interval: Collection interval in seconds
        """
        self.interval = interval
        self.running = False
        self.last_stats = {}
        self.metrics_history = {
            "timestamp": [],
            "cpu_percent": [],
            "memory_percent": [],
            "disk_io_read": [],
            "disk_io_write": [],
            "query_count": [],
            "query_latency_ms": []
        }
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = PROJECT_ROOT / "metrics"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.output_dir / f"wdbx_metrics_{timestamp}.csv"
        self.json_file = self.output_dir / f"wdbx_metrics_{timestamp}.json"
        
        # Set up system monitor if available
        self.system_monitor = SystemMonitor() if HAS_WDBX else None
        
    def start_collection(self, background=True):
        """
        Start collecting metrics.
        
        Args:
            background: Run collector in background thread if True
        """
        self.running = True
        
        if background:
            self.collector_thread = threading.Thread(
                target=self._collection_loop,
                daemon=True
            )
            self.collector_thread.start()
            logger.info(f"Started metrics collection (interval: {self.interval}s)")
        else:
            self._collection_loop()
    
    def stop_collection(self):
        """Stop collecting metrics."""
        self.running = False
        logger.info("Stopping metrics collection")
        self.save_metrics()
    
    def _collection_loop(self):
        """Main collection loop."""
        try:
            # Write CSV header
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "cpu_percent", "memory_percent", 
                    "disk_io_read", "disk_io_write", 
                    "query_count", "query_latency_ms"
                ])
            
            # Collection loop
            while self.running:
                self.collect_single_snapshot()
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            logger.info("Collection interrupted by user")
            self.running = False
        except Exception as e:
            logger.error(f"Error in collection loop: {e}")
            self.running = False
        finally:
            self.save_metrics()
    
    def collect_single_snapshot(self):
        """Collect a single snapshot of metrics."""
        try:
            timestamp = datetime.now().isoformat()
            
            # System metrics via psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            disk_io_counters = psutil.disk_io_counters()
            
            # Calculate deltas for disk IO
            if not self.last_stats.get("disk_io"):
                disk_io_read_delta = 0
                disk_io_write_delta = 0
            else:
                last_io = self.last_stats["disk_io"]
                disk_io_read_delta = disk_io_counters.read_bytes - last_io.read_bytes
                disk_io_write_delta = disk_io_counters.write_bytes - last_io.write_bytes
            
            # Save current disk IO stats for next comparison
            self.last_stats["disk_io"] = disk_io_counters
            
            # Get application metrics if WDBX is available
            if self.system_monitor:
                wdbx_metrics = self.system_monitor.get_metrics()
                query_count = wdbx_metrics.get("query_count", 0)
                query_latency_ms = wdbx_metrics.get("avg_query_latency_ms", 0)
            else:
                query_count = 0
                query_latency_ms = 0
            
            # Store metrics in history
            self.metrics_history["timestamp"].append(timestamp)
            self.metrics_history["cpu_percent"].append(cpu_percent)
            self.metrics_history["memory_percent"].append(memory_percent)
            self.metrics_history["disk_io_read"].append(disk_io_read_delta)
            self.metrics_history["disk_io_write"].append(disk_io_write_delta)
            self.metrics_history["query_count"].append(query_count)
            self.metrics_history["query_latency_ms"].append(query_latency_ms)
            
            # Write to CSV file
            with open(self.csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, cpu_percent, memory_percent,
                    disk_io_read_delta, disk_io_write_delta,
                    query_count, query_latency_ms
                ])
            
            logger.debug(f"Collected metrics - CPU: {cpu_percent}%, Memory: {memory_percent}%")
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    def save_metrics(self):
        """Save collected metrics to files."""
        try:
            # Save as JSON
            with open(self.json_file, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
            
            logger.info(f"Metrics saved to {self.json_file}")
            
            # Generate visualization if matplotlib is available
            if MATPLOTLIB_AVAILABLE and len(self.metrics_history["timestamp"]) > 0:
                self.generate_visualization()
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def generate_visualization(self):
        """Generate visualization of collected metrics."""
        try:
            # Skip if no data or matplotlib not available
            if not MATPLOTLIB_AVAILABLE or len(self.metrics_history["timestamp"]) < 2:
                return
                
            # Convert ISO timestamps to datetime objects for plotting
            timestamps = [datetime.fromisoformat(ts) for ts in self.metrics_history["timestamp"]]
            
            # Create a new figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            
            # Plot CPU and memory usage
            axes[0].plot(timestamps, self.metrics_history["cpu_percent"], "b-", label="CPU %")
            axes[0].plot(timestamps, self.metrics_history["memory_percent"], "r-", label="Memory %")
            axes[0].set_title("System Resource Usage")
            axes[0].set_ylabel("Percent (%)")
            axes[0].grid(True)
            axes[0].legend()
            
            # Plot disk I/O
            axes[1].plot(timestamps, self.metrics_history["disk_io_read"], "g-", label="Disk Read (B)")
            axes[1].plot(timestamps, self.metrics_history["disk_io_write"], "m-", label="Disk Write (B)")
            axes[1].set_title("Disk I/O")
            axes[1].set_ylabel("Bytes")
            axes[1].grid(True)
            axes[1].legend()
            
            # Plot query metrics
            axes[2].plot(timestamps, self.metrics_history["query_count"], "c-", label="Query Count")
            axes[2].set_title("WDBX Query Performance")
            axes[2].set_ylabel("Count")
            axes[2].grid(True)
            
            # Add secondary y-axis for latency
            ax2 = axes[2].twinx()
            ax2.plot(timestamps, self.metrics_history["query_latency_ms"], "y-", label="Latency (ms)")
            ax2.set_ylabel("Latency (ms)")
            
            # Combine legends
            lines1, labels1 = axes[2].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            axes[2].legend(lines1 + lines2, labels1 + labels2, loc="upper left")
            
            # Set common x-axis label and adjust layout
            fig.autofmt_xdate()
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / f"{self.json_file.stem}.png"
            plt.savefig(plot_file)
            plt.close()
            
            logger.info(f"Visualization saved to {plot_file}")
            
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")

def connect_to_wdbx_server(host, port, timeout=5):
    """
    Connect to a WDBX server to monitor metrics.
    
    Args:
        host: Server hostname
        port: Server port
        timeout: Connection timeout in seconds
        
    Returns:
        Socket connection if successful, None otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        logger.info(f"Connected to WDBX server at {host}:{port}")
        return sock
    except Exception as e:
        logger.error(f"Failed to connect to WDBX server: {e}")
        return None

def main():
    """Main entry point for the metrics collector."""
    parser = argparse.ArgumentParser(description="WDBX Metrics Collector")
    
    parser.add_argument("--output-dir", "-o", help="Directory to store metrics output")
    parser.add_argument("--interval", "-i", type=int, default=5, 
                        help="Collection interval in seconds (default: 5)")
    parser.add_argument("--duration", "-d", type=int, 
                        help="Collection duration in seconds (default: run until interrupted)")
    parser.add_argument("--server", "-s", help="Connect to WDBX server at host:port")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse server connection string if provided
    server_sock = None
    if args.server:
        try:
            host, port = args.server.split(":")
            port = int(port)
            server_sock = connect_to_wdbx_server(host, port)
        except ValueError:
            logger.error("Invalid server format. Use host:port")
            return 1
    
    # Create metrics collector
    collector = MetricsCollector(args.output_dir, args.interval)
    
    try:
        # Start collection
        logger.info("Starting metrics collection...")
        
        # Run for specific duration or until interrupted
        if args.duration:
            collector.start_collection(background=True)
            logger.info(f"Collecting for {args.duration} seconds...")
            time.sleep(args.duration)
            collector.stop_collection()
        else:
            logger.info("Press Ctrl+C to stop collection")
            collector.start_collection(background=False)
            
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
    finally:
        # Make sure collection is stopped and metrics are saved
        if collector.running:
            collector.stop_collection()
        
        # Close server connection if open
        if server_sock:
            server_sock.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 