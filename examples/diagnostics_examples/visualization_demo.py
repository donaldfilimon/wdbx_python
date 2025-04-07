"""
WDBX Diagnostics Visualization Demo

This example demonstrates how to use the diagnostics visualization module to:
1. Create interactive charts and dashboards
2. Visualize system metrics
3. Monitor component performance
4. Track events over time
"""

import logging
import os
import random
import time
from datetime import datetime

from wdbx.utils import log_event, register_component, start_monitoring, stop_monitoring
from wdbx.utils.diagnostics_viz import get_visualizer

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wdbx_viz_demo")


class DatabaseComponent:
    """Sample database component for demonstration."""
    
    def __init__(self):
        self.query_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self.avg_query_time_ms = 0
    
    def get_metrics(self):
        """Return metrics for monitoring."""
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_ratio = self.cache_hits / max(cache_total, 1)
        
        return {
            "queries": self.query_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": cache_hit_ratio,
            "errors": self.errors,
            "avg_query_time_ms": self.avg_query_time_ms
        }
    
    def simulate_query(self):
        """Simulate a database query."""
        self.query_count += 1
        
        # Simulate cache behavior (70% hit rate)
        if random.random() < 0.7:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
        # Update average query time (random between 5-50ms)
        query_time = random.uniform(5, 50)
        self.avg_query_time_ms = ((self.avg_query_time_ms * (self.query_count - 1)) + query_time) / self.query_count
        
        # Simulate occasional errors (2% chance)
        if random.random() < 0.02:
            self.errors += 1
            log_event("error", {
                "message": f"Database query error #{self.errors}",
                "query_id": f"q-{self.query_count}",
                "component": "database"
            })
            
        return {"result": f"Query result {self.query_count}"}


class MLModelComponent:
    """Sample ML model component for demonstration."""
    
    def __init__(self):
        self.inference_count = 0
        self.avg_inference_time_ms = 0
        self.errors = 0
        self.cache_hits = 0
    
    def get_metrics(self):
        """Return metrics for monitoring."""
        return {
            "inference_count": self.inference_count,
            "avg_inference_time_ms": self.avg_inference_time_ms,
            "errors": self.errors,
            "cache_hits": self.cache_hits
        }
    
    def simulate_inference(self):
        """Simulate an ML model inference."""
        self.inference_count += 1
        
        # Simulate inference time (20-200ms)
        inference_time = random.uniform(20, 200)
        self.avg_inference_time_ms = (
            (self.avg_inference_time_ms * (self.inference_count - 1)) + inference_time
        ) / self.inference_count
        
        # Simulate cached results (30% hit rate)
        if random.random() < 0.3:
            self.cache_hits += 1
            
        # Simulate occasional errors (5% chance)
        if random.random() < 0.05:
            self.errors += 1
            log_event("warning", {
                "message": f"ML model inference warning #{self.errors}",
                "inference_id": f"inf-{self.inference_count}",
                "component": "ml_model"
            })
            
        return {"prediction": random.random()}


def create_output_dir():
    """Create output directory for visualizations."""
    output_dir = os.path.join(os.path.dirname(__file__), "viz_output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    """Main function to run the visualization demo."""
    logger.info("Starting WDBX diagnostics visualization demo")
    
    # Create output directory
    output_dir = create_output_dir()
    logger.info(f"Visualizations will be saved to: {output_dir}")
    
    # Start monitoring
    start_monitoring()
    logger.info("Started system monitoring")
    
    # Create sample components
    db = DatabaseComponent()
    model = MLModelComponent()
    
    # Register components with monitoring system
    register_component("database", db)
    register_component("ml_model", model)
    logger.info("Registered components for monitoring")
    
    # Simulate activity to generate metrics
    logger.info("Simulating system activity for 60 seconds...")
    end_time = time.time() + 60  # 1 minute
    
    # Log some initial events
    log_event("info", {"message": "System startup", "environment": "demo"})
    
    while time.time() < end_time:
        # Simulate database queries (0-3 per second)
        db_queries = random.randint(0, 3)
        for _ in range(db_queries):
            db.simulate_query()
        
        # Simulate ML model inferences (0-2 per second)
        model_inferences = random.randint(0, 2)
        for _ in range(model_inferences):
            model.simulate_inference()
        
        # Log occasional system events
        if random.random() < 0.1:  # 10% chance each second
            event_type = random.choice(["info", "warning"])
            log_event(event_type, {
                "message": f"System event at {datetime.now().strftime('%H:%M:%S')}",
                "cpu": random.uniform(10, 90),
                "memory": random.uniform(30, 80)
            })
        
        # Wait a bit
        time.sleep(1)
        
        # Log progress every 10 seconds
        elapsed = time.time() - (end_time - 60)
        if elapsed % 10 < 1:
            logger.info(f"Running for {int(elapsed)}s. " +
                      f"DB: {db.query_count} queries, " +
                      f"ML: {model.inference_count} inferences")
    
    logger.info("Activity simulation completed")
    logger.info("Creating visualizations...")
    
    # Get visualizer
    visualizer = get_visualizer()
    
    # Check visualization support
    if not visualizer.check_visualization_support():
        logger.error("Visualization libraries not available. Install matplotlib or plotly.")
        stop_monitoring()
        return
    
    # Create system overview with plotly
    logger.info("Creating system overview visualization (Plotly)...")
    system_overview_file = os.path.join(output_dir, "system_overview_plotly.html")
    visualizer.create_system_overview(
        output_file=system_overview_file,
        time_range_minutes=5,
        use_plotly=True,
        dark_mode=True
    )
    
    # Create system overview with matplotlib
    logger.info("Creating system overview visualization (Matplotlib)...")
    system_overview_file_mpl = os.path.join(output_dir, "system_overview_matplotlib.png")
    visualizer.create_system_overview(
        output_file=system_overview_file_mpl,
        time_range_minutes=5,
        use_plotly=False
    )
    
    # Create component dashboard
    logger.info("Creating component metrics dashboard...")
    component_dashboard_file = os.path.join(output_dir, "component_dashboard.html")
    visualizer.create_component_dashboard(
        component_names=["database", "ml_model"],
        output_file=component_dashboard_file,
        time_range_minutes=5,
        use_plotly=True
    )
    
    # Create events timeline
    logger.info("Creating events timeline...")
    events_timeline_file = os.path.join(output_dir, "events_timeline.html")
    visualizer.create_events_timeline(
        output_file=events_timeline_file,
        time_range_hours=1,
        use_plotly=True
    )
    
    # Stop monitoring
    stop_monitoring()
    logger.info("Stopped monitoring system")
    
    # Summary
    logger.info("\nVisualization demo completed!")
    logger.info("The following files were created:")
    logger.info(f"1. {system_overview_file}")
    logger.info(f"2. {system_overview_file_mpl}")
    logger.info(f"3. {component_dashboard_file}")
    logger.info(f"4. {events_timeline_file}")
    logger.info("\nOpen the HTML files in a web browser to view the interactive visualizations")
    logger.info("or view the PNG files in an image viewer.")


if __name__ == "__main__":
    main() 