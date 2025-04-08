import logging
import time

from wdbx.utils.diagnostics import (
    get_metrics,
    get_monitor,
    log_event,
    start_monitoring,
    stop_monitoring,
)

# Configure basic logging to see output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("diagnostics_demo")


def simulate_task():
    """Simulates a task that takes some time."""
    logger.info("Simulating a task...")
    time.sleep(2)
    logger.info("Task simulation finished.")


if __name__ == "__main__":
    logger.info("Starting diagnostics demo...")

    # Start global monitoring (check interval defaults to 5 seconds)
    start_monitoring()
    logger.info("System monitoring started.")

    # Give monitor a moment to start
    time.sleep(1)

    # Log a custom event
    log_event("info", {"message": "Diagnostics demo started", "user": "demo_script"})
    logger.info("Logged a custom info event.")

    # Get the monitor instance to use the context manager
    monitor = get_monitor()

    # Demonstrate timing an operation using the context manager
    logger.info("Timing a simulated task using context manager...")
    with monitor.time_operation("simulated_task"):
        simulate_task()
    logger.info("Finished timing the simulated task.")

    # Get and print initial metrics
    logger.info("Fetching initial metrics...")
    initial_metrics = get_metrics()
    logger.info(f"Initial Metrics: {initial_metrics}")

    # Wait for another monitoring cycle
    logger.info("Waiting for 6 seconds for another monitoring cycle...")
    time.sleep(6)

    # Get and print updated metrics
    logger.info("Fetching updated metrics...")
    updated_metrics = get_metrics()
    logger.info(f"Updated Metrics: {updated_metrics}")

    # Stop monitoring
    stop_monitoring()
    logger.info("System monitoring stopped.")

    logger.info("Diagnostics demo finished.")
