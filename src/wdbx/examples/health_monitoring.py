"""
Health Monitoring and Prometheus Metrics Example.

This example demonstrates how to use the health monitoring and Prometheus metrics
capabilities of WDBX to monitor system health and performance.
"""

import argparse
import logging
import os
import random
import time
from threading import Thread

import numpy as np

from wdbx import WDBX
from wdbx.health import HealthMonitor, print_health_report
from wdbx.prometheus import get_metrics, instrument
from wdbx.utils.logging_utils import configure_logging, get_logger

# Setup logging
configure_logging(level=logging.INFO)
logger = get_logger("health_example")


@instrument(name="generate_random_vector", kind="create")
def generate_random_vector(dim=128):
    """Generate a random vector of the specified dimension."""
    return np.random.random(dim).astype(np.float32)


@instrument(name="simulate_load", kind="search")
def simulate_load(wdbx_instance, iterations=100, vector_dim=128, batch_size=10):
    """
    Simulate load on the WDBX system.
    
    Args:
        wdbx_instance: WDBX instance
        iterations: Number of iterations to run
        vector_dim: Dimension of vectors
        batch_size: Batch size for operations
    """
    metrics = get_metrics()
    
    logger.info(f"Starting load simulation: {iterations} iterations")
    
    for i in range(iterations):
        try:
            # Create vectors
            vector_batch = []
            for j in range(batch_size):
                vector = generate_random_vector(vector_dim)
                vector_id = f"load_test_{i}_{j}"
                metadata = {"iteration": i, "batch_index": j, "test": True}
                vector_batch.append((vector_id, vector, metadata))
            
            # Batch create vectors
            with metrics.time_create_operation():
                wdbx_instance.batch_create_vectors(vector_batch)
            
            metrics.record_vector_batch_create(success=True)
            
            # Search for similar vectors
            query_vector = generate_random_vector(vector_dim)
            
            with metrics.time_search_operation():
                results = wdbx_instance.find_similar_vectors(
                    query_vector=query_vector,
                    top_k=5
                )
            
            metrics.record_vector_search(success=True)
            
            # Random sleep to simulate varying load
            time.sleep(random.uniform(0.01, 0.1))
            
            # Every 10 iterations, perform a memory optimization
            if i > 0 and i % 10 == 0:
                logger.info(f"Iteration {i}: Optimizing memory")
                try:
                    wdbx_instance.optimize_memory()
                    metrics.record_memory_optimization(success=True)
                except Exception as e:
                    logger.error(f"Memory optimization failed: {str(e)}")
                    metrics.record_memory_optimization(success=False)
            
            # Every 25 iterations, save vectors
            if i > 0 and i % 25 == 0:
                logger.info(f"Iteration {i}: Saving vectors")
                with metrics.time_persistence_operation(operation="save"):
                    wdbx_instance.save_all()
                metrics.record_persistence_operation(operation="save", success=True)
            
            # Update metrics after each iteration
            if i % 5 == 0:
                metrics.collect_all_stats(wdbx_instance)
                
        except Exception as e:
            logger.error(f"Error in iteration {i}: {str(e)}")
            metrics.record_vector_batch_create(success=False)
            metrics.record_vector_search(success=False)
    
    logger.info("Load simulation completed")


def monitor_health(wdbx_instance, interval=10, duration=None):
    """
    Periodically monitor health and print reports.
    
    Args:
        wdbx_instance: WDBX instance
        interval: Interval between health checks in seconds
        duration: Duration to run in seconds (None for indefinite)
    """
    start_time = time.time()
    iterations = 0
    
    logger.info(f"Starting health monitoring with {interval}s interval")
    
    while True:
        iterations += 1
        
        # Check if duration has been reached
        if duration and (time.time() - start_time) > duration:
            logger.info(f"Health monitoring stopped after {duration}s")
            break
        
        logger.info(f"Health check #{iterations}")
        print_health_report(wdbx_instance)
        
        time.sleep(interval)


def run_example(data_dir="./data/health_example", 
                vector_dim=128, 
                load_iterations=100,
                prometheus_port=9090,
                health_interval=10,
                duration=None):
    """
    Run the health monitoring and metrics example.
    
    Args:
        data_dir: Directory for data storage
        vector_dim: Dimension of vectors
        load_iterations: Number of iterations for load simulation
        prometheus_port: Port for Prometheus metrics server
        health_interval: Interval between health checks in seconds
        duration: Duration to run in seconds (None for indefinite)
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize WDBX
    logger.info(f"Initializing WDBX with vector dimension {vector_dim}")
    wdbx = WDBX(
        vector_dim=vector_dim,
        data_dir=data_dir,
        debug_mode=True
    )
    
    # Initialize metrics
    metrics = get_metrics(prefix="wdbx_example")
    metrics.start_server(port=prometheus_port)
    logger.info(f"Prometheus metrics server started on port {prometheus_port}")
    
    # Collect initial metrics
    metrics.collect_all_stats(wdbx)
    
    # Start health monitoring in a separate thread
    health_thread = Thread(
        target=monitor_health,
        args=(wdbx, health_interval, duration),
        daemon=True
    )
    health_thread.start()
    
    # Simulate load
    simulate_load(
        wdbx_instance=wdbx,
        iterations=load_iterations,
        vector_dim=vector_dim
    )
    
    # Final health check
    logger.info("Final health check:")
    print_health_report(wdbx)
    
    # If duration is specified, wait for the health monitoring thread
    if duration:
        health_thread.join()
    
    logger.info(f"Example completed. Data saved to {data_dir}")
    logger.info(f"Prometheus metrics available at http://localhost:{prometheus_port}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WDBX Health Monitoring and Metrics Example")
    parser.add_argument("--data-dir", default="./data/health_example", help="Directory for data storage")
    parser.add_argument("--vector-dim", type=int, default=128, help="Dimension of vectors")
    parser.add_argument("--load-iterations", type=int, default=100, help="Number of iterations for load simulation")
    parser.add_argument("--prometheus-port", type=int, default=9090, help="Port for Prometheus metrics server")
    parser.add_argument("--health-interval", type=int, default=10, help="Interval between health checks in seconds")
    parser.add_argument("--duration", type=int, default=None, help="Duration to run in seconds (default: until load simulation completes)")
    
    args = parser.parse_args()
    
    run_example(
        data_dir=args.data_dir,
        vector_dim=args.vector_dim,
        load_iterations=args.load_iterations,
        prometheus_port=args.prometheus_port,
        health_interval=args.health_interval,
        duration=args.duration
    ) 