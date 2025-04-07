#!/usr/bin/env python
"""
Test script for the ML integration module.

This script tests the ML integration module's vector operations and backend selection.
It verifies that the ML backend correctly handles vector operations and benchmarks
performance across different backends.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("test_ml_integration")

# Add the project root to Python path if needed
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def import_ml_integration() -> Tuple[Any, bool, bool]:
    """
    Import the ML integration module and check available backends.
    
    Returns:
        Tuple of (ml_backend, JAX_AVAILABLE, TORCH_AVAILABLE)
    """
    try:
        from wdbx.ml.ml_integration import (
            JAX_AVAILABLE,
            TORCH_AVAILABLE,
            get_ml_backend,
        )
        return get_ml_backend(), JAX_AVAILABLE, TORCH_AVAILABLE
    except ImportError as e:
        logger.error(f"Failed to import ML integration module: {e}")
        logger.error("Make sure the wdbx package is in your Python path")
        sys.exit(1)

def benchmark_operation(
    operation_name: str,
    operation_func: callable,
    iterations: int = 5,
    warmup: int = 2
) -> Tuple[float, float]:
    """
    Benchmark an operation by running it multiple times.
    
    Args:
        operation_name: Name of the operation for logging
        operation_func: Function to benchmark (should take no args)
        iterations: Number of iterations to run
        warmup: Number of warmup iterations
        
    Returns:
        Tuple of (mean_time, std_dev)
    """
    logger.debug(f"Benchmarking {operation_name} with {iterations} iterations")
    
    # Run warmup iterations (not timed)
    for _ in range(warmup):
        operation_func()
    
    # Run timed iterations
    times = []
    for i in range(iterations):
        start_time = time.time()
        result = operation_func()
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        logger.debug(f"  Iteration {i+1}: {elapsed_time:.6f} seconds")
    
    # Calculate statistics
    mean_time = sum(times) / len(times)
    std_dev = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    
    logger.info(f"{operation_name}: mean={mean_time:.6f}s, std_dev={std_dev:.6f}s")
    return mean_time, std_dev

def benchmark_vector_operations(
    ml_backend: Any,
    dimensions: List[int] = [128, 768, 1536],
    batch_sizes: List[int] = [100, 1000]
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark vector operations using the ML backend.
    
    Args:
        ml_backend: The ML backend to use
        dimensions: List of vector dimensions to test
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dictionary of benchmark results
    """
    results = {}
    
    for dim in dimensions:
        logger.info(f"Testing {dim}-dimensional vectors")
        dim_results = {}
        
        # Create test vectors
        vector1 = np.random.rand(dim).astype(np.float32)
        vector2 = np.random.rand(dim).astype(np.float32)
        
        # Single vector operations
        logger.info("Testing single vector operations")
        
        # To backend format
        mean_time, _ = benchmark_operation(
            f"to_{ml_backend.backend}",
            lambda: ml_backend.to_backend_format(vector1)
        )
        dim_results[f"to_{ml_backend.backend}"] = mean_time
        
        # Vector normalization
        mean_time, _ = benchmark_operation(
            "normalize",
            lambda: ml_backend.normalize(vector1)
        )
        dim_results["normalize"] = mean_time
        
        # Cosine similarity
        mean_time, _ = benchmark_operation(
            "cosine_similarity",
            lambda: ml_backend.cosine_similarity(vector1, vector2)
        )
        dim_results["cosine_similarity"] = mean_time
        
        # Batch operations
        for batch_size in batch_sizes:
            logger.info(f"Testing batch operations with {batch_size} vectors")
            
            # Create batch of vectors
            vectors = [np.random.rand(dim).astype(np.float32) for _ in range(batch_size)]
            
            # Batch cosine similarity
            mean_time, _ = benchmark_operation(
                f"batch_cosine_similarity_{batch_size}",
                lambda: ml_backend.batch_cosine_similarity(vector1, vectors)
            )
            dim_results[f"batch_cosine_similarity_{batch_size}"] = mean_time
        
        results[f"dim_{dim}"] = dim_results
    
    return results

def verify_vectors_equal(
    v1: np.ndarray,
    v2: np.ndarray,
    tolerance: float = 1e-5
) -> bool:
    """
    Verify that two vectors are equal within tolerance.
    
    Args:
        v1: First vector
        v2: Second vector
        tolerance: Error tolerance
        
    Returns:
        True if vectors are equal within tolerance
    """
    if v1.shape != v2.shape:
        logger.error(f"Shape mismatch: {v1.shape} vs {v2.shape}")
        return False
    
    max_diff = np.max(np.abs(v1 - v2))
    if max_diff > tolerance:
        logger.error(f"Max difference: {max_diff} (exceeds tolerance {tolerance})")
        return False
    
    return True

def test_backend_conversions(ml_backend: Any) -> bool:
    """
    Test conversion between backends.
    
    Args:
        ml_backend: The ML backend to use
        
    Returns:
        True if all tests pass
    """
    logger.info("Testing backend conversions")
    
    # Create test vector
    vector_np = np.random.rand(128).astype(np.float32)
    
    # Test numpy conversion (identity)
    vector_np_copy = ml_backend.to_numpy(vector_np)
    if not verify_vectors_equal(vector_np, vector_np_copy):
        logger.error("NumPy to NumPy conversion failed")
        return False
    
    # Test backend conversion and back
    vector_backend = ml_backend.to_backend_format(vector_np)
    vector_np_round_trip = ml_backend.to_numpy(vector_backend)
    
    if not verify_vectors_equal(vector_np, vector_np_round_trip):
        logger.error("Round-trip conversion failed")
        return False
    
    logger.info("Backend conversions test passed")
    return True

def test_similarity_calculations(ml_backend: Any) -> bool:
    """
    Test similarity calculations.
    
    Args:
        ml_backend: The ML backend to use
        
    Returns:
        True if all tests pass
    """
    logger.info("Testing similarity calculations")
    
    # Create test vectors
    vector1 = np.random.rand(128).astype(np.float32)
    vector2 = np.random.rand(128).astype(np.float32)
    
    # Calculate cosine similarity using NumPy
    vector1_norm = vector1 / np.linalg.norm(vector1)
    vector2_norm = vector2 / np.linalg.norm(vector2)
    numpy_similarity = np.dot(vector1_norm, vector2_norm)
    
    # Calculate using ML backend
    backend_similarity = ml_backend.cosine_similarity(vector1, vector2)
    
    # Verify results match
    if abs(numpy_similarity - backend_similarity) > 1e-5:
        logger.error(f"Similarity mismatch: NumPy={numpy_similarity}, Backend={backend_similarity}")
        return False
    
    # Test with normalized vectors
    vector1_norm_backend = ml_backend.normalize(vector1)
    vector2_norm_backend = ml_backend.normalize(vector2)
    norm_similarity = ml_backend.cosine_similarity(vector1_norm_backend, vector2_norm_backend)
    
    if abs(numpy_similarity - norm_similarity) > 1e-5:
        logger.error(f"Normalized similarity mismatch: Expected={numpy_similarity}, Got={norm_similarity}")
        return False
    
    logger.info("Similarity calculations test passed")
    return True

def test_batch_operations(ml_backend: Any) -> bool:
    """
    Test batch vector operations.
    
    Args:
        ml_backend: The ML backend to use
        
    Returns:
        True if all tests pass
    """
    logger.info("Testing batch operations")
    
    # Create test vectors
    query = np.random.rand(128).astype(np.float32)
    batch_size = 10
    vectors = [np.random.rand(128).astype(np.float32) for _ in range(batch_size)]
    
    # Calculate similarities one by one
    individual_similarities = [
        ml_backend.cosine_similarity(query, vector) for vector in vectors
    ]
    
    # Calculate batch similarities
    batch_similarities = ml_backend.batch_cosine_similarity(query, vectors)
    
    # Verify results match
    if len(batch_similarities) != batch_size:
        logger.error(f"Expected {batch_size} similarities, got {len(batch_similarities)}")
        return False
    
    for i, (ind_sim, batch_sim) in enumerate(zip(individual_similarities, batch_similarities)):
        if abs(ind_sim - batch_sim) > 1e-5:
            logger.error(f"Similarity mismatch at index {i}: Individual={ind_sim}, Batch={batch_sim}")
            return False
    
    logger.info("Batch operations test passed")
    return True

def main() -> int:
    """
    Main function for testing the ML integration module.
    
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(description="Test ML integration module")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmarks"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--dimension", "-d",
        type=int,
        default=128,
        help="Vector dimension to use for tests"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1000,
        help="Batch size to use for tests"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Import ML integration module
    ml_backend, JAX_AVAILABLE, TORCH_AVAILABLE = import_ml_integration()
    logger.info(f"Using ML backend: {ml_backend.backend}")
    logger.info(f"Available backends: JAX={JAX_AVAILABLE}, PyTorch={TORCH_AVAILABLE}")
    
    # Run core tests
    test_passed = True
    
    test_passed = test_backend_conversions(ml_backend) and test_passed
    test_passed = test_similarity_calculations(ml_backend) and test_passed
    test_passed = test_batch_operations(ml_backend) and test_passed
    
    # Run benchmarks if requested
    if args.benchmark:
        logger.info("Running benchmarks")
        
        # Create test vectors for micro-benchmarks
        vector1 = np.random.rand(args.dimension).astype(np.float32)
        vector2 = np.random.rand(args.dimension).astype(np.float32)
        
        # Test conversion
        _, _ = benchmark_operation(
            "to_numpy",
            lambda: ml_backend.to_numpy(vector1)
        )
        
        # Test to backend format
        _, _ = benchmark_operation(
            f"to_{ml_backend.backend}",
            lambda: ml_backend.to_backend_format(vector1)
        )
        
        # Test cosine similarity
        _, _ = benchmark_operation(
            "cosine_similarity",
            lambda: ml_backend.cosine_similarity(vector1, vector2)
        )
        
        # Prepare batch test
        vectors = [np.random.rand(args.dimension).astype(np.float32) for _ in range(args.batch_size)]
        
        # Test batch cosine similarity
        _, _ = benchmark_operation(
            "batch_cosine_similarity",
            lambda: ml_backend.batch_cosine_similarity(vector1, vectors)
        )
    
    if test_passed:
        logger.info("All tests passed successfully")
        return 0
    logger.error("Some tests failed")
    return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.error("Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1) 