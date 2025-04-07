# wdbx/benchmarks.py
"""
Benchmarking utilities for WDBX performance testing.

This module provides functions to benchmark different aspects of WDBX,
particularly focusing on comparing performance between NumPy, JAX, and PyTorch
backends for vector operations.
"""
import contextlib
import logging
import random
import time
from functools import wraps
from typing import Callable, Dict

import numpy as np

from ..core.constants import VECTOR_DIMENSION, logger
from ..core.data_structures import EmbeddingVector
from ..ml import JAX_AVAILABLE, TORCH_AVAILABLE, get_ml_backend
from ..storage.vector_store import VectorStore

# Create ML backend for conversion functions
ml_backend = get_ml_backend()

# Function to create a timer context manager


@contextlib.contextmanager
def timer(name: str = None) -> float:
    """
    Context manager for timing code execution.

    Args:
        name: Optional name for the timer

    Yields:
        None

    Returns timing result via the context manager protocol.
    """
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    if name:
        logger.info(f"{name} took {elapsed_time:.6f} seconds")
    return elapsed_time


def benchmark_decorator(func: Callable) -> Callable:
    """
    Decorator to benchmark a function's execution time.

    Args:
        func: Function to benchmark

    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        func_name = func.__name__
        logger.info(f"Function {func_name} took {elapsed_time:.6f} seconds")
        return result, elapsed_time
    return wrapper


def generate_random_vectors(count: int, dimension: int = VECTOR_DIMENSION) -> np.ndarray:
    """
    Generate random vectors for benchmarking.

    Args:
        count: Number of vectors to generate
        dimension: Dimension of each vector

    Returns:
        Array of random vectors
    """
    return np.random.randn(count, dimension).astype(np.float32)


def benchmark_backend_conversion(vectors: np.ndarray, iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark conversion between backends.

    Args:
        vectors: Vectors to convert
        iterations: Number of iterations for the benchmark

    Returns:
        Dictionary of times for different conversion operations
    """
    results = {
        "numpy_to_jax": 0.0,
        "numpy_to_torch": 0.0,
        "jax_to_numpy": 0.0,
        "torch_to_numpy": 0.0,
        "jax_to_torch": 0.0,
        "torch_to_jax": 0.0
    }

    # Skip benchmarks for unavailable backends
    if not JAX_AVAILABLE:
        for key in list(results.keys()):
            if "jax" in key:
                del results[key]

    if not TORCH_AVAILABLE:
        for key in list(results.keys()):
            if "torch" in key:
                del results[key]

    # Perform conversion benchmarks
    if JAX_AVAILABLE:
        pass

        # NumPy to JAX
        with timer("NumPy to JAX") as t:
            for _ in range(iterations):
                jax_vectors = ml_backend.to_jax(vectors)
        results["numpy_to_jax"] = t

        # JAX to NumPy
        with timer("JAX to NumPy") as t:
            for _ in range(iterations):
                numpy_vectors = ml_backend.to_numpy(jax_vectors)
        results["jax_to_numpy"] = t

    if TORCH_AVAILABLE:
        pass

        # NumPy to PyTorch
        with timer("NumPy to PyTorch") as t:
            for _ in range(iterations):
                torch_vectors = ml_backend.to_torch(vectors)
        results["numpy_to_torch"] = t

        # PyTorch to NumPy
        with timer("PyTorch to NumPy") as t:
            for _ in range(iterations):
                _numpy_vectors = ml_backend.to_numpy(torch_vectors)
        results["torch_to_numpy"] = t

    # Cross-framework conversion if both are available
    if JAX_AVAILABLE and TORCH_AVAILABLE:
        pass

        jax_vectors = ml_backend.to_jax(vectors)
        torch_vectors = ml_backend.to_torch(vectors)

        # JAX to PyTorch
        with timer("JAX to PyTorch") as t:
            for _ in range(iterations):
                _torch_from_jax = ml_backend.to_torch(jax_vectors)
        results["jax_to_torch"] = t

        # PyTorch to JAX
        with timer("PyTorch to JAX") as t:
            for _ in range(iterations):
                _jax_from_torch = ml_backend.to_jax(torch_vectors)
        results["torch_to_jax"] = t

    return results


def benchmark_cosine_similarity(vectors: np.ndarray, batch_size: int = 100) -> Dict[str, float]:
    """
    Benchmark cosine similarity computation using different backends.

    Args:
        vectors: Vectors to use for benchmark
        batch_size: Number of similarity comparisons to perform

    Returns:
        Dictionary of times for different backends
    """
    results = {
        "numpy": 0.0,
        "jax": 0.0,
        "torch": 0.0,
        "torch_gpu": 0.0
    }

    # Create query vector and targets
    query = vectors[0]
    targets = vectors[1:batch_size + 1]

    # NumPy benchmark
    with timer("NumPy cosine similarity") as t:
        # Normalize vectors
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        similarities = []
        for target in targets:
            target_norm = np.linalg.norm(target)
            if target_norm > 0:
                target = target / target_norm
            similarity = np.dot(query, target)
            similarities.append(similarity)

    results["numpy"] = t

    # JAX benchmark
    if JAX_AVAILABLE:
        import jax
        import jax.numpy as jnp

        # Define and compile cosine similarity function
        @jax.jit
        def cosine_sim(v1, v2):
            v1 = v1 / jnp.maximum(jnp.linalg.norm(v1), 1e-8)
            v2 = v2 / jnp.maximum(jnp.linalg.norm(v2), 1e-8)
            return jnp.dot(v1, v2)

        # Convert to JAX
        jax_query = ml_backend.to_jax(query)
        jax_targets = ml_backend.to_jax(targets)

        # Warmup JIT
        _ = cosine_sim(jax_query, jax_targets[0])

        with timer("JAX cosine similarity") as t:
            _jax_similarities = [cosine_sim(jax_query, target) for target in jax_targets]
        results["jax"] = t
    else:
        del results["jax"]

    # PyTorch benchmark
    if TORCH_AVAILABLE:
        import torch
        import torch.nn.functional as F

        # CPU benchmark
        torch_query = ml_backend.to_torch(query)
        torch_targets = ml_backend.to_torch(targets)

        # Normalize vectors
        torch_query = F.normalize(torch_query, p=2, dim=0)

        with timer("PyTorch CPU cosine similarity") as t:
            torch_similarities = []
            for target in torch_targets:
                target = F.normalize(target, p=2, dim=0)
                similarity = torch.dot(torch_query, target)
                torch_similarities.append(similarity.item())

        results["torch"] = t

        # GPU benchmark if available
        if torch.cuda.is_available():
            torch_query_gpu = torch_query.cuda()
            torch_targets_gpu = [target.cuda() for target in torch_targets]

            with timer("PyTorch GPU cosine similarity") as t:
                torch_similarities_gpu = []
                for target in torch_targets_gpu:
                    target = F.normalize(target, p=2, dim=0)
                    similarity = torch.dot(torch_query_gpu, target)
                    torch_similarities_gpu.append(similarity.item())

            results["torch_gpu"] = t
        else:
            del results["torch_gpu"]
    else:
        del results["torch"]
        del results["torch_gpu"]

    return results


def benchmark_batch_operations(vectors: np.ndarray, batch_size: int = 1000) -> Dict[str, float]:
    """
    Benchmark batch vector operations using different backends.

    Args:
        vectors: Vectors to use for benchmark
        batch_size: Size of the batch

    Returns:
        Dictionary of times for different operations
    """
    results = {
        "numpy_normalize": 0.0,
        "jax_normalize": 0.0,
        "torch_normalize": 0.0,
        "numpy_matmul": 0.0,
        "jax_matmul": 0.0,
        "torch_matmul": 0.0
    }

    # Use a subset of vectors
    batch = vectors[:batch_size]

    # NumPy normalize
    with timer("NumPy batch normalize") as t:
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        _normalized = batch / np.maximum(norms, 1e-8)
    results["numpy_normalize"] = t

    # NumPy matmul
    with timer("NumPy batch matmul") as t:
        _result = np.matmul(batch, batch.T)
    results["numpy_matmul"] = t

    # JAX operations
    if JAX_AVAILABLE:
        import jax.numpy as jnp

        jax_batch = ml_backend.to_jax(batch)

        # JAX normalize
        with timer("JAX batch normalize") as t:
            jax_norms = jnp.linalg.norm(jax_batch, axis=1, keepdims=True)
            _jax_normalized = jax_batch / jnp.maximum(jax_norms, 1e-8)
        results["jax_normalize"] = t

        # JAX matmul
        with timer("JAX batch matmul") as t:
            _jax_result = jnp.matmul(jax_batch, jax_batch.T)
        results["jax_matmul"] = t
    else:
        del results["jax_normalize"]
        del results["jax_matmul"]

    # PyTorch operations
    if TORCH_AVAILABLE:
        import torch
        import torch.nn.functional as F

        torch_batch = ml_backend.to_torch(batch)

        # PyTorch normalize
        with timer("PyTorch batch normalize") as t:
            _torch_normalized = F.normalize(torch_batch, p=2, dim=1)
        results["torch_normalize"] = t

        # PyTorch matmul
        with timer("PyTorch batch matmul") as t:
            _torch_result = torch.matmul(torch_batch, torch_batch.T)
        results["torch_matmul"] = t
    else:
        del results["torch_normalize"]
        del results["torch_matmul"]

    return results


def benchmark_vector_store(vectors: np.ndarray, query_count: int = 10,
                           top_k: int = 100) -> Dict[str, float]:
    """
    Benchmark vector store operations.

    Args:
        vectors: Vectors to add to the store
        query_count: Number of queries to perform
        top_k: Number of results to return per query

    Returns:
        Dictionary of times for different operations
    """
    results = {
        "add_vectors": 0.0,
        "search_vectors": 0.0
    }

    vector_count = len(vectors)

    # Create vector store
    vector_store = VectorStore(dimension=vectors.shape[1])

    # Add vectors
    with timer(f"Adding {vector_count} vectors") as t:
        for i, vec in enumerate(vectors):
            embedding = EmbeddingVector(vec, {"index": i})
            vector_store.add(f"vec_{i}", embedding)

    results["add_vectors"] = t

    # Search vectors
    total_search_time = 0.0
    for i in range(query_count):
        query_idx = random.randint(0, vector_count - 1)
        query = vectors[query_idx]

        with timer() as t:
            _similar = vector_store.search_similar(query, top_k=top_k)
        total_search_time += t

    avg_search_time = total_search_time / query_count
    results["search_vectors"] = avg_search_time
    logger.info(f"Average search time for {top_k} results: {avg_search_time:.6f} seconds")

    return results


def benchmark_vector_conversion():
    """Benchmark vector format conversion between numpy, jax, and torch."""
    # Generate random vectors
    size = 1000
    dim = 1024

    # Create vectors in different formats
    try:
        # NumPy vectors (always available)
        start = time.time()
        _ = np.random.rand(size, dim).astype(np.float32)
        numpy_time = time.time() - start
        logger.info(f"NumPy vector creation: {numpy_time:.4f}s")

        # Try JAX if available
        if JAX_AVAILABLE:
            import jax.numpy as jnp
            import jax.random as random

            _key = random.PRNGKey(0)
            start = time.time()
            jax_vectors = jnp.array(np.random.rand(size, dim).astype(np.float32))
            jax_time = time.time() - start
            logger.info(f"JAX vector creation: {jax_time:.4f}s")

            # Convert JAX to NumPy
            start = time.time()
            _ = np.array(jax_vectors)
            jax_to_numpy_time = time.time() - start
            logger.info(f"JAX to NumPy conversion: {jax_to_numpy_time:.4f}s")

        # Try PyTorch if available
        if TORCH_AVAILABLE:
            import torch

            start = time.time()
            torch_vectors = torch.rand(size, dim, dtype=torch.float32)
            torch_time = time.time() - start
            logger.info(f"PyTorch vector creation: {torch_time:.4f}s")

            # Convert Torch to NumPy
            start = time.time()
            _ = torch_vectors.numpy()
            torch_to_numpy_time = time.time() - start
            logger.info(f"PyTorch to NumPy conversion: {torch_to_numpy_time:.4f}s")

        # Cross-framework conversions if both are available
        if JAX_AVAILABLE and TORCH_AVAILABLE:
            # JAX to PyTorch
            start = time.time()
            _ = torch.from_numpy(np.array(jax_vectors))
            jax_to_torch_time = time.time() - start
            logger.info(f"JAX to PyTorch conversion: {jax_to_torch_time:.4f}s")

            # PyTorch to JAX
            start = time.time()
            _ = jnp.array(torch_vectors.numpy())
            torch_to_jax_time = time.time() - start
            logger.info(f"PyTorch to JAX conversion: {torch_to_jax_time:.4f}s")

    except Exception as e:
        logger.error(f"Error in vector conversion benchmark: {e}")


def run_all_benchmarks(vector_count: int = 10000,
                       dimension: int = VECTOR_DIMENSION) -> Dict[str, Dict[str, float]]:
    """
    Run all benchmarks and return results.

    Args:
        vector_count: Number of vectors to use in benchmarks
        dimension: Dimension of vectors

    Returns:
        Dictionary of benchmark results
    """
    logger.info(f"Running benchmarks with {vector_count} vectors of dimension {dimension}")
    logger.info(
        f"Available backends: NumPy, {
            'JAX, ' if JAX_AVAILABLE else ''}{
            'PyTorch' if TORCH_AVAILABLE else ''}")

    # Generate random vectors
    vectors = generate_random_vectors(vector_count, dimension)

    results = {}

    # Run backend conversion benchmarks
    logger.info("Running backend conversion benchmarks...")
    results["conversion"] = benchmark_backend_conversion(vectors[:1000], iterations=10)

    # Run cosine similarity benchmarks
    logger.info("Running cosine similarity benchmarks...")
    results["cosine_similarity"] = benchmark_cosine_similarity(vectors, batch_size=1000)

    # Run batch operation benchmarks
    logger.info("Running batch operation benchmarks...")
    results["batch_operations"] = benchmark_batch_operations(vectors, batch_size=1000)

    # Run vector store benchmarks
    logger.info("Running vector store benchmarks...")
    results["vector_store"] = benchmark_vector_store(vectors[:5000], query_count=10, top_k=100)

    logger.info("All benchmarks completed")
    return results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Run benchmarks
    results = run_all_benchmarks(vector_count=10000)

    # Print summary
    print("\nBenchmark Summary:")
    for category, category_results in results.items():
        print(f"\n{category.upper()}:")
        for operation, time_taken in category_results.items():
            print(f"  {operation}: {time_taken:.6f} seconds")
