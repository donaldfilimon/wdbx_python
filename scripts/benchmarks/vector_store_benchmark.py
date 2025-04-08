#!/usr/bin/env python3

"""
WDBX Vector Store Benchmark Script
---------------------------------
Compares the performance of different backend configurations for vector search operations.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vector_store_benchmark")

# Add project root to sys.path to ensure wdbx can be imported
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import WDBX components
try:
    # Try to import from the correct modules
    try:
        from wdbx.core.data_structures import EmbeddingVector
    except ImportError:
        from wdbx.data_structures import EmbeddingVector

    # Try to get the backend availability flags
    try:
        from wdbx.ml import JAX_AVAILABLE, TORCH_AVAILABLE
    except ImportError:
        logger.warning("Could not import ML backend flags, assuming backends are not available")
        JAX_AVAILABLE = False
        TORCH_AVAILABLE = False

    # Try to import MLBackend with fallbacks
    try:
        from wdbx.ml.ml_integration import MLBackend
    except ImportError:
        try:
            from wdbx.ml.backend import MLBackend
        except ImportError:
            logger.warning("MLBackend not found. Creating a minimal mock class.")

            class MLBackend:
                default_backend = "numpy"

    from wdbx.storage.vector_store import VectorStore
except ImportError as e:
    logger.error(f"Failed to import WDBX modules: {e}")
    sys.exit(1)


def generate_random_embeddings(count: int, dimension: int) -> List[Tuple[str, EmbeddingVector]]:
    """
    Generate random embedding vectors for benchmarking.

    Args:
        count: Number of vectors to generate
        dimension: Vector dimension

    Returns:
        List of (id, embedding) tuples
    """
    embeddings = []
    for i in range(count):
        vector = np.random.randn(dimension).astype(np.float32)
        embedding = EmbeddingVector(
            vector=vector,
            metadata={"index": i, "type": "random", "norm": float(np.linalg.norm(vector))},
        )
        vector_id = f"vec_{i:06d}"
        embeddings.append((vector_id, embedding))
    return embeddings


def setup_vector_store(
    backend: str, dimension: int, embeddings: List[Tuple[str, EmbeddingVector]]
) -> VectorStore:
    """
    Set up a vector store with the specified backend and embeddings.

    Args:
        backend: Backend to use ('auto', 'numpy', 'jax', or 'torch')
        dimension: Vector dimension
        embeddings: List of embeddings to add to the store

    Returns:
        Configured VectorStore instance
    """
    # Configure ML backend first
    old_backend = None
    try:
        if hasattr(MLBackend, "default_backend"):
            old_backend = MLBackend.default_backend
            MLBackend.default_backend = backend
            logger.info(f"Set ML backend to {backend}")
        else:
            logger.warning("MLBackend doesn't have default_backend attribute. Using default.")
    except Exception as e:
        logger.warning(f"Could not set ML backend: {e}")

    # Create vector store
    try:
        # First try with vector_dimension parameter (correct one)
        store = VectorStore(vector_dimension=dimension)
    except TypeError:
        # Fallback for backward compatibility
        try:
            store = VectorStore(dimension=dimension)
        except Exception as e:
            logger.error(f"Could not create VectorStore: {e}")
            # Last resort - check what parameters the constructor expects
            import inspect

            sig = inspect.signature(VectorStore.__init__)
            logger.error(f"VectorStore constructor expects: {list(sig.parameters.keys())}")
            # Just try the simple version and let it fail naturally
            store = VectorStore(dimension)

    # Add embeddings
    for vector_id, embedding in embeddings:
        store.add(vector_id, embedding)

    # Restore original backend if changed
    try:
        if old_backend is not None and hasattr(MLBackend, "default_backend"):
            MLBackend.default_backend = old_backend
            logger.info(f"Restored ML backend to {old_backend}")
    except Exception as e:
        logger.warning(f"Could not restore ML backend: {e}")

    return store


def benchmark_vector_search(
    store: VectorStore,
    queries: List[np.ndarray],
    top_k: int = 10,
    threshold: float = 0.5,
    runs: int = 3,
) -> Dict[str, float]:
    """
    Benchmark vector similarity search.

    Args:
        store: Vector store to benchmark
        queries: Query vectors
        top_k: Maximum number of results to return
        threshold: Minimum similarity threshold
        runs: Number of runs to average over

    Returns:
        Dictionary with benchmark results
    """
    results = {
        "total_time": 0.0,
        "avg_time": 0.0,
        "min_time": float("inf"),
        "max_time": 0.0,
        "per_query": 0.0,
    }

    for _ in range(runs):
        start_time = time.time()

        for query in queries:
            _ = store.search_similar(query, top_k=top_k, threshold=threshold)

        elapsed = time.time() - start_time
        results["total_time"] += elapsed
        results["min_time"] = min(results["min_time"], elapsed)
        results["max_time"] = max(results["max_time"], elapsed)

    results["avg_time"] = results["total_time"] / runs
    results["per_query"] = results["avg_time"] / len(queries)

    return results


def main():
    """
    Main benchmark function.
    """
    # Parameters
    vector_count = 10000
    query_count = 100
    dimensions = [128, 768, 1536]

    # Determine available backends with better error handling
    backends = ["numpy"]  # NumPy should always be available

    try:
        if "TORCH_AVAILABLE" in globals() and TORCH_AVAILABLE:
            backends.append("torch")
            logger.info("PyTorch backend is available")
    except Exception as e:
        logger.warning(f"Error checking PyTorch availability: {e}")

    try:
        if "JAX_AVAILABLE" in globals() and JAX_AVAILABLE:
            backends.append("jax")
            logger.info("JAX backend is available")
    except Exception as e:
        logger.warning(f"Error checking JAX availability: {e}")

    logger.info("Starting WDBX Vector Store benchmark")
    logger.info(f"Testing {len(backends)} backends: {', '.join(backends)}")
    logger.info(f"Vector dimensions: {dimensions}")
    logger.info(f"Vector count: {vector_count}, Query count: {query_count}")

    # Results storage
    all_results = {}

    for dimension in dimensions:
        logger.info(f"\n===== BENCHMARKING VECTOR DIMENSION {dimension} =====")
        all_results[dimension] = {}

        # Generate embeddings and queries
        try:
            logger.info(f"Generating {vector_count} random embeddings with dimension {dimension}")
            embeddings = generate_random_embeddings(vector_count, dimension)

            logger.info(f"Generating {query_count} random query vectors")
            queries = [np.random.randn(dimension).astype(np.float32) for _ in range(query_count)]
        except Exception as e:
            logger.error(f"Error generating vectors: {e}")
            continue

        for backend in backends:
            logger.info(f"\n----- Testing {backend} backend -----")

            try:
                # Set up vector store
                logger.info(f"Setting up vector store with {backend} backend")
                store = setup_vector_store(backend, dimension, embeddings)

                # Warm up
                logger.info("Warming up...")
                _ = store.search_similar(queries[0], top_k=10)

                # Benchmark
                logger.info("Running search benchmark...")
                results = benchmark_vector_search(store, queries)

                logger.info(f"Average search time: {results['avg_time']:.6f} seconds")
                logger.info(f"Per query: {results['per_query']:.6f} seconds")

                all_results[dimension][backend] = results
            except Exception as e:
                logger.error(f"Error benchmarking {backend} backend: {e}")
                all_results[dimension][backend] = {
                    "error": str(e),
                    "avg_time": float("inf"),
                    "per_query": float("inf"),
                }

    # Print summary
    logger.info("\n===== BENCHMARK SUMMARY =====")
    logger.info("\nVector Search Performance (seconds):")
    logger.info(f"{'Dimension':10} {'Backend':10} {'Total':10} {'Per Query':10}")

    for dimension in all_results:
        for backend, results in all_results[dimension].items():
            if "error" in results:
                logger.info(f"{dimension:10d} {backend:10} {'ERROR':10} {'ERROR':10}")
            else:
                logger.info(
                    f"{dimension:10d} {backend:10} {results['avg_time']:10.6f} {results['per_query']:10.6f}"
                )

    # Calculate speedups
    if len(backends) > 1:
        logger.info("\nSpeedup over NumPy:")
        logger.info(f"{'Dimension':10} {'Backend':10} {'Speedup':10}")
        logger.info("-" * 40)

        for dimension in dimensions:
            numpy_time = all_results[dimension]["numpy"]["avg_time"]

            for backend in backends:
                if backend == "numpy":
                    continue

                backend_time = all_results[dimension][backend]["avg_time"]
                speedup = numpy_time / backend_time

                logger.info(f"{dimension:10} {backend:10} {speedup:.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
