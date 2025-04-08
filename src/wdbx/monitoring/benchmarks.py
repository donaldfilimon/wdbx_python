# wdbx/benchmarks.py
"""
Benchmarking utilities for WDBX performance testing.

This module provides functions to benchmark different aspects of WDBX,
particularly focusing on comparing performance between NumPy, JAX, and PyTorch
backends for vector operations.
"""
import contextlib
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from ..core.constants import VECTOR_DIMENSION, logger
from ..ml import JAX_AVAILABLE, TORCH_AVAILABLE, ArrayLike, get_ml_backend
from ..storage.vector_store import VectorStore

# Create ML backend for conversion functions
ml_backend = get_ml_backend()


class BenchmarkType(Enum):
    """Types of benchmarks that can be run."""

    BACKEND_CONVERSION = "backend_conversion"
    VECTOR_OPERATIONS = "vector_operations"
    SIMILARITY_SEARCH = "similarity_search"
    BATCH_OPERATIONS = "batch_operations"
    ATTENTION = "attention"
    NEURAL_BACKTRACKING = "neural_backtracking"
    VECTOR_STORE = "vector_store"
    ML_INTEGRATION = "ml_integration"


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    operation: str
    backend: str
    execution_time: float
    operations_per_second: float = 0.0
    vector_count: int = 0
    vector_dimension: int = 0
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def __post_init__(self):
        """Calculate operations per second if not provided."""
        if self.operations_per_second == 0.0 and self.execution_time > 0:
            if self.vector_count > 0:
                self.operations_per_second = self.vector_count / self.execution_time


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


def normalize_vectors(vectors: ArrayLike, backend: str = "auto") -> ArrayLike:
    """
    Normalize vectors using the specified backend.

    Args:
        vectors: Vectors to normalize
        backend: Backend to use ('numpy', 'jax', 'torch', or 'auto')

    Returns:
        Normalized vectors
    """
    # If backend is auto, use the best available
    if backend == "auto":
        if JAX_AVAILABLE:
            backend = "jax"
        elif TORCH_AVAILABLE:
            backend = "torch"
        else:
            backend = "numpy"

    # Convert vectors to the right format
    if backend == "jax" and JAX_AVAILABLE:
        import jax.numpy as jnp

        vectors_backend = ml_backend.to_jax(vectors)
        norm = jnp.linalg.norm(vectors_backend, axis=1, keepdims=True)
        normalized = vectors_backend / (norm + 1e-8)  # Avoid division by zero
        return normalized
    elif backend == "torch" and TORCH_AVAILABLE:
        import torch.nn.functional as F

        vectors_backend = ml_backend.to_torch(vectors)
        normalized = F.normalize(vectors_backend, p=2, dim=1)
        return normalized
    else:
        # NumPy fallback
        vectors_np = ml_backend.to_numpy(vectors)
        norm = np.linalg.norm(vectors_np, axis=1, keepdims=True)
        normalized = vectors_np / (norm + 1e-8)  # Avoid division by zero
        return normalized


def benchmark_backend_conversion(
    vectors: np.ndarray, iterations: int = 100
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark conversion between backends.

    Args:
        vectors: Vectors to convert
        iterations: Number of iterations for the benchmark

    Returns:
        Dictionary of benchmark results for different conversion operations
    """
    operations = {
        "numpy_to_jax": ("numpy", "jax"),
        "numpy_to_torch": ("numpy", "torch"),
        "jax_to_numpy": ("jax", "numpy"),
        "torch_to_numpy": ("torch", "numpy"),
        "jax_to_torch": ("jax", "torch"),
        "torch_to_jax": ("torch", "jax"),
    }

    results = {}

    # Skip benchmarks for unavailable backends
    if not JAX_AVAILABLE:
        for key in list(operations.keys()):
            if "jax" in key:
                results[key] = BenchmarkResult(
                    name=key,
                    operation="conversion",
                    backend="jax",
                    execution_time=0,
                    vector_count=vectors.shape[0],
                    vector_dimension=vectors.shape[1],
                    error="JAX not available",
                )
                del operations[key]

    if not TORCH_AVAILABLE:
        for key in list(operations.keys()):
            if "torch" in key:
                results[key] = BenchmarkResult(
                    name=key,
                    operation="conversion",
                    backend="torch",
                    execution_time=0,
                    vector_count=vectors.shape[0],
                    vector_dimension=vectors.shape[1],
                    error="PyTorch not available",
                )
                del operations[key]

    # Create backend-specific vectors
    backend_vectors = {"numpy": vectors}

    if JAX_AVAILABLE:
        import jax.numpy as jnp

        backend_vectors["jax"] = jnp.array(vectors)

    if TORCH_AVAILABLE:
        import torch

        backend_vectors["torch"] = torch.tensor(vectors)

    # Perform conversion benchmarks
    for name, (source, target) in operations.items():
        try:
            source_vec = backend_vectors[source]

            # Benchmark the conversion
            with timer(f"{source} to {target}") as t:
                for _ in range(iterations):
                    if target == "jax":
                        result = ml_backend.to_jax(source_vec)
                    elif target == "torch":
                        result = ml_backend.to_torch(source_vec)
                    else:  # numpy
                        result = ml_backend.to_numpy(source_vec)

            # Store the result
            results[name] = BenchmarkResult(
                name=name,
                operation="conversion",
                backend=f"{source}->{target}",
                execution_time=t,
                vector_count=vectors.shape[0],
                vector_dimension=vectors.shape[1],
                operations_per_second=iterations / t,
                additional_metrics={"batch_size": vectors.shape[0], "iterations": iterations},
            )
        except Exception as e:
            logger.error(f"Error in {name} benchmark: {e}")
            results[name] = BenchmarkResult(
                name=name,
                operation="conversion",
                backend=f"{source}->{target}",
                execution_time=0,
                vector_count=vectors.shape[0],
                vector_dimension=vectors.shape[1],
                error=str(e),
            )

    return results


def benchmark_cosine_similarity(
    vectors: np.ndarray, batch_size: int = 100
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark cosine similarity computation using different backends.

    Args:
        vectors: Vectors to use for benchmark
        batch_size: Number of similarity comparisons to perform

    Returns:
        Dictionary of benchmark results for different backends
    """
    results = {}
    backends = ["numpy"]

    if JAX_AVAILABLE:
        backends.append("jax")

    if TORCH_AVAILABLE:
        backends.append("torch")
        # Check if CUDA is available for PyTorch
        import torch

        if torch.cuda.is_available():
            backends.append("torch_gpu")

    # Create query vector and targets
    query = vectors[0]
    targets = vectors[1 : batch_size + 1]

    # Benchmark each backend
    for backend in backends:
        try:
            if backend == "numpy":
                # NumPy benchmark
                with timer("NumPy cosine similarity") as t:
                    # Use our ML backend's implementation
                    for target in targets:
                        _ = ml_backend.cosine_similarity(query, target)

                results["numpy"] = BenchmarkResult(
                    name="cosine_similarity",
                    operation="similarity",
                    backend="numpy",
                    execution_time=t,
                    vector_count=batch_size,
                    vector_dimension=vectors.shape[1],
                    operations_per_second=batch_size / t,
                    additional_metrics={"batch_size": batch_size},
                )

            elif backend == "jax":
                # JAX benchmark
                import jax
                import jax.numpy as jnp

                # Convert to JAX arrays
                query_jax = ml_backend.to_jax(query)
                targets_jax = ml_backend.to_jax(targets)

                # Define JIT-compiled cosine similarity function
                @jax.jit
                def cosine_sim(v1, v2):
                    v1_norm = v1 / jnp.linalg.norm(v1)
                    v2_norm = v2 / jnp.linalg.norm(v2)
                    return jnp.dot(v1_norm, v2_norm)

                # Warm-up JIT compilation
                _ = cosine_sim(query_jax, targets_jax[0])

                # Benchmark
                with timer("JAX cosine similarity") as t:
                    for i in range(batch_size):
                        _ = cosine_sim(query_jax, targets_jax[i % len(targets_jax)])

                results["jax"] = BenchmarkResult(
                    name="cosine_similarity",
                    operation="similarity",
                    backend="jax",
                    execution_time=t,
                    vector_count=batch_size,
                    vector_dimension=vectors.shape[1],
                    operations_per_second=batch_size / t,
                    additional_metrics={"batch_size": batch_size, "jit_compiled": True},
                )

            elif backend == "torch":
                # PyTorch CPU benchmark
                import torch
                import torch.nn.functional as F

                # Convert to PyTorch tensors on CPU
                query_torch = ml_backend.to_torch(query)
                targets_torch = ml_backend.to_torch(targets)

                # Benchmark
                with timer("PyTorch CPU cosine similarity") as t:
                    for target in targets_torch:
                        # Use PyTorch's cosine similarity
                        _ = F.cosine_similarity(
                            query_torch.unsqueeze(0), target.unsqueeze(0), dim=1
                        )

                results["torch"] = BenchmarkResult(
                    name="cosine_similarity",
                    operation="similarity",
                    backend="torch_cpu",
                    execution_time=t,
                    vector_count=batch_size,
                    vector_dimension=vectors.shape[1],
                    operations_per_second=batch_size / t,
                    additional_metrics={"batch_size": batch_size, "device": "cpu"},
                )

            elif backend == "torch_gpu":
                # PyTorch GPU benchmark
                import torch
                import torch.nn.functional as F

                # Get GPU device
                device = torch.device("cuda")

                # Convert to PyTorch tensors on GPU
                query_gpu = torch.tensor(ml_backend.to_numpy(query), device=device)
                targets_gpu = torch.tensor(ml_backend.to_numpy(targets), device=device)

                # Benchmark
                with timer("PyTorch GPU cosine similarity") as t:
                    for i in range(batch_size):
                        idx = i % len(targets_gpu)
                        # Use PyTorch's cosine similarity
                        _ = F.cosine_similarity(
                            query_gpu.unsqueeze(0), targets_gpu[idx].unsqueeze(0), dim=1
                        )

                results["torch_gpu"] = BenchmarkResult(
                    name="cosine_similarity",
                    operation="similarity",
                    backend="torch_gpu",
                    execution_time=t,
                    vector_count=batch_size,
                    vector_dimension=vectors.shape[1],
                    operations_per_second=batch_size / t,
                    additional_metrics={
                        "batch_size": batch_size,
                        "device": torch.cuda.get_device_name(0),
                        "cuda_version": torch.version.cuda,
                    },
                )

        except Exception as e:
            logger.error(f"Error in {backend} cosine similarity benchmark: {e}")
            results[backend] = BenchmarkResult(
                name="cosine_similarity",
                operation="similarity",
                backend=backend,
                execution_time=0,
                vector_count=batch_size,
                vector_dimension=vectors.shape[1],
                error=str(e),
            )

    return results


def benchmark_batch_operations(
    vectors: np.ndarray, batch_size: int = 1000
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark batch vector operations using different backends.

    Args:
        vectors: Vectors to use for benchmark
        batch_size: Number of vectors to process in a batch

    Returns:
        Dictionary of benchmark results for different operations and backends
    """
    results = {}

    # Only use up to batch_size vectors
    if vectors.shape[0] > batch_size:
        vectors = vectors[:batch_size]

    # Define backends to test
    backends = ["numpy"]
    if JAX_AVAILABLE:
        backends.append("jax")
    if TORCH_AVAILABLE:
        backends.append("torch")
        # Check if CUDA is available for PyTorch
        import torch

        if torch.cuda.is_available():
            backends.append("torch_gpu")

    # Define operations to benchmark
    operations = {
        "normalize": "Vector normalization",
        "batch_similarity": "Batch cosine similarity",
        "matrix_multiply": "Matrix multiplication",
    }

    # Benchmark each operation on each backend
    for op_name, op_desc in operations.items():
        for backend in backends:
            result_key = f"{backend}_{op_name}"

            try:
                if op_name == "normalize":
                    # Benchmark vector normalization
                    if backend == "numpy":
                        with timer(f"NumPy normalize ({batch_size} vectors)") as t:
                            vecs = ml_backend.to_numpy(vectors)
                            _ = normalize_vectors(vecs, backend="numpy")

                    elif backend == "jax":
                        import jax
                        import jax.numpy as jnp

                        # Define JIT-compiled normalization function
                        @jax.jit
                        def normalize_batch(vecs):
                            return vecs / jnp.linalg.norm(vecs, axis=1, keepdims=True)

                        # Convert to JAX
                        vecs_jax = ml_backend.to_jax(vectors)

                        # Warm-up JIT
                        _ = normalize_batch(vecs_jax)

                        # Benchmark
                        with timer(f"JAX normalize ({batch_size} vectors)") as t:
                            _ = normalize_batch(vecs_jax)

                    elif backend == "torch":
                        import torch
                        import torch.nn.functional as F

                        # Convert to PyTorch on CPU
                        vecs_torch = ml_backend.to_torch(vectors)

                        # Benchmark
                        with timer(f"PyTorch CPU normalize ({batch_size} vectors)") as t:
                            _ = F.normalize(vecs_torch, p=2, dim=1)

                    elif backend == "torch_gpu":
                        import torch
                        import torch.nn.functional as F

                        # Convert to PyTorch on GPU
                        device = torch.device("cuda")
                        vecs_gpu = torch.tensor(ml_backend.to_numpy(vectors), device=device)

                        # Benchmark
                        with timer(f"PyTorch GPU normalize ({batch_size} vectors)") as t:
                            _ = F.normalize(vecs_gpu, p=2, dim=1)

                elif op_name == "batch_similarity":
                    # Benchmark batch cosine similarity
                    # Use the first vector as query against all others
                    query = vectors[0]

                    if backend == "numpy":
                        with timer(f"NumPy batch similarity ({batch_size} vectors)") as t:
                            query_np = ml_backend.to_numpy(query)
                            vectors_np = ml_backend.to_numpy(vectors)

                            # Normalize query
                            query_norm = query_np / np.linalg.norm(query_np)

                            # Normalize all vectors
                            vectors_norm = vectors_np / np.linalg.norm(
                                vectors_np, axis=1, keepdims=True
                            )

                            # Compute batch similarity
                            _ = np.dot(vectors_norm, query_norm)

                    elif backend == "jax":
                        import jax
                        import jax.numpy as jnp

                        # Define JIT-compiled batch similarity function
                        @jax.jit
                        def batch_similarity(q, vs):
                            q_norm = q / jnp.linalg.norm(q)
                            vs_norm = vs / jnp.linalg.norm(vs, axis=1, keepdims=True)
                            return jnp.dot(vs_norm, q_norm)

                        # Convert to JAX
                        query_jax = ml_backend.to_jax(query)
                        vectors_jax = ml_backend.to_jax(vectors)

                        # Warm-up JIT
                        _ = batch_similarity(query_jax, vectors_jax)

                        # Benchmark
                        with timer(f"JAX batch similarity ({batch_size} vectors)") as t:
                            _ = batch_similarity(query_jax, vectors_jax)

                    elif backend == "torch":
                        import torch
                        import torch.nn.functional as F

                        # Convert to PyTorch on CPU
                        query_torch = ml_backend.to_torch(query)
                        vectors_torch = ml_backend.to_torch(vectors)

                        # Normalize
                        query_norm = F.normalize(query_torch.unsqueeze(0), p=2, dim=1)
                        vectors_norm = F.normalize(vectors_torch, p=2, dim=1)

                        # Benchmark
                        with timer(f"PyTorch CPU batch similarity ({batch_size} vectors)") as t:
                            _ = torch.matmul(vectors_norm, query_norm.squeeze().unsqueeze(1))

                    elif backend == "torch_gpu":
                        import torch
                        import torch.nn.functional as F

                        # Convert to PyTorch on GPU
                        device = torch.device("cuda")
                        query_gpu = torch.tensor(ml_backend.to_numpy(query), device=device)
                        vectors_gpu = torch.tensor(ml_backend.to_numpy(vectors), device=device)

                        # Normalize
                        query_norm = F.normalize(query_gpu.unsqueeze(0), p=2, dim=1)
                        vectors_norm = F.normalize(vectors_gpu, p=2, dim=1)

                        # Benchmark
                        with timer(f"PyTorch GPU batch similarity ({batch_size} vectors)") as t:
                            _ = torch.matmul(vectors_norm, query_norm.squeeze().unsqueeze(1))

                elif op_name == "matrix_multiply":
                    # Benchmark matrix multiplication (vectors · vectors.T)
                    if backend == "numpy":
                        with timer(f"NumPy matrix multiply ({batch_size}x{batch_size})") as t:
                            vectors_np = ml_backend.to_numpy(vectors)
                            _ = np.matmul(vectors_np, vectors_np.T)

                    elif backend == "jax":
                        import jax
                        import jax.numpy as jnp

                        # Define JIT-compiled matrix multiply function
                        @jax.jit
                        def matrix_multiply(vs):
                            return jnp.matmul(vs, vs.T)

                        # Convert to JAX
                        vectors_jax = ml_backend.to_jax(vectors)

                        # Warm-up JIT
                        _ = matrix_multiply(vectors_jax)

                        # Benchmark
                        with timer(f"JAX matrix multiply ({batch_size}x{batch_size})") as t:
                            _ = matrix_multiply(vectors_jax)

                    elif backend == "torch":
                        import torch

                        # Convert to PyTorch on CPU
                        vectors_torch = ml_backend.to_torch(vectors)

                        # Benchmark
                        with timer(f"PyTorch CPU matrix multiply ({batch_size}x{batch_size})") as t:
                            _ = torch.matmul(vectors_torch, vectors_torch.T)

                    elif backend == "torch_gpu":
                        import torch

                        # Convert to PyTorch on GPU
                        device = torch.device("cuda")
                        vectors_gpu = torch.tensor(ml_backend.to_numpy(vectors), device=device)

                        # Benchmark
                        with timer(f"PyTorch GPU matrix multiply ({batch_size}x{batch_size})") as t:
                            _ = torch.matmul(vectors_gpu, vectors_gpu.T)

                # Record result
                results[result_key] = BenchmarkResult(
                    name=op_name,
                    operation=op_name,
                    backend=backend,
                    execution_time=t,
                    vector_count=batch_size,
                    vector_dimension=vectors.shape[1],
                    operations_per_second=batch_size / t,
                    additional_metrics={"batch_size": batch_size},
                )

                # Add extra info for GPU benchmarks
                if backend == "torch_gpu":
                    import torch

                    results[result_key].additional_metrics.update(
                        {
                            "device": torch.cuda.get_device_name(0),
                            "cuda_version": torch.version.cuda,
                        }
                    )

            except Exception as e:
                logger.error(f"Error in {result_key} benchmark: {e}")
                results[result_key] = BenchmarkResult(
                    name=op_name,
                    operation=op_name,
                    backend=backend,
                    execution_time=0,
                    vector_count=batch_size,
                    vector_dimension=vectors.shape[1],
                    error=str(e),
                )

    return results


def benchmark_vector_store(
    vectors: np.ndarray, query_count: int = 10, top_k: int = 100
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark vector store operations.

    Args:
        vectors: Vectors to store and search
        query_count: Number of query vectors to use
        top_k: Number of nearest neighbors to retrieve

    Returns:
        Dictionary of benchmark results for different operations
    """
    results = {}

    # Create vector store
    try:
        # Create vector store in memory
        vector_store = VectorStore("benchmark_store", backend=ml_backend.selected_backend)

        # Benchmark vector insertion
        with timer(f"Vector insertion ({len(vectors)} vectors)") as t:
            for i, vector in enumerate(vectors):
                vector_id = f"vec_{i}"
                vector_store.add_vector(vector, vector_id, {"index": i})

        results["insertion"] = BenchmarkResult(
            name="vector_insertion",
            operation="insertion",
            backend=ml_backend.selected_backend,
            execution_time=t,
            vector_count=len(vectors),
            vector_dimension=vectors.shape[1],
            operations_per_second=len(vectors) / t,
            additional_metrics={"batch_insertion": False},
        )

        # Benchmark batch search
        query_vectors = vectors[:query_count]

        with timer(f"Vector search ({query_count} queries, top_{top_k})") as t:
            for query in query_vectors:
                results_ids = vector_store.search_similar(query, top_k=top_k)

        results["search"] = BenchmarkResult(
            name="vector_search",
            operation="search",
            backend=ml_backend.selected_backend,
            execution_time=t,
            vector_count=query_count,
            vector_dimension=vectors.shape[1],
            operations_per_second=query_count / t,
            additional_metrics={"top_k": top_k},
        )

        # Benchmark batch update
        update_count = min(100, len(vectors))
        update_vectors = generate_random_vectors(update_count, vectors.shape[1])

        with timer(f"Vector update ({update_count} vectors)") as t:
            for i in range(update_count):
                vector_id = f"vec_{i}"
                vector_store.update_vector(vector_id, update_vectors[i])

        results["update"] = BenchmarkResult(
            name="vector_update",
            operation="update",
            backend=ml_backend.selected_backend,
            execution_time=t,
            vector_count=update_count,
            vector_dimension=vectors.shape[1],
            operations_per_second=update_count / t,
        )

        # Benchmark delete
        delete_count = min(100, len(vectors))

        with timer(f"Vector deletion ({delete_count} vectors)") as t:
            for i in range(delete_count):
                vector_id = f"vec_{i}"
                vector_store.delete_vector(vector_id)

        results["deletion"] = BenchmarkResult(
            name="vector_deletion",
            operation="deletion",
            backend=ml_backend.selected_backend,
            execution_time=t,
            vector_count=delete_count,
            vector_dimension=vectors.shape[1],
            operations_per_second=delete_count / t,
        )

    except Exception as e:
        logger.error(f"Error in vector store benchmark: {e}")
        results["error"] = BenchmarkResult(
            name="vector_store",
            operation="all",
            backend=ml_backend.selected_backend if ml_backend else "unknown",
            execution_time=0,
            error=str(e),
        )

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
            from jax import random

            random.PRNGKey(0)
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


class BenchmarkRunner:
    """Runner for executing and aggregating benchmark tests."""

    def __init__(self, vector_count: int = 10000, dimension: int = VECTOR_DIMENSION):
        """
        Initialize the benchmark runner.

        Args:
            vector_count: Number of vectors to generate for benchmarks
            dimension: Dimension of vectors to generate
        """
        self.vector_count = vector_count
        self.dimension = dimension
        self.vectors = generate_random_vectors(vector_count, dimension)
        self.results: Dict[str, Dict[str, BenchmarkResult]] = {}

        # Initialize ML backend
        self.ml_backend = ml_backend
        self.backend_type = ml_backend.selected_backend if ml_backend else "unknown"

        # Log system info
        self._log_system_info()

    def _log_system_info(self) -> None:
        """Log system information for benchmark context."""
        logger.info("--- Benchmark System Information ---")
        logger.info(f"ML Backend: {self.backend_type}")
        logger.info(f"JAX Available: {JAX_AVAILABLE}")
        logger.info(f"PyTorch Available: {TORCH_AVAILABLE}")

        # Log backend-specific information
        if JAX_AVAILABLE:
            try:
                import jax

                devices = jax.devices()
                logger.info(f"JAX Devices: {[str(d) for d in devices]}")
            except ImportError:
                pass

        if TORCH_AVAILABLE:
            try:
                import torch

                logger.info(f"PyTorch version: {torch.__version__}")
                logger.info(f"CUDA Available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    logger.info(f"CUDA Devices: {torch.cuda.device_count()}")
                    for i in range(torch.cuda.device_count()):
                        logger.info(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
                    logger.info(f"CUDA Version: {torch.version.cuda}")

                # Check for MPS (Apple Silicon)
                has_mps = hasattr(torch, "mps") and torch.backends.mps.is_available()
                logger.info(f"MPS Available: {has_mps}")
            except ImportError:
                pass

    def run_benchmark(
        self, benchmark_type: Union[BenchmarkType, str], **kwargs
    ) -> Dict[str, BenchmarkResult]:
        """
        Run a specific benchmark.

        Args:
            benchmark_type: Type of benchmark to run
            **kwargs: Additional arguments to pass to the benchmark function

        Returns:
            Dictionary of benchmark results
        """
        if isinstance(benchmark_type, str):
            try:
                benchmark_type = BenchmarkType(benchmark_type)
            except ValueError:
                logger.error(f"Unknown benchmark type: {benchmark_type}")
                return {}

        logger.info(f"Running benchmark: {benchmark_type.value}")

        if benchmark_type == BenchmarkType.BACKEND_CONVERSION:
            # Default to 100 iterations for conversion benchmark
            iterations = kwargs.get("iterations", 100)
            results = benchmark_backend_conversion(self.vectors, iterations=iterations)

        elif benchmark_type == BenchmarkType.VECTOR_OPERATIONS:
            # Default to 100 operations for vector operations
            batch_size = kwargs.get("batch_size", 100)
            results = benchmark_cosine_similarity(self.vectors, batch_size=batch_size)

        elif benchmark_type == BenchmarkType.BATCH_OPERATIONS:
            # Default to 1000 vectors for batch operations
            batch_size = kwargs.get("batch_size", 1000)
            batch_size = min(batch_size, self.vector_count)
            results = benchmark_batch_operations(self.vectors, batch_size=batch_size)

        elif benchmark_type == BenchmarkType.VECTOR_STORE:
            # Vector store benchmark
            query_count = kwargs.get("query_count", 10)
            top_k = kwargs.get("top_k", 100)
            results = benchmark_vector_store(self.vectors, query_count=query_count, top_k=top_k)

        else:
            logger.error(f"Benchmark type not implemented: {benchmark_type}")
            results = {}

        # Store results
        self.results[benchmark_type.value] = results

        return results

    def run_all_benchmarks(self) -> Dict[str, Dict[str, BenchmarkResult]]:
        """
        Run all available benchmarks.

        Returns:
            Dictionary of all benchmark results
        """
        logger.info(
            f"Running all benchmarks with {self.vector_count} vectors of dimension {self.dimension}"
        )

        # Run each benchmark type
        benchmarks = [
            (BenchmarkType.BACKEND_CONVERSION, {}),
            (BenchmarkType.VECTOR_OPERATIONS, {"batch_size": 100}),
            (BenchmarkType.BATCH_OPERATIONS, {"batch_size": min(1000, self.vector_count)}),
            (BenchmarkType.VECTOR_STORE, {"query_count": 10, "top_k": 100}),
        ]

        for benchmark_type, kwargs in benchmarks:
            try:
                self.run_benchmark(benchmark_type, **kwargs)
            except Exception as e:
                logger.error(f"Error running benchmark {benchmark_type.value}: {e}")

        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of benchmark results.

        Returns:
            Dictionary with benchmark summary
        """
        if not self.results:
            return {"error": "No benchmark results available"}

        summary = {
            "system_info": {
                "ml_backend": self.backend_type,
                "jax_available": JAX_AVAILABLE,
                "torch_available": TORCH_AVAILABLE,
                "vector_count": self.vector_count,
                "vector_dimension": self.dimension,
            },
            "benchmarks": {},
        }

        # Add JAX device info if available
        if JAX_AVAILABLE:
            try:
                import jax

                summary["system_info"]["jax_devices"] = [str(d) for d in jax.devices()]
            except ImportError:
                pass

        # Add PyTorch GPU info if available
        if TORCH_AVAILABLE:
            try:
                import torch

                summary["system_info"]["torch_cuda_available"] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    summary["system_info"]["torch_cuda_devices"] = torch.cuda.device_count()
                    summary["system_info"]["torch_cuda_version"] = torch.version.cuda

                # Add MPS (Apple Silicon) info
                summary["system_info"]["torch_mps_available"] = (
                    hasattr(torch, "mps") and torch.backends.mps.is_available()
                )
            except ImportError:
                pass

        # Summarize each benchmark type
        for benchmark_type, results in self.results.items():
            benchmark_summary = {}

            for result_name, result in results.items():
                # Skip results with errors
                if result.error:
                    continue

                benchmark_summary[result_name] = {
                    "execution_time": result.execution_time,
                    "operations_per_second": result.operations_per_second,
                }

                # Add speedup compared to NumPy for non-NumPy backends
                if "numpy" not in result_name.lower() and f"numpy_{result.operation}" in results:
                    numpy_result = results[f"numpy_{result.operation}"]
                    if numpy_result.execution_time > 0:
                        speedup = numpy_result.execution_time / result.execution_time
                        benchmark_summary[result_name]["speedup_vs_numpy"] = speedup

            summary["benchmarks"][benchmark_type] = benchmark_summary

        return summary


def run_all_benchmarks(
    vector_count: int = 10000, dimension: int = VECTOR_DIMENSION
) -> Dict[str, Any]:
    """
    Run all benchmarks and return results.

    Args:
        vector_count: Number of vectors to use
        dimension: Dimension of vectors

    Returns:
        Dictionary of benchmark results
    """
    runner = BenchmarkRunner(vector_count=vector_count, dimension=dimension)
    runner.run_all_benchmarks()
    return runner.get_summary()


if __name__ == "__main__":
    # Run benchmarks if script is executed directly
    import json

    # Generate a smaller set of vectors for quicker testing
    results = run_all_benchmarks(vector_count=5000, dimension=128)

    # Print summary
    print(json.dumps(results, indent=2))

    # Print performance comparison
    if "benchmarks" in results and "batch_operations" in results["benchmarks"]:
        batch_ops = results["benchmarks"]["batch_operations"]
        print("\nPerformance Comparison (operations per second):")
        for op in ["normalize", "batch_similarity", "matrix_multiply"]:
            print(f"\n{op.upper()}:")
            for backend in ["numpy", "jax", "torch", "torch_gpu"]:
                key = f"{backend}_{op}"
                if key in batch_ops:
                    print(f"  {backend}: {batch_ops[key]['operations_per_second']:.2f} ops/sec")
                    if "speedup_vs_numpy" in batch_ops[key]:
                        print(f"    {batch_ops[key]['speedup_vs_numpy']:.2f}x faster than NumPy")
