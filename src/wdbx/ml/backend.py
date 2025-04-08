"""
Machine learning backend selection and management.

This module provides a unified interface for different ML backends
(NumPy, PyTorch, JAX, FAISS) for vector operations.
"""

from __future__ import annotations

import importlib.util
import os
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

# Use relative imports for core modules
from ..core.constants import DEFAULT_ML_BACKEND, VECTOR_DIMENSION, logger
from ..utils.memory_utils import register_for_memory_optimization

# Type alias for array-like objects
from . import JAX_AVAILABLE, TORCH_AVAILABLE, ArrayLike

# Type variable for generic vector operations
T = TypeVar("T", bound=ArrayLike)


class BackendType(Enum):
    """Enum for supported ML backends."""

    NUMPY = "numpy"
    PYTORCH = "pytorch"
    JAX = "jax"
    FAISS = "faiss"
    AUTO = "auto"

    @classmethod
    def from_string(cls, value: str) -> BackendType:
        """
        Convert string to BackendType.

        Args:
            value: String representation of backend type

        Returns:
            BackendType enum value
        """
        try:
            # Handle variations in naming
            value = value.lower()
            if value == "torch":
                value = "pytorch"
            return cls(value)
        except ValueError:
            logger.warning(f"Unknown backend type: {value}, using AUTO")
            return cls.AUTO


class MLBackend:
    """
    Machine learning backend manager.

    Provides a unified interface for different ML backends (NumPy, PyTorch, JAX, FAISS)
    for vector operations, automatically selecting the most efficient backend based on
    availability and performance characteristics.
    """

    def __init__(
        self, vector_dimension: int = VECTOR_DIMENSION, preferred_backend: Optional[str] = None
    ):
        """
        Initialize MLBackend.

        Args:
            vector_dimension: Dimension of vectors to be processed
            preferred_backend: Preferred backend type (numpy, pytorch, jax, faiss, or auto)
        """
        self.vector_dimension = vector_dimension
        self.selected_backend: str = self._select_backend(preferred_backend)
        self.backend_functions: Dict[str, Callable] = self._initialize_backend()

        # Register for memory optimization
        register_for_memory_optimization("ml_backend", self)

        logger.info(f"Initialized ML backend using {self.selected_backend}")

    def _select_backend(self, preferred_backend: Optional[str] = None) -> str:
        """
        Select the best backend to use.

        Args:
            preferred_backend: Preferred backend type

        Returns:
            Selected backend name
        """
        # Get backend from environment or parameter
        backend = preferred_backend or os.getenv("WDBX_ML_BACKEND", DEFAULT_ML_BACKEND)

        # Convert to enum for validation
        backend_type = BackendType.from_string(backend)

        # Auto-select if needed
        if backend_type == BackendType.AUTO:
            selected = self._auto_select_backend()
            logger.info(f"Auto-selected {selected} ML backend")
            return selected

        # Check if preferred backend is available
        available_backends = self._get_available_backends()

        if backend_type.value in available_backends:
            return backend_type.value

        # Fallback to numpy if preferred backend is not available
        logger.warning(
            f"Preferred backend {backend_type.value} not available. "
            f"Available backends: {available_backends}. "
            f"Falling back to numpy."
        )
        return "numpy"

    def _auto_select_backend(self) -> str:
        """
        Auto-select the best backend based on availability and performance.

        Returns:
            Selected backend name
        """
        available_backends = self._get_available_backends()

        # Prefer backends in this order: faiss, jax, pytorch, numpy
        for backend in ["faiss", "jax", "pytorch", "numpy"]:
            if backend in available_backends:
                return backend

        # Fallback to numpy (should always be available)
        return "numpy"

    def _get_available_backends(self) -> List[str]:
        """
        Check which backends are available in the environment.

        Returns:
            List of available backend names
        """
        available = []

        # NumPy should always be available (we import it at the module level)
        available.append("numpy")

        # Check for PyTorch
        if TORCH_AVAILABLE:
            available.append("pytorch")

        # Check for JAX
        if JAX_AVAILABLE:
            available.append("jax")

        # Check for FAISS
        if importlib.util.find_spec("faiss") is not None:
            try:
                import faiss

                available.append("faiss")
            except ImportError:
                logger.debug("FAISS is installed but cannot be imported")

        logger.debug(f"Available ML backends: {available}")
        return available

    def _initialize_backend(self) -> Dict[str, Callable]:
        """
        Initialize the selected backend.

        Returns:
            Dictionary of backend functions
        """
        if self.selected_backend == "numpy":
            return self._initialize_numpy_backend()
        elif self.selected_backend == "pytorch":
            return self._initialize_torch_backend()
        elif self.selected_backend == "jax":
            return self._initialize_jax_backend()
        elif self.selected_backend == "faiss":
            return self._initialize_faiss_backend()
        else:
            raise ValueError(f"Unsupported backend: {self.selected_backend}")

    def _initialize_numpy_backend(self) -> Dict[str, Callable]:
        """
        Initialize NumPy backend functions.

        Returns:
            Dictionary of backend functions
        """

        # Cosine similarity for single vector pair
        def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
            """
            Calculate cosine similarity between two vectors.

            Args:
                a: First vector
                b: Second vector

            Returns:
                Cosine similarity score (float)
            """
            # Ensure vectors are flattened
            a = a.reshape(-1)
            b = b.reshape(-1)

            # Calculate cosine similarity using dot product and norms
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            # Handle zero norms to avoid division by zero
            if norm_a == 0 or norm_b == 0:
                return 0.0

            return float(dot_product / (norm_a * norm_b))

        # Cosine similarity for batch processing
        def batch_cosine_similarity(
            query: NDArray[np.float32], vectors: NDArray[np.float32]
        ) -> NDArray[np.float32]:
            """
            Calculate cosine similarity between query vector and multiple vectors.

            Args:
                query: Query vector
                vectors: Matrix of vectors to compare against

            Returns:
                Array of similarity scores
            """
            # Ensure query is flattened
            query = query.reshape(1, -1)

            # Ensure vectors is 2D array
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)

            # Calculate dot products
            dot_products = np.dot(query, vectors.T).flatten()

            # Calculate norms
            query_norm = np.linalg.norm(query)
            vector_norms = np.linalg.norm(vectors, axis=1)

            # Calculate similarities
            similarities = dot_products / (query_norm * vector_norms)

            # Handle potential NaN values due to zero norms
            similarities = np.nan_to_num(similarities, nan=0.0)

            return similarities

        # Vector normalization
        def normalize(vector: NDArray[np.float32]) -> NDArray[np.float32]:
            """
            Normalize a vector to unit length.

            Args:
                vector: Input vector

            Returns:
                Normalized vector
            """
            # Flatten vector for consistent processing
            vector = vector.reshape(-1)

            # Calculate norm
            norm = np.linalg.norm(vector)

            # Handle zero norm
            if norm == 0:
                return vector

            return vector / norm

        # Convert to NumPy array
        def to_numpy(vector: ArrayLike) -> NDArray[np.float32]:
            """
            Convert vector to NumPy array.

            Args:
                vector: Input vector (any supported array-like type)

            Returns:
                NumPy array
            """
            if vector is None:
                return None

            if isinstance(vector, np.ndarray):
                return vector.astype(np.float32)

            # For other types, try to convert
            return np.array(vector, dtype=np.float32)

        return {
            "cosine_similarity": cosine_similarity,
            "batch_cosine_similarity": batch_cosine_similarity,
            "normalize": normalize,
            "to_numpy": to_numpy,
        }

    def _initialize_torch_backend(self) -> Dict[str, Callable]:
        """
        Initialize PyTorch backend functions.

        Returns:
            Dictionary of backend functions
        """
        import torch

        # Cosine similarity for single vector pair
        def cosine_similarity(
            a: Union[torch.Tensor, NDArray[np.float32]], b: Union[torch.Tensor, NDArray[np.float32]]
        ) -> float:
            """
            Calculate cosine similarity between two vectors using PyTorch.

            Args:
                a: First vector
                b: Second vector

            Returns:
                Cosine similarity score (float)
            """
            # Convert inputs to PyTorch tensors if they aren't already
            a_tensor = a if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.float32)
            b_tensor = b if isinstance(b, torch.Tensor) else torch.tensor(b, dtype=torch.float32)

            # Flatten tensors
            a_tensor = a_tensor.flatten()
            b_tensor = b_tensor.flatten()

            # Use PyTorch's built-in cosine similarity function
            cos_sim = torch.nn.functional.cosine_similarity(
                a_tensor.unsqueeze(0), b_tensor.unsqueeze(0)
            )

            return float(cos_sim.item())

        # Cosine similarity for batch processing
        def batch_cosine_similarity(
            query: Union[torch.Tensor, NDArray[np.float32]],
            vectors: Union[torch.Tensor, NDArray[np.float32]],
        ) -> NDArray[np.float32]:
            """
            Calculate cosine similarity between query vector and multiple vectors using PyTorch.

            Args:
                query: Query vector
                vectors: Matrix of vectors to compare against

            Returns:
                Array of similarity scores
            """
            # Convert inputs to PyTorch tensors if they aren't already
            query_tensor = (
                query
                if isinstance(query, torch.Tensor)
                else torch.tensor(query, dtype=torch.float32)
            )
            vectors_tensor = (
                vectors
                if isinstance(vectors, torch.Tensor)
                else torch.tensor(vectors, dtype=torch.float32)
            )

            # Ensure dimensions are correct
            query_tensor = query_tensor.flatten().unsqueeze(0)
            if vectors_tensor.dim() == 1:
                vectors_tensor = vectors_tensor.unsqueeze(0)

            # Use PyTorch's built-in cosine similarity function
            cos_sim = torch.nn.functional.cosine_similarity(
                query_tensor.unsqueeze(1), vectors_tensor.unsqueeze(0), dim=2
            )

            # Convert to NumPy for consistent return type
            return cos_sim.squeeze().detach().cpu().numpy()

        # Vector normalization
        def normalize(vector: Union[torch.Tensor, NDArray[np.float32]]) -> NDArray[np.float32]:
            """
            Normalize a vector to unit length using PyTorch.

            Args:
                vector: Input vector

            Returns:
                Normalized vector as NumPy array
            """
            # Convert input to PyTorch tensor if it isn't already
            vector_tensor = (
                vector
                if isinstance(vector, torch.Tensor)
                else torch.tensor(vector, dtype=torch.float32)
            )

            # Flatten tensor
            vector_tensor = vector_tensor.flatten()

            # Normalize using PyTorch's built-in function
            normalized = torch.nn.functional.normalize(vector_tensor.unsqueeze(0), p=2, dim=1)

            # Convert back to NumPy
            return normalized.squeeze().detach().cpu().numpy()

        # Convert to NumPy array
        def to_numpy(vector: ArrayLike) -> NDArray[np.float32]:
            """
            Convert vector to NumPy array.

            Args:
                vector: Input vector (any supported array-like type)

            Returns:
                NumPy array
            """
            if vector is None:
                return None

            if isinstance(vector, np.ndarray):
                return vector.astype(np.float32)

            if isinstance(vector, torch.Tensor):
                return vector.detach().cpu().numpy().astype(np.float32)

            # For other types, try to convert
            return np.array(vector, dtype=np.float32)

        # Convert to PyTorch tensor
        def to_torch(vector: ArrayLike) -> torch.Tensor:
            """
            Convert vector to PyTorch tensor.

            Args:
                vector: Input vector (any supported array-like type)

            Returns:
                PyTorch tensor
            """
            if vector is None:
                return None

            if isinstance(vector, torch.Tensor):
                return vector

            # For NumPy arrays and other types
            return torch.tensor(vector, dtype=torch.float32)

        return {
            "cosine_similarity": cosine_similarity,
            "batch_cosine_similarity": batch_cosine_similarity,
            "normalize": normalize,
            "to_numpy": to_numpy,
            "to_torch": to_torch,
        }

    def _initialize_jax_backend(self) -> Dict[str, Callable]:
        """
        Initialize JAX backend functions.

        Returns:
            Dictionary of backend functions
        """
        import jax
        import jax.numpy as jnp

        # JIT-compiled version of cosine similarity
        @jax.jit
        def _jax_cosine_similarity(a, b):
            """JIT-compiled cosine similarity between two vectors."""
            a = a.reshape(-1)
            b = b.reshape(-1)
            return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-8)

        # JIT-compiled version of batch cosine similarity
        @jax.jit
        def _jax_batch_cosine_similarity(query, vectors):
            """JIT-compiled cosine similarity between a query and multiple vectors."""
            query = query.reshape(1, -1)
            query_norm = jnp.linalg.norm(query)
            vector_norms = jnp.linalg.norm(vectors, axis=1)
            dot_products = jnp.dot(query, vectors.T).reshape(-1)
            return dot_products / (query_norm * vector_norms + 1e-8)

        # JIT-compiled version of normalization
        @jax.jit
        def _jax_normalize(vector):
            """JIT-compiled vector normalization."""
            vector = vector.reshape(-1)
            return vector / (jnp.linalg.norm(vector) + 1e-8)

        # Function wrappers to handle different input types
        def cosine_similarity(a: Any, b: Any) -> float:
            """
            Calculate cosine similarity between two vectors using JAX.

            Args:
                a: First vector
                b: Second vector

            Returns:
                Cosine similarity score (float)
            """
            a_jax = a if isinstance(a, jnp.ndarray) else jnp.array(a, dtype=jnp.float32)
            b_jax = b if isinstance(b, jnp.ndarray) else jnp.array(b, dtype=jnp.float32)

            result = _jax_cosine_similarity(a_jax, b_jax)
            return float(np.array(result))

        def batch_cosine_similarity(query: Any, vectors: Any) -> NDArray[np.float32]:
            """
            Calculate cosine similarity between query vector and multiple vectors using JAX.

            Args:
                query: Query vector
                vectors: Matrix of vectors to compare against

            Returns:
                Array of similarity scores
            """
            query_jax = (
                query if isinstance(query, jnp.ndarray) else jnp.array(query, dtype=jnp.float32)
            )
            vectors_jax = (
                vectors
                if isinstance(vectors, jnp.ndarray)
                else jnp.array(vectors, dtype=jnp.float32)
            )

            if vectors_jax.ndim == 1:
                vectors_jax = vectors_jax.reshape(1, -1)

            result = _jax_batch_cosine_similarity(query_jax, vectors_jax)
            return np.array(result)

        def normalize(vector: Any) -> NDArray[np.float32]:
            """
            Normalize a vector to unit length using JAX.

            Args:
                vector: Input vector

            Returns:
                Normalized vector as NumPy array
            """
            vector_jax = (
                vector if isinstance(vector, jnp.ndarray) else jnp.array(vector, dtype=jnp.float32)
            )
            result = _jax_normalize(vector_jax)
            return np.array(result)

        def to_numpy(vector: ArrayLike) -> NDArray[np.float32]:
            """
            Convert vector to NumPy array.

            Args:
                vector: Input vector (any supported array-like type)

            Returns:
                NumPy array
            """
            if vector is None:
                return None

            if isinstance(vector, np.ndarray):
                return vector.astype(np.float32)

            if isinstance(vector, jnp.ndarray):
                return np.array(vector, dtype=np.float32)

            # For other types, try to convert
            return np.array(vector, dtype=np.float32)

        def to_jax(vector: ArrayLike) -> Any:  # jax.Array
            """
            Convert vector to JAX array.

            Args:
                vector: Input vector (any supported array-like type)

            Returns:
                JAX array
            """
            if vector is None:
                return None

            if isinstance(vector, jnp.ndarray):
                return vector

            # For NumPy arrays and other types
            return jnp.array(vector, dtype=jnp.float32)

        return {
            "cosine_similarity": cosine_similarity,
            "batch_cosine_similarity": batch_cosine_similarity,
            "normalize": normalize,
            "to_numpy": to_numpy,
            "to_jax": to_jax,
        }

    def _initialize_faiss_backend(self) -> Dict[str, Callable]:
        """
        Initialize FAISS backend functions.

        Returns:
            Dictionary of backend functions
        """
        import faiss

        # Initialize an empty index
        dim = self.vector_dimension
        index = faiss.IndexFlatIP(dim)  # Inner product index (for cosine, we'll normalize vectors)

        def search_vectors(
            query: NDArray[np.float32], vectors: NDArray[np.float32], top_k: int
        ) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
            """
            Search for similar vectors using FAISS.

            Args:
                query: Query vector
                vectors: Matrix of vectors to search in
                top_k: Number of results to return

            Returns:
                Tuple of (distances, indices)
            """
            # Ensure inputs are NumPy arrays
            query_np = (
                np.array(query, dtype=np.float32)
                if not isinstance(query, np.ndarray)
                else query.astype(np.float32)
            )
            vectors_np = (
                np.array(vectors, dtype=np.float32)
                if not isinstance(vectors, np.ndarray)
                else vectors.astype(np.float32)
            )

            # Reshape query if needed
            if query_np.ndim == 1:
                query_np = query_np.reshape(1, -1)

            # Normalize vectors for cosine similarity
            faiss.normalize_L2(query_np)
            if vectors_np.ndim == 1:
                vectors_np = vectors_np.reshape(1, -1)
            faiss.normalize_L2(vectors_np)

            # Use temporary index for this search
            temp_index = faiss.IndexFlatIP(dim)
            temp_index.add(vectors_np)

            # Perform search
            distances, indices = temp_index.search(query_np, min(top_k, len(vectors_np)))

            return distances, indices

        # Reuse NumPy backend for basic operations
        numpy_backend = self._initialize_numpy_backend()

        return {
            **numpy_backend,  # Include all NumPy functions
            "search_vectors": search_vectors,
        }

    def cosine_similarity(self, a: ArrayLike, b: ArrayLike) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity score (float)
        """
        # Convert inputs to NumPy for consistent processing
        a_np = self.to_numpy(a)
        b_np = self.to_numpy(b)

        return self.backend_functions["cosine_similarity"](a_np, b_np)

    def batch_cosine_similarity(self, query: ArrayLike, vectors: ArrayLike) -> NDArray[np.float32]:
        """
        Calculate cosine similarity between query vector and multiple vectors.

        Args:
            query: Query vector
            vectors: Matrix of vectors to compare against

        Returns:
            Array of similarity scores
        """
        # Convert inputs to NumPy for consistent processing
        query_np = self.to_numpy(query)
        vectors_np = self.to_numpy(vectors)

        return self.backend_functions["batch_cosine_similarity"](query_np, vectors_np)

    def normalize(self, vector: ArrayLike) -> NDArray[np.float32]:
        """
        Normalize a vector to unit length.

        Args:
            vector: Input vector

        Returns:
            Normalized vector as NumPy array
        """
        # Convert input to NumPy for consistent processing
        vector_np = self.to_numpy(vector)

        return self.backend_functions["normalize"](vector_np)

    def to_numpy(self, vector: ArrayLike) -> NDArray[np.float32]:
        """
        Convert vector to NumPy array.

        Args:
            vector: Input vector (any supported array-like type)

        Returns:
            NumPy array
        """
        return self.backend_functions["to_numpy"](vector)

    def search_vectors(
        self, query: ArrayLike, vectors: ArrayLike, top_k: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        """
        Search for similar vectors.

        Args:
            query: Query vector
            vectors: Matrix of vectors to search in
            top_k: Number of results to return

        Returns:
            Tuple of (distances, indices)
        """
        # Convert inputs to NumPy for consistent processing
        query_np = self.to_numpy(query)
        vectors_np = self.to_numpy(vectors)

        # Use FAISS search if available
        if "search_vectors" in self.backend_functions:
            return self.backend_functions["search_vectors"](query_np, vectors_np, top_k)

        # Fallback implementation using batch_cosine_similarity
        similarities = self.batch_cosine_similarity(query_np, vectors_np)

        # Get top-k indices
        if len(similarities) <= top_k:
            indices = np.arange(len(similarities))
            distances = similarities
        else:
            # Get indices of top-k similarities
            indices = np.argsort(similarities)[-top_k:][::-1]
            distances = similarities[indices]

        return np.array(distances, dtype=np.float32), np.array(indices, dtype=np.int64)

    def optimize_memory(self) -> None:
        """
        Optimize memory usage by clearing caches and releasing temporary resources.
        """
        # Specific memory optimization for each backend
        if self.selected_backend == "pytorch":
            try:
                import torch

                # Clear CUDA cache if using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("Cleared PyTorch CUDA cache")

                # Clear other caches
                if hasattr(torch, "__cache__"):
                    torch.__cache__.clear()  # type: ignore
                    logger.debug("Cleared PyTorch function cache")
            except Exception as e:
                logger.warning(f"Error during PyTorch memory optimization: {e}")

        elif self.selected_backend == "jax":
            try:
                import jax

                # Clear JIT compilation cache
                jax.clear_caches()
                logger.debug("Cleared JAX caches")
            except Exception as e:
                logger.warning(f"Error during JAX memory optimization: {e}")

        # Trigger Python garbage collection
        gc_count = gc.collect()
        logger.debug(f"Garbage collected {gc_count} objects")


@lru_cache(maxsize=8)
def get_ml_backend(
    preferred_backend: Optional[str] = None, vector_dimension: int = VECTOR_DIMENSION
) -> MLBackend:
    """
    Get an MLBackend instance with the specified configuration.
    Uses LRU caching to avoid creating multiple instances with the same configuration.

    Args:
        preferred_backend: Preferred backend type (numpy, pytorch, jax, faiss, or auto)
        vector_dimension: Dimension of vectors to be processed

    Returns:
        MLBackend instance
    """
    return MLBackend(vector_dimension=vector_dimension, preferred_backend=preferred_backend)
