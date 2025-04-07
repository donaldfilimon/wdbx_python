"""
Machine learning backend selection and management.

This module provides a unified interface for different ML backends
(NumPy, PyTorch, JAX, FAISS) for vector operations.
"""

import importlib.util
import logging
import os
import time
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from ..core.constants import (DEFAULT_ML_BACKEND, ML_BACKENDS, VECTOR_DIMENSION,
                             logger)
from ..utils.memory_utils import register_for_memory_optimization

# Type alias for array-like objects (could be numpy arrays, PyTorch tensors, JAX arrays)
ArrayLike = Any


class BackendType(Enum):
    """Enum for supported ML backends."""
    
    NUMPY = "numpy"
    PYTORCH = "pytorch"
    JAX = "jax"
    FAISS = "faiss"
    AUTO = "auto"
    
    @classmethod
    def from_string(cls, value: str) -> 'BackendType':
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
    
    def __init__(self, 
                vector_dimension: int = VECTOR_DIMENSION,
                preferred_backend: Optional[str] = None):
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
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch
                available.append("pytorch")
            except ImportError:
                logger.debug("PyTorch is installed but cannot be imported")
        
        # Check for JAX
        if importlib.util.find_spec("jax") is not None:
            try:
                import jax
                available.append("jax")
            except ImportError:
                logger.debug("JAX is installed but cannot be imported")
        
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
            Dictionary of NumPy backend functions
        """
        def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
            """
            Calculate cosine similarity between two vectors.
            
            Args:
                a: First vector
                b: Second vector
                
            Returns:
                Cosine similarity value between -1 and 1
            """
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            if a_norm == 0 or b_norm == 0:
                return 0.0
            return float(np.dot(a, b) / (a_norm * b_norm))
        
        def batch_cosine_similarity(query: NDArray[np.float32], 
                              vectors: NDArray[np.float32]) -> NDArray[np.float32]:
            """
            Calculate cosine similarity between a query vector and a batch of vectors.
            
            Args:
                query: Query vector of shape (d,)
                vectors: Batch of vectors of shape (n, d)
                
            Returns:
                Array of similarity scores of shape (n,)
            """
            # Normalize query vector
            query_norm = np.linalg.norm(query)
            if query_norm == 0:
                return np.zeros(vectors.shape[0], dtype=np.float32)
            query_normalized = query / query_norm
            
            # Normalize all vectors in the batch
            vectors_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors_norm = np.maximum(vectors_norm, 1e-10)  # Avoid division by zero
            vectors_normalized = vectors / vectors_norm
            
            # Compute similarities
            similarities = np.dot(vectors_normalized, query_normalized)
            return similarities
        
        def normalize(vector: NDArray[np.float32]) -> NDArray[np.float32]:
            """
            Normalize a vector to unit length.
            
            Args:
                vector: Input vector
                
            Returns:
                Normalized vector
            """
            norm = np.linalg.norm(vector)
            if norm < 1e-10:
                return np.zeros_like(vector)
            return vector / norm
        
        def to_numpy(vector: ArrayLike) -> NDArray[np.float32]:
            """
            Convert vector to NumPy array.
            
            Args:
                vector: Input vector (any array-like)
                
            Returns:
                NumPy array representation
            """
            if isinstance(vector, np.ndarray):
                return vector.astype(np.float32)
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
            Dictionary of PyTorch backend functions
        """
        import torch
        
        def cosine_similarity(a: Union[torch.Tensor, NDArray[np.float32]], 
                        b: Union[torch.Tensor, NDArray[np.float32]]) -> float:
            """
            Calculate cosine similarity between two vectors using PyTorch.
            
            Args:
                a: First vector
                b: Second vector
                
            Returns:
                Cosine similarity value between -1 and 1
            """
            # Convert to torch tensors if needed
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, dtype=torch.float32)
            if not isinstance(b, torch.Tensor):
                b = torch.tensor(b, dtype=torch.float32)
            
            return float(torch.nn.functional.cosine_similarity(a.view(1, -1), b.view(1, -1)).item())
        
        def batch_cosine_similarity(query: Union[torch.Tensor, NDArray[np.float32]],
                              vectors: Union[torch.Tensor, NDArray[np.float32]]) -> NDArray[np.float32]:
            """
            Calculate cosine similarity between a query vector and a batch of vectors.
            
            Args:
                query: Query vector
                vectors: Batch of vectors
                
            Returns:
                Array of similarity scores
            """
            # Convert to torch tensors if needed
            if not isinstance(query, torch.Tensor):
                query = torch.tensor(query, dtype=torch.float32)
            if not isinstance(vectors, torch.Tensor):
                vectors = torch.tensor(vectors, dtype=torch.float32)
            
            # Normalize query vector
            query_norm = torch.norm(query)
            if query_norm == 0:
                return np.zeros(vectors.shape[0], dtype=np.float32)
            query_normalized = query / query_norm
            
            # Normalize all vectors in the batch
            vectors_norm = torch.norm(vectors, dim=1, keepdim=True)
            vectors_norm = torch.clamp(vectors_norm, min=1e-10)  # Avoid division by zero
            vectors_normalized = vectors / vectors_norm
            
            # Compute similarities
            similarities = torch.matmul(vectors_normalized, query_normalized)
            
            # Convert to numpy for consistent return type
            return similarities.detach().cpu().numpy().astype(np.float32)
        
        def normalize(vector: Union[torch.Tensor, NDArray[np.float32]]) -> NDArray[np.float32]:
            """
            Normalize a vector to unit length.
            
            Args:
                vector: Input vector
                
            Returns:
                Normalized vector as NumPy array
            """
            if not isinstance(vector, torch.Tensor):
                vector = torch.tensor(vector, dtype=torch.float32)
            
            norm = torch.norm(vector)
            if norm < 1e-10:
                return np.zeros(vector.shape, dtype=np.float32)
            
            normalized = vector / norm
            return normalized.detach().cpu().numpy().astype(np.float32)
        
        def to_numpy(vector: ArrayLike) -> NDArray[np.float32]:
            """
            Convert vector to NumPy array.
            
            Args:
                vector: Input vector (torch.Tensor or array-like)
                
            Returns:
                NumPy array representation
            """
            if isinstance(vector, torch.Tensor):
                return vector.detach().cpu().numpy().astype(np.float32)
            return np.array(vector, dtype=np.float32)
        
        def to_torch(vector: ArrayLike) -> torch.Tensor:
            """
            Convert vector to PyTorch tensor.
            
            Args:
                vector: Input vector (any array-like)
                
            Returns:
                PyTorch tensor
            """
            if isinstance(vector, torch.Tensor):
                return vector.float()
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
            Dictionary of JAX backend functions
        """
        import jax
        import jax.numpy as jnp
        
        # JIT-compiled functions for performance
        @jax.jit
        def _jax_cosine_similarity(a, b):
            """JAX implementation of cosine similarity."""
            a_norm = jnp.linalg.norm(a)
            b_norm = jnp.linalg.norm(b)
            # Avoid division by zero
            denominator = jnp.maximum(a_norm * b_norm, 1e-10)
            return jnp.dot(a, b) / denominator
        
        @jax.jit
        def _jax_batch_cosine_similarity(query, vectors):
            """JAX implementation of batch cosine similarity."""
            # Normalize query vector
            query_norm = jnp.linalg.norm(query)
            # Avoid division by zero
            query_norm = jnp.maximum(query_norm, 1e-10)
            query_normalized = query / query_norm
            
            # Normalize all vectors in the batch
            vectors_norm = jnp.linalg.norm(vectors, axis=1, keepdims=True)
            vectors_norm = jnp.maximum(vectors_norm, 1e-10)
            vectors_normalized = vectors / vectors_norm
            
            # Compute similarities
            return jnp.dot(vectors_normalized, query_normalized)
        
        @jax.jit
        def _jax_normalize(vector):
            """JAX implementation of vector normalization."""
            norm = jnp.linalg.norm(vector)
            # Avoid division by zero
            norm = jnp.maximum(norm, 1e-10)
            return vector / norm
        
        # Wrapper functions to handle different input types
        def cosine_similarity(a: Any, b: Any) -> float:
            """
            Calculate cosine similarity between two vectors using JAX.
            
            Args:
                a: First vector (any array-like convertible to JAX array)
                b: Second vector (any array-like convertible to JAX array)
                
            Returns:
                Cosine similarity value between -1 and 1
            """
            a_jax = jnp.array(a, dtype=jnp.float32)
            b_jax = jnp.array(b, dtype=jnp.float32)
            return float(_jax_cosine_similarity(a_jax, b_jax))
        
        def batch_cosine_similarity(query: Any, vectors: Any) -> NDArray[np.float32]:
            """
            Calculate cosine similarity between a query vector and a batch of vectors.
            
            Args:
                query: Query vector (any array-like convertible to JAX array)
                vectors: Batch of vectors (any array-like convertible to JAX array)
                
            Returns:
                Array of similarity scores
            """
            query_jax = jnp.array(query, dtype=jnp.float32)
            vectors_jax = jnp.array(vectors, dtype=jnp.float32)
            similarities = _jax_batch_cosine_similarity(query_jax, vectors_jax)
            return np.array(similarities, dtype=np.float32)
        
        def normalize(vector: Any) -> NDArray[np.float32]:
            """
            Normalize a vector to unit length.
            
            Args:
                vector: Input vector (any array-like convertible to JAX array)
                
            Returns:
                Normalized vector as NumPy array
            """
            vector_jax = jnp.array(vector, dtype=jnp.float32)
            normalized = _jax_normalize(vector_jax)
            return np.array(normalized, dtype=np.float32)
        
        def to_numpy(vector: ArrayLike) -> NDArray[np.float32]:
            """
            Convert vector to NumPy array.
            
            Args:
                vector: Input vector (JAX array or any array-like)
                
            Returns:
                NumPy array representation
            """
            if isinstance(vector, jax.Array):
                return np.array(vector, dtype=np.float32)
            return np.array(vector, dtype=np.float32)
        
        def to_jax(vector: ArrayLike) -> Any:  # jax.Array
            """
            Convert vector to JAX array.
            
            Args:
                vector: Input vector (any array-like)
                
            Returns:
                JAX array
            """
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
            Dictionary of FAISS backend functions
        """
        import faiss
        
        # Initialize FAISS index for search operations
        index = faiss.IndexFlatIP(self.vector_dimension)  # Inner product index (for cosine similarity)
        
        def search_vectors(query: NDArray[np.float32], 
                     vectors: NDArray[np.float32],
                     top_k: int) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
            """
            Search for nearest vectors using FAISS.
            
            Args:
                query: Query vector (must be normalized for cosine similarity)
                vectors: Batch of vectors (must be normalized for cosine similarity)
                top_k: Number of top results to return
                
            Returns:
                Tuple of (similarities, indices)
            """
            # Ensure vectors are in the right format
            query_np = np.array(query, dtype=np.float32).reshape(1, -1)
            vectors_np = np.array(vectors, dtype=np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_np)
            faiss.normalize_L2(vectors_np)
            
            # Create a temporary index if the batch size changed
            if index.ntotal != vectors_np.shape[0]:
                # Reset the index
                index.reset()
                index.add(vectors_np)
            
            # Search for nearest neighbors
            k = min(top_k, vectors_np.shape[0])  # Can't return more than we have
            similarities, indices = index.search(query_np, k)
            
            return similarities[0], indices[0]
        
        # Re-use NumPy functions for basic operations
        numpy_funcs = self._initialize_numpy_backend()
        
        return {
            **numpy_funcs,  # Include all NumPy functions
            "search_vectors": search_vectors,  # Override with FAISS implementation
        }
    
    def cosine_similarity(self, a: ArrayLike, b: ArrayLike) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity value between -1 and 1
        """
        # Convert to numpy arrays first for consistent handling
        a_np = self.to_numpy(a)
        b_np = self.to_numpy(b)
        
        return self.backend_functions["cosine_similarity"](a_np, b_np)
    
    def batch_cosine_similarity(self, query: ArrayLike, vectors: ArrayLike) -> NDArray[np.float32]:
        """
        Calculate cosine similarity between a query vector and a batch of vectors.
        
        Args:
            query: Query vector
            vectors: Batch of vectors
            
        Returns:
            Array of similarity scores
        """
        # Convert to numpy arrays first for consistent handling
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
        # Convert to numpy array first for consistent handling
        vector_np = self.to_numpy(vector)
        
        return self.backend_functions["normalize"](vector_np)
    
    def to_numpy(self, vector: ArrayLike) -> NDArray[np.float32]:
        """
        Convert vector to NumPy array.
        
        Args:
            vector: Input vector (any array-like)
            
        Returns:
            NumPy array representation
        """
        return self.backend_functions["to_numpy"](vector)
    
    def search_vectors(self, query: ArrayLike, vectors: ArrayLike, 
                 top_k: int) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        """
        Search for nearest vectors.
        
        Args:
            query: Query vector
            vectors: Batch of vectors
            top_k: Number of top results to return
            
        Returns:
            Tuple of (similarities, indices)
        """
        # Convert to numpy arrays first for consistent handling
        query_np = self.to_numpy(query)
        vectors_np = self.to_numpy(vectors)
        
        # Use backend-specific search if available, otherwise fallback to batch similarity
        if "search_vectors" in self.backend_functions:
            return self.backend_functions["search_vectors"](query_np, vectors_np, top_k)
        
        # Fallback implementation using batch cosine similarity
        similarities = self.batch_cosine_similarity(query_np, vectors_np)
        indices = np.argsort(-similarities)[:top_k]  # Descending order
        top_similarities = similarities[indices]
        
        return top_similarities, indices
    
    def optimize_memory(self) -> None:
        """
        Optimize memory usage by clearing caches.
        
        Different backends have different memory optimization strategies.
        """
        # Clear any caches specific to backends
        if self.selected_backend == "pytorch":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("Cleared PyTorch CUDA cache")
            except (ImportError, AttributeError):
                pass
        
        elif self.selected_backend == "jax":
            try:
                import jax
                jax.clear_caches()
                logger.debug("Cleared JAX cache")
            except (ImportError, AttributeError):
                pass
        
        # Explicitly run garbage collection
        import gc
        gc.collect()


# Singleton instance for reuse
_ML_BACKEND_INSTANCE: Optional[MLBackend] = None


@lru_cache(maxsize=8)
def get_ml_backend(preferred_backend: Optional[str] = None, 
              vector_dimension: int = VECTOR_DIMENSION) -> MLBackend:
    """
    Get the MLBackend instance, creating it if necessary.
    
    Args:
        preferred_backend: Preferred backend type
        vector_dimension: Dimension of vectors to be processed
        
    Returns:
        MLBackend instance
    """
    global _ML_BACKEND_INSTANCE
    
    if _ML_BACKEND_INSTANCE is None:
        _ML_BACKEND_INSTANCE = MLBackend(
            vector_dimension=vector_dimension,
            preferred_backend=preferred_backend
        )
    
    return _ML_BACKEND_INSTANCE 