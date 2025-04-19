"""
ML module for WDBX.
This module provides high-performance machine learning functionality.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Create module-level logger
logger = logging.getLogger("wdbx.ml")

# Import ArrayLike from typing_extensions if available, otherwise use Union
try:
    from typing_extensions import Protocol, TypeAlias

    # Use type alias to define ArrayLike with specific array types
    ArrayLike: TypeAlias = Union[np.ndarray, "jnp.ndarray", "torch.Tensor"]

    # Protocol for vector operations
    class VectorLike(Protocol):
        """Protocol for objects that can be used as vectors."""

        def __array__(self) -> np.ndarray: ...

except ImportError:
    # For backward compatibility
    # Define ArrayLike as a Union type directly
    ArrayLike = Union[np.ndarray, Any]  # Any will represent jnp.ndarray and torch.Tensor

    # Fallback class for vector operations
    class VectorLike:
        """Placeholder for vector-like objects."""

        pass


# Check for JAX availability
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
    logger.info("JAX available - using accelerated vector operations")
except ImportError:
    JAX_AVAILABLE = False
    jnp = None
    logger.debug("JAX not available - will fall back to other backends")

# Check for PyTorch availability
try:
    import torch

    TORCH_AVAILABLE = True
    logger.info("PyTorch available - using accelerated neural networks")
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    logger.debug("PyTorch not available - will fall back to other backends")


# Base classes for vector operations
class VectorOperation:
    """Base class for vector operations used in optimization modules."""

    @staticmethod
    def supports_batch_operations() -> bool:
        """Check if the operation supports batched processing."""
        return False


class DistanceMeasure(VectorOperation):
    """Base class for distance measurements in vector space."""

    def __call__(self, vec1: ArrayLike, vec2: ArrayLike) -> float:
        """Calculate distance between two vectors."""
        raise NotImplementedError("Subclasses must implement this method")


class OptimizerBase:
    """Base class for optimization algorithms."""

    def __init__(self, **kwargs):
        """Initialize the optimizer."""
        self.config = kwargs

    def optimize(self, data: ArrayLike) -> ArrayLike:
        """Optimize the given data."""
        raise NotImplementedError("Subclasses must implement this method")


# Import after defining constants to avoid circular imports
from .attention import MultiHeadAttention
from .backend import BackendType, MLBackend, get_ml_backend
from .ml_integration import FeatureStoreClient, MLOpsProvider, ModelServingClient
from .neural_backtracking import NeuralBacktracker

# Forward imports to make them available directly from wdbx.ml
# Type aliases for common ML types
ModelType = Union["torch.nn.Module", Any]  # Any represents other model types
TensorDict = Dict[str, "ArrayLike"]

__all__ = [
    # Types
    "ArrayLike",
    "VectorLike",
    "ModelType",
    "TensorDict",
    # Backend
    "MLBackend",
    "BackendType",
    "get_ml_backend",
    # Base classes
    "VectorOperation",
    "DistanceMeasure",
    "OptimizerBase",
    # Modules
    "MultiHeadAttention",
    "NeuralBacktracker",
    # ML integration
    "MLOpsProvider",
    "ModelServingClient",
    "FeatureStoreClient",
    # Constants
    "JAX_AVAILABLE",
    "TORCH_AVAILABLE",
]
