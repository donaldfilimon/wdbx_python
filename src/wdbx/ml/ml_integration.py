"""
JAX and PyTorch integration for WDBX.

This module provides utilities for integrating JAX and PyTorch with WDBX,
allowing for seamless conversion between NumPy, JAX, and PyTorch tensors,
and enabling the use of advanced ML capabilities in the database.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from . import ArrayLike

# Configure logging
logger = logging.getLogger("WDBX.ml_integration")

# Check for JAX availability
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
    logger.info("JAX available - using accelerated vector operations")
except ImportError:
    JAX_AVAILABLE = False
    jnp = None
    logger.warning("JAX not available - falling back to NumPy")

# Check for PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch available - using accelerated neural networks")
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    logger.warning("PyTorch not available - falling back to NumPy")

# Try importing torch2jax for PyTorch-JAX interop
try:
    if JAX_AVAILABLE and TORCH_AVAILABLE:
        from torch2jax import j2t, t2j
        TORCH2JAX_AVAILABLE = True
        logger.info("torch2jax available - using PyTorch-JAX interoperability")
    else:
        TORCH2JAX_AVAILABLE = False
except ImportError:
    TORCH2JAX_AVAILABLE = False
    logger.warning("torch2jax not available - PyTorch-JAX interoperability disabled")

# Use ArrayLike from wdbx.ml module instead of redefining
ModelType = Union["torch.nn.Module", Callable]


class MLBackend:
    """
    Represents the ML backend for WDBX operations.

    This class provides a unified interface for ML operations, regardless of
    whether JAX, PyTorch, or just NumPy is available.
    """

    def __init__(self, preferred_backend: str = "auto"):
        """
        Initialize the ML backend.

        Args:
            preferred_backend: One of 'auto', 'jax', 'torch', or 'numpy'
        """
        self.backends_available = {
            "jax": JAX_AVAILABLE,
            "torch": TORCH_AVAILABLE,
            "numpy": True  # NumPy is always available as it's a dependency
        }

        if preferred_backend == "auto":
            # Auto-select the best available backend
            # Prefer PyTorch over JAX based on benchmark performance
            if TORCH_AVAILABLE:
                self.backend = "torch"
            elif JAX_AVAILABLE:
                self.backend = "jax"
            else:
                self.backend = "numpy"
        else:
            # Use the specified backend if available
            if preferred_backend in self.backends_available and self.backends_available[preferred_backend]:
                self.backend = preferred_backend
            else:
                logger.warning(
                    f"Requested backend '{preferred_backend}' not available, falling back to auto-selection")
                if TORCH_AVAILABLE:
                    self.backend = "torch"
                elif JAX_AVAILABLE:
                    self.backend = "jax"
                else:
                    self.backend = "numpy"

        logger.info(f"Using ML backend: {self.backend}")

        # Initialize device information
        if self.backend == "jax":
            self.devices = jax.devices()
            self.default_device = self.devices[0]
            logger.info(f"JAX devices: {self.devices}")
        elif self.backend == "torch":
            self.cuda_available = torch.cuda.is_available()
            self.device = torch.device("cuda" if self.cuda_available else "cpu")
            logger.info(f"PyTorch device: {self.device}")
            if self.cuda_available:
                logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        else:
            self.device = None

    def to_numpy(self, arr: ArrayLike) -> np.ndarray:
        """
        Convert any array-like object to a NumPy array.

        Args:
            arr: Array-like object (NumPy array, JAX array, or PyTorch tensor)

        Returns:
            NumPy array
        """
        if arr is None:
            return None

        if isinstance(arr, np.ndarray):
            return arr

        if JAX_AVAILABLE and isinstance(arr, jnp.ndarray):
            return np.array(arr)

        if TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()

        # For any other type, try to convert via NumPy
        return np.array(arr)

    def to_jax(self, arr: ArrayLike) -> "jnp.ndarray":
        """
        Convert any array-like object to a JAX array.

        Args:
            arr: Array-like object (NumPy array, JAX array, or PyTorch tensor)

        Returns:
            JAX array
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not available")

        if arr is None:
            return None

        if isinstance(arr, jnp.ndarray):
            return arr

        if TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
            if TORCH2JAX_AVAILABLE:
                return t2j(arr)
            # Fall back to conversion via NumPy
            return jnp.array(arr.detach().cpu().numpy())

        # For NumPy arrays or any other type
        return jnp.array(arr)

    def to_torch(self, arr: ArrayLike, device: Optional[str] = None) -> "torch.Tensor":
        """
        Convert any array-like object to a PyTorch tensor.

        Args:
            arr: Array-like object (NumPy array, JAX array, or PyTorch tensor)
            device: Device to place the tensor on (None for default)

        Returns:
            PyTorch tensor
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")

        if arr is None:
            return None

        if isinstance(arr, torch.Tensor):
            if device is None:
                return arr
            return arr.to(device)

        if JAX_AVAILABLE and isinstance(arr, jnp.ndarray):
            if TORCH2JAX_AVAILABLE:
                tensor = j2t(arr)
            else:
                # Fall back to conversion via NumPy
                tensor = torch.tensor(np.array(arr))
        else:
            # For NumPy arrays or any other type
            tensor = torch.tensor(np.array(arr))

        # Move to specified device or default
        if device is not None:
            return tensor.to(device)
        if hasattr(self, "device") and self.device is not None:
            return tensor.to(self.device)
        return tensor

    def to_preferred(self, arr: ArrayLike) -> ArrayLike:
        """
        Convert any array-like object to the preferred backend format.

        Args:
            arr: Array-like object (NumPy array, JAX array, or PyTorch tensor)

        Returns:
            Array in the preferred backend format
        """
        if self.backend == "jax":
            return self.to_jax(arr)
        if self.backend == "torch":
            return self.to_torch(arr)
        return self.to_numpy(arr)

    def cosine_similarity(self, v1: ArrayLike, v2: ArrayLike) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Cosine similarity value
        """
        if self.backend == "jax":
            v1 = self.to_jax(v1)
            v2 = self.to_jax(v2)
            # Normalize vectors
            v1_norm = jnp.linalg.norm(v1)
            v2_norm = jnp.linalg.norm(v2)
            # Avoid division by zero
            if v1_norm == 0 or v2_norm == 0:
                return 0.0
            # Compute cosine similarity
            return jnp.dot(v1, v2) / (v1_norm * v2_norm)

        if self.backend == "torch":
            v1 = self.to_torch(v1)
            v2 = self.to_torch(v2)
            # Normalize vectors
            v1_norm = torch.norm(v1)
            v2_norm = torch.norm(v2)
            # Avoid division by zero
            if v1_norm == 0 or v2_norm == 0:
                return 0.0
            # Compute cosine similarity
            return torch.dot(v1, v2) / (v1_norm * v2_norm)

        v1 = self.to_numpy(v1)
        v2 = self.to_numpy(v2)
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        # Avoid division by zero
        if v1_norm == 0 or v2_norm == 0:
            return 0.0
        # Compute cosine similarity
        return np.dot(v1, v2) / (v1_norm * v2_norm)

    def normalize(self, v: ArrayLike) -> ArrayLike:
        """
        Normalize a vector to unit length.

        Args:
            v: Vector to normalize

        Returns:
            Normalized vector in the same format as input
        """
        if self.backend == "jax":
            v = self.to_jax(v)
            norm = jnp.linalg.norm(v)
            if norm > 0:
                return v / norm
            return v

        if self.backend == "torch":
            v = self.to_torch(v)
            norm = torch.norm(v)
            if norm > 0:
                return v / norm
            return v

        v = self.to_numpy(v)
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v

    def batch_cosine_similarity(self, query: ArrayLike, vectors: List[ArrayLike]) -> List[float]:
        """
        Compute cosine similarity between a query vector and multiple vectors.

        Args:
            query: Query vector
            vectors: List of vectors to compare against

        Returns:
            List of cosine similarity values
        """
        if not vectors:
            return []

        # Convert query to the preferred backend format
        query = self.to_preferred(query)

        if self.backend == "jax":
            import jax.numpy as jnp
            from jax import jit

            # Convert vectors to JAX arrays
            jax_vectors = [self.to_jax(v) for v in vectors]

            # Stack vectors into a single array if they're all the same shape
            try:
                stacked_vectors = jnp.stack(jax_vectors)
            except ValueError:
                # If stacking fails, handle vectors individually
                similarities = []
                for v in jax_vectors:
                    v_norm = jnp.linalg.norm(v)
                    q_norm = jnp.linalg.norm(query)
                    # Use jnp.where to avoid bool comparison
                    denominator = jnp.maximum(q_norm * v_norm, 1e-8)
                    similarity = jnp.dot(query.flatten(), v.flatten()) / denominator
                    similarities.append(float(similarity))
                return similarities

            # Define JIT-compatible cosine similarity
            @jit
            def cosine_sim_batch(q, vectors_batch):
                q_norm = jnp.linalg.norm(q)
                v_norms = jnp.linalg.norm(vectors_batch, axis=1)
                # Use safe division with minimum value
                denominators = jnp.maximum(q_norm * v_norms, 1e-8)
                # Compute dot products for the batch
                dots = jnp.sum(q * vectors_batch, axis=1)
                return dots / denominators

            # Compute similarities
            similarities = cosine_sim_batch(query, stacked_vectors)
            return [float(s) for s in similarities]

        if self.backend == "torch":
            import torch
            import torch.nn.functional as F

            # Convert query to PyTorch tensor and normalize
            torch_query = self.to_torch(query)
            torch_query = F.normalize(torch_query, p=2, dim=0)

            # Process in batches for potential memory saving
            batch_size = 1000
            similarities = []

            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                # Convert batch to PyTorch tensors
                torch_batch = [self.to_torch(v) for v in batch]

                # Try to stack if all tensors have the same shape
                try:
                    stacked_batch = torch.stack(torch_batch)
                    # Normalize batch
                    stacked_batch = F.normalize(stacked_batch, p=2, dim=1)
                    # Compute similarities
                    batch_similarities = torch.matmul(stacked_batch, torch_query)
                    similarities.extend(batch_similarities.cpu().tolist())
                except BaseException:
                    # Fall back to individual processing
                    for v in torch_batch:
                        v = F.normalize(v, p=2, dim=0)
                        similarity = torch.dot(v, torch_query)
                        similarities.append(similarity.item())

            return similarities

        # NumPy implementation
        query_np = self.to_numpy(query)
        query_norm = np.linalg.norm(query_np)

        similarities = []
        for v in vectors:
            v_np = self.to_numpy(v)
            v_norm = np.linalg.norm(v_np)

            if query_norm == 0 or v_norm == 0:
                similarities.append(0.0)
            else:
                similarity = np.dot(query_np.flatten(), v_np.flatten()) / (query_norm * v_norm)
                similarities.append(float(similarity))

        return similarities


# Create a default ML backend instance
default_ml_backend = MLBackend()


def get_ml_backend(preferred_backend: str = "auto") -> MLBackend:
    """
    Get an ML backend instance with the specified preference.

    Args:
        preferred_backend: One of 'auto', 'jax', 'torch', or 'numpy'

    Returns:
        MLBackend instance
    """
    return MLBackend(preferred_backend)


class MLOpsProvider:
    """Handles interactions with ML Operations platforms (e.g., MLflow, SageMaker)."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.tracking_uri = self.config.get("tracking_uri")
        # Initialize client based on config (e.g., MLflow client)
        print(f"MLOpsProvider initialized (Tracking URI: {self.tracking_uri})")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric to the MLOps platform."""
        print(f"[MLOps] Logging metric: {key}={value} (step={step})")
        # Add actual logging implementation (e.g., mlflow.log_metric)

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter to the MLOps platform."""
        print(f"[MLOps] Logging parameter: {key}={value}")
        # Add actual logging implementation (e.g., mlflow.log_param)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a local file or directory as an artifact."""
        print(f"[MLOps] Logging artifact: {local_path} (to {artifact_path})")
        # Add actual logging implementation (e.g., mlflow.log_artifact)

    def load_model(self, model_uri: str) -> Any:
        """Load a model from the MLOps platform."""
        print(f"[MLOps] Loading model from: {model_uri}")
        # Add actual model loading implementation (e.g., mlflow.pyfunc.load_model)
        # Return a placeholder model for now
        return lambda x: print(f"[Mock Model] Processing input of shape {getattr(x, 'shape', 'N/A')}")


class ModelServingClient:
    """Client for interacting with a model serving endpoint."""
    def __init__(self, endpoint_url: str, api_key: Optional[str] = None):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        # Initialize HTTP client (e.g., requests.Session)
        print(f"ModelServingClient initialized for endpoint: {endpoint_url}")

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send input data to the model serving endpoint and get predictions."""
        print(f"[Model Serving] Sending prediction request to {self.endpoint_url}")
        # Simulate request and response
        # headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        # response = requests.post(self.endpoint_url, json=input_data, headers=headers)
        # response.raise_for_status()
        # return response.json()
        
        # Mock response
        mock_prediction = np.random.rand(1, 10).tolist() # Example output shape
        return {"predictions": mock_prediction}


class FeatureStoreClient:
    """Client for interacting with a feature store."""
    def __init__(self, api_endpoint: str, api_key: Optional[str] = None):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        print(f"FeatureStoreClient initialized for endpoint: {api_endpoint}")

    def get_features(self, entity_ids: List[str], feature_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Retrieve features for given entity IDs."""
        print(f"[Feature Store] Getting features {feature_names} for entities {entity_ids}")
        # Simulate request and response
        # response = requests.post(f"{self.api_endpoint}/get-features", json={...})
        
        # Mock response
        mock_features = {}
        for entity_id in entity_ids:
            mock_features[entity_id] = {fname: random.random() for fname in feature_names}
        return mock_features

    def log_features(self, entity_id: str, features: Dict[str, Any], timestamp: Optional[float] = None) -> None:
        """Log features for a specific entity."""
        print(f"[Feature Store] Logging features for entity {entity_id}: {features}")
        # Simulate request
        # requests.post(f"{self.api_endpoint}/log-features", json={...})
        pass
