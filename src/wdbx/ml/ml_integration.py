"""
JAX and PyTorch integration for WDBX.

This module provides utilities for integrating JAX and PyTorch with WDBX,
allowing for seamless conversion between NumPy, JAX, and PyTorch tensors,
and enabling the use of advanced ML capabilities in the database.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from . import JAX_AVAILABLE, TORCH_AVAILABLE, ArrayLike
from . import logger as ml_logger
from .backend import get_ml_backend

# Get ML backend from the improved WDBX backend module
ml_backend = get_ml_backend()

# Type for ML models (PyTorch modules, JAX functions, etc.)
ModelType = Union["torch.nn.Module", Callable, Any]


class MLBackendIntegration:
    """
    Advanced integration for ML backends with WDBX.

    This class provides a unified interface for ML operations, handling
    tensor conversions between different frameworks and leveraging hardware
    acceleration for optimal performance.
    """

    def __init__(self, preferred_backend: str = "auto"):
        """
        Initialize the ML backend integration.

        Args:
            preferred_backend: One of 'auto', 'jax', 'torch', or 'numpy'
        """
        self.backends_available = {
            "jax": JAX_AVAILABLE,
            "torch": TORCH_AVAILABLE,
            "numpy": True,  # NumPy is always available as it's a dependency
        }

        if preferred_backend == "auto":
            # Auto-select the best available backend
            if TORCH_AVAILABLE:
                self.backend = "torch"
            elif JAX_AVAILABLE:
                self.backend = "jax"
            else:
                self.backend = "numpy"
        # Use the specified backend if available
        elif (
            preferred_backend in self.backends_available
            and self.backends_available[preferred_backend]
        ):
            self.backend = preferred_backend
        else:
            ml_logger.warning(
                f"Requested backend '{preferred_backend}' not available, falling back to auto-selection"
            )
            if TORCH_AVAILABLE:
                self.backend = "torch"
            elif JAX_AVAILABLE:
                self.backend = "jax"
            else:
                self.backend = "numpy"

        ml_logger.info(f"Using ML integration backend: {self.backend}")

        # Initialize device information
        if self.backend == "jax":
            import jax

            self.devices = jax.devices()
            self.default_device = self.devices[0] if self.devices else None
            ml_logger.info(f"JAX devices: {self.devices}")
        elif self.backend == "torch":
            import torch

            self.cuda_available = torch.cuda.is_available()
            self.mps_available = hasattr(torch, "mps") and torch.backends.mps.is_available()

            if self.cuda_available:
                self.device = torch.device("cuda")
                ml_logger.info(f"PyTorch using CUDA device: {torch.cuda.get_device_name(0)}")
            elif self.mps_available:
                self.device = torch.device("mps")
                ml_logger.info("PyTorch using Apple Metal Performance Shaders")
            else:
                self.device = torch.device("cpu")
                ml_logger.info("PyTorch using CPU")
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
        return ml_backend.to_numpy(arr)

    def to_jax(self, arr: ArrayLike) -> Any:
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

        import jax.numpy as jnp

        if isinstance(arr, jnp.ndarray):
            return arr

        # Use ml_backend for optimal conversion
        if hasattr(ml_backend, "to_jax"):
            return ml_backend.to_jax(arr)

        # Fallback: first convert to NumPy, then to JAX
        return jnp.array(self.to_numpy(arr))

    def to_torch(self, arr: ArrayLike, device: Optional[Union[str, "torch.device"]] = None) -> Any:
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

        import torch

        # Use the default device if none specified
        if device is None:
            device = self.device if hasattr(self, "device") else torch.device("cpu")

        # If already a PyTorch tensor, just move to the correct device if needed
        if isinstance(arr, torch.Tensor):
            return arr.to(device)

        # Use ml_backend for optimal conversion if available
        if hasattr(ml_backend, "to_torch"):
            torch_tensor = ml_backend.to_torch(arr)
            return torch_tensor.to(device)

        # Fallback: convert via NumPy
        return torch.tensor(self.to_numpy(arr), device=device)

    def to_preferred(self, arr: ArrayLike) -> ArrayLike:
        """
        Convert array to the format of the preferred backend.

        Args:
            arr: Array-like object to convert

        Returns:
            Converted array in the preferred backend's format
        """
        if self.backend == "jax":
            return self.to_jax(arr)
        elif self.backend == "torch":
            return self.to_torch(arr)
        else:
            return self.to_numpy(arr)

    def cosine_similarity(self, v1: ArrayLike, v2: ArrayLike) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Cosine similarity score (float between -1 and 1)
        """
        # Use ml_backend for optimal implementation
        return ml_backend.cosine_similarity(v1, v2)

    def normalize(self, v: ArrayLike) -> ArrayLike:
        """
        Normalize a vector to unit length.

        Args:
            v: Input vector

        Returns:
            Normalized vector in the same format as input
        """
        # Use ml_backend for optimal implementation
        return ml_backend.normalize(v)

    def batch_cosine_similarity(self, query: ArrayLike, vectors: List[ArrayLike]) -> List[float]:
        """
        Calculate cosine similarities between a query vector and multiple vectors.

        Args:
            query: Query vector
            vectors: List of vectors to compare against

        Returns:
            List of similarity scores
        """
        # Convert query to preferred format
        query_converted = self.to_preferred(query)

        if self.backend == "jax" and JAX_AVAILABLE:
            import jax.numpy as jnp

            # Convert vectors to JAX arrays
            vectors_jax = [self.to_jax(v) for v in vectors]

            # Stack vectors and use batch operation
            vectors_stack = jnp.stack(vectors_jax)

            # Normalize query and vectors
            query_norm = query_converted / jnp.linalg.norm(query_converted)
            vectors_norm = vectors_stack / jnp.linalg.norm(vectors_stack, axis=1, keepdims=True)

            # Compute dot products
            similarities = jnp.dot(query_norm, vectors_norm.T)

            # Convert to list of floats
            return [float(s) for s in similarities]

        elif self.backend == "torch" and TORCH_AVAILABLE:
            import torch
            import torch.nn.functional as F

            # Convert vectors to PyTorch tensors
            vectors_torch = [self.to_torch(v) for v in vectors]

            # Stack vectors
            vectors_stack = torch.stack(vectors_torch)

            # Use PyTorch's functional cosine similarity
            query_expanded = query_converted.unsqueeze(0)
            similarities = F.cosine_similarity(query_expanded, vectors_stack)

            # Convert to list of floats
            return similarities.detach().cpu().tolist()

        else:
            # NumPy implementation
            query_np = self.to_numpy(query)

            # Use ml_backend's batch operation if available
            if hasattr(ml_backend, "batch_cosine_similarity"):
                vectors_np = np.stack([self.to_numpy(v) for v in vectors])
                return ml_backend.batch_cosine_similarity(query_np, vectors_np).tolist()

            # Fallback implementation
            return [float(self.cosine_similarity(query, v)) for v in vectors]

    def create_tensor(self, data: Any, dtype: Optional[Any] = None) -> ArrayLike:
        """
        Create a tensor of the appropriate type for the current backend.

        Args:
            data: Data to convert to tensor
            dtype: Optional data type

        Returns:
            Tensor in the preferred format
        """
        if self.backend == "jax" and JAX_AVAILABLE:
            import jax.numpy as jnp

            # Set default dtype if none specified
            if dtype is None:
                dtype = jnp.float32

            return jnp.array(data, dtype=dtype)

        elif self.backend == "torch" and TORCH_AVAILABLE:
            import torch

            # Set default dtype if none specified
            if dtype is None:
                dtype = torch.float32

            return torch.tensor(data, dtype=dtype, device=self.device)

        else:
            # NumPy implementation
            if dtype is None:
                dtype = np.float32

            return np.array(data, dtype=dtype)

    def load_model(self, model_path: str, model_type: str = "auto") -> ModelType:
        """
        Load a machine learning model from the specified path.

        Args:
            model_path: Path to the model file
            model_type: Type of model ('pytorch', 'jax', 'onnx', or 'auto')

        Returns:
            Loaded model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Auto-detect model type if not specified
        if model_type == "auto":
            if model_path.endswith((".pt", ".pth")):
                model_type = "pytorch"
            elif model_path.endswith(".onnx"):
                model_type = "onnx"
            elif model_path.endswith((".pkl", ".joblib")):
                model_type = "sklearn"
            else:
                raise ValueError(f"Could not automatically determine model type for {model_path}")

        # Load based on model type
        if model_type == "pytorch":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not available for loading PyTorch models")

            import torch

            return torch.load(model_path, map_location=self.device)

        elif model_type == "onnx":
            try:
                import onnx
                import onnxruntime as ort

                # Load ONNX model
                onnx_model = onnx.load(model_path)
                onnx.checker.check_model(onnx_model)

                # Create ONNX Runtime session
                providers = []
                if (
                    self.backend == "torch"
                    and hasattr(self, "cuda_available")
                    and self.cuda_available
                ):
                    providers.append("CUDAExecutionProvider")
                providers.append("CPUExecutionProvider")

                session = ort.InferenceSession(model_path, providers=providers)
                return session

            except ImportError:
                raise ImportError("ONNX and ONNX Runtime are required for loading ONNX models")

        elif model_type == "sklearn":
            try:
                import joblib

                return joblib.load(model_path)
            except ImportError:
                raise ImportError("joblib is required for loading scikit-learn models")

        elif model_type == "jax":
            if not JAX_AVAILABLE:
                raise ImportError("JAX is not available for loading JAX models")

            import pickle

            with open(model_path, "rb") as f:
                return pickle.load(f)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def save_model(self, model: Any, path: str, model_type: Optional[str] = None) -> None:
        """
        Save a machine learning model to the specified path.

        Args:
            model: Model to save
            path: Path to save the model to
            model_type: Type of model ('pytorch', 'jax', 'onnx', or auto-detected)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Auto-detect model type if not specified
        if model_type is None:
            if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                model_type = "pytorch"
            elif hasattr(model, "predict") and hasattr(model, "fit"):
                model_type = "sklearn"
            else:
                model_type = "pickle"

        # Save based on model type
        if model_type == "pytorch":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not available for saving PyTorch models")

            import torch

            torch.save(model, path)

        elif model_type == "onnx":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is required for exporting to ONNX")

            import torch
            import torch.onnx

            # Ensure the model is a PyTorch module
            if not isinstance(model, torch.nn.Module):
                raise TypeError("Model must be a PyTorch module for ONNX export")

            # Create dummy input based on model's first parameter shape
            dummy_input = None
            for param in model.parameters():
                if hasattr(param, "shape"):
                    # Create a batch dimension input
                    shape = (1,) + param.shape[1:] if len(param.shape) > 1 else (1, 1)
                    dummy_input = torch.randn(shape, device=self.device)
                    break

            if dummy_input is None:
                raise ValueError("Could not determine input shape for ONNX export")

            # Export to ONNX
            torch.onnx.export(model, dummy_input, path)

        elif model_type == "sklearn":
            try:
                import joblib

                joblib.dump(model, path)
            except ImportError:
                raise ImportError("joblib is required for saving scikit-learn models")

        elif model_type == "jax":
            if not JAX_AVAILABLE:
                raise ImportError("JAX is not available for saving JAX models")

            import pickle

            with open(path, "wb") as f:
                pickle.dump(model, f)

        elif model_type == "pickle":
            import pickle

            with open(path, "wb") as f:
                pickle.dump(model, f)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")


def get_ml_integration(preferred_backend: str = "auto") -> MLBackendIntegration:
    """
    Get an MLBackendIntegration instance.

    Args:
        preferred_backend: Preferred backend ('auto', 'jax', 'torch', or 'numpy')

    Returns:
        MLBackendIntegration instance
    """
    return MLBackendIntegration(preferred_backend=preferred_backend)


class MLOpsProvider:
    """Interface for logging ML experiments and tracking metrics."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the ML Ops provider with configuration."""
        self.config = config or {}
        self._initialize_tracking()

    def _initialize_tracking(self) -> None:
        """Initialize the tracking backend based on configuration."""
        self.tracking_backend = self.config.get("tracking_backend", "local")
        self.experiment_name = self.config.get("experiment_name", "wdbx_experiment")
        self.tracking_uri = self.config.get("tracking_uri", None)

        # Initialize tracking based on specified backend
        if self.tracking_backend == "mlflow":
            try:
                import mlflow

                if self.tracking_uri:
                    mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self.mlflow = mlflow
                ml_logger.info(
                    f"MLflow tracking initialized for experiment: {self.experiment_name}"
                )
            except ImportError:
                ml_logger.warning("MLflow not available. Falling back to local tracking.")
                self.tracking_backend = "local"

        if self.tracking_backend == "local":
            self.metrics = {}
            self.params = {}
            self.artifacts = {}
            ml_logger.info("Using local tracking for ML operations")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric value.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        if self.tracking_backend == "mlflow":
            self.mlflow.log_metric(key, value, step=step)
        else:
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append((step, value))

    def log_param(self, key: str, value: Any) -> None:
        """
        Log a parameter value.

        Args:
            key: Parameter name
            value: Parameter value
        """
        if self.tracking_backend == "mlflow":
            self.mlflow.log_param(key, value)
        else:
            self.params[key] = value

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact file.

        Args:
            local_path: Path to the local file
            artifact_path: Optional path within the artifact directory
        """
        if self.tracking_backend == "mlflow":
            self.mlflow.log_artifact(local_path, artifact_path)
        else:
            if artifact_path not in self.artifacts:
                self.artifacts[artifact_path or "root"] = []
            self.artifacts[artifact_path or "root"].append(local_path)

    def log_model(self, model: Any, artifact_path: str, **kwargs) -> None:
        """
        Log a machine learning model.

        Args:
            model: Model to log
            artifact_path: Path within the artifact directory
            **kwargs: Additional keyword arguments for the specific backend
        """
        if self.tracking_backend == "mlflow":
            # Try to automatically determine the model flavor
            if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                self.mlflow.pytorch.log_model(model, artifact_path, **kwargs)
            elif "sklearn" in str(type(model)):
                self.mlflow.sklearn.log_model(model, artifact_path, **kwargs)
            else:
                self.mlflow.pyfunc.log_model(artifact_path, python_model=model, **kwargs)
        else:
            # For local tracking, save model as a pickle file
            import os
            import pickle

            os.makedirs(f"artifacts/{artifact_path}", exist_ok=True)
            with open(f"artifacts/{artifact_path}/model.pkl", "wb") as f:
                pickle.dump(model, f)

            if artifact_path not in self.artifacts:
                self.artifacts[artifact_path] = []
            self.artifacts[artifact_path].append(f"artifacts/{artifact_path}/model.pkl")

    def start_run(self, run_name: Optional[str] = None) -> Any:
        """
        Start a new tracking run.

        Args:
            run_name: Optional name for the run

        Returns:
            Run object or run ID
        """
        if self.tracking_backend == "mlflow":
            return self.mlflow.start_run(run_name=run_name)
        else:
            self.metrics = {}
            self.params = {}
            self.artifacts = {}
            self.current_run = {"name": run_name, "start_time": time.time()}
            return self.current_run

    def end_run(self) -> None:
        """End the current tracking run."""
        if self.tracking_backend == "mlflow":
            self.mlflow.end_run()
        elif hasattr(self, "current_run"):
            self.current_run["end_time"] = time.time()
            ml_logger.info(f"Ended run: {self.current_run['name']}")


class ModelServingClient:
    """Client for interacting with deployed ML models."""

    def __init__(self, endpoint_url: str, api_key: Optional[str] = None):
        """
        Initialize the model serving client.

        Args:
            endpoint_url: URL of the model endpoint
            api_key: Optional API key for authentication
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.session = self._create_session()

    def _create_session(self) -> Any:
        """Create an HTTP session for API requests."""
        try:
            import requests

            session = requests.Session()
            if self.api_key:
                session.headers.update({"Authorization": f"Bearer {self.api_key}"})
            return session
        except ImportError:
            ml_logger.warning("requests library not available. Using limited functionality.")
            return None

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a prediction request to the model endpoint.

        Args:
            input_data: Input data for the prediction

        Returns:
            Prediction results
        """
        if self.session is None:
            raise ImportError("requests library is required for API requests")

        try:
            response = self.session.post(
                self.endpoint_url, json=input_data, headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            ml_logger.error(f"Error making prediction request: {e}")
            raise


class FeatureStoreClient:
    """Client for interacting with a feature store."""

    def __init__(self, api_endpoint: str, api_key: Optional[str] = None):
        """
        Initialize the feature store client.

        Args:
            api_endpoint: URL of the feature store API
            api_key: Optional API key for authentication
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.session = self._create_session()

    def _create_session(self) -> Any:
        """Create an HTTP session for API requests."""
        try:
            import requests

            session = requests.Session()
            if self.api_key:
                session.headers.update({"Authorization": f"Bearer {self.api_key}"})
            return session
        except ImportError:
            ml_logger.warning("requests library not available. Using limited functionality.")
            return None

    def get_features(
        self, entity_ids: List[str], feature_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve features from the feature store.

        Args:
            entity_ids: List of entity IDs
            feature_names: List of feature names to retrieve

        Returns:
            Dictionary of features by entity ID
        """
        if self.session is None:
            raise ImportError("requests library is required for API requests")

        try:
            response = self.session.post(
                f"{self.api_endpoint}/get-features",
                json={"entity_ids": entity_ids, "feature_names": feature_names},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            ml_logger.error(f"Error retrieving features: {e}")
            raise

    def log_features(
        self, entity_id: str, features: Dict[str, Any], timestamp: Optional[float] = None
    ) -> None:
        """
        Log features to the feature store.

        Args:
            entity_id: Entity ID
            features: Dictionary of feature values
            timestamp: Optional timestamp for the features
        """
        if self.session is None:
            raise ImportError("requests library is required for API requests")

        data = {
            "entity_id": entity_id,
            "features": features,
        }

        if timestamp is not None:
            data["timestamp"] = timestamp

        try:
            response = self.session.post(
                f"{self.api_endpoint}/log-features",
                json=data,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except Exception as e:
            ml_logger.error(f"Error logging features: {e}")
            raise


import time
