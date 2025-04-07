"""
Tests for the ML backend implementation.

This module contains tests for the ML backend functionality,
verifying that it correctly handles different backend types,
vector operations, and memory optimization.
"""

import unittest
import numpy as np
from numpy.typing import NDArray

from ..core.constants import VECTOR_DIMENSION
from ..ml.backend import MLBackend, BackendType, get_ml_backend
from ..testing.test_helpers import WDBXTestCase


class TestBackendType(unittest.TestCase):
    """Tests for the BackendType enum."""

    def test_from_string(self):
        """Test conversion from string to BackendType."""
        self.assertEqual(BackendType.from_string("numpy"), BackendType.NUMPY)
        self.assertEqual(BackendType.from_string("pytorch"), BackendType.PYTORCH)
        self.assertEqual(BackendType.from_string("torch"), BackendType.PYTORCH)  # Alias
        self.assertEqual(BackendType.from_string("jax"), BackendType.JAX)
        self.assertEqual(BackendType.from_string("faiss"), BackendType.FAISS)
        self.assertEqual(BackendType.from_string("auto"), BackendType.AUTO)
        
        # Test invalid backend falls back to AUTO
        self.assertEqual(BackendType.from_string("invalid_backend"), BackendType.AUTO)


class TestMLBackend(WDBXTestCase):
    """Tests for the MLBackend class."""

    def setUp(self):
        """Set up test vectors."""
        super().setUp()
        
        # Create some test vectors
        self.vector1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.vector2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.vector3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.vector4 = np.array([0.7, 0.7, 0.0], dtype=np.float32)
        
        # Batch of vectors
        self.batch = np.stack([self.vector1, self.vector2, self.vector3, self.vector4])
        
        # Initialize backend with NumPy (always available)
        self.backend = MLBackend(vector_dimension=3, preferred_backend="numpy")

    def test_backend_initialization(self):
        """Test that the backend initializes correctly."""
        # NumPy backend should always be available
        backend = MLBackend(preferred_backend="numpy")
        self.assertEqual(backend.selected_backend, "numpy")
        
        # Test auto-selection
        auto_backend = MLBackend(preferred_backend="auto")
        self.assertIn(auto_backend.selected_backend, ["numpy", "pytorch", "jax", "faiss"])
        
        # Test fallback to numpy for unavailable backend
        unavailable_backend = MLBackend(preferred_backend="nonexistent_backend")
        self.assertEqual(unavailable_backend.selected_backend, "numpy")

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Orthogonal vectors should have similarity 0
        self.assertAlmostEqual(
            self.backend.cosine_similarity(self.vector1, self.vector2), 
            0.0, 
            places=6
        )
        
        # Same vector should have similarity 1
        self.assertAlmostEqual(
            self.backend.cosine_similarity(self.vector1, self.vector1), 
            1.0, 
            places=6
        )
        
        # Non-orthogonal vectors should have similarity between 0 and 1
        # vector4 = [0.7, 0.7, 0.0] and vector1 = [1.0, 0.0, 0.0]
        # Cosine similarity = 0.7 / (1.0 * sqrt(0.7^2 + 0.7^2)) = 0.7 / sqrt(0.98) ≈ 0.7071
        expected_similarity = 0.7 / np.sqrt(0.98)
        self.assertAlmostEqual(
            self.backend.cosine_similarity(self.vector1, self.vector4), 
            expected_similarity, 
            places=4
        )

    def test_batch_cosine_similarity(self):
        """Test batch cosine similarity calculation."""
        # Compare vector1 to all vectors in batch
        similarities = self.backend.batch_cosine_similarity(self.vector1, self.batch)
        
        # Check shape
        self.assertEqual(similarities.shape, (4,))
        
        # Check values
        self.assertAlmostEqual(similarities[0], 1.0, places=6)  # self-similarity
        self.assertAlmostEqual(similarities[1], 0.0, places=6)  # orthogonal
        self.assertAlmostEqual(similarities[2], 0.0, places=6)  # orthogonal
        self.assertAlmostEqual(
            similarities[3], 
            0.7 / np.sqrt(0.98), 
            places=4
        )  # partial similarity

    def test_normalize(self):
        """Test vector normalization."""
        # Create a non-normalized vector
        vector = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        
        # Normalize it
        normalized = self.backend.normalize(vector)
        
        # Check that the norm is 1.0
        self.assertAlmostEqual(np.linalg.norm(normalized), 1.0, places=6)
        
        # Check that the direction is preserved
        self.assertAlmostEqual(normalized[0] / normalized[1], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(normalized[1] / normalized[2], 3.0 / 4.0, places=6)
        
        # Test normalizing a zero vector returns a zero vector
        zero_vector = np.zeros(3, dtype=np.float32)
        normalized_zero = self.backend.normalize(zero_vector)
        self.assertTrue(np.allclose(normalized_zero, zero_vector))

    def test_search_vectors(self):
        """Test vector search functionality."""
        # Search for vectors similar to vector1
        similarities, indices = self.backend.search_vectors(
            self.vector1, self.batch, top_k=2
        )
        
        # Check shape
        self.assertEqual(similarities.shape, (2,))
        self.assertEqual(indices.shape, (2,))
        
        # First result should be vector1 itself
        self.assertEqual(indices[0], 0)
        self.assertAlmostEqual(similarities[0], 1.0, places=6)
        
        # Second result should be vector4 (since it's most similar to vector1)
        self.assertEqual(indices[1], 3)
        self.assertAlmostEqual(
            similarities[1], 
            0.7 / np.sqrt(0.98), 
            places=4
        )

    def test_singleton_instance(self):
        """Test that get_ml_backend returns a singleton instance."""
        backend1 = get_ml_backend(preferred_backend="numpy")
        backend2 = get_ml_backend(preferred_backend="numpy")
        
        # Should be the same instance
        self.assertIs(backend1, backend2)
        
        # Different parameters should still return the same instance
        # because of how the singleton pattern is implemented
        backend3 = get_ml_backend(preferred_backend="auto")
        self.assertIs(backend1, backend3)

    def test_optimize_memory(self):
        """Test memory optimization."""
        # Just make sure it doesn't raise exceptions
        self.backend.optimize_memory()

    def update_metrics(self):
        """Update Prometheus metrics with current WDBX data."""
        if not self.metrics or not self.wdbx:
            return
        
        try:
            # Update vector count
            if hasattr(self.wdbx, "vectors") and hasattr(self.metrics, "vector_count"):
                self.metrics.vector_count.set(len(self.wdbx.vectors))
            
            # Update memory usage
            if hasattr(self.wdbx, "get_memory_usage") and hasattr(self.metrics, "memory_usage"):
                self.metrics.memory_usage.set(self.wdbx.get_memory_usage())
            
            # Update other metrics as appropriate
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")


class TestDifferentBackends(WDBXTestCase):
    """Tests for different ML backends if available."""

    def test_available_backends(self):
        """Test which backends are available in the environment."""
        backend = MLBackend()
        available = backend._get_available_backends()
        
        # NumPy should always be available
        self.assertIn("numpy", available)
        
        # Print which backends are available (for debugging)
        print(f"Available ML backends: {available}")
        
        # If PyTorch is available, test it
        if "pytorch" in available:
            self._test_backend("pytorch")
        
        # If JAX is available, test it
        if "jax" in available:
            self._test_backend("jax")
        
        # If FAISS is available, test it
        if "faiss" in available:
            self._test_backend("faiss")

    def _test_backend(self, backend_name):
        """Test basic functionality with a specific backend."""
        backend = MLBackend(preferred_backend=backend_name)
        self.assertEqual(backend.selected_backend, backend_name)
        
        # Test basic operations
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        c = np.array([0.5, 0.5], dtype=np.float32)
        
        # Test cosine similarity
        self.assertAlmostEqual(backend.cosine_similarity(a, a), 1.0, places=6)
        self.assertAlmostEqual(backend.cosine_similarity(a, b), 0.0, places=6)
        
        # Test batch operations
        batch = np.stack([a, b, c])
        similarities = backend.batch_cosine_similarity(a, batch)
        self.assertEqual(similarities.shape, (3,))
        
        # Test search
        similarities, indices = backend.search_vectors(a, batch, top_k=2)
        self.assertEqual(similarities.shape, (2,))
        self.assertEqual(indices.shape, (2,))
        
        # Test normalization
        normalized = backend.normalize(c)
        self.assertAlmostEqual(np.linalg.norm(normalized), 1.0, places=6)
        
        # Test optimization
        backend.optimize_memory()


if __name__ == "__main__":
    unittest.main() 