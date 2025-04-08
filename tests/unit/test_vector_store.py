"""
Unit tests for the vector store module.
"""

import numpy as np
import pytest

from src.wdbx.core.data_structures import EmbeddingVector
from src.wdbx.storage.vector_store import VectorStore


@pytest.fixture
def vector_store():
    """Create a vector store for testing."""
    return VectorStore(vector_dimension=128)


@pytest.fixture
def embedding_vector():
    """Create an embedding vector for testing."""
    vector = np.random.rand(128).astype(np.float32)
    return EmbeddingVector(vector=vector, metadata={"test": "data"})


def test_vector_store_init():
    """Test initialization of the vector store."""
    dimension = 128
    store = VectorStore(vector_dimension=dimension)
    assert store.vector_dimension == dimension
    assert len(store.vectors) == 0


def test_add_vector(vector_store, embedding_vector):
    """Test adding a vector to the store."""
    vector_id = "test-vector-1"
    result = vector_store.add(vector_id, embedding_vector)
    assert result is True
    assert vector_id in vector_store.vectors


def test_search_similar(vector_store):
    """Test similar vector search."""
    # Add some test vectors
    dim = vector_store.vector_dimension
    for i in range(5):
        # Create slightly different vectors
        vector = np.random.rand(dim)
        vector_id = f"test-vector-{i}"
        ev = EmbeddingVector(vector=vector, metadata={"index": i})
        vector_store.add(vector_id, ev)

    # Search with one of the vectors
    # The vectors are stored directly, not as EmbeddingVector objects
    sample_vector = vector_store.vectors["test-vector-0"]
    results = vector_store.search_similar(sample_vector, top_k=3)

    # First result should be the exact match
    assert len(results) > 0
    assert results[0][0] == "test-vector-0"
    assert results[0][1] > 0.9  # Should have high similarity to itself
