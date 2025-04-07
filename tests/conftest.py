"""
Pytest configuration file for WDBX tests.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Add the project root directory to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from wdbx package
from wdbx import BlockChainManager, EmbeddingVector, VectorStore, WDBXConfig


@pytest.fixture
def vector_store():
    """Create a vector store instance for testing."""
    return VectorStore(dimension=128)

@pytest.fixture
def wdbx_config():
    """Create a standard WDBX config for testing."""
    return WDBXConfig(
        vector_dimension=128,
        num_shards=2,
        data_dir=os.path.join(os.path.dirname(__file__), "test_data"),
        enable_caching=True,
        cache_size=10,
        cache_ttl=None  # No TTL - use LRU
    )

@pytest.fixture
def embedding_vector():
    """Create a sample embedding vector for testing."""
    vector = np.random.rand(128)
    return EmbeddingVector(vector=vector, metadata={"test": True})

@pytest.fixture
def block_chain_manager():
    """Create a block chain manager instance for testing."""
    return BlockChainManager() 