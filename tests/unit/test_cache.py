"""
Unit tests for the caching functionality in WDBX.

These tests verify that:
1. Cache is properly initialized with different configurations
2. Cache hits and misses are correctly tracked in stats
3. LRU eviction and TTL expiration work as expected
4. The cache integration with search_similar_vectors functions correctly
"""

import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.wdbx.core.config import WDBXConfig
from src.wdbx.core.wdbx import WDBX
from src.wdbx.storage.vector_store import VectorStore


@pytest.fixture
def sample_vector():
    """Create a sample vector for testing."""
    return np.random.rand(128).astype(np.float32)


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store that returns predictable results."""
    mock = MagicMock(spec=VectorStore)
    mock.search.return_value = AsyncMock(return_value=[("id1", 0.9), ("id2", 0.8), ("id3", 0.7)])
    return mock


@pytest.fixture
def mock_persona_manager():
    """Create a properly mocked PersonaManager with persona_embeddings."""
    mock = MagicMock()
    # Mock the persona_embeddings dictionary with at least one entry to prevent StopIteration
    mock.persona_embeddings = {"default": np.zeros(128)}
    return mock


@pytest.fixture
def wdbx_with_cache(mock_persona_manager):
    """Create a WDBX instance with caching enabled (LRU cache)."""
    config = WDBXConfig(
        vector_dimension=128,
        num_shards=2,
        data_dir=os.path.join(os.path.dirname(__file__), "test_data"),
        enable_caching=True,
        cache_size=10,
        cache_ttl=None,  # No TTL - use LRU
    )

    # Create directory if it doesn't exist
    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(test_data_dir, exist_ok=True)

    # Set up all the mocks we need
    with (
        patch("src.wdbx.core.VectorStore") as mock_vs,
        patch("src.wdbx.core.ShardManager") as mock_sm,
        patch("src.wdbx.core.BlockChainManager") as mock_bcm,
        patch("src.wdbx.core.MVCCManager") as mock_mvcc,
        patch("src.wdbx.core.NeuralBacktracker") as mock_nt,
        patch("src.wdbx.core.MultiHeadAttention") as mock_mha,
        patch("src.wdbx.core.PersonaManager", return_value=mock_persona_manager) as mock_pm,
        patch("src.wdbx.core.ContentFilter") as mock_cf,
        patch("src.wdbx.core.BiasDetector") as mock_bd,
        patch("src.wdbx.core.PersonaTokenManager") as mock_ptm,
    ):

        # Setup the async mock for search
        mock_vs.return_value.search = AsyncMock()
        mock_vs.return_value.search.return_value = [("id1", 0.9), ("id2", 0.8), ("id3", 0.7)]

        # Create a mock for the distributed query planner
        mock_planner = AsyncMock()
        mock_planner.execute_search.return_value = [("id1", 0.9), ("id2", 0.8), ("id3", 0.7)]

        wdbx = WDBX(config=config)
        # Replace the distributed query planner with our mock
        wdbx.distributed_query_planner = mock_planner

        yield wdbx


@pytest.fixture
def wdbx_with_ttl_cache(mock_persona_manager):
    """Create a WDBX instance with TTL caching enabled."""
    config = WDBXConfig(
        vector_dimension=128,
        num_shards=2,
        data_dir=os.path.join(os.path.dirname(__file__), "test_data"),
        enable_caching=True,
        cache_size=10,
        cache_ttl=1,  # 1 second TTL for faster testing
    )

    # Create directory if it doesn't exist
    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(test_data_dir, exist_ok=True)

    # Set up all the mocks we need
    with (
        patch("src.wdbx.core.VectorStore") as mock_vs,
        patch("src.wdbx.core.ShardManager") as mock_sm,
        patch("src.wdbx.core.BlockChainManager") as mock_bcm,
        patch("src.wdbx.core.MVCCManager") as mock_mvcc,
        patch("src.wdbx.core.NeuralBacktracker") as mock_nt,
        patch("src.wdbx.core.MultiHeadAttention") as mock_mha,
        patch("src.wdbx.core.PersonaManager", return_value=mock_persona_manager) as mock_pm,
        patch("src.wdbx.core.ContentFilter") as mock_cf,
        patch("src.wdbx.core.BiasDetector") as mock_bd,
        patch("src.wdbx.core.PersonaTokenManager") as mock_ptm,
    ):

        # Setup the async mock for search
        mock_vs.return_value.search = AsyncMock()
        mock_vs.return_value.search.return_value = [("id1", 0.9), ("id2", 0.8), ("id3", 0.7)]

        # Create a mock for the distributed query planner
        mock_planner = AsyncMock()
        mock_planner.execute_search.return_value = [("id1", 0.9), ("id2", 0.8), ("id3", 0.7)]

        wdbx = WDBX(config=config)
        # Replace the distributed query planner with our mock
        wdbx.distributed_query_planner = mock_planner

        yield wdbx


@pytest.fixture
def wdbx_without_cache(mock_persona_manager):
    """Create a WDBX instance with caching disabled."""
    config = WDBXConfig(
        vector_dimension=128,
        num_shards=2,
        data_dir=os.path.join(os.path.dirname(__file__), "test_data"),
        enable_caching=False,
    )

    # Create directory if it doesn't exist
    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(test_data_dir, exist_ok=True)

    # Set up all the mocks we need
    with (
        patch("src.wdbx.core.VectorStore") as mock_vs,
        patch("src.wdbx.core.ShardManager") as mock_sm,
        patch("src.wdbx.core.BlockChainManager") as mock_bcm,
        patch("src.wdbx.core.MVCCManager") as mock_mvcc,
        patch("src.wdbx.core.NeuralBacktracker") as mock_nt,
        patch("src.wdbx.core.MultiHeadAttention") as mock_mha,
        patch("src.wdbx.core.PersonaManager", return_value=mock_persona_manager) as mock_pm,
        patch("src.wdbx.core.ContentFilter") as mock_cf,
        patch("src.wdbx.core.BiasDetector") as mock_bd,
        patch("src.wdbx.core.PersonaTokenManager") as mock_ptm,
    ):

        # Setup the async mock for search
        mock_vs.return_value.search = AsyncMock()
        mock_vs.return_value.search.return_value = [("id1", 0.9), ("id2", 0.8), ("id3", 0.7)]

        # Create a mock for the distributed query planner
        mock_planner = AsyncMock()
        mock_planner.execute_search.return_value = [("id1", 0.9), ("id2", 0.8), ("id3", 0.7)]

        wdbx = WDBX(config=config)
        # Replace the distributed query planner with our mock
        wdbx.distributed_query_planner = mock_planner

        yield wdbx


class TestCacheInitialization:
    """Tests for cache initialization."""

    def test_lru_cache_init(self, wdbx_with_cache):
        """Test that LRU cache is correctly initialized."""
        assert wdbx_with_cache.cache is not None
        assert wdbx_with_cache.config.enable_caching is True
        assert wdbx_with_cache.config.cache_ttl is None
        assert wdbx_with_cache.config.cache_size == 10

    def test_ttl_cache_init(self, wdbx_with_ttl_cache):
        """Test that TTL cache is correctly initialized."""
        assert wdbx_with_ttl_cache.cache is not None
        assert wdbx_with_ttl_cache.config.enable_caching is True
        assert wdbx_with_ttl_cache.config.cache_ttl == 1
        assert wdbx_with_ttl_cache.config.cache_size == 10

    def test_cache_disabled(self, wdbx_without_cache):
        """Test that cache is not initialized when disabled."""
        assert wdbx_without_cache.cache is None
        assert wdbx_without_cache.config.enable_caching is False


class TestCacheOperations:
    """Tests for cache operations during searches."""

    @pytest.mark.asyncio
    async def test_cache_hit(self, wdbx_with_cache, sample_vector):
        """Test cache hit scenario."""
        # First search (cache miss)
        results1 = await wdbx_with_cache.search_similar_vectors(sample_vector, top_k=10)

        # Second search with same vector (should be cache hit)
        results2 = await wdbx_with_cache.search_similar_vectors(sample_vector, top_k=10)

        # Verify that both results are identical
        assert results1 == results2

        # Verify that the distributed query planner was called only once
        assert wdbx_with_cache.distributed_query_planner.execute_search.call_count == 1

        # Verify the stats
        assert wdbx_with_cache.stats["cache_hits"] == 1
        assert wdbx_with_cache.stats["cache_misses"] == 1

    @pytest.mark.asyncio
    async def test_cache_miss(self, wdbx_with_cache, sample_vector):
        """Test cache miss scenario."""
        # Search with first vector (cache miss)
        await wdbx_with_cache.search_similar_vectors(sample_vector, top_k=10)

        # Search with different vector (should be another cache miss)
        different_vector = sample_vector + 1.0  # Create a different vector
        await wdbx_with_cache.search_similar_vectors(different_vector, top_k=10)

        # Verify that the distributed query planner was called twice
        assert wdbx_with_cache.distributed_query_planner.execute_search.call_count == 2

        # Verify the stats
        assert wdbx_with_cache.stats["cache_hits"] == 0
        assert wdbx_with_cache.stats["cache_misses"] == 2

    @pytest.mark.asyncio
    async def test_different_params_cause_miss(self, wdbx_with_cache, sample_vector):
        """Test that different parameters cause cache misses."""
        # Search with default parameters
        await wdbx_with_cache.search_similar_vectors(sample_vector, top_k=10)

        # Search with different top_k (should be a cache miss)
        await wdbx_with_cache.search_similar_vectors(sample_vector, top_k=5)

        # Search with different threshold (should be a cache miss)
        await wdbx_with_cache.search_similar_vectors(sample_vector, top_k=10, threshold=0.5)

        # Verify that the query planner was called for each different parameter set
        assert wdbx_with_cache.distributed_query_planner.execute_search.call_count == 3

        # Verify the stats
        assert wdbx_with_cache.stats["cache_hits"] == 0
        assert wdbx_with_cache.stats["cache_misses"] == 3

    @pytest.mark.asyncio
    async def test_no_cache(self, wdbx_without_cache, sample_vector):
        """Test search behavior when cache is disabled."""
        # Search multiple times with the same vector
        await wdbx_without_cache.search_similar_vectors(sample_vector, top_k=10)
        await wdbx_without_cache.search_similar_vectors(sample_vector, top_k=10)

        # Verify that the query planner was called for each search (no caching)
        assert wdbx_without_cache.distributed_query_planner.execute_search.call_count == 2

        # Cache stats should remain at zero
        assert wdbx_without_cache.stats["cache_hits"] == 0
        assert wdbx_without_cache.stats["cache_misses"] == 0

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, wdbx_with_ttl_cache, sample_vector):
        """Test that TTL cache entries expire correctly."""
        # First search (cache miss)
        await wdbx_with_ttl_cache.search_similar_vectors(sample_vector, top_k=10)

        # Second search immediately (should be cache hit)
        await wdbx_with_ttl_cache.search_similar_vectors(sample_vector, top_k=10)

        # Wait for TTL to expire (cache set to 1 second)
        time.sleep(1.1)

        # Third search after expiration (should be cache miss)
        await wdbx_with_ttl_cache.search_similar_vectors(sample_vector, top_k=10)

        # Verify that the query planner was called twice (initial + after expiration)
        assert wdbx_with_ttl_cache.distributed_query_planner.execute_search.call_count == 2

        # Verify the stats
        assert wdbx_with_ttl_cache.stats["cache_hits"] == 1
        assert wdbx_with_ttl_cache.stats["cache_misses"] == 2

    @pytest.mark.asyncio
    async def test_stats_updated(self, wdbx_with_cache, sample_vector):
        """Test that system stats are correctly updated with cache info."""
        # Make some cache hits and misses
        await wdbx_with_cache.search_similar_vectors(sample_vector, top_k=10)
        await wdbx_with_cache.search_similar_vectors(sample_vector, top_k=10)
        await wdbx_with_cache.search_similar_vectors(sample_vector + 1.0, top_k=10)

        # Get system stats
        stats = wdbx_with_cache.get_system_stats()

        # Verify cache-related stats are present and correct
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "cache_hit_rate" in stats
        assert "cache_size" in stats

        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2
        assert stats["cache_hit_rate"] == 1 / 3
        assert stats["cache_size"] == 2  # Two distinct queries were cached

    @pytest.mark.asyncio
    async def test_lru_eviction(self, mock_persona_manager):
        """Test that LRU cache evicts entries when full."""
        # Create a temporary wdbx with a very small cache
        config = WDBXConfig(
            vector_dimension=128,
            num_shards=2,
            enable_caching=True,
            cache_size=3,  # Small cache size to force eviction
        )

        # Set up all the mocks we need
        with (
            patch("src.wdbx.core.VectorStore") as mock_vs,
            patch("src.wdbx.core.ShardManager") as mock_sm,
            patch("src.wdbx.core.BlockChainManager") as mock_bcm,
            patch("src.wdbx.core.MVCCManager") as mock_mvcc,
            patch("src.wdbx.core.NeuralBacktracker") as mock_nt,
            patch("src.wdbx.core.MultiHeadAttention") as mock_mha,
            patch("src.wdbx.core.PersonaManager", return_value=mock_persona_manager) as mock_pm,
            patch("src.wdbx.core.ContentFilter") as mock_cf,
            patch("src.wdbx.core.BiasDetector") as mock_bd,
            patch("src.wdbx.core.PersonaTokenManager") as mock_ptm,
        ):

            wdbx = WDBX(config=config)

            # Create a mock for the distributed query planner
            mock_planner = AsyncMock()
            mock_planner.execute_search.return_value = [("id1", 0.9)]
            wdbx.distributed_query_planner = mock_planner

            # Create multiple distinct vectors to fill the cache
            for i in range(5):  # More vectors than cache size
                vector = np.random.rand(128).astype(np.float32)
                await wdbx.search_similar_vectors(vector, top_k=10)

            # Verify the cache size matches the configuration
            assert wdbx.cache is not None
            assert len(wdbx.cache) <= config.cache_size

            # Verify the query planner was called for each unique vector
            assert wdbx.distributed_query_planner.execute_search.call_count == 5
