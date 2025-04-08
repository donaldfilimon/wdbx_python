"""
Unit tests for the DistributedQueryPlanner class.

These tests focus on the DistributedQueryPlanner class's functionality
without requiring a full WDBX instance, using mocks where appropriate.
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.wdbx.core.wdbx import DistributedQueryPlanner
from src.wdbx.storage.shard_manager import ShardManager
from src.wdbx.storage.vector_store import VectorStore


@pytest.fixture
def mock_shard_manager():
    """Create a mock ShardManager."""
    mock = MagicMock(spec=ShardManager)
    mock.get_shard_info.return_value = {"id": "test-shard-1", "status": "active"}
    mock.get_all_shard_info.return_value = [
        {"id": "test-shard-1", "status": "active"},
        {"id": "test-shard-2", "status": "active"},
    ]
    return mock


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore with a synchronous search_similar method."""
    mock = MagicMock(spec=VectorStore)
    # Setup a synchronous mock for search_similar method
    mock.search_similar.return_value = [
        ("vector-1", 0.95),
        ("vector-2", 0.85),
        ("vector-3", 0.75),
    ]
    return mock


@pytest.fixture
def query_planner(mock_shard_manager, mock_vector_store):
    """Create a DistributedQueryPlanner instance with mocked dependencies."""
    return DistributedQueryPlanner(mock_shard_manager, mock_vector_store)


@pytest.fixture
def sample_query_vector():
    """Create a sample query vector."""
    return np.random.rand(128).astype(np.float32)


class TestQueryPlannerInit:
    """Tests for DistributedQueryPlanner initialization."""

    def test_init_with_dependencies(self, mock_shard_manager, mock_vector_store):
        """Test initialization with dependencies."""
        planner = DistributedQueryPlanner(mock_shard_manager, mock_vector_store)
        assert planner.shard_manager == mock_shard_manager
        assert planner.vector_store == mock_vector_store


class TestQueryPlannerSearch:
    """Tests for DistributedQueryPlanner search functionality."""

    @pytest.mark.asyncio
    async def test_plan_and_execute_search(self, query_planner, sample_query_vector):
        """Test the basic search functionality."""
        # Execute search
        results = await query_planner.plan_and_execute_search(sample_query_vector, top_k=3)

        # Verify the search was called on the vector store
        query_planner.vector_store.search_similar.assert_called_once_with(sample_query_vector, 3)

        # Verify results
        assert len(results) == 3
        assert results[0][0] == "vector-1"
        assert results[0][1] == 0.95

        # Results should be sorted by score (highest first)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_with_different_top_k(self, query_planner, sample_query_vector):
        """Test search with different top_k values."""
        # First search with top_k=2
        await query_planner.plan_and_execute_search(sample_query_vector, top_k=2)
        query_planner.vector_store.search_similar.assert_called_with(sample_query_vector, 2)

        # Reset mock to clear call history
        query_planner.vector_store.search_similar.reset_mock()

        # Second search with top_k=5
        await query_planner.plan_and_execute_search(sample_query_vector, top_k=5)
        query_planner.vector_store.search_similar.assert_called_with(sample_query_vector, 5)

    @pytest.mark.asyncio
    async def test_result_truncation(self, query_planner, sample_query_vector):
        """Test that results are truncated to the requested top_k."""
        # Change mock to return more results than we'll request
        query_planner.vector_store.search_similar.return_value = [
            ("vector-1", 0.95),
            ("vector-2", 0.85),
            ("vector-3", 0.75),
            ("vector-4", 0.65),
            ("vector-5", 0.55),
        ]

        # Request only top 2
        results = await query_planner.plan_and_execute_search(sample_query_vector, top_k=2)

        # Should only get the top 2
        assert len(results) == 2
        assert results[0][0] == "vector-1"
        assert results[1][0] == "vector-2"


class TestQueryPlanningLogic:
    """Tests for the query planning logic itself."""

    @pytest.mark.asyncio
    async def test_shard_interaction(self, query_planner, sample_query_vector, mock_shard_manager):
        """Test interaction with the shard manager."""
        # In the current implementation, the distributed query planner does not
        # directly interact with the shard manager for queries; it just uses the vector store.
        # This test simply verifies that the search operation completes successfully.

        # Execute search
        results = await query_planner.plan_and_execute_search(sample_query_vector, top_k=3)

        # Verify the search was called on the vector store
        query_planner.vector_store.search_similar.assert_called_once_with(sample_query_vector, 3)

        # Verify results
        assert len(results) == 3
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)

        # Note: In a future implementation, shard_manager.get_all_shard_info might be called,
        # but for now, we're just verifying the current behavior.

    @pytest.mark.asyncio
    async def test_empty_result_handling(self, query_planner, sample_query_vector):
        """Test handling of empty search results."""
        # Set the mock to return empty results
        query_planner.vector_store.search_similar.return_value = []

        # Execute search
        results = await query_planner.plan_and_execute_search(sample_query_vector, top_k=5)

        # Should get empty results
        assert results == []

    @pytest.mark.asyncio
    async def test_error_handling(self, query_planner, sample_query_vector):
        """Test handling of errors during search."""
        # Make the search raise an exception
        query_planner.vector_store.search_similar.side_effect = Exception("Test error")

        # Execute search - should propagate the exception
        with pytest.raises(Exception) as exc_info:
            await query_planner.plan_and_execute_search(sample_query_vector, top_k=3)

        assert "Test error" in str(exc_info.value)


class TestFutureExtensions:
    """Tests to validate extensions that might be implemented in the future."""

    @pytest.mark.asyncio
    async def test_multi_shard_search_simulation(self, mock_shard_manager, mock_vector_store):
        """
        Simulate what a multi-shard search might look like. This tests a potential
        future implementation rather than the current one.
        """
        # Create test vector
        query_vector = np.random.rand(128).astype(np.float32)

        # In a real multi-shard implementation, we might search each shard separately
        # and merge results. Here we're simulating that pattern.

        # Mock shard-specific search results
        shard1_results = [("vector-1", 0.95), ("vector-2", 0.85)]
        shard2_results = [("vector-3", 0.90), ("vector-4", 0.80)]

        # Create separate vector store mocks for different shards
        shard1_store = MagicMock(spec=VectorStore)
        shard1_store.search_similar = AsyncMock(return_value=shard1_results)

        shard2_store = MagicMock(spec=VectorStore)
        shard2_store.search_similar = AsyncMock(return_value=shard2_results)

        # Simulate the pattern of searching multiple shards and merging results
        import asyncio

        results = await asyncio.gather(
            shard1_store.search_similar(query_vector, 10),
            shard2_store.search_similar(query_vector, 10),
        )

        # Flatten and sort the combined results
        combined_results = []
        for shard_results in results:
            combined_results.extend(shard_results)

        combined_results.sort(key=lambda x: x[1], reverse=True)

        # Verify the combined results are as expected
        assert combined_results[0] == ("vector-1", 0.95)
        assert combined_results[1] == ("vector-3", 0.90)
        assert combined_results[2] == ("vector-2", 0.85)
        assert combined_results[3] == ("vector-4", 0.80)

        # This is the functionality we'd expect from a future implementation of
        # DistributedQueryPlanner.plan_and_execute_search
