"""
Asynchronous API for WDBX.

This module provides asynchronous versions of the WDBX core API methods,
allowing for better scalability and non-blocking operation in asyncio applications.
"""

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import Block, EmbeddingVector
from ..core.wdbx_core import WDBXCore, get_wdbx_core
from ..utils.logging_utils import LogContext, get_logger

# Initialize logger
logger = get_logger("wdbx.async_api")

# Thread pool for executing CPU-bound operations
_THREAD_POOL = ThreadPoolExecutor(max_workers=4)


class AsyncWDBXClient:
    """
    Asynchronous client for WDBX operations.

    This class provides asynchronous versions of the WDBX core API methods,
    allowing for better scalability and non-blocking operation in asyncio applications.
    """

    def __init__(self, wdbx_instance: Optional[WDBXCore] = None, max_workers: int = 4):
        """
        Initialize the async WDBX client.

        Args:
            wdbx_instance: WDBX core instance to wrap (or None to create a new one)
            max_workers: Maximum number of worker threads for the thread pool
        """
        self.wdbx = wdbx_instance or get_wdbx_core()
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        # Override thread pool if required
        global _THREAD_POOL
        if max_workers != 4:
            _THREAD_POOL = self._thread_pool

    async def create_vector(
        self,
        vector_data: Union[List[float], NDArray[np.float32]],
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None,
        bias_score: float = 0.0,
    ) -> EmbeddingVector:
        """
        Create a new embedding vector asynchronously.

        Args:
            vector_data: Vector data as list or numpy array
            metadata: Additional metadata for the vector
            vector_id: Optional ID for the vector (generated if not provided)
            bias_score: Bias score for the vector

        Returns:
            Created EmbeddingVector instance
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(
                _THREAD_POOL,
                functools.partial(
                    self.wdbx.create_vector,
                    vector_data=vector_data,
                    metadata=metadata,
                    vector_id=vector_id,
                    bias_score=bias_score,
                ),
            )
        except Exception as e:
            logger.error(f"Error creating vector: {e}")
            raise

    async def create_block(
        self,
        data: Dict[str, Any],
        embeddings: Optional[List[EmbeddingVector]] = None,
        block_id: Optional[str] = None,
        references: Optional[List[str]] = None,
    ) -> Block:
        """
        Create a new data block asynchronously.

        Args:
            data: Block data
            embeddings: Embedding vectors associated with this block
            block_id: Optional ID for the block (generated if not provided)
            references: Optional references to other blocks

        Returns:
            Created Block instance
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(
                _THREAD_POOL,
                functools.partial(
                    self.wdbx.create_block,
                    data=data,
                    embeddings=embeddings,
                    block_id=block_id,
                    references=references,
                ),
            )
        except Exception as e:
            logger.error(f"Error creating block: {e}")
            raise

    async def find_similar_vectors(
        self,
        query_vector: Union[EmbeddingVector, List[float], NDArray[np.float32]],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Find vectors similar to the query vector asynchronously.

        Args:
            query_vector: Query vector (EmbeddingVector, list, or numpy array)
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (vector_id, similarity) tuples sorted by similarity
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(
                _THREAD_POOL,
                functools.partial(
                    self.wdbx.find_similar_vectors,
                    query_vector=query_vector,
                    top_k=top_k,
                    threshold=threshold,
                ),
            )
        except Exception as e:
            logger.error(f"Error finding similar vectors: {e}")
            raise

    async def search_blocks(
        self,
        query: Union[str, Dict[str, Any], EmbeddingVector, NDArray[np.float32]],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[Block, float]]:
        """
        Search for blocks relevant to the query asynchronously.

        Args:
            query: Query as text, dict, embedding vector, or numpy array
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (block, similarity) tuples sorted by similarity
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(
                _THREAD_POOL,
                functools.partial(
                    self.wdbx.search_blocks, query=query, top_k=top_k, threshold=threshold
                ),
            )
        except Exception as e:
            logger.error(f"Error searching blocks: {e}")
            raise

    async def save_vector(self, vector: EmbeddingVector, overwrite: bool = False) -> bool:
        """
        Save a vector to disk asynchronously.

        Args:
            vector: Vector to save
            overwrite: Whether to overwrite if the file exists

        Returns:
            True if successful
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(
                _THREAD_POOL,
                functools.partial(self.wdbx.save_vector, vector=vector, overwrite=overwrite),
            )
        except Exception as e:
            logger.error(f"Error saving vector: {e}")
            raise

    async def save_block(self, block: Block, overwrite: bool = False) -> bool:
        """
        Save a block to disk asynchronously.

        Args:
            block: Block to save
            overwrite: Whether to overwrite if the file exists

        Returns:
            True if successful
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(
                _THREAD_POOL, functools.partial(self.wdbx.save_block, block=block, overwrite=overwrite)
            )
        except Exception as e:
            logger.error(f"Error saving block: {e}")
            raise

    async def load_vector(self, vector_id: str) -> Optional[EmbeddingVector]:
        """
        Load a vector from disk asynchronously.

        Args:
            vector_id: ID of the vector to load

        Returns:
            Loaded vector or None if not found
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(
                _THREAD_POOL, functools.partial(self.wdbx.load_vector, vector_id=vector_id)
            )
        except Exception as e:
            logger.error(f"Error loading vector: {e}")
            raise

    async def load_block(self, block_id: str) -> Optional[Block]:
        """
        Load a block from disk asynchronously.

        Args:
            block_id: ID of the block to load

        Returns:
            Loaded block or None if not found
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(
                _THREAD_POOL, functools.partial(self.wdbx.load_block, block_id=block_id)
            )
        except Exception as e:
            logger.error(f"Error loading block: {e}")
            raise

    async def optimize_memory(self) -> None:
        """
        Optimize memory usage asynchronously.
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(
                _THREAD_POOL, self.wdbx.optimize_memory
            )
        except Exception as e:
            logger.error(f"Error optimizing memory: {e}")
            raise

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics asynchronously.

        Returns:
            Dictionary of statistics
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(_THREAD_POOL, self.wdbx.get_stats)
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise

    async def clear(self) -> None:
        """
        Clear all in-memory data asynchronously.
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(_THREAD_POOL, self.wdbx.clear)
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            raise

    async def shutdown(self) -> None:
        """
        Shut down WDBX core asynchronously and release resources.
        """
        try:
            await asyncio.get_event_loop().run_in_executor(_THREAD_POOL, self.wdbx.shutdown)
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise

        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)


# Singleton instance for the async client
_ASYNC_CLIENT_INSTANCE: Optional[AsyncWDBXClient] = None


def get_async_client(
    wdbx_instance: Optional[WDBXCore] = None, max_workers: int = 4
) -> AsyncWDBXClient:
    """
    Get the async WDBX client instance.

    Args:
        wdbx_instance: WDBX core instance to wrap (or None to use the singleton)
        max_workers: Maximum number of worker threads

    Returns:
        AsyncWDBXClient instance
    """
    global _ASYNC_CLIENT_INSTANCE

    if _ASYNC_CLIENT_INSTANCE is None:
        _ASYNC_CLIENT_INSTANCE = AsyncWDBXClient(
            wdbx_instance=wdbx_instance or get_wdbx_core(), max_workers=max_workers
        )

    return _ASYNC_CLIENT_INSTANCE


# Example usage
async def example_async_usage():
    """
    Example usage of the async API.
    """
    with LogContext(component="AsyncAPI"):
        # Get the async client
        client = get_async_client()

        # Create a vector
        vector = await client.create_vector(
            vector_data=np.random.rand(128).astype(np.float32),
            metadata={"name": "Test Vector", "tags": ["test"]},
        )
        logger.info(f"Created vector: {vector.vector_id}")

        # Find similar vectors
        similar = await client.find_similar_vectors(vector, top_k=5)
        logger.info(f"Found {len(similar)} similar vectors")

        # Create a block
        block = await client.create_block(
            data={"name": "Test Block", "tags": ["test"]}, embeddings=[vector]
        )
        logger.info(f"Created block: {block.block_id}")

        # Clean up
        await client.clear()
        await client.shutdown()
