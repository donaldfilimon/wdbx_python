"""
WDBX Client Library.

This module provides a simple, high-level client for interacting with WDBX,
making it easier to integrate WDBX into applications.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, overload

import numpy as np
from numpy.typing import NDArray

from .api.async_api import AsyncWDBXClient, get_async_client
from .core.data_structures import Block, EmbeddingVector
from .core.wdbx_core import WDBXCore, get_wdbx_core
from .utils.errors import WDBXError, error_context, handle_error
from .utils.logging_utils import LogContext, get_logger
from .utils.profiler import profile

# Initialize logger
logger = get_logger("wdbx.client")


class WDBXClient:
    """
    High-level client for interacting with WDBX.
    
    This class provides a simple, intuitive interface for working with WDBX,
    hiding the complexity of the underlying implementation and providing
    convenience methods for common operations.
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        vector_dimension: int = 1536,
        enable_memory_optimization: bool = True,
        auto_connect: bool = True,
        config_path: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the WDBX client.
        
        Args:
            data_dir: Directory for storing vector data (if None, will use a temporary directory)
            vector_dimension: Dimension of vectors
            enable_memory_optimization: Whether to enable memory optimization
            auto_connect: Whether to connect automatically
            config_path: Path to configuration file (if provided, other parameters are ignored)
            **kwargs: Additional keyword arguments for WDBXCore
        """
        self.data_dir = data_dir
        self.vector_dimension = vector_dimension
        self.enable_memory_optimization = enable_memory_optimization
        self.config_path = config_path
        self.kwargs = kwargs
        
        # Initialize internal state
        self._core: Optional[WDBXCore] = None
        self._async_client: Optional[AsyncWDBXClient] = None
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        
        # Connect if auto_connect is True
        if auto_connect:
            self.connect()
    
    def connect(self) -> 'WDBXClient':
        """
        Connect to WDBX.
        
        Returns:
            Self for chaining
        """
        if self._core is not None:
            logger.warning("Already connected to WDBX")
            return self
        
        # Create a temporary directory if data_dir is None
        if self.data_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="wdbx_")
            self.data_dir = self._temp_dir.name
            logger.info(f"Using temporary directory for data: {self.data_dir}")
        
        # Initialize WDBX core
        with LogContext(component="WDBXClient"):
            logger.info("Connecting to WDBX...")
            try:
                if self.config_path:
                    logger.info(f"Using configuration from {self.config_path}")
                    self._core = get_wdbx_core(config_path=self.config_path)
                else:
                    logger.info(
                        f"Initializing with data_dir={self.data_dir}, "
                        f"vector_dimension={self.vector_dimension}, "
                        f"enable_memory_optimization={self.enable_memory_optimization}"
                    )
                    self._core = get_wdbx_core(
                        data_dir=self.data_dir,
                        vector_dimension=self.vector_dimension,
                        enable_memory_optimization=self.enable_memory_optimization,
                        **self.kwargs,
                    )
                logger.info("Successfully connected to WDBX")
            except Exception as e:
                error = handle_error(e, message="Failed to connect to WDBX")
                raise error
        
        return self
    
    def disconnect(self) -> None:
        """Disconnect from WDBX and release resources."""
        if self._core is None:
            logger.warning("Not connected to WDBX")
            return
        
        with LogContext(component="WDBXClient"):
            logger.info("Disconnecting from WDBX...")
            try:
                # Shutdown the core
                self._core.shutdown()
                self._core = None
                
                # Cleanup the async client if it exists
                if self._async_client is not None:
                    asyncio.run(self._async_client.shutdown())
                    self._async_client = None
                
                # Cleanup the temporary directory if it was created
                if self._temp_dir is not None:
                    self._temp_dir.cleanup()
                    self._temp_dir = None
                    self.data_dir = None
                
                logger.info("Successfully disconnected from WDBX")
            except Exception as e:
                error = handle_error(e, message="Failed to disconnect from WDBX")
                logger.error(str(error))
    
    @property
    def core(self) -> WDBXCore:
        """
        Get the WDBX core instance.
        
        Returns:
            WDBX core instance
        
        Raises:
            WDBXError: If not connected to WDBX
        """
        if self._core is None:
            raise WDBXError("Not connected to WDBX, call connect() first")
        return self._core
    
    def get_async_client(self) -> AsyncWDBXClient:
        """
        Get the async WDBX client.
        
        Returns:
            Async WDBX client
        
        Raises:
            WDBXError: If not connected to WDBX
        """
        if self._core is None:
            raise WDBXError("Not connected to WDBX, call connect() first")
        
        # Create the async client if it doesn't exist
        if self._async_client is None:
            self._async_client = get_async_client(self._core)
        
        return self._async_client
    
    @profile("create_vector")
    def create_vector(
        self,
        vector_data: Union[List[float], NDArray[np.float32]],
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None,
    ) -> EmbeddingVector:
        """
        Create a new embedding vector.
        
        Args:
            vector_data: Vector data as list or numpy array
            metadata: Additional metadata for the vector
            vector_id: Optional ID for the vector (generated if not provided)
            
        Returns:
            Created embedding vector
        
        Raises:
            WDBXError: If the vector creation fails
        """
        try:
            return self.core.create_vector(
                vector_data=vector_data,
                metadata=metadata,
                vector_id=vector_id,
            )
        except Exception as e:
            error = handle_error(e, message="Failed to create vector")
            raise error
    
    @profile("create_block")
    def create_block(
        self,
        data: Dict[str, Any],
        embeddings: Optional[List[EmbeddingVector]] = None,
        block_id: Optional[str] = None,
        references: Optional[List[str]] = None,
    ) -> Block:
        """
        Create a new data block.
        
        Args:
            data: Block data
            embeddings: Embedding vectors associated with this block
            block_id: Optional ID for the block (generated if not provided)
            references: Optional references to other blocks
            
        Returns:
            Created block
        
        Raises:
            WDBXError: If the block creation fails
        """
        try:
            return self.core.create_block(
                data=data,
                embeddings=embeddings,
                block_id=block_id,
                references=references,
            )
        except Exception as e:
            error = handle_error(e, message="Failed to create block")
            raise error
    
    @profile("find_similar_vectors")
    def find_similar_vectors(
        self,
        query: Union[EmbeddingVector, List[float], NDArray[np.float32], str],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Find vectors similar to the query.
        
        Args:
            query: Query vector or vector ID (if string)
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (vector_id, similarity) tuples sorted by similarity
        
        Raises:
            WDBXError: If the search fails
        """
        try:
            # If query is a string, treat it as a vector ID
            if isinstance(query, str):
                vector = self.get_vector(query)
                if vector is None:
                    raise WDBXError(f"Vector not found: {query}")
                return self.core.find_similar_vectors(
                    query_vector=vector,
                    top_k=top_k,
                    threshold=threshold,
                )
            else:
                return self.core.find_similar_vectors(
                    query_vector=query,
                    top_k=top_k,
                    threshold=threshold,
                )
        except Exception as e:
            error = handle_error(e, message="Failed to find similar vectors")
            raise error
    
    @profile("search_blocks")
    def search_blocks(
        self,
        query: Union[str, Dict[str, Any], EmbeddingVector, NDArray[np.float32]],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[Block, float]]:
        """
        Search for blocks relevant to the query.
        
        Args:
            query: Query as text, dict, embedding vector, or numpy array
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (block, similarity) tuples sorted by similarity
        
        Raises:
            WDBXError: If the search fails
        """
        try:
            return self.core.search_blocks(
                query=query,
                top_k=top_k,
                threshold=threshold,
            )
        except Exception as e:
            error = handle_error(e, message="Failed to search blocks")
            raise error
    
    @profile("get_vector")
    def get_vector(self, vector_id: str) -> Optional[EmbeddingVector]:
        """
        Get a vector by ID.
        
        Args:
            vector_id: ID of the vector
            
        Returns:
            Vector instance or None if not found
        """
        # Try to get from in-memory cache first
        if vector_id in self.core.vectors:
            return self.core.vectors[vector_id]
        
        # Try to load from disk
        return self.core.load_vector(vector_id)
    
    @profile("get_block")
    def get_block(self, block_id: str) -> Optional[Block]:
        """
        Get a block by ID.
        
        Args:
            block_id: ID of the block
            
        Returns:
            Block instance or None if not found
        """
        # Try to get from in-memory cache first
        if block_id in self.core.blocks:
            return self.core.blocks[block_id]
        
        # Try to load from disk
        return self.core.load_block(block_id)
    
    @profile("save_vector")
    def save_vector(self, vector: EmbeddingVector, overwrite: bool = False) -> bool:
        """
        Save a vector to disk.
        
        Args:
            vector: Vector to save
            overwrite: Whether to overwrite if the file exists
            
        Returns:
            True if successful
        """
        try:
            return self.core.save_vector(vector, overwrite=overwrite)
        except Exception as e:
            error = handle_error(e, message="Failed to save vector")
            logger.error(str(error))
            return False
    
    @profile("save_block")
    def save_block(self, block: Block, overwrite: bool = False) -> bool:
        """
        Save a block to disk.
        
        Args:
            block: Block to save
            overwrite: Whether to overwrite if the file exists
            
        Returns:
            True if successful
        """
        try:
            return self.core.save_block(block, overwrite=overwrite)
        except Exception as e:
            error = handle_error(e, message="Failed to save block")
            logger.error(str(error))
            return False
    
    @profile("batch_create_vectors")
    def batch_create_vectors(
        self,
        vector_data_list: List[Union[List[float], NDArray[np.float32]]],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        vector_ids: Optional[List[str]] = None,
    ) -> List[EmbeddingVector]:
        """
        Create multiple vectors in batch.
        
        Args:
            vector_data_list: List of vector data
            metadata_list: List of metadata for each vector
            vector_ids: List of IDs for each vector
            
        Returns:
            List of created vectors
        """
        result = []
        
        # Make sure metadata and vector IDs have the right length
        if metadata_list is None:
            metadata_list = [None] * len(vector_data_list)
        if vector_ids is None:
            vector_ids = [None] * len(vector_data_list)
        
        # Validate input lengths
        if len(vector_data_list) != len(metadata_list) or len(vector_data_list) != len(vector_ids):
            raise ValueError(
                f"Length mismatch: {len(vector_data_list)} vectors, "
                f"{len(metadata_list)} metadata, {len(vector_ids)} vector IDs"
            )
        
        # Create vectors in batch
        for i, vector_data in enumerate(vector_data_list):
            try:
                vector = self.create_vector(
                    vector_data=vector_data,
                    metadata=metadata_list[i],
                    vector_id=vector_ids[i],
                )
                result.append(vector)
            except Exception as e:
                logger.error(f"Failed to create vector at index {i}: {str(e)}")
                result.append(None)
        
        return result
    
    @profile("batch_find_similar_vectors")
    def batch_find_similar_vectors(
        self,
        queries: List[Union[EmbeddingVector, List[float], NDArray[np.float32], str]],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[List[Tuple[str, float]]]:
        """
        Find vectors similar to multiple queries in batch.
        
        Args:
            queries: List of query vectors or vector IDs
            top_k: Number of results to return for each query
            threshold: Minimum similarity threshold
            
        Returns:
            List of result lists, each containing (vector_id, similarity) tuples
        """
        results = []
        
        for i, query in enumerate(queries):
            try:
                result = self.find_similar_vectors(
                    query=query,
                    top_k=top_k,
                    threshold=threshold,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to find similar vectors for query at index {i}: {str(e)}")
                results.append([])
        
        return results
    
    @profile("optimize_memory")
    def optimize_memory(self) -> None:
        """Optimize memory usage."""
        try:
            self.core.optimize_memory()
        except Exception as e:
            error = handle_error(e, message="Failed to optimize memory")
            logger.error(str(error))
    
    @profile("clear")
    def clear(self) -> None:
        """Clear all in-memory data."""
        try:
            self.core.clear()
        except Exception as e:
            error = handle_error(e, message="Failed to clear data")
            logger.error(str(error))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary of statistics
        """
        try:
            return self.core.get_stats()
        except Exception as e:
            error = handle_error(e, message="Failed to get statistics")
            logger.error(str(error))
            return {}
    
    def export_data(self, output_dir: str, format: str = "json") -> bool:
        """
        Export all data to a directory.
        
        Args:
            output_dir: Output directory
            format: Export format (json or binary)
            
        Returns:
            True if successful
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Export all vectors
            vectors_dir = os.path.join(output_dir, "vectors")
            os.makedirs(vectors_dir, exist_ok=True)
            
            for vector_id, vector in self.core.vectors.items():
                if format == "json":
                    file_path = os.path.join(vectors_dir, f"{vector_id}.json")
                    with open(file_path, "w") as f:
                        json.dump(vector.to_dict(), f, indent=2)
                else:
                    file_path = os.path.join(vectors_dir, f"{vector_id}.bin")
                    # Use the core's binary serialization
                    with open(file_path, "wb") as f:
                        f.write(self.core._serialize_vector(vector))
            
            # Export all blocks
            blocks_dir = os.path.join(output_dir, "blocks")
            os.makedirs(blocks_dir, exist_ok=True)
            
            for block_id, block in self.core.blocks.items():
                if format == "json":
                    file_path = os.path.join(blocks_dir, f"{block_id}.json")
                    with open(file_path, "w") as f:
                        json.dump(block.to_dict(), f, indent=2)
                else:
                    file_path = os.path.join(blocks_dir, f"{block_id}.bin")
                    # Use the core's binary serialization
                    with open(file_path, "wb") as f:
                        f.write(self.core._serialize_block(block))
            
            # Export metadata
            metadata = {
                "vector_count": len(self.core.vectors),
                "block_count": len(self.core.blocks),
                "vector_dimension": self.core.vector_dimension,
                "export_time": time.time(),
                "format": format,
            }
            
            with open(os.path.join(output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            return True
        except Exception as e:
            error = handle_error(e, message="Failed to export data")
            logger.error(str(error))
            return False
    
    def import_data(self, input_dir: str, format: str = "json") -> bool:
        """
        Import data from a directory.
        
        Args:
            input_dir: Input directory
            format: Import format (json or binary)
            
        Returns:
            True if successful
        """
        try:
            # Check if input directory exists
            if not os.path.isdir(input_dir):
                raise ValueError(f"Input directory does not exist: {input_dir}")
            
            # Import vectors
            vectors_dir = os.path.join(input_dir, "vectors")
            if os.path.isdir(vectors_dir):
                for file_name in os.listdir(vectors_dir):
                    file_path = os.path.join(vectors_dir, file_name)
                    
                    # Skip directories
                    if os.path.isdir(file_path):
                        continue
                    
                    # Import based on format
                    if format == "json" and file_name.endswith(".json"):
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            self.core.create_vector_from_dict(data)
                    elif format != "json" and file_name.endswith(".bin"):
                        with open(file_path, "rb") as f:
                            data = f.read()
                            self.core._deserialize_vector(data)
            
            # Import blocks
            blocks_dir = os.path.join(input_dir, "blocks")
            if os.path.isdir(blocks_dir):
                for file_name in os.listdir(blocks_dir):
                    file_path = os.path.join(blocks_dir, file_name)
                    
                    # Skip directories
                    if os.path.isdir(file_path):
                        continue
                    
                    # Import based on format
                    if format == "json" and file_name.endswith(".json"):
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            self.core.create_block_from_dict(data)
                    elif format != "json" and file_name.endswith(".bin"):
                        with open(file_path, "rb") as f:
                            data = f.read()
                            self.core._deserialize_block(data)
            
            return True
        except Exception as e:
            error = handle_error(e, message="Failed to import data")
            logger.error(str(error))
            return False
    
    def __enter__(self) -> 'WDBXClient':
        """Context manager entry."""
        if self._core is None:
            self.connect()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.disconnect()


# Convenience functions


@contextmanager
def wdbx_session(
    data_dir: Optional[str] = None,
    vector_dimension: int = 1536,
    enable_memory_optimization: bool = True,
    **kwargs: Any,
) -> WDBXClient:
    """
    Context manager for a WDBX session.
    
    Args:
        data_dir: Directory for storing vector data (if None, will use a temporary directory)
        vector_dimension: Dimension of vectors
        enable_memory_optimization: Whether to enable memory optimization
        **kwargs: Additional keyword arguments for WDBXCore
        
    Yields:
        WDBX client
    """
    client = WDBXClient(
        data_dir=data_dir,
        vector_dimension=vector_dimension,
        enable_memory_optimization=enable_memory_optimization,
        **kwargs,
    )
    
    try:
        client.connect()
        yield client
    finally:
        client.disconnect()


def create_vector(
    vector_data: Union[List[float], NDArray[np.float32]],
    metadata: Optional[Dict[str, Any]] = None,
    vector_id: Optional[str] = None,
    data_dir: Optional[str] = None,
    **kwargs: Any,
) -> EmbeddingVector:
    """
    Create a vector without explicitly creating a client.
    
    Args:
        vector_data: Vector data as list or numpy array
        metadata: Additional metadata for the vector
        vector_id: Optional ID for the vector (generated if not provided)
        data_dir: Directory for storing vector data (if None, will use a temporary directory)
        **kwargs: Additional keyword arguments for WDBXCore
        
    Returns:
        Created embedding vector
    """
    with wdbx_session(data_dir=data_dir, **kwargs) as client:
        return client.create_vector(
            vector_data=vector_data,
            metadata=metadata,
            vector_id=vector_id,
        )


def find_similar_vectors(
    query: Union[EmbeddingVector, List[float], NDArray[np.float32], str],
    top_k: int = 10,
    threshold: float = 0.0,
    data_dir: Optional[str] = None,
    **kwargs: Any,
) -> List[Tuple[str, float]]:
    """
    Find similar vectors without explicitly creating a client.
    
    Args:
        query: Query vector or vector ID (if string)
        top_k: Number of results to return
        threshold: Minimum similarity threshold
        data_dir: Directory for storing vector data (if None, will use a temporary directory)
        **kwargs: Additional keyword arguments for WDBXCore
        
    Returns:
        List of (vector_id, similarity) tuples sorted by similarity
    """
    with wdbx_session(data_dir=data_dir, **kwargs) as client:
        return client.find_similar_vectors(
            query=query,
            top_k=top_k,
            threshold=threshold,
        ) 