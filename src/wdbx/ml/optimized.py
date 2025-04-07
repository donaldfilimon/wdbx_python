"""
Optimized module for WDBX.

This module provides high-performance implementations of core WDBX components.
"""

import gc
import json
import logging
import os
import pickle
import sys
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import diagnostics utilities
from ..utils.diagnostics import time_operation

# Constants to avoid circular imports
VECTOR_DIMENSION = 1536
DEFAULT_SIMILARITY_THRESHOLD = 0.77

# Type checking imports are handled dynamically where needed

# Create a fallback LRU cache 
# You can install lru-dict package with: pip install lru-dict
try:
    from lru import LRU  # type: ignore
except ImportError:  # pragma: no cover
    # Try using functools.lru_cache if available (Python 3.2+)
    try:
        from functools import lru_cache
        
        class LRU(dict):
            """LRU cache implementation using functools.lru_cache."""
            
            def __init__(self, size: int = 100):
                """
                Initialize LRU cache with maximum size.
                
                Args:
                    size: Maximum number of items to store
                """
                super().__init__()
                self.size = max(size, 1)  # Ensure at least size 1
                self._cache_func = lru_cache(maxsize=self.size)(
                    lambda k: self.get_raw(k)
                )
                self.order: List[Any] = []
            
            def get_raw(self, key: Any) -> Any:
                """Get item from underlying dictionary without LRU tracking."""
                return super().__getitem__(key)
                
            def __getitem__(self, key: Any) -> Any:
                """
                Get item from cache, moving it to the front of LRU order.
                
                Args:
                    key: Key to retrieve
                    
                Returns:
                    Value associated with key
                    
                Raises:
                    KeyError: If key is not in the cache
                """
                if key not in self:
                    raise KeyError(key)
                    
                # Access through lru_cache to update recency
                self._cache_func(key)
                return super().__getitem__(key)
                
            def __setitem__(self, key: Any, value: Any) -> None:
                """
                Add or update an item in the cache.
                
                Args:
                    key: Key to store
                    value: Value to store
                """
                # Invalidate lru_cache when we change values
                self._cache_func.cache_clear()
                
                if key in self.order:
                    self.order.remove(key)
                self.order.append(key)
                
                # Maintain maximum size
                while len(self.order) > self.size:
                    oldest_key = self.order.pop(0)
                    super().__delitem__(oldest_key)
                    
                super().__setitem__(key, value)
                
            def __contains__(self, key: Any) -> bool:
                """
                Check if key is in the cache.
                
                Args:
                    key: Key to check
                    
                Returns:
                    True if key is in cache, False otherwise
                """
                return key in self.order
    except ImportError:
        # Pure Python fallback
        class LRU(dict):
            """Simple LRU implementation as fallback."""
            
            def __init__(self, size: int = 100):
                """
                Initialize LRU cache with maximum size.
                
                Args:
                    size: Maximum number of items to store
                """
                super().__init__()
                self.size = max(size, 1)  # Ensure at least size 1
                self.order: List[Any] = []
            
            def __getitem__(self, key: Any) -> Any:
                """
                Get item from cache, moving it to the front of LRU order.

                Args:
                    key: Key to retrieve

                Returns:
                    Value associated with key

                Raises:
                    KeyError: If key is not in the cache
                """
                if key not in self.order:
                    raise KeyError(key)

                self.order.remove(key)
                self.order.append(key)
                return super().__getitem__(key)
            
            def __setitem__(self, key: Any, value: Any) -> None:
                """
                Add or update an item in the cache.

                Args:
                    key: Key to store
                    value: Value to store
                """
                if key in self.order:
                    self.order.remove(key)
                self.order.append(key)
                
                # Maintain maximum size
                while len(self.order) > self.size:
                    oldest_key = self.order.pop(0)
                    super().__delitem__(oldest_key)
                    
                super().__setitem__(key, value)
                
            def __contains__(self, key: Any) -> bool:
                """
                Check if key is in the cache.

                Args:
                    key: Key to check

                Returns:
                    True if key is in cache, False otherwise
                """
                return key in self.order


# Function to dynamically import MVCCTransaction to avoid circular imports
def get_mvcc_transaction():
    """Dynamically import MVCCTransaction to avoid circular imports."""
    from ..mvcc import MVCCTransaction
    return MVCCTransaction


class OptimizedVectorStore:
    """High-performance vector store implementation."""

    def __init__(
        self,
        dimensions: int = VECTOR_DIMENSION,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ):
        """
        Initialize the optimized vector store.
        
        Args:
            dimensions: Dimension of vectors to store
            similarity_threshold: Minimum similarity threshold for queries
        """
        self.dimensions = dimensions
        self.similarity_threshold = similarity_threshold
        self.vectors: Dict[str, np.ndarray] = {}
        self.index = None
        self._lock = threading.RLock()
        
        # We might use HNSWLIB or other fast nearest-neighbor libraries
        self._has_hnswlib = False
        try:
            # Only import if available, but don't store the module
            import hnswlib  # noqa: F401
            self._has_hnswlib = True
        except ImportError:
            pass
        
        self._rebuild_index()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        # Clean up resources
        self.close()
        # Don't suppress exceptions
        return False
        
    def close(self):
        """Clean up resources."""
        with self._lock:
            self.vectors = {}
            self.index = None

    @time_operation("OptimizedVectorStore", "_rebuild_index")
    def _rebuild_index(self):
        """Rebuild the vector index for fast similarity search."""
        if not self._has_hnswlib or not self.vectors:
            return

        try:
            import hnswlib
            vec_count = len(self.vectors)
            if vec_count == 0:
                return

            self.index = hnswlib.Index(space="cosine", dim=self.dimensions)
            self.index.init_index(
                max_elements=max(1000, vec_count * 2), 
                ef_construction=200, 
                M=16
            )

            for i, (_, vector) in enumerate(self.vectors.items()):
                self.index.add_items(vector, i)

            # Create mappings between IDs and indices
            self.id_to_index = {
                vec_id: i for i, vec_id in enumerate(self.vectors.keys())
            }
            self.index_to_id = {i: vec_id for vec_id, i in self.id_to_index.items()}

            self.index.set_ef(50)  # Higher values improve recall but reduce speed
        except Exception as e:
            logging.error(f"Error rebuilding index: {e}")
            self.index = None

    @time_operation("OptimizedVectorStore", "add_batch")
    def add_batch(self, vectors: Dict[str, np.ndarray]) -> List[str]:
        """
        Add multiple vectors to the store.

        Args:
            vectors: Dictionary of vector_id -> vector

        Returns:
            List of added vector IDs
        """
        with self._lock:
            for vec_id, vector in vectors.items():
                self.vectors[vec_id] = vector

            self._rebuild_index()
        return list(vectors.keys())

    @time_operation("OptimizedVectorStore", "add")
    def add(self, vector_id: str, vector: np.ndarray) -> bool:
        """
        Add a vector to the store.

        Args:
            vector_id: ID for the vector
            vector: Vector data as numpy array

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If vector dimensions don't match the expected dimensions
            TypeError: If vector_id is not a string or vector is not a numpy array
        """
        # Type checks
        if not isinstance(vector_id, str):
            raise TypeError("vector_id must be a string")
        
        if not isinstance(vector, np.ndarray):
            raise TypeError("vector must be a numpy array")
        
        # Dimension check
        if self.dimensions and vector.shape[0] != self.dimensions:
            raise ValueError(
                f"Vector dimensions {vector.shape[0]} do not match "
                f"expected dimensions {self.dimensions}"
            )
            
        with self._lock:
            try:
                if self.index is None:
                    if not self._rebuild_index():
                        logging.error("Failed to rebuild index")
                        return False
                        
                # Store vector in database
                self.vectors[vector_id] = vector
                
                # Update index if needed
                if len(self.vectors) % 100 == 0:
                    self._rebuild_index()
                    
                return True
            except Exception as e:
                logging.error(f"Error adding vector {vector_id}: {e}")
                return False

    @time_operation("OptimizedVectorStore", "search_similar")
    def search_similar(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            query_vector: The query vector to search for
            top_k: Maximum number of results to return

        Returns:
            List of (vector_id, similarity_score) tuples
            
        Raises:
            TypeError: If query_vector is not a numpy array
            ValueError: If dimensions don't match or top_k is not positive
        """
        # Guard clause: handle empty vectors
        if not self.vectors:
            logging.warning("Search attempted on empty vector store")
            return []
            
        # Guard clause: validate query vector
        if not isinstance(query_vector, np.ndarray):
            raise TypeError("Query vector must be a numpy array")
            
        # Guard clause: ensure correct dimensions
        if len(query_vector.shape) != 1 or query_vector.shape[0] != self.dimensions:
            raise ValueError(f"Query vector must have shape ({self.dimensions},)")
            
        # Guard clause: ensure positive top_k
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        with self._lock:
            try:
                # Try to use HNSW index if available
                if self.index is not None and self._has_hnswlib:
                    try:
                        k = min(top_k, len(self.vectors))
                        labels, distances = self.index.knn_query(query_vector, k=k)

                        results = [
                            (self.index_to_id[labels[0][i]], 1.0 - distances[0][i])
                            for i in range(len(labels[0]))
                        ]
                        return sorted(results, key=lambda x: x[1], reverse=True)
                    except Exception as e:
                        logging.error(
                            f"Error in HNSW search: {e}. "
                            f"Falling back to brute force."
                        )

                # Fallback to brute force - use batch processing
                vector_ids = list(self.vectors.keys())
                vector_data = list(self.vectors.values())
                
                # Process all vectors in a batch for efficiency
                similarities = self._batch_cosine_similarity(query_vector, vector_data)
                
                # Filter by threshold and create result tuples
                results = [
                    (vector_ids[i], float(similarities[i]))
                    for i in range(len(similarities))
                    if similarities[i] >= self.similarity_threshold
                ]
                
                # Sort by similarity (descending) and limit to top_k
                return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
            except Exception as e:
                logging.error(f"Error during similarity search: {e}")
                return []

    @time_operation("OptimizedVectorStore", "delete")
    def delete(self, vector_id: str) -> bool:
        """
        Delete a vector from the store.

        Args:
            vector_id: ID of the vector to delete

        Returns:
            True if deleted, False if not found or error occurred
            
        Raises:
            TypeError: If vector_id is not a string
        """
        if not isinstance(vector_id, str):
            raise TypeError("vector_id must be a string")
            
        with self._lock:
            try:
                if vector_id in self.vectors:
                    del self.vectors[vector_id]
                    # Rebuild index to reflect the deletion
                    self._rebuild_index()
                    logging.info(f"Vector {vector_id} deleted successfully")
                    return True
                logging.warning(f"Vector {vector_id} not found for deletion")
                return False
            except Exception as e:
                logging.error(f"Error deleting vector {vector_id}: {e}")
                return False

    @time_operation("OptimizedVectorStore", "optimize_memory")
    def optimize_memory(self) -> bool:
        """
        Optimize memory usage by cleaning up resources when necessary.

        Returns:
            True if optimization was successful, False otherwise
        """
        with self._lock:
            try:
                # Log current memory usage
                mem_before = get_memory_usage()
                logging.info(
                    f"Memory usage before optimization: "
                    f"{mem_before.get('percent', 'unknown')}%"
                )
                
                # Check if we need to free memory
                vector_count = len(self.vectors)
                logging.info(f"Vector count: {vector_count}")
                
                if self.index is not None and vector_count > 10000:
                    # For large indices, explicitly free memory
                    logging.info("Large index detected, freeing memory...")
                    gc.collect()  # Force garbage collection before rebuilding
                    
                    # Free the current index to release memory
                    self.index = None
                    gc.collect()  # Force garbage collection after freeing
                    
                    # Rebuild the index with optimized parameters
                    success = self._rebuild_index()
                    
                    # Log memory usage after optimization
                    mem_after = get_memory_usage()
                    mem_diff = (
                        mem_after.get("percent", 0) - mem_before.get("percent", 0)
                    )
                    logging.info(
                        f"Memory usage after optimization: "
                        f"{mem_after.get('percent', 'unknown')}%"
                    )
                    logging.info(f"Memory change: {mem_diff:.2f}%")
                    
                    return success
                # Always rebuild the index to ensure it's optimized
                success = self._rebuild_index()
                
                # Log memory usage after optimization
                mem_after = get_memory_usage()
                mem_diff = (
                    mem_after.get("percent", 0) - mem_before.get("percent", 0)
                )
                logging.info(
                    f"Memory usage after optimization: "
                    f"{mem_after.get('percent', 'unknown')}%"
                )
                logging.info(f"Memory change: {mem_diff:.2f}%")
                
                return success
            except Exception as e:
                logging.error(f"Error optimizing memory: {e}")
                return False

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0-1)
        """
        # Handle zero vectors - avoid division by zero
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0
        
        # Normalize vectors for better numerical stability
        vec1_normalized = vec1 / norm_a
        vec2_normalized = vec2 / norm_b
        
        # Calculate dot product of normalized vectors
        # This is equivalent to cosine similarity
        return np.clip(np.dot(vec1_normalized, vec2_normalized), 0, 1)
        
    def _batch_cosine_similarity(
        self, query_vector: np.ndarray, vectors: List[np.ndarray]
    ) -> np.ndarray:
        """
        Calculate cosine similarity between query vector and multiple vectors efficiently.

        Args:
            query_vector: Query vector
            vectors: List of vectors to compare against
            
        Returns:
            Array of similarity scores
        """
        # Normalize query once
        query_norm = np.linalg.norm(query_vector)
        if query_norm < 1e-10:
            return np.zeros(len(vectors))
            
        query_normalized = query_vector / query_norm
        
        # Prepare output array
        similarities = np.zeros(len(vectors))
        
        # Calculate similarities in batch
        for i, vec in enumerate(vectors):
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-10:
                similarities[i] = 0
            else:
                vec_normalized = vec / vec_norm
                similarities[i] = np.clip(
                    np.dot(query_normalized, vec_normalized), 0, 1
                )
                
        return similarities


class OptimizedBlockManager:
    """High-performance block manager implementation."""

    def __init__(self, lmdb_path: Optional[str] = None):
        """
        Initialize a new OptimizedBlockManager.

        Args:
            lmdb_path: Optional path to LMDB database for persistence
        """
        self.blocks = {}
        self.chain_heads = {}
        self.lock = threading.RLock()
        self.lmdb_path = lmdb_path
        self.lmdb_env = None

        if lmdb_path:
            self._init_lmdb()

        # Load blocks from LMDB if available
        if self.lmdb_env:
            self._load_from_lmdb()
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        # Clean up resources
        self.close()
        # Don't suppress exceptions
        return False
        
    def close(self):
        """Clean up resources and close database connections."""
        with self.lock:
            if self.lmdb_env:
                self.lmdb_env.close()
                self.lmdb_env = None

    def _init_lmdb(self) -> None:
        """
        Initialize LMDB environment.

        Sets up LMDB for persistent storage if available.
        """
        try:
            import lmdb
            os.makedirs(os.path.dirname(self.lmdb_path), exist_ok=True)
            self.lmdb_env = lmdb.open(self.lmdb_path, map_size=1024*1024*1024)  # 1GB
        except ImportError:
            logging.warning("LMDB not available. Falling back to in-memory storage.")
            self.lmdb_env = None
        except Exception as e:
            logging.error(f"Error initializing LMDB: {e}")
            self.lmdb_env = None

    @time_operation("OptimizedBlockManager", "_load_from_lmdb")
    def _load_from_lmdb(self):
        """Load blocks from LMDB."""
        if not self.lmdb_env:
            return

        try:
            with self.lmdb_env.begin() as txn:
                # Load chain heads
                chain_heads_bytes = txn.get(b"chain_heads")
                if chain_heads_bytes:
                    self.chain_heads = json.loads(chain_heads_bytes.decode())

                # Load blocks
                block_count = 0
                cursor = txn.cursor()
                for key, value in cursor:
                    key_str = key.decode()
                    if key_str != "chain_heads" and not key_str.startswith("chain_"):
                        # SECURITY: Ensure this is only used with trusted data sources
                        # Consider alternative serialization methods for untrusted data
                        block_data = pickle.loads(value)
                        self.blocks[key_str] = block_data
                        block_count += 1

                # Validate blocks and chains
                chain_heads_copy = dict(self.chain_heads)
                for chain_id, head_id in chain_heads_copy.items():
                    if head_id not in self.blocks:
                        logging.warning(
                            f"Chain head {head_id} for chain {chain_id} not found."
                        )
                        del self.chain_heads[chain_id]

                logging.info(
                    f"Loaded {block_count} blocks and {len(self.chain_heads)} "
                    f"chains from LMDB."
                )
        except Exception as e:
            logging.error(f"Error loading from LMDB: {e}")

    @time_operation("OptimizedBlockManager", "create_block_batch")
    def create_block_batch(self, block_data_batch):
        """
        Create multiple blocks efficiently.
        
        Args:
            block_data_batch: List of (data, embedding, chain_id, context_ref) tuples
            
        Returns:
            List of created block IDs
        """
        # Guard clause: handle empty batch
        if not block_data_batch:
            return []
            
        # Guard clause: validate input format
        if not isinstance(block_data_batch, (list, tuple)):
            raise TypeError("block_data_batch must be a list or tuple")
            
        # Import here to avoid circular dependencies 
        # Not using the TYPE_CHECKING imports
        from ..data_structures import Block, EmbeddingVector
        
        block_ids = []
        with self.lock:
            for idx, item in enumerate(block_data_batch):
                # Guard clause: validate tuple format
                if not isinstance(item, (list, tuple)) or len(item) != 4:
                    raise ValueError(f"Item at index {idx} must be a 4-element tuple")
                    
                block_data, embedding, chain_id, context_ref = item
                
                # Validate embedding
                if not isinstance(embedding, (list, tuple, np.ndarray)):
                    raise TypeError(
                        f"Embedding at index {idx} must be a vector-like object"
                    )
                
                block_id = str(uuid.uuid4())
                
                # Create block
                block = Block(
                    id=block_id,
                    data=block_data,
                    embedding=EmbeddingVector(vector=embedding),
                    chain_id=chain_id,
                    timestamp=datetime.now().isoformat(),
                    context_ref=context_ref
                )
                
                # Update chain
                if chain_id in self.chain_heads:
                    prev_head = self.chain_heads[chain_id]
                    block.prev_block_id = prev_head
                
                self.blocks[block_id] = block
                self.chain_heads[chain_id] = block_id
                block_ids.append(block_id)

            # Persist to LMDB if available
            if self.lmdb_env:
                with self.lmdb_env.begin(write=True) as txn:
                    for block_id in block_ids:
                        block = self.blocks[block_id]
                        txn.put(block_id.encode(), pickle.dumps(block))

                    # Update chain heads
                    txn.put(
                        b"chain_heads",
                        json.dumps(self.chain_heads).encode()
                    )

        return block_ids

    @time_operation("OptimizedBlockManager", "get_block")
    def get_block(self, block_id: str) -> Optional[Any]:
        """
        Get a block by ID.

        Args:
            block_id: ID of the block to retrieve

        Returns:
            Block if found, None otherwise
        """
        with self.lock:
            return self.blocks.get(block_id)

    @time_operation("OptimizedBlockManager", "get_blocks_batch")
    def get_blocks_batch(self, block_ids: List[str]) -> Dict[str, Any]:
        """
        Get multiple blocks by IDs.

        Args:
            block_ids: List of block IDs to retrieve

        Returns:
            Dictionary of block_id -> block
        """
        with self.lock:
            return {block_id: self.blocks.get(block_id) for block_id in block_ids}

    @time_operation("OptimizedBlockManager", "validate_chain")
    def validate_chain(self, chain_id: str) -> bool:
        """
        Validate the integrity of a block chain.

        Args:
            chain_id: ID of the chain to validate

        Returns:
            True if chain is valid, False otherwise
        """
        with self.lock:
            if chain_id not in self.chain_heads:
                return False

            current_id = self.chain_heads[chain_id]
            visited = set()

            while current_id:
                if current_id in visited:
                    logging.error(f"Cycle detected in chain {chain_id}")
                    return False

                visited.add(current_id)
                block = self.blocks.get(current_id)

                if not block:
                    logging.error(f"Missing block {current_id} in chain {chain_id}")
                    return False

                current_id = block.prev_block_id

            return True


class OptimizedTransactionManager:
    """High-performance transaction manager implementation."""

    def __init__(self):
        """Initialize a new OptimizedTransactionManager."""
        self.active_transactions = {}
        self.lock = threading.RLock()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        # Clean up resources
        self.close()
        # Don't suppress exceptions
        return False
        
    def close(self):
        """Abort all active transactions and clean up resources."""
        with self.lock:
            active_ids = list(self.active_transactions.keys())
            for tx_id in active_ids:
                try:
                    self.abort(tx_id)
                except ValueError:
                    # Transaction might have been removed by another thread
                    pass
            self.active_transactions = {}

    @time_operation("OptimizedTransactionManager", "start_transaction")
    def start_transaction(self) -> Any:
        """
        Start a new transaction.

        Returns:
            New transaction object
            
        Raises:
            RuntimeError: If unable to create transaction
        """
        # Dynamically import MVCCTransaction to avoid circular imports
        try:
            mvcc_transaction_class = get_mvcc_transaction()
        except ImportError as e:
            raise RuntimeError(f"Failed to import MVCCTransaction: {e}") from e

        with self.lock:
            transaction_id = str(uuid.uuid4())
            transaction = mvcc_transaction_class(transaction_id)
            self.active_transactions[transaction_id] = transaction
            logging.debug(f"Started transaction {transaction_id}")
            return transaction

    @time_operation("OptimizedTransactionManager", "read")
    def read(self, transaction_id: str, key: str) -> Any:
        """
        Read a value using the specified transaction.

        Args:
            transaction_id: ID of the transaction
            key: Key to read

        Returns:
            Value if found, None otherwise

        Raises:
            ValueError: If transaction is not found
            TypeError: If key is not a string
        """
        # Guard clause: validate key type
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
            
        with self.lock:
            if transaction_id not in self.active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found")

            transaction = self.active_transactions[transaction_id]
            try:
                return transaction.read(key)
            except Exception as e:
                logging.error(
                    f"Error reading key {key} in transaction {transaction_id}: {e}"
                )
                raise

    @time_operation("OptimizedTransactionManager", "write")
    def write(self, transaction_id: str, key: str, value: Any) -> bool:
        """
        Write a value using the specified transaction.

        Args:
            transaction_id: ID of the transaction
            key: Key to write
            value: Value to write

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If transaction is not found
            TypeError: If key is not a string
        """
        # Guard clause: validate key type
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
            
        with self.lock:
            if transaction_id not in self.active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found")

            transaction = self.active_transactions[transaction_id]
            try:
                result = transaction.write(key, value)
                return result
            except Exception as e:
                logging.error(
                    f"Error writing key {key} in transaction {transaction_id}: {e}"
                )
                raise

    @time_operation("OptimizedTransactionManager", "commit")
    def commit(self, transaction_id: str) -> bool:
        """
        Commit a transaction.

        Args:
            transaction_id: ID of the transaction to commit

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If transaction is not found
        """
        with self.lock:
            if transaction_id not in self.active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found")

            transaction = self.active_transactions[transaction_id]
            try:
                result = transaction.commit()
                del self.active_transactions[transaction_id]
                logging.info(f"Transaction {transaction_id} committed successfully")
                return result
            except Exception as e:
                logging.error(f"Error committing transaction {transaction_id}: {e}")
                raise

    @time_operation("OptimizedTransactionManager", "abort")
    def abort(self, transaction_id: str) -> bool:
        """
        Abort a transaction.

        Args:
            transaction_id: ID of the transaction to abort

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If transaction is not found
        """
        with self.lock:
            if transaction_id not in self.active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found")

            transaction = self.active_transactions[transaction_id]
            try:
                result = transaction.abort()
                del self.active_transactions[transaction_id]
                logging.info(f"Transaction {transaction_id} aborted successfully")
                return result
            except Exception as e:
                logging.error(f"Error aborting transaction {transaction_id}: {e}")
                raise

# Function to get memory usage
def get_memory_usage() -> Dict[str, Any]:
    """
    Get current process memory usage.
    
    Returns:
        Dictionary with memory usage information (bytes, percent)
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss,  # Resident Set Size
            "vms": memory_info.vms,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "timestamp": time.time()
        }
    except ImportError:
        # Fallback method for systems without psutil
        import gc
        gc.collect()  # Force garbage collection before measuring
        
        # Get size of tracked objects
        total_size = 0
        objects = gc.get_objects()
        for obj in objects[:1000]:  # Limit to avoid excessive processing
            try:
                total_size += sys.getsizeof(obj)
            except (TypeError, AttributeError):
                pass
                
        return {
            "rss": total_size,
            "vms": total_size,
            "percent": -1,  # Cannot determine percent
            "timestamp": time.time()
        }
