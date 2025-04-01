# wdbx/optimized.py
"""
Performance-optimized components for WDBX.

This module provides high-performance implementations of core WDBX
components, using advanced techniques like memory mapping, connection
pooling, and batched operations for maximum throughput.
"""
import os
import mmap
import threading
import numpy as np
import time
import json
import hashlib
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from concurrent.futures import ThreadPoolExecutor
import logging
import lru

from wdbx.constants import (
    VECTOR_DIMENSION, SHARD_COUNT, NETWORK_OVERHEAD,
    DEFAULT_SIMILARITY_THRESHOLD, MAX_BATCH_SIZE,
    USE_COMPRESSION
)

logger = logging.getLogger("WDBX.Optimized")

# Try to import optional dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Using pure Python vector similarity search.")

try:
    import lmdb
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False
    logger.warning("LMDB not available. Using standard storage backend.")

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    logger.warning("JAX not available. Using standard NumPy for computations.")

# LRU cache for frequently accessed objects
CACHE_SIZE = int(os.environ.get("WDBX_CACHE_SIZE", 10000))
vector_cache = lru.LRU(CACHE_SIZE)
block_cache = lru.LRU(CACHE_SIZE)

class OptimizedVectorStore:
    """
    High-performance implementation of vector storage and similarity search.
    Uses memory mapping, connection pooling, and advanced indexing for speed.
    """
    def __init__(self, dimension: int = VECTOR_DIMENSION, index_refresh_interval: float = 1.0):
        self.dimension = dimension
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.index_lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 2))
        self.index_refresh_interval = index_refresh_interval
        self.last_index_refresh = 0
        self.index_dirty = False
        
        # Initialize the appropriate index based on availability
        if FAISS_AVAILABLE:
            # Use an IVF index for larger datasets, which is much faster
            self.index = faiss.IndexFlatIP(dimension)  # Start with flat index
            self.use_ivf = False  # Will switch to IVF once we have enough vectors
            logger.info(f"OptimizedVectorStore initialized with FAISS, dimension={dimension}")
        else:
            from wdbx.vector_store import PythonVectorIndex
            self.index = PythonVectorIndex(dimension)
            logger.info(f"OptimizedVectorStore initialized with Python fallback, dimension={dimension}")
        
        # Initialize LMDB if available
        if LMDB_AVAILABLE:
            self.db_path = os.path.join(os.environ.get("WDBX_DATA_DIR", "./wdbx_data"), "vector_store")
            os.makedirs(self.db_path, exist_ok=True)
            self.env = lmdb.open(self.db_path, map_size=1024*1024*1024*10)  # 10GB
            logger.info(f"Using LMDB for persistent storage at {self.db_path}")
            self._load_from_lmdb()
        else:
            self.env = None
            logger.info("Using in-memory storage only")
        
        # Start background index refresher
        threading.Thread(target=self._background_index_refresh, daemon=True).start()
    
    def _background_index_refresh(self):
        """Periodically refresh the index in the background."""
        while True:
            time.sleep(self.index_refresh_interval)
            with self.index_lock:
                if self.index_dirty and time.time() - self.last_index_refresh > self.index_refresh_interval:
                    self._rebuild_index()
                    self.index_dirty = False
                    self.last_index_refresh = time.time()
    
    def _rebuild_index(self):
        """Rebuild the search index."""
        if not self.vectors:
            return
        
        try:
            vectors_array = np.zeros((len(self.vectors), self.dimension), dtype=np.float32)
            self.id_map = {}
            self.reverse_id_map = {}
            
            for i, (vid, vec) in enumerate(self.vectors.items()):
                # Normalize the vector for cosine similarity
                norm = np.linalg.norm(vec)
                if norm >= 1e-10:
                    vectors_array[i] = vec / norm
                else:
                    vectors_array[i] = np.zeros(self.dimension, dtype=np.float32)
                self.id_map[i] = vid
                self.reverse_id_map[vid] = i
            
            if FAISS_AVAILABLE:
                # If we have enough vectors, switch to IVF index for faster search
                if len(self.vectors) > 1000 and not self.use_ivf:
                    n_centroids = min(int(len(self.vectors) / 39), 256)  # Rule of thumb: sqrt(n)
                    quantizer = faiss.IndexFlatIP(self.dimension)
                    self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_centroids, faiss.METRIC_INNER_PRODUCT)
                    self.index.train(vectors_array)
                    self.use_ivf = True
                    logger.info(f"Switched to IVF index with {n_centroids} centroids")
                
                # Reset index and add vectors
                if self.use_ivf:
                    if not self.index.is_trained:
                        self.index.train(vectors_array)
                    self.index.reset()
                else:
                    self.index = faiss.IndexFlatIP(self.dimension)
                
                self.index.add(vectors_array)
            else:
                # Python fallback
                self.index.reset()
                self.index.add(vectors_array)
            
            logger.debug(f"Index rebuilt with {len(self.vectors)} vectors")
        
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
    
    def _load_from_lmdb(self):
        """Load vectors and metadata from LMDB."""
        if not self.env:
            return
        
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                key_str = key.decode('utf-8')
                if key_str.startswith('vector:'):
                    vector_id = key_str[7:]
                    vector = np.frombuffer(value, dtype=np.float32).reshape(self.dimension)
                    self.vectors[vector_id] = vector
                elif key_str.startswith('metadata:'):
                    metadata_id = key_str[9:]
                    metadata = json.loads(value.decode('utf-8'))
                    self.metadata[metadata_id] = metadata
        
        if self.vectors:
            logger.info(f"Loaded {len(self.vectors)} vectors from LMDB")
            self._rebuild_index()
    
    def add_batch(self, vectors: Dict[str, np.ndarray], metadata: Dict[str, Dict[str, Any]]) -> bool:
        """
        Add a batch of vectors to the store efficiently.
        
        Args:
            vectors: Dictionary of vector_id -> vector
            metadata: Dictionary of vector_id -> metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            try:
                # Check for duplicates
                duplicates = set(vectors.keys()) & set(self.vectors.keys())
                if duplicates:
                    logger.warning(f"Found {len(duplicates)} duplicate vector IDs")
                    for dup in duplicates:
                        del vectors[dup]
                        if dup in metadata:
                            del metadata[dup]
                
                if not vectors:
                    return True  # Nothing to add after removing duplicates
                
                # Add to in-memory store
                self.vectors.update(vectors)
                self.metadata.update(metadata)
                
                # Mark index as dirty
                with self.index_lock:
                    self.index_dirty = True
                
                # Add to LMDB if available
                if self.env:
                    with self.env.begin(write=True) as txn:
                        for vid, vec in vectors.items():
                            key = f"vector:{vid}".encode('utf-8')
                            txn.put(key, vec.tobytes())
                        
                        for mid, meta in metadata.items():
                            key = f"metadata:{mid}".encode('utf-8')
                            txn.put(key, json.dumps(meta).encode('utf-8'))
                
                return True
            
            except Exception as e:
                logger.error(f"Error adding batch: {e}")
                return False
    
    def add(self, vector_id: str, vector: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a single vector to the store.
        
        Args:
            vector_id: Unique identifier for the vector
            vector: The vector to store
            metadata: Optional metadata for the vector
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Convert to NumPy array if needed
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        
        # Ensure correct shape
        if vector.shape != (self.dimension,):
            if len(vector.shape) > 1:
                vector = vector.reshape(self.dimension)
            else:
                logger.error(f"Vector has wrong dimension: {vector.shape} != ({self.dimension},)")
                return False
        
        return self.add_batch({vector_id: vector}, {vector_id: metadata or {}})
    
    def search_similar(self, query_vector: np.ndarray, top_k: int = 10,
                    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                    include_metadata: bool = False) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """
        Search for vectors similar to the query vector.
        
        Args:
            query_vector: The query vector
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of (vector_id, similarity_score, metadata) tuples
        """
        with self.index_lock:
            if len(self.vectors) == 0:
                return []
            
            # Convert to NumPy array if needed
            if not isinstance(query_vector, np.ndarray):
                query_vector = np.array(query_vector, dtype=np.float32)
            
            # Ensure correct shape
            if query_vector.shape != (self.dimension,):
                try:
                    query_vector = query_vector.reshape(self.dimension)
                except:
                    logger.error(f"Query vector has wrong dimension and cannot be reshaped: {query_vector.shape}")
                    return []
            
            # Normalize the query vector
            norm = np.linalg.norm(query_vector)
            if norm < 1e-10:
                logger.warning("Query vector has near-zero norm")
                return []
            
            query_vector = query_vector / norm
            
            # Limit top_k to number of vectors
            top_k = min(top_k, len(self.vectors))
            
            # If index is dirty and hasn't been rebuilt recently, rebuild it now
            if self.index_dirty and time.time() - self.last_index_refresh > self.index_refresh_interval:
                self._rebuild_index()
                self.index_dirty = False
                self.last_index_refresh = time.time()
            
            try:
                # Search for similar vectors
                if FAISS_AVAILABLE and self.use_ivf:
                    # Use IVF index with nprobe parameter for better recall
                    nprobe = min(32, self.index.nlist // 4)  # Rule of thumb: nlist/4
                    self.index.nprobe = nprobe
                
                scores, indices = self.index.search(query_vector.reshape(1, -1), top_k)
                
                # Build the results
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx < 0 or idx >= len(self.id_map):
                        continue
                    
                    similarity = float(scores[0][i])
                    if similarity < threshold:
                        continue
                    
                    vector_id = self.id_map[idx]
                    if include_metadata:
                        meta = self.metadata.get(vector_id, {})
                        results.append((vector_id, similarity, meta))
                    else:
                        results.append((vector_id, similarity, None))
                
                return results
            
            except Exception as e:
                logger.error(f"Error searching for similar vectors: {e}")
                return []
    
    def delete(self, vector_id: str) -> bool:
        """
        Delete a vector from the store.
        
        Args:
            vector_id: ID of the vector to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            if vector_id not in self.vectors:
                return False
            
            try:
                # Remove from in-memory store
                del self.vectors[vector_id]
                if vector_id in self.metadata:
                    del self.metadata[vector_id]
                
                # Mark index as dirty
                with self.index_lock:
                    self.index_dirty = True
                
                # Remove from LMDB if available
                if self.env:
                    with self.env.begin(write=True) as txn:
                        txn.delete(f"vector:{vector_id}".encode('utf-8'))
                        txn.delete(f"metadata:{vector_id}".encode('utf-8'))
                
                return True
            
            except Exception as e:
                logger.error(f"Error deleting vector: {e}")
                return False
    
    def close(self):
        """Close the vector store and release resources."""
        if self.env:
            self.env.close()
        self.executor.shutdown(wait=False)


class OptimizedBlockManager:
    """
    High-performance block management with integrity verification.
    Uses batched operations and parallel processing for speed.
    """
    def __init__(self, data_dir: str = None, max_workers: int = None):
        self.blocks = {}
        self.chain_heads = {}
        self.block_chain = {}
        self.lock = threading.RLock()
        self.data_dir = data_dir or os.path.join(os.environ.get("WDBX_DATA_DIR", "./wdbx_data"), "blocks")
        self.max_workers = max_workers or min(32, os.cpu_count() * 2)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize LMDB if available
        if LMDB_AVAILABLE:
            map_size = 1024 * 1024 * 1024 * 10  # 10GB
            self.env = lmdb.open(self.data_dir, map_size=map_size)
            logger.info(f"Using LMDB for persistent storage at {self.data_dir}")
            self._load_from_lmdb()
        else:
            self.env = None
            logger.info("Using in-memory storage only")
    
    def _load_from_lmdb(self):
        """Load blocks and chain metadata from LMDB."""
        if not self.env:
            return
        
        from wdbx.data_structures import Block
        
        with self.env.begin(write=False) as txn:
            # Load chain heads
            chain_heads_data = txn.get(b'chain_heads')
            if chain_heads_data:
                self.chain_heads = json.loads(chain_heads_data.decode('utf-8'))
            
            # Load block chain
            block_chain_data = txn.get(b'block_chain')
            if block_chain_data:
                self.block_chain = json.loads(block_chain_data.decode('utf-8'))
            
            # Load blocks
            cursor = txn.cursor()
            for key, value in cursor:
                key_str = key.decode('utf-8')
                if key_str.startswith('block:'):
                    block_id = key_str[6:]
                    block = Block.deserialize(value)
                    if block.validate():
                        self.blocks[block_id] = block
                    else:
                        logger.warning(f"Block {block_id} failed validation during loading")
        
        logger.info(f"Loaded {len(self.blocks)} blocks and {len(self.chain_heads)} chains from LMDB")
    
    def create_block_batch(self, blocks_data: List[Tuple[Dict[str, Any], List[np.ndarray], Optional[str], Optional[List[str]]]]) -> List[str]:
        """
        Create multiple blocks efficiently.
        
        Args:
            blocks_data: List of (data, embeddings, chain_id, context_references) tuples
            
        Returns:
            List of created block IDs
        """
        from wdbx.data_structures import Block, EmbeddingVector
        
        with self.lock:
            block_ids = []
            new_blocks = {}
            block_chain_updates = {}
            chain_heads_updates = {}
            
            for data, embeddings, chain_id, context_references in blocks_data:
                block_id = str(uuid.uuid4())
                previous_hash = ""
                
                if chain_id and chain_id in self.chain_heads:
                    head_block_id = self.chain_heads[chain_id]
                    previous_hash = self.blocks[head_block_id].hash
                
                # Convert embeddings to EmbeddingVector objects if they're not already
                embedding_objects = []
                for emb in embeddings:
                    if isinstance(emb, EmbeddingVector):
                        embedding_objects.append(emb)
                    else:
                        embedding_objects.append(EmbeddingVector(
                            vector=emb,
                            metadata={"timestamp": time.time()}
                        ))
                
                block = Block(
                    id=block_id,
                    timestamp=time.time(),
                    data=data,
                    embeddings=embedding_objects,
                    previous_hash=previous_hash,
                    context_references=context_references or []
                )
                
                new_blocks[block_id] = block
                block_ids.append(block_id)
                
                if chain_id:
                    chain_heads_updates[chain_id] = block_id
                    block_chain_updates[block_id] = chain_id
                else:
                    new_chain_id = str(uuid.uuid4())
                    chain_heads_updates[new_chain_id] = block_id
                    block_chain_updates[block_id] = new_chain_id
            
            # Update in-memory state
            self.blocks.update(new_blocks)
            self.chain_heads.update(chain_heads_updates)
            self.block_chain.update(block_chain_updates)
            
            # Persist to LMDB if available
            if self.env:
                with self.env.begin(write=True) as txn:
                    # Store blocks
                    for block_id, block in new_blocks.items():
                        txn.put(f"block:{block_id}".encode('utf-8'), block.serialize())
                    
                    # Update chain heads
                    txn.put(b'chain_heads', json.dumps(self.chain_heads).encode('utf-8'))
                    
                    # Update block chain
                    txn.put(b'block_chain', json.dumps(self.block_chain).encode('utf-8'))
            
            return block_ids
    
    def create_block(self, data: Dict[str, Any], embeddings: List, chain_id: Optional[str] = None, 
                  context_references: Optional[List[str]] = None) -> str:
        """
        Create a single block.
        
        Args:
            data: Block data
            embeddings: Embeddings to include in the block
            chain_id: Optional chain ID
            context_references: Optional context references
            
        Returns:
            ID of the created block
        """
        block_ids = self.create_block_batch([(data, embeddings, chain_id, context_references)])
        return block_ids[0]
    
    def get_block(self, block_id: str) -> Optional[object]:
        """
        Get a block by ID.
        
        Args:
            block_id: ID of the block to retrieve
            
        Returns:
            Block if found, None otherwise
        """
        # Check if block is in memory
        block = self.blocks.get(block_id)
        if block:
            return block
        
        # Check if block is in cache
        if block_id in block_cache:
            return block_cache[block_id]
        
        # Try to load from LMDB
        if self.env:
            with self.env.begin(write=False) as txn:
                from wdbx.data_structures import Block
                block_data = txn.get(f"block:{block_id}".encode('utf-8'))
                if block_data:
                    block = Block.deserialize(block_data)
                    if block.validate():
                        # Cache the block
                        block_cache[block_id] = block
                        return block
        
        return None
    
    def get_blocks_batch(self, block_ids: List[str]) -> Dict[str, object]:
        """
        Get multiple blocks efficiently.
        
        Args:
            block_ids: List of block IDs to retrieve
            
        Returns:
            Dictionary of block_id -> block
        """
        result = {}
        missing_ids = []
        
        # Check which blocks are in memory
        for block_id in block_ids:
            block = self.blocks.get(block_id)
            if block:
                result[block_id] = block
            elif block_id in block_cache:
                result[block_id] = block_cache[block_id]
            else:
                missing_ids.append(block_id)
        
        # Load missing blocks from LMDB
        if missing_ids and self.env:
            with self.env.begin(write=False) as txn:
                from wdbx.data_structures import Block
                for block_id in missing_ids:
                    block_data = txn.get(f"block:{block_id}".encode('utf-8'))
                    if block_data:
                        block = Block.deserialize(block_data)
                        if block.validate():
                            result[block_id] = block
                            block_cache[block_id] = block
        
        return result
    
    def validate_chain(self, chain_id: str) -> bool:
        """
        Validate the integrity of a chain.
        
        Args:
            chain_id: ID of the chain to validate
            
        Returns:
            True if chain is valid, False otherwise
        """
        if chain_id not in self.chain_heads:
            return False
        
        blocks = self.get_chain(chain_id)
        if not blocks:
            return False
        
        # Validate blocks in parallel
        def validate_block_link(i):
            if i >= len(blocks) - 1:
                return blocks[i].validate()
            return blocks[i].validate() and blocks[i].previous_hash == blocks[i + 1].hash
        
        results = list(self.executor.map(validate_block_link, range(len(blocks))))
        return all(results)
    
    def get_chain(self, chain_id: str) -> List[object]:
        """
        Get all blocks in a chain.
        
        Args:
            chain_id: ID of the chain to retrieve
            
        Returns:
            List of blocks in the chain, newest first
        """
        if chain_id not in self.chain_heads:
            return []
        
        blocks = []
        current_block_id = self.chain_heads[chain_id]
        
        # First try to get blocks from memory
        while current_block_id:
            block = self.get_block(current_block_id)
            if not block:
                break
            
            blocks.append(block)
            
            # Find the previous block
            prev_hash = block.previous_hash
            if not prev_hash:
                break
            
            # Look for a block with matching hash
            prev_block_id = None
            for bid, b in self.blocks.items():
                if b.hash == prev_hash:
                    prev_block_id = bid
                    break
            
            current_block_id = prev_block_id
        
        return blocks
    
    def close(self):
        """Close the block manager and release resources."""
        if self.env:
            self.env.close()
        self.executor.shutdown(wait=False)


class OptimizedTransactionManager:
    """
    High-performance transaction manager with MVCC.
    Uses optimistic concurrency control for better performance.
    """
    def __init__(self, data_dir: str = None):
        self.transactions = {}
        self.versions = {}
        self.lock = threading.RLock()
        self.data_dir = data_dir or os.path.join(os.environ.get("WDBX_DATA_DIR", "./wdbx_data"), "transactions")
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize LMDB if available
        if LMDB_AVAILABLE:
            map_size = 1024 * 1024 * 1024  # 1GB
            self.env = lmdb.open(self.data_dir, map_size=map_size)
            logger.info(f"Using LMDB for persistent storage at {self.data_dir}")
        else:
            self.env = None
            logger.info("Using in-memory storage only")
    
    def start_transaction(self):
        """
        Start a new transaction.
        
        Returns:
            Transaction object
        """
        from wdbx.mvcc import MVCCTransaction
        
        with self.lock:
            transaction = MVCCTransaction()
            self.transactions[transaction.transaction_id] = transaction
            return transaction
    
    def read(self, transaction_id: str, key: str):
        """
        Read a value from the database.
        
        Args:
            transaction_id: ID of the transaction
            key: Key to read
            
        Returns:
            Value if found, None otherwise
            
        Raises:
            ValueError: If transaction is invalid or inactive
        """
        with self.lock:
            transaction = self.transactions.get(transaction_id)
            if not transaction or not transaction.is_active():
                raise ValueError(f"Invalid or inactive transaction: {transaction_id}")
            
            transaction.read(key)
            
            # Try to get from memory first
            versions = self.versions.get(key, [])
            valid_versions = [(v, val, tid) for v, val, tid in versions 
                            if v <= transaction.version or tid == transaction_id]
            
            if valid_versions:
                valid_versions.sort(reverse=True)
                return valid_versions[0][1]
            
            # Try to get from LMDB
            if self.env:
                with self.env.begin(write=False) as txn:
                    # Get all versions of the key
                    cursor = txn.cursor()
                    prefix = f"version:{key}:".encode('utf-8')
                    if cursor.set_range(prefix):
                        valid_versions = []
                        for db_key, value in cursor:
                            if not db_key.startswith(prefix):
                                break
                            
                            # Parse key to get version and transaction ID
                            # Format: version:{key}:{version}:{transaction_id}
                            key_parts = db_key.decode('utf-8').split(':')
                            if len(key_parts) < 4:
                                continue
                            
                            version = int(key_parts[2])
                            tid = key_parts[3]
                            
                            if version <= transaction.version or tid == transaction_id:
                                valid_versions.append((version, value, tid))
                        
                        if valid_versions:
                            valid_versions.sort(reverse=True, key=lambda x: x[0])
                            return pickle.loads(valid_versions[0][1])
            
            return None
    
    def write(self, transaction_id: str, key: str, value):
        """
        Write a value to the database.
        
        Args:
            transaction_id: ID of the transaction
            key: Key to write
            value: Value to write
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ValueError: If transaction is invalid or inactive
        """
        with self.lock:
            transaction = self.transactions.get(transaction_id)
            if not transaction or not transaction.is_active():
                raise ValueError(f"Invalid or inactive transaction: {transaction_id}")
            
            transaction.write(key)
            
            # Add to in-memory versions
            if key not in self.versions:
                self.versions[key] = []
            
            self.versions[key].append((transaction.version, value, transaction_id))
            
            return True
    
    def commit(self, transaction_id: str) -> bool:
        """
        Commit a transaction.
        
        Args:
            transaction_id: ID of the transaction to commit
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            transaction = self.transactions.get(transaction_id)
            if not transaction or not transaction.is_active():
                return False
            
            # Commit to LMDB if available
            if self.env and transaction.write_set:
                with self.env.begin(write=True) as txn:
                    for key in transaction.write_set:
                        versions = self.versions.get(key, [])
                        for version, value, tid in versions:
                            if tid == transaction_id:
                                # Format: version:{key}:{version}:{transaction_id}
                                db_key = f"version:{key}:{version}:{tid}".encode('utf-8')
                                txn.put(db_key, pickle.dumps(value))
            
            transaction.commit()
            return True
    
    def abort(self, transaction_id: str) -> bool:
        """
        Abort a transaction.
        
        Args:
            transaction_id: ID of the transaction to abort
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            transaction = self.transactions.get(transaction_id)
            if not transaction or not transaction.is_active():
                return False
            
            # Remove versions written by this transaction
            for key in transaction.write_set:
                if key in self.versions:
                    self.versions[key] = [(v, val, tid) for v, val, tid in self.versions[key] 
                                        if tid != transaction_id]
                    if not self.versions[key]:
                        del self.versions[key]
            
            transaction.abort()
            return True
    
    def close(self):
        """Close the transaction manager and release resources."""
        if self.env:
            self.env.close()