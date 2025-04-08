"""
Core WDBX engine implementation.

This module provides the main WDBX class that serves as the central
interface for interacting with the WDBX system, integrating various
components such as data structures, ML backends, and memory management.
"""

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..config.config_manager import ConfigManager, get_config
from ..ml.backend import get_ml_backend
from ..utils.logging_utils import LogContext
from ..utils.memory_utils import get_memory_monitor, get_memory_usage, optimize_memory
from .constants import DEFAULT_BIAS_SCORE, VECTOR_DIMENSION, logger
from .data_structures import Block, EmbeddingVector


@dataclass
class ProcessingStats:
    """
    Statistics for WDBX processing operations.

    This class tracks performance metrics for WDBX operations including
    processing time, memory usage, and operation counts. It provides
    methods to record and finalize statistics during processing.

    Attributes:
        start_time: Time when processing started
        end_time: Time when processing ended (0.0 if not finished)
        memory_before: Memory usage when processing started (in MB)
        memory_after: Memory usage when processing ended (in MB)
        memory_peak: Peak memory usage during processing (in MB)
        num_vectors_processed: Number of vectors processed
        num_blocks_processed: Number of blocks processed
        optimization_count: Number of memory optimizations performed
    """

    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    memory_before: float = field(default_factory=get_memory_usage)
    memory_after: float = 0.0
    memory_peak: float = field(default_factory=get_memory_usage)
    num_vectors_processed: int = 0
    num_blocks_processed: int = 0
    optimization_count: int = 0
    _last_memory_check: float = field(default_factory=time.time, repr=False)
    _memory_check_interval: float = 0.5  # Check memory at most every 0.5 seconds

    def finalize(self) -> None:
        """
        Finalize statistics after processing is complete.

        Records the end time and final memory usage statistics.
        """
        self.end_time = time.time()
        self.memory_after = get_memory_usage()
        self.memory_peak = max(self.memory_peak, self.memory_after)

    @property
    def processing_time(self) -> float:
        """
        Get the total processing time in seconds.

        Returns:
            Processing time duration
        """
        if self.end_time == 0.0:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def memory_used(self) -> float:
        """
        Get the memory used during processing in MB.

        Returns:
            Peak memory used minus initial memory
        """
        return self.memory_peak - self.memory_before

    @property
    def vectors_per_second(self) -> float:
        """
        Get the rate of vector processing in vectors per second.

        Returns:
            Processing rate or 0.0 if no vectors were processed
        """
        processing_time = max(self.processing_time, 0.001)  # Avoid division by zero
        return (
            self.num_vectors_processed / processing_time if self.num_vectors_processed > 0 else 0.0
        )

    @property
    def blocks_per_second(self) -> float:
        """
        Get the rate of block processing in blocks per second.

        Returns:
            Processing rate or 0.0 if no blocks were processed
        """
        processing_time = max(self.processing_time, 0.001)  # Avoid division by zero
        return self.num_blocks_processed / processing_time if self.num_blocks_processed > 0 else 0.0

    def record_memory_usage(self) -> None:
        """
        Record current memory usage and update peak if necessary.

        Throttles memory checks to avoid excessive system calls.
        """
        current_time = time.time()
        # Only check memory usage if enough time has passed since last check
        if current_time - self._last_memory_check >= self._memory_check_interval:
            current = get_memory_usage()
            self.memory_peak = max(self.memory_peak, current)
            self._last_memory_check = current_time

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert stats to dictionary for logging.

        Automatically finalizes the stats if not already done.

        Returns:
            Dictionary of statistics with calculated values
        """
        if self.end_time == 0.0:
            self.finalize()

        # Basic stats
        stats = {
            "processing_time_seconds": self.processing_time,
            "memory_before_mb": self.memory_before,
            "memory_after_mb": self.memory_after,
            "memory_peak_mb": self.memory_peak,
            "memory_used_mb": self.memory_used,
            "vectors_processed": self.num_vectors_processed,
            "blocks_processed": self.num_blocks_processed,
            "optimization_count": self.optimization_count,
        }

        # Add derived statistics
        if self.processing_time > 0:
            stats["vectors_per_second"] = self.vectors_per_second
            stats["blocks_per_second"] = self.blocks_per_second

        # Add memory efficiency metrics
        if self.num_vectors_processed > 0:
            stats["memory_per_vector_kb"] = (self.memory_used * 1024) / self.num_vectors_processed

        if self.num_blocks_processed > 0:
            stats["memory_per_block_kb"] = (self.memory_used * 1024) / self.num_blocks_processed

        return stats

    def log_stats(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        """
        Log the current statistics.

        Args:
            logger: Logger instance to use
            level: Logging level (default: INFO)
        """
        stats = self.to_dict()

        # Format a readable summary
        summary = (
            f"Processing completed in {stats['processing_time_seconds']:.2f}s: "
            f"{stats['vectors_processed']} vectors, {stats['blocks_processed']} blocks. "
            f"Memory: {stats['memory_used_mb']:.2f}MB peak."
        )

        # Log the summary and detailed stats
        logger.log(level, summary)
        logger.log(level, f"Detailed stats: {json.dumps(stats, indent=2)}")

    def reset(self) -> None:
        """Reset statistics for a new processing session."""
        self.start_time = time.time()
        self.end_time = 0.0
        self.memory_before = get_memory_usage()
        self.memory_after = 0.0
        self.memory_peak = self.memory_before
        self.num_vectors_processed = 0
        self.num_blocks_processed = 0
        self.optimization_count = 0
        self._last_memory_check = time.time()


class WDBXCore:
    """
    Core WDBX engine for data processing and analysis.

    This class integrates various components of the WDBX system, providing a unified
    interface for vector operations, block management, and data analysis.

    The core is designed for high performance, memory efficiency, and thread safety.
    It provides both synchronous and asynchronous interfaces for operations.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        vector_dimension: int = VECTOR_DIMENSION,
        enable_memory_optimization: bool = True,
    ):
        """
        Initialize the WDBX core engine.

        Args:
            data_dir: Directory for storing data (defaults to config value)
            vector_dimension: Dimension of embedding vectors
            enable_memory_optimization: Whether to enable automatic memory optimization
        """
        # Initialize logging context
        log_context = LogContext(component="WDBXCore")
        with log_context:
            logger.info(f"Initializing WDBX Core (vector_dim={vector_dimension})")

            # Set up configuration
            self.config = ConfigManager.get_instance()

            # Set up data directory
            self.data_dir = Path(data_dir or get_config("data_dir"))
            self._ensure_data_directory()

            # Initialize core attributes
            self.vector_dimension = vector_dimension
            self.enable_memory_optimization = enable_memory_optimization
            self._lock = asyncio.Lock()  # For async operations

            # Initialize ML backend
            self.ml_backend = get_ml_backend(
                preferred_backend=get_config("ml_backend"), vector_dimension=vector_dimension
            )

            # Initialize memory monitoring
            if enable_memory_optimization:
                self.memory_monitor = get_memory_monitor()
                self.memory_monitor.register_optimization_callback(self.optimize_memory)
                self.memory_monitor.register_object("wdbx_core", self)

                # Start monitoring if enabled in config
                if get_config("memory_optimization_enabled", True):
                    self.memory_monitor.start_monitoring()

            # Initialize storage for blocks and vectors with initial capacity
            self.blocks: Dict[str, Block] = {}
            self.vectors: Dict[str, EmbeddingVector] = {}

            # Initialize indexes for faster lookups
            self._block_by_vector_id: Dict[str, Set[str]] = {}  # vector_id -> {block_ids}
            self._vector_by_metadata: Dict[str, Set[str]] = {}  # metadata_key:value -> {vector_ids}

            # Initialize statistics
            self.stats = ProcessingStats()

            logger.info(f"WDBX Core initialized (data_dir={self.data_dir})")

    def _ensure_data_directory(self) -> None:
        """Ensure the data directory exists."""
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created data directory: {self.data_dir}")

    async def create_vector_async(
        self,
        vector_data: Union[List[float], NDArray[np.float32]],
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None,
        bias_score: float = DEFAULT_BIAS_SCORE,
    ) -> EmbeddingVector:
        """
        Create a new embedding vector asynchronously.

        Thread-safe method that can be called from multiple async contexts.

        Args:
            vector_data: Vector data as list or numpy array
            metadata: Additional metadata for the vector
            vector_id: Optional ID for the vector (generated if not provided)
            bias_score: Bias score for the vector

        Returns:
            Created EmbeddingVector instance
        """
        async with self._lock:
            return self.create_vector(
                vector_data=vector_data,
                metadata=metadata,
                vector_id=vector_id,
                bias_score=bias_score,
            )

    def create_vector(
        self,
        vector_data: Union[List[float], NDArray[np.float32]],
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None,
        bias_score: float = DEFAULT_BIAS_SCORE,
    ) -> EmbeddingVector:
        """
        Create a new embedding vector.

        Args:
            vector_data: Vector data as list or numpy array
            metadata: Additional metadata for the vector
            vector_id: Optional ID for the vector (generated if not provided)
            bias_score: Bias score for the vector

        Returns:
            Created EmbeddingVector instance

        Raises:
            ValueError: If vector dimensions don't match
            TypeError: If vector_data is not a list or numpy array
        """
        # Convert vector to numpy array if needed
        if not isinstance(vector_data, np.ndarray):
            try:
                vector_data = np.array(vector_data, dtype=np.float32)
            except (ValueError, TypeError) as e:
                raise TypeError(f"Vector data must be convertible to numpy array: {e}") from e

        # Ensure vector is float32 for memory efficiency
        if vector_data.dtype != np.float32:
            vector_data = vector_data.astype(np.float32)

        # Validate vector dimension
        if vector_data.shape[0] != self.vector_dimension:
            raise ValueError(
                f"Vector dimension mismatch: got {vector_data.shape[0]}, "
                f"expected {self.vector_dimension}"
            )

        # Ensure metadata is a dictionary
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            raise TypeError("Metadata must be a dictionary")

        # Create embedding vector
        final_vector_id = vector_id or str(uuid.uuid4())
        embedding = EmbeddingVector(
            vector=vector_data,
            metadata=metadata,
            vector_id=final_vector_id,
            bias_score=bias_score,
        )

        # Register vector
        self.vectors[embedding.vector_id] = embedding
        self.stats.num_vectors_processed += 1

        # Update metadata index
        for key, value in metadata.items():
            # Only index string and numeric values
            if isinstance(value, (str, int, float, bool)):
                index_key = f"{key}:{value}"
                if index_key not in self._vector_by_metadata:
                    self._vector_by_metadata[index_key] = set()
                self._vector_by_metadata[index_key].add(embedding.vector_id)

        # Check if memory optimization is needed
        if self.enable_memory_optimization and self.memory_monitor.should_optimize():
            self.optimize_memory()

        # Update memory tracking
        self.stats.record_memory_usage()

        return embedding

    async def create_block_async(
        self,
        data: Dict[str, Any],
        embeddings: Optional[List[EmbeddingVector]] = None,
        block_id: Optional[str] = None,
        references: Optional[List[str]] = None,
    ) -> Block:
        """
        Create a new data block asynchronously.

        Thread-safe method that can be called from multiple async contexts.

        Args:
            data: Block data
            embeddings: Embedding vectors associated with the block
            block_id: Optional ID for the block (generated if not provided)
            references: Optional references to other blocks

        Returns:
            Created Block instance
        """
        async with self._lock:
            return self.create_block(
                data=data, embeddings=embeddings, block_id=block_id, references=references
            )

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
            embeddings: Embedding vectors associated with the block
            block_id: Optional ID for the block (generated if not provided)
            references: Optional references to other blocks

        Returns:
            Created Block instance

        Raises:
            TypeError: If data is not a dictionary
            ValueError: If any referenced block doesn't exist
        """
        # Validate input
        if not isinstance(data, dict):
            raise TypeError("Block data must be a dictionary")

        # Validate references
        if references:
            for ref in references:
                if ref not in self.blocks:
                    raise ValueError(f"Referenced block {ref} does not exist")

        # Convert embeddings to the format expected by blockchain module
        processed_embeddings = None
        if embeddings:
            processed_embeddings = [
                {
                    "vector_id": e.vector_id,
                    "vector": e.vector.tolist() if hasattr(e.vector, "tolist") else e.vector,
                    "metadata": e.metadata,
                }
                for e in embeddings
            ]

        # Create block using the blockchain manager
        if hasattr(self, "blockchain") and self.blockchain:
            block_id = self.blockchain.add_block(
                data=data, embeddings=processed_embeddings, context_references=references
            )
            # Retrieve the block from the blockchain
            block = self.blockchain.get_block(block_id)
        else:
            # Fallback to local block creation if blockchain is not available
            final_block_id = block_id or str(uuid.uuid4())
            block = Block(
                data=data,
                embeddings=embeddings or [],
                block_id=final_block_id,
                context_references=references or [],
            )

        # Register block in local storage
        self.blocks[block.block_id] = block
        self.stats.num_blocks_processed += 1

        # Update vector-to-block index
        for embedding in embeddings or []:
            vector_id = embedding.vector_id
            if vector_id not in self._block_by_vector_id:
                self._block_by_vector_id[vector_id] = set()
            self._block_by_vector_id[vector_id].add(block.block_id)

        # Check if memory optimization is needed
        if self.enable_memory_optimization and self.memory_monitor.should_optimize():
            self.optimize_memory()

        # Update memory tracking
        self.stats.record_memory_usage()

        return block

    async def find_similar_vectors_async(
        self,
        query_vector: Union[EmbeddingVector, List[float], NDArray[np.float32]],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Find vectors similar to the query vector asynchronously.

        Thread-safe method that can be called from multiple async contexts.

        Args:
            query_vector: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (vector_id, similarity_score) tuples
        """
        # This method doesn't need the lock since it's read-only
        # and the vectors dictionary is not modified during the search
        # Only thread local operations are performed
        return self.find_similar_vectors(
            query_vector=query_vector, top_k=top_k, threshold=threshold
        )

    def find_similar_vectors(
        self,
        query_vector: Union[EmbeddingVector, List[float], NDArray[np.float32]],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Find vectors similar to the query vector.

        Args:
            query_vector: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (vector_id, similarity_score) tuples sorted by similarity (highest first)

        Raises:
            ValueError: If query vector dimensions don't match
            TypeError: If query vector is of unsupported type
        """
        # Get a numpy vector from the query
        if isinstance(query_vector, EmbeddingVector):
            query_embedding = query_vector.vector
        elif isinstance(query_vector, list):
            query_embedding = np.array(query_vector, dtype=np.float32)
        elif isinstance(query_vector, np.ndarray):
            query_embedding = query_vector.astype(np.float32)
        else:
            raise TypeError(f"Unsupported query vector type: {type(query_vector)}")

        # Check dimensions
        if query_embedding.shape[0] != self.vector_dimension:
            raise ValueError(
                f"Query vector dimension mismatch: got {query_embedding.shape[0]}, "
                f"expected {self.vector_dimension}"
            )

        # Create a normalized query vector for efficient similarity computation
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            normalized_query = query_embedding / query_norm
        else:
            normalized_query = query_embedding

        # Compute similarities and find top k
        results = []
        for vector_id, embedding in self.vectors.items():
            similarity = embedding.cosine_similarity(normalized_query)

            # Only consider vectors above threshold
            if similarity >= threshold:
                results.append((vector_id, similarity))

        # Sort by similarity (descending) and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def find_vectors_by_metadata(
        self,
        metadata_filters: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> List[str]:
        """
        Find vectors that match metadata filters.

        Args:
            metadata_filters: Dictionary of metadata key-value pairs to match
            top_k: Maximum number of results to return (None for all)

        Returns:
            List of vector IDs that match all filters
        """
        # Empty filters returns nothing
        if not metadata_filters:
            return []

        # Start with all vector IDs
        result_set: Optional[Set[str]] = None

        # Apply each filter
        for key, value in metadata_filters.items():
            # Only filter by string and numeric values
            if not isinstance(value, (str, int, float, bool)):
                continue

            index_key = f"{key}:{value}"
            matching_ids = self._vector_by_metadata.get(index_key, set())

            # First filter initializes the result set
            if result_set is None:
                result_set = matching_ids.copy()
            # Subsequent filters narrow down the results (AND logic)
            else:
                result_set &= matching_ids

            # Early termination if no matches
            if not result_set:
                return []

        # No matching filters
        if result_set is None:
            return []

        # Convert to list and apply limit if needed
        result_list = list(result_set)
        if top_k is not None and top_k > 0:
            return result_list[:top_k]
        return result_list

    async def search_blocks_async(
        self,
        query: Union[str, Dict[str, Any], EmbeddingVector, NDArray[np.float32]],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[Block, float]]:
        """
        Search blocks based on a query vector or metadata asynchronously.

        Thread-safe method that can be called from multiple async contexts.

        Args:
            query: Query vector, text, or metadata
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (block, similarity_score) tuples
        """
        # This method doesn't need the lock since it's read-only
        return self.search_blocks(query=query, top_k=top_k, threshold=threshold)

    def search_blocks(
        self,
        query: Union[str, Dict[str, Any], EmbeddingVector, NDArray[np.float32]],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[Block, float]]:
        """
        Search blocks based on a query vector or metadata.

        Args:
            query: Query vector, text, or metadata
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (block, similarity_score) tuples sorted by similarity (highest first)
        """
        # Handle different query types
        if isinstance(query, dict):
            # Metadata search
            vector_ids = self.find_vectors_by_metadata(query, top_k=None)

            # Get relevant blocks
            block_scores: Dict[str, float] = {}
            for vector_id in vector_ids:
                block_ids = self._block_by_vector_id.get(vector_id, set())
                for block_id in block_ids:
                    # Simple scoring: each matching vector adds 1.0 to the block score
                    if block_id in block_scores:
                        block_scores[block_id] += 1.0
                    else:
                        block_scores[block_id] = 1.0

            # Sort blocks by score
            sorted_block_ids = sorted(
                block_scores.keys(), key=lambda bid: block_scores[bid], reverse=True
            )

            # Convert to result format
            results = [
                (self.blocks[block_id], block_scores[block_id])
                for block_id in sorted_block_ids[:top_k]
                if block_scores[block_id] >= threshold
            ]
            return results

        elif isinstance(query, str):
            # TODO: Implement text-based search
            logger.warning("Text-based search not fully implemented yet")
            return []

        else:
            # Vector similarity search
            # First find similar vectors
            similar_vectors = self.find_similar_vectors(
                query_vector=query,
                top_k=min(top_k * 3, 100),  # Get more vectors to ensure enough blocks
                threshold=threshold,
            )

            # Then find blocks containing those vectors
            block_scores: Dict[str, float] = {}
            for vector_id, score in similar_vectors:
                block_ids = self._block_by_vector_id.get(vector_id, set())
                for block_id in block_ids:
                    # Use maximum similarity score for the block
                    if block_id in block_scores:
                        block_scores[block_id] = max(block_scores[block_id], score)
                    else:
                        block_scores[block_id] = score

            # Sort blocks by score
            sorted_block_ids = sorted(
                block_scores.keys(), key=lambda bid: block_scores[bid], reverse=True
            )

            # Convert to result format
            results = [
                (self.blocks[block_id], block_scores[block_id])
                for block_id in sorted_block_ids[:top_k]
                if block_scores[block_id] >= threshold
            ]
            return results

    async def save_vector_async(self, vector: EmbeddingVector, overwrite: bool = False) -> bool:
        """
        Save vector to persistent storage asynchronously.

        Thread-safe method that can be called from multiple async contexts.

        Args:
            vector: Vector to save
            overwrite: Whether to overwrite existing vector

        Returns:
            True if saved successfully, False otherwise
        """
        async with self._lock:
            return self.save_vector(vector, overwrite)

    def save_vector(self, vector: EmbeddingVector, overwrite: bool = False) -> bool:
        """
        Save vector to persistent storage.

        Args:
            vector: Vector to save
            overwrite: Whether to overwrite existing vector

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Check if vector exists and we're not overwriting
            vector_path = self.data_dir / "vectors" / f"{vector.vector_id}.json"
            if vector_path.exists() and not overwrite:
                logger.warning(f"Vector {vector.vector_id} already exists and overwrite=False")
                return False

            # Ensure directory exists
            vector_path.parent.mkdir(parents=True, exist_ok=True)

            # Save vector as JSON
            with open(vector_path, "w") as f:
                json.dump(vector.to_dict(), f, indent=2)

            return True
        except Exception as e:
            logger.error(f"Error saving vector {vector.vector_id}: {e}")
            return False

    async def save_block_async(self, block: Block, overwrite: bool = False) -> bool:
        """
        Save block to persistent storage asynchronously.

        Thread-safe method that can be called from multiple async contexts.

        Args:
            block: Block to save
            overwrite: Whether to overwrite existing block

        Returns:
            True if saved successfully, False otherwise
        """
        async with self._lock:
            return self.save_block(block, overwrite)

    def save_block(self, block: Block, overwrite: bool = False) -> bool:
        """
        Save block to persistent storage.

        Args:
            block: Block to save
            overwrite: Whether to overwrite existing block

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Check if block exists and we're not overwriting
            block_path = self.data_dir / "blocks" / f"{block.block_id}.json"
            if block_path.exists() and not overwrite:
                logger.warning(f"Block {block.block_id} already exists and overwrite=False")
                return False

            # Ensure directory exists
            block_path.parent.mkdir(parents=True, exist_ok=True)

            # Save block as JSON
            with open(block_path, "w") as f:
                json.dump(block.to_dict(), f, indent=2)

            return True
        except Exception as e:
            logger.error(f"Error saving block {block.block_id}: {e}")
            return False

    async def load_vector_async(self, vector_id: str) -> Optional[EmbeddingVector]:
        """
        Load vector from persistent storage asynchronously.

        Thread-safe method that can be called from multiple async contexts.

        Args:
            vector_id: ID of the vector to load

        Returns:
            Loaded vector or None if not found
        """
        # Loading doesn't modify the vectors dictionary, so no lock needed
        return self.load_vector(vector_id)

    def load_vector(self, vector_id: str) -> Optional[EmbeddingVector]:
        """
        Load vector from persistent storage.

        Args:
            vector_id: ID of the vector to load

        Returns:
            Loaded vector or None if not found
        """
        # Check if vector is already in memory
        if vector_id in self.vectors:
            return self.vectors[vector_id]

        try:
            vector_path = self.data_dir / "vectors" / f"{vector_id}.json"
            if not vector_path.exists():
                return None

            # Load vector from JSON
            with open(vector_path) as f:
                vector_dict = json.load(f)

            # Create vector from dict
            vector = EmbeddingVector.from_dict(vector_dict)

            # Add to in-memory store
            self.vectors[vector_id] = vector

            # Update indices
            for key, value in vector.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    index_key = f"{key}:{value}"
                    if index_key not in self._vector_by_metadata:
                        self._vector_by_metadata[index_key] = set()
                    self._vector_by_metadata[index_key].add(vector_id)

            return vector
        except Exception as e:
            logger.error(f"Error loading vector {vector_id}: {e}")
            return None

    async def load_block_async(self, block_id: str) -> Optional[Block]:
        """
        Load block from persistent storage asynchronously.

        Thread-safe method that can be called from multiple async contexts.

        Args:
            block_id: ID of the block to load

        Returns:
            Loaded block or None if not found
        """
        # This operation doesn't modify shared state, so lock not required
        return self.load_block(block_id)

    def load_block(self, block_id: str) -> Optional[Block]:
        """
        Load block from persistent storage.

        Args:
            block_id: ID of the block to load

        Returns:
            Loaded block or None if not found
        """
        # Check if block is already in memory
        if block_id in self.blocks:
            return self.blocks[block_id]

        try:
            block_path = self.data_dir / "blocks" / f"{block_id}.json"
            if not block_path.exists():
                return None

            # Load block from JSON
            with open(block_path) as f:
                block_dict = json.load(f)

            # Process embeddings from the block
            embeddings = []
            for embedding_dict in block_dict.get("embeddings", []):
                embedding_id = embedding_dict.get("vector_id")

                # Try to get the embedding if it's already loaded
                embedding = self.load_vector(embedding_id)
                if embedding is None:
                    # Create the embedding if it doesn't exist
                    embedding = EmbeddingVector.from_dict(embedding_dict)
                    self.vectors[embedding_id] = embedding

                    # Update metadata index
                    for key, value in embedding.metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            index_key = f"{key}:{value}"
                            if index_key not in self._vector_by_metadata:
                                self._vector_by_metadata[index_key] = set()
                            self._vector_by_metadata[index_key].add(embedding_id)

                embeddings.append(embedding)

            # Update the block dict with loaded embeddings
            block_dict["embeddings"] = embeddings

            # Create block
            block = Block.from_dict(block_dict)

            # Add to in-memory store and update indices
            self.blocks[block_id] = block
            for embedding in block.embeddings:
                vector_id = embedding.vector_id
                if vector_id not in self._block_by_vector_id:
                    self._block_by_vector_id[vector_id] = set()
                self._block_by_vector_id[vector_id].add(block_id)

            return block
        except Exception as e:
            logger.error(f"Error loading block {block_id}: {e}")
            return None

    def optimize_memory(self) -> None:
        """
        Optimize memory usage of vectors and blocks.

        This method frees memory by optimizing vector storage and
        potentially removing unused vectors from memory.
        """
        self.stats.optimization_count += 1
        logger.debug(f"Running memory optimization (count={self.stats.optimization_count})")

        try:
            # Optimize vector storage
            for vector in self.vectors.values():
                vector.optimize_memory()

            # Optimize block storage
            for block in self.blocks.values():
                block.optimize_memory()

            # Invoke system-wide memory optimization
            if optimize_memory:
                optimize_memory()
        except Exception as e:
            logger.error(f"Error during memory optimization: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the WDBX core.

        Returns:
            Dictionary of statistics
        """
        stats_dict = self.stats.to_dict()
        stats_dict.update(
            {
                "total_vectors": len(self.vectors),
                "total_blocks": len(self.blocks),
                "vector_dimension": self.vector_dimension,
            }
        )
        return stats_dict

    def clear(self) -> None:
        """
        Clear all in-memory data.

        This method removes all vectors and blocks from memory,
        but does not delete persisted data.
        """
        self.vectors.clear()
        self.blocks.clear()
        self._block_by_vector_id.clear()
        self._vector_by_metadata.clear()

        # Reset statistics
        self.stats.reset()

        logger.info("Cleared all in-memory data")

    def shutdown(self) -> None:
        """
        Shut down the WDBX core.

        This method performs cleanup operations before shutdown,
        such as saving unsaved data, stopping background processes,
        and releasing resources.
        """
        logger.info("Shutting down WDBX Core...")

        try:
            # Stop memory monitoring
            if self.enable_memory_optimization and hasattr(self, "memory_monitor"):
                self.memory_monitor.stop_monitoring()

            # Final memory optimization
            self.optimize_memory()

            # Log final statistics
            self.stats.finalize()
            logger.info(f"Final statistics: {json.dumps(self.stats.to_dict(), indent=2)}")

            logger.info("WDBX Core shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


async def get_wdbx_core_async(create_if_missing: bool = True) -> Optional[WDBXCore]:
    """
    Get or create a WDBXCore instance asynchronously.

    This function ensures only one instance exists across async contexts.

    Args:
        create_if_missing: Whether to create an instance if none exists

    Returns:
        WDBXCore instance or None
    """
    global _wdbx_core_instance

    if _wdbx_core_instance is None and create_if_missing:
        _wdbx_core_instance = WDBXCore()

    return _wdbx_core_instance
