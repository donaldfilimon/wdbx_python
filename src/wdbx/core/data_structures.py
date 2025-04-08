# wdbx/data_structures.py
"""
Core data structures for the WDBX system.

This module defines the fundamental data structures used throughout the WDBX
system, including embedding vectors and data blocks. These structures are
designed for efficient memory usage, serialization, and vector operations.
"""
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Union, cast

import numpy as np
from numpy.typing import NDArray

from ..utils.memory_utils import register_for_memory_optimization
from .constants import (
    DEFAULT_BIAS_SCORE,
)


@dataclass(frozen=False)
class EmbeddingVector:
    """
    Vector embedding with associated metadata.

    This class represents a vector embedding with its associated metadata,
    providing methods for vector operations and serialization. The implementation
    focuses on memory efficiency and performance for similarity computations.

    Attributes:
        vector: The embedding vector as a NumPy array or list of floats
        metadata: Associated metadata dictionary
        vector_id: Unique identifier for the vector
        timestamp: Creation time of the vector
        bias_score: Bias score for the vector
    """

    vector: Union[NDArray[np.float32], List[float]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    bias_score: float = DEFAULT_BIAS_SCORE
    _norm: Optional[float] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize vector after creation."""
        # Ensure vector is a numpy array with correct dtype
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.array(self.vector, dtype=np.float32)
        elif self.vector.dtype != np.float32:
            self.vector = self.vector.astype(np.float32)

        # Pre-compute vector norm for faster similarity calculations
        self._compute_norm()

        # Register for memory optimization
        register_for_memory_optimization(f"vector_{self.vector_id}", self)

    def _compute_norm(self) -> None:
        """Compute and cache the vector norm."""
        self._norm = float(np.linalg.norm(self.vector))

    def normalize(self) -> "EmbeddingVector":
        """
        Normalize the vector to unit length.

        Returns:
            Self with normalized vector
        """
        if self._norm is None:
            self._compute_norm()

        norm = cast(float, self._norm)
        if norm > 0:
            self.vector = self.vector / norm
            self._norm = 1.0  # After normalization, norm is 1
        return self

    def dot_product(self, other: Union["EmbeddingVector", NDArray[np.float32]]) -> float:
        """
        Calculate the dot product with another vector.

        Args:
            other: Another vector to calculate dot product with

        Returns:
            Dot product value
        """
        if isinstance(other, EmbeddingVector):
            return float(np.dot(self.vector, other.vector))
        return float(np.dot(self.vector, other))

    def cosine_similarity(self, other: Union["EmbeddingVector", NDArray[np.float32]]) -> float:
        """
        Calculate cosine similarity with another vector.

        The cosine similarity is measured by the cosine of the angle between two vectors,
        ranging from -1 (exactly opposite) to 1 (exactly the same).

        Args:
            other: Another vector to calculate similarity with

        Returns:
            Cosine similarity value between -1 and 1
        """
        other_vector = other.vector if isinstance(other, EmbeddingVector) else other
        other_norm = (
            other._norm
            if isinstance(other, EmbeddingVector) and other._norm is not None
            else np.linalg.norm(other_vector)
        )

        # Ensure self norm is computed
        if self._norm is None:
            self._compute_norm()

        norm_self = cast(float, self._norm)

        # Handle zero vectors to avoid division by zero
        if norm_self == 0 or other_norm == 0:
            return 0.0

        # Calculate cosine similarity
        return float(np.dot(self.vector, other_vector) / (norm_self * other_norm))

    def euclidean_distance(self, other: Union["EmbeddingVector", NDArray[np.float32]]) -> float:
        """
        Calculate Euclidean distance to another vector.

        The Euclidean distance measures the straight-line distance between two vectors
        in the embedding space.

        Args:
            other: Another vector to calculate distance to

        Returns:
            Euclidean distance value (non-negative)
        """
        other_vector = other.vector if isinstance(other, EmbeddingVector) else other
        return float(np.linalg.norm(self.vector - other_vector))

    def manhattan_distance(self, other: Union["EmbeddingVector", NDArray[np.float32]]) -> float:
        """
        Calculate Manhattan (L1) distance to another vector.

        The Manhattan distance measures the sum of absolute differences between coordinates.

        Args:
            other: Another vector to calculate distance to

        Returns:
            Manhattan distance value (non-negative)
        """
        other_vector = other.vector if isinstance(other, EmbeddingVector) else other
        return float(np.sum(np.abs(self.vector - other_vector)))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "vector_id": self.vector_id,
            "vector": self.vector.tolist(),
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "bias_score": self.bias_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingVector":
        """
        Create from dictionary representation.

        Args:
            data: Dictionary data

        Returns:
            Created EmbeddingVector instance
        """
        return cls(
            vector=np.array(data["vector"], dtype=np.float32),
            metadata=data.get("metadata", {}),
            vector_id=data["vector_id"],
            timestamp=data.get("timestamp", time.time()),
            bias_score=data.get("bias_score", DEFAULT_BIAS_SCORE),
        )

    def optimize_memory(self) -> None:
        """
        Optimize memory usage.

        Ensures the vector uses the most memory-efficient representation
        and is contiguous in memory.
        """
        # Ensure the vector is stored as float32 (not float64)
        if self.vector.dtype != np.float32:
            self.vector = self.vector.astype(np.float32)

        # Ensure the vector is contiguous in memory
        if not self.vector.flags.c_contiguous:
            self.vector = np.ascontiguousarray(self.vector)

        # Recompute norm after optimization
        self._compute_norm()

    def __eq__(self, other: object) -> bool:
        """Check if two embedding vectors are equal."""
        if not isinstance(other, EmbeddingVector):
            return False
        return (
            self.vector_id == other.vector_id
            and np.array_equal(self.vector, other.vector)
            and self.timestamp == other.timestamp
            and self.bias_score == other.bias_score
        )

    def __hash__(self) -> int:
        """Generate hash for embedding vector."""
        return hash(self.vector_id)


@dataclass
class Block:
    """
    Data block with associated embeddings.

    This class represents a block of data with its associated embedding vectors,
    providing methods for serialization, validation, and cryptographic security.

    Attributes:
        data: The main data content of the block
        embeddings: Associated embedding vectors
        block_id: Unique identifier for the block
        timestamp: Creation time of the block
        previous_block_hash: Hash of the previous block in a chain
        hash: Cryptographic hash of the block
        nonce: Nonce value for blockchain consensus
        context_references: References to related blocks
        signature: Cryptographic signature
    """

    data: Dict[str, Any]
    embeddings: List[EmbeddingVector] = field(default_factory=list)
    block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    previous_block_hash: Optional[str] = None
    hash: Optional[str] = None
    nonce: int = 0
    context_references: List[str] = field(default_factory=list)
    signature: Optional[str] = None
    _embedding_ids: Set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        """Initialize block after creation."""
        # Create set of embedding IDs for faster lookups
        self._embedding_ids = {e.vector_id for e in self.embeddings}

        # Calculate hash if not provided
        if self.hash is None:
            self.calculate_hash()

        # Register for memory optimization
        register_for_memory_optimization(f"block_{self.block_id}", self)

    @lru_cache(maxsize=1)
    def _get_block_string(self) -> str:
        """Get string representation of block data for hashing."""
        return json.dumps(
            {
                "block_id": self.block_id,
                "data": self.data,
                "timestamp": self.timestamp,
                "previous_block_hash": self.previous_block_hash,
                "nonce": self.nonce,
                "context_references": self.context_references,
                "embedding_ids": sorted([e.vector_id for e in self.embeddings]),
            },
            sort_keys=True,
        )

    def calculate_hash(self) -> str:
        """
        Calculate the hash of the block.

        Returns:
            Hash string
        """
        # Calculate the hash using block string representation
        block_string = self._get_block_string()
        self.hash = hashlib.sha256(block_string.encode()).hexdigest()
        return self.hash

    def compute_hash(self) -> str:
        """
        Compute the hash of the block.

        This is an alias for calculate_hash() to maintain compatibility.

        Returns:
            Hash string
        """
        return self.calculate_hash()

    def is_valid(self) -> bool:
        """
        Check if the block is valid.

        Returns:
            True if valid, False otherwise
        """
        # Recalculate the hash
        calculated_hash = self.calculate_hash()

        # Check if the hash matches
        return calculated_hash == self.hash

    def validate(self) -> bool:
        """
        Validate the block's hash.

        Returns:
            True if the block's hash is valid, False otherwise
        """
        return self.is_valid()

    def contains_embedding(self, vector_id: str) -> bool:
        """
        Check if block contains an embedding with the given ID.

        Args:
            vector_id: Vector ID to check

        Returns:
            True if the block contains the embedding, False otherwise
        """
        return vector_id in self._embedding_ids

    def add_embedding(self, embedding: EmbeddingVector) -> None:
        """
        Add an embedding to the block.

        Args:
            embedding: Embedding vector to add
        """
        if embedding.vector_id not in self._embedding_ids:
            self.embeddings.append(embedding)
            self._embedding_ids.add(embedding.vector_id)
            # Invalidate hash
            self.hash = None
            # Clear lru_cache
            self._get_block_string.cache_clear()

    def remove_embedding(self, vector_id: str) -> bool:
        """
        Remove an embedding from the block.

        Args:
            vector_id: ID of the embedding to remove

        Returns:
            True if found and removed, False otherwise
        """
        if vector_id in self._embedding_ids:
            self.embeddings = [e for e in self.embeddings if e.vector_id != vector_id]
            self._embedding_ids.remove(vector_id)
            # Invalidate hash
            self.hash = None
            # Clear lru_cache
            self._get_block_string.cache_clear()
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "block_id": self.block_id,
            "data": self.data,
            "embeddings": [e.to_dict() for e in self.embeddings],
            "timestamp": self.timestamp,
            "previous_block_hash": self.previous_block_hash,
            "hash": self.hash,
            "nonce": self.nonce,
            "context_references": self.context_references,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Block":
        """
        Create from dictionary representation.

        Args:
            data: Dictionary data

        Returns:
            Created Block instance
        """
        # Extract embeddings
        embeddings = [
            EmbeddingVector.from_dict(e) if isinstance(e, dict) else e
            for e in data.get("embeddings", [])
        ]

        return cls(
            data=data["data"],
            embeddings=embeddings,
            block_id=data["block_id"],
            timestamp=data.get("timestamp", time.time()),
            previous_block_hash=data.get("previous_block_hash"),
            hash=data.get("hash"),
            nonce=data.get("nonce", 0),
            context_references=data.get("context_references", []),
            signature=data.get("signature"),
        )

    def optimize_memory(self) -> None:
        """
        Optimize memory usage.

        Optimizes memory usage of the block and its embeddings.
        """
        # Optimize embeddings
        for embedding in self.embeddings:
            embedding.optimize_memory()

    def __eq__(self, other: object) -> bool:
        """Check if two blocks are equal based on their IDs and hashes."""
        if not isinstance(other, Block):
            return False
        return self.block_id == other.block_id and self.hash == other.hash

    def __hash__(self) -> int:
        """Generate hash for the block."""
        return hash(self.block_id)


@dataclass
class ShardInfo:
    """
    Information about a data shard.

    Attributes:
        shard_id: Unique identifier for the shard.
        node_id: ID of the node hosting the shard.
        start_key: Start key range for the shard.
        end_key: End key range for the shard.
        size_bytes: Current size of the shard in bytes.
        item_count: Number of items in the shard.
        status: Current status of the shard (e.g., "active", "offline", "removed").
        load: Current load factor of the shard (0.0 to 1.0).
        block_count: Number of blocks stored in the shard.
        last_updated: Timestamp of the last update to this shard info.
    """

    shard_id: str
    node_id: str
    start_key: Optional[str] = None
    end_key: Optional[str] = None
    size_bytes: int = 0
    item_count: int = 0
    status: str = "active"
    load: float = 0.0
    block_count: int = 0
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "shard_id": self.shard_id,
            "node_id": self.node_id,
            "start_key": self.start_key,
            "end_key": self.end_key,
            "size_bytes": self.size_bytes,
            "item_count": self.item_count,
            "status": self.status,
            "load": self.load,
            "block_count": self.block_count,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShardInfo":
        """Create from dictionary representation."""
        return cls(
            shard_id=data["shard_id"],
            node_id=data["node_id"],
            start_key=data.get("start_key"),
            end_key=data.get("end_key"),
            size_bytes=data.get("size_bytes", 0),
            item_count=data.get("item_count", 0),
            status=data.get("status", "active"),
            load=data.get("load", 0.0),
            block_count=data.get("block_count", 0),
            last_updated=data.get("last_updated", time.time()),
        )


@dataclass
class TransactionLogEntry:
    """
    Represents an entry in the transaction log.

    Attributes:
        transaction_id: Unique ID of the transaction.
        operation: Type of operation (e.g., 'write', 'delete').
        key: Key affected by the operation.
        value: Value associated with the operation (optional).
        timestamp: Time of the operation.
    """

    transaction_id: str
    operation: str
    key: str
    value: Optional[Any] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "transaction_id": self.transaction_id,
            "operation": self.operation,
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransactionLogEntry":
        """Create from dictionary representation."""
        return cls(
            transaction_id=data["transaction_id"],
            operation=data["operation"],
            key=data["key"],
            value=data.get("value"),
            timestamp=data.get("timestamp", time.time()),
        )

    def __eq__(self, other: object) -> bool:
        """Check if two transaction log entries are equal."""
        if not isinstance(other, TransactionLogEntry):
            return False
        return (
            self.transaction_id == other.transaction_id
            and self.operation == other.operation
            and self.key == other.key
            and self.timestamp == other.timestamp
        )
