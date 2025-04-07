# wdbx/data_structures.py
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray

from .constants import (BLOCKCHAIN_DIFFICULTY, VectorType, MetadataType, 
                       VectorId, logger, DEFAULT_BIAS_SCORE, VECTOR_DIMENSION)
from ..utils.memory_utils import register_for_memory_optimization


@dataclass
class EmbeddingVector:
    """
    Vector embedding with associated metadata.
    
    This class represents a vector embedding with its associated metadata,
    providing methods for vector operations and serialization.
    """
    
    vector: Union[NDArray[np.float32], List[float]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    bias_score: float = DEFAULT_BIAS_SCORE
    
    def __post_init__(self) -> None:
        """Initialize vector after creation."""
        # Ensure vector is a numpy array with correct dtype
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.array(self.vector, dtype=np.float32)
        elif self.vector.dtype != np.float32:
            self.vector = self.vector.astype(np.float32)
        
        # Register for memory optimization
        register_for_memory_optimization(f"vector_{self.vector_id}", self)
    
    def normalize(self) -> 'EmbeddingVector':
        """
        Normalize the vector to unit length.
        
        Returns:
            Self with normalized vector
        """
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm
        return self
    
    def dot_product(self, other: Union['EmbeddingVector', NDArray[np.float32]]) -> float:
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
    
    def cosine_similarity(self, other: Union['EmbeddingVector', NDArray[np.float32]]) -> float:
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
        
        # Calculate the norms of the vectors
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other_vector)
        
        # Handle zero vectors to avoid division by zero
        if norm_self == 0 or norm_other == 0:
            return 0.0
        
        # Calculate cosine similarity
        return float(np.dot(self.vector, other_vector) / (norm_self * norm_other))
    
    def euclidean_distance(self, other: Union['EmbeddingVector', NDArray[np.float32]]) -> float:
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
            "bias_score": self.bias_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingVector':
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
            bias_score=data.get("bias_score", DEFAULT_BIAS_SCORE)
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


@dataclass
class Block:
    """
    Data block with associated embeddings.
    
    This class represents a block of data with its associated embedding vectors,
    providing methods for serialization and validation.
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
    
    def __post_init__(self) -> None:
        """Initialize block after creation."""
        if self.hash is None:
            self.calculate_hash()
        
        # Register for memory optimization
        register_for_memory_optimization(f"block_{self.block_id}", self)
    
    def calculate_hash(self) -> str:
        """
        Calculate the hash of the block.
        
        Returns:
            Hash string
        """
        import hashlib
        
        # Create a string representation of the block data
        block_string = json.dumps({
            "block_id": self.block_id,
            "data": self.data,
            "timestamp": self.timestamp,
            "previous_block_hash": self.previous_block_hash,
            "nonce": self.nonce,
            "context_references": self.context_references,
            "embedding_ids": [e.vector_id for e in self.embeddings]
        }, sort_keys=True)
        
        # Calculate the hash
        self.hash = hashlib.sha256(block_string.encode()).hexdigest()
        return self.hash
    
    def is_valid(self) -> bool:
        """
        Check if the block is valid.
        
        Returns:
            True if valid, False otherwise
        """
        # Recalculate the hash
        calculated_hash = self.calculate_hash()
        
        # Check if the hash matches
        if calculated_hash != self.hash:
            return False
        
        return True
    
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
            "signature": self.signature
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Block':
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
            signature=data.get("signature")
        )
    
    def optimize_memory(self) -> None:
        """
        Optimize memory usage.
        
        Optimizes memory usage of the block and its embeddings.
        """
        # Optimize embeddings
        for embedding in self.embeddings:
            embedding.optimize_memory()


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
    """
    shard_id: str
    node_id: str
    start_key: Optional[str] = None
    end_key: Optional[str] = None
    size_bytes: int = 0
    item_count: int = 0


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
