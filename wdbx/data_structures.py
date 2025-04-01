# wdbx/data_structures.py
import time
import uuid
import json
import pickle
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union, ClassVar
import numpy as np
from wdbx.constants import logger
from config import BLOCKCHAIN_DIFFICULTY

@dataclass
class EmbeddingVector:
    """Represents a high-dimensional embedding vector with metadata."""
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.array(self.vector, dtype=np.float32)
        if len(self.vector.shape) != 1:
            raise ValueError(f"Expected 1D vector, got shape {self.vector.shape}")

    def normalize(self) -> np.ndarray:
        """Normalize vector to unit length."""
        norm = np.linalg.norm(self.vector)
        return self.vector / norm if norm >= 1e-10 else np.zeros_like(self.vector)

    def get_normalized_vector(self) -> 'EmbeddingVector':
        """Return a new EmbeddingVector with normalized vector."""
        return EmbeddingVector(
            vector=self.normalize(),
            metadata=self.metadata.copy(),
            timestamp=self.timestamp,
            id=self.id
        )

    def dot_product(self, other: 'EmbeddingVector') -> float:
        """Calculate dot product with another vector."""
        return np.dot(self.vector, other.vector)

    def cosine_similarity(self, other: 'EmbeddingVector') -> float:
        """Calculate cosine similarity with another vector."""
        return np.dot(self.normalize(), other.normalize())

    def euclidean_distance(self, other: 'EmbeddingVector') -> float:
        """Calculate Euclidean distance to another vector."""
        return np.linalg.norm(self.vector - other.vector)

    def serialize(self) -> bytes:
        """Serialize to bytes for storage."""
        data = {
            "vector": self.vector.tolist(),
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "id": self.id
        }
        return pickle.dumps(data)

    @classmethod
    def deserialize(cls, data: bytes) -> "EmbeddingVector":
        """Deserialize from bytes."""
        try:
            loaded = pickle.loads(data)
            return cls(
                vector=np.array(loaded["vector"], dtype=np.float32),
                metadata=loaded["metadata"],
                timestamp=loaded["timestamp"],
                id=loaded.get("id", str(uuid.uuid4()))
            )
        except Exception as e:
            logger.error(f"Error deserializing EmbeddingVector: {e}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "vector": self.vector.tolist(),
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "id": self.id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingVector":
        """Create from dictionary representation."""
        return cls(
            vector=np.array(data["vector"], dtype=np.float32),
            metadata=data["metadata"],
            timestamp=data["timestamp"],
            id=data.get("id", str(uuid.uuid4()))
        )

    def __len__(self) -> int:
        return len(self.vector)


@dataclass
class Block:
    """
    Data block that groups embedding vectors and supports a blockchain-like integrity mechanism.

    Attributes:
        id: Unique identifier for the block
        timestamp: Time of block creation
        data: Dictionary of associated data
        embeddings: List of embedding vectors
        previous_hash: Hash of previous block in chain
        context_references: List of related block IDs
        hash: Current block hash
        nonce: Value used in mining process
    """
    id: str
    timestamp: float
    data: Dict[str, Any]
    embeddings: List[EmbeddingVector]
    previous_hash: str = ""
    context_references: List[str] = field(default_factory=list)
    hash: str = ""
    nonce: int = 0
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.hash:
            self.mine_block()

    def calculate_hash(self) -> str:
        """Calculate the cryptographic hash of the block contents."""
        # Create a stable representation of block data
        block_content = (
            f"{self.id}{self.timestamp}{json.dumps(self.data, sort_keys=True)}"
            f"{self.previous_hash}{self.nonce}"
        )
        # Include embedding hashes
        for embedding in self.embeddings:
            vector_hash = hashlib.sha256(embedding.vector.tobytes()).hexdigest()
            block_content += vector_hash
        # Include context references
        block_content += "".join(sorted(self.context_references))
        return hashlib.sha256(block_content.encode()).hexdigest()

    def mine_block(self) -> None:
        """Mine the block by finding a hash with the required difficulty."""
        target = '0' * BLOCKCHAIN_DIFFICULTY
        logger.debug(f"Mining block {self.id} with difficulty {BLOCKCHAIN_DIFFICULTY}")
        start_time = time.time()

        while True:
            self.hash = self.calculate_hash()
            if self.hash.startswith(target):
                break
            self.nonce += 1

            # Optional: Add a timeout or progress report
            if self.nonce % 100000 == 0:
                elapsed = time.time() - start_time
                logger.debug(f"Mining in progress: tried {self.nonce} nonces in {elapsed:.2f}s")

        elapsed = time.time() - start_time
        logger.info(f"Block {self.id} mined in {elapsed:.2f}s with nonce {self.nonce}")

    def validate(self) -> bool:
        """Verify the block's hash is valid."""
        return self.hash == self.calculate_hash()

    def add_embedding(self, embedding: EmbeddingVector) -> None:
        """Add a new embedding to the block (requires re-mining)."""
        self.embeddings.append(embedding)
        self.hash = ""  # Invalidate current hash
        self.mine_block()  # Re-mine the block

    def add_context_reference(self, block_id: str) -> None:
        """Add a reference to a related block (requires re-mining)."""
        if block_id not in self.context_references:
            self.context_references.append(block_id)
            self.hash = ""  # Invalidate current hash
            self.mine_block()  # Re-mine the block

    def serialize(self) -> bytes:
        """Serialize the block to bytes."""
        embeddings_serialized = [e.serialize() for e in self.embeddings]
        data = {
            "id": self.id,
            "timestamp": self.timestamp,
            "data": self.data,
            "embeddings_serialized": embeddings_serialized,
            "previous_hash": self.previous_hash,
            "context_references": self.context_references,
            "hash": self.hash,
            "nonce": self.nonce,
            "created_at": self.created_at
        }
        return pickle.dumps(data)

    @classmethod
    def deserialize(cls, data: bytes) -> "Block":
        """Deserialize the block from bytes."""
        try:
            loaded = pickle.loads(data)
            embeddings = [EmbeddingVector.deserialize(e) for e in loaded["embeddings_serialized"]]
            block = cls(
                id=loaded["id"],
                timestamp=loaded["timestamp"],
                data=loaded["data"],
                embeddings=embeddings,
                previous_hash=loaded["previous_hash"],
                context_references=loaded["context_references"],
                hash=loaded["hash"],
                nonce=loaded["nonce"],
                created_at=loaded.get("created_at", loaded["timestamp"])
            )
            if not block.validate():
                logger.warning(f"Block {block.id} failed validation after deserialization")
            return block
        except Exception as e:
            logger.error(f"Error deserializing Block: {e}")
            raise

    def to_dict(self, include_embeddings: bool = True) -> Dict[str, Any]:
        """Convert block to dictionary representation."""
        result = {
            "id": self.id,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "context_references": self.context_references,
            "hash": self.hash,
            "nonce": self.nonce,
            "created_at": self.created_at,
            "embedding_count": len(self.embeddings)
        }
        if include_embeddings:
            result["embeddings"] = [e.to_dict() for e in self.embeddings]
        return result

    def get_size(self) -> int:
        """Calculate approximate memory size of block in bytes."""
        return len(self.serialize())


@dataclass
class ShardInfo:
    """
    Holds information about a storage shard.

    Attributes:
        shard_id: Unique identifier for the shard
        host: Hostname or IP address
        port: Port number
        status: Current operational status
        load: Current load factor (0.0-1.0)
        block_count: Number of blocks stored
        last_updated: Timestamp of last status update
        capacity: Maximum storage capacity in bytes
        used_space: Currently used storage in bytes
    """
    shard_id: int
    host: str
    port: int
    status: str = "active"
    load: float = 0.0
    block_count: int = 0
    last_updated: float = field(default_factory=time.time)
    capacity: int = 1_000_000_000  # 1GB default capacity
    used_space: int = 0

    # Valid status values
    VALID_STATUSES: ClassVar[List[str]] = ["active", "inactive", "maintenance", "failed", "syncing"]

    def __post_init__(self) -> None:
        # Validate status
        if self.status not in self.VALID_STATUSES:
            logger.warning(f"Invalid shard status '{self.status}', defaulting to 'active'")
            self.status = "active"

        # Ensure load is between 0 and 1
        self.load = max(0.0, min(1.0, self.load))

    def calculate_latency(self, retrieval_size: int) -> float:
        """
        Calculate estimated latency for retrieving data from this shard.

        Args:
            retrieval_size: Size of data to be retrieved in bytes

        Returns:
            Estimated latency in seconds
        """
        # Using a more sophisticated model:
        # 1. Base network latency
        # 2. Load-dependent processing time
        # 3. Size-dependent transfer time

        base_latency = 0.05  # 50ms base network overhead
        processing_factor = 0.1 * self.load  # Higher load = more processing time
        transfer_factor = 0.00001 * retrieval_size  # Transfer time based on size

        return base_latency + processing_factor + transfer_factor

    def is_available(self) -> bool:
        """Check if the shard is available for operations."""
        return self.status == "active" and self.load < 0.95

    def has_capacity(self, size_needed: int) -> bool:
        """Check if the shard has enough remaining capacity."""
        return (self.used_space + size_needed) <= self.capacity

    def update_load(self, new_load: float) -> None:
        """Update the load factor with validation."""
        self.load = max(0.0, min(1.0, new_load))
        self.last_updated = time.time()

    def update_block_count(self, delta: int = 1) -> None:
        """Update the block count (add or remove blocks)."""
        self.block_count = max(0, self.block_count + delta)
        self.last_updated = time.time()

    def update_used_space(self, space_bytes: int) -> None:
        """Update the used space value."""
        self.used_space = max(0, space_bytes)
        self.last_updated = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "shard_id": self.shard_id,
            "host": self.host,
            "port": self.port,
            "status": self.status,
            "load": self.load,
            "block_count": self.block_count,
            "last_updated": self.last_updated,
            "capacity": self.capacity,
            "used_space": self.used_space,
            "free_space": self.capacity - self.used_space,
            "utilization_percent": (self.used_space / self.capacity * 100) if self.capacity > 0 else 0
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShardInfo":
        """Create from dictionary representation."""
        return cls(
            shard_id=data["shard_id"],
            host=data["host"],
            port=data["port"],
            status=data.get("status", "active"),
            load=data.get("load", 0.0),
            block_count=data.get("block_count", 0),
            last_updated=data.get("last_updated", time.time()),
            capacity=data.get("capacity", 1_000_000_000),
            used_space=data.get("used_space", 0)
        )
