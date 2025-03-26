# wdbx/data_structures.py
import time
import uuid
import json
import pickle
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np
from wdbx.constants import logger
from config import BLOCKCHAIN_DIFFICULTY

@dataclass
class EmbeddingVector:
    """Represents a high-dimensional embedding vector with metadata."""
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self) -> None:
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.array(self.vector, dtype=np.float32)
        if len(self.vector.shape) != 1:
            raise ValueError(f"Expected 1D vector, got shape {self.vector.shape}")
    
    def normalize(self) -> np.ndarray:
        norm = np.linalg.norm(self.vector)
        return self.vector / norm if norm >= 1e-10 else np.zeros_like(self.vector)
    
    def serialize(self) -> bytes:
        data = {
            "vector": self.vector.tolist(),
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
        return pickle.dumps(data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> "EmbeddingVector":
        loaded = pickle.loads(data)
        return cls(
            vector=np.array(loaded["vector"], dtype=np.float32),
            metadata=loaded["metadata"],
            timestamp=loaded["timestamp"]
        )
    
    def __len__(self) -> int:
        return len(self.vector)


@dataclass
class Block:
    """
    Data block that groups embedding vectors and supports a blockchain-like integrity mechanism.
    """
    id: str
    timestamp: float
    data: Dict[str, Any]
    embeddings: List[EmbeddingVector]
    previous_hash: str = ""
    context_references: List[str] = field(default_factory=list)
    hash: str = ""
    nonce: int = 0
    
    def __post_init__(self) -> None:
        if not self.hash:
            self.mine_block()
    
    def calculate_hash(self) -> str:
        block_content = (
            f"{self.id}{self.timestamp}{json.dumps(self.data, sort_keys=True)}"
            f"{self.previous_hash}{self.nonce}"
        )
        # Include embedding hashes
        for embedding in self.embeddings:
            vector_hash = hashlib.sha256(embedding.vector.tobytes()).hexdigest()
            block_content += vector_hash
        # Include context references
        block_content += "".join(self.context_references)
        return hashlib.sha256(block_content.encode()).hexdigest()
    
    def mine_block(self) -> None:
        target = '0' * BLOCKCHAIN_DIFFICULTY
        while True:
            self.hash = self.calculate_hash()
            if self.hash.startswith(target):
                break
            self.nonce += 1
    
    def validate(self) -> bool:
        return self.hash == self.calculate_hash()
    
    def serialize(self) -> bytes:
        embeddings_serialized = [e.serialize() for e in self.embeddings]
        data = {
            "id": self.id,
            "timestamp": self.timestamp,
            "data": self.data,
            "embeddings_serialized": embeddings_serialized,
            "previous_hash": self.previous_hash,
            "context_references": self.context_references,
            "hash": self.hash,
            "nonce": self.nonce
        }
        return pickle.dumps(data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> "Block":
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
            nonce=loaded["nonce"]
        )
        if not block.validate():
            logger.warning(f"Block {block.id} failed validation after deserialization")
        return block


@dataclass
class ShardInfo:
    """Holds information about a storage shard."""
    shard_id: int
    host: str
    port: int
    status: str = "active"
    load: float = 0.0
    block_count: int = 0
    last_updated: float = field(default_factory=time.time)
    
    def calculate_latency(self, retrieval_size: int) -> float:
        """Calculate estimated latency for retrieving data from this shard"""
        # Using a base latency plus load-dependent factor
        base_latency = 0.05  # 50ms base network overhead
        return base_latency + (retrieval_size * self.load) / 8  # assuming 8 shards as default
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "host": self.host,
            "port": self.port,
            "status": self.status,
            "load": self.load,
            "block_count": self.block_count,
            "last_updated": self.last_updated
        }
