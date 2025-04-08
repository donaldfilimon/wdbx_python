"""
Data structures for WDBX.

This module defines the core data structures used in WDBX, including
vectors, blocks, and other fundamental objects.
"""

import hashlib
import json
import time
import uuid
from typing import Any, Dict, List, Union

import numpy as np


class EmbeddingVector:
    """
    Represents a vector embedding with associated metadata.

    Attributes:
        vector: The actual vector data
        metadata: Associated metadata for the vector
        id: Unique identifier for the vector
    """

    def __init__(
        self,
        vector: Union[List[float], np.ndarray],
        metadata: Dict[str, Any] = None,
        vector_id: str = None,
    ):
        """
        Initialize an embedding vector.

        Args:
            vector: Vector data
            metadata: Associated metadata
            vector_id: Unique identifier (generated if not provided)
        """
        if isinstance(vector, list):
            self.vector = np.array(vector, dtype=np.float32)
        else:
            self.vector = vector

        self.metadata = metadata or {"timestamp": time.time()}
        self.id = vector_id or str(uuid.uuid4())

    def __len__(self) -> int:
        """Get the dimension of the vector."""
        return len(self.vector)

    def serialize(self) -> bytes:
        """
        Serialize the vector to bytes.

        Returns:
            Serialized representation of the vector
        """
        data = {"id": self.id, "vector": self.vector.tolist(), "metadata": self.metadata}
        return json.dumps(data).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "EmbeddingVector":
        """
        Deserialize bytes to an EmbeddingVector.

        Args:
            data: Serialized vector data

        Returns:
            Deserialized EmbeddingVector
        """
        parsed = json.loads(data.decode("utf-8"))
        return cls(vector=parsed["vector"], metadata=parsed["metadata"], vector_id=parsed["id"])


class Block:
    """
    Represents a block in the blockchains used by WDBX.

    Attributes:
        id: Unique identifier for the block
        timestamp: Time when the block was created
        data: Arbitrary data stored in the block
        embeddings: List of embedding vectors
        previous_hash: Hash of the previous block in the chain
        hash: Hash of this block
        context_references: References to related blocks
    """

    def __init__(
        self,
        id: str,
        timestamp: float,
        data: Dict[str, Any],
        embeddings: List[EmbeddingVector],
        previous_hash: str = "",
        context_references: List[str] = None,
    ):
        """
        Initialize a block.

        Args:
            id: Unique identifier
            timestamp: Creation timestamp
            data: Block data
            embeddings: List of embedding vectors
            previous_hash: Hash of the previous block
            context_references: References to related blocks
        """
        self.id = id
        self.timestamp = timestamp
        self.data = data
        self.embeddings = embeddings
        self.previous_hash = previous_hash
        self.context_references = context_references or []
        self.hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """
        Calculate the hash of the block.

        Returns:
            SHA-256 hash of the block
        """
        # Create a string representation of the block
        block_string = (
            f"{self.id}{self.timestamp}{json.dumps(self.data, sort_keys=True)}"
            f"{self.previous_hash}{','.join(self.context_references)}"
        )

        # Add hashes of embeddings
        for embedding in self.embeddings:
            if hasattr(embedding, "id"):
                block_string += embedding.id

        # Calculate SHA-256 hash
        return hashlib.sha256(block_string.encode("utf-8")).hexdigest()

    def validate(self) -> bool:
        """
        Validate the block's integrity.

        Returns:
            True if valid, False otherwise
        """
        return self.hash == self._calculate_hash()

    def serialize(self) -> bytes:
        """
        Serialize the block to bytes.

        Returns:
            Serialized representation of the block
        """
        # Serialize embeddings
        serialized_embeddings = []
        for emb in self.embeddings:
            if isinstance(emb, EmbeddingVector):
                emb_data = {
                    "id": emb.id,
                    "vector": emb.vector.tolist() if hasattr(emb.vector, "tolist") else emb.vector,
                    "metadata": emb.metadata,
                }
                serialized_embeddings.append(emb_data)

        data = {
            "id": self.id,
            "timestamp": self.timestamp,
            "data": self.data,
            "embeddings": serialized_embeddings,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
            "context_references": self.context_references,
        }

        return json.dumps(data).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "Block":
        """
        Deserialize bytes to a Block.

        Args:
            data: Serialized block data

        Returns:
            Deserialized Block
        """
        parsed = json.loads(data.decode("utf-8"))

        # Deserialize embeddings
        embeddings = []
        for emb_data in parsed["embeddings"]:
            embedding = EmbeddingVector(
                vector=emb_data["vector"], metadata=emb_data["metadata"], vector_id=emb_data["id"]
            )
            embeddings.append(embedding)

        # Create block without calculating hash
        block = cls(
            id=parsed["id"],
            timestamp=parsed["timestamp"],
            data=parsed["data"],
            embeddings=embeddings,
            previous_hash=parsed["previous_hash"],
            context_references=parsed["context_references"],
        )

        # Set the hash directly to avoid recalculation
        block.hash = parsed["hash"]

        return block


class ShardInfo:
    """
    Information about a shard in the distributed system.

    Attributes:
        shard_id: Unique identifier for the shard
        node_id: ID of the node hosting the shard
        size: Number of vectors in the shard
        vector_ids: List of vector IDs in the shard
        metadata: Additional shard metadata
    """

    def __init__(
        self,
        shard_id: str,
        node_id: str,
        size: int = 0,
        vector_ids: List[str] = None,
        metadata: Dict[str, Any] = None,
    ):
        """
        Initialize shard information.

        Args:
            shard_id: Unique identifier
            node_id: ID of the hosting node
            size: Number of vectors
            vector_ids: List of vector IDs
            metadata: Additional metadata
        """
        self.shard_id = shard_id
        self.node_id = node_id
        self.size = size
        self.vector_ids = vector_ids or []
        self.metadata = metadata or {}
