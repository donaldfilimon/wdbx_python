#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WDBX Core Components
-------------------
Core data structures and utilities for the Wide Distributed Block Exchange system.
This module provides the fundamental building blocks used throughout the system.
"""

import os
import sys
import time
import uuid
import json
import pickle
import hashlib
import threading
import array
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("WDBX")

###############################################################################
#                              UTILITY CLASSES                                #
###############################################################################

class NumpyLikeArray:
    """
    A lightweight numpy-like array implementation using Python's built-in array module.
    Provides vector operations needed for similarity search without external dependencies.
    """
    def __init__(self, data, dtype='f'):
        """Initialize with data - can be list, array.array, or another NumpyLikeArray."""
        if isinstance(data, NumpyLikeArray):
            self.data = array.array(dtype, data.data)
        elif isinstance(data, array.array):
            self.data = array.array(dtype, data)
        else:
            self.data = array.array(dtype, data)
        self.shape = (len(self.data),)
        self.dtype = dtype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return NumpyLikeArray(self.data[idx], self.dtype)
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

    def tolist(self):
        return list(self.data)

    def tobytes(self):
        return self.data.tobytes()

    def copy(self):
        """Create a copy of this array."""
        return NumpyLikeArray(self.data, self.dtype)

    @classmethod
    def zeros(cls, shape, dtype='f'):
        """Create array of zeros with given shape."""
        if isinstance(shape, (list, tuple)):
            size = shape[0]
        else:
            size = shape
        return cls([0.0] * size, dtype)

    @classmethod
    def from_bytes(cls, data, dtype='f'):
        """Create array from bytes."""
        result = array.array(dtype)
        result.frombytes(data)
        return cls(result, dtype)

    def dot(self, other):
        """Compute dot product with another vector."""
        if len(self) != len(other):
            raise ValueError(f"Vector dimensions don't match: {len(self)} vs {len(other)}")
        return sum(a * b for a, b in zip(self.data, other.data))

    def norm(self):
        """Compute L2 (Euclidean) norm of vector."""
        return math.sqrt(sum(x * x for x in self.data))

    def normalize(self):
        """Return normalized vector (unit length)."""
        norm = self.norm()
        if norm < 1e-10:
            return NumpyLikeArray([0.0] * len(self.data), self.dtype)
        return NumpyLikeArray([x / norm for x in self.data], self.dtype)

    def __add__(self, other):
        """Vector addition."""
        if len(self) != len(other):
            raise ValueError(f"Vector dimensions don't match: {len(self)} vs {len(other)}")
        return NumpyLikeArray([a + b for a, b in zip(self.data, other.data)], self.dtype)

    def __sub__(self, other):
        """Vector subtraction."""
        if len(self) != len(other):
            raise ValueError(f"Vector dimensions don't match: {len(self)} vs {len(other)}")
        return NumpyLikeArray([a - b for a, b in zip(self.data, other.data)], self.dtype)

    def __mul__(self, scalar):
        """Scalar multiplication."""
        return NumpyLikeArray([x * scalar for x in self.data], self.dtype)

    def __truediv__(self, scalar):
        """Scalar division."""
        if abs(scalar) < 1e-10:
            raise ValueError("Division by near-zero value")
        return NumpyLikeArray([x / scalar for x in self.data], self.dtype)

    def cosine_similarity(self, other):
        """Compute cosine similarity with another vector."""
        dot_product = self.dot(other)
        norm_product = self.norm() * other.norm()
        if norm_product < 1e-10:
            return 0.0
        return dot_product / norm_product


class Timer:
    """Context manager for timing code execution."""
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed = end_time - self.start_time
        logger.debug(f"{self.name} took {elapsed:.6f} seconds")

    @property
    def elapsed(self):
        """Return elapsed time since timer started."""
        if self.start_time == 0:
            return 0
        return time.time() - self.start_time


###############################################################################
#                           CORE DATA STRUCTURES                              #
###############################################################################

@dataclass
class EmbeddingVector:
    """
    Represents a high-dimensional embedding vector with metadata.
    The foundation of WDBX's vector-based operations.

    Attributes:
        vector (NumpyLikeArray): The embedding vector.
        metadata (Dict[str, Any]): Additional metadata associated with the vector.
        timestamp (float): Creation or modification timestamp.
    """
    vector: NumpyLikeArray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    _normalized: Optional[NumpyLikeArray] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.vector, NumpyLikeArray):
            self.vector = NumpyLikeArray(self.vector)

    def normalize(self) -> NumpyLikeArray:
        """Return normalized vector for similarity calculations (cached for efficiency)."""
        if self._normalized is None:
            self._normalized = self.vector.normalize()
        return self._normalized

    def cosine_similarity(self, other: 'EmbeddingVector') -> float:
        """Calculate cosine similarity with another embedding vector."""
        return self.vector.cosine_similarity(other.vector)

    def serialize(self, use_compression: bool = False) -> bytes:
        """
        Serialize vector to bytes for storage.

        Args:
            use_compression (bool): Whether to use compression to reduce size.
        """
        data = {
            "vector": self.vector.tolist(),
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        if use_compression:
            import zlib
            return zlib.compress(serialized)
        return serialized

    @classmethod
    def deserialize(cls, data: bytes, is_compressed: bool = False) -> "EmbeddingVector":
        """
        Deserialize vector from bytes.

        Args:
            data (bytes): The serialized data to deserialize.
            is_compressed (bool): Whether the data is compressed.
        """
        if is_compressed:
            import zlib
            data = zlib.decompress(data)

        loaded = pickle.loads(data)
        return cls(
            vector=NumpyLikeArray(loaded["vector"]),
            metadata=loaded["metadata"],
            timestamp=loaded["timestamp"]
        )

    def __len__(self) -> int:
        """Return dimension of the vector."""
        return len(self.vector)

    def dimension(self) -> int:
        """Return the dimensionality of the vector."""
        return len(self.vector)

    def update_metadata(self, new_metadata: Dict[str, Any]) -> None:
        """Update metadata with new key-value pairs."""
        self.metadata.update(new_metadata)
        self.timestamp = time.time()


@dataclass
class Block:
    """
    Data block that groups embedding vectors with blockchain-like integrity.
    Provides cryptographic verification and tamper resistance.

    Attributes:
        id (str): Unique identifier for the block.
        timestamp (float): Block creation timestamp.
        data (Dict[str, Any]): Block data payload.
        embeddings (List[EmbeddingVector]): Vector embeddings contained in the block.
        previous_hash (str): Hash of the previous block (for chain linking).
        context_references (List[str]): References to related blocks or contexts.
        hash (str): Cryptographic hash of the block contents.
        nonce (int): Number used once for mining the block.
    """
    id: str
    timestamp: float
    data: Dict[str, Any]
    embeddings: List[EmbeddingVector]
    previous_hash: str = ""
    context_references: List[str] = field(default_factory=list)
    hash: str = ""
    nonce: int = 0
    _embedding_hashes: Dict[int, str] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Automatically mine block if no hash is provided."""
        if not self.hash:
            self.mine_block()

    def calculate_hash(self) -> str:
        """Calculate cryptographic hash of block contents."""
        # Pre-compute embedding hashes if not already cached
        if not self._embedding_hashes or len(self._embedding_hashes) != len(self.embeddings):
            self._embedding_hashes = {
                i: hashlib.sha256(e.vector.tobytes()).hexdigest()
                for i, e in enumerate(self.embeddings)
            }

        # Use a more efficient string building approach
        block_content = [
            self.id,
            str(self.timestamp),
            json.dumps(self.data, sort_keys=True),
            self.previous_hash,
            str(self.nonce),
            *self._embedding_hashes.values(),
            *self.context_references
        ]

        return hashlib.sha256("".join(block_content).encode()).hexdigest()

    def mine_block(self, difficulty: int = 2, max_iterations: int = 10000000) -> bool:
        """
        Mine block by finding nonce that produces hash with required difficulty.

        Args:
            difficulty (int): Number of leading zeros required in hash.
            max_iterations (int): Maximum number of attempts before giving up.

        Returns:
            bool: True if successfully mined, False if max_iterations reached.
        """
        target = '0' * difficulty
        iteration = 0

        while iteration < max_iterations:
            self.hash = self.calculate_hash()
            if self.hash.startswith(target):
                return True
            self.nonce += 1
            iteration += 1

            # Log progress periodically
            if iteration % 100000 == 0:
                logger.debug(f"Mining block {self.id}: {iteration} iterations completed")

        logger.warning(f"Failed to mine block {self.id} after {max_iterations} iterations")
        return False

    def validate(self) -> bool:
        """Validate block integrity by recomputing and comparing hash."""
        return self.hash == self.calculate_hash()

    def serialize(self, use_compression: bool = False) -> bytes:
        """
        Serialize block to bytes for storage.

        Args:
            use_compression (bool): Whether to use compression to reduce size.
        """
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
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        if use_compression:
            import zlib
            return zlib.compress(serialized)
        return serialized

    @classmethod
    def deserialize(cls, data: bytes, is_compressed: bool = False) -> "Block":
        """
        Deserialize block from bytes.

        Args:
            data (bytes): The serialized data to deserialize.
            is_compressed (bool): Whether the data is compressed.
        """
        if is_compressed:
            import zlib
            data = zlib.decompress(data)

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

    def add_embedding(self, embedding: EmbeddingVector) -> None:
        """Add a new embedding to the block and update the hash."""
        self.embeddings.append(embedding)
        # Clear the hash cache as the block content has changed
        self._embedding_hashes = {}
        # Remine the block to update the hash
        self.mine_block()

    def get_mean_embedding(self) -> Optional[EmbeddingVector]:
        """Calculate and return the mean embedding vector of all embeddings in the block."""
        if not self.embeddings:
            return None

        vectors = [e.vector for e in self.embeddings]
        mean_vector = mean_vectors(vectors)
        return EmbeddingVector(
            vector=mean_vector,
            metadata={"source": f"mean_of_block_{self.id}", "vector_count": len(vectors)},
            timestamp=time.time()
        )


@dataclass
class ShardInfo:
    """
    Information about a storage shard in the distributed system.
    Used for load balancing and routing.

    Attributes:
        shard_id (int): Unique identifier for the shard.
        host (str): Hostname or IP address of the shard.
        port (int): Port number to connect to the shard.
        status (str): Current operational status of the shard.
        load (float): Current load factor (0.0 - 1.0).
        block_count (int): Number of blocks stored in this shard.
        last_updated (float): Timestamp of the last status update.
    """
    shard_id: int
    host: str
    port: int
    status: str = "active"
    load: float = 0.0
    block_count: int = 0
    last_updated: float = field(default_factory=time.time)
    capacity: int = 1000  # Maximum number of blocks this shard can efficiently handle

    def __post_init__(self):
        """Validate the shard information."""
        if self.shard_id < 0:
            raise ValueError("Shard ID must be non-negative")

        if not self.host:
            raise ValueError("Host cannot be empty")

        if not (0 < self.port < 65536):
            raise ValueError("Port must be between 1 and 65535")

        if not (0.0 <= self.load <= 1.0):
            self.load = max(0.0, min(1.0, self.load))
            logger.warning(f"Load factor for shard {self.shard_id} adjusted to valid range [0.0, 1.0]: {self.load}")

    def calculate_latency(self, retrieval_size: int, network_overhead: float = 0.02) -> float:
        """
        Estimate latency for retrieving data of given size from this shard.

        Args:
            retrieval_size (int): Size of data to retrieve in bytes.
            network_overhead (float): Base network latency in seconds.

        Returns:
            float: Estimated latency in seconds.
        """
        # More sophisticated latency model considering:
        # - Network distance (constant per shard)
        # - Current load
        # - Size of data transfer
        # - Block count as a factor in lookup time
        lookup_factor = 0.001 * min(1.0, self.block_count / self.capacity)
        transfer_factor = 0.000001 * retrieval_size

        return network_overhead + (lookup_factor * self.block_count) + (transfer_factor * self.load)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "shard_id": self.shard_id,
            "host": self.host,
            "port": self.port,
            "status": self.status,
            "load": self.load,
            "block_count": self.block_count,
            "last_updated": self.last_updated,
            "capacity": self.capacity
        }

    def update_status(self, status: str, load: float = None) -> None:
        """Update shard status information."""
        self.status = status
        if load is not None:
            self.load = max(0.0, min(1.0, load))
        self.last_updated = time.time()

    def is_available(self) -> bool:
        """Check if the shard is currently available for operations."""
        return (self.status == "active" and
                self.load < 0.95 and
                self.block_count < self.capacity and
                (time.time() - self.last_updated) < 300)  # 5 minutes freshness


def mean_vectors(vectors: List[NumpyLikeArray]) -> NumpyLikeArray:
    """
    Calculate mean of multiple vectors more efficiently.

    Args:
        vectors (List[NumpyLikeArray]): List of vectors to average.

    Returns:
        NumpyLikeArray: Mean vector.

    Raises:
        ValueError: If the vectors list is empty or vectors have different dimensions.
    """
    if not vectors:
        raise ValueError("Cannot compute mean of empty list")

    # Check dimensions
    dim = len(vectors[0])
    if not all(len(v) == dim for v in vectors):
        raise ValueError("All vectors must have the same dimension")

    n = len(vectors)
    if n == 1:
        return vectors[0].copy()

    # Pre-allocate result array and accumulate sum
    result = NumpyLikeArray.zeros(dim)
    for vec in vectors:
        result = result + vec

    # Divide by count to get mean
    return result * (1.0 / n)


class SimpleKMeans:
    """
    Improved K-means clustering implementation without external dependencies.
    Includes k-means++ initialization for better starting centroids.

    Attributes:
        n_clusters (int): Number of clusters to form.
        max_iterations (int): Maximum number of iterations.
        random_seed (int): Seed for random number generation.
        centroids (List[NumpyLikeArray]): Cluster centroids.
        labels (List[int]): Cluster assignments for each input point.
        inertia (float): Sum of squared distances to closest centroid.
        use_kmeans_plus_plus (bool): Whether to use k-means++ initialization.
    """
    def __init__(self, n_clusters=3, max_iterations=100, random_seed=42, use_kmeans_plus_plus=True):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.centroids = []
        self.labels = []
        self.inertia = float('inf')
        self.use_kmeans_plus_plus = use_kmeans_plus_plus

    def _initialize_centroids(self, X: List[NumpyLikeArray]) -> List[NumpyLikeArray]:
        """
        Initialize centroids using k-means++ algorithm or random selection.

        Args:
            X (List[NumpyLikeArray]): Data points.

        Returns:
            List[NumpyLikeArray]: Initial centroids.
        """
        import random
        random.seed(self.random_seed)

        if not self.use_kmeans_plus_plus:
            indices = random.sample(range(len(X)), self.n_clusters)
            return [X[idx].copy() for idx in indices]

        # K-means++ initialization
        centroids = [X[random.randint(0, len(X) - 1)].copy()]

        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            # Compute squared distances to closest centroid
            distances = []
            for point in X:
                min_dist = min(1.0 - point.cosine_similarity(centroid) for centroid in centroids)
                distances.append(min_dist ** 2)

            # Choose next centroid with probability proportional to squared distance
            sum_distances = sum(distances)
            if sum_distances < 1e-10:  # Avoid division by near-zero
                # Fall back to random selection if all points are very close to centroids
                while len(centroids) < self.n_clusters:
                    idx = random.randint(0, len(X) - 1)
                    if not any(X[idx].cosine_similarity(c) > 0.99 for c in centroids):
                        centroids.append(X[idx].copy())
                return centroids

            target = random.uniform(0, sum_distances)
            cumsum = 0
            for i, dist in enumerate(distances):
                cumsum += dist
                if cumsum >= target:
                    centroids.append(X[i].copy())
                    break

        return centroids

    def fit_predict(self, X: List[NumpyLikeArray], early_stopping_threshold: float = 0.001) -> List[int]:
        """
        Fit K-means and return cluster assignments.

        Args:
            X (List[NumpyLikeArray]): Data points to cluster.
            early_stopping_threshold (float): Stop iterations if less than this fraction
                                             of points change clusters.

        Returns:
            List[int]: Cluster assignments for each point.
        """
        import random

        if len(X) < self.n_clusters:
            raise ValueError(f"Number of samples {len(X)} less than number of clusters {self.n_clusters}")

        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        prev_labels = [-1] * len(X)

        for iteration in range(self.max_iterations):
            # Assign points to nearest centroids
            clusters = [[] for _ in range(self.n_clusters)]
            self.labels = []
            total_distance = 0.0

            for i, point in enumerate(X):
                closest, dist = self._closest_centroid_with_distance(point)
                clusters[closest].append(i)
                self.labels.append(closest)
                total_distance += dist

            # Update inertia (sum of squared distances)
            self.inertia = total_distance

            # Check for early stopping - fraction of points that changed clusters
            changes = sum(1 for i, label in enumerate(self.labels) if label != prev_labels[i])
            change_ratio = changes / len(X)

            if change_ratio < early_stopping_threshold and iteration > 0:
                logger.debug(f"K-means converged after {iteration+1} iterations (change ratio: {change_ratio:.4f})")
                break

            prev_labels = self.labels.copy()

            # Update centroids
            old_centroids = self.centroids.copy()
            for i in range(self.n_clusters):
                if not clusters[i]:  # Empty cluster
                    # Reinitialize empty cluster with point furthest from any centroid
                    max_dist = -1
                    max_idx = -1
                    for j, point in enumerate(X):
                        min_dist = min(1.0 - point.cosine_similarity(c) for c in self.centroids)
                        if min_dist > max_dist:
                            max_dist = min_dist
                            max_idx = j

                    if max_idx >= 0:
                        self.centroids[i] = X[max_idx].copy()
                    continue

                self.centroids[i] = mean_vectors([X[idx] for idx in clusters[i]])

            # Check for convergence
            if all(old.cosine_similarity(new) > 0.999 for old, new in zip(old_centroids, self.centroids)):
                logger.debug(f"K-means converged after {iteration+1} iterations (centroid stability)")
                break

        return self.labels

    def _closest_centroid_with_distance(self, point: NumpyLikeArray) -> Tuple[int, float]:
        """
        Find index of closest centroid to the given point and the distance.

        Args:
            point (NumpyLikeArray): Data point.

        Returns:
            Tuple[int, float]: (centroid index, distance)
        """
        similarities = [point.cosine_similarity(centroid) for centroid in self.centroids]
        max_sim = max(similarities)
        closest = similarities.index(max_sim)
        # Convert similarity to distance (1 - similarity)
        distance = 1.0 - max_sim
        return closest, distance

    def _closest_centroid(self, point: NumpyLikeArray) -> int:
        """Find index of closest centroid to the given point."""
        closest, _ = self._closest_centroid_with_distance(point)
        return closest

    def predict(self, X: List[NumpyLikeArray]) -> List[int]:
        """
        Predict the closest cluster for each sample in X.

        Args:
            X (List[NumpyLikeArray]): New data points.

        Returns:
            List[int]: Cluster assignments for each point.
        """
        if not self.centroids:
            raise ValueError("Model not fitted yet. Call fit_predict first.")

        return [self._closest_centroid(point) for point in X]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WDBX Database Connectors
-----------------------
Database connection managers for the Wide Distributed Block Exchange system.

This module provides:
- Abstract database client interface
- Protocol-specific implementations (HTTP, Socket, Filesystem)
- Connection management with failover and retry capabilities
- Command-line interface for basic database operations

Each connector implements the same interface, allowing transparent switching
between storage backends based on availability and performance requirements.
"""

# Standard library imports
import os
import sys
import time
import uuid
import json
import pickle
import hashlib
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set

# Network-related imports
import socket
import urllib.request
import urllib.parse
import urllib.error

# Concurrent execution imports
import asyncio
import concurrent.futures

# Logging
import logging

# Configure module logger
logger = logging.getLogger("WDBX.Database")

###############################################################################
#                          DATABASE CLIENT INTERFACE                          #
###############################################################################

class DatabaseClient(ABC):
    """
    Abstract base class for database clients.
    Defines the interface that all database client implementations must adhere to.
    """

    @abstractmethod
    def close(self) -> None:
        """Close the connection and release any resources."""
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        """
        Get a value from the database by key.

        Args:
            key (str): The key to retrieve

        Returns:
            Optional[bytes]: The value associated with the key, or None if not found
        """
        pass

    @abstractmethod
    def put(self, key: str, value: bytes) -> bool:
        """
        Put a value into the database with the given key.

        Args:
            key (str): The key to store
            value (bytes): The value to store

        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a value from the database by key.

        Args:
            key (str): The key to delete

        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query the database with the given parameters.

        Args:
            query_params (Dict[str, Any]): Query parameters to filter results

        Returns:
            List[Dict[str, Any]]: List of matching results
        """
        pass

    def health_check(self) -> bool:
        """
        Check if the database is accessible and functioning correctly.
        Default implementation uses a test key/value operation.

        Returns:
            bool: True if the database is healthy, False otherwise
        """
        test_key = f"__health_check_{uuid.uuid4()}"
        test_value = f"health_check_{time.time()}".encode('utf-8')

        try:
            if not self.put(test_key, test_value):
                return False

            retrieved = self.get(test_key)
            result = retrieved == test_value

            self.delete(test_key)
            return result
        except Exception:
            return False

    def __enter__(self):
        """Support for context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are cleaned up when exiting context."""
        self.close()


class HttpDatabaseClient(DatabaseClient):
    """
    Client for interacting with a database via HTTP protocol.
    Uses only Python's built-in libraries with connection pooling.
    """
    def __init__(self, host: str = "localhost", port: int = 8080,
                 timeout: int = 5, pool_connections: int = 10,
                 retry_count: int = 2) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.retry_count = retry_count
        self.base_url = f"http://{host}:{port}"

        # Create connection pool handler
        self.opener = self._create_connection_pool(pool_connections)
        logger.info(f"Initialized HTTP database client to {self.base_url} with {pool_connections} pooled connections")

    def _create_connection_pool(self, pool_size: int) -> urllib.request.OpenerDirector:
        """Create and return a connection pool."""
        import http.client

        class PooledHTTPConnection(http.client.HTTPConnection):
            def __init__(self, host, port=None, timeout=None):
                super().__init__(host, port, timeout=timeout)

        class PooledHTTPHandler(urllib.request.HTTPHandler):
            def http_open(self, req):
                return self.do_open(
                    lambda host, **kwargs: PooledHTTPConnection(host, timeout=timeout), req)

        timeout = self.timeout
        handler = PooledHTTPHandler()
        opener = urllib.request.build_opener(handler)

        # Set default headers
        opener.addheaders = [('User-Agent', 'WDBX/1.0'), ('Connection', 'keep-alive')]
        return opener

    def close(self) -> None:
        """Close the HTTP connection pool."""
        # No explicit close needed for urllib pools
        pass

    def get(self, key: str) -> Optional[bytes]:
        """Get a value from the database by key with retry logic."""
        errors = []
        for attempt in range(self.retry_count + 1):
            try:
                url = f"{self.base_url}/db/{urllib.parse.quote(key)}"
                with self.opener.open(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        return response.read()
                    return None
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                errors.append(f"HTTP error {e.code}: {e.reason}")
                if attempt == self.retry_count:
                    logger.error(f"HTTP get failed after {attempt+1} attempts: {e.code} - {e.reason}")
            except Exception as e:
                errors.append(str(e))
                if attempt == self.retry_count:
                    logger.error(f"HTTP get error after {attempt+1} attempts: {str(e)}")

            if attempt < self.retry_count:
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff

        logger.error(f"HTTP get failed: {'; '.join(errors)}")
        return None

    def put(self, key: str, value: bytes) -> bool:
        """Put a value into the database with the given key with retry logic."""
        errors = []
        for attempt in range(self.retry_count + 1):
            try:
                url = f"{self.base_url}/db/{urllib.parse.quote(key)}"
                headers = {"Content-Type": "application/octet-stream"}
                request = urllib.request.Request(url, data=value, headers=headers, method="PUT")
                with self.opener.open(request, timeout=self.timeout) as response:
                    return response.status in (200, 201, 204)
            except Exception as e:
                errors.append(str(e))
                if attempt < self.retry_count:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"HTTP put error after {attempt+1} attempts: {str(e)}")

        logger.error(f"HTTP put failed: {'; '.join(errors)}")
        return False

    def delete(self, key: str) -> bool:
        """Delete a value from the database by key with retry logic."""
        errors = []
        for attempt in range(self.retry_count + 1):
            try:
                url = f"{self.base_url}/db/{urllib.parse.quote(key)}"
                request = urllib.request.Request(url, method="DELETE")
                with self.opener.open(request, timeout=self.timeout) as response:
                    return response.status in (200, 202, 204)
            except Exception as e:
                errors.append(str(e))
                if attempt < self.retry_count:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"HTTP delete error after {attempt+1} attempts: {str(e)}")

        logger.error(f"HTTP delete failed: {'; '.join(errors)}")
        return False

    def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query the database with the given parameters with retry logic."""
        errors = []
        for attempt in range(self.retry_count + 1):
            try:
                url = f"{self.base_url}/db/query"
                data = json.dumps(query_params).encode('utf-8')
                headers = {"Content-Type": "application/json"}
                request = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with self.opener.open(request, timeout=self.timeout) as response:
                    if response.status == 200:
                        return json.loads(response.read().decode('utf-8'))
                    return []
            except Exception as e:
                errors.append(str(e))
                if attempt < self.retry_count:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"HTTP query error after {attempt+1} attempts: {str(e)}")

        logger.error(f"HTTP query failed: {'; '.join(errors)}")
        return []


class SocketDatabaseClient(DatabaseClient):
    """
    Client for interacting with a database via raw socket protocol.
    More efficient for large data transfers compared to HTTP.
    Implements connection pooling and advanced buffer management.
    """
    # Command codes for the socket protocol
    CMD_GET = 1
    CMD_PUT = 2
    CMD_DELETE = 3
    CMD_QUERY = 4
    CMD_HEALTH = 5

    # Response codes
    RESP_OK = 10
    RESP_NOT_FOUND = 11
    RESP_ERROR = 12

    def __init__(self, host: str = "localhost", port: int = 9090, timeout: int = 5,
                 pool_size: int = 3, buffer_size: int = 8192) -> None:
        """
        Initialize the socket database client with connection pooling.

        Args:
            host (str): Server hostname
            port (int): Server port
            timeout (int): Socket timeout in seconds
            pool_size (int): Maximum number of connections to keep open
            buffer_size (int): Size of the socket buffer for data transfer
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.buffer_size = buffer_size
        self.pool_size = pool_size
        self._connection_pool = []
        self._pool_lock = threading.RLock()
        logger.info(f"Initialized socket database client to {host}:{port} with connection pool of {pool_size}")

    def _get_connection(self) -> Optional[socket.socket]:
        """Get a connection from the pool or create a new one."""
        with self._pool_lock:
            if self._connection_pool:
                return self._connection_pool.pop()

        # Create new connection
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            return sock
        except Exception as e:
            logger.error(f"Socket connection error: {str(e)}")
            return None

    def _return_connection(self, sock: socket.socket) -> None:
        """Return a connection to the pool if it's still usable."""
        if sock is None:
            return

        try:
            # Check if the socket is still usable
            if sock.fileno() == -1:
                return

            # Add back to pool if there's room
            with self._pool_lock:
                if len(self._connection_pool) < self.pool_size:
                    self._connection_pool.append(sock)
                else:
                    sock.close()
        except Exception:
            try:
                sock.close()
            except Exception:
                pass

    def close(self) -> None:
        """Close all pooled socket connections."""
        with self._pool_lock:
            for sock in self._connection_pool:
                try:
                    sock.close()
                except Exception:
                    pass
            self._connection_pool = []

    def _send_command(self, command: int, key: str, value: Optional[bytes] = None) -> Tuple[int, Optional[bytes]]:
        """Send a command to the database server and receive the response with improved buffer handling."""
        sock = self._get_connection()
        if not sock:
            return self.RESP_ERROR, None

        try:
            # Protocol format:
            # [1 byte command][4 bytes key length][key bytes]
            # [4 bytes value length (if applicable)][value bytes (if applicable)]
            key_bytes = key.encode('utf-8')
            key_len = len(key_bytes)

            # Build the message header
            message = bytearray()
            message.append(command)
            message.extend(key_len.to_bytes(4, byteorder='big'))
            message.extend(key_bytes)

            # Add value if provided
            if value and command in (self.CMD_PUT, self.CMD_QUERY):
                val_len = len(value)
                message.extend(val_len.to_bytes(4, byteorder='big'))

                # For large values, send the header first then stream the data
                if val_len > self.buffer_size:
                    sock.sendall(message)
                    # Send value in chunks
                    sent = 0
                    while sent < val_len:
                        chunk_size = min(self.buffer_size, val_len - sent)
                        sock.sendall(value[sent:sent+chunk_size])
                        sent += chunk_size
                else:
                    # Small values can be sent with the header
                    message.extend(value)
                    sock.sendall(message)
            else:
                # Just send the header for commands without values
                sock.sendall(message)

            # Receive the response code
            resp_data = sock.recv(1)
            if not resp_data:
                return self.RESP_ERROR, None

            resp_code = int.from_bytes(resp_data, byteorder='big')

            # For GET responses, read the data
            if resp_code == self.RESP_OK and command == self.CMD_GET:
                # Get data length
                length_data = sock.recv(4)
                if not length_data or len(length_data) < 4:
                    return self.RESP_ERROR, None

                data_len = int.from_bytes(length_data, byteorder='big')

                # Receive data in chunks for better memory usage
                chunks = []
                remaining = data_len
                while remaining > 0:
                    chunk_size = min(self.buffer_size, remaining)
                    chunk = sock.recv(chunk_size)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)

                if remaining > 0:
                    logger.warning(f"Incomplete data received, missing {remaining} bytes")
                    return self.RESP_ERROR, None

                # Combine chunks efficiently
                if len(chunks) == 1:
                    result_data = chunks[0]
                else:
                    result_data = b''.join(chunks)

                return resp_code, result_data

            # For QUERY responses, read the JSON data
            if resp_code == self.RESP_OK and command == self.CMD_QUERY:
                # Get data length
                length_data = sock.recv(4)
                if not length_data or len(length_data) < 4:
                    return self.RESP_ERROR, None

                data_len = int.from_bytes(length_data, byteorder='big')

                # Receive data in chunks
                data = bytearray()
                remaining = data_len
                while remaining > 0:
                    chunk_size = min(self.buffer_size, remaining)
                    chunk = sock.recv(chunk_size)
                    if not chunk:
                        break
                    data.extend(chunk)
                    remaining -= len(chunk)

                return resp_code, bytes(data)

            return resp_code, None

        except Exception as e:
            logger.error(f"Socket command error: {str(e)}")
            # Close failed connections instead of returning to pool
            try:
                sock.close()
            except Exception:
                pass
            sock = None
            return self.RESP_ERROR, None
        finally:
            if sock:
                self._return_connection(sock)

    def get(self, key: str) -> Optional[bytes]:
        """Get a value from the database by key."""
        resp_code, data = self._send_command(self.CMD_GET, key)
        if resp_code == self.RESP_OK:
            return data
        return None

    def put(self, key: str, value: bytes) -> bool:
        """Put a value into the database with the given key."""
        resp_code, _ = self._send_command(self.CMD_PUT, key, value)
        return resp_code == self.RESP_OK

    def delete(self, key: str) -> bool:
        """Delete a value from the database by key."""
        resp_code, _ = self._send_command(self.CMD_DELETE, key)
        return resp_code == self.RESP_OK

    def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query the database with the given parameters."""
        try:
            # Serialize the query params
            query_bytes = json.dumps(query_params).encode('utf-8')
            # Use a special query key
            resp_code, data = self._send_command(self.CMD_QUERY, "_query", query_bytes)
            if resp_code == self.RESP_OK and data:
                return json.loads(data.decode('utf-8'))
            return []
        except Exception as e:
            logger.error(f"Socket query error: {str(e)}")
            return []


class FilesystemDatabaseClient(DatabaseClient):
    """
    Client for using the filesystem as a database.
    Useful for single-node deployments or testing.
    Implements directory sharding and file locking for better performance.
    """
    def __init__(self, data_dir: str = "./wdbx_data",
                 shard_level: int = 2,
                 use_compression: bool = False) -> None:
        """
        Initialize the filesystem database client.

        Args:
            data_dir (str): Base directory for data storage
            shard_level (int): Number of subdirectory levels for sharding (0-3)
            use_compression (bool): Whether to compress stored data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.shard_level = max(0, min(3, shard_level))  # Limit to 0-3 levels
        self.use_compression = use_compression

        # Create metadata index directory
        self.index_dir = self.data_dir / "_index"
        self.index_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized filesystem database client at {self.data_dir} "
                   f"(sharding: {self.shard_level}, compression: {self.use_compression})")

    def close(self) -> None:
        """Nothing to close with filesystem."""
        pass

    def _get_shard_path(self, key_hash: str) -> Path:
        """Get the sharded directory path for a key hash."""
        if self.shard_level == 0:
            return self.data_dir

        # Use first N characters of hash for sharding
        paths = [key_hash[i:i+2] for i in range(0, self.shard_level * 2, 2)]
        path = self.data_dir

        for subdir in paths:
            path = path / subdir
            path.mkdir(parents=True, exist_ok=True)

        return path

    def _get_file_paths(self, key: str) -> Tuple[Path, Path]:
        """Get the data and metadata file paths for a key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        shard_path = self._get_shard_path(key_hash)
        data_path = shard_path / key_hash
        meta_path = self.index_dir / f"{key_hash}.meta"
        return data_path, meta_path

    def check_connection(self) -> bool:
        """Test database connection by attempting to create and read a test file."""
        test_key = f"_connection_test_{uuid.uuid4()}"
        test_data = b"connection_test"

        try:
            if not self.put(test_key, test_data):
                return False

            read_data = self.get(test_key)
            result = read_data == test_data

            self.delete(test_key)
            return result
        except Exception as e:
            logger.error(f"Connection check failed: {str(e)}")
            return False

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if compression is enabled."""
        if not self.use_compression:
            return data

        import zlib
        return zlib.compress(data)

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data if compression is enabled."""
        if not self.use_compression:
            return data

        import zlib
        return zlib.decompress(data)

    def get(self, key: str) -> Optional[bytes]:
        """Get a value from the database by key."""
        data_path, _ = self._get_file_paths(key)

        if not data_path.exists():
            return None

        try:
            import fcntl
            with open(data_path, 'rb') as f:
                try:
                    # Get shared lock for reading
                    fcntl.flock(f, fcntl.LOCK_SH)
                    data = f.read()
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

            return self._decompress_data(data)
        except ImportError:
            # fcntl not available on non-Unix platforms
            with open(data_path, 'rb') as f:
                data = f.read()
            return self._decompress_data(data)
        except Exception as e:
            logger.error(f"Filesystem get error: {str(e)}")
            return None

    def put(self, key: str, value: bytes) -> bool:
        """Put a value into the database with the given key."""
        data_path, meta_path = self._get_file_paths(key)

        try:
            # Prepare data and metadata
            compressed_data = self._compress_data(value)
            metadata = {
                "key": key,
                "original_size": len(value),
                "stored_size": len(compressed_data),
                "timestamp": time.time(),
                "compressed": self.use_compression
            }

            with self.lock:
                import fcntl

                # Write data file with exclusive lock
                with open(data_path, 'wb') as f:
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        f.write(compressed_data)
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)

                # Write metadata file
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)

            return True
        except ImportError:
            # fcntl not available on non-Unix platforms
            try:
                with self.lock:
                    with open(data_path, 'wb') as f:
                        f.write(compressed_data)
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f)
                return True
            except Exception as e:
                logger.error(f"Filesystem put error: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Filesystem put error: {str(e)}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a value from the database by key."""
        data_path, meta_path = self._get_file_paths(key)

        try:
            with self.lock:
                if data_path.exists():
                    data_path.unlink()
                if meta_path.exists():
                    meta_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Filesystem delete error: {str(e)}")
            return False

    def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query the database based on keys with optimized file access."""
        results = []
        try:
            # Get metadata files more efficiently
            meta_files = list(self.index_dir.glob("*.meta"))

            # Apply query filters
            key_prefix = query_params.get('key_prefix')
            key_suffix = query_params.get('key_suffix')
            key_contains = query_params.get('key_contains')
            key_exact = query_params.get('key_exact')

            # Use cache for faster lookups
            temp_cache = {}

            for meta_path in meta_files:
                try:
                    # Lazy loading of metadata
                    def get_metadata():
                        if meta_path not in temp_cache:
                            with open(meta_path, 'r') as f:
                                temp_cache[meta_path] = json.load(f)
                        return temp_cache[meta_path]

                    # Fast path for exact match
                    if key_exact:
                        metadata = get_metadata()
                        if metadata['key'] == key_exact:
                            results.append(metadata)
                            continue

                    # Apply other filters only if needed
                    if key_prefix or key_suffix or key_contains:
                        metadata = get_metadata()
                        key = metadata['key']

                        if key_prefix and key.startswith(key_prefix):
                            results.append(metadata)
                        elif key_suffix and key.endswith(key_suffix):
                            results.append(metadata)
                        elif key_contains and key_contains in key:
                            results.append(metadata)

                except Exception as e:
                    logger.error(f"Error processing metadata file {meta_path}: {str(e)}")

            # Sort results by timestamp (newest first)
            results.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            return results

        except Exception as e:
            logger.error(f"Filesystem query error: {str(e)}")
            return []


class DatabaseConnectionManager:
    """
    Manages multiple database client types and provides a unified interface.
    Handles failover and automatic retries.
    """
    def __init__(
        self,
        http_host: str = "localhost",
        http_port: int = 8080,
        socket_host: str = "localhost",
        socket_port: int = 9090,
        data_dir: str = "./wdbx_data",
        use_socket_by_default: bool = True,
        max_retries: int = 3,
        retry_delay: float = 0.5
    ) -> None:
        self.http_client = HttpDatabaseClient(http_host, http_port)
        self.socket_client = SocketDatabaseClient(socket_host, socket_port)
        self.filesystem_client = FilesystemDatabaseClient(data_dir)
        self.use_socket_by_default = use_socket_by_default
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.clients = {
            'http': self.http_client,
            'socket': self.socket_client,
            'filesystem': self.filesystem_client
        }
        self.default_client = 'socket' if use_socket_by_default else 'http'
        self.lock = threading.RLock()

        # Track client health
        self.client_health = {client: True for client in self.clients}
        self.last_health_check = {client: 0 for client in self.clients}
        self.health_check_interval = 60  # seconds

        logger.info(f"Database connection manager initialized (default: {self.default_client})")

    def close_all_connections(self) -> None:
        """Close all database connections."""
        with self.lock:
            for client in self.clients.values():
                client.close()

    def _get_client(self, client_type: Optional[str] = None) -> DatabaseClient:
        """Get the specified client or the default one, considering health."""
        with self.lock:
            # Use specified client if provided and healthy
            if client_type and client_type in self.clients:
                client = self.clients[client_type]
                current_time = time.time()

                # Check health periodically
                if (current_time - self.last_health_check.get(client_type, 0) >
                        self.health_check_interval and not self.client_health.get(client_type, True)):
                    try:
                        self.client_health[client_type] = client.health_check()
                        self.last_health_check[client_type] = current_time
                    except Exception:
                        self.client_health[client_type] = False

                if self.client_health.get(client_type, True):
                    return client

            # Use default client if healthy
            default_client = self.default_client
            if self.client_health.get(default_client, True):
                return self.clients[default_client]

            # Fall back to filesystem which should always work
            return self.filesystem_client

    def _execute_with_retry(self, operation: str, func: callable, *args, **kwargs) -> Any:
        """Execute a database operation with retry logic."""
        client = kwargs.pop('client', None)
        client_type = kwargs.pop('client_type', None)
        if client is None:
            client = self._get_client(client_type)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                return func(client, *args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Database {operation} attempt {attempt+1}/{self.max_retries} failed: {
                    e
                }")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise last_error
                # No need for finally block here - the exception is already caught in the try/except

                # Sleep between retry attempts
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

            # If we exhaust all retries, raise the last error
            if last_error:
                logger.error(f"Database {operation} failed after {self.max_retries} attempts")
                raise last_error

                return None


        ###############################################################################
        #                        COMMAND-LINE INTERFACE                               #
        ###############################################################################


def main():
            """Command-line interface for WDBX database operations."""
            import argparse
            import sys

            # Set up argument parser
            parser = argparse.ArgumentParser(description="WDBX Database CLI")
            parser.add_argument('--host', default='localhost', help='Database host')
            parser.add_argument('--http-port', type=int, default=8080, help='HTTP port')
            parser.add_argument('--socket-port', type=int, default=9090, help='Socket port')
            parser.add_argument('--data-dir', default='./wdbx_data', help='Data directory for filesystem storage')
            parser.add_argument('--client', choices=['http', 'socket', 'filesystem'], default='auto',
                               help='Client type to use')

            # Create subparsers for different commands
            subparsers = parser.add_subparsers(dest='command', help='Command to execute')

            # Get command
            get_parser = subparsers.add_parser('get', help='Get a value by key')
            get_parser.add_argument('key', help='Key to retrieve')
            get_parser.add_argument('--output', '-o', help='Output file (stdout if not specified)')

            # Put command
            put_parser = subparsers.add_parser('put', help='Store a value with key')
            put_parser.add_argument('key', help='Key to store')
            put_parser.add_argument('--value', '-v', help='Value to store (as string)')
            put_parser.add_argument('--file', '-f', help='File containing value to store')

            # Delete command
            delete_parser = subparsers.add_parser('delete', help='Delete a value by key')
            delete_parser.add_argument('key', help='Key to delete')

            # Query command
            query_parser = subparsers.add_parser('query', help='Query database by parameters')
            query_parser.add_argument('--key-prefix', help='Filter by key prefix')
            query_parser.add_argument('--key-suffix', help='Filter by key suffix')
            query_parser.add_argument('--key-contains', help='Filter by substring in key')
            query_parser.add_argument('--key-exact', help='Filter by exact key match')
            query_parser.add_argument('--format', choices=['json', 'text'], default='text',
                                     help='Output format')

            # Health check command
            health_parser = subparsers.add_parser('health', help='Check database health')

            # Interactive mode
            interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')

            # Parse arguments
            args = parser.parse_args()

            if not args.command:
                parser.print_help()
                return 1

            # Set up logging
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # Initialize connection manager
            db_manager = DatabaseConnectionManager(
                http_host=args.host,
                http_port=args.http_port,
                socket_host=args.host,
                socket_port=args.socket_port,
                data_dir=args.data_dir,
                use_socket_by_default=(args.client == 'socket')
            )

            # Determine client type
            client_type = None if args.client == 'auto' else args.client

            try:
                if args.command == 'get':
                    # Get value
                    value = db_manager._execute_with_retry(
                        'get', lambda client, k: client.get(k), args.key, client_type=client_type
                    )

                    if value is None:
                        print(f"Key not found: {args.key}")
                        return 1

                    if args.output:
                        with open(args.output, 'wb') as f:
                            f.write(value)
                        print(f"Value written to {args.output}")
                    else:
                        # Try to decode as UTF-8 for display
                        try:
                            print(value.decode('utf-8'))
                        except UnicodeDecodeError:
                            print(f"Binary data retrieved ({len(value)} bytes)")

                elif args.command == 'put':
                    # Get value from either --value or --file
                    if args.value is not None:
                        value = args.value.encode('utf-8')
                    elif args.file:
                        with open(args.file, 'rb') as f:
                            value = f.read()
                    else:
                        print("Error: Either --value or --file must be specified")
                        return 1

                    success = db_manager._execute_with_retry(
                        'put', lambda client, k, v: client.put(k, v), args.key, value, client_type=client_type
                    )

                    if success:
                        print(f"Successfully stored key: {args.key} ({len(value)} bytes)")
                    else:
                        print(f"Failed to store key: {args.key}")
                        return 1

                elif args.command == 'delete':
                    success = db_manager._execute_with_retry(
                        'delete', lambda client, k: client.delete(k), args.key, client_type=client_type
                    )

                    if success:
                        print(f"Successfully deleted key: {args.key}")
                    else:
                        print(f"Failed to delete key: {args.key}")
                        return 1

                elif args.command == 'query':
                    # Build query parameters
                    query_params = {}
                    if args.key_prefix:
                        query_params['key_prefix'] = args.key_prefix
                    if args.key_suffix:
                        query_params['key_suffix'] = args.key_suffix
                    if args.key_contains:
                        query_params['key_contains'] = args.key_contains
                    if args.key_exact:
                        query_params['key_exact'] = args.key_exact

                    if not query_params:
                        print("Error: At least one query parameter must be specified")
                        return 1

                    results = db_manager._execute_with_retry(
                        'query', lambda client, qp: client.query(qp), query_params, client_type=client_type
                    )

                    if args.format == 'json':
                        print(json.dumps(results, indent=2))
                    else:
                        print(f"Found {len(results)} results:")
                        for i, item in enumerate(results, 1):
                            print(f"{i}. Key: {item.get('key')}")
                            print(f"   Size: {item.get('original_size', 'unknown')} bytes")
                            print(f"   Timestamp: {time.ctime(item.get('timestamp', 0))}")
                            print()

                elif args.command == 'health':
                    clients = ['http', 'socket', 'filesystem']
                    for client_name in clients:
                        client = db_manager.clients[client_name]
                        try:
                            is_healthy = client.health_check()
                            status = "HEALTHY" if is_healthy else "UNHEALTHY"
                            print(f"{client_name.upper()} client: {status}")
                        except Exception as e:
                            print(f"{client_name.upper()} client: ERROR - {str(e)}")

                elif args.command == 'interactive':
                    print("WDBX Database Interactive Shell")
                    print("Type 'help' for available commands, 'exit' to quit")

                    while True:
                        try:
                            cmd = input("> ").strip().split()
                            if not cmd:
                                continue

                            if cmd[0] == 'exit':
                                break
                            elif cmd[0] == 'help':
                                print("Available commands:")
                                print("  get <key>                  - Retrieve value by key")
                                print("  put <key> <value>          - Store string value with key")
                                print("  putfile <key> <filename>   - Store file contents with key")
                                print("  delete <key>               - Delete value by key")
                                print("  query <param> <value>      - Query database (param: prefix, suffix, contains, exact)")
                                print("  health                     - Check database health")
                                print("  client <type>              - Switch client type (http, socket, filesystem)")
                                print("  exit                       - Exit interactive mode")
                            elif cmd[0] == 'get' and len(cmd) == 2:
                                value = db_manager._execute_with_retry(
                                    'get', lambda client, k: client.get(k), cmd[1], client_type=client_type
                                )
                                if value is None:
                                    print(f"Key not found: {cmd[1]}")
                                else:
                                    try:
                                        print(value.decode('utf-8'))
                                    except UnicodeDecodeError:
                                        print(f"Binary data retrieved ({len(value)} bytes)")
                            elif cmd[0] == 'put' and len(cmd) >= 3:
                                value = ' '.join(cmd[2:]).encode('utf-8')
                                success = db_manager._execute_with_retry(
                                    'put', lambda client, k, v: client.put(k, v), cmd[1], value, client_type=client_type
                                )
                                if success:
                                    print(f"Successfully stored key: {cmd[1]}")
                                else:
                                    print(f"Failed to store key: {cmd[1]}")
                            elif cmd[0] == 'putfile' and len(cmd) == 3:
                                try:
                                    with open(cmd[2], 'rb') as f:
                                        value = f.read()
                                    success = db_manager._execute_with_retry(
                                        'put', lambda client, k, v: client.put(k, v), cmd[1], value, client_type=client_type
                                    )
                                    if success:
                                        print(f"Successfully stored file as key: {cmd[1]}")
                                    else:
                                        print(f"Failed to store file as key: {cmd[1]}")
                                except FileNotFoundError:
                                    print(f"File not found: {cmd[2]}")
                            elif cmd[0] == 'delete' and len(cmd) == 2:
                                success = db_manager._execute_with_retry(
                                    'delete', lambda client, k: client.delete(k), cmd[1], client_type=client_type
                                )
                                if success:
                                    print(f"Successfully deleted key: {cmd[1]}")
                                else:
                                    print(f"Failed to delete key: {cmd[1]}")
                            elif cmd[0] == 'query' and len(cmd) == 3:
                                param, value = cmd[1], cmd[2]
                                query_params = {}
                                if param == 'prefix':
                                    query_params['key_prefix'] = value
                                elif param == 'suffix':
                                    query_params['key_suffix'] = value
                                elif param == 'contains':
                                    query_params['key_contains'] = value
                                elif param == 'exact':
                                    query_params['key_exact'] = value
                                else:
                                    print(f"Unknown query parameter: {param}")
                                    continue

                                results = db_manager._execute_with_retry(
                                    'query', lambda client, qp: client.query(qp), query_params, client_type=client_type
                                )
                                print(f"Found {len(results)} results:")
                                for i, item in enumerate(results, 1):
                                    print(f"{i}. Key: {item.get('key')}")
                                    print(f"   Size: {item.get('original_size', 'unknown')} bytes")
                                    print(f"   Timestamp: {time.ctime(item.get('timestamp', 0))}")
                                    print()
                            elif cmd[0] == 'health':
                                clients = ['http', 'socket', 'filesystem']
                                for client_name in clients:
                                    client = db_manager.clients[client_name]
                                    try:
                                        is_healthy = client.health_check()
                                        status = "HEALTHY" if is_healthy else "UNHEALTHY"
                                        print(f"{client_name.upper()} client: {status}")
                                    except Exception as e:
                                        print(f"{client_name.upper()} client: ERROR - {str(e)}")
                            elif cmd[0] == 'client' and len(cmd) == 2:
                                if cmd[1] in ['http', 'socket', 'filesystem', 'auto']:
                                    client_type = None if cmd[1] == 'auto' else cmd[1]
                                    print(f"Switched to client type: {cmd[1]}")
                                else:
                                    print(f"Unknown client type: {cmd[1]}")
                            else:
                                print("Unknown command or invalid syntax. Type 'help' for help.")
                        except KeyboardInterrupt:
                            print("\nOperation cancelled")
                        except Exception as e:
                            print(f"Error: {str(e)}")
            finally:
                db_manager.close_all_connections()

            return 0

if __name__ == "__main__":
    import asyncio

    async def async_main():
        return main()

    sys.exit(asyncio.run(async_main()))
