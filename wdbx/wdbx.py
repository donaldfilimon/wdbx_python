#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WDBX - Wide Distributed Block Exchange
-------------------------------------
Core implementation module that integrates all WDBX components.

This module provides:
- Vector operations and embedding functionality
- Block chain implementation with cryptographic validation
- Database interfaces for various storage backends
- MVCC (Multi-Version Concurrency Control) for transaction management
- Neural backtracking for explainable AI reasoning
- Multi-persona management for AI systems
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
import platform
import tempfile
import shutil
import socket
import logging
import concurrent.futures
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Callable, TypeVar, Generic
from collections import defaultdict, deque
from functools import lru_cache
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("WDBX")

# Try to import optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    logger.info("NumPy available - using optimized vector operations")
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    logger.warning("NumPy not available - using simplified vector operations")

try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS available - using optimized vector search")
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    logger.warning("FAISS not available - using basic vector search")

try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch available - using optimized embeddings")
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    logger.warning("PyTorch not available - using simplified embeddings")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
    logger.info("SentenceTransformer available - using pre-trained embeddings")
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformer = None
    logger.warning("SentenceTransformer not available - using simple embeddings")

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
    logger.info("aiohttp available - HTTP server functionality enabled")
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None
    logger.warning("aiohttp not available - HTTP server functionality disabled")

# Import WDBX submodules if available in the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(current_dir, "__init__.py")):
    try:
        from . import (
            constants,
            data_structures,
            blockchain,
            vector_store,
            mvcc,
            neural_backtracking,
            attention,
            persona,
            content_filter,
            shard_manager
        )
        # Use imported modules in preference to local implementations
        MODULES_AVAILABLE = True
        logger.info("Using WDBX submodules")
    except ImportError:
        MODULES_AVAILABLE = False
        logger.warning("WDBX submodules not found - using integrated implementation")
else:
    MODULES_AVAILABLE = False
    logger.warning("Not in a package context - using integrated implementation")

# Constants
VECTOR_DIMENSION = 1024
SHARD_COUNT = 8
DEFAULT_SIMILARITY_THRESHOLD = 0.6
MAX_BATCH_SIZE = 1000
BLOCKCHAIN_DIFFICULTY = 2
DEFAULT_DATA_DIR = "./wdbx_data"
HTTP_HOST = "localhost"
HTTP_PORT = 8080
SOCKET_HOST = "localhost"
SOCKET_PORT = 9090
MAX_RETRIES = 3
ENABLE_CACHING = True
CACHE_TTL = 60 * 60  # 1 hour
USE_COMPRESSION = True

# Version information
__version__ = "1.0.1"

###############################################################################
#                              UTILITY CLASSES                                #
###############################################################################

class NumpyLikeArray:
    """
    A numpy-like array implementation for vector operations
    when numpy is not available.
    """
    def __init__(self, data):
        if isinstance(data, list):
            self.data = data
        else:
            self.data = list(data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"NumpyLikeArray({self.data})"

    def __iter__(self):
        return iter(self.data)

    def tolist(self):
        return self.data

T = TypeVar('T')

class LRUCache(Generic[T]):
    """LRU Cache implementation"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> (value, timestamp)
        self.lock = threading.RLock()

    def get(self, key: Any) -> Optional[T]:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                return None
            # Update timestamp on access
            value, _ = self.cache[key]
            self.cache[key] = (value, time.time())
            return value

    def put(self, key: Any, value: T) -> None:
        """Put value into cache"""
        with self.lock:
            # If cache is full, remove least recently used item
            if len(self.cache) >= self.capacity and key not in self.cache:
                self._evict_lru()
            self.cache[key] = (value, time.time())

    def remove(self, key: Any) -> None:
        """Remove key from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]

    def clear(self) -> None:
        """Clear the cache"""
        with self.lock:
            self.cache.clear()

    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self.cache:
            return
        # Find key with oldest timestamp
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
        del self.cache[lru_key]

    def __contains__(self, key: Any) -> bool:
        """Check if key in cache"""
        with self.lock:
            return key in self.cache

@contextmanager
def error_handling(operation_name, fallback_value=None):
    """Context manager for consistent error handling"""
    try:
        yield
    except Exception as e:
        logger.error(f"Error during {operation_name}: {e}", exc_info=True)
        return fallback_value

###############################################################################
#                           CORE DATA STRUCTURES                              #
###############################################################################

class EmbeddingVector:
    """Represents a vector embedding with metadata"""
    def __init__(self, vector, metadata=None):
        self.vector = vector
        self.metadata = metadata or {}

    def normalize(self):
        """Normalize the vector to unit length"""
        if NUMPY_AVAILABLE and isinstance(self.vector, np.ndarray):
            norm = np.linalg.norm(self.vector)
            if norm > 0:
                return EmbeddingVector(self.vector / norm, self.metadata)
            return EmbeddingVector(self.vector, self.metadata)

        # Pure Python implementation
        vec = self.vector.tolist() if hasattr(self.vector, 'tolist') else self.vector
        norm = math.sqrt(sum(x*x for x in vec))
        if norm > 0:
            return EmbeddingVector([x/norm for x in vec], self.metadata)
        return EmbeddingVector(vec, self.metadata)

    def serialize(self):
        """Serialize to bytes for storage"""
        return pickle.dumps((self.vector, self.metadata))

    @classmethod
    def deserialize(cls, data):
        """Create an EmbeddingVector from serialized bytes"""
        vector, metadata = pickle.loads(data)
        return cls(vector, metadata)

class Block:
    """Represents a block in the blockchain"""
    def __init__(self, id=None, prev_hash=None, timestamp=None, data=None, embeddings=None, hash=None, nonce=0):
        self.id = id or str(uuid.uuid4())
        self.prev_hash = prev_hash
        self.timestamp = timestamp or time.time()
        self.data = data or {}
        self.embeddings = embeddings or []
        self.nonce = nonce
        self.hash = hash or self.calculate_hash()

    def calculate_hash(self):
        """Calculate the cryptographic hash of this block"""
        # Convert data to a stable string representation
        data_str = json.dumps(self.data, sort_keys=True)

        # For efficiency, just use the IDs from embeddings
        embedding_ids = [e.metadata.get('vector_id', '') for e in self.embeddings]
        embedding_str = json.dumps(embedding_ids, sort_keys=True)

        block_content = f"{self.id}{self.prev_hash or ''}{self.timestamp}{data_str}{embedding_str}{self.nonce}"
        return hashlib.sha256(block_content.encode()).hexdigest()

    def mine_block(self, difficulty=BLOCKCHAIN_DIFFICULTY):
        """
        Mine the block by finding a nonce that produces a hash with the required difficulty.
        The difficulty is the number of leading zeros required in the hash.
        """
        target = '0' * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.calculate_hash()
        return self.hash

    def validate(self):
        """Validate block integrity"""
        return self.hash == self.calculate_hash()

    def serialize(self):
        """Serialize to bytes for storage"""
        # Serialize embeddings separately
        embedding_data = [e.serialize() for e in self.embeddings]
        block_data = {
            'id': self.id,
            'prev_hash': self.prev_hash,
            'timestamp': self.timestamp,
            'data': self.data,
            'embedding_data': embedding_data,
            'hash': self.hash,
            'nonce': self.nonce
        }
        return pickle.dumps(block_data)

    @classmethod
    def deserialize(cls, data):
        """Create a Block from serialized bytes"""
        block_data = pickle.loads(data)
        # Deserialize embeddings
        embeddings = [EmbeddingVector.deserialize(e) for e in block_data['embedding_data']]

        return cls(
            id=block_data['id'],
            prev_hash=block_data['prev_hash'],
            timestamp=block_data['timestamp'],
            data=block_data['data'],
            embeddings=embeddings,
            hash=block_data['hash'],
            nonce=block_data['nonce']
        )

class ShardInfo:
    """Information about a shard in the distributed system"""
    def __init__(self, shard_id, host=None, port=None, status="active"):
        self.shard_id = shard_id
        self.host = host
        self.port = port
        self.status = status
        self.last_heartbeat = time.time()

    def is_active(self, timeout_seconds=60):
        """Check if shard is currently active based on heartbeat"""
        return (self.status == "active" and
                time.time() - self.last_heartbeat < timeout_seconds)

class MVCCTransaction:
    """Represents a transaction in the MVCC system"""
    def __init__(self, transaction_id=None):
        self.transaction_id = transaction_id or str(uuid.uuid4())
        self.start_time = time.time()
        self.status = "active"  # active, committed, aborted
        self.read_set = set()
        self.write_set = set()

class SimpleEmbeddingModel:
    """
    A simple embedding model implementation when SentenceTransformer is not available.
    Uses random projections to create embeddings.
    """
    def __init__(self, dimension=VECTOR_DIMENSION):
        self.dimension = dimension
        # Create a simple random projection matrix
        if NUMPY_AVAILABLE:
            self.projection_matrix = np.random.randn(5000, dimension).astype(np.float32)
            self.projection_matrix /= np.linalg.norm(self.projection_matrix, axis=1, keepdims=True) + 1e-8
        else:
            self.projection_matrix = [[random.gauss(0, 1) for _ in range(dimension)] for _ in range(5000)]
            # Normalize each row
            for i in range(len(self.projection_matrix)):
                norm = math.sqrt(sum(x*x for x in self.projection_matrix[i]))
                if norm > 0:
                    self.projection_matrix[i] = [x/norm for x in self.projection_matrix[i]]

    def encode(self, text, **kwargs):
        """
        Create a simple embedding for the given text using random projections
        """
        if not text:
            if NUMPY_AVAILABLE:
                return np.zeros(self.dimension, dtype=np.float32)
            else:
                return [0.0] * self.dimension

        # Simple character-level hashing for consistency
        hash_val = sum(ord(c) for c in text) % 5000

        if NUMPY_AVAILABLE:
            embedding = self.projection_matrix[hash_val].copy()

            # Add some variation based on text length
            embedding *= (0.8 + 0.4 * (len(text) % 10) / 10)

            # Add some variation based on first characters
            if len(text) > 0:
                embedding += 0.1 * self.projection_matrix[ord(text[0]) % 5000]

            # Normalize
            embedding /= np.linalg.norm(embedding) + 1e-8
            return embedding
        else:
            embedding = list(self.projection_matrix[hash_val])

            # Add some variation based on text length
            factor = (0.8 + 0.4 * (len(text) % 10) / 10)
            embedding = [x * factor for x in embedding]

            # Add some variation based on first characters
            if len(text) > 0:
                first_char_embedding = self.projection_matrix[ord(text[0]) % 5000]
                embedding = [e + 0.1 * f for e, f in zip(embedding, first_char_embedding)]

            # Normalize
            norm = math.sqrt(sum(x*x for x in embedding))
            if norm > 0:
                embedding = [x/norm for x in embedding]

            return embedding

###############################################################################
#                          DATABASE CLIENT INTERFACE                          #
###############################################################################

class DatabaseClient(ABC):
    """Abstract base class for database clients"""
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the database"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the database"""
        pass

    @abstractmethod
    def store(self, key, value) -> bool:
        """Store a value in the database"""
        pass

    @abstractmethod
    def retrieve(self, key) -> Any:
        """Retrieve a value from the database"""
        pass

    @abstractmethod
    def delete(self, key) -> bool:
        """Delete a value from the database"""
        pass

    @abstractmethod
    def exists(self, key) -> bool:
        """Check if key exists in the database"""
        pass

    def health_check(self) -> bool:
        """Check if database is healthy and accessible"""
        try:
            # Generate a random test key
            test_key = f"health_check_{uuid.uuid4()}"
            test_value = {"timestamp": time.time()}

            # Try store and retrieve
            store_result = self.store(test_key, test_value)
            if not store_result:
                return False

            retrieved = self.retrieve(test_key)
            if retrieved is None or retrieved.get("timestamp") != test_value.get("timestamp"):
                return False

            # Clean up test key
            self.delete(test_key)
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

class HttpDatabaseClient(DatabaseClient):
    """Client that connects to database over HTTP"""
    def __init__(self, host=HTTP_HOST, port=HTTP_PORT, connection_timeout=5, retry_count=MAX_RETRIES):
        self.host = host
        self.port = port
        self.connected = False
        self.base_url = f"http://{host}:{port}"
        self.connection_timeout = connection_timeout
        self.retry_count = retry_count
        self.session = None

    def connect(self) -> bool:
        """Connect to HTTP database server"""
        try:
            # Try to connect and check if server is responsive
            url = f"{self.base_url}/health"

            # Use urllib3 or requests if available for connection pooling
            try:
                import requests
                self.session = requests.Session()
                response = self.session.get(url, timeout=self.connection_timeout)
                self.connected = response.status_code == 200
            except ImportError:
                # Fallback to urllib
                import urllib.request
                with urllib.request.urlopen(url, timeout=self.connection_timeout) as response:
                    self.connected = response.status == 200

            if self.connected:
                logger.info(f"Connected to HTTP database at {self.base_url}")
            return self.connected

        except Exception as e:
            logger.warning(f"Failed to connect to HTTP database: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from HTTP database"""
        if self.session:
            try:
                self.session.close()
            except Exception:
                pass
            self.session = None

        self.connected = False
        logger.info("Disconnected from HTTP database")
        return True

    def store(self, key, value) -> bool:
        """Store data via HTTP POST request"""
        if not self.connected:
            raise ConnectionError("Not connected to database")

        for attempt in range(self.retry_count):
            try:
                url = f"{self.base_url}/store/{key}"
                data = pickle.dumps(value)
                headers = {"Content-Type": "application/octet-stream"}

                if self.session:
                    response = self.session.post(url, data=data, headers=headers, timeout=self.connection_timeout)
                    return response.status_code == 200
                else:
                    import urllib.request
                    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=self.connection_timeout) as response:
                        return response.status == 200

            except Exception as e:
                logger.error(f"Error storing data via HTTP (attempt {attempt+1}/{self.retry_count}): {e}")
                if attempt == self.retry_count - 1:
                    return False
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff

        return False

    def retrieve(self, key) -> Any:
        """Retrieve data via HTTP GET request"""
        if not self.connected:
            raise ConnectionError("Not connected to database")

        for attempt in range(self.retry_count):
            try:
                url = f"{self.base_url}/retrieve/{key}"

                if self.session:
                    response = self.session.get(url, timeout=self.connection_timeout)
                    if response.status_code == 200:
                        return pickle.loads(response.content)
                    elif response.status_code == 404:
                        return None
                else:
                    import urllib.request
                    import urllib.error
                    try:
                        with urllib.request.urlopen(url, timeout=self.connection_timeout) as response:
                            if response.status == 200:
                                return pickle.loads(response.read())
                    except urllib.error.HTTPError as e:
                        if e.code == 404:
                            return None
                        raise
                return None

            except Exception as e:
                logger.error(f"Error retrieving data via HTTP (attempt {attempt+1}/{self.retry_count}): {e}")
                if attempt == self.retry_count - 1:
                    return None
                time.sleep(0.5 * (attempt + 1))

        return None

    def delete(self, key) -> bool:
        """Delete data via HTTP DELETE request"""
        if not self.connected:
            raise ConnectionError("Not connected to database")

        for attempt in range(self.retry_count):
            try:
                url = f"{self.base_url}/delete/{key}"

                if self.session:
                    response = self.session.delete(url, timeout=self.connection_timeout)
                    return response.status_code in (200, 204)
                else:
                    import urllib.request
                    req = urllib.request.Request(url, method="DELETE")
                    with urllib.request.urlopen(req, timeout=self.connection_timeout) as response:
                        return response.status in (200, 204)

            except Exception as e:
                logger.error(f"Error deleting data via HTTP (attempt {attempt+1}/{self.retry_count}): {e}")
                if attempt == self.retry_count - 1:
                    return False
                time.sleep(0.5 * (attempt + 1))

        return False

    def exists(self, key) -> bool:
        """Check if key exists via HTTP HEAD request"""
        if not self.connected:
            raise ConnectionError("Not connected to database")

        for attempt in range(self.retry_count):
            try:
                url = f"{self.base_url}/exists/{key}"

                if self.session:
                    response = self.session.head(url, timeout=self.connection_timeout)
                    return response.status_code == 200
                else:
                    import urllib.request
                    import urllib.error
                    req = urllib.request.Request(url, method="HEAD")
                    try:
                        with urllib.request.urlopen(req, timeout=self.connection_timeout) as response:
                            return response.status == 200
                    except urllib.error.HTTPError as e:
                        if e.code == 404:
                            return False
                        raise

            except Exception as e:
                logger.error(f"Error checking key existence via HTTP (attempt {attempt+1}/{self.retry_count}): {e}")
                if attempt == self.retry_count - 1:
                    return False
                time.sleep(0.5 * (attempt + 1))

        return False

class SocketDatabaseClient(DatabaseClient):
    """Client that connects to database over socket"""
    def __init__(self, host=SOCKET_HOST, port=SOCKET_PORT, timeout=10, retry_attempts=MAX_RETRIES):
        self.host = host
        self.port = port
        self.socket = None
        self.timeout = timeout
        self.buffer_size = 4096
        self.retry_attempts = retry_attempts
        self.lock = threading.RLock()  # For thread safety

    def connect(self) -> bool:
        """Connect to socket database server"""
        with self.lock:
            try:
                if self.socket:
                    # Already connected
                    return True

                import socket
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(self.timeout)
                self.socket.connect((self.host, self.port))
                logger.info(f"Connected to socket database at {self.host}:{self.port}")
                return True
            except Exception as e:
                logger.warning(f"Failed to connect to socket database: {e}")
                self.socket = None
                return False

    def disconnect(self) -> bool:
        """Disconnect from socket database"""
        with self.lock:
            if self.socket:
                try:
                    self.socket.close()
                    logger.info("Disconnected from socket database")
                except Exception as e:
                    logger.warning(f"Error disconnecting from socket: {e}")
                finally:
                    self.socket = None
            return True

    def store(self, key, value) -> bool:
        """Store data via socket connection"""
        return self._retry_operation("store", lambda: self._store_impl(key, value))

    def _store_impl(self, key, value) -> bool:
        """Implementation of store operation with proper locking"""
        with self.lock:
            if not self.socket and not self.connect():
                raise ConnectionError("Not connected to database")

            try:
                # Prepare message: STORE <key> <data>
                data = pickle.dumps(value)
                message = f"STORE {key}".encode() + b" " + data

                # Send message
                self.socket.sendall(message)

                # Receive acknowledgment
                response = self.socket.recv(self.buffer_size).decode().strip()
                return response == "OK"
            except Exception as e:
                logger.error(f"Error storing data via socket: {e}")
                self.disconnect()
                raise

    def retrieve(self, key) -> Any:
        """Retrieve data via socket connection"""
        return self._retry_operation("retrieve", lambda: self._retrieve_impl(key))

    def _retrieve_impl(self, key) -> Any:
        """Implementation of retrieve operation with proper locking"""
        with self.lock:
            if not self.socket and not self.connect():
                raise ConnectionError("Not connected to database")

            try:
                # Send request: GET <key>
                message = f"GET {key}".encode()
                self.socket.sendall(message)

                # Receive data
                chunks = []
                while True:
                    chunk = self.socket.recv(self.buffer_size)
                    if not chunk or chunk.endswith(b"END"):
                        break
                    chunks.append(chunk)

                if not chunks:
                    return None

                # Process and deserialize the data
                data = b"".join(chunks).rstrip(b"END")
                return pickle.loads(data) if data else None
            except Exception as e:
                logger.error(f"Error retrieving data via socket: {e}")
                self.disconnect()
                raise

    def delete(self, key) -> bool:
        """Delete data via socket connection"""
        return self._retry_operation("delete", lambda: self._delete_impl(key))

    def _delete_impl(self, key) -> bool:
        """Implementation of delete operation with proper locking"""
        with self.lock:
            if not self.socket and not self.connect():
                raise ConnectionError("Not connected to database")

            try:
                # Send request: DELETE <key>
                message = f"DELETE {key}".encode()
                self.socket.sendall(message)

                # Receive acknowledgment
                response = self.socket.recv(self.buffer_size).decode().strip()
                return response == "OK"
            except Exception as e:
                logger.error(f"Error deleting data via socket: {e}")
                self.disconnect()
                raise

    def exists(self, key) -> bool:
        """Check if key exists via socket connection"""
        return self._retry_operation("exists", lambda: self._exists_impl(key))

    def _exists_impl(self, key) -> bool:
        """Implementation of exists operation with proper locking"""
        with self.lock:
            if not self.socket and not self.connect():
                raise ConnectionError("Not connected to database")

            try:
                # Send request: EXISTS <key>
                message = f"EXISTS {key}".encode()
                self.socket.sendall(message)

                # Receive response
                response = self.socket.recv(self.buffer_size).decode().strip()
                return response == "TRUE"
            except Exception as e:
                logger.error(f"Error checking key existence via socket: {e}")
                self.disconnect()
                raise

    def _retry_operation(self, operation_name, operation_func):
        """Retry a database operation with exponential backoff"""
        for attempt in range(self.retry_attempts):
            try:
                return operation_func()
            except Exception as e:
                logger.warning(f"Operation {operation_name} failed (attempt {attempt+1}/{self.retry_attempts}): {e}")
                if attempt == self.retry_attempts - 1:
                    if operation_name == "retrieve":
                        return None
                    return False
                # Exponential backoff
                time.sleep(0.5 * (2 ** attempt))
                # Try to reconnect before next attempt
                self.connect()

        # Should never reach here due to return in the loop
        return None if operation_name == "retrieve" else False

class FilesystemDatabaseClient(DatabaseClient):
    """Client that stores data in the filesystem"""
    def __init__(self, data_dir=DEFAULT_DATA_DIR, use_compression=USE_COMPRESSION):
        self.data_dir = data_dir
        self.connected = False
        self.use_compression = use_compression
        self.cache = LRUCache(1000) if ENABLE_CACHING else None

    def connect(self) -> bool:
        """Initialize filesystem storage"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)

            # Create subdirectories for better file distribution
            for i in range(16):
                subdir = os.path.join(self.data_dir, f"{i:x}")
                os.makedirs(subdir, exist_ok=True)

            # Check if the directory is writable
            test_file = os.path.join(self.data_dir, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)

            self.connected = True
            logger.info(f"Initialized filesystem database at {self.data_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize filesystem database: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from filesystem storage"""
        self.connected = False
        if self.cache:
            self.cache.clear()
        logger.info("Disconnected from filesystem database")
        return True

    def store(self, key, value) -> bool:
        """Store data in filesystem"""
        if not self.connected:
            raise ConnectionError("Not connected to database")

        try:
            # Ensure key is filesystem-safe
            safe_key = hashlib.md5(str(key).encode()).hexdigest()
            subdir = os.path.join(self.data_dir, safe_key[0])
            file_path = os.path.join(subdir, f"{safe_key}.pickle")

            # Serialize data, optionally with compression
            if self.use_compression:
                try:
                    import gzip
                    serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                    data = gzip.compress(serialized)
                except ImportError:
                    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

            # Write data atomically using a temporary file
            with tempfile.NamedTemporaryFile(dir=subdir, delete=False) as temp_file:
                temp_file.write(data)
                temp_path = temp_file.name

            # Move the file atomically to ensure consistency
            shutil.move(temp_path, file_path)

            # Update cache
            if self.cache:
                self.cache.put(key, value)

            return True
        except Exception as e:
            logger.error(f"Error storing data in filesystem: {e}")
            return False

    def retrieve(self, key) -> Any:
        """Retrieve data from filesystem"""
        if not self.connected:
            raise ConnectionError("Not connected to database")

        # Check cache first
        if self.cache:
            cached_value = self.cache.get(key)
            if cached_value is not None:
                return cached_value

        try:
            # Ensure key is filesystem-safe
            safe_key = hashlib.md5(str(key).encode()).hexdigest()
            subdir = os.path.join(self.data_dir, safe_key[0])
            file_path = os.path.join(subdir, f"{safe_key}.pickle")

            if not os.path.exists(file_path):
                return None

            with open(file_path, 'rb') as f:
                data = f.read()

            # Detect and decompress if needed
            if self.use_compression:
                try:
                    import gzip
                    # Check if data is compressed
                    if data[:2] == b'\x1f\x8b':  # gzip magic number
                        data = gzip.decompress(data)
                except ImportError:
                    pass  # Compression not available

            value = pickle.loads(data)

            # Update cache
            if self.cache:
                self.cache.put(key, value)

            return value
        except Exception as e:
            logger.error(f"Error retrieving data from filesystem: {e}")
            return None

    def delete(self, key) -> bool:
        """Delete data from filesystem"""
        if not self.connected:
            raise ConnectionError("Not connected to database")

        try:
            # Remove from cache
            if self.cache:
                self.cache.remove(key)

            # Ensure key is filesystem-safe
            safe_key = hashlib.md5(str(key).encode()).hexdigest()
            subdir = os.path.join(self.data_dir, safe_key[0])
            file_path = os.path.join(subdir, f"{safe_key}.pickle")

            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting data from filesystem: {e}")
            return False

    def exists(self, key) -> bool:
        """Check if key exists in filesystem"""
        if not self.connected:
            raise ConnectionError("Not connected to database")

        # Check cache first
        if self.cache and key in self.cache:
            return True

        try:
            # Ensure key is filesystem-safe
            safe_key = hashlib.md5(str(key).encode()).hexdigest()
            subdir = os.path.join(self.data_dir, safe_key[0])
            file_path = os.path.join(subdir, f"{safe_key}.pickle")

            return os.path.exists(file_path)
        except Exception as e:
            logger.error(f"Error checking key existence in filesystem: {e}")
            return False

class DatabaseConnectionManager:
    """Manages connections to different database backends"""
    def __init__(self, http_host=HTTP_HOST, http_port=HTTP_PORT,
                 socket_host=SOCKET_HOST, socket_port=SOCKET_PORT,
                 data_dir=DEFAULT_DATA_DIR, use_socket_by_default=True):
        self.http_client = HttpDatabaseClient(http_host, http_port)
        self.socket_client = SocketDatabaseClient(socket_host, socket_port)
        self.fs_client = FilesystemDatabaseClient(data_dir)
        self.default_client = "socket" if use_socket_by_default else "filesystem"
        self.clients = {}
        self.active_connections = set()
        self.lock = threading.RLock()

    def get_client(self, client_type=None):
        """Get a client by type and ensure it's connected"""
        with self.lock:
            if client_type is None:
                client_type = self.default_client

            if client_type == "http":
                client = self.http_client
            elif client_type == "socket":
                client = self.socket_client
            elif client_type == "filesystem":
                client = self.fs_client
            else:
                raise ValueError(f"Unknown client type: {client_type}")

            # Ensure client is connected
            if client_type not in self.active_connections:
                if client.connect():
                    self.active_connections.add(client_type)
                else:
                    # If primary client fails, try fallback to filesystem
                    if client_type != "filesystem":
                        logger.warning(f"Failed to connect to {client_type} database, falling back to filesystem")
                        client = self.fs_client
                        client_type = "filesystem"
                        if client.connect():
                            self.active_connections.add(client_type)
                        else:
                            raise ConnectionError("Failed to connect to any database backend")

            return client

    def close_all_connections(self):
        """Close all active database connections"""
        with self.lock:
            for client_type in list(self.active_connections):
                client = self.get_client(client_type)
                try:
                    client.disconnect()
                    self.active_connections.remove(client_type)
                except Exception as e:
                    logger.error(f"Error closing {client_type} connection: {e}")

###############################################################################
#                        VECTOR STORE AND SEARCH INDEX                        #
###############################################################################

class SearchIndex:
    """Abstract base class for vector search indexes"""
    @abstractmethod
    def add_vector(self, id, vector) -> bool:
        """Add a vector to the index"""
        pass

    @abstractmethod
    def search(self, query_vector, top_k=5) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        pass

    @abstractmethod
    def remove_vector(self, id) -> bool:
        """Remove a vector from the index"""
        pass

    @abstractmethod
    def get_vector(self, id) -> Optional[Any]:
        """Get a vector by ID"""
        pass

    @abstractmethod
    def count(self) -> int:
        """Get the number of vectors in the index"""
        pass

class BasicSearchIndex(SearchIndex):
    """Basic vector search index implementation"""
    def __init__(self, dimension):
        self.dimension = dimension
        self.vectors = {}  # id -> vector mapping
        self.lock = threading.RLock()

    def add_vector(self, id, vector) -> bool:
        """Add a vector to the index"""
        with self.lock:
            self.vectors[id] = vector
            return True

    def search(self, query_vector, top_k=5) -> List[Tuple[str, float]]:
        """Search for similar vectors using cosine similarity"""
        with self.lock:
            if not self.vectors:
                return []

            similarities = []
            for id, vector in self.vectors.items():
                sim = self._cosine_similarity(query_vector, vector)
                similarities.append((id, sim))

            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

    def remove_vector(self, id) -> bool:
        """Remove a vector from the index"""
        with self.lock:
            if id in self.vectors:
                del self.vectors[id]
                return True
            return False

    def _cosine_similarity(self, v1, v2):
        """Calculate cosine similarity between two vectors"""
        # Convert to list for uniform processing
        v1_list = v1 if isinstance(v1, list) else list(v1)
        v2_list = v2 if isinstance(v2, list) else list(v2)

        # Check dimensions match
        if len(v1_list) != len(v2_list):
            raise ValueError(f"Vector dimensions don't match: {len(v1_list)} vs {len(v2_list)}")

        dot_product = sum(a*b for a, b in zip(v1_list, v2_list))
        norm_v1 = math.sqrt(sum(a*a for a in v1_list))
        norm_v2 = math.sqrt(sum(b*b for b in v2_list))

        if norm_v1 == 0 or norm_v2 == 0:
            return 0

        return dot_product / (norm_v1 * norm_v2)

    def get_vector(self, id) -> Optional[Any]:
        """Get a vector by ID"""
        with self.lock:
            return self.vectors.get(id)

    def count(self) -> int:
        """Get the number of vectors in the index"""
        with self.lock:
            return len(self.vectors)

class FaissSearchIndex(SearchIndex):
    """FAISS-based vector search index for efficient similarity search"""
    def __init__(self, dimension):
        if not FAISS_AVAILABLE or faiss is None:
            raise ImportError("FAISS library is required for FaissSearchIndex")

        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
        self.id_map = {}  # position -> id mapping
        self.reverse_map = {}  # id -> position mapping
        self.vector_store = {}  # id -> vector mapping for retrieval
        self.lock = threading.RLock()

    def add_vector(self, id, vector) -> bool:
        """Add a vector to the FAISS index"""
        with self.lock:
            if id in self.reverse_map:
                self.remove_vector(id)

            # Convert to numpy array
            if not isinstance(vector, np.ndarray):
                vector_np = np.array(vector, dtype=np.float32).reshape(1, -1)
            else:
                vector_np = vector.reshape(1, -1).astype(np.float32)

            # Add to FAISS index
            position = len(self.id_map)
            self.index.add(vector_np)
            self.id_map[position] = id
            self.reverse_map[id] = position

            # Store the original vector for retrieval
            self.vector_store[id] = vector

            return True

    def search(self, query_vector, top_k=5) -> List[Tuple[str, float]]:
        """Search for similar vectors using FAISS"""
        with self.lock:
            if not self.id_map:
                return []

            # Convert to numpy array
            if not isinstance(query_vector, np.ndarray):
                query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            else:
                query_vector = query_vector.reshape(1, -1).astype(np.float32)

            # Perform search
            distances, indices = self.index.search(query_vector, min(top_k, len(self.id_map)))

            # Convert to (id, similarity) format
            results = []
            for i in range(len(indices[0])):
                position = indices[0][i]
                distance = distances[0][i]

                # Convert distance to similarity (smaller distance = higher similarity)
                # For L2 distance: similarity = 1 / (1 + distance)
                similarity = 1.0 / (1.0 + distance)

                if position in self.id_map:
                    results.append((self.id_map[position], similarity))

            return results

    def remove_vector(self, id) -> bool:
        """Remove a vector from the index"""
        with self.lock:
            if id not in self.reverse_map:
                return False

            # FAISS doesn't support direct removal, so we need to rebuild the index
            position_to_remove = self.reverse_map[id]
            del self.reverse_map[id]
            if id in self.vector_store:
                del self.vector_store[id]

            # Collect all remaining vectors
            vectors = []
            ids = []
            for pos, vec_id in self.id_map.items():
                if pos != position_to_remove:
                    if vec_id in self.vector_store:
                        vector = self.vector_store[vec_id]
                        if not isinstance(vector, np.ndarray):
                            vector = np.array(vector, dtype=np.float32).reshape(1, -1)
                        vectors.append(vector)
                        ids.append(vec_id)

            # Rebuild index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.id_map = {}
            self.reverse_map = {}

            if vectors:
                vectors_np = np.vstack(vectors) if vectors else np.empty((0, self.dimension), dtype=np.float32)
                self.index.add(vectors_np)
                for i, vec_id in enumerate(ids):
                    self.id_map[i] = vec_id
                    self.reverse_map[vec_id] = i

            return True

    def get_vector(self, id) -> Optional[Any]:
        """Get a vector by ID"""
        with self.lock:
            return self.vector_store.get(id)

    def count(self) -> int:
        """Get the number of vectors in the index"""
        with self.lock:
            return len(self.reverse_map)

###############################################################################
#                          CORE MANAGERS & COMPONENTS                         #
###############################################################################

class VectorStore:
    """Manages storage and retrieval of embedding vectors"""
    def __init__(self, dimension=VECTOR_DIMENSION):
        self.dimension = dimension

        # Choose the appropriate index implementation
        if FAISS_AVAILABLE:
            try:
                self.index = FaissSearchIndex(dimension)
                logger.info("Using FAISS for vector indexing")
            except Exception as e:
                logger.warning(f"Failed to initialize FAISS index: {e}")
                self.index = BasicSearchIndex(dimension)
                logger.info("Using basic vector index")
        else:
            self.index = BasicSearchIndex(dimension)
            logger.info("Using basic vector index")

        # Keep track of vector metadata separately
        self.metadata = {}  # id -> metadata mapping
        self.lock = threading.RLock()

    def add(self, id, embedding_vector):
        """Add an embedding vector to the store"""
        with self.lock:
            if isinstance(embedding_vector, EmbeddingVector):
                vector = embedding_vector.vector
                metadata = embedding_vector.metadata
            else:
                vector = embedding_vector
                metadata = {}

            # Store metadata
            self.metadata[id] = metadata

            # Add to index
            return self.index.add_vector(id, vector)

    def get(self, id):
        """Get an embedding vector by ID"""
        with self.lock:
            vector = self.index.get_vector(id)
            metadata = self.metadata.get(id, {})

            if vector is not None:
                return EmbeddingVector(vector, metadata)
            return None

    def search_similar(self, query_vector, top_k=10, threshold=0.0):
        """Search for vectors similar to the query vector"""
        with self.lock:
            results = self.index.search(query_vector, top_k)

            # Apply threshold if specified
            if threshold > 0:
                results = [(id, score) for id, score in results if score >= threshold]

            return results

    def remove(self, id):
        """Remove a vector from the store"""
        with self.lock:
            if id in self.metadata:
                del self.metadata[id]
            return self.index.remove_vector(id)

    def count(self):
        """Get the number of vectors in the store"""
        return self.index.count()

    def get_stats(self):
        """Get statistics about the vector store"""
        return {
            "vector_count": self.count(),
            "dimension": self.dimension,
            "index_type": type(self.index).__name__
        }

    def is_ready(self):
        """Check if the vector store is ready for operations"""
        return True

class BlockChainManager:
    """Manages blockchain operations and integrity checking"""
    def __init__(self, db_manager=None, mining_difficulty=BLOCKCHAIN_DIFFICULTY):
        self.db_manager = db_manager
        self.mining_difficulty = mining_difficulty
        self.lock = threading.RLock()

        # Keep track of chains and blocks
        self.chains = {}  # chain_id -> list of block_ids
        self.blocks = {}  # block_id -> Block object (in-memory cache)
        self.chain_heads = {}  # chain_id -> head block_id

        # Load existing chains from database if available
        if db_manager:
            self._load_chains()

    def _load_chains(self):
        """Load existing chains from the database"""
        try:
            client = self.db_manager.get_client("filesystem")
            chain_list = client.retrieve("chains")

            if chain_list:
                for chain_id in chain_list:
                    chain_blocks = client.retrieve(f"chain:{chain_id}")
                    if chain_blocks:
                        self.chains[chain_id] = chain_blocks
                        self.chain_heads[chain_id] = chain_blocks[-1]

                        # Load the head block into memory
                        head_block = client.retrieve(f"block:{chain_blocks[-1]}")
                        if head_block:
                            self.blocks[chain_blocks[-1]] = head_block

            logger.info(f"Loaded {len(self.chains)} chains from database")
        except Exception as e:
            logger.error(f"Error loading chains from database: {e}")
            # Initialize empty if loading fails
            self.chains = {}
            self.blocks = {}
            self.chain_heads = {}

    def create_block(self, data, embeddings, chain_id=None, context_references=None):
        """Create a new block with the given data and embeddings"""
        with self.lock:
            # Get the previous hash if this is part of a chain
            prev_hash = None
            if chain_id and chain_id in self.chain_heads:
                head_block_id = self.chain_heads[chain_id]
                head_block = self.get_block(head_block_id)
                if head_block:
                    prev_hash = head_block.hash

            # Create a new block
            block = Block(
                prev_hash=prev_hash,
                data=data,
                embeddings=embeddings
            )

            # Add context references to the block data
            if context_references:
                block.data["_context_references"] = context_references

            # Mine the block to find a valid hash
            block.mine_block(difficulty=self.mining_difficulty)

            # Store in memory
            self.blocks[block.id] = block

            # Update chain information
            if chain_id:
                if chain_id not in self.chains:
                    self.chains[chain_id] = []
                self.chains[chain_id].append(block.id)
                self.chain_heads[chain_id] = block.id
            else:
                # Create a new chain with this block as the genesis block
                new_chain_id = str(uuid.uuid4())
                self.chains[new_chain_id] = [block.id]
                self.chain_heads[new_chain_id] = block.id
                # Add chain ID to block data
                block.data["_chain_id"] = new_chain_id

            # Persist to database if available
            if self.db_manager:
                self._persist_block(block)
                self._persist_chain_info()

            return block

    def _persist_block(self, block):
        """Persist a block to the database"""
        try:
            client = self.db_manager.get_client("filesystem")
            client.store(f"block:{block.id}", block)
        except Exception as e:
            logger.error(f"Error persisting block {block.id}: {e}")

    def _persist_chain_info(self):
        """Persist chain information to the database"""
        try:
            client = self.db_manager.get_client("filesystem")
            # Save list of all chains
            client.store("chains", list(self.chains.keys()))

            # Save each chain's blocks
            for chain_id, blocks in self.chains.items():
                client.store(f"chain:{chain_id}", blocks)
        except Exception as e:
            logger.error(f"Error persisting chain info: {e}")

    def get_block(self, block_id):
        """Get a block by ID, either from memory or database"""
        # Check memory cache first
        if block_id in self.blocks:
            return self.blocks[block_id]

        # Try to load from database
        if self.db_manager:
            try:
                client = self.db_manager.get_client("filesystem")
                block = client.retrieve(f"block:{block_id}")
                if block:
                    # Cache in memory for future use
                    self.blocks[block_id] = block
                    return block
            except Exception as e:
                logger.error(f"Error loading block {block_id} from database: {e}")

        return None

    def get_chain(self, chain_id):
        """Get all blocks in a chain"""
        if chain_id not in self.chains:
            return []

        # Return the blocks in the chain
        blocks = []
        for block_id in self.chains[chain_id]:
            block = self.get_block(block_id)
            if block:
                blocks.append(block)

        return blocks

    def get_chains_for_block(self, block_id):
        """Find all chains containing the given block"""
        chains = []
        for chain_id, block_ids in self.chains.items():
            if block_id in block_ids:
                chains.append(chain_id)
        return chains

    def get_context_blocks(self, block_id, max_depth=3):
        """Get context blocks for the given block"""
        block = self.get_block(block_id)
        if not block:
            return []

        context_blocks = []

        # Add blocks from context references
        if "_context_references" in block.data:
            for ref_id in block.data["_context_references"]:
                ref_block = self.get_block(ref_id)
                if ref_block:
                    context_blocks.append(ref_block)

        # Add previous blocks from the same chain
        chain_ids = self.get_chains_for_block(block_id)
        for chain_id in chain_ids:
            chain = self.get_chain(chain_id)
            if not chain:
                continue

            # Find the position of the block in the chain
            block_positions = {b.id: i for i, b in enumerate(chain)}
            if block_id in block_positions:
                pos = block_positions[block_id]

                # Get previous blocks up to max_depth
                start = max(0, pos - max_depth)
                for i in range(start, pos):
                    if chain[i].id != block_id and chain[i] not in context_blocks:
                        context_blocks.append(chain[i])

        return context_blocks

    def verify_chain_integrity(self, chain_id=None):
        """Verify the integrity of a chain or all chains"""
        if chain_id:
            chains_to_check = [chain_id]
        else:
            chains_to_check = list(self.chains.keys())

        for cid in chains_to_check:
            if cid not in self.chains:
                logger.warning(f"Chain {cid} not found")
                return False

            blocks = self.get_chain(cid)
            if not blocks:
                logger.warning(f"Chain {cid} has no blocks")
                return False

            # Verify each block
            prev_hash = None
            for block in blocks:
                # Verify block hash
                if not block.validate():
                    logger.warning(f"Block {block.id} in chain {cid} has invalid hash")
                    return False

                # Verify prev_hash link
                if prev_hash is not None and block.prev_hash != prev_hash:
                    logger.warning(f"Block {block.id} in chain {cid} has invalid prev_hash link")
                    return False

                prev_hash = block.hash

        return True

    def get_latest_block_id(self, chain_id=None):
        """Get the ID of the latest block in a chain or across all chains"""
        if chain_id:
            return self.chain_heads.get(chain_id)

        # Get the latest block across all chains
        latest_block_id = None
        latest_timestamp = 0

        for chain_id, head_id in self.chain_heads.items():
            block = self.get_block(head_id)
            if block and block.timestamp > latest_timestamp:
                latest_timestamp = block.timestamp
                latest_block_id = block.id

        return latest_block_id

    def get_block_count(self):
        """Get the total number of blocks across all chains"""
        count = 0
        for chain_id, block_ids in self.chains.items():
            count += len(block_ids)
        return count

    def get_stats(self):
        """Get statistics about the blockchain"""
        return {
            "block_count": self.get_block_count(),
            "chain_count": len(self.chains),
            "mining_difficulty": self.mining_difficulty
        }

class MVCCManager:
    """Manages multi-version concurrency control for transactions"""
    def __init__(self):
        self.transactions = {}  # transaction_id -> MVCCTransaction
        self.locked_keys = {}  # key -> transaction_id
        self.lock = threading.RLock()

    def start_transaction(self):
        """Start a new transaction"""
        with self.lock:
            transaction = MVCCTransaction()
            self.transactions[transaction.transaction_id] = transaction
            return transaction

    def read(self, transaction_id, key, client):
        """Read a value within a transaction"""
        with self.lock:
            if transaction_id not in self.transactions:
                raise ValueError(f"Invalid transaction ID: {transaction_id}")

            transaction = self.transactions[transaction_id]
            if transaction.status != "active":
                raise ValueError(f"Transaction {transaction_id} is not active")

            # Record the read operation
            transaction.read_set.add(key)

            # Perform the read
            return client.retrieve(key)

    def write(self, transaction_id, key, value, client):
        """Write a value within a transaction"""
        with self.lock:
            if transaction_id not in self.transactions:
                raise ValueError(f"Invalid transaction ID: {transaction_id}")

            transaction = self.transactions[transaction_id]
            if transaction.status != "active":
                raise ValueError(f"Transaction {transaction_id} is not active")

            # Check if the key is locked by another transaction
            if key in self.locked_keys and self.locked_keys[key] != transaction_id:
                raise ValueError(f"Key {key} is locked by another transaction")

            # Lock the key for this transaction
            self.locked_keys[key] = transaction_id

            # Record the write operation
            transaction.write_set.add(key)

            # Perform the write
            return client.store(key, value)

    def commit(self, transaction_id):
        """Commit a transaction"""
        with self.lock:
            if transaction_id not in self.transactions:
                raise ValueError(f"Invalid transaction ID: {transaction_id}")

            transaction = self.transactions[transaction_id]
            if transaction.status != "active":
                raise ValueError(f"Transaction {transaction_id} is not active")

            # Update transaction status
            transaction.status = "committed"

            # Release locks
            for key in transaction.write_set:
                if key in self.locked_keys and self.locked_keys[key] == transaction_id:
                    del self.locked_keys[key]

            return True

    def abort(self, transaction_id):
        """Abort a transaction"""
        with self.lock:
            if transaction_id not in self.transactions:
                raise ValueError(f"Invalid transaction ID: {transaction_id}")

            transaction = self.transactions[transaction_id]
            if transaction.status != "active":
                raise ValueError(f"Transaction {transaction_id} is not active")

            # Update transaction status
            transaction.status = "aborted"

            # Release locks
            for key in transaction.write_set:
                if key in self.locked_keys and self.locked_keys[key] == transaction_id:
                    del self.locked_keys[key]

            return True

    def cleanup(self, max_age=3600):
        """Clean up old transactions"""
        with self.lock:
            current_time = time.time()
            to_remove = []

            for transaction_id, transaction in self.transactions.items():
                if transaction.status != "active" and current_time - transaction.start_time > max_age:
                    to_remove.append(transaction_id)

            for transaction_id in to_remove:
                del self.transactions[transaction_id]

            return len(to_remove)

class ShardManager:
    """Manages shards for distributed operation"""
    def __init__(self, num_shards=SHARD_COUNT):
        self.num_shards = num_shards
        self.shards = {}  # shard_id -> ShardInfo
        self.lock = threading.RLock()

        # Initialize shards
        for i in range(num_shards):
            shard_id = f"shard-{i}"
            self.shards[shard_id] = ShardInfo(shard_id)

    def get_shard_for_block(self, block_id):
        """Determine which shard a block belongs to"""
        # Simple hash-based sharding
        shard_index = int(hashlib.md5(block_id.encode()).hexdigest(), 16) % self.num_shards
        return f"shard-{shard_index}"

    def get_all_shards(self):
        """Get information about all shards"""
        with self.lock:
            return list(self.shards.values())

    def check_all_shards(self):
        """Check if all shards are active"""
        with self.lock:
            return all(shard.is_active() for shard in self.shards.values())

    def get_stats(self):
        """Get statistics about the shards"""
        with self.lock:
            active_shards = sum(1 for shard in self.shards.values() if shard.is_active())
            return {
                "total_shards": self.num_shards,
                "active_shards": active_shards
            }


class NeuralBacktracker:
    """Implements neural backtracking for explainable AI reasoning"""
    def __init__(self, block_chain_manager=None, vector_store=None):
        self.block_chain_manager = block_chain_manager
        self.vector_store = vector_store
        self.traces = {}  # trace_id -> trace data
        self.lock = threading.RLock()

    def trace_activation(self, query_vector, threshold=DEFAULT_
        class NeuralBacktracker:
            """Implements neural backtracking for explainable AI reasoning"""
            def __init__(self, block_chain_manager=None, vector_store=None):
                self.block_chain_manager = block_chain_manager
                self.vector_store = vector_store
                self.traces = {}  # trace_id -> trace data
                self.lock = threading.RLock()

            def trace_activation(self, query_vector, threshold=DEFAULT_SIMILARITY_THRESHOLD, top_k=20):
                """
                Create a neural trace for the query vector.

                A neural trace tracks activation patterns through the system,
                showing how different blocks contribute to the response.
                """
                trace_id = str(uuid.uuid4())

                with self.lock:
                    # Find similar vectors
                    if not self.vector_store:
                        logger.warning("Vector store not available for tracing")
                        return trace_id

                    similar_vectors = self.vector_store.search_similar(query_vector, top_k=top_k, threshold=threshold)

                    # Find blocks containing these vectors
                    blocks = []
                    if self.block_chain_manager:
                        # Get all blocks and check if they contain the similar vectors
                        for chain_id, block_ids in self.block_chain_manager.chains.items():
                            for block_id in block_ids:
                                block = self.block_chain_manager.get_block(block_id)
                                if block:
                                    # Check if any of the block's embeddings match our similar vectors
                                    for vector_id, _ in similar_vectors:
                                        if any(e.metadata.get('vector_id') == vector_id for e in block.embeddings):
                                            blocks.append(block)
                                            break

                    # Create the trace
                    trace = {
                        'id': trace_id,
                        'timestamp': time.time(),
                        'query_vector': query_vector,
                        'similar_vectors': similar_vectors,
                        'blocks': [block.id for block in blocks],
                        'chains': list(set(block.data.get('_chain_id', '') for block in blocks if '_chain_id' in block.data)),
                        'activation_path': self._compute_activation_path(query_vector, blocks)
                    }

                    self.traces[trace_id] = trace
                    return trace_id

            def _compute_activation_path(self, query_vector, blocks):
                """Compute the activation path through blocks"""
                # Sort blocks by similarity to query vector
                block_similarities = []

                for block in blocks:
                    # Compute average similarity of block embeddings to query vector
                    similarities = []
                    for embedding in block.embeddings:
                        if self.vector_store and hasattr(self.vector_store.index, '_cosine_similarity'):
                            similarity = self.vector_store.index._cosine_similarity(query_vector, embedding.vector)
                            similarities.append(similarity)

                    if similarities:
                        avg_similarity = sum(similarities) / len(similarities)
                        block_similarities.append((block.id, avg_similarity))

                # Sort by similarity (highest first)
                block_similarities.sort(key=lambda x: x[1], reverse=True)

                # Create activation path: connections between blocks
                path = []
                for i in range(len(block_similarities) - 1):
                    source_id, source_sim = block_similarities[i]
                    target_id, target_sim = block_similarities[i + 1]
                    path.append({
                        'source': source_id,
                        'target': target_id,
                        'weight': (source_sim + target_sim) / 2
                    })

                return path

            def get_trace(self, trace_id):
                """Get a neural trace by ID"""
                with self.lock:
                    return self.traces.get(trace_id)

            def get_traces_for_block(self, block_id):
                """Get all traces involving a specific block"""
                with self.lock:
                    return [trace_id for trace_id, trace in self.traces.items()
                           if block_id in trace['blocks']]

        class MultiHeadAttention:
            """Implements multi-head attention for sequence modeling"""
            def __init__(self, num_heads=8, d_model=512):
                self.num_heads = num_heads
                self.d_model = d_model
                self.d_k = d_model // num_heads

                # Initialize weights for query, key, value projections
                if NUMPY_AVAILABLE:
                    # Initialize with Xavier/Glorot initialization
                    scale = np.sqrt(2.0 / (d_model + self.d_k))
                    self.w_q = np.random.randn(d_model, d_model) * scale
                    self.w_k = np.random.randn(d_model, d_model) * scale
                    self.w_v = np.random.randn(d_model, d_model) * scale
                    self.w_o = np.random.randn(d_model, d_model) * scale
                else:
                    # Simple random initialization
                    import random
                    self.w_q = [[random.gauss(0, 0.02) for _ in range(d_model)] for _ in range(d_model)]
                    self.w_k = [[random.gauss(0, 0.02) for _ in range(d_model)] for _ in range(d_model)]
                    self.w_v = [[random.gauss(0, 0.02) for _ in range(d_model)] for _ in range(d_model)]
                    self.w_o = [[random.gauss(0, 0.02) for _ in range(d_model)] for _ in range(d_model)]

            def apply(self, vectors, query=None):
                """Apply multi-head attention to a sequence of vectors"""
                if not vectors:
                    return None

                # If no explicit query is provided, use the last vector
                if query is None:
                    query = vectors[-1]

                if NUMPY_AVAILABLE:
                    return self._apply_numpy(vectors, query)
                else:
                    return self._apply_python(vectors, query)

            def _apply_numpy(self, vectors, query):
                """NumPy implementation of multi-head attention"""
                # Convert to numpy arrays
                vectors_np = np.array(vectors)
                query_np = np.array(query).reshape(1, -1)

                # Project query, key, value
                q = np.matmul(query_np, self.w_q).reshape(self.num_heads, self.d_k)
                k = np.matmul(vectors_np, self.w_k).reshape(len(vectors), self.num_heads, self.d_k)
                v = np.matmul(vectors_np, self.w_v).reshape(len(vectors), self.num_heads, self.d_k)

                # Compute attention scores
                scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(self.d_k)

                # Apply softmax
                attn_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)

                # Apply attention weights
                context = np.matmul(attn_weights, v)

                # Concatenate heads and apply output projection
                context = context.reshape(1, -1)
                output = np.matmul(context, self.w_o).flatten()

                return output

            def _apply_python(self, vectors, query):
                """Pure Python implementation of multi-head attention"""
                # Simple implementation that computes weighted average based on similarity
                similarities = []

                for vec in vectors:
                    # Compute cosine similarity
                    dot_product = sum(a*b for a, b in zip(query, vec))
                    norm_q = math.sqrt(sum(a*a for a in query))
                    norm_v = math.sqrt(sum(b*b for b in vec))

                    if norm_q == 0 or norm_v == 0:
                        similarity = 0
                    else:
                        similarity = dot_product / (norm_q * norm_v)

                    similarities.append(similarity)

                # Convert to weights using softmax
                exp_similarities = [math.exp(s) for s in similarities]
                sum_exp = sum(exp_similarities)
                weights = [e / sum_exp for e in exp_similarities]

                # Compute weighted average
                d = len(query)
                output = [0] * d
                for i, vec in enumerate(vectors):
                    for j in range(d):
                        output[j] += weights[i] * vec[j]

                return output

        class ContentFilter:
            """Filters content to ensure it's safe and compliant"""
            def __init__(self, sensitive_topics=None, offensive_patterns=None):
                self.sensitive_topics = sensitive_topics or []
                self.offensive_patterns = offensive_patterns or []

            def should_filter_content(self, data):
                """Check if content should be filtered"""
                if not isinstance(data, dict):
                    return False

                # Check text fields in the data
                for key, value in data.items():
                    if isinstance(value, str):
                        if self._contains_sensitive_content(value):
                            return True

                return False

            def should_filter_vector(self, embedding_vector):
                """Check if embedding vector should be filtered"""
                if not isinstance(embedding_vector, EmbeddingVector):
                    return False

                # Check metadata for sensitive content
                metadata = embedding_vector.metadata
                for key, value in metadata.items():
                    if isinstance(value, str):
                        if self._contains_sensitive_content(value):
                            return True

                return False

            def _contains_sensitive_content(self, text):
                """Check if text contains sensitive topics or offensive patterns"""
                text = text.lower()

                # Check for sensitive topics
                for topic in self.sensitive_topics:
                    if topic.lower() in text:
                        return True

                # Check for offensive patterns
                for pattern in self.offensive_patterns:
                    if pattern.lower() in text:
                        return True

                return False

        class BiasDetector:
            """Detects and mitigates bias in content"""
            def __init__(self, bias_attributes=None):
                self.bias_attributes = bias_attributes or []

            def detect_bias(self, text):
                """Detect bias in text"""
                bias_scores = {}

                for attribute in self.bias_attributes:
                    # Simple approach: check for mentions of the attribute
                    score = text.lower().count(attribute.lower()) / (len(text.split()) + 1)
                    bias_scores[attribute] = score

                return bias_scores

            def get_overall_bias_score(self, text):
                """Get an overall bias score"""
                bias_scores = self.detect_bias(text)
                if not bias_scores:
                    return 0

                return sum(bias_scores.values()) / len(bias_scores)

        class PersonaManager:
            """Manages multiple AI personas and response generation"""
            def __init__(self, wdbx_instance):
                self.wdbx = wdbx_instance
                self.personas = {
                    "default": {
                        "name": "Default",
                        "description": "The default AI assistant persona.",
                        "traits": ["helpful", "neutral", "informative"],
                        "embeddings": []
                    },
                    "technical": {
                        "name": "Technical Expert",
                        "description": "Technical and detailed persona for complex topics.",
                        "traits": ["precise", "technical", "detailed"],
                        "embeddings": []
                    },
                    "creative": {
                        "name": "Creative",
                        "description": "Creative and imaginative persona for brainstorming.",
                        "traits": ["creative", "imaginative", "expressive"],
                        "embeddings": []
                    }
                }

                # Initialize persona embeddings
                self.create_persona_embeddings()

            def create_persona_embeddings(self):
                """Create embeddings for each persona"""
                for persona_id, persona in self.personas.items():
                    # Create embedding from persona description and traits
                    text = f"{persona['name']}. {persona['description']} {' '.join(persona['traits'])}"

                    if hasattr(self.wdbx, 'model') and self.wdbx.model:
                        embedding = self.wdbx.model.encode(text)

                        # Create an embedding vector
                        vector = EmbeddingVector(
                            vector=embedding,
                            metadata={
                                "persona_id": persona_id,
                                "description": persona["description"],
                                "traits": persona["traits"]
                            }
                        )

                        persona["embeddings"] = [vector]

            def determine_optimal_persona(self, user_input, context=None):
                """Determine the optimal persona for the user input"""
                if not hasattr(self.wdbx, 'model') or not self.wdbx.model:
                    return "default"

                # Convert user input to an embedding
                user_embedding = self.wdbx.model.encode(user_input)

                # Compute similarity to each persona
                similarities = {}
                for persona_id, persona in self.personas.items():
                    if not persona["embeddings"]:
                        continue

                    # Get the persona embedding
                    persona_embedding = persona["embeddings"][0].vector

                    # Compute similarity
                    if hasattr(self.wdbx.vector_store.index, '_cosine_similarity'):
                        similarity = self.wdbx.vector_store.index._cosine_similarity(user_embedding, persona_embedding)
                    else:
                        # Fallback to simple cosine similarity
                        dot_product = sum(a*b for a, b in zip(user_embedding, persona_embedding))
                        norm_u = math.sqrt(sum(a*a for a in user_embedding))
                        norm_p = math.sqrt(sum(b*b for b in persona_embedding))

                        if norm_u == 0 or norm_p == 0:
                            similarity = 0
                        else:
                            similarity = dot_product / (norm_u * norm_p)

                    similarities[persona_id] = similarity

                # Return the persona with the highest similarity
                if similarities:
                    return max(similarities.items(), key=lambda x: x[1])[0]

                return "default"

            def generate_response(self, user_input, persona_id="default", context=None):
                """Generate a response using the specified persona"""
                # This is a placeholder for actual response generation
                # In a real implementation, this would use an LLM or other model

                persona = self.personas.get(persona_id, self.personas["default"])

                # Simplistic response generation based on persona traits
                if "technical" in persona["traits"]:
                    response = f"From a technical perspective, I would analyze '{user_input}' as follows..."
                elif "creative" in persona["traits"]:
                    response = f"Creatively thinking about '{user_input}', I imagine..."
                else:
                    response = f"I understand you're asking about '{user_input}'. Here's what I know..."

                return response

            def process_user_input(self, user_input, context=None):
                """Process a user input and generate a response"""
                # Determine the optimal persona
                persona_id = self.determine_optimal_persona(user_input, context)

                # Generate a response
                response = self.generate_response(user_input, persona_id, context)

                # Create an embedding for the user input
                if hasattr(self.wdbx, 'model') and self.wdbx.model:
                    user_embedding = EmbeddingVector(
                        vector=self.wdbx.model.encode(user_input),
                        metadata={
                            "type": "user_input",
                            "text": user_input,
                            "timestamp": time.time()
                        }
                    )

                    # Create an embedding for the response
                    response_embedding = EmbeddingVector(
                        vector=self.wdbx.model.encode(response),
                        metadata={
                            "type": "ai_response",
                            "text": response,
                            "persona_id": persona_id,
                            "timestamp": time.time()
                        }
                    )

                    # Create a conversation block
                    block_data = {
                        "user_input": user_input,
                        "ai_response": response,
                        "persona_id": persona_id,
                        "timestamp": time.time()
                    }

                    # Add context if available
                    if context and "chain_id" in context:
                        chain_id = context["chain_id"]
                    else:
                        chain_id = None

                    if context and "block_ids" in context:
                        context_references = context["block_ids"]
                    else:
                        context_references = None

                    # Create the block
                    block_id = None
                    if hasattr(self.wdbx, 'create_conversation_block'):
                        try:
                            block_id = self.wdbx.create_conversation_block(
                                data=block_data,
                                embeddings=[user_embedding, response_embedding],
                                chain_id=chain_id,
                                context_references=context_references,
                                persona_id=persona_id
                            )
                        except Exception as e:
                            logger.error(f"Error creating conversation block: {e}")

                    return response, block_id

                return response, None

        class PersonaTokenManager:
            """Manages persona-specific token generation and blending"""
            def __init__(self, persona_embeddings):
                self.persona_embeddings = persona_embeddings

            def blend_responses(self, responses, weights=None):
                """Blend multiple responses based on weights"""
                if not responses:
                    return ""

                if len(responses) == 1:
                    return responses[0]

                # Use equal weights if not provided
                if not weights or len(weights) != len(responses):
                    weights = [1.0 / len(responses)] * len(responses)

                # Ensure weights sum to 1
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

                # Simplistic blending for illustration
                # In a real implementation, this would use more sophisticated methods
                parts = []
                for i, (response, weight) in enumerate(zip(responses, weights)):
                    if weight > 0.2:  # Only include significant contributions
                        parts.append(response)

                return " ".join(parts)

        ###############################################################################
        #                              WDBX MAIN CLASS                                #
        ###############################################################################

        class WDBX:
            """
            Core implementation of the Wide Distributed Block Exchange (WDBX) system.

            This class integrates all components of the WDBX system, providing a streamlined
            interface for storing embedding vectors, creating conversation blocks with blockchain
            integrity, and managing multi-persona AI systems.

            Args:
                vector_dimension (int): Dimensionality of embedding vectors
                num_shards (int): Number of shards for distributed storage
                enable_persona_management (bool): Whether to enable the persona management system
                content_filter_level (str): Level of content filtering ('none', 'low', 'medium', 'high')
                data_dir (str): Directory for data storage
                model_name (str): Name of the pre-trained model to use for embeddings
            """
            def __init__(self,
                         vector_dimension=VECTOR_DIMENSION,
                         num_shards=SHARD_COUNT,
                         enable_persona_management=True,
                         content_filter_level='medium',
                         data_dir=DEFAULT_DATA_DIR,
                         model_name=None):
                self.vector_dimension = vector_dimension
                self.num_shards = num_shards
                self.embedding_dimension = vector_dimension
                self.documents = []  # For simple document storage

                # Initialize embedding model
                self.model = None
                if model_name and SENTENCE_TRANSFORMER_AVAILABLE:
                    try:
                        self.model = SentenceTransformer(model_name)
                        logger.info(f"Loaded SentenceTransformer model: {model_name}")
                    except Exception as e:
                        logger.error(f"Error loading SentenceTransformer model: {e}")
                        self.model = SimpleEmbeddingModel(dimension=vector_dimension)
                else:
                    self.model = SimpleEmbeddingModel(dimension=vector_dimension)
                    logger.info(f"Using simple embedding model with dimension {vector_dimension}")

                # Initialize database connection manager
                self.db_manager = DatabaseConnectionManager(
                    data_dir=data_dir,
                    use_socket_by_default=False  # Use filesystem by default for simplicity
                )

                # Initialize vector store
                self.vector_store = VectorStore(dimension=vector_dimension)

                # Initialize shard manager
                self.shard_manager = ShardManager(num_shards=num_shards)

                # Initialize blockchain manager
                self.blockchain_manager = BlockChainManager(
                    db_manager=self.db_manager,
                    mining_difficulty=BLOCKCHAIN_DIFFICULTY
                )

                # Initialize MVCC manager
                self.mvcc_manager = MVCCManager()

                # Initialize neural backtracker
                self.neural_backtracker = NeuralBacktracker(
                    block_chain_manager=self.blockchain_manager,
                    vector_store=self.vector_store
                )

                # Initialize content filtering
                self._initialize_content_filter(content_filter_level)

                # Initialize persona management if enabled
                if enable_persona_management:
                    self.persona_manager = PersonaManager(self)
                    self.persona_token_manager = PersonaTokenManager(
                        self.persona_manager.persona_embeddings if hasattr(self.persona_manager, 'persona_embeddings') else {}
                    )
                else:
                    self.persona_manager = None
                    self.persona_token_manager = None

                # Initialize multi-head attention
                self.attention = MultiHeadAttention(num_heads=8, d_model=vector_dimension)

                # Initialize metrics
                self.stats = {
                    "blocks_created": 0,
                    "vectors_stored": 0,
                    "transactions_processed": 0,
                    "traces_created": 0,
                    "queries_processed": 0,
                    "failed_transactions": 0,
                    "content_filter_blocks": 0,
                    "start_time": time.time()
                }

                # Initialize search index (FAISS or basic)
                if FAISS_AVAILABLE:
                    try:
                        self.vector_db = faiss.IndexFlatL2(vector_dimension)
                        logger.info("Using FAISS for vector database")
                    except Exception as e:
                        logger.error(f"Error initializing FAISS: {e}")
                        self.vector_db = self.vector_store.index
                else:
                    self.vector_db = self.vector_store.index

                # Add initial data
                self.add_initial_data()

                logger.info(f"WDBX initialized with {vector_dimension}-dimension vectors across {num_shards} shards")

            def _initialize_content_filter(self, level):
                """Initialize content filter based on specified level"""
                sensitive_topics = []
                offensive_patterns = []

                if level in ('low', 'medium', 'high'):
                    sensitive_topics = ["illegal activities", "dangerous substances"]
                    offensive_patterns = ["hate speech", "discriminatory language"]

                if level in ('medium', 'high'):
                    sensitive_topics.extend(["explicit content", "self-harm"])
                    offensive_patterns.extend(["offensive language", "threatening language"])

                if level == 'high':
                    sensitive_topics.extend(["political extremism", "personal identifiable information"])
                    offensive_patterns.extend(["profanity", "controversial topics"])

                self.content_filter = ContentFilter(
                    sensitive_topics=sensitive_topics,
                    offensive_patterns=offensive_patterns
                )

                self.bias_detector = BiasDetector(
                    bias_attributes=["gender", "race", "age", "religion", "nationality", "disability", "sexual orientation"]
                )

            def add_initial_data(self):
                """Adds initial data to the vector database."""
                try:
                    # Load initial data from a file or database
                    initial_data = self.load_initial_data()

                    # Embed the initial data
                    embeddings = self.embed_data(initial_data)

                    # Add embeddings to the vector database
                    if hasattr(self.vector_db, 'add') and NUMPY_AVAILABLE:
                        self.vector_db.add(np.array(embeddings))

                    # Also store documents for retrieval
                    self.documents = initial_data

                    logger.info(f"Added {len(initial_data)} initial documents")
                except Exception as e:
                    logger.error(f"Error adding initial data: {e}")

            def load_initial_data(self):
                """Loads initial data from a file or database."""
                # Try to load from file if it exists
                data_file = os.path.join(self.db_manager.fs_client.data_dir, "initial_data.json")
                if os.path.exists(data_file):
                    try:
                        with open(data_file, 'r') as f:
                            return json.load(f)
                    except Exception as e:
                        logger.error(f"Error loading initial data from file: {e}")

                # Return default data
                return [
                    "WDBX is a high-performance, distributed database system for multi-personality AI systems.",
                    "It combines vector similarity search with blockchain-style integrity and MVCC concurrency control.",
                    "Neural backtracking allows tracing activation patterns for explainable AI reasoning."
                ]

            def embed_data(self, data):
                """Embeds the given data using the pre-trained model."""
                embeddings = []
                for text in data:
                    embedding = self.model.encode(text)
                    embeddings.append(embedding)
                return embeddings

            def add_document(self, document):
                """Adds a new document to the vector database."""
                try:
                    # Embed the document
                    embedding = self.model.encode(document)

                    # Add the embedding to the vector database
                    if hasattr(self.vector_db, 'add') and NUMPY_AVAILABLE:
                        self.vector_db.add(np.array([embedding]))
                    else:
                        # Use our vector store as fallback
                        vector_id = str(uuid.uuid4())
                        self.vector_store.add(vector_id, EmbeddingVector(
                            vector=embedding,
                            metadata={"text": document, "timestamp": time.time()}
                        ))

                    # Store the document
                    self.documents.append(document)
                    return True
                except Exception as e:
                    logger.error(f"Error adding document: {e}")
                    return False

            def search(self, query, top_k=5):
                """Searches the vector database for the most similar documents to the given query."""
                try:
                    # Embed the query
                    query_embedding = self.model.encode(query)

                    # Search the vector database
                    if hasattr(self.vector_db, 'search') and NUMPY_AVAILABLE:
                        distances, indices = self.vector_db.search(np.array([query_embedding]), top_k)
                        # Retrieve the corresponding documents
                        results = [self.documents[i] for i in indices[0] if i < len(self.documents)]
                    else:
                        # Use our vector store as fallback
                        similar_vectors = self.vector_store.search_similar(query_embedding, top_k=top_k)
                        # Map back to documents if possible
                        results = []
                        for vector_id, _ in similar_vectors:
                            vector = self.vector_store.get(vector_id)
                            if vector and 'text' in vector.metadata:
                                results.append(vector.metadata['text'])

                    self.stats["queries_processed"] += 1
                    return results
                except Exception as e:
                    logger.error(f"Error searching the vector database: {e}")
                    return []

            def is_connected(self) -> bool:
                """
                Check if the system is ready and connected

                Returns:
                    bool: True if the system is operational, False otherwise
                """
                try:
                    # Check shard manager status
                    shard_status = all(s.is_active() for s in self.shard_manager.get_all_shards())

                    # Check vector store status
                    store_status = hasattr(self.vector_store, 'is_ready') and self.vector_store.is_ready()

                    # Check database connection
                    db_status = any(
                        client_type in self.db_manager.active_connections
                        for client_type in ['filesystem', 'socket', 'http']
                    )

                    return shard_status and store_status and db_status
                except Exception as e:
                    logger.error(f"Error checking connection status: {e}")
                    return False

            def store_embedding(self, embedding_vector) -> str:
                """
                Store an embedding vector in the system.

                Args:
                    embedding_vector (EmbeddingVector): The embedding vector to store

                Returns:
                    str: ID of the stored vector

                Raises:
                    ValueError: If the vector could not be stored
                    TypeError: If the provided embedding is not an EmbeddingVector instance
                """
                if not isinstance(embedding_vector, EmbeddingVector):
                    raise TypeError("Expected an EmbeddingVector instance")

                transaction = self.mvcc_manager.start_transaction()
                vector_id = str(uuid.uuid4())
                try:
                    # Filter content if necessary
                    if hasattr(self.content_filter, 'should_filter_vector') and self.content_filter.should_filter_vector(embedding_vector):
                        logger.warning(f"Content filter blocked vector {vector_id}")
                        self.stats["content_filter_blocks"] += 1
                        self.mvcc_manager.abort(transaction.transaction_id)
                        raise ValueError(f"Vector content blocked by filter")

                    embedding_vector.metadata["vector_id"] = vector_id
                    embedding_vector.metadata["timestamp"] = time.time()

                    # Store the vector
                    success = self.vector_store.add(vector_id, embedding_vector)
                    if not success:
                        self.mvcc_manager.abort(transaction.transaction_id)
                        self.stats["failed_transactions"] += 1
                        raise ValueError(f"Failed to store vector {vector_id}")

                    # Store in database for persistence
                    client = self.db_manager.get_client("filesystem")
                    if not client.store(f"vector:{vector_id}", embedding_vector):
                        logger.warning(f"Failed to persist vector {vector_id} to database")

                    self.mvcc_manager.commit(transaction.transaction_id)
                    self.stats["vectors_stored"] += 1
                    self.stats["transactions_processed"] += 1
                    return vector_id
                except Exception as e:
                    self.mvcc_manager.abort(transaction.transaction_id)
                    self.stats["failed_transactions"] += 1
                    raise e

            def create_conversation_block(self, data, embeddings, chain_id=None, context_references=None, persona_id=None) -> str:
                """
                Create a block containing conversation data and embedding vectors.

                Args:
                    data (Dict[str, Any]): Data to store in the block
                    embeddings (List[EmbeddingVector]): Embedding vectors to include in the block
                    chain_id (Optional[str]): ID of the chain to add this block to, or None for a new chain
                    context_references (Optional[List[str]]): References to other blocks for context
                    persona_id (Optional[str]): ID of the persona creating this block

                Returns:
                    str: ID of the created block

                Raises:
                    ValueError: If block creation fails due to invalid inputs
                    RuntimeError: If an operational error occurs during block creation
                """
                transaction = self.mvcc_manager.start_transaction()
                try:
                    # Check for content filtering
                    if hasattr(self.content_filter, 'should_filter_content') and self.content_filter.should_filter_content(data):
                        logger.warning("Content filter blocked block creation")
                        self.stats["content_filter_blocks"] += 1
                        self.mvcc_manager.abort(transaction.transaction_id)
                        raise ValueError("Block content blocked by filter")

                    # Store embeddings first
                    vector_ids = []
                    for embedding in embeddings:
                        vector_id = self.store_embedding(embedding)
                        embedding.metadata["vector_id"] = vector_id
                        vector_ids.append(vector_id)

                    # Add metadata
                    data["_meta"] = {
                        "timestamp": time.time(),
                        "vector_count": len(embeddings),
                        "vector_ids": vector_ids
                    }

                    if persona_id and self.persona_manager:
                        data["_meta"]["persona_id"] = persona_id
                        # Add persona-specific processing here if needed

                    # Create the block
                    block = self.blockchain_manager.create_block(
                        data=data,
                        embeddings=embeddings,
                        chain_id=chain_id,
                        context_references=context_references or []
                    )

                    # Assign to shard
                    shard_id = self.shard_manager.get_shard_for_block(block.id)
                    data["_meta"]["shard_id"] = shard_id

                    self.mvcc_manager.commit(transaction.transaction_id)
                    self.stats["blocks_created"] += 1
                    logger.debug(f"Created block {block.id} in chain {chain_id or 'new'}")
                    return block.id
                except Exception as e:
                    self.mvcc_manager.abort(transaction.transaction_id)
                    self.stats["failed_transactions"] += 1
                    if isinstance(e, ValueError):
                        raise
                    else:
                        raise RuntimeError(f"Failed to create block: {str(e)}") from e

            def search_similar_vectors(self, query_vector, top_k=10, threshold=0.0) -> List[Tuple[str, float]]:
                """
                Search for vectors similar to the query vector.

                Args:
                    query_vector: The query vector to search for
                    top_k (int): Maximum number of results to return
                    threshold (float): Minimum similarity score (0.0-1.0) to include in results

                Returns:
                    List[Tuple[str, float]]: List of (vector_id, similarity_score) tuples
                """
                self.stats["queries_processed"] += 1
                return self.vector_store.search_similar(
                    query_vector,
                    top_k=top_k,
                    threshold=threshold
                )

            def create_neural_trace(self, query_vector, trace_depth=3) -> str:
                """
                Create a neural trace for the query vector.

                A neural trace tracks activation patterns through the system,
                allowing for backtracking and understanding of how the system
                responds to specific inputs.

                Args:
                    query_vector: The query vector to trace
                    trace_depth (int): Depth of the neural trace

                Returns:
                    str: ID of the created trace
                """
                trace_id = self.neural_backtracker.trace_activation(
                    query_vector,
                    threshold=0.6,
                    top_k=min(20, trace_depth * 5)
                )
                self.stats["traces_created"] += 1
                return trace_id

            def get_conversation_context(self, block_ids, include_embeddings=True, max_context_blocks=10) -> Dict:
                """
                Get the conversation context for the given block IDs.

                Args:
                    block_ids (List[str]): IDs of blocks to get context for
                    include_embeddings (bool): Whether to include embeddings in the response
                    max_context_blocks (int): Maximum number of context blocks to include

                Returns:
                    Dict[str, Any]: Context information including blocks, chains, and aggregated embedding
                """
                if not block_ids:
                    return {"blocks": [], "context_blocks": [], "chains": [], "aggregated_embedding": None}

                blocks = []
                embeddings = []
                chains = set()

                # Collect block information
                for block_id in block_ids:
                    block = self.blockchain_manager.get_block(block_id)
                    if block:
                        blocks.append(block)
                        if include_embeddings:
                            embeddings.extend(block.embeddings)

                        # Find which chains this block belongs to
                        for chain_id in self.blockchain_manager.get_chains_for_block(block_id):
                            chains.add(chain_id)

                # Get context blocks with limit
                context_blocks = []
                for block in blocks:
                    context_blocks.extend(
                        self.blockchain_manager.get_context_blocks(block.id, max_depth=max_context_blocks)
                    )

                # Limit context blocks to max_context_blocks
                if len(context_blocks) > max_context_blocks:
                    context_blocks = context_blocks[:max_context_blocks]

                # Create aggregated embedding if needed
                aggregated_embedding = None
                if embeddings and include_embeddings:
                    vectors = [e.vector for e in embeddings]

                    # Apply attention mechanism
                    if hasattr(self.attention, 'apply'):
                        aggregated_embedding = self.attention.apply(
                            vectors,
                            query=vectors[-1] if vectors else None
                        )

                return {
                    "blocks": blocks,
                    "context_blocks": context_blocks,
                    "chains": list(chains),
                    "aggregated_embedding": aggregated_embedding,
                    "block_count": len(blocks),
                    "context_block_count": len(context_blocks),
                    "chain_count": len(chains)
                }

            def get_block(self, block_id) -> Optional[Block]:
                """
                Get a specific block by ID.

                Args:
                    block_id (str): ID of the block to retrieve

                Returns:
                    Optional[Block]: The block if found, None otherwise
                """
                return self.blockchain_manager.get_block(block_id)

            def get_chain(self, chain_id) -> List[Block]:
                """
                Get all blocks in a chain.

                Args:
                    chain_id (str): ID of the chain to retrieve

                Returns:
                    List[Block]: All blocks in the chain in order
                """
                return self.blockchain_manager.get_chain(chain_id)

            def get_system_stats(self) -> Dict:
                """
                Get comprehensive statistics about system usage.

                Returns:
                    Dict[str, Any]: System statistics
                """
                uptime = time.time() - self.stats["start_time"]

                # Calculate derived metrics
                blocks_per_second = self.stats["blocks_created"] / uptime if uptime > 0 else 0
                vectors_per_second = self.stats["vectors_stored"] / uptime if uptime > 0 else 0
                queries_per_second = self.stats["queries_processed"] / uptime if uptime > 0 else 0
                failure_rate = (self.stats["failed_transactions"] /
                              (self.stats["transactions_processed"] + self.stats["failed_transactions"])) * 100 if \
                              (self.stats["transactions_processed"] + self.stats["failed_transactions"]) > 0 else 0

                # Get component-specific stats
                vector_store_stats = self.vector_store.get_stats() if hasattr(self.vector_store, 'get_stats') else {
                    "vector_count": self.stats["vectors_stored"],
                    "dimension": self.vector_dimension
                }

                blockchain_stats = self.blockchain_manager.get_stats() if hasattr(self.blockchain_manager, 'get_stats') else {
                    "block_count": self.stats["blocks_created"],
                    "chain_count": len(getattr(self.blockchain_manager, 'chain_heads', {}))
                }

                shard_stats = self.shard_manager.get_stats() if hasattr(self.shard_manager, 'get_stats') else {
                    "shard_count": self.num_shards,
                    "active_shards": self.num_shards
                }

                return {
                    **self.stats,
                    "uptime": uptime,
                    "uptime_formatted": self._format_uptime(uptime),
                    "blocks_per_second": blocks_per_second,
                    "vectors_per_second": vectors_per_second,
                    "queries_per_second": queries_per_second,
                    "failure_rate_percent": failure_rate,
                    "shard_count": self.num_shards,
                    "vector_dimension": self.vector_dimension,
                    "vector_store": vector_store_stats,
                    "blockchain": blockchain_stats,
                    "shards": shard_stats,
                    "memory_usage_mb": self._get_memory_usage(),
                    "index_type": type(getattr(self, 'vector_db', None)).__name__,
                    "content_filter_enabled": hasattr(self, 'content_filter'),
                    "persona_management_enabled": self.persona_manager is not None,
                    "available_components": {
                        "numpy": NUMPY_AVAILABLE,
                        "faiss": FAISS_AVAILABLE,
                        "aiohttp": AIOHTTP_AVAILABLE,
                        "torch": TORCH_AVAILABLE,
                        "sentence_transformer": SENTENCE_TRANSFORMER_AVAILABLE
                    }
                }

            def _format_uptime(self, seconds):
                """Format uptime in human-readable format"""
                days, remainder = divmod(int(seconds), 86400)
                hours, remainder = divmod(remainder, 3600)
                minutes, seconds = divmod(remainder, 60)
                return f"{days}d {hours}h {minutes}m {seconds}s"

            def _get_memory_usage(self):
                """Get approximate memory usage in MB"""
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    return process.memory_info().rss / 1024 / 1024
                except ImportError:
                    return -1  # Couldn't determine

            def close(self):
                """
                Close any open resources and perform cleanup operations.

                This method should be called when shutting down the system to ensure
                proper cleanup of resources.
                """
                logger.info("Shutting down WDBX...")

                # Close database connections
                self.db_manager.close_all_connections()

                # Final log message
                logger.info(f"WDBX shutdown complete. Stats: {self.stats['blocks_created']} blocks, "
                           f"{self.stats['vectors_stored']} vectors stored")

        # Factory method for creating a new WDBX instance
        def create_wdbx(vector_dimension=VECTOR_DIMENSION, num_shards=SHARD_COUNT, **kwargs) -> WDBX:
            """
            Create a new WDBX instance with the specified configuration.

            Args:
                vector_dimension (int): Dimensionality of embedding vectors
                num_shards (int): Number of shards for distributed storage
                **kwargs: Additional configuration options

            Returns:
                WDBX: Configured WDBX instance
            """
            return WDBX(vector_dimension=vector_dimension, num_shards=num_shards, **kwargs)

        # Module version
        __version__ = "1.0.0"

        # If this module is run directly, show usage example
        if __name__ == "__main__":
            print(f"WDBX version {__version__}")
            print("Initializing WDBX with default settings...")

            # Create a WDBX instance
            wdbx = create_wdbx(vector_dimension=512, num_shards=4)

            # Add some documents
            print("Adding sample documents...")
            wdbx.add_document("WDBX is a high-performance vector database with blockchain integrity.")
            wdbx.add_document("It provides efficient storage and retrieval of embedding vectors.")
            wdbx.add_document("The system includes MVCC for concurrency control and neural backtracking.")

            # Search for documents
            print("\nSearching for documents about 'vector database':")
            results = wdbx.search("vector database", top_k=2)
            for result in results:
                print(f"- {result}")

            # Create an embedding vector
            print("\nCreating an embedding vector...")
            if NUMPY_AVAILABLE:
                vector = np.random.randn(512).astype(np.float32)
            else:
                vector = [random.gauss(0, 1) for _ in range(512)]

            embedding = EmbeddingVector(
                vector=vector,
                metadata={"description": "Sample embedding", "source": "example"}
            )

            # Store the embedding
            try:
                vector_id = wdbx.store_embedding(embedding)
                print(f"Stored embedding with ID: {vector_id}")
            except Exception as e:
                print(f"Error storing embedding: {e}")

            # Create a conversation block
            print("\nCreating a conversation block...")
            try:
                block_id = wdbx.create_conversation_block(
                    data={"user_input": "Tell me about WDBX", "ai_response": "WDBX is a vector database..."},
                    embeddings=[embedding],
                    persona_id="default"
                )
                print(f"Created block with ID: {block_id}")
            except Exception as e:
                print(f"Error creating block: {e}")

            # Get system stats
            print("\nSystem statistics:")
            stats = wdbx.get_system_stats()
            for key, value in stats.items():
                if not isinstance(value, dict):
                    print(f"  {key}: {value}")

            # Shutdown
            print("\nShutting down WDBX...")
            wdbx.close()
