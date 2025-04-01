# wdbx/cluster.py
"""
High Availability Cluster Configuration for WDBX.

This module provides a comprehensive high availability solution for WDBX,
including:
- Leader election
- Data replication
- Automatic failover
- Consensus protocol
- Distributed coordination
- Synchronization mechanisms
"""
import os
import sys
import time
import uuid
import json
import pickle
import hashlib
import logging
import threading
import socket
import random
import binascii
import struct
import fcntl
import errno
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Callable
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
import queue

try:
    import asyncio
    ASYNCIO_AVAILABLE = True
except ImportError:
    ASYNCIO_AVAILABLE = False
    logging.warning("AsyncIO not available. Using synchronous fallbacks.")

try:
    import etcd3
    ETCD_AVAILABLE = True
except ImportError:
    ETCD_AVAILABLE = False
    logging.warning("etcd3 not available. Using file-based coordination.")

try:
    import zookeeper
    ZOOKEEPER_AVAILABLE = True
except ImportError:
    ZOOKEEPER_AVAILABLE = False
    logging.warning("ZooKeeper not available. Using alternative coordination.")

from wdbx.constants import logger
from wdbx.data_structures import Block, EmbeddingVector

# Cluster configuration
CLUSTER_NAME = os.environ.get("WDBX_CLUSTER_NAME", "wdbx-cluster")
CLUSTER_NODES = os.environ.get("WDBX_CLUSTER_NODES", "127.0.0.1:8080").split(",")
CLUSTER_PORT = int(os.environ.get("WDBX_CLUSTER_PORT", 8500))
CLUSTER_SECRET = os.environ.get("WDBX_CLUSTER_SECRET", "")
CLUSTER_DATA_DIR = os.environ.get("WDBX_CLUSTER_DATA_DIR", "./wdbx_cluster")
CLUSTER_ETCD_ENDPOINTS = os.environ.get("WDBX_ETCD_ENDPOINTS", "").split(",") if os.environ.get("WDBX_ETCD_ENDPOINTS") else []
CLUSTER_ZOOKEEPER_HOSTS = os.environ.get("WDBX_ZOOKEEPER_HOSTS", "").split(",") if os.environ.get("WDBX_ZOOKEEPER_HOSTS") else []
CLUSTER_HEARTBEAT_INTERVAL = float(os.environ.get("WDBX_HEARTBEAT_INTERVAL", 1.0))
CLUSTER_ELECTION_TIMEOUT = float(os.environ.get("WDBX_ELECTION_TIMEOUT", 5.0))
CLUSTER_REPLICATION_FACTOR = int(os.environ.get("WDBX_REPLICATION_FACTOR", 3))


class NodeState:
    """Possible states for a cluster node."""
    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
    OBSERVER = "observer"
    OFFLINE = "offline"


class Node:
    """
    Represents a node in the WDBX cluster.
    """
    def __init__(self, node_id: str, host: str, port: int):
        """
        Initialize a node.

        Args:
            node_id: Unique identifier for the node
            host: Hostname or IP address
            port: Port number
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.state = NodeState.UNKNOWN
        self.last_heartbeat: float = 0.0
        self.term = 0
        self.voted_for: Optional[str] = None
        self.leader_id: Optional[str] = None
        self.data_version = 0
        self.capabilities: Set[str] = set()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the node
        """
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "state": self.state,
            "last_heartbeat": self.last_heartbeat,
            "term": self.term,
            "voted_for": self.voted_for,
            "leader_id": self.leader_id,
            "data_version": self.data_version,
            "capabilities": list(self.capabilities)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """
        Create a node from a dictionary.

        Args:
            data: Dictionary representation of the node

        Returns:
            Node object
        """
        node = cls(data["node_id"], data["host"], data["port"])
        node.state = data["state"]
        node.last_heartbeat = float(data["last_heartbeat"])
        node.term = data["term"]
        node.voted_for = data["voted_for"]
        node.leader_id = data["leader_id"]
        node.data_version = data["data_version"]
        node.capabilities = set(data.get("capabilities", []))
        return node

    def is_alive(self, timeout: float = CLUSTER_ELECTION_TIMEOUT) -> bool:
        """
        Check if the node is alive based on heartbeat.

        Args:
            timeout: Heartbeat timeout in seconds

        Returns:
            True if the node is alive, False otherwise
        """
        return self.last_heartbeat > time.time() - timeout


class ClusterChange:
    """
    Represents a change in the cluster state.
    """
    def __init__(self, node_id: str, change_type: str, old_state: Optional[str] = None,
                new_state: Optional[str] = None, timestamp: Optional[float] = None,
                details: Optional[Dict[str, Any]] = None):
        """
        Initialize a cluster change.

        Args:
            node_id: ID of the node that changed
            change_type: Type of change (e.g., "state", "heartbeat", "term", etc.)
            old_state: Previous state, or None
            new_state: New state, or None
            timestamp: Timestamp of the change, or None for current time
            details: Additional details about the change
        """
        self.node_id = node_id
        self.change_type = change_type
        self.old_state = old_state
        self.new_state = new_state
        self.timestamp = timestamp or time.time()
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the change
        """
        return {
            "node_id": self.node_id,
            "change_type": self.change_type,
            "old_state": self.old_state,
            "new_state": self.new_state,
            "timestamp": self.timestamp,
            "details": self.details
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterChange':
        """
        Create a cluster change from a dictionary.

        Args:
            data: Dictionary representation of the change

        Returns:
            ClusterChange object
        """
        return cls(
            node_id=data["node_id"],
            change_type=data["change_type"],
            old_state=data["old_state"],
            new_state=data["new_state"],
            timestamp=data["timestamp"],
            details=data["details"]
        )


class CoordinationBackend:
    """
    Base class for coordination backends.
    """
    def __init__(self):
        """Initialize the coordination backend."""
        pass

    def initialize(self) -> bool:
        """
        Initialize the backend.

        Returns:
            True if initialization was successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement initialize()")

    def get(self, key: str) -> Optional[bytes]:
        """
        Get a value from the backend.

        Args:
            key: Key to get

        Returns:
            Value if found, None otherwise
        """
        raise NotImplementedError("Subclasses must implement get()")

    def put(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """
        Put a value into the backend.

        Args:
            key: Key to store
            value: Value to store
            ttl: Time-to-live in seconds, or None for no expiration

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement put()")

    def delete(self, key: str) -> bool:
        """
        Delete a key from the backend.

        Args:
            key: Key to delete

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement delete()")

    def watch(self, key: str, callback: Callable[[str, Optional[bytes]], None]) -> Any:
        """
        Watch a key for changes.

        Args:
            key: Key to watch
            callback: Function to call when the key changes

        Returns:
            Watch identifier or object
        """
        raise NotImplementedError("Subclasses must implement watch()")

    def unwatch(self, watch_id: Any) -> bool:
        """
        Stop watching a key.

        Args:
            watch_id: Watch identifier or object

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement unwatch()")

    def lock(self, name: str, ttl: int = 60) -> Any:
        """
        Acquire a lock.

        Args:
            name: Lock name
            ttl: Time-to-live in seconds

        Returns:
            Lock object or identifier
        """
        raise NotImplementedError("Subclasses must implement lock()")

    def unlock(self, lock_id: Any) -> bool:
        """
        Release a lock.

        Args:
            lock_id: Lock object or identifier

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement unlock()")

    def close(self) -> None:
        """Close the backend and release resources."""
        raise NotImplementedError("Subclasses must implement close()")


class EtcdBackend(CoordinationBackend):
    """
    Coordination backend using etcd.
    """
    def __init__(self, endpoints: List[str] = CLUSTER_ETCD_ENDPOINTS):
        """
        Initialize the etcd backend.

        Args:
            endpoints: List of etcd endpoints (host:port)
        """
        super().__init__()
        self.endpoints = endpoints
        self.client = None
        self.watches = {}
        self.locks = {}

    def initialize(self) -> bool:
        """
        Initialize the backend.

        Returns:
            True if initialization was successful, False otherwise
        """
        if not ETCD_AVAILABLE:
            logger.error("etcd3 library not available.")
            return False

        try:
            # Parse endpoints
            hosts = []
            for endpoint in self.endpoints:
                parts = endpoint.split(":")
                if len(parts) == 2:
                    host, port = parts
                    hosts.append((host, int(port)))
                else:
                    hosts.append((endpoint, 2379))  # Default etcd port

            # Connect to etcd
            if hosts:
                host, port = hosts[0]
                self.client = etcd3.client(host=host, port=port)
                # Test connection
                self.client.status()
                logger.info(f"Connected to etcd at {host}:{port}")
                return True
            else:
                logger.error("No valid etcd endpoints provided.")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize etcd backend: {e}")
            return False

    def get(self, key: str) -> Optional[bytes]:
        """
        Get a value from etcd.

        Args:
            key: Key to get

        Returns:
            Value if found, None otherwise
        """
        if not self.client:
            return None

        try:
            result = self.client.get(key)
            if result[0]:
                return result[0]
            return None
        except Exception as e:
            logger.error(f"Failed to get key from etcd: {e}")
            return None

    def put(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """
        Put a value into etcd.

        Args:
            key: Key to store
            value: Value to store
            ttl: Time-to-live in seconds, or None for no expiration

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False

        try:
            if ttl:
                lease = self.client.lease(ttl)
                self.client.put(key, value, lease=lease)
            else:
                self.client.put(key, value)
            return True
        except Exception as e:
            logger.error(f"Failed to put key into etcd: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete a key from etcd.

        Args:
            key: Key to delete

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False

        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete key from etcd: {e}")
            return False

    def watch(self, key: str, callback: Callable[[str, Optional[bytes]], None]) -> Any:
        """
        Watch a key for changes.

        Args:
            key: Key to watch
            callback: Function to call when the key changes

        Returns:
            Watch identifier
        """
        if not self.client:
            return None

        try:
            def etcd_callback(event):
                if event.type == "PUT":
                    callback(key, event.value)
                elif event.type == "DELETE":
                    callback(key, None)

            watch_id = str(uuid.uuid4())
            watch = self.client.add_watch_callback(key, etcd_callback)
            self.watches[watch_id] = watch
            return watch_id
        except Exception as e:
            logger.error(f"Failed to watch key in etcd: {e}")
            return None

    def unwatch(self, watch_id: str) -> bool:
        """
        Stop watching a key.

        Args:
            watch_id: Watch identifier

        Returns:
            True if successful, False otherwise
        """
        if not self.client or watch_id not in self.watches:
            return False

        try:
            watch = self.watches[watch_id]
            self.client.cancel_watch(watch)
            del self.watches[watch_id]
            return True
        except Exception as e:
            logger.error(f"Failed to unwatch key in etcd: {e}")
            return False

    def lock(self, name: str, ttl: int = 60) -> Any:
        """
        Acquire a lock.

        Args:
            name: Lock name
            ttl: Time-to-live in seconds

        Returns:
            Lock object
        """
        if not self.client:
            return None

        try:
            lock_id = str(uuid.uuid4())
            lock = self.client.lock(name, ttl=ttl)
            if lock.acquire():
                self.locks[lock_id] = lock
                return lock_id
            return None
        except Exception as e:
            logger.error(f"Failed to acquire lock in etcd: {e}")
            return None

    def unlock(self, lock_id: Any) -> bool:
        """
        Release a lock.

        Args:
            lock_id: Lock identifier

        Returns:
            True if successful, False otherwise
        """
        if not self.client or lock_id not in self.locks:
            return False

        try:
            lock = self.locks[lock_id]
            lock.release()
            del self.locks[lock_id]
            return True
        except Exception as e:
            logger.error(f"Failed to release lock in etcd: {e}")
            return False

    def close(self) -> None:
        """Close the etcd connection and release resources."""
        if self.client:
            try:
                # Cancel all watches
                for watch_id in list(self.watches.keys()):
                    self.unwatch(watch_id)

                # Release all locks
                for lock_id in list(self.locks.keys()):
                    self.unlock(lock_id)

                # Close the client
                self.client.close()
                self.client = None
            except Exception as e:
                logger.error(f"Failed to close etcd connection: {e}")


class FileBackend(CoordinationBackend):
    """
    Coordination backend using the filesystem.
    Simpler alternative when etcd or ZooKeeper are not available.
    """
    def __init__(self, data_dir: str = CLUSTER_DATA_DIR):
        """
        Initialize the file backend.

        Args:
            data_dir: Directory to store data in
        """
        super().__init__()
        self.data_dir = data_dir
        self.values_dir = os.path.join(data_dir, "values")
        self.locks_dir = os.path.join(data_dir, "locks")
        self.watches = {}
        self.watch_thread = None
        self.watch_running = False
        self.watch_queue = queue.Queue()

    def initialize(self) -> bool:
        """
        Initialize the backend.

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Create directories
            os.makedirs(self.values_dir, exist_ok=True)
            os.makedirs(self.locks_dir, exist_ok=True)

            # Start watch thread
            self.watch_running = True
            self.watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
            self.watch_thread.start()

            logger.info(f"Initialized file backend at {self.data_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize file backend: {e}")
            return False

    def _key_to_path(self, key: str) -> str:
        """
        Convert a key to a file path.

        Args:
            key: Key to convert

        Returns:
            File path
        """
        # Make the key safe for the filesystem
        safe_key = key.replace("/", "_").replace(":", "_").replace("\\", "_")
        return os.path.join(self.values_dir, safe_key)

    def _lock_to_path(self, name: str) -> str:
        """
        Convert a lock name to a file path.

        Args:
            name: Lock name

        Returns:
            File path
        """
        # Make the name safe for the filesystem
        safe_name = name.replace("/", "_").replace(":", "_").replace("\\", "_")
        return os.path.join(self.locks_dir, safe_name + ".lock")

    def get(self, key: str) -> Optional[bytes]:
        """
        Get a value from the file backend.

        Args:
            key: Key to get

        Returns:
            Value if found, None otherwise
        """
        path = self._key_to_path(key)
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return f.read()
            return None
        except Exception as e:
            logger.error(f"Failed to get key from file backend: {e}")
            return None

    def put(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """
        Put a value into the file backend.

        Args:
            key: Key to store
            value: Value to store
            ttl: Time-to-live in seconds, or None for no expiration

        Returns:
            True if successful, False otherwise
        """
        path = self._key_to_path(key)
        try:
            with open(path, "wb") as f:
                f.write(value)

            # Handle TTL
            if ttl:
                expiration = time.time() + ttl
                ttl_path = path + ".ttl"
                with open(ttl_path, "w") as f:
                    f.write(str(expiration))

            # Notify watchers
            self.watch_queue.put((key, value))

            return True
        except Exception as e:
            logger.error(f"Failed to put key into file backend: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete a key from the file backend.

        Args:
            key: Key to delete

        Returns:
            True if successful, False otherwise
        """
        path = self._key_to_path(key)
        try:
            if os.path.exists(path):
                os.remove(path)

                # Remove TTL file if it exists
                ttl_path = path + ".ttl"
                if os.path.exists(ttl_path):
                    os.remove(ttl_path)

                # Notify watchers
                self.watch_queue.put((key, None))

                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete key from file backend: {e}")
            return False

    def _watch_loop(self):
        """Watch loop for processing watch events."""
        while self.watch_running:
            try:
                # Process TTL expirations
                self._check_ttl_expirations()

                # Process watch events
                try:
                    key, value = self.watch_queue.get(timeout=1.0)
                    if key in self.watches:
                        for callback in self.watches[key].values():
                            try:
                                callback(key, value)
                            except Exception as e:
                                logger.error(f"Error in watch callback: {e}")
                except queue.Empty:
                    pass
            except Exception as e:
                logger.error(f"Error in watch loop: {e}")

            # Sleep a bit
            time.sleep(0.1)

    def _check_ttl_expirations(self):
        """Check for TTL expirations."""
        now = time.time()
        try:
            for filename in os.listdir(self.values_dir):
                if filename.endswith(".ttl"):
                    ttl_path = os.path.join(self.values_dir, filename)
                    try:
                        with open(ttl_path, "r") as f:
                            expiration = float(f.read().strip())

                        if now > expiration:
                            # TTL expired, delete the key
                            key_path = ttl_path[:-4]  # Remove ".ttl"
                            key = os.path.basename(key_path)

                            # Delete the files
                            if os.path.exists(key_path):
                                os.remove(key_path)
                            os.remove(ttl_path)

                            # Notify watchers
                            self.watch_queue.put((key, None))
                    except Exception as e:
                        logger.error(f"Error checking TTL expiration for {filename}: {e}")
        except Exception as e:
            logger.error(f"Error checking TTL expirations: {e}")

    def watch(self, key: str, callback: Callable[[str, Optional[bytes]], None]) -> Any:
        """
        Watch a key for changes.

        Args:
            key: Key to watch
            callback: Function to call when the key changes

        Returns:
            Watch identifier
        """
        watch_id = str(uuid.uuid4())

        # Initialize the watches for this key
        if key not in self.watches:
            self.watches[key] = {}

        # Add the callback
        self.watches[key][watch_id] = callback

        return watch_id

    def unwatch(self, watch_id: Any) -> bool:
        """
        Stop watching a key.

        Args:
            watch_id: Watch identifier

        Returns:
            True if successful, False otherwise
        """
        # Find the key for this watch ID
        for key, watches in self.watches.items():
            if watch_id in watches:
                del watches[watch_id]
                if not watches:
                    del self.watches[key]
                return True

        return False

    def lock(self, name: str, ttl: int = 60) -> Any:
        """
        Acquire a lock.

        Args:
            name: Lock name
            ttl: Time-to-live in seconds

        Returns:
            Lock object
        """
        lock_path = self._lock_to_path(name)
        lock_id = str(uuid.uuid4())

        # Try to create the lock file
        try:
            with open(lock_path, "x") as f:
                expiration = time.time() + ttl
                lock_data = {
                    "id": lock_id,
                    "expiration": expiration
                }
                f.write(json.dumps(lock_data))

            return lock_id
        except FileExistsError:
            # Lock already exists, check if it's expired
            try:
                with open(lock_path, "r") as f:
                    lock_data = json.loads(f.read())

                if time.time() > lock_data["expiration"]:
                    # Lock expired, try to acquire it
                    os.remove(lock_path)
                    return self.lock(name, ttl)
            except Exception:
                pass

            return None
        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            return None

    def unlock(self, lock_id: Any) -> bool:
        """
        Release a lock.

        Args:
            lock_id: Lock identifier

        Returns:
            True if successful, False otherwise
        """
        # Find the lock file for this lock ID
        try:
            for filename in os.listdir(self.locks_dir):
                lock_path = os.path.join(self.locks_dir, filename)
                try:
                    with open(lock_path, "r") as f:
                        lock_data = json.loads(f.read())

                    if lock_data["id"] == lock_id:
                        os.remove(lock_path)
                        return True
                except Exception:
                    pass

            return False
        except Exception as e:
            logger.error(f"Failed to unlock: {e}")
            return False

    def close(self) -> None:
        """Close the file backend and release resources."""
        self.watch_running = False
        if self.watch_thread:
            self.watch_thread.join(timeout=1.0)


class ClusterCoordinator:
    """
    Coordinates cluster operations.
    """
    def __init__(self, node_id: str, backends: Optional[List[str]] = None):
        """
        Initialize the cluster coordinator.

        Args:
            node_id: ID of this node
            backends: List of backend types to try, in order of preference
        """
        self.node_id = node_id
        self.backends = backends or ["etcd", "file"]
        self.backend = None
        self.cluster_key_prefix = f"/wdbx/cluster/{CLUSTER_NAME}"
        self.nodes_key = f"{self.cluster_key_prefix}/nodes"
        self.leader_key = f"{self.cluster_key_prefix}/leader"
        self.locks = {}
        self.watches = []

    def initialize(self) -> bool:
        """
        Initialize the cluster coordinator.

        Returns:
            True if initialization was successful, False otherwise
        """
        # Try each backend in order of preference
        for backend_type in self.backends:
            if backend_type == "etcd" and ETCD_AVAILABLE:
                self.backend = EtcdBackend()
                if self.backend.initialize():
                    break
            elif backend_type == "file":
                self.backend = FileBackend()
                if self.backend.initialize():
                    break

        if not self.backend:
            logger.error("Failed to initialize any coordination backend.")
            return False

        logger.info(f"Initialized cluster coordinator with {type(self.backend).__name__}")
        return True

    def register_node(self, node: Node, ttl: int = 30) -> bool:
        """
        Register a node in the cluster.

        Args:
            node: Node to register
            ttl: Time-to-live in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self.backend:
            return False

        node_key = f"{self.nodes_key}/{node.node_id}"
        try:
            # Convert node to bytes
            node_data = json.dumps(node.to_dict()).encode("utf-8")

            # Store in the backend
            return self.backend.put(node_key, node_data, ttl)
        except Exception as e:
            logger.error(f"Failed to register node: {e}")
            return False

    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Get a node from the cluster.

        Args:
            node_id: ID of the node to get

        Returns:
            Node if found, None otherwise
        """
        if not self.backend:
            return None

        node_key = f"{self.nodes_key}/{node_id}"
        try:
            node_data = self.backend.get(node_key)
            if node_data:
                node_dict = json.loads(node_data.decode("utf-8"))
                return Node.from_dict(node_dict)
            return None
        except Exception as e:
            logger.error(f"Failed to get node: {e}")
            return None

    def get_all_nodes(self) -> Dict[str, Node]:
        """
        Get all nodes in the cluster.

        Returns:
            Dictionary of node_id -> Node
        """
        if not self.backend:
            return {}

        try:
            nodes = {}

            # List all node keys
            # For a real implementation, we would need a "list" operation
            # For now, use some backend-specific hacks
            if isinstance(self.backend, EtcdBackend) and self.backend.client:
                # For etcd, we can use the prefix feature
                for item in self.backend.client.get_prefix(self.nodes_key):
                    key = item[1].key.decode("utf-8")
                    node_id = key.split("/")[-1]
                    node_dict = json.loads(item[0].decode("utf-8"))
                    nodes[node_id] = Node.from_dict(node_dict)
            elif isinstance(self.backend, FileBackend):
                # For file backend, we can list the directory
                prefix = self.nodes_key.replace("/", "_")
                for filename in os.listdir(self.backend.values_dir):
                    if filename.startswith(prefix) and not filename.endswith(".ttl"):
                        node_id = filename.split("_")[-1]
                        node_data = self.backend.get(f"{self.nodes_key}/{node_id}")
                        if node_data:
                            node_dict = json.loads(node_data.decode("utf-8"))
                            nodes[node_id] = Node.from_dict(node_dict)

            return nodes
        except Exception as e:
            logger.error(f"Failed to get all nodes: {e}")
            return {}

    def watch_nodes(self, callback: Callable[[str, Optional[Node]], None]) -> bool:
        """
        Watch for node changes.

        Args:
            callback: Function to call when a node changes

        Returns:
            True if successful, False otherwise
        """
        if not self.backend:
            return False

        try:
            def node_callback(key, value):
                node_id = key.split("/")[-1]
                if value:
                    node_dict = json.loads(value.decode("utf-8"))
                    node = Node.from_dict(node_dict)
                    callback(node_id, node)
                else:
                    callback(node_id, None)

            watch_id = self.backend.watch(self.nodes_key, node_callback)
            if watch_id:
                self.watches.append(watch_id)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to watch nodes: {e}")
            return False

    def elect_leader(self, node_id: str, term: int) -> bool:
        """
        Try to elect a leader.

        Args:
            node_id: ID of the node to elect
            term: Election term

        Returns:
            True if the node was elected, False otherwise
        """
        if not self.backend:
            return False

        try:
            # Try to acquire the leader lock
            lock_id = self.backend.lock(f"{self.cluster_key_prefix}/leader-election", ttl=10)
            if not lock_id:
                return False

            try:
                # Check current leader
                leader_data = self.backend.get(self.leader_key)
                if leader_data:
                    leader_info = json.loads(leader_data.decode("utf-8"))
                    current_term = leader_info.get("term", 0)

                    # Only replace leader if term is higher
                    if term <= current_term:
                        return False

                # Set the new leader
                leader_info = {
                    "node_id": node_id,
                    "term": term,
                    "timestamp": time.time()
                }
                leader_bytes = json.dumps(leader_info).encode("utf-8")

                if self.backend.put(self.leader_key, leader_bytes):
                    logger.info(f"Node {node_id} elected as leader for term {term}")
                    return True

                return False
            finally:
                # Release the lock
                self.backend.unlock(lock_id)
        except Exception as e:
            logger.error(f"Failed to elect leader: {e}")
            return False

    def get_leader(self) -> Tuple[Optional[str], int]:
        """
        Get the current leader.

        Returns:
            Tuple of (leader_id, term), or (None, 0) if no leader
        """
        if not self.backend:
            return None, 0

        try:
            leader_data = self.backend.get(self.leader_key)
            if leader_data:
                leader_info = json.loads(leader_data.decode("utf-8"))
                return leader_info.get("node_id"), leader_info.get("term", 0)
            return None, 0
        except Exception as e:
            logger.error(f"Failed to get leader: {e}")
            return None, 0

    def watch_leader(self, callback: Callable[[Optional[str], int], None]) -> bool:
        """
        Watch for leader changes.

        Args:
            callback: Function to call when the leader changes

        Returns:
            True if successful, False otherwise
        """
        if not self.backend:
            return False

        try:
            def leader_callback(key, value):
                if value:
                    leader_info = json.loads(value.decode("utf-8"))
                    callback(leader_info.get("node_id"), leader_info.get("term", 0))
                else:
                    callback(None, 0)

            watch_id = self.backend.watch(self.leader_key, leader_callback)
            if watch_id:
                self.watches.append(watch_id)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to watch leader: {e}")
            return False

    def acquire_lock(self, name: str, ttl: int = 60) -> Optional[str]:
        """
        Acquire a cluster-wide lock.

        Args:
            name: Lock name
            ttl: Time-to-live in seconds

        Returns:
            Lock ID if successful, None otherwise
        """
        if not self.backend:
            return None

        try:
            lock_id = self.backend.lock(f"{self.cluster_key_prefix}/locks/{name}", ttl)
            if lock_id:
                self.locks[name] = lock_id
                return name
            return None
        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            return None

    def release_lock(self, name: str) -> bool:
        """
        Release a cluster-wide lock.

        Args:
            name: Lock name

        Returns:
            True if successful, False otherwise
        """
        if not self.backend or name not in self.locks:
            return False

        try:
            lock_id = self.locks[name]
            if self.backend.unlock(lock_id):
                del self.locks[name]
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to release lock: {e}")
            return False

    def close(self) -> None:
        """Close the coordinator and release resources."""
        if self.backend:
            try:
                # Release all locks
                for name in list(self.locks.keys()):
                    self.release_lock(name)

                # Remove all watches
                for watch_id in self.watches:
                    self.backend.unwatch(watch_id)

                # Close the backend
                self.backend.close()
            except Exception as e:
                logger.error(f"Failed to close coordinator: {e}")
            finally:
                self.backend = None


class ReplicationManager:
    """
    Manages data replication between nodes.
    """
    def __init__(self, node_id: str, wdbx_instance: Any, coordinator: ClusterCoordinator):
        """
        Initialize the replication manager.

        Args:
            node_id: ID of this node
            wdbx_instance: WDBX instance
            coordinator: Cluster coordinator
        """
        self.node_id = node_id
        self.wdbx = wdbx_instance
        self.coordinator = coordinator
        self.replication_queue = queue.Queue()
        self.replication_thread = None
        self.running = False

    def start(self) -> bool:
        """
        Start the replication manager.

        Returns:
            True if successful, False otherwise
        """
        if self.running:
            return True

        try:
            self.running = True
            self.replication_thread = threading.Thread(target=self._replication_loop, daemon=True)
            self.replication_thread.start()
            logger.info("Replication manager started")
            return True
        except Exception as e:
            logger.error(f"Failed to start replication manager: {e}")
            self.running = False
            return False

    def stop(self) -> None:
        """Stop the replication manager."""
        self.running = False
        if self.replication_thread:
            self.replication_thread.join(timeout=5.0)
            self.replication_thread = None

    def _replication_loop(self) -> None:
        """Background thread for processing replication events."""
        while self.running:
            try:
                # Get the next replication task
                try:
                    item = self.replication_queue.get(timeout=1.0)
                    self._process_replication_item(item)
                except queue.Empty:
                    pass

                # For leader: check for un-replicated blocks
                leader_id, _ = self.coordinator.get_leader()
                if leader_id == self.node_id:
                    self._check_unreplicated_blocks()
            except Exception as e:
                logger.error(f"Error in replication loop: {e}")

            time.sleep(0.1)

    def _process_replication_item(self, item: dict) -> None:
        """
        Process a replication item.

        Args:
            item: Replication item
        """
        op_type = item.get("type")

        if op_type == "block":
            # Replicate a block
            block_id = item.get("block_id")
            block_data = item.get("data")
            if block_id and block_data:
                self._replicate_block(block_id, block_data)
        elif op_type == "embedding":
            # Replicate an embedding vector
            vector_id = item.get("vector_id")
            vector_data = item.get("data")
            if vector_id and vector_data:
                self._replicate_embedding(vector_id, vector_data)

    def _check_unreplicated_blocks(self) -> None:
        """Check for blocks that need replication."""
        # In a real implementation, this would check which blocks don't have enough replicas
        # For now, just log a message
        logger.debug("Checking for unreplicated blocks")

    def _replicate_block(self, block_id: str, block_data: bytes) -> bool:
        """
        Replicate a block.

        Args:
            block_id: ID of the block to replicate
            block_data: Serialized block data

        Returns:
            True if successful, False otherwise
        """
        try:
            from wdbx.data_structures import Block

            # Deserialize the block
            block = Block.deserialize(block_data)

            # Validate the block
            if not block.validate():
                logger.warning(f"Invalid block received for replication: {block_id}")
                return False

            # Store the block
            self.wdbx.block_chain_manager.blocks[block_id] = block

            # Update chain information
            if block.previous_hash:
                # Find the chain this block belongs to
                for chain_id, head_id in self.wdbx.block_chain_manager.chain_heads.items():
                    head_block = self.wdbx.block_chain_manager.blocks.get(head_id)
                    if head_block and head_block.hash == block.previous_hash:
                        # Update chain head
                        self.wdbx.block_chain_manager.chain_heads[chain_id] = block_id
                        self.wdbx.block_chain_manager.block_chain[block_id] = chain_id
                        break

            logger.debug(f"Replicated block {block_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to replicate block {block_id}: {e}")
            return False

    def _replicate_embedding(self, vector_id: str, vector_data: bytes) -> bool:
        """
        Replicate an embedding vector.

        Args:
            vector_id: ID of the vector to replicate
            vector_data: Serialized vector data

        Returns:
            True if successful, False otherwise
        """
        try:
            from wdbx.data_structures import EmbeddingVector

            # Deserialize the vector
            vector = EmbeddingVector.deserialize(vector_data)

            # Store the vector
            self.wdbx.vector_store.vectors[vector_id] = vector

            logger.debug(f"Replicated vector {vector_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to replicate vector {vector_id}: {e}")
            return False

    def replicate_block(self, block_id: str) -> bool:
        """
        Schedule a block for replication.

        Args:
            block_id: ID of the block to replicate

        Returns:
            True if scheduled successfully, False otherwise
        """
        try:
            # Get the block
            block = self.wdbx.block_chain_manager.get_block(block_id)
            if not block:
                logger.warning(f"Block not found for replication: {block_id}")
                return False

            # Serialize the block
            block_data = block.serialize()

            # Add to replication queue
            self.replication_queue.put({
                "type": "block",
                "block_id": block_id,
                "data": block_data
            })

            return True
        except Exception as e:
            logger.error(f"Failed to schedule block replication: {e}")
            return False

    def replicate_embedding(self, vector_id: str) -> bool:
        """
        Schedule an embedding vector for replication.

        Args:
            vector_id: ID of the vector to replicate

        Returns:
            True if scheduled successfully, False otherwise
        """
        try:
            # Get the vector
            vector = self.wdbx.vector_store.get(vector_id)
            if not vector:
                logger.warning(f"Vector not found for replication: {vector_id}")
                return False

            # Serialize the vector
            vector_data = vector.serialize()

            # Add to replication queue
            self.replication_queue.put({
                "type": "embedding",
                "vector_id": vector_id,
                "data": vector_data
            })

            return True
        except Exception as e:
            logger.error(f"Failed to schedule vector replication: {e}")
            return False


class ClusterNode:
    """
    Represents a node in the WDBX cluster, with state machine replication.
    """
    def __init__(self, wdbx_instance: Any, node_id: Optional[str] = None,
                host: Optional[str] = None, port: Optional[int] = None):
        """
        Initialize a cluster node.

        Args:
            wdbx_instance: WDBX instance
            node_id: Node ID, or None to generate one
            host: Host address, or None for default
            port: Port number, or None for default
        """
        self.wdbx = wdbx_instance
        self.node_id = node_id or str(uuid.uuid4())
        self.host = host or socket.gethostname()
        self.port = port or CLUSTER_PORT

        # Node state
        self.state = NodeState.INITIALIZING
        self.term = 0
        self.voted_for = None
        self.leader_id = None
        self.votes_received = set()

        # Timers
        self.last_heartbeat = 0
        self.election_timeout = CLUSTER_ELECTION_TIMEOUT
        self.heartbeat_interval = CLUSTER_HEARTBEAT_INTERVAL

        # Consensus components
        self.coordinator = ClusterCoordinator(self.node_id)
        self.replication_manager = ReplicationManager(self.node_id, wdbx_instance, self.coordinator)

        # Node operation
        self.running = False
        self.consensus_thread = None
        self.heartbeat_thread = None

    def start(self) -> bool:
        """
        Start the cluster node.

        Returns:
            True if successful, False otherwise
        """
        if self.running:
            return True

        try:
            # Initialize coordinator
            if not self.coordinator.initialize():
                logger.error("Failed to initialize cluster coordinator.")
                return False

            # Initialize replication manager
            if not self.replication_manager.start():
                logger.error("Failed to start replication manager.")
                return False

            # Register this node in the cluster
            node = Node(self.node_id, self.host, self.port)
            node.state = NodeState.FOLLOWER
            node.term = self.term
            node.last_heartbeat = time.time()

            if not self.coordinator.register_node(node):
                logger.error("Failed to register node in the cluster.")
                return False

            # Start consensus thread
            self.running = True
            self.state = NodeState.FOLLOWER
            self.consensus_thread = threading.Thread(target=self._consensus_loop, daemon=True)
            self.consensus_thread.start()

            # Start heartbeat thread
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()

            logger.info(f"Cluster node {self.node_id} started")
            return True
        except Exception as e:
            logger.error(f"Failed to start cluster node: {e}")
            self.stop()
            return False

    def stop(self) -> None:
        """Stop the cluster node."""
        self.running = False

        # Stop threads
        if self.consensus_thread:
            self.consensus_thread.join(timeout=5.0)
            self.consensus_thread = None

        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5.0)
            self.heartbeat_thread = None

        # Stop replication manager
        if self.replication_manager:
            self.replication_manager.stop()

        # Close coordinator
        if self.coordinator:
            self.coordinator.close()

        logger.info(f"Cluster node {self.node_id} stopped")

    def _consensus_loop(self) -> None:
        """Consensus algorithm main loop."""
        random.seed()

        # Randomize first election timeout
        election_timeout = self.election_timeout * (1 + random.random())
        election_timer = time.time() + election_timeout

        # Set up leader watch
        self.coordinator.watch_leader(self._on_leader_change)

        while self.running:
            try:
                current_time = time.time()

                if self.state == NodeState.FOLLOWER:
                    # Check if we should become a candidate
                    if current_time > election_timer:
                        logger.info(f"Election timeout, becoming candidate for term {self.term + 1}")
                        self._become_candidate()

                        # Reset election timer
                        election_timeout = self.election_timeout * (1 + random.random())
                        election_timer = current_time + election_timeout

                elif self.state == NodeState.CANDIDATE:
                    # Check if we have enough votes to become leader
                    nodes = self.coordinator.get_all_nodes()
                    required_votes = (len(nodes) // 2) + 1

                    if len(self.votes_received) >= required_votes:
                        logger.info(f"Received majority votes ({len(self.votes_received)}), becoming leader")
                        self._become_leader()
                    elif current_time > election_timer:
                        logger.info(f"Election timeout, starting new election for term {self.term + 1}")
                        self.term += 1
                        self.voted_for = self.node_id
                        self.votes_received = {self.node_id}

                        # Request votes from other nodes
                        self._request_votes()

                        # Reset election timer
                        election_timeout = self.election_timeout * (1 + random.random())
                        election_timer = current_time + election_timeout

                elif self.state == NodeState.LEADER:
                    # Send heartbeats periodically
                    if current_time - self.last_heartbeat >= self.heartbeat_interval:
                        self._send_heartbeat()
                        self.last_heartbeat = current_time

                # Update node status
                self._update_node_status()
            except Exception as e:
                logger.error(f"Error in consensus loop: {e}")

            # Sleep a bit
            time.sleep(0.1)

    def _heartbeat_loop(self) -> None:
        """Heartbeat loop for registering node status."""
        while self.running:
            try:
                # Register node with TTL
                node = Node(self.node_id, self.host, self.port)
                node.state = self.state
                node.term = self.term
                node.voted_for = self.voted_for
                node.leader_id = self.leader_id
                node.last_heartbeat = time.time()

                self.coordinator.register_node(node, ttl=30)
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

            # Sleep for heartbeat interval
            time.sleep(self.heartbeat_interval)

    def _update_node_status(self) -> None:
        """Update node status in the cluster."""
        try:
            # Get current nodes
            nodes = self.coordinator.get_all_nodes()

            # Check for dead nodes
            for node_id, node in nodes.items():
                if node.state != NodeState.OFFLINE and not node.is_alive():
                    node.state = NodeState.OFFLINE
                    logger.info(f"Node {node_id} is offline")
        except Exception as e:
            logger.error(f"Error updating node status: {e}")

    def _become_candidate(self) -> None:
        """Become a candidate and start an election."""
        self.state = NodeState.CANDIDATE
        self.term += 1
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}

        # Request votes from other nodes
        self._request_votes()

    def _request_votes(self) -> None:
        """Request votes from other nodes."""
        try:
            # Get all nodes
            nodes = self.coordinator.get_all_nodes()

            # Request votes from other nodes
            for node_id, node in nodes.items():
                if node_id != self.node_id and node.state != NodeState.OFFLINE:
                    # In a real implementation, this would send an RPC to each node
                    # For now, just simulate with the coordinator
                    self._simulate_vote_request(node_id)
        except Exception as e:
            logger.error(f"Error requesting votes: {e}")

    def _simulate_vote_request(self, node_id: str) -> None:
        """
        Simulate a vote request to another node.

        Args:
            node_id: ID of the node to request a vote from
        """
        try:
            # Get the node
            node = self.coordinator.get_node(node_id)
            if not node:
                return

            # Check if the node has already voted for this term
            if node.voted_for is not None and node.voted_for != self.node_id and node.term >= self.term:
                return

            # Simulate the node granting the vote
            if node.term <= self.term:
                self.votes_received.add(node_id)
                logger.debug(f"Received vote from node {node_id}")
        except Exception as e:
            logger.error(f"Error simulating vote request: {e}")

    def _become_leader(self) -> None:
        """Become the leader of the cluster."""
        if self.state == NodeState.CANDIDATE:
            # Try to elect as leader
            if self.coordinator.elect_leader(self.node_id, self.term):
                self.state = NodeState.LEADER
                self.leader_id = self.node_id
                logger.info(f"Node {self.node_id} is now the leader for term {self.term}")

                # Send initial heartbeat
                self._send_heartbeat()
            else:
                logger.warning(f"Failed to elect as leader for term {self.term}")

    def _send_heartbeat(self) -> None:
        """Send heartbeat to all nodes."""
        if self.state != NodeState.LEADER:
            return

        try:
            # In a real implementation, this would send heartbeats to all nodes
            # For now, just update the leader key with fresh TTL
            self.coordinator.elect_leader(self.node_id, self.term)

            # Debug log
            logger.debug(f"Leader {self.node_id} sent heartbeat for term {self.term}")
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")

    def _on_leader_change(self, leader_id: Optional[str], term: int) -> None:
        """
        Handle leader change events.

        Args:
            leader_id: ID of the new leader, or None if no leader
            term: Term of the new leader
        """
        try:
            if leader_id == self.node_id:
                # We are the leader
                if self.state != NodeState.LEADER:
                    self.state = NodeState.LEADER
                    logger.info(f"This node is now the leader for term {term}")
            else:
                # Someone else is the leader
                if leader_id:
                    if term > self.term:
                        # Update term
                        self.term = term
                        self.voted_for = None

                    # Update leader ID
                    self.leader_id = leader_id

                    # If we were a candidate or leader, step down
                    if self.state in (NodeState.CANDIDATE, NodeState.LEADER):
                        self.state = NodeState.FOLLOWER
                        logger.info(f"Stepping down, node {leader_id} is the leader for term {term}")
                else:
                    # No leader
                    self.leader_id = None
        except Exception as e:
            logger.error(f"Error handling leader change: {e}")

    def handle_vote_request(self, candidate_id: str, term: int) -> bool:
        """
        Handle a vote request from a candidate.

        Args:
            candidate_id: ID of the candidate
            term: Candidate's term

        Returns:
            True if the vote is granted, False otherwise
        """
        try:
            # Check if the term is valid
            if term < self.term:
                return False

            # If we haven't voted for this term, or we voted for this candidate
            if self.voted_for is None or self.voted_for == candidate_id:
                # Grant vote
                self.voted_for = candidate_id
                return True

            return False
        except Exception as e:
            logger.error(f"Error handling vote request: {e}")
            return False

    def get_cluster_state(self) -> Dict[str, Any]:
        """
        Get the current state of the cluster.

        Returns:
            Dictionary with cluster state information
        """
        try:
            # Get all nodes
            nodes = self.coordinator.get_all_nodes()

            # Get leader
            leader_id, term = self.coordinator.get_leader()

            # Collect active nodes
            active_nodes = {}
            for node_id, node in nodes.items():
                if node.is_alive():
                    active_nodes[node_id] = node.to_dict()

            return {
                "nodes": {node_id: node.to_dict() for node_id, node in nodes.items()},
                "active_nodes": active_nodes,
                "leader": leader_id,
                "term": term,
                "node_count": len(nodes),
                "active_count": len(active_nodes)
            }
        except Exception as e:
            logger.error(f"Error getting cluster state: {e}")
            return {
                "error": str(e)
            }


def init_cluster(app, wdbx_instance: Any) -> Optional[ClusterNode]:
    """
    Initialize cluster functionality for a Flask application.

    Args:
        app: Flask application
        wdbx_instance: WDBX instance

    Returns:
        ClusterNode instance
    """
    try:
        # Generate node ID using hostname and port
        hostname = socket.gethostname()
        port = int(os.environ.get("PORT", CLUSTER_PORT))
        node_id = f"{hostname}:{port}"

        # Initialize cluster node
        cluster_node = ClusterNode(wdbx_instance, node_id=node_id, host=hostname, port=port)
        app.cluster_node = cluster_node

        # Start the cluster node
        if not cluster_node.start():
            logger.warning("Failed to start cluster node.")

        # Add cluster-related routes
        @app.route("/cluster/status", methods=["GET"])
        def get_cluster_status():
            """Get the status of the cluster."""
            from flask import jsonify
            return jsonify(cluster_node.get_cluster_state())

        @app.route("/cluster/node/<node_id>", methods=["GET"])
        def get_cluster_node(node_id):
            """Get information about a specific node."""
            from flask import jsonify

            node = cluster_node.coordinator.get_node(node_id)
            if node:
                return jsonify(node.to_dict())
            return jsonify({"error": f"Node {node_id} not found"}), 404

        @app.before_request
        def check_leader():
            """Check if this node is the leader for certain operations."""
            from flask import request, jsonify

            # List of paths that require leader status
            leader_paths = [
                "/v1/blocks/create",
                "/v1/blocks/update",
                "/v1/blocks/delete"
            ]

            if request.path in leader_paths and request.method != "GET":
                if cluster_node.state != NodeState.LEADER:
                    # Redirect to leader if known
                    if cluster_node.leader_id:
                        leader_node = cluster_node.coordinator.get_node(cluster_node.leader_id)
                        if leader_node:
                            return jsonify({
                                "error": "Not the leader",
                                "leader": {
                                    "node_id": leader_node.node_id,
                                    "host": leader_node.host,
                                    "port": leader_node.port
                                }
                            }), 307

                    # No leader known
                    return jsonify({"error": "Not the leader"}), 503

        @app.teardown_appcontext
        def shutdown_cluster(exception=None):
            """Shut down the cluster node when the app is terminated."""
            if hasattr(app, "cluster_node"):
                app.cluster_node.stop()

        logger.info("Cluster functionality initialized")
        return cluster_node
    except Exception as e:
        logger.error(f"Failed to initialize cluster: {e}")
        return None


# Sample usage
if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(description="WDBX Cluster Node")
    parser.add_argument("--port", type=int, default=CLUSTER_PORT, help="Port to listen on")
    parser.add_argument("--host", default=socket.gethostname(), help="Host to bind to")
    parser.add_argument("--nodes", default=",".join(CLUSTER_NODES), help="Comma-separated list of cluster nodes")
    args = parser.parse_args()

    # Update environment variables
    os.environ["WDBX_CLUSTER_PORT"] = str(args.port)
    os.environ["WDBX_CLUSTER_NODES"] = args.nodes

    # Generate node ID
    node_id = f"{args.host}:{args.port}"

    # Create a mock WDBX instance for testing
    class MockWDBX:
        def __init__(self):
            from wdbx.blockchain import BlockChainManager
            class MockVectorStore:
                def __init__(self):
                    self.vectors = {}
                    self.get = self.vectors.get
            self.vector_store = MockVectorStore()
            self.block_chain_manager = BlockChainManager()

    wdbx = MockWDBX()

    # Create and start a cluster node
    node = ClusterNode(wdbx, node_id=node_id, host=args.host, port=args.port)

    try:
        if node.start():
            logger.info(f"Cluster node {node_id} started")

            # Run until interrupted
            while True:
                time.sleep(1)
        else:
            logger.error("Failed to start cluster node.")
    except KeyboardInterrupt:
        logger.info("Stopping cluster node...")
    finally:
        node.stop()
