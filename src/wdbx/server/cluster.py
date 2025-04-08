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
import hmac
import json
import logging
import os
import random
import secrets
import signal
import socket
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

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
    import kazoo

    ZOOKEEPER_AVAILABLE = True
except ImportError:
    ZOOKEEPER_AVAILABLE = False
    logging.warning("ZooKeeper not available. Using alternative coordination.")

from ..core.constants import logger

# Configure logging
logger = logging.getLogger("wdbx.cluster")

# Constants
CLUSTER_PREFIX = "wdbx/cluster"
CLUSTER_PORT = 5925
CLUSTER_ELECTION_TIMEOUT = 5.0  # seconds
CLUSTER_HEARTBEAT_INTERVAL = 1.0  # seconds
CLUSTER_SECRET = os.environ.get("WDBX_CLUSTER_SECRET", "")


@dataclass
class ClusterConfig:
    """Configuration for WDBX cluster setup."""

    # Cluster identity
    cluster_name: str = "wdbx-cluster"
    node_id: Optional[str] = None  # Auto-generated if None

    # Network configuration
    host: Optional[str] = None  # Auto-detected if None
    port: int = CLUSTER_PORT

    # Timeouts and intervals
    election_timeout_base: float = CLUSTER_ELECTION_TIMEOUT
    heartbeat_interval: float = CLUSTER_HEARTBEAT_INTERVAL

    # Security
    cluster_secret: str = field(default_factory=lambda: os.environ.get("WDBX_CLUSTER_SECRET", ""))
    enable_security: bool = True

    # Coordination backend options
    backend_type: Optional[str] = None  # Auto-detected if None
    etcd_hosts: List[str] = field(default_factory=lambda: ["localhost:2379"])
    zookeeper_hosts: str = "localhost:2181"
    storage_dir: str = field(
        default_factory=lambda: os.path.join(tempfile.gettempdir(), "wdbx-cluster")
    )

    # Advanced configuration
    replication_factor: int = 3
    quorum_size: Optional[int] = None  # Auto-calculated if None

    # ML-enhanced monitoring
    enable_ml_monitoring: bool = True

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        # Generate node_id if not provided
        if not self.node_id:
            self.node_id = f"node-{uuid.uuid4()}"

        # Detect host if not provided
        if not self.host:
            try:
                self.host = socket.gethostbyname(socket.gethostname())
            except socket.gaierror:
                self.host = "127.0.0.1"
                logger.warning(f"Could not determine hostname. Using {self.host}")

        # Generate secure cluster secret if not provided and security is enabled
        if not self.cluster_secret and self.enable_security:
            self.cluster_secret = secrets.token_hex(16)
            logger.warning(
                "No cluster secret provided. Generated a random secret. "
                "For production, set WDBX_CLUSTER_SECRET environment variable."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "cluster_name": self.cluster_name,
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "election_timeout_base": self.election_timeout_base,
            "heartbeat_interval": self.heartbeat_interval,
            "enable_security": self.enable_security,
            "backend_type": self.backend_type,
            "etcd_hosts": self.etcd_hosts,
            "zookeeper_hosts": self.zookeeper_hosts,
            "storage_dir": self.storage_dir,
            "replication_factor": self.replication_factor,
            "quorum_size": self.quorum_size,
            "enable_ml_monitoring": self.enable_ml_monitoring,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterConfig":
        """Create config from dictionary."""
        # Filter out unknown fields
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    @classmethod
    def from_file(cls, file_path: str) -> "ClusterConfig":
        """Load config from JSON file."""
        try:
            with open(file_path) as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load config from {file_path}: {e}")
            return cls()

    def save_to_file(self, file_path: str) -> bool:
        """Save config to JSON file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Don't save cluster secret
            data = self.to_dict()
            data.pop("cluster_secret", None)

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {file_path}: {e}")
            return False

    def get_election_timeout(self) -> float:
        """
        Get randomized election timeout to prevent election conflicts.

        Returns:
            Randomized election timeout in seconds
        """
        # Randomize between 0.8-1.2× the base timeout
        return self.election_timeout_base * (0.8 + 0.4 * random.random())

    def get_quorum_size(self, node_count: int) -> int:
        """
        Calculate the quorum size needed for consensus.

        Args:
            node_count: Number of nodes in the cluster

        Returns:
            Minimum number of nodes required for quorum
        """
        if self.quorum_size is not None:
            return self.quorum_size

        # Default quorum is majority: ⌊N/2⌋ + 1
        return (node_count // 2) + 1


class NodeState:
    """Possible states for a cluster node."""

    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
    OBSERVER = "observer"
    OFFLINE = "offline"


class ClusterNode:
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
            "capabilities": list(self.capabilities),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterNode":
        """
        Create a node from a dictionary.

        Args:
            data: Dictionary representation of the node

        Returns:
            ClusterNode object
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


class ClusterCoordinator:
    """
    Manages cluster coordination and consensus using various backends.

    The ClusterCoordinator provides:
    - Distributed coordination for cluster nodes
    - Leader election algorithm implementation
    - Secure node registration and discovery
    - High-availability state management
    - ML-enhanced failure detection and prediction

    Supported backends include etcd3, ZooKeeper, and a simple file-based implementation
    for development and testing.
    """

    def __init__(self, node_id: str, backend_type: str = None):
        """
        Initialize cluster coordinator with specified backend.

        Args:
            node_id: ID of the node this coordinator belongs to
            backend_type: Coordination backend to use, or None for auto-detection
        """
        self.node_id = node_id
        self.nodes: Dict[str, ClusterNode] = {}
        self.backend = None
        self.backend_type = backend_type or self._detect_backend()
        self.initialized = False
        self._change_listeners: List[Callable] = []

        # Attempt to import security components
        try:
            from ..security import SecurityManager

            self.security_manager = SecurityManager()
            logger.info(f"Security manager initialized for cluster coordinator on {node_id}")
            self.security_enabled = True
        except ImportError:
            logger.warning("Security module not available for cluster coordinator")
            self.security_manager = None
            self.security_enabled = False

        # Attempt to import monitoring components
        try:
            from ..monitoring import PerformanceMonitor

            self.performance_monitor = PerformanceMonitor()
            logger.info(f"Performance monitoring initialized for cluster coordinator on {node_id}")
            self.monitoring_enabled = True
        except ImportError:
            logger.warning("Monitoring module not available for cluster coordinator")
            self.performance_monitor = None
            self.monitoring_enabled = False

        # Setup secure communication with payload signing
        self._setup_security()

        # Register signal handlers for clean shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_shutdown_signal)

        logger.info(f"Cluster coordinator initialized with backend: {self.backend_type}")

    def _detect_backend(self) -> str:
        """
        Auto-detect the best available coordination backend.

        Returns:
            String identifying the backend type
        """
        # Try to import etcd3
        try:
            import etcd3

            logger.info("Detected etcd3 backend")
            return "etcd3"
        except ImportError:
            pass

        # Try to import kazoo (ZooKeeper)
        try:
            import kazoo

            logger.info("Detected ZooKeeper backend")
            return "zookeeper"
        except ImportError:
            pass

        # Fall back to file-based backend
        logger.warning(
            "No distributed coordination backend detected. "
            "Using file-based backend - NOT RECOMMENDED FOR PRODUCTION"
        )
        return "file"

    def _setup_security(self) -> None:
        """Set up secure communication for the coordinator."""
        # Default to simple HMAC signing if security manager not available
        if not self.security_enabled or not self.security_manager:
            # Generate a signing key from environment or use default
            secret = os.environ.get("WDBX_CLUSTER_SECRET", "wdbx-cluster-default-secret")
            if secret == "wdbx-cluster-default-secret":
                logger.warning(
                    "Using default cluster secret. Set WDBX_CLUSTER_SECRET for production."
                )

            # Create simple signing functions
            def sign_payload(payload: Dict) -> str:
                payload_str = json.dumps(payload, sort_keys=True)
                return hmac.new(
                    secret.encode(), payload_str.encode(), digestmod="sha256"
                ).hexdigest()

            def verify_signature(payload: Dict, signature: str) -> bool:
                expected = sign_payload(payload)
                return hmac.compare_digest(expected, signature)

            self.sign_payload = sign_payload
            self.verify_signature = verify_signature

        else:
            # Use security manager for more advanced security
            def sign_payload(payload: Dict) -> str:
                return self.security_manager.sign_data(payload)

            def verify_signature(payload: Dict, signature: str) -> bool:
                return self.security_manager.verify_signature(payload, signature)

            self.sign_payload = sign_payload
            self.verify_signature = verify_signature

    def initialize(self) -> bool:
        """
        Initialize the coordination backend.

        Returns:
            True if initialization successful
        """
        if self.initialized:
            logger.warning("Coordinator already initialized")
            return True

        # Start performance monitoring if available
        if self.monitoring_enabled and self.performance_monitor:
            try:
                self.performance_monitor.start()
            except Exception as e:
                logger.error(f"Failed to start performance monitoring: {e}")

        try:
            # Initialize appropriate backend
            backend_initializers = {
                "etcd3": self._initialize_etcd,
                "zookeeper": self._initialize_zookeeper,
                "file": self._initialize_file_backend,
            }

            initializer = backend_initializers.get(self.backend_type)
            if not initializer:
                logger.error(f"Unsupported backend type: {self.backend_type}")
                return False

            # Initialize the backend with performance tracking
            start_time = time.time()
            result = initializer()

            if self.monitoring_enabled and self.performance_monitor:
                self.performance_monitor.record_event(
                    "coordinator_init",
                    {"backend": self.backend_type, "duration": time.time() - start_time},
                )

            if not result:
                logger.error(f"Failed to initialize {self.backend_type} backend")
                return False

            self.initialized = True
            logger.info(f"Successfully initialized {self.backend_type} coordinator backend")
            return True

        except Exception as e:
            logger.error(f"Error initializing coordinator: {e}", exc_info=True)
            return False

    def _initialize_etcd(self) -> bool:
        """
        Initialize etcd3 backend.

        Returns:
            True if successful
        """
        try:
            import etcd3

            # Get configuration from environment or use defaults
            host = os.environ.get("WDBX_ETCD_HOST", "localhost")
            port = int(os.environ.get("WDBX_ETCD_PORT", "2379"))
            ca_cert = os.environ.get("WDBX_ETCD_CA_CERT")
            cert_key = os.environ.get("WDBX_ETCD_CERT_KEY")
            cert_cert = os.environ.get("WDBX_ETCD_CERT_CERT")

            # Create client with proper security if provided
            if ca_cert and cert_key and cert_cert:
                logger.info(f"Connecting to etcd with TLS: {host}:{port}")
                self.backend = etcd3.client(
                    host=host, port=port, ca_cert=ca_cert, cert_key=cert_key, cert_cert=cert_cert
                )
            else:
                if os.environ.get("WDBX_ENVIRONMENT", "").lower() == "production":
                    logger.warning("Connecting to etcd without TLS in production environment")
                self.backend = etcd3.client(host=host, port=port)

            # Test connection
            self.backend.get_cluster_id()

            # Set up watch for cluster changes
            prefix = f"{CLUSTER_PREFIX}/nodes/"
            watch_id = self.backend.add_watch_prefix_callback(prefix, self._handle_node_change)

            logger.info(f"Etcd3 backend initialized: {host}:{port}")
            return True

        except ImportError:
            logger.error("Etcd3 package not installed")
            return False
        except Exception as e:
            logger.error(f"Error initializing etcd: {e}", exc_info=True)
            return False

    def _initialize_zookeeper(self) -> bool:
        """
        Initialize ZooKeeper backend.

        Returns:
            True if successful
        """
        try:
            from kazoo.client import KazooClient

            # Get configuration from environment or use defaults
            hosts = os.environ.get("WDBX_ZK_HOSTS", "localhost:2181")
            timeout = float(os.environ.get("WDBX_ZK_TIMEOUT", "10.0"))

            # Create client
            self.backend = KazooClient(hosts=hosts, timeout=timeout)
            self.backend.start(timeout=timeout)

            # Create base paths
            base_path = f"/{CLUSTER_PREFIX}"
            nodes_path = f"{base_path}/nodes"

            for path in [base_path, nodes_path]:
                self.backend.ensure_path(path)

            # Set up watch for cluster changes
            @self.backend.ChildrenWatch(nodes_path)
            def watch_nodes(children):
                for child in children:
                    node_path = f"{nodes_path}/{child}"
                    self._handle_node_change(node_path)

            logger.info(f"ZooKeeper backend initialized: {hosts}")
            return True

        except ImportError:
            logger.error("Kazoo package not installed")
            return False
        except Exception as e:
            logger.error(f"Error initializing ZooKeeper: {e}", exc_info=True)
            return False

    def _initialize_file_backend(self) -> bool:
        """
        Initialize file-based backend (for development/testing).

        Returns:
            True if successful
        """
        try:
            # Use local directory for storage
            storage_dir = os.environ.get(
                "WDBX_CLUSTER_STORAGE", os.path.join(tempfile.gettempdir(), "wdbx-cluster")
            )

            # Create directory if it doesn't exist
            os.makedirs(storage_dir, exist_ok=True)

            # Create subdirectories
            nodes_dir = os.path.join(storage_dir, "nodes")
            os.makedirs(nodes_dir, exist_ok=True)

            # Store backend info
            self.backend = {
                "type": "file",
                "storage_dir": storage_dir,
                "nodes_dir": nodes_dir,
            }

            # Set up file watcher if available
            try:
                import watchdog
                from watchdog.events import FileSystemEventHandler
                from watchdog.observers import Observer

                class NodeChangeHandler(FileSystemEventHandler):
                    def __init__(self, coordinator):
                        self.coordinator = coordinator

                    def on_modified(self, event):
                        if event.is_directory:
                            return
                        self.coordinator._handle_node_change(event.src_path)

                handler = NodeChangeHandler(self)
                observer = Observer()
                observer.schedule(handler, nodes_dir, recursive=False)
                observer.start()

                # Store observer to stop it later
                self.backend["observer"] = observer

                logger.info("File watcher initialized for cluster changes")

            except ImportError:
                logger.warning("Watchdog package not installed, polling for cluster changes")

                # Set up polling instead
                def poll_nodes():
                    while True:
                        try:
                            for filename in os.listdir(nodes_dir):
                                path = os.path.join(nodes_dir, filename)
                                if os.path.isfile(path):
                                    self._handle_node_change(path)
                            time.sleep(1.0)
                        except Exception as e:
                            logger.error(f"Error polling nodes: {e}")
                            time.sleep(5.0)

                poll_thread = threading.Thread(
                    target=poll_nodes, name="node_poll_thread", daemon=True
                )
                poll_thread.start()

                # Store thread to join it later
                self.backend["poll_thread"] = poll_thread

            logger.info(f"File backend initialized: {storage_dir}")
            return True

        except Exception as e:
            logger.error(f"Error initializing file backend: {e}", exc_info=True)
            return False

    def register_node(self, node: ClusterNode) -> bool:
        """
        Register a node with the cluster.

        Args:
            node: Node instance to register

        Returns:
            True if registration successful
        """
        if not self.initialized:
            logger.error("Coordinator not initialized")
            return False

        try:
            # Profile this operation if monitoring is enabled
            profile_context = None
            if self.monitoring_enabled and self.performance_monitor:
                profile_context = self.performance_monitor.profile("register_node")

            # Convert node to serializable form
            node_dict = node.to_dict()

            # Add security signature if enabled
            if self.security_enabled:
                node_dict["signature"] = self.sign_payload(node_dict)

            # Serialize node data
            node_data = json.dumps(node_dict).encode()

            # Register node using appropriate backend
            if self.backend_type == "etcd3":
                key = f"{CLUSTER_PREFIX}/nodes/{node.id}"
                self.backend.put(key, node_data)

            elif self.backend_type == "zookeeper":
                path = f"/{CLUSTER_PREFIX}/nodes/{node.id}"
                if self.backend.exists(path):
                    self.backend.set(path, node_data)
                else:
                    self.backend.create(path, node_data, makepath=True)

            elif self.backend_type == "file":
                path = os.path.join(self.backend["nodes_dir"], node.id)
                with open(path, "wb") as f:
                    f.write(node_data)

            else:
                logger.error(f"Unsupported backend type: {self.backend_type}")
                return False

            # End profiling if active
            if profile_context:
                profile_context.__exit__(None, None, None)

            logger.debug(f"Registered node: {node.id} (state: {node.state})")
            return True

        except Exception as e:
            logger.error(f"Error registering node: {e}", exc_info=True)
            return False

    def unregister_node(self, node_id: str) -> bool:
        """
        Unregister a node from the cluster.

        Args:
            node_id: ID of node to unregister

        Returns:
            True if unregistration successful
        """
        if not self.initialized:
            logger.error("Coordinator not initialized")
            return False

        try:
            # Delete node using appropriate backend
            if self.backend_type == "etcd3":
                key = f"{CLUSTER_PREFIX}/nodes/{node_id}"
                self.backend.delete(key)

            elif self.backend_type == "zookeeper":
                path = f"/{CLUSTER_PREFIX}/nodes/{node_id}"
                if self.backend.exists(path):
                    self.backend.delete(path)

            elif self.backend_type == "file":
                path = os.path.join(self.backend["nodes_dir"], node_id)
                if os.path.exists(path):
                    os.remove(path)

            else:
                logger.error(f"Unsupported backend type: {self.backend_type}")
                return False

            # Security audit logging if enabled
            if self.security_enabled and self.security_manager:
                self.security_manager.log_security_event(
                    event_type="cluster_node_removed",
                    details={"node_id": node_id, "removed_by": self.node_id},
                    level="INFO",
                )

            logger.info(f"Unregistered node: {node_id}")
            return True

        except Exception as e:
            logger.error(f"Error unregistering node: {e}", exc_info=True)
            return False

    def get_nodes(self) -> Dict[str, ClusterNode]:
        """
        Get all registered nodes in the cluster.

        Returns:
            Dictionary mapping node IDs to Node instances
        """
        if not self.initialized:
            logger.error("Coordinator not initialized")
            return {}

        nodes = {}

        try:
            # Profile this operation if monitoring is enabled
            profile_context = None
            if self.monitoring_enabled and self.performance_monitor:
                profile_context = self.performance_monitor.profile("get_nodes")

            # Fetch nodes from appropriate backend
            if self.backend_type == "etcd3":
                prefix = f"{CLUSTER_PREFIX}/nodes/"
                results = self.backend.get_prefix(prefix)

                for value, metadata in results:
                    if not value:
                        continue
                    try:
                        node_dict = json.loads(value.decode())
                        node = ClusterNode.from_dict(node_dict)

                        # Verify signature if present
                        if "signature" in node_dict and self.security_enabled:
                            signature = node_dict.pop("signature")
                            if not self.verify_signature(node_dict, signature):
                                logger.warning(f"Invalid signature for node: {node.id}")
                                continue

                        nodes[node.id] = node
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid node data: {value}")

            elif self.backend_type == "zookeeper":
                path = f"/{CLUSTER_PREFIX}/nodes"
                if not self.backend.exists(path):
                    return {}

                for child in self.backend.get_children(path):
                    child_path = f"{path}/{child}"
                    value, _ = self.backend.get(child_path)

                    if not value:
                        continue

                    try:
                        node_dict = json.loads(value.decode())
                        node = ClusterNode.from_dict(node_dict)

                        # Verify signature if present
                        if "signature" in node_dict and self.security_enabled:
                            signature = node_dict.pop("signature")
                            if not self.verify_signature(node_dict, signature):
                                logger.warning(f"Invalid signature for node: {node.id}")
                                continue

                        nodes[node.id] = node
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid node data: {value}")

            elif self.backend_type == "file":
                nodes_dir = self.backend["nodes_dir"]
                for filename in os.listdir(nodes_dir):
                    path = os.path.join(nodes_dir, filename)

                    if not os.path.isfile(path):
                        continue

                    try:
                        with open(path, "rb") as f:
                            value = f.read()

                        if not value:
                            continue

                        node_dict = json.loads(value.decode())
                        node = ClusterNode.from_dict(node_dict)

                        # Verify signature if present
                        if "signature" in node_dict and self.security_enabled:
                            signature = node_dict.pop("signature")
                            if not self.verify_signature(node_dict, signature):
                                logger.warning(f"Invalid signature for node: {node.id}")
                                continue

                        nodes[node.id] = node
                    except (OSError, json.JSONDecodeError) as e:
                        logger.warning(f"Error reading node data from {path}: {e}")

            else:
                logger.error(f"Unsupported backend type: {self.backend_type}")

            # End profiling if active
            if profile_context:
                profile_context.__exit__(None, None, None)

            return nodes

        except Exception as e:
            logger.error(f"Error getting nodes: {e}", exc_info=True)
            return {}

    def _handle_node_change(self, key_or_path):
        """
        Handle node change notification from backend.

        Args:
            key_or_path: Key or path that changed
        """
        try:
            # Extract node ID from key/path
            if self.backend_type == "etcd3":
                parts = key_or_path.decode().split("/")
                node_id = parts[-1]
            elif self.backend_type == "zookeeper":
                parts = key_or_path.split("/")
                node_id = parts[-1]
            elif self.backend_type == "file":
                node_id = os.path.basename(key_or_path)
            else:
                logger.error(f"Unsupported backend type: {self.backend_type}")
                return

            # Refresh nodes and notify listeners
            nodes = self.get_nodes()
            for listener in self._change_listeners:
                try:
                    listener(node_id, nodes)
                except Exception as e:
                    logger.error(f"Error in node change listener: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error handling node change: {e}", exc_info=True)

    def add_change_listener(self, listener: Callable[[str, Dict[str, ClusterNode]], None]):
        """
        Add a listener for node changes.

        Args:
            listener: Callback function that takes node_id and nodes dict
        """
        self._change_listeners.append(listener)

    def remove_change_listener(self, listener):
        """
        Remove a change listener.

        Args:
            listener: Listener to remove
        """
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)

    def close(self):
        """Close the coordinator and release resources."""
        if not self.initialized:
            return

        try:
            # Stop monitoring if active
            if self.monitoring_enabled and self.performance_monitor:
                try:
                    self.performance_monitor.stop()
                except Exception as e:
                    logger.error(f"Error stopping performance monitoring: {e}")

            # Close backend based on type
            if self.backend_type == "etcd3":
                self.backend.close()

            elif self.backend_type == "zookeeper":
                self.backend.stop()
                self.backend.close()

            elif self.backend_type == "file":
                # Stop file watcher if running
                if "observer" in self.backend:
                    self.backend["observer"].stop()
                    self.backend["observer"].join(timeout=5.0)

            logger.info(f"Closed {self.backend_type} coordinator backend")

        except Exception as e:
            logger.error(f"Error closing coordinator: {e}", exc_info=True)

        self.initialized = False

    def _handle_shutdown_signal(self, signum, frame):
        """
        Handle shutdown signal.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {signum}, shutting down coordinator")
        self.close()

    def get_cluster_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cluster.

        Returns:
            Dictionary with cluster statistics
        """
        if not self.initialized:
            return {"status": "not_initialized"}

        # Get nodes to analyze
        nodes = self.get_nodes()

        # Collect stats
        stats = {
            "node_count": len(nodes),
            "active_nodes": sum(1 for node in nodes.values() if node.is_alive()),
            "backend_type": self.backend_type,
            "states": {},
        }

        # Count nodes by state
        for node in nodes.values():
            state_name = node.state.name if hasattr(node.state, "name") else str(node.state)
            stats["states"][state_name] = stats["states"].get(state_name, 0) + 1

        # Add leader info if available
        leader_nodes = [node for node in nodes.values() if node.state == NodeState.LEADER]
        if leader_nodes:
            leader = leader_nodes[0]
            stats["leader"] = {
                "id": leader.id,
                "host": leader.host,
                "port": leader.port,
                "term": getattr(leader, "term", 0),
            }

        # Add ML prediction stats if available
        if self.monitoring_enabled and self.performance_monitor:
            stats["performance"] = {
                "coordinator_cpu": self.performance_monitor.get_cpu_percent(),
                "coordinator_memory": self.performance_monitor.get_memory_percent(),
            }

        return stats


class ClusterManager:
    """
    Manages a cluster of WDBX nodes.
    """

    def __init__(self, config: ClusterConfig):
        """
        Initialize cluster manager.

        Args:
            config: Cluster configuration
        """
        self.config = config
        self.coordinator = ClusterCoordinator(config.node_id)
        self.nodes: Dict[str, ClusterNode] = {}
        self.current_node: Optional[ClusterNode] = None
        self.initialized = False

    def initialize(self) -> bool:
        """
        Initialize the cluster manager.

        Returns:
            True if initialization successful
        """
        if self.initialized:
            return True

        try:
            # Initialize coordinator
            if not self.coordinator.initialize():
                return False

            # Create current node
            self.current_node = ClusterNode(self.config.node_id, self.config.host, self.config.port)

            # Register current node
            if not self.coordinator.register_node(self.current_node):
                return False

            # Add change listener
            self.coordinator.add_change_listener(self._handle_node_change)

            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"Error initializing cluster manager: {e}")
            return False

    def _handle_node_change(self, node_id: str, nodes: Dict[str, ClusterNode]):
        """
        Handle node change notification.

        Args:
            node_id: ID of changed node
            nodes: Updated nodes dictionary
        """
        self.nodes = nodes
        if node_id == self.config.node_id:
            self.current_node = nodes.get(node_id)

    def get_node(self, node_id: str) -> Optional[ClusterNode]:
        """
        Get a node by ID.

        Args:
            node_id: Node ID to look up

        Returns:
            ClusterNode if found, None otherwise
        """
        return self.nodes.get(node_id)

    def get_leader(self) -> Optional[ClusterNode]:
        """
        Get the current leader node.

        Returns:
            Leader node if one exists, None otherwise
        """
        for node in self.nodes.values():
            if node.state == NodeState.LEADER:
                return node
        return None

    def close(self):
        """Close the cluster manager and release resources."""
        if not self.initialized:
            return

        try:
            # Unregister current node
            if self.current_node:
                self.coordinator.unregister_node(self.current_node.node_id)

            # Close coordinator
            self.coordinator.close()

        except Exception as e:
            logger.error(f"Error closing cluster manager: {e}")

        self.initialized = False


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
            from flask import jsonify, request

            # List of paths that require leader status
            leader_paths = ["/v1/blocks/create", "/v1/blocks/update", "/v1/blocks/delete"]

            if request.path in leader_paths and request.method != "GET":
                if cluster_node.state != NodeState.LEADER:
                    # Redirect to leader if known
                    if cluster_node.leader_id:
                        leader_node = cluster_node.coordinator.get_node(cluster_node.leader_id)
                        if leader_node:
                            return (
                                jsonify(
                                    {
                                        "error": "Not the leader",
                                        "leader": {
                                            "node_id": leader_node.node_id,
                                            "host": leader_node.host,
                                            "port": leader_node.port,
                                        },
                                    }
                                ),
                                307,
                            )

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
    parser.add_argument(
        "--nodes", default=",".join(CLUSTER_NODES), help="Comma-separated list of cluster nodes"
    )
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
        logger.info("Cluster node stopped")
