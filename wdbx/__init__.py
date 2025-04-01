# wdbx/__init__.py
"""
WDBX: Wide Distributed Block Database

A high-performance, responsive data store designed for multi-personality AI systems.
This system integrates vector similarity search, blockchain-inspired integrity,
multiversion concurrency control, and multi-head attention mechanisms to ensure
AI applications operate efficiently and reliably.

Features:
- High-performance vector embedding storage and similarity search
- Blockchain-inspired data integrity and auditability
- Multiversion concurrency control for atomic transactions
- Distributed sharding for horizontal scaling
- Neural backtracking for explainable AI decisions
- Multi-persona management for complex AI systems
"""

# Core imports
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable

# Internal module imports
from wdbx.attention import MultiHeadAttention
from wdbx.blockchain import BlockChainManager
from wdbx.cli import main as cli_main, run_example, interactive_mode, run_server, batch_process
from wdbx.cluster import (
    NodeState, Node, ClusterChange, CoordinationBackend, EtcdBackend,
    FileBackend, ClusterCoordinator, ReplicationManager, ClusterNode,
    init_cluster
)
from wdbx.constants import (
    VECTOR_DIMENSION, SHARD_COUNT, NETWORK_OVERHEAD, AES_KEY_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD, MVCC_WRITE_LOCK_TIMEOUT, MAX_BATCH_SIZE,
    BLOCKCHAIN_DIFFICULTY, MAX_RETRIES, READ_TIMEOUT, WRITE_TIMEOUT,
    DEFAULT_DATA_DIR, CACHE_DIR, LOG_DIR, TEMP_DIR, HTTP_HOST, HTTP_PORT,
    HTTP_WORKERS, HTTP_TIMEOUT, HTTP_MAX_REQUEST_SIZE, SOCKET_HOST,
    SOCKET_PORT, SOCKET_BACKLOG, SOCKET_TIMEOUT, LOG_LEVEL, LOG_FORMAT,
    LOG_FILE, LOG_MAX_SIZE, LOG_BACKUP_COUNT, DEFAULT_PERSONAS,
    PERSONA_CONFIGS, USE_JIT, USE_COMPRESSION, USE_MMAP, ENABLE_CACHING,
    CACHE_TTL, PARALLELISM, AUTH_REQUIRED, JWT_SECRET, JWT_EXPIRATION,
    ENCRYPTION_ENABLED, SSL_CERT_FILE, SSL_KEY_FILE, logger
)
from wdbx.content_filter import ContentFilter, BiasDetector
from wdbx.data_structures import EmbeddingVector, Block, ShardInfo
from wdbx.monitoring import (
    METRICS_PORT, METRICS_ENABLED, TRACING_ENABLED, LOG_SAMPLING_RATE,
    HEALTH_CHECK_INTERVAL, MetricsRegistry, PrometheusCounter, PrometheusGauge,
    PrometheusHistogram, PrometheusSummary, PrometheusInfo, SimpleCounter,
    SimpleGauge, SimpleHistogram, SimpleSummary, SimpleInfo, TracingManager,
    DummySpan, HealthChecker, LogCollector, MonitoringSystem, OperationTracker,
    init_monitoring
)
from wdbx.mvcc import MVCCTransaction, MVCCManager
from wdbx.neural_backtracking import NeuralBacktracker
from wdbx.optimized import OptimizedVectorStore, OptimizedBlockManager, OptimizedTransactionManager
from wdbx.performance import PerformanceAnalyzer
from wdbx.persona import PersonaTokenManager, PersonaManager
from wdbx.security import (
    SECRET_KEY, TOKEN_EXPIRATION, RATE_LIMIT_WINDOW, RATE_LIMIT_MAX_REQUESTS,
    DEFAULT_ROLES, AUDIT_LOG_FILE, ApiKey, RateLimiter, SecurityManager,
    require_auth, init_security
)
from wdbx.server import run_server as start_server
from wdbx.shard_manager import ShardManager
from wdbx.terminal_ui import WDBXMonitor, WDBXTerminalUI, run_terminal_ui, run_simple_dashboard
from wdbx.vector_store import PythonVectorIndex, VectorStore, VectorOperations
from wdbx.web_ui import create_web_app, run_web_ui

# Set up the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("WDBX")

# Version information
__version__ = "1.0.1"
__author__ = "Donald Filimon"

# Check for optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not found. Using built-in array implementation which may be slower.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not found. Using built-in vector search which may be slower.")

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not found. HTTP server functionality will not be available.")

try:
    from sklearn import cluster
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not found. Using built-in clustering implementation.")

# Try importing OpenTelemetry for tracing
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning("OpenTelemetry not found. Advanced tracing will not be available.")

# Import core components from submodules
from wdbx.attention import *
from wdbx.cli import *
from wdbx.cluster import *
from wdbx.data_structures import *
from wdbx.vector_store import *
from wdbx.blockchain import *
from wdbx.mvcc import *
from wdbx.neural_backtracking import *
from wdbx.shard_manager import *
from wdbx.persona import *
from wdbx.content_filter import *


# The main WDBX class
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

    Attributes:
        vector_dimension (int): Dimensionality of embedding vectors
        num_shards (int): Number of shards for distributed storage
        vector_store (VectorStore): Storage for embedding vectors
        shard_manager (ShardManager): Manager for distributed shards
        block_chain_manager (BlockChainManager): Manager for blockchain functionality
        mvcc_manager (MVCCManager): Manager for multiversion concurrency control
        neural_backtracker (NeuralBacktracker): Component for neural backtracking
        persona_manager (PersonaManager): Manager for multiple personas
        content_filter (ContentFilter): Filter for content moderation
        attention (MultiHeadAttention): Multi-head attention mechanism
        stats (Dict): Statistics about system usage
    """
    def __init__(self,
                 vector_dimension: int = 1024,
                 num_shards: int = 8,
                 enable_persona_management: bool = True,
                 content_filter_level: str = 'medium') -> None:
        self.vector_dimension = vector_dimension
        self.num_shards = num_shards
        self.vector_store = VectorStore(dimension=vector_dimension)
        self.shard_manager = ShardManager(num_shards=num_shards)
        self.block_chain_manager = BlockChainManager()
        self.mvcc_manager = MVCCManager()
        self.neural_backtracker = NeuralBacktracker(
            block_chain_manager=self.block_chain_manager,
            vector_store=self.vector_store
        )

        # Initialize additional components
        if enable_persona_management:
            self.persona_manager = PersonaManager(self)
            self.persona_token_manager = PersonaTokenManager(self.persona_manager.persona_embeddings)
        else:
            self.persona_manager = None
            self.persona_token_manager = None

        from wdbx.content_filter import ContentFilter, BiasDetector
        self.content_filter = ContentFilter(
            sensitive_topics=["illegal activities", "dangerous substances", "explicit content"],
            offensive_patterns=["offensive language", "hate speech", "discriminatory language"]
        )
        self.bias_detector = BiasDetector(
            bias_attributes=["gender", "race", "age", "religion", "nationality", "disability", "sexual orientation"]
        )
        self.attention = MultiHeadAttention(num_heads=8, d_model=vector_dimension)

        # System metrics
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

        logger.info(f"WDBX initialized with {vector_dimension}-dimension vectors across {num_shards} shards")

    def is_connected(self) -> bool:
        """
        Check if the system is ready and connected

        Returns:
            bool: True if the system is operational, False otherwise
        """
        # shard_status = self.shard_manager.check_all_shards()
        # store_status = self.vector_store.is_ready()
        # return all([shard_status, store_status])
        return True # Place holder until check_all_shards and is_ready are implemented

    def store_embedding(self, embedding_vector: EmbeddingVector) -> str:
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
            # if self.content_filter and self.content_filter.should_filter_vector(embedding_vector):
            #     logger.warning(f"Content filter blocked vector {vector_id}")
            #     self.stats["content_filter_blocks"] += 1
            #     self.mvcc_manager.abort(transaction.transaction_id)
            #     raise ValueError(f"Vector content blocked by filter")

            embedding_vector.metadata["vector_id"] = vector_id
            embedding_vector.metadata["timestamp"] = time.time()

            # Store the vector
            if not self.vector_store.add(vector_id, embedding_vector):
                self.mvcc_manager.abort(transaction.transaction_id)
                self.stats["failed_transactions"] += 1
                raise ValueError(f"Failed to store vector {vector_id}")

            self.mvcc_manager.commit(transaction.transaction_id)
            self.stats["vectors_stored"] += 1
            self.stats["transactions_processed"] += 1
            return vector_id
        except Exception as e:
            self.mvcc_manager.abort(transaction.transaction_id)
            self.stats["failed_transactions"] += 1
            raise e

    def create_conversation_block(self, data: Dict[str, Any], embeddings: List[EmbeddingVector],
                                 chain_id: Optional[str] = None,
                                 context_references: Optional[List[str]] = None,
                                 persona_id: Optional[str] = None) -> str:
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
            # if self.content_filter and self.content_filter.should_filter_content(data):
            #     logger.warning("Content filter blocked block creation")
            #     self.stats["content_filter_blocks"] += 1
            #     self.mvcc_manager.abort(transaction.transaction_id)
            #     raise ValueError("Block content blocked by filter")

            # Store embeddings first
            vector_ids = []
            for embedding in embeddings:
                vector_id = self.store_embedding(embedding)
                embedding.metadata.setdefault("vector_ids", []).append(vector_id)
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
            block = self.block_chain_manager.create_block(
                data=data,
                embeddings=embeddings,
                chain_id=chain_id,
                context_references=context_references or []
            )

            # Assign to shard
            shard = self.shard_manager.get_shard_for_block(block.id)
            # Fix: Access proper shard attribute instead of shard_id on the int type
            data["_meta"]["shard_id"] = shard if shard else None

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

    def search_similar_vectors(self, query_vector: Any, top_k: int = 10,
                              threshold: float = 0.0) -> List[Tuple[str, float]]:
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

    def create_neural_trace(self, query_vector: Any, trace_depth: int = 3) -> str:
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
            top_k=20
        )
        self.stats["traces_created"] += 1
        return trace_id

    def get_conversation_context(self, block_ids: List[str],
                                include_embeddings: bool = True,
                                max_context_blocks: int = 10) -> Dict[str, Any]:
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
            block = self.block_chain_manager.get_block(block_id)
            if block:
                blocks.append(block)
                if include_embeddings:
                    embeddings.extend(block.embeddings)

                # Find which chains this block belongs to
                for cid, head in self.block_chain_manager.chain_heads.items():
                    chain_blocks = self.block_chain_manager.get_chain(cid)
                    if any(b.id == block_id for b in chain_blocks):
                        chains.add(cid)
                        break

        # Get context blocks with limit
        context_blocks = []
        for block in blocks:
            context_blocks.extend(
                self.block_chain_manager.get_context_blocks(block.id)
            )

        # Limit context blocks to max_context_blocks
        if len(context_blocks) > max_context_blocks:
            context_blocks = context_blocks[:max_context_blocks]

        # Create aggregated embedding if needed
        aggregated_embedding = None
        if embeddings and include_embeddings:
            vectors = [e.vector for e in embeddings]
            aggregated_embedding = VectorOperations.average_vectors(vectors)

            # Apply attention mechanism if available
            if hasattr(self, 'attention'):
                #aggregated_embedding = self.attention.apply(
                #    vectors,
                #    query=vectors[-1] if vectors else None
                #)
                pass # Place holder until attention.apply is implemented

        return {
            "blocks": blocks,
            "context_blocks": context_blocks,
            "chains": list(chains),
            "aggregated_embedding": aggregated_embedding,
            "block_count": len(blocks),
            "context_block_count": len(context_blocks),
            "chain_count": len(chains)
        }

    def get_system_stats(self) -> Dict[str, Any]:
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
        vector_store_stats = {} #self.vector_store.get_stats() if hasattr(self.vector_store, 'get_stats') else {}
        blockchain_stats = {} #self.block_chain_manager.get_stats() if hasattr(self.block_chain_manager, 'get_stats') else {}
        shard_stats = {} #self.shard_manager.get_stats() if hasattr(self.shard_manager, 'get_stats') else {}

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
            "available_components": {
                "numpy": NUMPY_AVAILABLE,
                "faiss": FAISS_AVAILABLE,
                "aiohttp": AIOHTTP_AVAILABLE,
                "sklearn": SKLEARN_AVAILABLE,
                "persona_management": self.persona_manager is not None
            }
        }

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format"""
        days, remainder = divmod(int(seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{days}d {hours}h {minutes}m {seconds}s"

    def _get_memory_usage(self) -> float:
        """Get approximate memory usage in MB"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return -1  # Couldn't determine

    def get_block(self, block_id: str) -> Optional[Block]:
        """
        Get a specific block by ID.

        Args:
            block_id (str): ID of the block to retrieve

        Returns:
            Optional[Block]: The block if found, None otherwise
        """
        return self.block_chain_manager.get_block(block_id)

    def get_chain(self, chain_id: str) -> List[Block]:
        """
        Get all blocks in a chain.

        Args:
            chain_id (str): ID of the chain to retrieve

        Returns:
            List[Block]: All blocks in the chain in order
        """
        return self.block_chain_manager.get_chain(chain_id)

    def close(self) -> None:
        """
        Close any open resources and perform cleanup operations.

        This method should be called when shutting down the system to ensure
        proper cleanup of resources.
        """
        logger.info("Shutting down WDBX...")

        # Close vector store connections
        #if hasattr(self.vector_store, 'close'):
        #    self.vector_store.close()

        # Close shard connections
        #if hasattr(self.shard_manager, 'close'):
        #    self.shard_manager.close()

        # Finalize any pending transactions
        #if hasattr(self.mvcc_manager, 'close'):
        #    self.mvcc_manager.close()

        # Final log message
        logger.info(f"WDBX shutdown complete. Stats: {self.stats['blocks_created']} blocks, "
                   f"{self.stats['vectors_stored']} vectors stored")

# Factory method for creating a new WDBX instance
def create_wdbx(vector_dimension: int = 1024, num_shards: int = 8, **kwargs) -> WDBX:
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

# Make commonly used classes available at module level
__all__ = [
    'WDBX',
    'create_wdbx',
    'EmbeddingVector',
    'Block',
    'VectorStore',
    'VectorOperations',
    'BlockChainManager',
    'MVCCManager',
    'MVCCTransaction',
    'NeuralBacktracker',
    'ShardManager',
    'PersonaManager',
    'PersonaTokenManager',
    'ContentFilter',
    'BiasDetector',
    'MultiHeadAttention',
    # 'DatabaseClient',
    # 'HttpDatabaseClient',
    # 'SocketDatabaseClient',
    # 'FilesystemDatabaseClient'
]
