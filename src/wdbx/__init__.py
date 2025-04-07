"""
WDBX - High-performance vector database with blockchain-inspired features.

This package provides a vector database with distributed query planning,
multi-version concurrency control, and blockchain-based storage for data provenance.
"""

# Version information
__version__ = "0.1.0"

# Import core constants
from .core.constants import DEFAULT_SIMILARITY_THRESHOLD, SHARD_COUNT, VECTOR_DIMENSION

# Import logging utilities
from .utils.logging_utils import configure_logging, get_logger

# Initialize default logger
logger = get_logger("WDBX")

# Import commonly used components for easy access
from .core import WDBX, WDBXConfig, create_wdbx
from .core.data_structures import Block, EmbeddingVector
from .storage.blockchain import BlockChainManager
from .storage.mvcc import MVCCManager, MVCCTransaction
from .storage.vector_store import VectorStore

# Public exports
__all__ = [
    # Core Classes
    "WDBX",
    "WDBXConfig",
    "create_wdbx",
    "EmbeddingVector",
    "Block",
    "VectorStore",
    "BlockChainManager",
    "MVCCTransaction",
    "MVCCManager",
    # Logging & Constants
    "configure_logging",
    "get_logger",
    "VECTOR_DIMENSION",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "SHARD_COUNT",
]
