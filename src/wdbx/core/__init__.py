"""
Core module of WDBX.

This module contains the main components and functionality of the WDBX system.
"""

# Re-export main classes for backward compatibility with tests
from ..ml.attention import MultiHeadAttention
from ..ml.neural_backtracking import NeuralBacktracker
from ..security.content_filter import BiasDetector, ContentFilter
from ..security.persona import PersonaManager, PersonaTokenManager
from ..storage.blockchain import BlockChainManager
from ..storage.mvcc import MVCCManager
from ..storage.shard_manager import ShardManager

# Re-export components from their new locations
from ..storage.vector_store import VectorStore
from .config import WDBXConfig
from .constants import (
    DEFAULT_BIAS_SCORE,
    DEFAULT_DATA_DIR,
    DEFAULT_SIMILARITY_THRESHOLD,
    SHARD_COUNT,
    VECTOR_DIMENSION,
    logger,
)
from .data_structures import Block, EmbeddingVector
from .wdbx import WDBX, DistributedQueryPlanner, create_wdbx

__all__ = [
    # Core classes
    "WDBX",
    "DistributedQueryPlanner",
    "create_wdbx",
    # Configuration
    "WDBXConfig",
    # Data structures
    "EmbeddingVector",
    "Block",
    # Constants
    "VECTOR_DIMENSION",
    "SHARD_COUNT",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_DATA_DIR",
    "DEFAULT_BIAS_SCORE",
    "logger",
    # New components
    "VectorStore",
    "ShardManager",
    "BlockChainManager",
    "MVCCManager",
    "NeuralBacktracker",
    "MultiHeadAttention",
    "PersonaManager",
    "PersonaTokenManager",
    "ContentFilter",
    "BiasDetector",
]
