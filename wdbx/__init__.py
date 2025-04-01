# wdbx/__init__.py
"""
WDBX: Wide Distributed Block Database

A super-speedy, ultra-responsive data store made for multi-personality AI systems.
It mixes vector similarity search, blockchain-style integrity, multiversion concurrency control, 
and multi-head attention to keep AI applications running smooth and strong.
"""
import logging
from typing import Dict, List, Optional, Union, Any

# Set up the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("WDBX")

# Version information
__version__ = "1.0.0"
__author__ = "Donald Filimon"

# Try importing optional dependencies
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

# Import core components from submodules
from wdbx.data_structures import EmbeddingVector, Block, ShardInfo
from wdbx.vector_store import VectorStore, VectorOperations
from wdbx.blockchain import BlockChainManager
from wdbx.mvcc import MVCCManager, MVCCTransaction
from wdbx.neural_backtracking import NeuralBacktracker
from wdbx.shard_manager import ShardManager
from wdbx.persona import PersonaManager, PersonaTokenManager
from wdbx.content_filter import ContentFilter, BiasDetector
from wdbx.attention import MultiHeadAttention

# Import database-related components
from wdbx.database import DatabaseClient, HttpDatabaseClient, SocketDatabaseClient, FilesystemDatabaseClient

# The main WDBX class
class WDBX:
    """
    Core implementation of the Wide Distributed Block Exchange (WDBX) system.
    
    This class brings together all the components of the WDBX system, providing a simple
    interface for storing embedding vectors, creating conversation blocks with blockchain
    integrity, and managing multi-persona AI systems.
    
    Args:
        vector_dimension (int): Dimensionality of embedding vectors
        num_shards (int): Number of shards for distributed storage
    
    Attributes:
        vector_dimension (int): Dimensionality of embedding vectors
        num_shards (int): Number of shards for distributed storage
        vector_store (VectorStore): Storage for embedding vectors
        shard_manager (ShardManager): Manager for distributed shards
        block_chain_manager (BlockChainManager): Manager for blockchain functionality
        mvcc_manager (MVCCManager): Manager for multiversion concurrency control
        neural_backtracker (NeuralBacktracker): Component for neural backtracking
        stats (Dict): Statistics about system usage
    """
    def __init__(self, vector_dimension: int = 1024, num_shards: int = 8) -> None:
        import time
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
        self.stats = {
            "blocks_created": 0,
            "vectors_stored": 0,
            "transactions_processed": 0,
            "traces_created": 0,
            "start_time": time.time()
        }

    def is_connected(self) -> bool:
        """Check if the system is ready and connected"""
        return True

    def store_embedding(self, embedding_vector: EmbeddingVector) -> str:
        """
        Store an embedding vector in the system.
        
        Args:
            embedding_vector (EmbeddingVector): The embedding vector to store
            
        Returns:
            str: ID of the stored vector
            
        Raises:
            ValueError: If the vector could not be stored
        """
        transaction = self.mvcc_manager.start_transaction()
        import uuid
        vector_id = str(uuid.uuid4())
        try:
            embedding_vector.metadata["vector_id"] = vector_id
            if not self.vector_store.add(vector_id, embedding_vector):
                self.mvcc_manager.abort(transaction.transaction_id)
                raise ValueError(f"Failed to store vector {vector_id}")
            self.mvcc_manager.commit(transaction.transaction_id)
            self.stats["vectors_stored"] += 1
            self.stats["transactions_processed"] += 1
            return vector_id
        except Exception as e:
            self.mvcc_manager.abort(transaction.transaction_id)
            raise e

    def create_conversation_block(self, data: Dict[str, Any], embeddings: List[EmbeddingVector], 
                                 chain_id: Optional[str] = None, 
                                 context_references: Optional[List[str]] = None) -> str:
        """
        Create a block containing conversation data and embedding vectors.
        
        Args:
            data (Dict[str, Any]): Data to store in the block
            embeddings (List[EmbeddingVector]): Embedding vectors to include in the block
            chain_id (Optional[str]): ID of the chain to add this block to, or None for a new chain
            context_references (Optional[List[str]]): References to other blocks for context
            
        Returns:
            str: ID of the created block
            
        Raises:
            Exception: If block creation fails
        """
        transaction = self.mvcc_manager.start_transaction()
        try:
            for embedding in embeddings:
                vector_id = self.store_embedding(embedding)
                embedding.metadata.setdefault("vector_ids", []).append(vector_id)
            block = self.block_chain_manager.create_block(
                data=data,
                embeddings=embeddings,
                chain_id=chain_id,
                context_references=context_references or []
            )
            self.shard_manager.get_shard_for_block(block.id)
            self.mvcc_manager.commit(transaction.transaction_id)
            self.stats["blocks_created"] += 1
            return block.id
        except Exception as e:
            self.mvcc_manager.abort(transaction.transaction_id)
            raise e

    def search_similar_vectors(self, query_vector: Any, top_k: int = 10) -> List[tuple]:
        """
        Search for vectors similar to the query vector.
        
        Args:
            query_vector: The query vector to search for
            top_k (int): Maximum number of results to return
            
        Returns:
            List[tuple]: List of (vector_id, similarity_score) tuples
        """
        return self.vector_store.search_similar(query_vector, top_k=top_k)

    def create_neural_trace(self, query_vector: Any) -> str:
        """
        Create a neural trace for the query vector.
        
        A neural trace tracks activation patterns through the system,
        allowing for backtracking and understanding of how the system
        responds to specific inputs.
        
        Args:
            query_vector: The query vector to trace
            
        Returns:
            str: ID of the created trace
        """
        trace_id = self.neural_backtracker.trace_activation(query_vector)
        self.stats["traces_created"] += 1
        return trace_id

    def get_conversation_context(self, block_ids: List[str]) -> Dict[str, Any]:
        """
        Get the conversation context for the given block IDs.
        
        Args:
            block_ids (List[str]): IDs of blocks to get context for
            
        Returns:
            Dict[str, Any]: Context information including blocks, chains, and aggregated embedding
        """
        blocks = []
        embeddings = []
        chains = set()
        for block_id in block_ids:
            block = self.block_chain_manager.get_block(block_id)
            if block:
                blocks.append(block)
                embeddings.extend(block.embeddings)
                for cid, head in self.block_chain_manager.chain_heads.items():
                    chain_blocks = self.block_chain_manager.get_chain(cid)
                    if any(b.id == block_id for b in chain_blocks):
                        chains.add(cid)
                        break
        context_blocks = []
        for block in blocks:
            context_blocks.extend(self.block_chain_manager.get_context_blocks(block.id))
        aggregated_embedding = None
        if embeddings:
            vectors = [e.vector for e in embeddings]
            aggregated_embedding = VectorOperations.average_vectors(vectors)
        return {
            "blocks": blocks,
            "context_blocks": context_blocks,
            "chains": list(chains),
            "aggregated_embedding": aggregated_embedding
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get statistics about system usage.
        
        Returns:
            Dict[str, Any]: System statistics
        """
        import time
        uptime = time.time() - self.stats["start_time"]
        return {
            **self.stats,
            "uptime": uptime,
            "blocks_per_second": self.stats["blocks_created"] / uptime if uptime > 0 else 0,
            "vectors_per_second": self.stats["vectors_stored"] / uptime if uptime > 0 else 0,
            "shard_count": self.num_shards,
            "vector_dimension": self.vector_dimension
        }

    def close(self) -> None:
        """Close any open resources"""
        # Implementation would close database connections, etc.
        pass

# Make commonly used classes available at module level
__all__ = [
    'WDBX',
    'EmbeddingVector',
    'Block',
    'VectorStore',
    'VectorOperations',
    'BlockChainManager',
    'MVCCManager',
    'NeuralBacktracker',
    'ShardManager',
    'PersonaManager',
    'ContentFilter',
    'BiasDetector',
    'MultiHeadAttention',
    'DatabaseClient',
    'HttpDatabaseClient',
    'SocketDatabaseClient',
    'FilesystemDatabaseClient'
]