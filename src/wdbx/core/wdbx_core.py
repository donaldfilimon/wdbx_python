"""
Core WDBX engine implementation.

This module provides the main WDBX class that serves as the central
interface for interacting with the WDBX system, integrating various
components such as data structures, ML backends, and memory management.
"""

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from .constants import VECTOR_DIMENSION, DEFAULT_BIAS_SCORE, logger
from .data_structures import Block, EmbeddingVector
from ..config.config_manager import ConfigManager, get_config
from ..ml.backend import MLBackend, get_ml_backend
from ..utils.memory_utils import (MemoryMonitor, get_memory_monitor,
                                 get_memory_usage, optimize_memory)
from ..utils.logging_utils import LogContext, log_execution_time


@dataclass
class ProcessingStats:
    """Statistics for WDBX processing operations."""
    
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    memory_before: float = field(default_factory=get_memory_usage)
    memory_after: float = 0.0
    memory_peak: float = field(default_factory=get_memory_usage)
    num_vectors_processed: int = 0
    num_blocks_processed: int = 0
    optimization_count: int = 0
    
    def finalize(self) -> None:
        """Finalize statistics after processing is complete."""
        self.end_time = time.time()
        self.memory_after = get_memory_usage()
    
    @property
    def processing_time(self) -> float:
        """Get the total processing time in seconds."""
        if self.end_time == 0.0:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def memory_used(self) -> float:
        """Get the memory used during processing in MB."""
        return self.memory_peak - self.memory_before
    
    def record_memory_usage(self) -> None:
        """Record current memory usage and update peak if necessary."""
        current = get_memory_usage()
        if current > self.memory_peak:
            self.memory_peak = current
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for logging."""
        if self.end_time == 0.0:
            self.finalize()
        
        return {
            "processing_time_seconds": self.processing_time,
            "memory_before_mb": self.memory_before,
            "memory_after_mb": self.memory_after,
            "memory_peak_mb": self.memory_peak,
            "memory_used_mb": self.memory_used,
            "vectors_processed": self.num_vectors_processed,
            "blocks_processed": self.num_blocks_processed,
            "optimization_count": self.optimization_count,
            "vectors_per_second": self.num_vectors_processed / max(self.processing_time, 0.001),
            "blocks_per_second": self.num_blocks_processed / max(self.processing_time, 0.001)
        }


class WDBXCore:
    """
    Core WDBX engine for data processing and analysis.
    
    This class integrates various components of the WDBX system, providing a unified
    interface for vector operations, block management, and data analysis.
    """
    
    def __init__(self, 
                data_dir: Optional[str] = None,
                vector_dimension: int = VECTOR_DIMENSION,
                enable_memory_optimization: bool = True):
        """
        Initialize the WDBX core engine.
        
        Args:
            data_dir: Directory for storing data (defaults to config value)
            vector_dimension: Dimension of embedding vectors
            enable_memory_optimization: Whether to enable automatic memory optimization
        """
        # Initialize logging context
        log_context = LogContext(component="WDBXCore")
        with log_context:
            logger.info(f"Initializing WDBX Core (vector_dim={vector_dimension})")
            
            # Set up configuration
            self.config = ConfigManager.get_instance()
            
            # Set up data directory
            self.data_dir = Path(data_dir or get_config("data_dir"))
            self._ensure_data_directory()
            
            # Initialize core attributes
            self.vector_dimension = vector_dimension
            self.enable_memory_optimization = enable_memory_optimization
            
            # Initialize ML backend
            self.ml_backend = get_ml_backend(
                preferred_backend=get_config("ml_backend"),
                vector_dimension=vector_dimension
            )
            
            # Initialize memory monitoring
            if enable_memory_optimization:
                self.memory_monitor = get_memory_monitor()
                self.memory_monitor.register_optimization_callback(self.optimize_memory)
                self.memory_monitor.register_object("wdbx_core", self)
                
                # Start monitoring if enabled in config
                if get_config("memory_optimization_enabled", True):
                    self.memory_monitor.start_monitoring()
            
            # Initialize storage for blocks and vectors
            self.blocks: Dict[str, Block] = {}
            self.vectors: Dict[str, EmbeddingVector] = {}
            
            # Initialize statistics
            self.stats = ProcessingStats()
            
            logger.info(f"WDBX Core initialized (data_dir={self.data_dir})")
    
    def _ensure_data_directory(self) -> None:
        """Ensure the data directory exists."""
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created data directory: {self.data_dir}")
    
    def create_vector(self, 
                     vector_data: Union[List[float], NDArray[np.float32]],
                     metadata: Optional[Dict[str, Any]] = None,
                     vector_id: Optional[str] = None,
                     bias_score: float = DEFAULT_BIAS_SCORE) -> EmbeddingVector:
        """
        Create a new embedding vector.
        
        Args:
            vector_data: Vector data as list or numpy array
            metadata: Additional metadata for the vector
            vector_id: Optional ID for the vector (generated if not provided)
            bias_score: Bias score for the vector
            
        Returns:
            Created EmbeddingVector instance
        """
        # Convert vector to numpy array if needed
        if not isinstance(vector_data, np.ndarray):
            vector_data = np.array(vector_data, dtype=np.float32)
        
        # Validate vector dimension
        if vector_data.shape[0] != self.vector_dimension:
            raise ValueError(
                f"Vector dimension mismatch: got {vector_data.shape[0]}, "
                f"expected {self.vector_dimension}"
            )
        
        # Create embedding vector
        embedding = EmbeddingVector(
            vector=vector_data,
            metadata=metadata or {},
            vector_id=vector_id or str(uuid.uuid4()),
            bias_score=bias_score
        )
        
        # Register vector
        self.vectors[embedding.vector_id] = embedding
        self.stats.num_vectors_processed += 1
        
        # Check if memory optimization is needed
        if (self.enable_memory_optimization and 
            self.memory_monitor.should_optimize()):
            self.optimize_memory()
        
        return embedding
    
    def create_block(self,
                   data: Dict[str, Any],
                   embeddings: Optional[List[EmbeddingVector]] = None,
                   block_id: Optional[str] = None,
                   references: Optional[List[str]] = None) -> Block:
        """
        Create a new data block.
        
        Args:
            data: Block data
            embeddings: Embedding vectors associated with this block
            block_id: Optional ID for the block (generated if not provided)
            references: Optional references to other blocks
            
        Returns:
            Created Block instance
        """
        # Create block
        block = Block(
            data=data,
            embeddings=embeddings or [],
            block_id=block_id or str(uuid.uuid4()),
            context_references=references or []
        )
        
        # Register block
        self.blocks[block.block_id] = block
        self.stats.num_blocks_processed += 1
        
        # Update statistics
        self.stats.record_memory_usage()
        
        # Check if memory optimization is needed
        if (self.enable_memory_optimization and 
            self.memory_monitor.should_optimize()):
            self.optimize_memory()
        
        return block
    
    def find_similar_vectors(self, 
                          query_vector: Union[EmbeddingVector, List[float], NDArray[np.float32]],
                          top_k: int = 10,
                          threshold: float = 0.0) -> List[Tuple[str, float]]:
        """
        Find vectors similar to the query vector.
        
        Args:
            query_vector: Query vector (EmbeddingVector, list, or numpy array)
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (vector_id, similarity) tuples sorted by similarity
        """
        with log_execution_time("find_similar_vectors"):
            # Convert query to numpy array if needed
            if isinstance(query_vector, EmbeddingVector):
                query_array = query_vector.vector
            elif isinstance(query_vector, list):
                query_array = np.array(query_vector, dtype=np.float32)
            else:
                query_array = query_vector
            
            # Validate vector dimension
            if query_array.shape[0] != self.vector_dimension:
                raise ValueError(
                    f"Query vector dimension mismatch: got {query_array.shape[0]}, "
                    f"expected {self.vector_dimension}"
                )
            
            # Early return if no vectors
            if not self.vectors:
                return []
            
            # Normalize query vector
            query_array = self.ml_backend.normalize(query_array)
            
            # Get all vector IDs and vectors
            vector_ids = list(self.vectors.keys())
            vectors = np.stack([self.vectors[vid].vector for vid in vector_ids])
            
            # Compute similarities
            similarities, indices = self.ml_backend.search_vectors(
                query_array, vectors, top_k
            )
            
            # Filter by threshold and format results
            results = []
            for idx, similarity in zip(indices, similarities):
                if similarity < threshold:
                    continue
                if idx < len(vector_ids):  # Safety check
                    results.append((vector_ids[idx], float(similarity)))
            
            # Update statistics
            self.stats.record_memory_usage()
            
            return results
    
    def search_blocks(self,
                    query: Union[str, Dict[str, Any], EmbeddingVector, NDArray[np.float32]],
                    top_k: int = 10,
                    threshold: float = 0.0) -> List[Tuple[Block, float]]:
        """
        Search for blocks relevant to the query.
        
        Args:
            query: Query as text, dict, embedding vector, or numpy array
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (block, similarity) tuples sorted by similarity
        """
        with log_execution_time("search_blocks"):
            # Early return if no blocks
            if not self.blocks:
                return []
            
            # Process query based on type
            if isinstance(query, str) or isinstance(query, dict):
                # TODO: Convert text/dict query to embedding vector
                # This would use a text embedding model
                raise NotImplementedError(
                    "Text and dictionary queries not yet implemented"
                )
            elif isinstance(query, EmbeddingVector):
                query_vector = query.vector
            else:  # numpy array
                query_vector = query
            
            # Find blocks with matching embeddings
            results = []
            
            # Group blocks by their embeddings
            block_embeddings = {}
            for block_id, block in self.blocks.items():
                if not block.embeddings:
                    continue
                
                # Average the embeddings for each block
                vectors = [emb.vector for emb in block.embeddings]
                avg_vector = np.mean(vectors, axis=0)
                block_embeddings[block_id] = self.ml_backend.normalize(avg_vector)
            
            # Early return if no blocks with embeddings
            if not block_embeddings:
                return []
            
            # Get all block IDs and normalized embeddings
            block_ids = list(block_embeddings.keys())
            vectors = np.stack([block_embeddings[bid] for bid in block_ids])
            
            # Compute similarities
            similarities, indices = self.ml_backend.search_vectors(
                self.ml_backend.normalize(query_vector), vectors, top_k
            )
            
            # Filter by threshold and format results
            for idx, similarity in zip(indices, similarities):
                if similarity < threshold:
                    continue
                if idx < len(block_ids):  # Safety check
                    block_id = block_ids[idx]
                    results.append((self.blocks[block_id], float(similarity)))
            
            # Update statistics
            self.stats.record_memory_usage()
            
            return results
    
    def save_vector(self, vector: EmbeddingVector, overwrite: bool = False) -> bool:
        """
        Save a vector to disk.
        
        Args:
            vector: Vector to save
            overwrite: Whether to overwrite if the file exists
            
        Returns:
            True if successful
        """
        vector_path = self.data_dir / "vectors" / f"{vector.vector_id}.json"
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        
        if vector_path.exists() and not overwrite:
            logger.warning(f"Vector file exists and overwrite=False: {vector_path}")
            return False
        
        # Serialize and save
        vector_data = vector.to_dict()
        with open(vector_path, "w") as f:
            import json
            json.dump(vector_data, f, indent=2)
        
        return True
    
    def save_block(self, block: Block, overwrite: bool = False) -> bool:
        """
        Save a block to disk.
        
        Args:
            block: Block to save
            overwrite: Whether to overwrite if the file exists
            
        Returns:
            True if successful
        """
        block_path = self.data_dir / "blocks" / f"{block.block_id}.json"
        block_path.parent.mkdir(parents=True, exist_ok=True)
        
        if block_path.exists() and not overwrite:
            logger.warning(f"Block file exists and overwrite=False: {block_path}")
            return False
        
        # Serialize and save
        block_data = block.to_dict()
        with open(block_path, "w") as f:
            import json
            json.dump(block_data, f, indent=2)
        
        return True
    
    def load_vector(self, vector_id: str) -> Optional[EmbeddingVector]:
        """
        Load a vector from disk.
        
        Args:
            vector_id: ID of the vector to load
            
        Returns:
            Loaded vector or None if not found
        """
        vector_path = self.data_dir / "vectors" / f"{vector_id}.json"
        
        if not vector_path.exists():
            logger.warning(f"Vector file not found: {vector_path}")
            return None
        
        try:
            with open(vector_path, "r") as f:
                import json
                vector_data = json.load(f)
            
            vector = EmbeddingVector.from_dict(vector_data)
            
            # Register in memory
            self.vectors[vector.vector_id] = vector
            
            return vector
        except Exception as e:
            logger.error(f"Error loading vector {vector_id}: {e}")
            return None
    
    def load_block(self, block_id: str) -> Optional[Block]:
        """
        Load a block from disk.
        
        Args:
            block_id: ID of the block to load
            
        Returns:
            Loaded block or None if not found
        """
        block_path = self.data_dir / "blocks" / f"{block_id}.json"
        
        if not block_path.exists():
            logger.warning(f"Block file not found: {block_path}")
            return None
        
        try:
            with open(block_path, "r") as f:
                import json
                block_data = json.load(f)
            
            block = Block.from_dict(block_data)
            
            # Load referenced vectors if not already in memory
            for i, embedding_ref in enumerate(block_data.get("embeddings", [])):
                vector_id = embedding_ref.get("vector_id")
                if vector_id and vector_id not in self.vectors:
                    self.load_vector(vector_id)
            
            # Register in memory
            self.blocks[block.block_id] = block
            
            return block
        except Exception as e:
            logger.error(f"Error loading block {block_id}: {e}")
            return None
    
    def optimize_memory(self) -> None:
        """
        Optimize memory usage by clearing caches and unnecessary data.
        """
        # Increment optimization count
        self.stats.optimization_count += 1
        
        # Optimize ML backend
        self.ml_backend.optimize_memory()
        
        # Optimize vectors (convert to optimal dtype and ensure contiguous)
        for vector_id, vector in self.vectors.items():
            vector.optimize_memory()
        
        # Optimize blocks
        for block_id, block in self.blocks.items():
            block.optimize_memory()
        
        # Force garbage collection
        optimize_memory()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats.to_dict()
    
    def clear(self) -> None:
        """
        Clear all in-memory data.
        """
        self.vectors.clear()
        self.blocks.clear()
        self.optimize_memory()
        
        # Reset statistics
        self.stats = ProcessingStats()
        
        logger.info("Cleared all in-memory data")
    
    def shutdown(self) -> None:
        """
        Shut down WDBX core and release resources.
        """
        # Stop memory monitoring if enabled
        if self.enable_memory_optimization:
            self.memory_monitor.stop_monitoring()
        
        # Clear all data
        self.clear()
        
        logger.info("WDBX Core shut down")


# Singleton instance
_WDBX_CORE_INSTANCE: Optional[WDBXCore] = None


def get_wdbx_core(create_if_missing: bool = True) -> Optional[WDBXCore]:
    """
    Get the WDBX core instance.
    
    Args:
        create_if_missing: Whether to create the instance if it doesn't exist
        
    Returns:
        WDBX core instance or None
    """
    global _WDBX_CORE_INSTANCE
    
    if _WDBX_CORE_INSTANCE is None and create_if_missing:
        _WDBX_CORE_INSTANCE = WDBXCore(
            data_dir=get_config("data_dir"),
            vector_dimension=VECTOR_DIMENSION,
            enable_memory_optimization=get_config("memory_optimization_enabled", True)
        )
    
    return _WDBX_CORE_INSTANCE


def initialize_wdbx() -> WDBXCore:
    """
    Initialize the WDBX system.
    
    Returns:
        Initialized WDBX core instance
    """
    return get_wdbx_core(create_if_missing=True) 