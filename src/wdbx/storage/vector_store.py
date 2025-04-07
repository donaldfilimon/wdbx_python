"""
Vector storage module for WDBX.

This module provides classes for storing and retrieving vector embeddings.
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Set up logger
logger = logging.getLogger("wdbx.storage.vector_store")

# Local type definition to avoid circular imports
if TYPE_CHECKING:
    from ..data_structures import EmbeddingVector
    VectorType = Union[List[float], np.ndarray, "EmbeddingVector"]
else:
    VectorType = Union[List[float], np.ndarray]


class PythonVectorIndex:
    """
    Pure Python implementation of a vector index for similarity search.
    Used as a fallback when FAISS is not available.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize the vector index.
        
        Args:
            dimension: Dimension of the vectors
        """
        self.dimension = dimension
        self.vectors = []
        self.ids = []
        
    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the index.
        
        Args:
            vectors: Array of vectors to add
        """
        # Reset before adding
        self.vectors = []
        self.ids = []
        
        # Add each vector
        for i in range(vectors.shape[0]):
            self.vectors.append(vectors[i])
            self.ids.append(i)
    
    def reset(self) -> None:
        """Clear the index."""
        self.vectors = []
        self.ids = []
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for similar vectors.
        
        Args:
            query: Query vector
            k: Number of results to return
            
        Returns:
            List of (id, score) tuples
        """
        if not self.vectors:
            return []
        
        # Calculate similarity scores
        scores = []
        for i, vec in enumerate(self.vectors):
            # Inner product for normalized vectors is equivalent to cosine similarity
            score = np.dot(query, vec)
            scores.append((i, float(score)))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return scores[:k]


class VectorStore:
    """
    Storage for vector embeddings with efficient similarity search.
    
    Attributes:
        dimension: Dimension of the vectors
        vectors: Dictionary of vector_id -> vector
        metadata: Dictionary of vector_id -> metadata
    """
    
    def __init__(self, dimension: int = 768):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the vectors
        """
        self.dimension = dimension
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"VectorStore initialized with dimension={dimension}")
    
    def add(self, vector_id: str, vector: VectorType, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a vector to the store.
        
        Args:
            vector_id: Unique identifier for the vector
            vector: The vector to store (numpy array, list of floats, or EmbeddingVector)
            metadata: Optional metadata for the vector
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Handle EmbeddingVector objects
            if hasattr(vector, "vector") and hasattr(vector, "metadata"):
                metadata = vector.metadata
                vector = vector.vector
            
            # Ensure correct shape
            if not isinstance(vector, np.ndarray):
                vector = np.array(vector, dtype=np.float32)
                
            # Handle dimension mismatch - try to resize if possible
            if vector.shape != (self.dimension,):
                try:
                    # Case 1: Vector needs reshaping
                    if len(vector.shape) > 1 and vector.shape[0] * vector.shape[1] == self.dimension:
                        vector = vector.reshape(self.dimension)
                    # Case 2: Vector has different dimension, try to resize
                    elif len(vector.shape) == 1:
                        # If vector is smaller, pad with zeros
                        if vector.shape[0] < self.dimension:
                            padded = np.zeros(self.dimension, dtype=np.float32)
                            padded[:vector.shape[0]] = vector
                            vector = padded
                        # If vector is larger, truncate
                        elif vector.shape[0] > self.dimension:
                            vector = vector[:self.dimension]
                    else:
                        logger.error(f"Vector has wrong dimension: {vector.shape} != ({self.dimension},)")
                        return False
                    
                    # Normalize the vector after resizing to ensure valid similarity calculations
                    vector_norm = np.linalg.norm(vector)
                    if vector_norm > 0:
                        vector = vector / vector_norm
                    logger.info(f"Resized and normalized vector for ID {vector_id} to dimension {self.dimension}")
                    
                except Exception as e:
                    logger.error(f"Error resizing vector: {e}")
                    return False
            
            # Store the vector and metadata
            self.vectors[vector_id] = vector
            self.metadata[vector_id] = metadata or {"timestamp": time.time()}
            
            return True
        except Exception as e:
            logger.error(f"Unexpected error in vector store add: {str(e)}")
            return False
    
    def get(self, vector_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Get a vector by ID.
        
        Args:
            vector_id: ID of the vector
            
        Returns:
            Tuple of (vector, metadata) if found, None otherwise
        """
        if vector_id not in self.vectors:
            return None
        
        return (self.vectors[vector_id], self.metadata.get(vector_id, {}))
    
    def delete(self, vector_id: str) -> bool:
        """
        Delete a vector from the store.
        
        Args:
            vector_id: ID of the vector
            
        Returns:
            True if deleted, False if not found
        """
        if vector_id not in self.vectors:
            return False
        
        del self.vectors[vector_id]
        if vector_id in self.metadata:
            del self.metadata[vector_id]
        
        return True
    
    def search_similar(self, query_vector: VectorType, top_k: int = 10,
                      threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Search for vectors similar to the query vector.
        
        Args:
            query_vector: Query vector (numpy array, list of floats, or EmbeddingVector)
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (vector_id, similarity_score) tuples
        """
        if not self.vectors:
            return []
        
        # Handle EmbeddingVector objects
        if hasattr(query_vector, "vector"):
            query_vector = query_vector.vector
        
        # Ensure query vector is properly shaped
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
            
        # Handle dimension mismatch - try to resize if possible
        if query_vector.shape != (self.dimension,):
            try:
                # Case 1: Vector needs reshaping
                if len(query_vector.shape) > 1 and query_vector.shape[0] * query_vector.shape[1] == self.dimension:
                    query_vector = query_vector.reshape(self.dimension)
                # Case 2: Vector has different dimension, try to resize
                elif len(query_vector.shape) == 1:
                    # If vector is smaller, pad with zeros
                    if query_vector.shape[0] < self.dimension:
                        padded = np.zeros(self.dimension, dtype=np.float32)
                        padded[:query_vector.shape[0]] = query_vector
                        query_vector = padded
                    # If vector is larger, truncate
                    elif query_vector.shape[0] > self.dimension:
                        query_vector = query_vector[:self.dimension]
                    # Normalize after resizing
                    query_vector = query_vector / (np.linalg.norm(query_vector) or 1.0)
                else:
                    logger.error(f"Query vector has wrong dimension: {query_vector.shape} != ({self.dimension},)")
                    return []
            except Exception as e:
                logger.error(f"Error resizing query vector: {e}")
                return []
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
        
        # Calculate similarities using dot product (cosine similarity for normalized vectors)
        results = []
        for vid, vec in self.vectors.items():
            # Normalize vector
            vec_norm = np.linalg.norm(vec)
            if vec_norm > 0:
                vec = vec / vec_norm
            
            similarity = float(np.dot(query_vector, vec))
            if similarity >= threshold:
                results.append((vid, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return results[:top_k]
    
    async def search_in_shard(self, query_vector: VectorType, shard_id: str, top_k: int = 10,
                      threshold: float = 0.7, transaction_id: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Search for vectors similar to the query vector within a specific shard.
        This method exists for compatibility with sharded implementations.
        In this simple implementation, it's just an alias for search_similar.
        
        Args:
            query_vector: Query vector (numpy array, list of floats, or EmbeddingVector)
            shard_id: ID of the shard to search in (ignored in this implementation)
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            transaction_id: Optional transaction ID for MVCC consistency (ignored in this implementation)
            
        Returns:
            List of (vector_id, similarity_score) tuples
        """
        try:
            # In this simple implementation, we ignore the shard_id and transaction_id
            # and just call the regular search_similar method
            return self.search_similar(query_vector, top_k, threshold)
        except Exception as e:
            logger.error(f"Error in search_in_shard for shard {shard_id}: {e}")
            # Return empty result on error rather than propagating exception
            return []
    
    def count(self) -> int:
        """
        Get the number of vectors in the store.
        
        Returns:
            Number of vectors
        """
        return len(self.vectors)


class VectorOperations:
    """Utility class for vector operations."""
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate the cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity (between -1 and 1)
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate the Euclidean distance between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(a - b))
    
    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to unit length.
        
        Args:
            vector: Vector to normalize
            
        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
