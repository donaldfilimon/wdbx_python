"""
Vector Store Module

This module provides classes for storing and retrieving vector embeddings.
It includes a pure Python vector index implementation and a vector store class
that manages vector embeddings and their associated metadata.
"""

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

# Import EmbeddingVector for type hints
from ..core.data_structures import EmbeddingVector

# Set up logging
logger = logging.getLogger(__name__)

# Try to import JAX for faster vector operations
try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
    logger.info("JAX is available and will be used for vector operations")
except ImportError:
    HAS_JAX = False
    logger.info("JAX not available, falling back to NumPy")

# Type aliases for better code readability
VectorId = str
MetadataDict = Dict[str, Any]
VectorType = NDArray[np.float32]
VectorDict = Dict[VectorId, VectorType]
MetadataStorage = Dict[VectorId, MetadataDict]
T = TypeVar("T")

# Constants
DEFAULT_BATCH_SIZE = 1000
DEFAULT_TOP_K = 10


class PythonVectorIndex:
    """
    Pure Python vector index implementation for similarity searches.

    This provides a simple but efficient vector search capability without
    requiring external dependencies like FAISS.
    """

    def __init__(self, dimension: int) -> None:
        """
        Initialize a new vector index.

        Args:
            dimension: The dimension of vectors to be indexed
        """
        self.dimension = dimension
        self.vectors: VectorDict = {}
        self.reset_stats()
        logger.info(f"Initialized Python vector index with dimension {dimension}")

    def reset_stats(self) -> None:
        """Reset statistics tracking for the index."""
        self.stats = {
            "total_searches": 0,
            "total_adds": 0,
            "search_time_ms": 0.0,
            "last_search_time_ms": 0.0,
        }

    def add_vectors(self, vectors: VectorDict) -> None:
        """
        Add multiple vectors to the index at once.

        Args:
            vectors: Dictionary mapping vector IDs to vector values

        Raises:
            ValueError: If any vector has an incorrect dimension
        """
        if not vectors:
            return

        # Dimension check
        for vector_id, vector in vectors.items():
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}"
                )

        # Add all vectors
        self.vectors.update(vectors)
        self.stats["total_adds"] += len(vectors)
        logger.debug(f"Added {len(vectors)} vectors to Python index")

    def reset(self) -> None:
        """Clear all vectors from the index."""
        self.vectors = {}
        self.reset_stats()
        logger.info("Reset Python vector index")

    def search(
        self,
        query_vector: VectorType,
        top_k: int = DEFAULT_TOP_K,
        exclude_ids: Optional[Set[VectorId]] = None,
    ) -> List[Tuple[VectorId, float]]:
        """
        Search for the top_k most similar vectors to the query vector.

        Args:
            query_vector: The query vector
            top_k: Number of top results to return
            exclude_ids: Set of vector IDs to exclude from search

        Returns:
            List of tuples containing (vector_id, similarity_score)

        Raises:
            ValueError: If the query vector has incorrect dimension
        """
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}"
            )

        start_time = time.time()

        # Skip computation if index is empty
        if not self.vectors:
            self.stats["total_searches"] += 1
            self.stats["last_search_time_ms"] = 0.0
            return []

        # Prepare excluded IDs set
        exclude_set = exclude_ids or set()

        # Compute similarities - optimized by using numpy operations
        ids = [vid for vid in self.vectors.keys() if vid not in exclude_set]

        if not ids:
            self.stats["total_searches"] += 1
            self.stats["last_search_time_ms"] = 0.0
            return []

        # Convert to numpy arrays for vectorized computation
        search_vectors = np.array([self.vectors[vid] for vid in ids], dtype=np.float32)

        # Normalize query vector
        if HAS_JAX:
            # Use JAX for faster computation
            query_norm = jnp.linalg.norm(query_vector)
            if query_norm > 0:
                query_vector = query_vector / query_norm

            # Convert to JAX arrays
            jax_search_vectors = jnp.array(search_vectors)
            jax_query_vector = jnp.array(query_vector)

            # Compute dot products (cosine similarity for normalized vectors)
            similarities = jnp.dot(jax_search_vectors, jax_query_vector)
            similarities = np.array(similarities)  # Convert back to numpy
        else:
            # Use NumPy
            query_norm = np.linalg.norm(query_vector)
            if query_norm > 0:
                query_vector = query_vector / query_norm

            # Compute dot products (cosine similarity for normalized vectors)
            similarities = np.dot(search_vectors, query_vector)

        # Get indices of top_k similarities
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        # Create result list
        results = [(ids[i], float(similarities[i])) for i in top_indices]

        end_time = time.time()
        search_time_ms = (end_time - start_time) * 1000

        # Update stats
        self.stats["total_searches"] += 1
        self.stats["last_search_time_ms"] = search_time_ms
        self.stats["search_time_ms"] += search_time_ms

        return results


class VectorStore:
    """
    Manages vector embeddings for storage and retrieval.

    This class provides methods for adding, retrieving, deleting, and searching
    for vectors, as well as managing their associated metadata.
    """

    def __init__(
        self,
        vector_dimension: int,
        storage_path: Optional[Union[str, Path]] = None,
        use_numpy: bool = True,
        auto_save: bool = True,
        auto_save_interval: int = 100,
    ) -> None:
        """
        Initialize a new vector store.

        Args:
            vector_dimension: Dimension of vectors to store
            storage_path: Path to directory for persistent storage
            use_numpy: Whether to use numpy for vector operations
            auto_save: Whether to automatically save vectors periodically
            auto_save_interval: Number of operations between auto-saves

        Raises:
            ValueError: If vector dimension is not positive
        """
        if vector_dimension <= 0:
            raise ValueError("Vector dimension must be positive")

        self.vector_dimension = vector_dimension
        self.vectors: VectorDict = {}
        self.metadata: MetadataStorage = {}
        self._modified = False
        self._operation_count = 0
        self.auto_save = auto_save
        self.auto_save_interval = auto_save_interval

        # Create index
        self.index = PythonVectorIndex(dimension=vector_dimension)

        # Handle storage path
        if storage_path is not None:
            self.storage_path = Path(storage_path).resolve()
            os.makedirs(self.storage_path, exist_ok=True)
            self._try_load()
        else:
            self.storage_path = None

        logger.info(
            f"Initialized VectorStore with dimension={vector_dimension}, "
            f"storage_path={self.storage_path}, auto_save={auto_save}"
        )

    def _try_load(self) -> None:
        """
        Attempt to load vectors and metadata from storage.

        This method gracefully handles missing or corrupt files.
        """
        if not self.storage_path:
            return

        vectors_path = self.storage_path / "vectors.pkl"
        metadata_path = self.storage_path / "metadata.json"

        # Try to load vectors
        if vectors_path.exists():
            try:
                logger.info(f"Loading vectors from {vectors_path}")
                with open(vectors_path, "rb") as f:
                    self.vectors = pickle.load(f)

                # Update index
                self.index.add_vectors(self.vectors)
                logger.info(f"Loaded {len(self.vectors)} vectors")
            except Exception as e:
                logger.error(f"Failed to load vectors: {str(e)}")
                # Create backup of corrupt file
                if vectors_path.exists():
                    backup_path = vectors_path.with_suffix(".pkl.bak")
                    vectors_path.rename(backup_path)
                    logger.info(f"Created backup of corrupt vectors file at {backup_path}")

        # Try to load metadata
        if metadata_path.exists():
            try:
                logger.info(f"Loading metadata from {metadata_path}")
                with open(metadata_path, encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.metadata)} vectors")
            except Exception as e:
                logger.error(f"Failed to load metadata: {str(e)}")
                # Create backup of corrupt file
                if metadata_path.exists():
                    backup_path = metadata_path.with_suffix(".json.bak")
                    metadata_path.rename(backup_path)
                    logger.info(f"Created backup of corrupt metadata file at {backup_path}")

    def save(self) -> bool:
        """
        Save vectors and metadata to storage.

        Returns:
            bool: True if saved successfully, False otherwise
        """
        if not self.storage_path or not self._modified:
            return False

        vectors_path = self.storage_path / "vectors.pkl"
        metadata_path = self.storage_path / "metadata.json"

        success = True

        # Save vectors
        try:
            logger.info(f"Saving {len(self.vectors)} vectors to {vectors_path}")
            # Create temp file first to prevent corruption
            temp_vectors_path = vectors_path.with_suffix(".pkl.tmp")
            with open(temp_vectors_path, "wb") as f:
                pickle.dump(self.vectors, f)
            # Replace original file atomically
            temp_vectors_path.replace(vectors_path)
        except Exception as e:
            logger.error(f"Failed to save vectors: {str(e)}")
            success = False

        # Save metadata
        try:
            logger.info(f"Saving metadata for {len(self.metadata)} vectors to {metadata_path}")
            # Create temp file first to prevent corruption
            temp_metadata_path = metadata_path.with_suffix(".json.tmp")
            with open(temp_metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            # Replace original file atomically
            temp_metadata_path.replace(metadata_path)
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
            success = False

        if success:
            self._modified = False
            logger.info("Successfully saved vectors and metadata")

        return success

    def add(
        self,
        vector_id: VectorId,
        vector: Union[VectorType, "EmbeddingVector"],
        metadata: Optional[MetadataDict] = None,
    ) -> bool:
        """
        Add a vector to the store.

        Args:
            vector_id: Unique ID for the vector
            vector: The vector to add (can be a numpy array, list, or EmbeddingVector)
            metadata: Optional metadata to associate with the vector

        Returns:
            bool: True if added successfully, False otherwise

        Raises:
            ValueError: If the vector dimension doesn't match the store's dimension
        """
        # Handle EmbeddingVector objects
        if hasattr(vector, "vector") and hasattr(vector, "metadata"):
            # Extract vector and metadata from EmbeddingVector
            embedding_vector = vector
            vector_data = embedding_vector.vector
            if metadata is None:
                metadata = embedding_vector.metadata
        else:
            vector_data = vector

        # Input validation
        if len(vector_data) != self.vector_dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.vector_dimension}, got {len(vector_data)}"
            )

        try:
            # Convert to numpy array if needed
            if not isinstance(vector_data, np.ndarray):
                vector_data = np.array(vector_data, dtype=np.float32)

            # Normalize the vector
            if HAS_JAX:
                # Use JAX for normalization
                jax_vector = jnp.array(vector_data)
                norm = jnp.linalg.norm(jax_vector)
                if norm > 0:
                    vector_data = np.array(jax_vector / norm)
            else:
                # Use NumPy
                norm = np.linalg.norm(vector_data)
                if norm > 0:
                    vector_data = vector_data / norm

            # Store the vector
            self.vectors[vector_id] = vector_data

            # Update the index
            self.index.add_vectors({vector_id: vector_data})

            # Store metadata
            if metadata is not None:
                self.metadata[vector_id] = metadata
            elif vector_id not in self.metadata:
                self.metadata[vector_id] = {}

            self._modified = True
            self._operation_count += 1

            # Auto-save if needed
            if self.auto_save and self._operation_count >= self.auto_save_interval:
                self.save()
                self._operation_count = 0

            return True

        except Exception as e:
            logger.error(f"Unexpected error when adding vector {vector_id}: {str(e)}")
            return False

    def add_batch(
        self,
        vectors: Dict[VectorId, Union[VectorType, "EmbeddingVector"]],
        metadata: Optional[Dict[VectorId, MetadataDict]] = None,
    ) -> int:
        """
        Add multiple vectors to the store in a batch operation.

        Args:
            vectors: Dictionary mapping vector IDs to vectors (numpy arrays, lists, or EmbeddingVector objects)
            metadata: Optional dictionary mapping vector IDs to metadata

        Returns:
            int: Number of vectors successfully added

        Raises:
            ValueError: If any vector dimension doesn't match the store's dimension
        """
        if not vectors:
            return 0

        # Input validation and processing
        processed_vectors = {}
        processed_metadata = {} if metadata is None else metadata.copy()

        for vid, vec in vectors.items():
            # Handle EmbeddingVector objects
            if hasattr(vec, "vector") and hasattr(vec, "metadata"):
                embedding_vector = vec
                vector_data = embedding_vector.vector
                # Extract metadata if not explicitly provided
                if vid not in processed_metadata:
                    processed_metadata[vid] = embedding_vector.metadata
            else:
                vector_data = vec

            # Dimension check
            if len(vector_data) != self.vector_dimension:
                raise ValueError(
                    f"Vector dimension mismatch for {vid}: expected {self.vector_dimension}, got {len(vector_data)}"
                )

            processed_vectors[vid] = vector_data

        # Process vectors in batches for better performance
        count = 0
        normalized_vectors = {}

        try:
            # Normalize all vectors
            for vid, vec in processed_vectors.items():
                # Convert to numpy array if needed
                if not isinstance(vec, np.ndarray):
                    vec = np.array(vec, dtype=np.float32)

                # Normalize
                if HAS_JAX:
                    # Use JAX for normalization
                    jax_vec = jnp.array(vec)
                    norm = jnp.linalg.norm(jax_vec)
                    if norm > 0:
                        vec = np.array(jax_vec / norm)
                else:
                    # Use NumPy
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm

                normalized_vectors[vid] = vec
                count += 1

            # Store vectors
            self.vectors.update(normalized_vectors)

            # Update index
            self.index.add_vectors(normalized_vectors)

            # Store metadata
            if processed_metadata:
                for vid, meta in processed_metadata.items():
                    if vid in normalized_vectors:
                        self.metadata[vid] = meta

            # For vectors without metadata, ensure they have an empty dict
            for vid in normalized_vectors:
                if vid not in self.metadata:
                    self.metadata[vid] = {}

            self._modified = True
            self._operation_count += count

            # Auto-save if needed
            if self.auto_save and self._operation_count >= self.auto_save_interval:
                self.save()
                self._operation_count = 0

            return count

        except Exception as e:
            logger.error(f"Unexpected error when adding batch of {len(vectors)} vectors: {str(e)}")
            return count

    def get(self, vector_id: VectorId) -> Optional[Tuple[VectorType, MetadataDict]]:
        """
        Retrieve a vector and its metadata by ID.

        Args:
            vector_id: ID of the vector to retrieve

        Returns:
            Tuple of (vector, metadata) if found, None otherwise
        """
        if vector_id not in self.vectors:
            return None

        vector = self.vectors[vector_id]
        metadata = self.metadata.get(vector_id, {})

        return vector, metadata

    def delete(self, vector_id: VectorId) -> bool:
        """
        Delete a vector and its metadata by ID.

        Args:
            vector_id: ID of the vector to delete

        Returns:
            bool: True if deleted, False if not found
        """
        if vector_id not in self.vectors:
            return False

        # Delete vector from storage
        del self.vectors[vector_id]

        # Delete metadata if it exists
        if vector_id in self.metadata:
            del self.metadata[vector_id]

        self._modified = True
        self._operation_count += 1

        # Auto-save if needed
        if self.auto_save and self._operation_count >= self.auto_save_interval:
            self.save()
            self._operation_count = 0

        # Note: We don't remove from the index, it will be rebuilt on next search
        # This is a tradeoff - removing is complex, and index is rebuilt on next search

        return True

    def search(
        self,
        query_vector: VectorType,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = 0.0,
        include_vectors: bool = False,
        include_metadata: bool = True,
        exclude_ids: Optional[Set[VectorId]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector to search with
            top_k: Number of top results to return
            threshold: Minimum similarity threshold (0-1)
            include_vectors: Whether to include vector values in results
            include_metadata: Whether to include metadata in results
            exclude_ids: Set of vector IDs to exclude from search

        Returns:
            List of results sorted by similarity (descending)
        """
        # Input validation
        if len(query_vector) != self.vector_dimension:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.vector_dimension}, got {len(query_vector)}"
            )

        # Convert to numpy array if needed
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)

        # Normalize query vector
        if HAS_JAX:
            # Use JAX for normalization
            jax_query = jnp.array(query_vector)
            norm = jnp.linalg.norm(jax_query)
            if norm > 0:
                query_vector = np.array(jax_query / norm)
        else:
            # Use NumPy
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

        # Perform search using index
        results = self.index.search(query_vector, top_k, exclude_ids)

        # Filter by threshold and format results
        formatted_results = []
        for vector_id, similarity in results:
            if similarity < threshold:
                continue

            result = {"id": vector_id, "similarity": similarity}

            if include_metadata:
                result["metadata"] = self.metadata.get(vector_id, {})

            if include_vectors:
                result["vector"] = self.vectors[vector_id].tolist()

            formatted_results.append(result)

        return formatted_results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary containing statistics
        """
        return {
            "vector_count": len(self.vectors),
            "metadata_count": len(self.metadata),
            "dimension": self.vector_dimension,
            "storage_path": str(self.storage_path) if self.storage_path else None,
            "auto_save": self.auto_save,
            "auto_save_interval": self.auto_save_interval,
            "index_stats": self.index.stats,
            "modified": self._modified,
            "operation_count": self._operation_count,
            "using_jax": HAS_JAX,
        }

    def clear(self) -> None:
        """
        Clear all vectors and metadata from the store.
        """
        self.vectors = {}
        self.metadata = {}
        self.index.reset()
        self._modified = True
        self._operation_count += 1

        # Auto-save if needed
        if self.auto_save and self._operation_count >= self.auto_save_interval:
            self.save()
            self._operation_count = 0

        logger.info("Cleared all vectors and metadata from store")

    def export_json(self, file_path: Union[str, Path]) -> bool:
        """
        Export vectors and metadata to a JSON file.

        Args:
            file_path: Path to save the JSON file

        Returns:
            bool: True if exported successfully, False otherwise
        """
        try:
            # Convert vectors to lists for JSON serialization
            export_data = {
                "dimension": self.vector_dimension,
                "count": len(self.vectors),
                "vectors": {
                    vid: vec.tolist() if isinstance(vec, np.ndarray) else vec
                    for vid, vec in self.vectors.items()
                },
                "metadata": self.metadata,
            }

            # Ensure directory exists
            file_path = Path(file_path).resolve()
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file first
            temp_path = file_path.with_suffix(".json.tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            # Replace original file atomically
            temp_path.replace(file_path)

            logger.info(f"Exported {len(self.vectors)} vectors to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export vectors to JSON: {str(e)}")
            return False

    def import_json(self, file_path: Union[str, Path]) -> int:
        """
        Import vectors and metadata from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            int: Number of vectors imported

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        file_path = Path(file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Import file not found: {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                import_data = json.load(f)

            # Validate format
            if not isinstance(import_data, dict):
                raise ValueError("Invalid import format: root must be an object")

            if "dimension" not in import_data or import_data["dimension"] != self.vector_dimension:
                raise ValueError(
                    f"Dimension mismatch: file has {import_data.get('dimension')}, "
                    f"store has {self.vector_dimension}"
                )

            if "vectors" not in import_data or not isinstance(import_data["vectors"], dict):
                raise ValueError("Invalid import format: missing vectors object")

            # Import vectors and metadata
            vectors_to_add = {}
            metadata_to_add = {}

            for vid, vec in import_data["vectors"].items():
                # Convert to numpy array
                vector = np.array(vec, dtype=np.float32)

                # Validate dimension
                if len(vector) != self.vector_dimension:
                    logger.warning(
                        f"Skipping vector {vid}: dimension mismatch "
                        f"(expected {self.vector_dimension}, got {len(vector)})"
                    )
                    continue

                vectors_to_add[vid] = vector

                # Get metadata if available
                if "metadata" in import_data and vid in import_data["metadata"]:
                    metadata_to_add[vid] = import_data["metadata"][vid]

            # Add all vectors and metadata
            count = self.add_batch(vectors_to_add, metadata_to_add)

            logger.info(f"Imported {count} vectors from {file_path}")
            return count

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse import file: {str(e)}")
            raise ValueError(f"Invalid JSON in import file: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to import vectors from JSON: {str(e)}")
            raise

    def search_similar(
        self,
        query_vector: Union[VectorType, "EmbeddingVector"],
        top_k: int = DEFAULT_TOP_K,
        threshold: float = 0.0,
        exclude_ids: Optional[Set[VectorId]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors. Returns vector IDs and similarity scores.

        Args:
            query_vector: Query vector to search with (can be a numpy array, list, or EmbeddingVector)
            top_k: Number of top results to return
            threshold: Minimum similarity threshold (0-1)
            exclude_ids: Set of vector IDs to exclude from search

        Returns:
            List of tuples (vector_id, similarity) sorted by similarity (descending)

        Raises:
            ValueError: If the query vector dimension doesn't match the store's dimension
        """
        # Handle None input
        if query_vector is None:
            raise ValueError("Query vector cannot be None")

        # Handle EmbeddingVector objects
        if hasattr(query_vector, "vector") and hasattr(query_vector, "metadata"):
            query_vector = query_vector.vector

        # Input validation
        if len(query_vector) != self.vector_dimension:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.vector_dimension}, got {len(query_vector)}"
            )

        # Convert to numpy array if needed
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)

        # Normalize query vector
        if HAS_JAX:
            # Use JAX for normalization
            jax_query = jnp.array(query_vector)
            norm = jnp.linalg.norm(jax_query)
            if norm > 0:
                query_vector = np.array(jax_query / norm)
        else:
            # Use NumPy
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

        # Perform search using index
        results = self.index.search(query_vector, top_k, exclude_ids)

        # Filter by threshold
        if threshold > 0:
            results = [(vid, sim) for vid, sim in results if sim >= threshold]

        return results

    def count(self) -> int:
        """
        Get the total number of vectors in the store.

        Returns:
            int: Number of vectors stored
        """
        return len(self.vectors)

    def process_in_batches(
        self,
        vectors: List[Union[VectorType, "EmbeddingVector"]],
        batch_size: int = DEFAULT_BATCH_SIZE,
        process_fn: Optional[
            Callable[[List[Union[VectorType, "EmbeddingVector"]]], List[Any]]
        ] = None,
    ) -> List[Any]:
        """
        Process a large list of vectors in batches to avoid memory issues.

        Args:
            vectors: List of vectors to process
            batch_size: Size of each batch
            process_fn: Function to apply to each batch (defaults to normalization)

        Returns:
            List of processed results
        """
        if not vectors:
            return []

        results = []
        batches = [vectors[i : i + batch_size] for i in range(0, len(vectors), batch_size)]

        logger.info(
            f"Processing {len(vectors)} vectors in {len(batches)} batches of size {batch_size}"
        )

        for i, batch in enumerate(batches):
            logger.debug(f"Processing batch {i+1}/{len(batches)} with {len(batch)} vectors")

            # Default processing is normalization
            if process_fn is None:
                # Convert to numpy arrays
                batch_arrays = []
                for vec in batch:
                    # Handle EmbeddingVector objects
                    if hasattr(vec, "vector") and hasattr(vec, "metadata"):
                        vector_data = vec.vector
                    else:
                        vector_data = vec

                    # Convert to numpy array if needed
                    if not isinstance(vector_data, np.ndarray):
                        vector_data = np.array(vector_data, dtype=np.float32)

                    # Normalize
                    if HAS_JAX:
                        jax_vec = jnp.array(vector_data)
                        norm = jnp.linalg.norm(jax_vec)
                        if norm > 0:
                            vector_data = np.array(jax_vec / norm)
                    else:
                        norm = np.linalg.norm(vector_data)
                        if norm > 0:
                            vector_data = vector_data / norm

                    batch_arrays.append(vector_data)

                results.extend(batch_arrays)
            else:
                # Apply custom process function
                batch_results = process_fn(batch)
                results.extend(batch_results)

        return results
