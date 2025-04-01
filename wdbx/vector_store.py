# wdbx/vector_store.py
"""
Vector storage and similarity search for WDBX.

This module provides efficient storage and retrieval of embedding vectors,
with support for cosine similarity search and clustering operations.
"""
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
import math

from wdbx.constants import logger, VECTOR_DIMENSION, DEFAULT_SIMILARITY_THRESHOLD
from wdbx.data_structures import EmbeddingVector

# Try to import FAISS, but provide a fallback if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Using pure Python vector similarity search.")

# Try to import scikit-learn for clustering, but provide a fallback if not available
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Using pure Python clustering implementation.")


class PythonVectorIndex:
    """
    Pure Python implementation of a vector index for similarity search.
    Used as a fallback when FAISS is not available.
    """
    def __init__(self, dimension: int):
        """
        Initialize the Python vector index.

        Args:
            dimension (int): Dimension of vectors to be indexed
        """
        self.dimension = dimension
        self.vectors: List[np.ndarray] = []

    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            vectors (np.ndarray): Vectors to add, shape (n, dimension)
        """
        # Convert to list of vectors if needed
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        for i in range(vectors.shape[0]):
            self.vectors.append(vectors[i])

    def reset(self) -> None:
        """Reset the index, removing all vectors."""
        self.vectors.clear()

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the k nearest vectors to the query.

        Args:
            query (np.ndarray): Query vector, shape (1, dimension)
            k (int): Number of results to return

        Returns:
            Tuple[np.ndarray, np.ndarray]: (similarities, indices)
        """
        if not self.vectors:
            return np.array([[0.0]]), np.array([[-1]])

        # Ensure query is normalized
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        # Calculate cosine similarities
        similarities = []
        for vector in self.vectors:
            vec_norm = np.linalg.norm(vector)
            if vec_norm > 0:
                vector = vector / vec_norm
            similarity = np.dot(query.flatten(), vector)
            similarities.append(similarity)

        # Sort and get top k
        if not similarities:
            return np.array([[0.0]]), np.array([[-1]])

        k = min(k, len(similarities))
        if k == 0:
            return np.array([[0.0]]), np.array([[-1]])

        sorted_indices = np.argsort(similarities)[::-1][:k]
        sorted_similarities = np.array([similarities[i] for i in sorted_indices])

        return sorted_similarities.reshape(1, -1), sorted_indices.reshape(1, -1)


class VectorStore:
    """
    Optimized storage for embedding vectors with similarity search capabilities.
    Uses FAISS for efficient similarity search when available, with a pure Python
    fallback implementation.
    """
    def __init__(self, dimension: int = VECTOR_DIMENSION) -> None:
        """
        Initialize the vector store.

        Args:
            dimension (int): Dimension of vectors to be stored
        """
        self.dimension = dimension
        self.vectors: Dict[str, EmbeddingVector] = {}
        self.id_map: Dict[int, str] = {}
        self.reverse_id_map: Dict[str, int] = {}
        self.lock = threading.RLock()

        # Initialize the appropriate index based on availability
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dimension)
            logger.info(f"VectorStore initialized with FAISS, dimension={dimension}")
        else:
            self.index = PythonVectorIndex(dimension)
            logger.info(f"VectorStore initialized with Python fallback, dimension={dimension}")

    def add(self, vector_id: str, vector: EmbeddingVector) -> bool:
        """
        Add a vector to the store.

        Args:
            vector_id (str): Unique identifier for the vector
            vector (EmbeddingVector): Vector to store

        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            if vector_id in self.vectors:
                logger.warning(f"Vector ID {vector_id} already exists.")
                return False

            if not isinstance(vector, EmbeddingVector):
                logger.error(f"Expected EmbeddingVector, got {type(vector)}")
                return False

            try:
                # Store the vector
                self.vectors[vector_id] = vector

                # Normalize and add to index
                normalized_vector = vector.normalize()
                idx = len(self.id_map)
                self.id_map[idx] = vector_id
                self.reverse_id_map[vector_id] = idx

                # Add to the index
                vector_array = normalized_vector.reshape(1, -1)
                self.index.add(vector_array)

                return True

            except Exception as e:
                logger.error(f"Error adding vector: {str(e)}")
                # Clean up if there was an error
                if vector_id in self.vectors:
                    del self.vectors[vector_id]
                if vector_id in self.reverse_id_map:
                    idx = self.reverse_id_map[vector_id]
                    del self.reverse_id_map[vector_id]
                    if idx in self.id_map:
                        del self.id_map[idx]
                return False

    def update(self, vector_id: str, vector: EmbeddingVector) -> bool:
        """
        Update a vector in the store.

        Args:
            vector_id (str): Identifier of the vector to update
            vector (EmbeddingVector): New vector

        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            if vector_id not in self.vectors:
                logger.warning(f"Vector ID {vector_id} not found for update.")
                return False

            try:
                # Update the vector
                self.vectors[vector_id] = vector

                # Rebuild the index
                vectors_array = np.zeros((len(self.vectors), self.dimension), dtype=np.float32)
                self.id_map.clear()
                self.reverse_id_map.clear()

                for i, (vid, vec) in enumerate(self.vectors.items()):
                    vectors_array[i] = vec.normalize()
                    self.id_map[i] = vid
                    self.reverse_id_map[vid] = i

                # Reset and rebuild the index
                if FAISS_AVAILABLE:
                    self.index = faiss.IndexFlatIP(self.dimension)
                else:
                    self.index = PythonVectorIndex(self.dimension)

                if len(vectors_array) > 0:
                    self.index.add(vectors_array)

                return True

            except Exception as e:
                logger.error(f"Error updating vector: {str(e)}")
                return False

    def get(self, vector_id: str) -> Optional[EmbeddingVector]:
        """
        Get a vector from the store.

        Args:
            vector_id (str): Identifier of the vector to retrieve

        Returns:
            Optional[EmbeddingVector]: The vector if found, None otherwise
        """
        return self.vectors.get(vector_id)

    def delete(self, vector_id: str) -> bool:
        """
        Delete a vector from the store.

        Args:
            vector_id (str): Identifier of the vector to delete

        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            if vector_id not in self.vectors:
                logger.warning(f"Vector ID {vector_id} not found for deletion.")
                return False

            try:
                # Remove the vector
                del self.vectors[vector_id]

                # Rebuild the index
                vectors_array = np.zeros((len(self.vectors), self.dimension), dtype=np.float32)
                new_id_map = {}
                new_reverse_id_map = {}

                for i, (vid, vec) in enumerate(self.vectors.items()):
                    vectors_array[i] = vec.normalize()
                    new_id_map[i] = vid
                    new_reverse_id_map[vid] = i

                self.id_map = new_id_map
                self.reverse_id_map = new_reverse_id_map

                # Reset and rebuild the index
                if FAISS_AVAILABLE:
                    self.index = faiss.IndexFlatIP(self.dimension)
                else:
                    self.index = PythonVectorIndex(self.dimension)

                if len(vectors_array) > 0:
                    self.index.add(vectors_array)

                return True

            except Exception as e:
                logger.error(f"Error deleting vector: {str(e)}")
                return False

    def search_similar(self, query_vector: Union[np.ndarray, List[float]], top_k: int = 10,
                       threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> List[Tuple[str, float]]:
        """
        Search for vectors similar to the query vector.

        Args:
            query_vector (Union[np.ndarray, List[float]]): Query vector
            top_k (int): Maximum number of results to return
            threshold (float): Minimum similarity threshold

        Returns:
            List[Tuple[str, float]]: List of (vector_id, similarity_score) tuples
        """
        with self.lock:
            if len(self.vectors) == 0:
                return []

            # Convert to numpy array if needed
            if not isinstance(query_vector, np.ndarray):
                query_vector = np.array(query_vector, dtype=np.float32)

            # Normalize the query vector
            norm = np.linalg.norm(query_vector)
            if norm < 1e-10:
                logger.warning("Query vector has near-zero norm")
                return []

            query_vector = query_vector / norm

            # Limit top_k to number of vectors
            top_k = min(top_k, len(self.vectors))

            try:
                # Search for similar vectors
                scores, indices = self.index.search(query_vector.reshape(1, -1), top_k)

                # Build the results
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx < 0 or idx >= len(self.id_map):
                        continue

                    similarity = float(scores[0][i])
                    if similarity < threshold:
                        continue

                    vector_id = self.id_map[idx]
                    results.append((vector_id, similarity))

                return results

            except Exception as e:
                logger.error(f"Error searching for similar vectors: {str(e)}")
                return []

    def get_vector_count(self) -> int:
        """
        Get the number of vectors in the store.

        Returns:
            int: Number of vectors
        """
        return len(self.vectors)

    def get_all_vectors(self) -> Dict[str, EmbeddingVector]:
        """
        Get all vectors in the store.

        Returns:
            Dict[str, EmbeddingVector]: Dictionary of vector_id -> vector
        """
        return self.vectors.copy()


class VectorOperations:
    """
    Collection of static methods for operating on vectors.
    """
    @staticmethod
    def average_vectors(vectors: List[np.ndarray]) -> np.ndarray:
        """
        Calculate the average of multiple vectors.

        Args:
            vectors (List[np.ndarray]): List of vectors to average

        Returns:
            np.ndarray: Average vector

        Raises:
            ValueError: If the vectors list is empty or vectors have different dimensions
        """
        if not vectors:
            raise ValueError("Cannot compute mean of empty list")

        # Convert to numpy arrays if needed
        vectors = [np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v for v in vectors]

        # Check dimensions
        dim = vectors[0].shape[0]
        if not all(v.shape[0] == dim for v in vectors):
            raise ValueError("All vectors must have the same dimension")

        # Calculate mean
        return np.mean(vectors, axis=0)

    @staticmethod
    def project_vector(vector: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
        """
        Project a vector using a projection matrix.

        Args:
            vector (np.ndarray): Vector to project
            projection_matrix (np.ndarray): Projection matrix

        Returns:
            np.ndarray: Projected vector
        """
        return np.dot(projection_matrix, vector)

    @staticmethod
    def transform_vector(vector: np.ndarray, transform_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Apply a transformation function to a vector.

        Args:
            vector (np.ndarray): Vector to transform
            transform_fn (Callable[[np.ndarray], np.ndarray]): Transformation function

        Returns:
            np.ndarray: Transformed vector
        """
        return transform_fn(vector)

    @staticmethod
    def cluster_vectors(vectors: List[np.ndarray], n_clusters: int) -> Tuple[List[int], List[np.ndarray]]:
        """
        Cluster vectors using K-means.

        Args:
            vectors (List[np.ndarray]): List of vectors to cluster
            n_clusters (int): Number of clusters

        Returns:
            Tuple[List[int], List[np.ndarray]]: (cluster_assignments, cluster_centers)
        """
        # Convert to numpy arrays if needed
        vectors = [np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v for v in vectors]

        # Stack vectors
        X = np.vstack(vectors)

        if SKLEARN_AVAILABLE:
            # Use scikit-learn K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_assignments = kmeans.fit_predict(X)
            return list(cluster_assignments), list(kmeans.cluster_centers_)
        else:
            # Use our own K-means implementation
            return VectorOperations._kmeans_cluster(X, n_clusters)

    @staticmethod
    def _kmeans_cluster(X: np.ndarray, n_clusters: int) -> Tuple[List[int], List[np.ndarray]]:
        """
        Pure Python implementation of K-means clustering.

        Args:
            X (np.ndarray): Data matrix, shape (n_samples, n_features)
            n_clusters (int): Number of clusters

        Returns:
            Tuple[List[int], List[np.ndarray]]: (cluster_assignments, cluster_centers)
        """
        import random

        # Initialize centroids using K-means++
        centroids = [X[random.randint(0, X.shape[0]-1)]]

        for _ in range(1, n_clusters):
            # Calculate distance from each point to nearest centroid
            distances = []
            for x in X:
                min_dist = min(np.sum((x - c) ** 2) for c in centroids)
                distances.append(min_dist)

            # Choose next centroid with probability proportional to distance squared
            distances = np.array(distances)
            probs = distances / distances.sum()
            cumprobs = probs.cumsum()
            r = random.random()
            i = np.searchsorted(cumprobs, r)
            centroids.append(X[i])

        # Iterate until convergence or max iterations
        max_iterations = 100
        for _ in range(max_iterations):
            # Assign points to nearest centroid
            clusters = [[] for _ in range(n_clusters)]
            for i, x in enumerate(X):
                distances = [np.sum((x - c) ** 2) for c in centroids]
                cluster_idx = np.argmin(distances)
                clusters[cluster_idx].append(i)

            # Update centroids
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    new_centroid = np.mean([X[i] for i in cluster], axis=0)
                    new_centroids.append(new_centroid)
                else:
                    # If cluster is empty, initialize with a random point
                    new_centroids.append(X[random.randint(0, X.shape[0]-1)])

            # Check for convergence
            if np.allclose(np.array(centroids), np.array(new_centroids), atol=1e-4):
                break

            centroids = new_centroids

        # Assign final clusters
        cluster_assignments = np.zeros(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            distances = [np.sum((x - c) ** 2) for c in centroids]
            cluster_idx = np.argmin(distances)
            cluster_assignments[i] = cluster_idx

        return list(cluster_assignments), centroids