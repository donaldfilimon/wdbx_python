# wdbx/vector_store.py
import threading
import numpy as np
import faiss
from typing import Dict, List, Optional, Tuple
from wdbx.data_structures import EmbeddingVector
from wdbx.constants import logger
from config import VECTOR_DIMENSION, DEFAULT_SIMILARITY_THRESHOLD

class VectorStore:
    """
    Optimized storage for embedding vectors using FAISS for cosine similarity search.
    Note: Updates and deletions currently rebuild the entire index. Future work could
    implement incremental updates.
    """
    def __init__(self, dimension: int = VECTOR_DIMENSION) -> None:
        self.dimension = dimension
        self.vectors: Dict[str, EmbeddingVector] = {}
        self.index = faiss.IndexFlatIP(dimension)
        self.id_map: Dict[int, str] = {}
        self.reverse_id_map: Dict[str, int] = {}
        self.lock = threading.RLock()

    def add(self, vector_id: str, vector: EmbeddingVector) -> bool:
        with self.lock:
            if vector_id in self.vectors:
                logger.error(f"Vector ID {vector_id} already exists.")
                return False
            self.vectors[vector_id] = vector
            normalized_vector = vector.normalize()
            idx = len(self.id_map)
            self.id_map[idx] = vector_id
            self.reverse_id_map[vector_id] = idx
            self.index.add(normalized_vector.reshape(1, -1))
            return True

    def update(self, vector_id: str, vector: EmbeddingVector) -> bool:
        with self.lock:
            if vector_id not in self.vectors:
                logger.error(f"Vector ID {vector_id} not found for update.")
                return False
            self.vectors[vector_id] = vector
            # Rebuild index (inefficient; TODO: implement incremental update)
            vectors_array = np.zeros((len(self.vectors), self.dimension), dtype=np.float32)
            self.id_map.clear()
            self.reverse_id_map.clear()
            for i, (vid, vec) in enumerate(self.vectors.items()):
                vectors_array[i] = vec.normalize()
                self.id_map[i] = vid
                self.reverse_id_map[vid] = i
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(vectors_array)
            return True

    def get(self, vector_id: str) -> Optional[EmbeddingVector]:
        return self.vectors.get(vector_id)

    def delete(self, vector_id: str) -> bool:
        with self.lock:
            if vector_id not in self.vectors:
                logger.error(f"Vector ID {vector_id} not found for deletion.")
                return False
            del self.vectors[vector_id]
            # Rebuild index after deletion
            vectors_array = np.zeros((len(self.vectors), self.dimension), dtype=np.float32)
            new_id_map = {}
            new_reverse_id_map = {}
            for i, (vid, vec) in enumerate(self.vectors.items()):
                vectors_array[i] = vec.normalize()
                new_id_map[i] = vid
                new_reverse_id_map[vid] = i
            self.id_map = new_id_map
            self.reverse_id_map = new_reverse_id_map
            self.index = faiss.IndexFlatIP(self.dimension)
            if len(self.vectors) > 0:
                self.index.add(vectors_array)
            return True

    def search_similar(self, query_vector: np.ndarray, top_k: int = 10,
                       threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> List[Tuple[str, float]]:
        if len(self.vectors) == 0:
            return []
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(query_vector)
        if norm < 1e-10:
            return []
        query_vector = query_vector / norm
        top_k = min(top_k, len(self.vectors))
        scores, indices = self.index.search(query_vector.reshape(1, -1), top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue
            similarity = scores[0][i]
            if similarity < threshold:
                continue
            vector_id = self.id_map[idx]
            results.append((vector_id, float(similarity)))
        return results


class VectorOperations:
    """
    Collection of static methods for operating on vectors.
    """
    @staticmethod
    def average_vectors(vectors: List[np.ndarray]) -> np.ndarray:
        if not vectors:
            raise ValueError("Cannot average an empty list of vectors.")
        vectors = [np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v for v in vectors]
        dim = vectors[0].shape[0]
        for v in vectors[1:]:
            if v.shape[0] != dim:
                raise ValueError("Vector dimension mismatch.")
        return np.mean(vectors, axis=0)

    @staticmethod
    def project_vector(vector: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
        return np.dot(projection_matrix, vector)

    @staticmethod
    def transform_vector(vector: np.ndarray, transform_fn) -> np.ndarray:
        return transform_fn(vector)

    @staticmethod
    def cluster_vectors(vectors: List[np.ndarray], n_clusters: int) -> Tuple[List[int], List[np.ndarray]]:
        from sklearn.cluster import KMeans
        vectors = [np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v for v in vectors]
        X = np.vstack(vectors)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_assignments = kmeans.fit_predict(X)
        return list(cluster_assignments), list(kmeans.cluster_centers_)

    @staticmethod
    def cluster_vectors_faiss(vectors: List[np.ndarray], n_clusters: int) -> Tuple[List[int], List[np.ndarray]]:
        vectors = [np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v for v in vectors]
        X = np.vstack(vectors)
        d = X.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(X)
        _, I = index.search(X, n_clusters)
        cluster_assignments = I.flatten().tolist()
        cluster_centers = [np.mean(X[I == i], axis=0) for i in range(n_clusters)]
        return cluster_assignments, cluster_centers

