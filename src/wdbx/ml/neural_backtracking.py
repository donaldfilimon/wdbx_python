# wdbx/ml/neural_backtracking.py
"""
Neural backtracking implementation for WDBX.

This module provides methods for tracing activation patterns and detecting semantic drift
by tracking neural network activations to identify reasoning paths and potential errors.
"""

import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ..core.data_structures import Block
from ..storage.blockchain import BlockChainManager
from ..storage.vector_store import VectorStore

# Import diagnostics components
from ..utils.diagnostics import get_performance_profiler
from . import JAX_AVAILABLE, TORCH_AVAILABLE, ArrayLike
from . import logger as ml_logger
from .backend import get_ml_backend

# Get ML backend for optimized operations
ml_backend = get_ml_backend()


class NeuralBacktracker:
    """
    Provides methods for tracing activation patterns and detecting semantic drift.
    Enables backtracking through neural network activations to identify reasoning paths,
    semantic drift points, and potential reasoning errors.

    Optimized to use JAX or PyTorch for accelerated vector operations when available.
    """

    def __init__(self, block_chain_manager: BlockChainManager, vector_store: VectorStore) -> None:
        """
        Initialize the neural backtracker.

        Args:
            block_chain_manager: Manager for blockchain operations
            vector_store: Store for embedding vectors
        """
        self.block_chain_manager = block_chain_manager
        self.vector_store = vector_store
        self.activation_traces: Dict[str, Dict[str, float]] = {}
        self.semantic_paths: Dict[str, List[str]] = {}
        self.lock = threading.RLock()
        self.profiler = get_performance_profiler()  # Get profiler instance

        # Store trace creation timestamps for cleanup
        self.trace_timestamps: Dict[str, float] = {}

        # Handle case where ML backend might not be initialized
        if ml_backend is None:
            ml_logger.warning(
                "NeuralBacktracker initialized with no ML backend, using fallback methods"
            )
        else:
            ml_logger.info(
                f"NeuralBacktracker initialized with ML backend: {ml_backend.selected_backend}"
            )

        # Set ml_backend reference
        self.ml_backend = ml_backend

        # Initialize backend-specific operations
        self._initialize_backend_ops()

    def _initialize_backend_ops(self) -> None:
        """Initialize backend-specific operations for vector analysis."""
        self.backend_type = self.ml_backend.selected_backend if self.ml_backend else "numpy"

        # Set up JIT compilation for JAX functions if available
        if self.backend_type == "jax" and JAX_AVAILABLE:
            import jax
            import jax.numpy as jnp

            # JIT-compile common operations for performance
            @jax.jit
            def _jax_analyze_vector(vec):
                """Perform various analyses on a vector using JAX."""
                mean_val = jnp.mean(vec[:10])
                std_val = jnp.std(vec)
                magnitude = jnp.linalg.norm(vec)
                return mean_val, std_val, magnitude

            @jax.jit
            def _jax_compare_vectors(vec1, vec2):
                """Calculate vector similarity using JAX."""
                vec1_norm = vec1 / (jnp.linalg.norm(vec1) + 1e-8)
                vec2_norm = vec2 / (jnp.linalg.norm(vec2) + 1e-8)
                return jnp.dot(vec1_norm, vec2_norm)

            self._jax_analyze_vector = _jax_analyze_vector
            self._jax_compare_vectors = _jax_compare_vectors

        # Set up PyTorch operations if available
        elif self.backend_type == "pytorch" and TORCH_AVAILABLE:
            import torch

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _analyze_activation(self, vector: ArrayLike) -> Dict[str, float]:
        """
        Analyze neural activations within a vector.

        Args:
            vector: Vector to analyze

        Returns:
            Dictionary of analysis results
        """
        analysis = {}

        # Convert vector to the appropriate format for the backend
        if self.backend_type == "jax" and JAX_AVAILABLE:
            import jax.numpy as jnp

            # Use pre-compiled JAX operations
            vector_jax = vector if isinstance(vector, jnp.ndarray) else jnp.array(vector)
            mean_val, std_val, magnitude = self._jax_analyze_vector(vector_jax)

            # Convert results to Python scalars
            if mean_val > 0.6:
                analysis["pattern_A"] = float(mean_val)
            if std_val < 0.2:
                analysis["low_variance"] = float(std_val)
            analysis["overall_magnitude"] = float(magnitude)

        elif self.backend_type == "pytorch" and TORCH_AVAILABLE:
            import torch

            # Use PyTorch operations
            vector_torch = (
                vector
                if isinstance(vector, torch.Tensor)
                else torch.tensor(vector, device=self.device, dtype=torch.float32)
            )

            mean_val = torch.mean(vector_torch[:10]).item()
            std_val = torch.std(vector_torch).item()
            magnitude = torch.norm(vector_torch).item()

            if mean_val > 0.6:
                analysis["pattern_A"] = float(mean_val)
            if std_val < 0.2:
                analysis["low_variance"] = float(std_val)
            analysis["overall_magnitude"] = float(magnitude)

        else:
            # Fallback to NumPy
            vector_np = self.ml_backend.to_numpy(vector)

            if np.mean(vector_np[:10]) > 0.6:
                analysis["pattern_A"] = float(np.mean(vector_np[:10]))
            if np.std(vector_np) < 0.2:
                analysis["low_variance"] = float(np.std(vector_np))
            analysis["overall_magnitude"] = float(np.linalg.norm(vector_np))

        return analysis

    def _calculate_semantic_similarity(self, vec1: ArrayLike, vec2: ArrayLike) -> float:
        """
        Calculate semantic similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score (higher is more similar)
        """
        # Use ml_backend's cosine_similarity for optimal performance
        try:
            similarity = self.ml_backend.cosine_similarity(vec1, vec2)
            return float(similarity)
        except Exception as e:
            ml_logger.warning(
                f"Error using ml_backend cosine_similarity: {e}. Using fallback method."
            )

            # Fallback to NumPy implementation
            vec1_np = self.ml_backend.to_numpy(vec1)
            vec2_np = self.ml_backend.to_numpy(vec2)

            vec1_np = vec1_np.reshape(-1)
            vec2_np = vec2_np.reshape(-1)

            # Normalize vectors
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            vec1_normalized = vec1_np / norm1
            vec2_normalized = vec2_np / norm2

            # Calculate cosine similarity
            return float(np.dot(vec1_normalized, vec2_normalized))

    def _detect_reasoning_errors(self, block_sequence: List[Block]) -> Dict[str, Any]:
        """
        Detect logical inconsistencies or errors in a block sequence.

        Args:
            block_sequence: Sequence of blocks to analyze

        Returns:
            Dictionary containing detected errors
        """
        errors = {}
        if len(block_sequence) < 2:
            return errors

        # Look for large semantic jumps between consecutive blocks
        for i in range(1, len(block_sequence)):
            prev_block = block_sequence[i - 1]
            curr_block = block_sequence[i]

            # Skip if either block doesn't have embeddings
            if not prev_block.embeddings or not curr_block.embeddings:
                continue

            # Get the primary embedding vectors
            vec1 = prev_block.embeddings[0].vector
            vec2 = curr_block.embeddings[0].vector

            # Calculate similarity
            similarity = self._calculate_semantic_similarity(vec1, vec2)

            # Check for potential semantic jump
            if similarity < 0.4:  # Threshold for potential inconsistency
                errors[f"jump_{prev_block.block_id}_{curr_block.block_id}"] = {
                    "type": "semantic_jump",
                    "from_block": prev_block.block_id,
                    "to_block": curr_block.block_id,
                    "similarity": similarity,
                    "severity": "high" if similarity < 0.2 else "medium",
                }

        return errors

    def trace_activation(
        self, query_vector: ArrayLike, threshold: float = 0.6, top_k: int = 20
    ) -> str:
        """
        Traces activation patterns based on a query vector.

        Args:
            query_vector: Vector to trace activations for
            threshold: Minimum similarity threshold for activations
            top_k: Maximum number of similar vectors to consider

        Returns:
            trace_id: Unique identifier for the activation trace
        """
        with self.profiler.profile_block("NeuralBacktracker.trace_activation"):
            with self.lock:
                trace_id = str(uuid.uuid4())
                activations: Dict[str, float] = {}

                try:
                    # Find similar vectors
                    similar_vectors = self.vector_store.search_similar(
                        query_vector, top_k=top_k, threshold=threshold
                    )

                    # Create a fast lookup for blocks containing specific vectors
                    vector_to_blocks: Dict[str, Set[str]] = {}
                    for block_id, block in self.block_chain_manager.blocks.items():
                        for embedding in block.embeddings:
                            vector_ids = embedding.metadata.get("vector_ids", [])
                            for vid in vector_ids:
                                if vid not in vector_to_blocks:
                                    vector_to_blocks[vid] = set()
                                vector_to_blocks[vid].add(block_id)

                    # Calculate activations
                    for vector_id, similarity in similar_vectors:
                        blocks_with_vector = vector_to_blocks.get(vector_id, set())
                        for block_id in blocks_with_vector:
                            activations[block_id] = max(activations.get(block_id, 0.0), similarity)

                    self.activation_traces[trace_id] = activations
                    self.trace_timestamps[trace_id] = time.time()

                    ml_logger.debug(
                        f"Created activation trace {trace_id} with {len(activations)} activations"
                    )
                except Exception as e:
                    ml_logger.error(f"Error during activation tracing: {e}")
                    # Return empty trace ID with warning
                    self.activation_traces[trace_id] = {}
                    self.trace_timestamps[trace_id] = time.time()
                    ml_logger.warning(f"Created empty activation trace {trace_id} due to error")

                return trace_id

    def detect_semantic_drift(self, trace_id: str, threshold: float = 0.2) -> List[str]:
        """
        Detect points of semantic drift in an activation trace.

        Args:
            trace_id: ID of the activation trace to analyze
            threshold: Activation difference threshold for detecting drift

        Returns:
            List of block IDs where semantic drift occurs
        """
        with self.profiler.profile_block("NeuralBacktracker.detect_semantic_drift"):
            with self.lock:
                trace = self.activation_traces.get(trace_id)
                if not trace:
                    ml_logger.warning(f"Invalid trace ID: {trace_id}")
                    return []

                # Sort blocks by activation strength
                block_ids = sorted(trace.keys(), key=lambda x: trace[x], reverse=True)

                drift_points = []
                prev_activation = None
                for block_id in block_ids:
                    activation = trace[block_id]
                    if prev_activation is not None and (prev_activation - activation) > threshold:
                        drift_points.append(block_id)
                    prev_activation = activation

                ml_logger.debug(
                    f"Detected {len(drift_points)} semantic drift points for trace {trace_id}"
                )
                return drift_points

    def identify_reasoning_errors(
        self, trace_id: str, error_patterns: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Identify potential reasoning errors based on predefined patterns.

        Args:
            trace_id: ID of the activation trace to analyze
            error_patterns: Optional list of pattern dictionaries defining error conditions

        Returns:
            Dictionary of identified errors by block ID
        """
        with self.profiler.profile_block("NeuralBacktracker.identify_reasoning_errors"):
            with self.lock:
                trace = self.activation_traces.get(trace_id)
                if not trace:
                    ml_logger.warning(f"Invalid trace ID: {trace_id}")
                    return {}

                # Default error patterns if none provided
                if error_patterns is None:
                    error_patterns = [
                        {"name": "high_to_low_jump", "type": "activation_drop", "threshold": 0.4},
                        {"name": "anomaly", "type": "outlier", "z_score_threshold": 2.0},
                    ]

                errors = {}

                # Get activated blocks sorted by activation strength
                block_ids_sorted = sorted(trace.keys(), key=lambda x: trace[x], reverse=True)

                # Skip if we don't have enough blocks
                if len(block_ids_sorted) < 2:
                    return errors

                # Get actual Block objects
                blocks = []
                for block_id in block_ids_sorted:
                    block = self.block_chain_manager.get_block(block_id)
                    if block is not None:
                        blocks.append(block)

                # Check activation patterns
                activations = [trace[block_id] for block_id in block_ids_sorted]
                mean_activation = sum(activations) / len(activations)
                stddev = (
                    sum((a - mean_activation) ** 2 for a in activations) / len(activations)
                ) ** 0.5

                # Apply error detection methods
                for pattern in error_patterns:
                    if pattern["type"] == "activation_drop":
                        threshold = pattern.get("threshold", 0.4)
                        for i in range(1, len(blocks)):
                            prev_activation = trace[blocks[i - 1].block_id]
                            curr_activation = trace[blocks[i].block_id]
                            if (prev_activation - curr_activation) > threshold:
                                errors[blocks[i].block_id] = {
                                    "error_type": "activation_drop",
                                    "severity": (
                                        "high"
                                        if (prev_activation - curr_activation) > 0.6
                                        else "medium"
                                    ),
                                    "prev_block": blocks[i - 1].block_id,
                                    "activation_drop": prev_activation - curr_activation,
                                }

                    elif pattern["type"] == "outlier":
                        z_threshold = pattern.get("z_score_threshold", 2.0)
                        for i, block in enumerate(blocks):
                            activation = trace[block.block_id]
                            z_score = (activation - mean_activation) / (stddev if stddev > 0 else 1)
                            if abs(z_score) > z_threshold:
                                errors[block.block_id] = {
                                    "error_type": "outlier",
                                    "severity": "high" if abs(z_score) > 3.0 else "medium",
                                    "z_score": z_score,
                                    "activation": activation,
                                }

                # Extend with semantic analysis if we have enough blocks
                if len(blocks) >= 2:
                    seq_errors = self._detect_reasoning_errors(blocks)
                    errors.update(seq_errors)

                return errors

    def compare_activations(self, trace_id1: str, trace_id2: str) -> float:
        """
        Compare two activation traces to determine similarity.

        Args:
            trace_id1: ID of first activation trace
            trace_id2: ID of second activation trace

        Returns:
            Similarity score between traces (0.0 to 1.0)
        """
        with self.profiler.profile_block("NeuralBacktracker.compare_activations"):
            with self.lock:
                trace1 = self.activation_traces.get(trace_id1)
                trace2 = self.activation_traces.get(trace_id2)

                if not trace1 or not trace2:
                    ml_logger.warning(f"Invalid trace ID(s): {trace_id1}, {trace_id2}")
                    return 0.0

                # Create vectors from traces for comparison
                all_blocks = set(trace1.keys()) | set(trace2.keys())

                if not all_blocks:
                    return 0.0  # No blocks to compare

                # Create vectors with activation values for each trace
                vec1 = np.zeros(len(all_blocks))
                vec2 = np.zeros(len(all_blocks))

                for i, block_id in enumerate(all_blocks):
                    vec1[i] = trace1.get(block_id, 0.0)
                    vec2[i] = trace2.get(block_id, 0.0)

                # Calculate cosine similarity between vectors
                return self._calculate_semantic_similarity(vec1, vec2)

    def merge_traces(self, trace_ids: List[str], weights: Optional[List[float]] = None) -> str:
        """
        Merge multiple activation traces into a single trace.

        Args:
            trace_ids: List of trace IDs to merge
            weights: Optional weights for each trace (must match length of trace_ids)

        Returns:
            ID of the merged trace
        """
        with self.profiler.profile_block("NeuralBacktracker.merge_traces"):
            with self.lock:
                # Validate inputs
                if not trace_ids:
                    ml_logger.warning("No trace IDs provided for merging")
                    return str(uuid.uuid4())  # Return a new empty trace

                # If no weights provided, use equal weights
                if weights is None:
                    weights = [1.0 / len(trace_ids)] * len(trace_ids)
                elif len(weights) != len(trace_ids):
                    ml_logger.warning(
                        f"Number of weights ({len(weights)}) does not match number of traces ({len(trace_ids)}). Using equal weights."
                    )
                    weights = [1.0 / len(trace_ids)] * len(trace_ids)

                # Get all traces and normalize weights
                traces = []
                valid_weights = []
                valid_trace_ids = []

                weight_sum = sum(weights)
                if weight_sum == 0:
                    ml_logger.warning("Sum of weights is zero. Using equal weights.")
                    weights = [1.0 / len(trace_ids)] * len(trace_ids)
                    weight_sum = 1.0

                for tid, weight in zip(trace_ids, weights):
                    trace = self.activation_traces.get(tid)
                    if trace:
                        traces.append(trace)
                        valid_weights.append(weight / weight_sum)  # Normalize weight
                        valid_trace_ids.append(tid)

                if not traces:
                    ml_logger.warning("No valid traces found for merging")
                    return str(uuid.uuid4())  # Return a new empty trace

                # Create merged trace
                merged_trace: Dict[str, float] = {}
                new_trace_id = str(uuid.uuid4())

                # Collect all block IDs from all traces
                all_blocks = set()
                for trace in traces:
                    all_blocks.update(trace.keys())

                # Calculate weighted average of activations for each block
                for block_id in all_blocks:
                    weighted_sum = 0.0
                    for trace, weight in zip(traces, valid_weights):
                        weighted_sum += trace.get(block_id, 0.0) * weight
                    merged_trace[block_id] = weighted_sum

                # Store the merged trace
                self.activation_traces[new_trace_id] = merged_trace
                self.trace_timestamps[new_trace_id] = time.time()

                ml_logger.debug(
                    f"Created merged trace {new_trace_id} from traces {valid_trace_ids} with {len(merged_trace)} activations"
                )
                return new_trace_id

    def create_semantic_path(self, blocks: List[str]) -> str:
        """
        Create a semantic path from a list of block IDs.

        Args:
            blocks: List of block IDs forming the path

        Returns:
            path_id: Unique identifier for the semantic path
        """
        with self.lock:
            # Validate input
            if not blocks:
                ml_logger.warning("No blocks provided for semantic path creation")
                return ""

            # Check if blocks exist
            existing_blocks = []
            for block_id in blocks:
                if self.block_chain_manager.get_block(block_id) is not None:
                    existing_blocks.append(block_id)
                else:
                    ml_logger.warning(f"Block {block_id} not found, excluding from path")

            if not existing_blocks:
                ml_logger.warning("No valid blocks found for semantic path creation")
                return ""

            # Create path ID
            path_id = str(uuid.uuid4())
            self.semantic_paths[path_id] = existing_blocks

            ml_logger.debug(f"Created semantic path {path_id} with {len(existing_blocks)} blocks")
            return path_id

    def follow_semantic_path(self, path_id: str) -> List[Block]:
        """
        Follow a semantic path and return the blocks in order.

        Args:
            path_id: ID of the semantic path to follow

        Returns:
            List of blocks in the path
        """
        with self.lock:
            # Get path
            block_ids = self.semantic_paths.get(path_id)
            if not block_ids:
                ml_logger.warning(f"Invalid path ID: {path_id}")
                return []

            # Get blocks
            blocks = []
            for block_id in block_ids:
                block = self.block_chain_manager.get_block(block_id)
                if block is not None:
                    blocks.append(block)
                else:
                    ml_logger.warning(f"Block {block_id} in path {path_id} not found")

            return blocks

    def get_activation_vector(self, trace_id: str) -> ArrayLike:
        """
        Get a representative vector for an activation trace.

        Args:
            trace_id: ID of the activation trace

        Returns:
            Vector representation of the trace
        """
        with self.lock:
            trace = self.activation_traces.get(trace_id)
            if not trace:
                ml_logger.warning(f"Invalid trace ID: {trace_id}")
                return np.zeros(self.ml_backend.vector_dimension, dtype=np.float32)

            # Get top activated blocks
            sorted_blocks = sorted(trace.items(), key=lambda x: x[1], reverse=True)
            top_blocks = sorted_blocks[: min(5, len(sorted_blocks))]

            if not top_blocks:
                return np.zeros(self.ml_backend.vector_dimension, dtype=np.float32)

            # Get embeddings for these blocks
            vectors = []
            weights = []
            for block_id, activation in top_blocks:
                block = self.block_chain_manager.get_block(block_id)
                if block is not None and block.embeddings:
                    vector = block.embeddings[0].vector
                    vectors.append(self.ml_backend.to_numpy(vector))
                    weights.append(activation)

            if not vectors:
                return np.zeros(self.ml_backend.vector_dimension, dtype=np.float32)

            # Compute weighted average
            vectors = np.array(vectors)
            weights = np.array(weights) / sum(weights)  # Normalize weights

            weighted_avg = np.sum(vectors * weights[:, np.newaxis], axis=0)
            return weighted_avg

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance statistics for the neural backtracker.

        Returns:
            Dictionary of performance metrics
        """
        with self.lock:
            stats = {}

            # Get profiler stats if available
            if self.profiler:
                stats["profiler"] = self.profiler.get_stats().get("NeuralBacktracker", {})

            # Add trace stats
            stats["traces"] = {
                "count": len(self.activation_traces),
                "average_size": sum(len(t) for t in self.activation_traces.values())
                / max(1, len(self.activation_traces)),
                "total_memory_kb": sum(
                    sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in t.items())
                    for t in self.activation_traces.values()
                )
                / 1024,
            }

            # Add path stats
            stats["paths"] = {
                "count": len(self.semantic_paths),
                "average_length": sum(len(p) for p in self.semantic_paths.values())
                / max(1, len(self.semantic_paths)),
            }

            return stats

    def clear_traces(self, older_than: Optional[float] = None) -> int:
        """
        Clear old activation traces to free memory.

        Args:
            older_than: Clear traces older than this timestamp (seconds since epoch)
                        If None, uses 24 hours ago as default

        Returns:
            Number of traces cleared
        """
        with self.lock:
            if older_than is None:
                older_than = time.time() - (24 * 60 * 60)  # 24 hours ago

            # Find traces to clear
            trace_ids_to_clear = [
                tid for tid, timestamp in self.trace_timestamps.items() if timestamp < older_than
            ]

            # Clear traces
            for tid in trace_ids_to_clear:
                self.activation_traces.pop(tid, None)
                self.trace_timestamps.pop(tid, None)

            ml_logger.info(f"Cleared {len(trace_ids_to_clear)} old activation traces")
            return len(trace_ids_to_clear)

    def check_results(self, results: List[Dict[str, Any]]) -> bool:
        """
        Check if results are valid based on internal heuristics.

        Args:
            results: List of result dictionaries to validate

        Returns:
            True if results are valid, False otherwise
        """
        # This is a placeholder for more sophisticated validation
        if not results:
            return False

        # Basic sanity checks
        for result in results:
            if not isinstance(result, dict):
                return False

        return True

    def log_traces(self, traces: Any) -> None:
        """
        Log trace information for debugging purposes.

        Args:
            traces: Trace data to log
        """
        ml_logger.debug(f"NeuralBacktracker traces: {traces}")


import sys
