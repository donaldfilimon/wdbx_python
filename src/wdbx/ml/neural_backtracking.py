# wdbx/neural_backtracking.py
import random
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ..core.constants import logger
from ..core.data_structures import Block
from ..storage.blockchain import BlockChainManager
from ..storage.vector_store import VectorStore

# Import diagnostics components
from ..utils.diagnostics import get_performance_profiler
from . import ArrayLike, get_ml_backend

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
        self.profiler = get_performance_profiler() # Get profiler instance

        # Store trace creation timestamps for cleanup
        self.trace_timestamps: Dict[str, float] = {}

        # Handle case where ML backend might not be initialized
        if ml_backend is None:
            logger.info("NeuralBacktracker initialized with no ML backend")
        else:
            logger.info(f"NeuralBacktracker initialized with ML backend: {ml_backend.backend}")
        
        # Set ml_backend reference
        self.ml_backend = ml_backend

    def _mock_activation_analysis(self, vector: ArrayLike) -> Dict[str, float]:
        """Simulate analysis of neural activations within a vector."""
        # Simulate identifying key activation patterns
        analysis = {}
        vector_np = self.ml_backend.to_numpy(vector)
        if np.mean(vector_np[:10]) > 0.6:
            analysis["pattern_A"] = float(np.mean(vector_np[:10]))
        if np.std(vector_np) < 0.2:
            analysis["low_variance"] = float(np.std(vector_np))
        analysis["overall_magnitude"] = float(np.linalg.norm(vector_np))
        time.sleep(random.uniform(0.001, 0.005)) # Simulate computation time
        return analysis

    def _mock_semantic_comparison(self, vec1: ArrayLike, vec2: ArrayLike) -> float:
        """Simulate calculating semantic similarity (cosine distance here)."""
        similarity = self.ml_backend.cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
        time.sleep(random.uniform(0.002, 0.006)) # Simulate computation time
        return float(similarity)

    def _mock_reasoning_error_detection(self, block_sequence: List[Block]) -> Dict[str, Any]:
        """Simulate detecting logical fallacies or inconsistencies in a block sequence."""
        errors = {}
        if len(block_sequence) < 2:
            return errors

        # Simple check: Look for large jumps in embedding space between consecutive blocks
        vec1 = self.ml_backend.create_tensor(block_sequence[-2].embeddings[0].vector)
        vec2 = self.ml_backend.create_tensor(block_sequence[-1].embeddings[0].vector)
        similarity = self._mock_semantic_comparison(vec1, vec2)
        if similarity < 0.4: # Arbitrary threshold for potential inconsistency
            errors["potential_jump"] = {"from_block": block_sequence[-2].block_id, "to_block": block_sequence[-1].block_id, "similarity": similarity}
            
        # Simulate checking for contradictions (requires more complex logic)
        # if "positive_statement" in block_sequence[-2].data and "negative_statement" in block_sequence[-1].data:
        #     errors["contradiction"] = {"blocks": [block_sequence[-2].block_id, block_sequence[-1].block_id]}
            
        time.sleep(random.uniform(0.005, 0.015)) # Simulate computation time
        return errors

    def trace_activation(
            self,
            query_vector: ArrayLike,
            threshold: float = 0.6,
            top_k: int = 20) -> str:
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

                # Find similar vectors - the vector store now handles different array types
                similar_vectors = self.vector_store.search_similar(
                    query_vector, top_k=top_k, threshold=threshold)

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

                logger.debug(f"Created activation trace {trace_id} with {len(activations)} activations")
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
                    raise ValueError(f"Invalid trace ID: {trace_id}")

                # Sort blocks by activation strength
                block_ids = sorted(trace.keys(), key=lambda x: trace[x], reverse=True)

                drift_points = []
                prev_activation = None
                for block_id in block_ids:
                    activation = trace[block_id]
                    if prev_activation is not None and (prev_activation - activation) > threshold:
                        drift_points.append(block_id)
                    prev_activation = activation

                logger.debug(f"Detected {len(drift_points)} semantic drift points for trace {trace_id}")
                return drift_points

    def identify_reasoning_errors(
            self, trace_id: str, error_patterns: List[Dict[str, Any]]) -> List[str]:
        """
        Identify potential reasoning errors based on predefined patterns.

        Args:
            trace_id: ID of the activation trace to analyze
            error_patterns: List of pattern dictionaries defining error conditions

        Returns:
            List of block IDs matching error patterns
        """
        with self.profiler.profile_block("NeuralBacktracker.identify_reasoning_errors"):
            with self.lock:
                trace = self.activation_traces.get(trace_id)
                if not trace:
                    raise ValueError(f"Invalid trace ID: {trace_id}")

                error_blocks = []
                for block_id in trace.keys():
                    block = self.block_chain_manager.get_block(block_id)
                    if not block:
                        continue

                    for pattern in error_patterns:
                        # Check if block data matches pattern
                        data_match = all(
                            block.data.get(k) == v for k,
                            v in pattern.get(
                                "data",
                                {}).items())

                        # Check if any embedding metadata matches pattern
                        embedding_match = False
                        if "embedding_metadata" in pattern:
                            for embedding in block.embeddings:
                                if all(
                                        embedding.metadata.get(k) == v for k,
                                        v in pattern["embedding_metadata"].items()):
                                    embedding_match = True
                                    break
                        else:
                            embedding_match = True  # No metadata criteria specified

                        if data_match and embedding_match:
                            error_blocks.append(block_id)
                            break

                logger.debug(
                    f"Identified {
                        len(error_blocks)} blocks with reasoning errors in trace {trace_id}")
                return error_blocks

    def compare_activations(self, trace_id1: str, trace_id2: str) -> float:
        """
        Compare two activation traces to determine similarity.

        Args:
            trace_id1: First trace ID
            trace_id2: Second trace ID

        Returns:
            Similarity score between the two traces (0-1)
        """
        with self.lock:
            trace1 = self.activation_traces.get(trace_id1)
            trace2 = self.activation_traces.get(trace_id2)

            if not trace1 or not trace2:
                raise ValueError("Invalid trace ID(s)")

            # Get all block IDs from both traces
            all_blocks = set(trace1.keys()) | set(trace2.keys())

            if not all_blocks:
                return 0.0

            # Create vectors from activation values
            vec1 = [trace1.get(block_id, 0.0) for block_id in all_blocks]
            vec2 = [trace2.get(block_id, 0.0) for block_id in all_blocks]

            # Convert to the preferred array format for the ML backend
            vec1_array = self.ml_backend.to_preferred(vec1)
            vec2_array = self.ml_backend.to_preferred(vec2)

            # Calculate cosine similarity
            similarity = self.ml_backend.cosine_similarity(vec1_array, vec2_array)

            return float(similarity)

    def merge_traces(self, trace_ids: List[str], weights: Optional[List[float]] = None) -> str:
        """
        Merge multiple activation traces into a new trace, optionally with weights.

        Args:
            trace_ids: List of trace IDs to merge
            weights: Optional weights for each trace (must match trace_ids length)

        Returns:
            New trace ID for the merged trace
        """
        with self.lock:
            if not trace_ids:
                raise ValueError("No trace IDs provided")

            # Validate trace IDs
            traces = []
            for tid in trace_ids:
                trace = self.activation_traces.get(tid)
                if not trace:
                    raise ValueError(f"Invalid trace ID: {tid}")
                traces.append(trace)

            # Validate weights
            if weights is None:
                weights = [1.0] * len(traces)
            elif len(weights) != len(traces):
                raise ValueError("Number of weights must match number of traces")

            # Create a new merged trace
            merged_activations: Dict[str, float] = {}

            # Get all unique block IDs across traces
            all_blocks = set()
            for trace in traces:
                all_blocks.update(trace.keys())

            # Compute weighted average of activations for each block
            total_weight = sum(weights)
            if total_weight == 0:
                raise ValueError("Sum of weights cannot be zero")

            for block_id in all_blocks:
                weighted_sum = 0.0
                for i, trace in enumerate(traces):
                    if block_id in trace:
                        weighted_sum += trace[block_id] * weights[i]
                merged_activations[block_id] = weighted_sum / total_weight

            # Create and store the new trace
            new_trace_id = str(uuid.uuid4())
            self.activation_traces[new_trace_id] = merged_activations
            self.trace_timestamps[new_trace_id] = time.time()

            logger.debug(
                f"Created merged trace {new_trace_id} from {
                    len(trace_ids)} traces with {
                    len(merged_activations)} activations")
            return new_trace_id

    def create_semantic_path(self, blocks: List[str]) -> str:
        """
        Create a semantic path through specified blocks.

        Args:
            blocks: List of block IDs to include in the path

        Returns:
            path_id: Unique identifier for the semantic path
        """
        with self.lock:
            # Validate all block IDs
            invalid_blocks = []
            for block_id in blocks:
                if not self.block_chain_manager.get_block(block_id):
                    invalid_blocks.append(block_id)

            if invalid_blocks:
                raise ValueError(f"Invalid block IDs: {', '.join(invalid_blocks)}")

            path_id = str(uuid.uuid4())
            self.semantic_paths[path_id] = blocks
            logger.debug(f"Created semantic path {path_id} with {len(blocks)} blocks")
            return path_id

    def follow_semantic_path(self, path_id: str) -> List[Block]:
        """
        Retrieve blocks along a semantic path.

        Args:
            path_id: ID of the semantic path to follow

        Returns:
            List of Block objects along the path
        """
        with self.lock:
            path = self.semantic_paths.get(path_id)
            if not path:
                raise ValueError(f"Invalid path ID: {path_id}")

            blocks = []
            for block_id in path:
                block = self.block_chain_manager.get_block(block_id)
                if block:
                    blocks.append(block)

            return blocks

    def get_activation_vector(self, trace_id: str) -> ArrayLike:
        """
        Convert an activation trace to a vector representation.

        Args:
            trace_id: ID of the trace to convert

        Returns:
            Vector representation of the activation trace
        """
        with self.lock:
            trace = self.activation_traces.get(trace_id)
            if not trace:
                raise ValueError(f"Invalid trace ID: {trace_id}")

            # Sort block IDs for consistent representation
            block_ids = sorted(trace.keys())

            # Create a vector from activation values
            vector = [trace[block_id] for block_id in block_ids]

            # Convert to ML backend's preferred format
            return self.ml_backend.to_preferred(vector)

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance statistics.

        Returns:
            A dictionary with performance statistics
        """
        with self.lock:
            stats = {k: v.copy() for k, v in self.performance_stats.items()}
            # Add average times
            for operation, metrics in stats.items():
                if metrics["calls"] > 0:
                    metrics["avg_time"] = metrics["total_time"] / metrics["calls"]
                else:
                    metrics["avg_time"] = 0.0
            return stats

    def clear_traces(self, older_than: Optional[float] = None) -> int:
        """
        Clear activation traces.

        Args:
            older_than: Optional time in seconds; if provided, only traces older than this are cleared

        Returns:
            Number of traces cleared
        """
        with self.lock:
            if older_than is None:
                count = len(self.activation_traces)
                self.activation_traces.clear()
                self.trace_timestamps.clear()
                return count

            # Remove traces older than the specified time
            traces_to_remove = []
            current_time = time.time()

            for trace_id, timestamp in self.trace_timestamps.items():
                if current_time - timestamp > older_than:
                    traces_to_remove.append(trace_id)

            for trace_id in traces_to_remove:
                if trace_id in self.activation_traces:
                    del self.activation_traces[trace_id]
                if trace_id in self.trace_timestamps:
                    del self.trace_timestamps[trace_id]

            return len(traces_to_remove)

    def check_results(self, results: List[Dict[str, Any]]) -> bool:
        """Check if the results meet the required accuracy."""
        self.logger.debug("Checking results for accuracy")
        return len(results) > 0

    def log_traces(self, traces):
        """Log traced paths for debugging and auditing."""
        self.logger.debug("Logging neural backtracking traces")
