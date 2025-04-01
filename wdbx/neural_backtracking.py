# wdbx/neural_backtracking.py
import uuid
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from wdbx.data_structures import Block
from wdbx.vector_store import VectorStore
from wdbx.blockchain import BlockChainManager
from wdbx.constants import logger

class NeuralBacktracker:
    """
    Provides methods for tracing activation patterns and detecting semantic drift.
    Enables backtracking through neural network activations to identify reasoning paths,
    semantic drift points, and potential reasoning errors.
    """
    def __init__(self, block_chain_manager: BlockChainManager, vector_store: VectorStore) -> None:
            self.block_chain_manager = block_chain_manager
            self.vector_store = vector_store
            self.activation_traces: Dict[str, Dict[str, float]] = {}
            self.semantic_paths: Dict[str, List[str]] = {}

            # Import threading module
            import threading
            self.lock = threading.RLock()

            # Performance tracking
            self.performance_stats: Dict[str, Dict[str, float]] = {
                "trace_activation": {"calls": 0, "total_time": 0.0},
                "detect_semantic_drift": {"calls": 0, "total_time": 0.0},
                "identify_reasoning_errors": {"calls": 0, "total_time": 0.0}
            }

    def trace_activation(self, query_vector: np.ndarray, threshold: float = 0.6, top_k: int = 20) -> str:
        """
        Traces activation patterns based on a query vector.

        Args:
            query_vector: Vector to trace activations for
            threshold: Minimum similarity threshold for activations
            top_k: Maximum number of similar vectors to consider

        Returns:
            trace_id: Unique identifier for the activation trace
        """
        start_time = time.time()
        with self.lock:
            trace_id = str(uuid.uuid4())
            activations: Dict[str, float] = {}

            # Find similar vectors
            similar_vectors = self.vector_store.search_similar(query_vector, top_k=top_k, threshold=threshold)

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

            # Update performance stats
            self.performance_stats["trace_activation"]["calls"] += 1
            self.performance_stats["trace_activation"]["total_time"] += time.time() - start_time

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
        start_time = time.time()
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

            # Update performance stats
            self.performance_stats["detect_semantic_drift"]["calls"] += 1
            self.performance_stats["detect_semantic_drift"]["total_time"] += time.time() - start_time

            logger.debug(f"Detected {len(drift_points)} semantic drift points for trace {trace_id}")
            return drift_points

    def identify_reasoning_errors(self, trace_id: str, error_patterns: List[Dict[str, Any]]) -> List[str]:
        """
        Identify potential reasoning errors based on predefined patterns.

        Args:
            trace_id: ID of the activation trace to analyze
            error_patterns: List of pattern dictionaries defining error conditions

        Returns:
            List of block IDs matching error patterns
        """
        start_time = time.time()
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
                    data_match = all(block.data.get(k) == v for k, v in pattern.get("data", {}).items())

                    # Check if any embedding metadata matches pattern
                    embedding_match = False
                    if "embedding_metadata" in pattern:
                        for embedding in block.embeddings:
                            if all(embedding.metadata.get(k) == v for k, v in pattern["embedding_metadata"].items()):
                                embedding_match = True
                                break
                    else:
                        embedding_match = True  # No metadata criteria specified

                    if data_match and embedding_match:
                        error_blocks.append(block_id)
                        break

            # Update performance stats
            self.performance_stats["identify_reasoning_errors"]["calls"] += 1
            self.performance_stats["identify_reasoning_errors"]["total_time"] += time.time() - start_time

            logger.debug(f"Identified {len(error_blocks)} blocks with reasoning errors in trace {trace_id}")
            return error_blocks

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
                if block is not None:
                    blocks.append(block)

            logger.debug(f"Following semantic path {path_id} with {len(blocks)} valid blocks")
            return blocks

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance statistics for the neural backtracker operations.

        Returns:
            Dictionary of operation statistics including calls and execution time
        """
        with self.lock:
            stats = {}
            for op, metrics in self.performance_stats.items():
                stats[op] = metrics.copy()
                if metrics["calls"] > 0:
                    stats[op]["avg_time"] = metrics["total_time"] / metrics["calls"]
            return stats

    def clear_traces(self, older_than: Optional[float] = None) -> int:
        """
        Clear activation traces, optionally only those older than a specified time.

        Args:
            older_than: Optional timestamp; traces created before this will be removed

        Returns:
            Number of traces cleared
        """
        with self.lock:
            if older_than is None:
                count = len(self.activation_traces)
                self.activation_traces.clear()
                return count

            to_remove = []
            current_time = time.time()
            for trace_id in self.activation_traces:
                trace_age = current_time - self.activation_traces[trace_id].get("created_at", 0)
                if trace_age > older_than:
                    to_remove.append(trace_id)

            for trace_id in to_remove:
                del self.activation_traces[trace_id]

            return len(to_remove)