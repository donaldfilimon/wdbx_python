# wdbx/neural_backtracking.py
import uuid
import time
import numpy as np
from typing import List, Dict
from wdbx.data_structures import Block
from wdbx.vector_store import VectorStore
from wdbx.blockchain import BlockChainManager
from wdbx.constants import logger

class NeuralBacktracker:
    """
    Provides methods for tracing activation patterns and detecting semantic drift.
    """
    def __init__(self, block_chain_manager: BlockChainManager, vector_store: VectorStore) -> None:
            self.block_chain_manager = block_chain_manager
            self.vector_store = vector_store
            self.activation_traces: Dict[str, Dict[str, float]] = {}
            self.semantic_paths: Dict[str, List[str]] = {}

            # Import threading module
            import threading
            self.lock = threading.RLock()
    
    def trace_activation(self, query_vector: np.ndarray, threshold: float = 0.6) -> str:
        with self.lock:
            trace_id = str(uuid.uuid4())
            activations: Dict[str, float] = {}
            similar_vectors = self.vector_store.search_similar(query_vector, top_k=20, threshold=threshold)
            for vector_id, similarity in similar_vectors:
                blocks_with_vector = []
                for block_id, block in self.block_chain_manager.blocks.items():
                    for embedding in block.embeddings:
                        if vector_id in embedding.metadata.get("vector_ids", []):
                            blocks_with_vector.append(block_id)
                            break
                for block_id in blocks_with_vector:
                    activations[block_id] = max(activations.get(block_id, 0.0), similarity)
            self.activation_traces[trace_id] = activations
            return trace_id
    
    def detect_semantic_drift(self, trace_id: str, threshold: float = 0.2) -> List[str]:
        with self.lock:
            trace = self.activation_traces.get(trace_id)
            if not trace:
                raise ValueError(f"Invalid trace ID: {trace_id}")
            block_ids = sorted(trace.keys(), key=lambda x: trace[x], reverse=True)
            drift_points = []
            prev_activation = None
            for block_id in block_ids:
                activation = trace[block_id]
                if prev_activation is not None and (prev_activation - activation) > threshold:
                    drift_points.append(block_id)
                prev_activation = activation
            return drift_points
    
    def identify_reasoning_errors(self, trace_id: str, error_patterns: List[Dict[str, any]]) -> List[str]:
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
                    data_match = all(block.data.get(k) == v for k, v in pattern.get("data", {}).items())
                    embedding_match = any(
                        all(embedding.metadata.get(k) == v for k, v in pattern.get("embedding_metadata", {}).items())
                        for embedding in block.embeddings
                    )
                    if data_match and (not pattern.get("embedding_metadata") or embedding_match):
                        error_blocks.append(block_id)
                        break
            return error_blocks
    
    def create_semantic_path(self, blocks: List[str]) -> str:
        with self.lock:
            for block_id in blocks:
                if not self.block_chain_manager.get_block(block_id):
                    raise ValueError(f"Invalid block ID: {block_id}")
            path_id = str(uuid.uuid4())
            self.semantic_paths[path_id] = blocks
            return path_id
    
    def follow_semantic_path(self, path_id: str) -> List[Block]:
            with self.lock:
                path = self.semantic_paths.get(path_id)
                if not path:
                    raise ValueError(f"Invalid path ID: {path_id}")
                blocks = []
                for block_id in path:
                    block = self.block_chain_manager.get_block(block_id)
                    if block is not None:
                        blocks.append(block)
                return blocks