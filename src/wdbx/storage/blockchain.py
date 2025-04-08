import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Try to import JAX and PyTorch for potential acceleration
try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..core.data_structures import Block, EmbeddingVector


class BlockChainManager:
    """
    Manages the blockchain protocol for data integrity and provides methods
    for creating, retrieving, validating, and searching blocks in chains.
    """

    def __init__(
        self,
        enable_logging: bool = True,
        data_dir: Optional[str] = None,
        use_acceleration: bool = True,
    ) -> None:
        self.blocks: Dict[str, Block] = {}
        self.chain_heads: Dict[str, str] = {}  # Maps chain_id -> head block_id
        self.block_chain: Dict[str, str] = {}  # Maps block_id -> chain_id
        self.lock = threading.RLock()
        self.pending_blocks: Dict[str, Block] = {}  # Temporary storage for blocks being processed
        self.deleted_blocks: Set[str] = set()  # Track deleted blocks
        self.data_dir = Path(data_dir) if data_dir else None
        self.logger = logging.getLogger(__name__) if enable_logging else None

        # Configure acceleration options
        self.use_acceleration = use_acceleration
        self.acceleration_backend = None
        if use_acceleration:
            if HAS_JAX:
                self.acceleration_backend = "jax"
            elif HAS_TORCH:
                self.acceleration_backend = "torch"

        if self.logger:
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

            if self.acceleration_backend:
                self.logger.info(f"Using {self.acceleration_backend} for acceleration")
            elif use_acceleration:
                self.logger.warning(
                    "Acceleration requested but neither JAX nor PyTorch is available"
                )

    def add_block(
        self,
        data: Any,
        embeddings: Optional[List[Dict[str, Any]]] = None,
        chain_id: Optional[str] = None,
        context_references: Optional[List[str]] = None,
    ) -> str:
        """
        Add a new block to the blockchain.

        Args:
            data: The data to store in the block
            embeddings: Optional list of embeddings to associate with the block
            chain_id: Optional chain ID to append to (creates new chain if None)
            context_references: Optional list of block IDs to reference

        Returns:
            The ID of the newly created block

        Raises:
            ValueError: If an invalid context reference is provided
        """
        if self.logger:
            self.logger.debug(
                f"Adding block with {len(embeddings) if embeddings else 0} embeddings"
            )

        with self.lock:
            block_id = str(uuid.uuid4())
            previous_hash = ""

            # Validate context references exist
            if context_references:
                valid_refs = [ref for ref in context_references if ref in self.blocks]
                if len(valid_refs) != len(context_references) and self.logger:
                    invalid_refs = set(context_references) - set(valid_refs)
                    self.logger.warning(f"Some context references don't exist: {invalid_refs}")
                context_references = valid_refs

            # Set previous hash if adding to existing chain
            if chain_id and chain_id in self.chain_heads:
                head_block_id = self.chain_heads[chain_id]
                previous_hash = self.blocks[head_block_id].hash

            # Create new block
            block = Block(
                block_id=block_id,
                timestamp=time.time(),
                data=data,
                embeddings=embeddings,
                previous_block_hash=previous_hash,
                context_references=context_references or [],
            )

            # Store block in pending state first
            self.pending_blocks[block_id] = block

            try:
                # Compute hash (this may fail if block data is invalid)
                block.compute_hash()

                # Add block to storage
                self.blocks[block_id] = block

                # Update chain references
                if chain_id:
                    self.chain_heads[chain_id] = block_id
                    self.block_chain[block_id] = chain_id
                else:
                    new_chain_id = str(uuid.uuid4())
                    self.chain_heads[new_chain_id] = block_id
                    self.block_chain[block_id] = new_chain_id
                    block.chain_id = new_chain_id
                    if self.logger:
                        self.logger.info(f"Created new chain with id {new_chain_id}")

                if self.logger:
                    self.logger.debug(
                        f"Created block {block_id} in chain {self.block_chain[block_id]}"
                    )

                # Remove from pending blocks
                del self.pending_blocks[block_id]

                # Auto-save if data directory is configured
                if self.data_dir:
                    self._auto_save_chain(chain_id or block.chain_id)

                return block_id

            except Exception as e:
                # Clean up pending block on error
                if block_id in self.pending_blocks:
                    del self.pending_blocks[block_id]
                if self.logger:
                    self.logger.error(f"Failed to create block: {str(e)}")
                raise RuntimeError(f"Failed to create block: {str(e)}")

    def get_block(self, block_id: str) -> Optional[Block]:
        """
        Retrieve a block by its ID.

        Args:
            block_id: The ID of the block to retrieve

        Returns:
            The Block if found, None otherwise
        """
        with self.lock:
            return self.blocks.get(block_id)

    def get_chain(self, chain_id: str) -> List[Block]:
        """
        Retrieve all blocks in a chain in reverse chronological order (newest first).

        Args:
            chain_id: The ID of the chain to retrieve

        Returns:
            List of Blocks in the chain, or empty list if chain not found
        """
        with self.lock:
            if chain_id not in self.chain_heads:
                if self.logger:
                    self.logger.warning(f"Chain {chain_id} not found")
                return []

            blocks = []
            current_block_id = self.chain_heads[chain_id]
            block_hash_map = {block.hash: block_id for block_id, block in self.blocks.items()}

            while current_block_id:
                block = self.blocks.get(current_block_id)
                if not block:
                    if self.logger:
                        self.logger.error(f"Block {current_block_id} referenced but not found")
                    break

                # Ensure the block actually belongs to the requested chain
                if self.block_chain.get(current_block_id) != chain_id:
                    if self.logger:
                        self.logger.error(
                            f"Chain integrity error: Block {current_block_id} expected in chain {chain_id} but found in {self.block_chain.get(current_block_id)}"
                        )
                    break  # Stop traversal if chain mismatch found

                blocks.append(block)

                # More efficient lookup using hash map instead of iteration
                prev_block_id = block_hash_map.get(block.previous_block_hash)
                current_block_id = prev_block_id

            return blocks

    def validate_chain(self, chain_id: str) -> bool:
        """
        Validate the integrity of an entire chain.

        Args:
            chain_id: The ID of the chain to validate

        Returns:
            True if chain is valid, False otherwise
        """
        with self.lock:
            blocks = self.get_chain(chain_id)
            if not blocks:
                return False

            # Use accelerated validation if available
            if self.use_acceleration and len(blocks) > 10:
                return self._accelerated_validate_chain(blocks)

            # Standard validation
            for i, block in enumerate(blocks):
                # Validate block hash
                if not block.validate():
                    if self.logger:
                        self.logger.error(f"Block {block.block_id} failed hash validation")
                    return False

                # Validate chain linkage (previous block's hash should match current
                # block's previous_block_hash)
                if i < len(blocks) - 1:
                    if block.previous_block_hash != blocks[i + 1].hash:
                        if self.logger:
                            self.logger.error(
                                f"Chain broken between blocks {block.block_id} and {blocks[i + 1].block_id}"
                            )
                        return False

            return True

    def _accelerated_validate_chain(self, blocks: List[Block]) -> bool:
        """
        Validate chain integrity using hardware acceleration.

        Args:
            blocks: List of blocks to validate

        Returns:
            True if chain is valid, False otherwise
        """
        try:
            # First validate all block hashes
            if not all(block.validate() for block in blocks):
                return False

            # Then validate chain linkage
            if self.acceleration_backend == "jax":
                # Extract hashes and previous hashes
                hashes = jnp.array([block.hash for block in blocks])
                prev_hashes = jnp.array([block.previous_block_hash for block in blocks[:-1]])
                next_hashes = hashes[1:]

                # Check if all previous hashes match the next block's hash
                return bool(jnp.all(prev_hashes == next_hashes))

            elif self.acceleration_backend == "torch":
                # Extract hashes and previous hashes
                hashes = [block.hash for block in blocks]
                prev_hashes = [block.previous_block_hash for block in blocks[:-1]]
                next_hashes = hashes[1:]

                # Check if all previous hashes match the next block's hash
                return all(p == n for p, n in zip(prev_hashes, next_hashes))

            else:
                # Fallback to standard validation
                for i in range(len(blocks) - 1):
                    if blocks[i].previous_block_hash != blocks[i + 1].hash:
                        return False
                return True

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Accelerated validation failed, falling back to standard: {str(e)}"
                )

            # Fallback to standard validation
            for i in range(len(blocks) - 1):
                if blocks[i].previous_block_hash != blocks[i + 1].hash:
                    return False
            return True

    def get_context_blocks(self, block_id: str) -> List[Block]:
        """
        Get all blocks referenced by a block's context references.

        Args:
            block_id: The ID of the block whose context to retrieve

        Returns:
            List of referenced Blocks
        """
        with self.lock:
            block = self.blocks.get(block_id)
            if not block:
                return []

            return [
                self.blocks[ref_id] for ref_id in block.context_references if ref_id in self.blocks
            ]

    def search_blocks_by_data(self, query: Dict[str, Any]) -> List[str]:
        """
        Search for blocks that match all key-value pairs in the query.

        Args:
            query: Dictionary of key-value pairs to match against block data

        Returns:
            List of block IDs that match the query
        """
        with self.lock:
            matching_blocks = []
            for block_id, block in self.blocks.items():
                if all(block.data.get(k) == v for k, v in query.items()):
                    matching_blocks.append(block_id)
            return matching_blocks

    def search_blocks_by_embedding(
        self, query_embedding: EmbeddingVector, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for blocks with similar embeddings using vector similarity.

        Args:
            query_embedding: The embedding vector to search for
            top_k: Number of results to return

        Returns:
            List of tuples (block_id, similarity_score) sorted by similarity
        """
        with self.lock:
            if not self.blocks:
                return []

            # Use accelerated similarity search if available
            if self.use_acceleration and self.acceleration_backend:
                return self._accelerated_embedding_search(query_embedding, top_k)

            # Standard similarity search
            results = []
            for block_id, block in self.blocks.items():
                if not block.embeddings:
                    continue

                # Use the first embedding for comparison
                block_embedding = block.embeddings[0]
                similarity = self._cosine_similarity(query_embedding.vector, block_embedding.vector)
                results.append((block_id, similarity))

            # Sort by similarity (highest first) and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    def _accelerated_embedding_search(
        self, query_embedding: EmbeddingVector, top_k: int
    ) -> List[Tuple[str, float]]:
        """
        Perform accelerated embedding similarity search using JAX or PyTorch.

        Args:
            query_embedding: The embedding vector to search for
            top_k: Number of results to return

        Returns:
            List of tuples (block_id, similarity_score) sorted by similarity
        """
        try:
            # Filter blocks that have embeddings
            blocks_with_embeddings = [
                (block_id, block) for block_id, block in self.blocks.items() if block.embeddings
            ]

            if not blocks_with_embeddings:
                return []

            block_ids = [b[0] for b in blocks_with_embeddings]

            if self.acceleration_backend == "jax":
                # Convert query to JAX array
                query_vec = jnp.array(query_embedding.vector)

                # Extract first embedding from each block and convert to JAX array
                block_vecs = jnp.array([b[1].embeddings[0].vector for b in blocks_with_embeddings])

                # Normalize vectors
                query_norm = query_vec / jnp.linalg.norm(query_vec)
                block_norms = block_vecs / jnp.linalg.norm(block_vecs, axis=1, keepdims=True)

                # Compute cosine similarities
                similarities = jnp.dot(block_norms, query_norm)

                # Get indices of top_k results
                if len(similarities) <= top_k:
                    top_indices = jnp.argsort(similarities)[::-1]
                else:
                    top_indices = jnp.argsort(similarities)[::-1][:top_k]

                # Convert to list of (block_id, similarity) tuples
                results = [(block_ids[i], float(similarities[i])) for i in top_indices]

            elif self.acceleration_backend == "torch":
                # Convert query to PyTorch tensor
                query_vec = torch.tensor(query_embedding.vector, dtype=torch.float32)

                # Extract first embedding from each block and convert to PyTorch tensor
                block_vecs = torch.tensor(
                    [b[1].embeddings[0].vector for b in blocks_with_embeddings], dtype=torch.float32
                )

                # Normalize vectors
                query_norm = query_vec / torch.norm(query_vec)
                block_norms = block_vecs / torch.norm(block_vecs, dim=1, keepdim=True)

                # Compute cosine similarities
                similarities = torch.matmul(block_norms, query_norm)

                # Get indices of top_k results
                if len(similarities) <= top_k:
                    top_indices = torch.argsort(similarities, descending=True)
                else:
                    top_indices = torch.argsort(similarities, descending=True)[:top_k]

                # Convert to list of (block_id, similarity) tuples
                results = [(block_ids[i], float(similarities[i])) for i in top_indices.tolist()]

            else:
                # This shouldn't happen, but fall back to standard search
                return self.search_blocks_by_embedding(query_embedding, top_k)

            return results

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Accelerated embedding search failed, falling back to standard: {str(e)}"
                )

            # Fall back to standard search
            results = []
            for block_id, block in self.blocks.items():
                if not block.embeddings:
                    continue

                # Use the first embedding for comparison
                block_embedding = block.embeddings[0]
                similarity = self._cosine_similarity(query_embedding.vector, block_embedding.vector)
                results.append((block_id, similarity))

            # Sort by similarity (highest first) and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_chain_id_for_block(self, block_id: str) -> Optional[str]:
        """
        Get the chain ID that a block belongs to.

        Args:
            block_id: The ID of the block

        Returns:
            The chain ID if found, None otherwise
        """
        with self.lock:
            return self.block_chain.get(block_id)

    def get_block_count(self) -> int:
        """
        Get the total number of blocks managed by this instance.

        Returns:
            int: Number of blocks
        """
        with self.lock:
            return len(self.blocks)

    def merge_chains(self, source_chain_id: str, target_chain_id: str) -> bool:
        """
        Merge two chains by appending the source chain to the target chain.

        Args:
            source_chain_id: ID of the source chain to merge from
            target_chain_id: ID of the target chain to merge into

        Returns:
            True if merge was successful, False otherwise
        """
        with self.lock:
            # Validate both chains exist
            if source_chain_id not in self.chain_heads or target_chain_id not in self.chain_heads:
                if self.logger:
                    self.logger.error(
                        f"Cannot merge chains: source={source_chain_id}, target={target_chain_id}"
                    )
                return False

            # Get blocks from both chains
            source_blocks = self.get_chain(source_chain_id)
            target_blocks = self.get_chain(target_chain_id)

            if not source_blocks or not target_blocks:
                return False

            # Get the head blocks
            source_head = source_blocks[0]
            target_head = target_blocks[0]

            try:
                # Create a new block that links the chains
                link_block = Block(
                    block_id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    data={
                        "merge": {"source_chain": source_chain_id, "target_chain": target_chain_id}
                    },
                    embeddings=[],  # No embeddings for a merge block
                    previous_block_hash=target_head.hash,
                    context_references=[source_head.block_id],
                )

                link_block.compute_hash()

                # Add the link block to the target chain
                self.blocks[link_block.block_id] = link_block
                self.block_chain[link_block.block_id] = target_chain_id
                self.chain_heads[target_chain_id] = link_block.block_id

                # Update all source chain blocks to belong to target chain
                for block in source_blocks:
                    self.block_chain[block.block_id] = target_chain_id

                # Remove the source chain
                del self.chain_heads[source_chain_id]

                if self.logger:
                    self.logger.info(f"Merged chain {source_chain_id} into {target_chain_id}")

                # Auto-save if data directory is configured
                if self.data_dir:
                    self._auto_save_chain(target_chain_id)

                return True
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error merging chains: {str(e)}")
                return False

    def export_chain_data(self, chain_id: str) -> List[Dict[str, Any]]:
        """
        Export chain data for external storage or transmission.

        Args:
            chain_id: The ID of the chain to export

        Returns:
            List of serializable dictionaries representing blocks in the chain
        """
        with self.lock:
            blocks = self.get_chain(chain_id)
            return [block.to_dict() for block in blocks]

    def verify(self) -> Tuple[bool, List[str]]:
        """
        Verify the integrity of all chains.

        Returns:
            Tuple of (is_valid, list_of_invalid_chain_ids)
        """
        with self.lock:
            invalid_chains = []
            for chain_id in self.chain_heads:
                if not self.validate_chain(chain_id):
                    invalid_chains.append(chain_id)

            return len(invalid_chains) == 0, invalid_chains

    def save_chain(self, filepath: str) -> bool:
        """
        Save blockchain data to a file.

        Args:
            filepath: Path to save the data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use pathlib for cross-platform path handling
            save_path = Path(filepath)

            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with self.lock:
                # Prepare data for serialization
                data = {
                    "blocks": {
                        block_id: block.to_dict() for block_id, block in self.blocks.items()
                    },
                    "chain_heads": self.chain_heads,
                    "block_chain": self.block_chain,
                    "timestamp": time.time(),
                }

                # Use atomic write to prevent data corruption
                temp_path = save_path.with_suffix(".tmp")
                with open(temp_path, "w") as f:
                    json.dump(data, f, indent=2)

                # Move temporary file to final location (atomic on most systems)
                temp_path.replace(save_path)

                if self.logger:
                    self.logger.info(
                        f"Saved blockchain with {len(self.blocks)} blocks to {filepath}"
                    )
                return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save blockchain: {str(e)}")
            return False

    def _auto_save_chain(self, chain_id: str) -> bool:
        """
        Automatically save a chain to the data directory.

        Args:
            chain_id: ID of the chain to save

        Returns:
            True if successful, False otherwise
        """
        if not self.data_dir:
            return False

        try:
            chain_file = self.data_dir / f"chain_{chain_id}.json"
            return self.save_chain(str(chain_file))
        except Exception as e:
            if self.logger:
                self.logger.error(f"Auto-save failed for chain {chain_id}: {str(e)}")
            return False

    @classmethod
    def load_chain(
        cls, filepath: str, data_dir: Optional[str] = None, use_acceleration: bool = True
    ) -> "BlockChainManager":
        """
        Load blockchain data from a file.

        Args:
            filepath: Path to the file containing chain data
            data_dir: Optional data directory for auto-saving
            use_acceleration: Whether to use hardware acceleration if available

        Returns:
            A new BlockChainManager instance with the loaded data
        """
        manager = cls(data_dir=data_dir, use_acceleration=use_acceleration)

        try:
            # Use pathlib for cross-platform path handling
            load_path = Path(filepath)

            if not load_path.exists():
                if manager.logger:
                    manager.logger.error(f"Blockchain file not found: {filepath}")
                return manager

            with open(load_path) as f:
                data = json.load(f)

            with manager.lock:
                # Load blocks first
                for block_id, block_data in data["blocks"].items():
                    try:
                        block = Block.from_dict(block_data)
                        manager.blocks[block_id] = block
                    except Exception as e:
                        if manager.logger:
                            manager.logger.warning(f"Failed to load block {block_id}: {str(e)}")

                # Then load chain heads and block-chain mappings
                manager.chain_heads = data["chain_heads"]
                manager.block_chain = data["block_chain"]

                if manager.logger:
                    manager.logger.info(
                        f"Loaded blockchain with {len(manager.blocks)} blocks from {filepath}"
                    )

        except Exception as e:
            if manager.logger:
                manager.logger.error(f"Failed to load blockchain from {filepath}: {str(e)}")

        return manager

    def _add_block_to_chain(self, block: Block, chain_id: Optional[str] = None) -> None:
        """
        Add an existing block to a chain (internal method).

        Args:
            block: The block to add
            chain_id: Optional chain ID (creates a new chain if None)
        """
        with self.lock:
            if block.block_id in self.blocks:
                if self.logger:
                    self.logger.warning(f"Block {block.block_id} already exists in blockchain")
                return

            self.blocks[block.block_id] = block

            # Determine chain ID
            if chain_id and chain_id in self.chain_heads:
                actual_chain_id = chain_id
            else:
                # Create a new chain
                actual_chain_id = chain_id or str(uuid.uuid4())
                if self.logger and not chain_id:
                    self.logger.info(f"Created new chain with id {actual_chain_id}")

            # Update chain references
            block_id = block.block_id
            self.chain_heads[actual_chain_id] = block_id
            self.block_chain[block_id] = actual_chain_id

            # Auto-save if data directory is configured
            if self.data_dir:
                self._auto_save_chain(actual_chain_id)

    def delete_block(self, block_id: str) -> bool:
        """
        Delete a block and all blocks that depend on it.

        Args:
            block_id: ID of the block to delete

        Returns:
            True if successful, False if block not found
        """
        with self.lock:
            if block_id not in self.blocks:
                return False

            # Find all blocks that depend on this one
            chain_id = self.block_chain.get(block_id)
            if not chain_id:
                if self.logger:
                    self.logger.warning(f"Block {block_id} has no chain assignment")
                return False

            # Check if this is the head block
            is_head = self.chain_heads.get(chain_id) == block_id

            # Get the chain
            chain_blocks = self.get_chain(chain_id)

            # Find the position of this block in the chain
            block_index = next(
                (i for i, b in enumerate(chain_blocks) if b.block_id == block_id), -1
            )

            if block_index == -1:
                if self.logger:
                    self.logger.error(f"Block {block_id} not found in its claimed chain {chain_id}")
                return False

            # Delete this block and all newer blocks (they depend on this one)
            blocks_to_delete = chain_blocks[: block_index + 1]

            for block in blocks_to_delete:
                if block.block_id in self.blocks:
                    del self.blocks[block.block_id]
                if block.block_id in self.block_chain:
                    del self.block_chain[block.block_id]

                # Track the deletion
                self.deleted_blocks.add(block.block_id)

            # Update the chain head if needed
            if is_head:
                if block_index + 1 < len(chain_blocks):
                    # Set the previous block as the new head
                    new_head = chain_blocks[block_index + 1].block_id
                    self.chain_heads[chain_id] = new_head
                else:
                    # Chain is now empty
                    del self.chain_heads[chain_id]

            if self.logger:
                self.logger.info(
                    f"Deleted block {block_id} and {len(blocks_to_delete) - 1} dependent blocks"
                )

            # Auto-save if data directory is configured
            if self.data_dir and chain_id in self.chain_heads:
                self._auto_save_chain(chain_id)

            return True

    def get_chain_stats(self, chain_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about chains.

        Args:
            chain_id: Optional chain ID to get stats for a specific chain

        Returns:
            Dictionary with statistics
        """
        with self.lock:
            stats = {
                "total_blocks": len(self.blocks),
                "total_chains": len(self.chain_heads),
                "pending_blocks": len(self.pending_blocks),
                "deleted_blocks": len(self.deleted_blocks),
            }

            if chain_id:
                if chain_id not in self.chain_heads:
                    stats["chain_found"] = False
                    return stats

                chain_blocks = self.get_chain(chain_id)
                stats.update(
                    {
                        "chain_found": True,
                        "chain_id": chain_id,
                        "chain_length": len(chain_blocks),
                        "head_block": self.chain_heads.get(chain_id),
                        "oldest_timestamp": chain_blocks[-1].timestamp if chain_blocks else None,
                        "newest_timestamp": chain_blocks[0].timestamp if chain_blocks else None,
                    }
                )
            else:
                # General stats about all chains
                chain_lengths = {cid: len(self.get_chain(cid)) for cid in self.chain_heads}
                stats.update(
                    {
                        "chains": [
                            {"id": cid, "length": length, "head": self.chain_heads[cid]}
                            for cid, length in chain_lengths.items()
                        ],
                        "longest_chain": (
                            max(chain_lengths.items(), key=lambda x: x[1])[0]
                            if chain_lengths
                            else None
                        ),
                        "shortest_chain": (
                            min(chain_lengths.items(), key=lambda x: x[1])[0]
                            if chain_lengths
                            else None
                        ),
                        "average_chain_length": (
                            sum(chain_lengths.values()) / len(chain_lengths) if chain_lengths else 0
                        ),
                    }
                )

            return stats

    def export_chain(self, chain_id: str, output_path: str, format: str = "json") -> bool:
        """
        Export a chain to a file.

        Args:
            chain_id: ID of the chain to export
            output_path: Path to save the exported chain
            format: Export format (currently only 'json' is supported)

        Returns:
            True if export was successful, False otherwise
        """
        if format.lower() != "json":
            if self.logger:
                self.logger.error(f"Unsupported export format: {format}")
            return False

        with self.lock:
            if chain_id not in self.chain_heads:
                if self.logger:
                    self.logger.error(f"Chain {chain_id} not found")
                return False

            try:
                chain_blocks = self.get_chain(chain_id)

                # Convert blocks to serializable format
                serialized_blocks = []
                for block in chain_blocks:
                    # Handle embeddings based on acceleration backend
                    serialized_embeddings = []
                    if block.embeddings:
                        for embedding in block.embeddings:
                            # Convert vector to standard Python list regardless of backend
                            vector = embedding.vector
                            if (
                                self.use_acceleration
                                and self.acceleration_backend == "jax"
                                and HAS_JAX
                            ):
                                if isinstance(vector, jnp.ndarray):
                                    vector = vector.tolist()
                            elif (
                                self.use_acceleration
                                and self.acceleration_backend == "torch"
                                and HAS_TORCH
                            ):
                                if isinstance(vector, torch.Tensor):
                                    vector = vector.cpu().numpy().tolist()

                            serialized_embeddings.append(
                                {
                                    "id": embedding.id,
                                    "vector": vector,
                                    "metadata": embedding.metadata,
                                }
                            )

                    block_data = {
                        "block_id": block.block_id,
                        "previous_block_id": block.previous_block_id,
                        "timestamp": block.timestamp,
                        "data": block.data,
                        "hash": block.hash,
                        "embeddings": serialized_embeddings,
                    }
                    serialized_blocks.append(block_data)

                # Create export data structure
                export_data = {
                    "chain_id": chain_id,
                    "head_block_id": self.chain_heads[chain_id],
                    "block_count": len(chain_blocks),
                    "export_timestamp": time.time(),
                    "acceleration_backend": self.acceleration_backend,
                    "blocks": serialized_blocks,
                }

                # Write to file
                with open(output_path, "w") as f:
                    json.dump(export_data, f, indent=2)

                if self.logger:
                    self.logger.info(
                        f"Exported chain {chain_id} with {len(chain_blocks)} blocks to {output_path}"
                    )
                return True

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to export chain {chain_id}: {str(e)}")
                return False
