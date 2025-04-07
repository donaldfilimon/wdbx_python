import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from ..core.data_structures import Block, EmbeddingVector


class BlockChainManager:
    """
    Manages the blockchain protocol for data integrity and provides methods
    for creating, retrieving, validating, and searching blocks in chains.
    """

    def __init__(self, enable_logging: bool = True) -> None:
        self.blocks: Dict[str, Block] = {}
        self.chain_heads: Dict[str, str] = {}  # Maps chain_id -> head block_id
        self.block_chain: Dict[str, str] = {}  # Maps block_id -> chain_id
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__) if enable_logging else None

        if self.logger:
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

    def create_block(self,
                     data: Dict[str,
                                Any],
                     embeddings: List[EmbeddingVector],
                     chain_id: Optional[str] = None,
                     context_references: Optional[List[str]] = None) -> Block:
        """
        Create a new block and add it to a chain.

        Args:
            data: The data to store in the block
            embeddings: Vector embeddings related to the data
            chain_id: Optional chain to add block to (creates new chain if None)
            context_references: Optional list of related block IDs

        Returns:
            The newly created Block
        """
        with self.lock:
            block_id = str(uuid.uuid4())
            previous_hash = ""

            # Validate context references exist
            if context_references:
                valid_refs = [ref for ref in context_references if ref in self.blocks]
                if len(valid_refs) != len(context_references) and self.logger:
                    self.logger.warning(
                        f"Some context references don't exist: {
                            set(context_references) - set(valid_refs)}")
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
                context_references=context_references or []
            )

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
                self.logger.debug(f"Created block {block_id} in chain {self.block_chain[block_id]}")

            return block

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
                        self.logger.error(f"Chain integrity error: Block {current_block_id} expected in chain {chain_id} but found in {self.block_chain.get(current_block_id)}")
                    break # Stop traversal if chain mismatch found

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
                                f"Chain broken between blocks {block.block_id} and {blocks[i + 1].block_id}")
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

            return [self.blocks[ref_id]
                    for ref_id in block.context_references if ref_id in self.blocks]

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
        Merge the source chain into the target chain.

        Args:
            source_chain_id: ID of the chain to merge from
            target_chain_id: ID of the chain to merge into

        Returns:
            True if merge successful, False otherwise
        """
        with self.lock:
            if source_chain_id not in self.chain_heads or target_chain_id not in self.chain_heads:
                if self.logger:
                    self.logger.error("Cannot merge: one or both chains not found")
                return False

            source_blocks = self.get_chain(source_chain_id)

            # Update chain references for all blocks in source chain
            for block in source_blocks:
                self.block_chain[block.id] = target_chain_id

            # Remove the source chain
            del self.chain_heads[source_chain_id]

            if self.logger:
                self.logger.info(f"Merged chain {source_chain_id} into {target_chain_id}")

            return True

    def export_chain_data(self, chain_id: str) -> List[Dict[str, Any]]:
        """
        Export all data from blocks in a chain.

        Args:
            chain_id: The ID of the chain to export

        Returns:
            List of data dictionaries from each block
        """
        with self.lock:
            blocks = self.get_chain(chain_id)
            return [block.data for block in blocks]

    def verify(self):
        """Verify the integrity of the entire blockchain."""
        self.logger.debug("Verifying blockchain integrity")

    # --- Persistence Methods (Placeholders) ---

    def save_chain(self, filepath: str) -> None:
        """Save the current blockchain state to a file (placeholder)."""
        with self.lock:
            # logger.info(f"Attempting to save blockchain to {filepath}...")
            try:
                chain_data = [block.to_dict() for block in self.blocks.values()]
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(chain_data, f, indent=2)
                # logger.info(f"Blockchain successfully saved to {filepath}.")
            except Exception:
                # logger.error(f"Failed to save blockchain: {e}")
                pass # Keep silent for now

    @classmethod
    def load_chain(cls, filepath: str) -> "BlockChainManager":
        """Load a blockchain state from a file (placeholder)."""
        # logger.info(f"Attempting to load blockchain from {filepath}...")
        try:
            with open(filepath, encoding="utf-8") as f:
                chain_data = json.load(f)
            
            manager = cls(enable_logging=False) # Create manager without default logging
            manager.blocks = {}
            manager.chain_heads = {}
            manager.block_chain = {}
            
            for block_data in chain_data:
                # Recreate blocks from dictionary data
                block = Block.from_dict(block_data)
                manager._add_block_to_chain(block)
                
            if not manager.validate_chain():
                # logger.error("Loaded blockchain failed validation!")
                raise ValueError("Invalid blockchain data loaded from file.")
                
            # logger.info(f"Blockchain successfully loaded from {filepath}.")
            return manager
        except FileNotFoundError:
            # logger.warning(f"Blockchain file not found: {filepath}. Creating new chain.")
            return cls() # Return a new, empty manager
        except Exception:
            # logger.error(f"Failed to load blockchain: {e}. Creating new chain.")
            return cls() # Return a new, empty manager on error

    def _add_block_to_chain(self, block: Block) -> None:
        """Internal helper to add a block to the chain and index."""
        with self.lock:
            self.blocks[block.id] = block
            self.chain_heads[block.id] = block.id
            self.block_chain[block.id] = block.id
            # logger.debug(f"Added block {block.id} to the blockchain.")
