import uuid
import time
import threading
from typing import Any, Dict, List, Optional
from wdbx.data_structures import Block, EmbeddingVector

class BlockChainManager:
    """
    Manages the blockchain protocol for data integrity.
    """
    def __init__(self) -> None:
        self.blocks: Dict[str, Block] = {}
        self.chain_heads: Dict[str, str] = {}
        self.block_chain: Dict[str, str] = {}
        self.lock = threading.RLock()

    def create_block(self, data: Dict[str, Any], embeddings: List[EmbeddingVector],
                     chain_id: Optional[str] = None, context_references: Optional[List[str]] = None) -> Block:
        with self.lock:
            block_id = str(uuid.uuid4())
            previous_hash = ""
            if chain_id and chain_id in self.chain_heads:
                head_block_id = self.chain_heads[chain_id]
                previous_hash = self.blocks[head_block_id].hash
            block = Block(
                id=block_id,
                timestamp=time.time(),
                data=data,
                embeddings=embeddings,
                previous_hash=previous_hash,
                context_references=context_references or []
            )
            self.blocks[block_id] = block
            if chain_id:
                self.chain_heads[chain_id] = block_id
                self.block_chain[block_id] = chain_id
            else:
                new_chain_id = str(uuid.uuid4())
                self.chain_heads[new_chain_id] = block_id
                self.block_chain[block_id] = new_chain_id
            return block

    def get_block(self, block_id: str) -> Optional[Block]:
        with self.lock:
            return self.blocks.get(block_id)

    def get_chain(self, chain_id: str) -> List[Block]:
        with self.lock:
            if chain_id not in self.chain_heads:
                return []
            blocks = []
            current_block_id = self.chain_heads[chain_id]
            while current_block_id:
                block = self.blocks.get(current_block_id)
                if not block:
                    break
                blocks.append(block)
                prev_block_id = next((bid for bid, b in self.blocks.items() if b.hash == block.previous_hash), None)
                current_block_id = prev_block_id
            return blocks

    def validate_chain(self, chain_id: str) -> bool:
        with self.lock:
            blocks = self.get_chain(chain_id)
            if not blocks:
                return False
            for i, block in enumerate(blocks):
                if not block.validate():
                    return False
                if i < len(blocks) - 1:
                    if block.previous_hash != blocks[i + 1].hash:
                        return False
            return True

    def get_context_blocks(self, block_id: str) -> List[Block]:
        with self.lock:
            block = self.blocks.get(block_id)
            if not block:
                return []
            return [self.blocks[ref_id] for ref_id in block.context_references if ref_id in self.blocks]

    def search_blocks_by_data(self, query: Dict[str, Any]) -> List[str]:
        with self.lock:
            matching_blocks = []
            for block_id, block in self.blocks.items():
                if all(block.data.get(k) == v for k, v in query.items()):
                    matching_blocks.append(block_id)
            return matching_blocks
