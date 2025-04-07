"""
Unit tests for the blockchain module.
"""
import time

import numpy as np

from src.wdbx.core.data_structures import Block, EmbeddingVector


def test_block_creation():
    """Test creation of a block."""
    block = Block(
        block_id="test-block-1",
        timestamp=time.time(),
        data={"message": "Test block"},
        embeddings=[],
        previous_block_hash="",
    )

    # Verify block properties
    assert block.block_id == "test-block-1"
    assert block.previous_block_hash == ""
    assert isinstance(block.timestamp, float)
    assert block.data["message"] == "Test block"

    # Hash should be generated
    assert block.hash is not None


def test_block_chain_manager(block_chain_manager):
    """Test the blockchain manager."""
    # Create a test block
    data = {"message": "Test blockchain"}
    embeddings = []
    
    # Create a block and add it to the chain
    block = block_chain_manager.create_block(
        data=data,
        embeddings=embeddings
    )
    
    # Verify the block was created and added
    assert block is not None
    assert block.block_id is not None
    assert block_chain_manager.get_block(block.block_id) == block
    
    # Create a second block in the same chain
    second_block = block_chain_manager.create_block(
        data={"message": "Second block"},
        embeddings=[],
        chain_id=block.chain_id
    )
    
    # Verify chain links
    assert second_block.previous_block_hash == block.hash
    
    # Verify chain integrity
    assert block_chain_manager.validate_chain(chain_id=block.chain_id)

    # Perform some operation that creates a block
    data = {"message": "Test block 1"}
    embeddings = [EmbeddingVector(vector=np.random.rand(128))]
    block1 = block_chain_manager.create_block(data, embeddings, chain_id=block.chain_id)
    
    assert block1 is not None
    assert block1.block_id is not None
    assert block_chain_manager.get_block(block1.block_id) == block1
    assert block_chain_manager.get_chain_id_for_block(block1.block_id) == block.chain_id 