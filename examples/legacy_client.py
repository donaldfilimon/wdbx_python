#!/usr/bin/env python3
"""
Example showing how to use the legacy functionality from wdbx.py
"""

import os
import sys

import numpy as np

# Add parent directory to path so we can import wdbx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import from reorganized structure
from src.wdbx import EmbeddingVector, WDBXConfig, create_wdbx


def example_usage():
    """Show example usage of the WDBX module"""
    # Create a configuration
    config = WDBXConfig(vector_dimension=128, num_shards=4, data_dir="./wdbx_data")

    print("Creating WDBX instance...")
    db = create_wdbx(
        vector_dimension=config.vector_dimension,
        num_shards=config.num_shards,
        data_dir=config.data_dir,
    )

    print(f"WDBX initialized with {db.vector_dimension}-dimensional vectors")

    # Create and store an embedding vector
    print("\nCreating and storing embedding vector...")
    vector = np.random.rand(128).astype(np.float32)
    embedding = EmbeddingVector(vector=vector, metadata={"text": "Example from legacy client"})
    vector_id = db.store_embedding(embedding)
    print(f"Stored vector with ID: {vector_id}")

    # Create a conversation block
    print("\nCreating conversation block...")
    block_id = db.create_conversation_block(
        data={"message": "This is a test message", "source": "legacy_client.py"},
        embeddings=[embedding],
    )
    print(f"Created block with ID: {block_id}")

    # Search for similar vectors
    print("\nSearching for similar vectors...")
    results = db.search_similar_vectors(vector, top_k=5)
    for i, (vid, sim) in enumerate(results):
        print(f"  Result {i+1}: Vector ID {vid}, Similarity: {sim:.4f}")

    # Get system stats
    print("\nSystem statistics:")
    stats = db.get_system_stats()
    print(f"  Uptime: {stats['uptime_formatted']}")
    print(f"  Vectors stored: {stats['vectors']['stored']}")
    print(f"  Blocks created: {stats['blocks']['created']}")

    # Close the database
    print("\nClosing database...")
    db.close()
    print("Done!")


if __name__ == "__main__":
    # Run the example
    example_usage()
