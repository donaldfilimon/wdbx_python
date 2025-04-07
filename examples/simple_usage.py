#!/usr/bin/env python3
"""
Simple usage example for WDBX.

This example demonstrates the basic functionality of WDBX, including:
- Creating and configuring a WDBX instance
- Storing embedding vectors
- Creating conversation blocks
- Searching for similar vectors
- Retrieving conversation context
"""

import os
import sys

import numpy as np

# Add the project root directory to the Python path
# This is only necessary if WDBX is not installed in your Python environment
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import from the reorganized package
from src.wdbx import EmbeddingVector, WDBXConfig, create_wdbx


def main():
    # Create a WDBX instance with 128-dimensional vectors
    print("Initializing WDBX...")
    
    # Create a configuration
    config = WDBXConfig(
        vector_dimension=128,
        num_shards=4,
        data_dir="./wdbx_data",
        enable_persona_management=True,
        content_filter_level="medium"
    )
    
    # Create the WDBX instance
    db = create_wdbx(
        vector_dimension=config.vector_dimension,
        num_shards=config.num_shards,
        data_dir=config.data_dir,
        enable_persona_management=config.enable_persona_management,
        content_filter_level=config.content_filter_level
    )
    
    print(f"WDBX initialized with {db.vector_dimension}-dimensional vectors")
    
    # Create some sample embedding vectors
    print("\nCreating and storing embedding vectors...")
    embeddings = []
    vector_ids = []
    
    for i in range(5):
        # Create a random vector with metadata
        vector = np.random.rand(db.vector_dimension).astype(np.float32)
        metadata = {
            "text": f"Sample text {i}",
            "source": "example script",
            "index": i
        }
        
        # Create an embedding vector
        embedding = EmbeddingVector(vector=vector, metadata=metadata)
        embeddings.append(embedding)
        
        # Store the embedding
        vector_id = db.store_embedding(embedding)
        vector_ids.append(vector_id)
        print(f"  Stored vector {i+1} with ID: {vector_id}")
    
    # Create a conversation block
    print("\nCreating a conversation block...")
    block_data = {
        "user_message": "What is the meaning of life?",
        "response": "The meaning of life is a philosophical question that has been pondered throughout human history.",
        "timestamp": "2023-01-01T12:00:00Z",
        "conversation_id": "example-conversation",
        "turn_number": 1
    }
    
    block_id = db.create_conversation_block(
        data=block_data,
        embeddings=embeddings[:2],  # Use the first two embeddings
        persona_id="thoughtful"  # Use a specific persona
    )
    print(f"Created block with ID: {block_id}")
    
    # Create a second block in the same chain
    print("\nCreating a second block in the same chain...")
    second_block_data = {
        "user_message": "Why do we ask such questions?",
        "response": "Humans are naturally curious and seek meaning in their existence.",
        "timestamp": "2023-01-01T12:01:00Z",
        "conversation_id": "example-conversation",
        "turn_number": 2
    }
    
    second_block_id = db.create_conversation_block(
        data=second_block_data,
        embeddings=embeddings[2:4],  # Use the next two embeddings
        previous_block_hash=block_id,  # Use the first block's ID as reference
        context_references=[block_id],  # Reference the first block
        persona_id="philosophical"  # Use a different persona
    )
    print(f"Created second block with ID: {second_block_id}")
    
    # Search for similar vectors
    print("\nSearching for similar vectors...")
    query_vector = embeddings[0].vector  # Use the first embedding as query
    results = db.search_similar_vectors(query_vector, top_k=3)
    
    print("Search results:")
    for i, (vector_id, similarity) in enumerate(results):
        print(f"  Result {i+1}: Vector ID {vector_id}, Similarity: {similarity:.4f}")
    
    # Get conversation context
    print("\nRetrieving conversation context...")
    context = db.get_conversation_context([block_id, second_block_id])
    
    print(f"Retrieved context with {len(context['blocks'])} blocks")
    
    # Get system stats
    print("\nSystem statistics:")
    stats = db.get_system_stats()
    print(f"  Uptime: {stats['uptime_formatted']}")
    print(f"  Vectors stored: {stats['vectors']['stored']}")
    print(f"  Blocks created: {stats['blocks']['created']}")
    print(f"  Memory usage: {stats['memory_usage_mb']:.2f} MB")
    
    # Clean up
    print("\nClosing WDBX...")
    db.close()
    print("Done!")

if __name__ == "__main__":
    main() 