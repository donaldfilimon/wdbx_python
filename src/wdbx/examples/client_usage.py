"""
WDBX Client Library Usage Example.

This example demonstrates the usage of the WDBXClient library for common vector operations.
"""

import os
import json
import numpy as np
import time
from pathlib import Path

from wdbx.client import WDBXClient, wdbx_session
from wdbx.utils.logging_utils import setup_logging, get_logger

# Set up logging
setup_logging(log_level="INFO")
logger = get_logger("wdbx.examples.client_usage")


def example_client_instance():
    """Demonstrate creating a client instance and basic usage."""
    logger.info("=== Example 1: Creating a WDBX client instance ===")
    
    # Create a client with a specified data directory
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    client = WDBXClient(
        data_dir=data_dir,
        vector_dimension=384,  # Using a smaller dimension for example
        enable_memory_optimization=True,
        auto_connect=True,  # Connect automatically
    )
    
    # Display client information
    logger.info(f"Client created with data directory: {client.data_dir}")
    logger.info(f"Vector dimension: {client.vector_dimension}")
    
    # Perform basic operations
    vector_data = np.random.rand(384).astype(np.float32)
    metadata = {"description": "Example vector", "tags": ["example", "test"]}
    
    # Create a vector
    vector = client.create_vector(vector_data=vector_data, metadata=metadata)
    logger.info(f"Created vector with ID: {vector.id}")
    
    # Save the vector
    client.save_vector(vector)
    
    # Create another vector for comparison
    vector2 = client.create_vector(
        vector_data=np.random.rand(384).astype(np.float32),
        metadata={"description": "Another example vector"}
    )
    client.save_vector(vector2)
    
    # Find similar vectors
    results = client.find_similar_vectors(query=vector, top_k=5)
    logger.info(f"Found {len(results)} similar vectors:")
    for vector_id, similarity in results:
        logger.info(f"  Vector ID: {vector_id}, Similarity: {similarity:.4f}")
    
    # Clean up
    client.disconnect()
    logger.info("Client disconnected")


def example_context_manager():
    """Demonstrate using the WDBX client with a context manager."""
    logger.info("\n=== Example 2: Using the WDBX client with a context manager ===")
    
    # Using a temporary directory (automatically cleaned up)
    with wdbx_session(vector_dimension=384) as client:
        logger.info(f"Session created with temporary directory: {client.data_dir}")
        
        # Create a batch of vectors
        vector_data_list = [np.random.rand(384).astype(np.float32) for _ in range(5)]
        metadata_list = [
            {"index": i, "description": f"Batch vector {i}"} 
            for i in range(5)
        ]
        
        # Create vectors in batch
        vectors = client.batch_create_vectors(
            vector_data_list=vector_data_list,
            metadata_list=metadata_list
        )
        
        logger.info(f"Created {len(vectors)} vectors in batch")
        
        # Create a block containing the vectors
        block = client.create_block(
            data={"name": "Example block", "description": "Contains batch vectors"},
            embeddings=vectors
        )
        
        logger.info(f"Created block with ID: {block.id}")
        client.save_block(block)
        
        # Search for similar vectors to the first vector
        query = vectors[0]
        results = client.find_similar_vectors(query=query, top_k=3)
        logger.info(f"Found {len(results)} similar vectors to {query.id}:")
        for vector_id, similarity in results:
            logger.info(f"  Vector ID: {vector_id}, Similarity: {similarity:.4f}")
    
    logger.info("Session ended and resources cleaned up")


def example_import_export():
    """Demonstrate importing and exporting data."""
    logger.info("\n=== Example 3: Importing and exporting data ===")
    
    # Create a client and generate some data
    export_dir = os.path.join(os.path.dirname(__file__), "export_data")
    
    with wdbx_session(vector_dimension=384) as client:
        # Create some vectors and blocks
        for i in range(10):
            vector = client.create_vector(
                vector_data=np.random.rand(384).astype(np.float32),
                metadata={"index": i, "description": f"Export vector {i}"}
            )
            client.save_vector(vector)
        
        # Create a few blocks
        for i in range(3):
            # Get some random vectors
            vector_ids = list(client.core.vectors.keys())
            selected_vectors = [client.core.vectors[vid] for vid in vector_ids[:3]]
            
            block = client.create_block(
                data={"name": f"Block {i}", "description": f"Export block {i}"},
                embeddings=selected_vectors
            )
            client.save_block(block)
        
        # Export data
        logger.info(f"Exporting data to {export_dir}")
        client.export_data(output_dir=export_dir, format="json")
        
        # Show statistics before clearing
        stats = client.get_stats()
        logger.info(f"Statistics before clearing: {json.dumps(stats, indent=2)}")
        
        # Clear data
        client.clear()
        logger.info("Cleared all in-memory data")
        
        # Import the data back
        logger.info(f"Importing data from {export_dir}")
        client.import_data(input_dir=export_dir, format="json")
        
        # Show statistics after import
        stats = client.get_stats()
        logger.info(f"Statistics after import: {json.dumps(stats, indent=2)}")
    
    logger.info("Import/export example completed")


def example_memory_optimization():
    """Demonstrate memory optimization."""
    logger.info("\n=== Example 4: Memory optimization ===")
    
    with wdbx_session(vector_dimension=384, enable_memory_optimization=True) as client:
        # Create a large number of vectors to test memory optimization
        logger.info("Creating 100 vectors to test memory optimization")
        
        for i in range(100):
            vector = client.create_vector(
                vector_data=np.random.rand(384).astype(np.float32),
                metadata={"index": i}
            )
            # Only save every 10th vector to see the effect on memory usage
            if i % 10 == 0:
                client.save_vector(vector)
        
        # Get memory stats before optimization
        stats_before = client.get_stats()
        logger.info(f"Memory usage before optimization: {stats_before.get('memory_usage', 'N/A')} bytes")
        
        # Optimize memory
        logger.info("Optimizing memory...")
        client.optimize_memory()
        
        # Get memory stats after optimization
        stats_after = client.get_stats()
        logger.info(f"Memory usage after optimization: {stats_after.get('memory_usage', 'N/A')} bytes")
        
        # Verify vectors are still accessible
        vector_ids = list(client.core.vectors.keys())
        logger.info(f"Number of vectors in memory after optimization: {len(vector_ids)}")
        
        # Try to access a saved vector (should be loaded from disk)
        for vid in vector_ids[:5]:
            vector = client.get_vector(vid)
            if vector:
                logger.info(f"Successfully loaded vector {vid} after optimization")
    
    logger.info("Memory optimization example completed")


def example_async_client():
    """Demonstrate using the async client."""
    logger.info("\n=== Example 5: Using the async client ===")
    
    import asyncio
    
    async def async_operations():
        with wdbx_session(vector_dimension=384) as client:
            # Get the async client
            async_client = client.get_async_client()
            logger.info("Created async client")
            
            # Create a vector asynchronously
            vector_data = np.random.rand(384).astype(np.float32)
            vector = await async_client.create_vector(
                vector_data=vector_data,
                metadata={"description": "Async vector"}
            )
            logger.info(f"Created vector asynchronously with ID: {vector.id}")
            
            # Create another vector for similarity comparison
            vector2 = await async_client.create_vector(
                vector_data=np.random.rand(384).astype(np.float32),
                metadata={"description": "Another async vector"}
            )
            
            # Find similar vectors asynchronously
            similar_vectors = await async_client.find_similar_vectors(
                query_vector=vector,
                top_k=5
            )
            logger.info(f"Found {len(similar_vectors)} similar vectors asynchronously")
            
            # Create a block asynchronously
            block = await async_client.create_block(
                data={"name": "Async block", "description": "Created asynchronously"},
                embeddings=[vector, vector2]
            )
            logger.info(f"Created block asynchronously with ID: {block.id}")
            
            # Search blocks asynchronously
            similar_blocks = await async_client.search_blocks(
                query=vector,
                top_k=3
            )
            logger.info(f"Found {len(similar_blocks)} similar blocks asynchronously")
            
            # Shut down the async client explicitly (optional, as it happens in client.disconnect())
            await async_client.shutdown()
    
    # Run the async function
    asyncio.run(async_operations())
    logger.info("Async client example completed")


def run_all_examples():
    """Run all examples."""
    logger.info("Starting WDBX client examples")
    
    example_client_instance()
    example_context_manager()
    example_import_export()
    example_memory_optimization()
    example_async_client()
    
    logger.info("All examples completed successfully")


if __name__ == "__main__":
    run_all_examples() 