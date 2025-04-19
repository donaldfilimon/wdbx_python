"""
Basic usage example for the WDBX system.

This example demonstrates how to use the core components of the WDBX system,
including vector operations, block creation, search, and memory optimization.
"""

import time
from pathlib import Path

import numpy as np

from wdbx.client import WDBXClient
from wdbx.core.data_structures import Block, EmbeddingVector

# Use absolute imports instead of relative imports
from wdbx.core.wdbx_core import WDBXCore, get_wdbx_core
from wdbx.plugins import get_plugin, load_plugin_module
from wdbx.utils.logging_utils import LogContext, get_logger

# Initialize logger
logger = get_logger("examples.basic_usage")


# Mock implementation for missing module
class ConfigManager:
    @staticmethod
    def get_instance():
        return ConfigManager()

    def set(self, key, value):
        pass


def setup_test_environment():
    """Set up a test environment with temporary directories."""
    # Create a temporary directory for data
    temp_dir = Path("./temp_data")
    temp_dir.mkdir(exist_ok=True)

    # Set up configuration
    config = ConfigManager.get_instance()
    config.set("data_dir", str(temp_dir))
    config.set("debug_mode", True)
    config.set("memory_optimization_enabled", True)

    return temp_dir


def create_test_vectors(wdbx: WDBXCore, count: int = 5):
    """Create test vectors with random data."""
    logger.info(f"Creating {count} test vectors")

    vectors = []
    for i in range(count):
        # Create a random vector
        vector_data = np.random.rand(wdbx.vector_dimension).astype(np.float32)

        # Create metadata
        metadata = {
            "name": f"Vector {i}",
            "description": f"Test vector {i}",
            "tags": ["test", f"group-{i % 3}"],
        }

        # Create embedding vector
        vector = wdbx.create_vector(
            vector_data=vector_data, metadata=metadata, vector_id=f"vector-{i}"
        )

        vectors.append(vector)
        logger.info(f"Created vector {vector.vector_id} with metadata: {metadata}")

    return vectors


def create_test_blocks(wdbx: WDBXCore, vectors: list[EmbeddingVector], count: int = 3):
    """Create test blocks containing vectors."""
    logger.info(f"Creating {count} test blocks")

    blocks = []
    for i in range(count):
        # Select vectors for this block (each block gets a subset of vectors)
        block_vectors = vectors[i: i + 3]

        # Create block data
        data = {
            "name": f"Block {i}",
            "description": f"Test block {i}",
            "timestamp": time.time(),
            "tags": ["test", f"group-{i % 2}"],
        }

        # Create block
        block = wdbx.create_block(data=data, embeddings=block_vectors, block_id=f"block-{i}")

        blocks.append(block)
        logger.info(f"Created block {block.block_id} with {len(block.embeddings)} vectors")

    return blocks


def demonstrate_vector_operations(vectors: list[EmbeddingVector]):
    """Demonstrate vector operations."""
    logger.info("Demonstrating vector operations")

    # Select two vectors
    vector1 = vectors[0]
    vector2 = vectors[1]

    # Calculate cosine similarity
    similarity = vector1.cosine_similarity(vector2)
    logger.info(
        f"Cosine similarity between '{vector1.metadata['name']}' and '{vector2.metadata['name']}': {similarity:.4f}"
    )

    # Calculate Euclidean distance
    distance = vector1.euclidean_distance(vector2)
    logger.info(
        f"Euclidean distance between '{vector1.metadata['name']}' and '{vector2.metadata['name']}': {distance:.4f}"
    )

    # Normalize a vector
    normalized = vector1.normalize()
    logger.info(f"Normalized vector norm: {np.linalg.norm(normalized.vector):.4f}")


def demonstrate_search(wdbx: WDBXCore, query_vector: EmbeddingVector):
    """Demonstrate search functionality."""
    logger.info("Demonstrating vector search")

    # Search for similar vectors
    similar_vectors = wdbx.find_similar_vectors(query_vector, top_k=3)

    logger.info(
        f"Found {len(similar_vectors)} vectors similar to '{query_vector.metadata['name']}':"
    )
    for vector_id, similarity in similar_vectors:
        vector = wdbx.vectors.get(vector_id)
        if vector:
            logger.info(f"  - {vector.metadata['name']}: similarity {similarity:.4f}")

    # Search for relevant blocks
    similar_blocks = wdbx.search_blocks(query_vector, top_k=2)

    logger.info(
        f"Found {len(similar_blocks)} blocks relevant to '{query_vector.metadata['name']}':"
    )
    for block, similarity in similar_blocks:
        logger.info(f"  - {block.data['name']}: similarity {similarity:.4f}")


def demonstrate_persistence(wdbx: WDBXCore, vectors: list[EmbeddingVector], blocks: list[Block]):
    """Demonstrate persistence functionality."""
    logger.info("Demonstrating persistence")

    # Save vectors
    for vector in vectors:
        success = wdbx.save_vector(vector)
        logger.info(f"Saved vector {vector.vector_id}: {success}")

    # Save blocks
    for block in blocks:
        success = wdbx.save_block(block)
        logger.info(f"Saved block {block.block_id}: {success}")

    # Clear in-memory data
    wdbx.clear()
    logger.info("Cleared in-memory data")

    # Load a vector
    vector_id = vectors[0].vector_id
    loaded_vector = wdbx.load_vector(vector_id)
    logger.info(f"Loaded vector {vector_id}: {loaded_vector.metadata['name']}")

    # Load a block
    block_id = blocks[0].block_id
    loaded_block = wdbx.load_block(block_id)
    logger.info(f"Loaded block {block_id}: {loaded_block.data['name']}")


def demonstrate_memory_optimization(wdbx: WDBXCore):
    """Demonstrate memory optimization."""
    logger.info("Demonstrating memory optimization")

    # Get memory usage before optimization
    memory_before = wdbx.get_stats()["memory_peak_mb"]
    logger.info(f"Memory usage before optimization: {memory_before:.2f} MB")

    # Optimize memory
    wdbx.optimize_memory()

    # Get memory usage after optimization
    memory_after = wdbx.get_stats()["memory_peak_mb"]
    logger.info(f"Memory usage after optimization: {memory_after:.2f} MB")
    logger.info(f"Optimization count: {wdbx.stats.optimization_count}")


def demonstrate_plugin_integration():
    """Demonstrate Discord bot plugin integration."""
    logger.info("Demonstrating Discord bot plugin integration")

    try:
        # Connect to WDBX
        client = WDBXClient()
        client.connect(host="127.0.0.1", port=8080)
        logger.info("Connected to WDBX client")

        # Load the Discord bot plugin
        load_plugin_module("wdbx.plugins.discord_bot")
        discord_plugin = get_plugin("discord_bot")

        if discord_plugin:
            logger.info(
                f"Found Discord bot plugin: {discord_plugin.name} v{discord_plugin.version}"
            )
            discord_plugin()

            # Initialize with the client
            # Note: In a real application, you would need to set up the Discord bot config file first
            logger.info(
                "Discord bot plugin found. To use it, create a config file and initialize the plugin."
            )
            logger.info("See docs/plugins/discord_bot.rst for details.")
        else:
            logger.info("Discord bot plugin not found. Make sure it's installed correctly.")

    except Exception as e:
        logger.error(f"Error demonstrating plugin integration: {e}")


def run_example():
    """Run the complete example."""
    with LogContext(component="BasicUsage"):
        logger.info("Starting WDBX basic usage example")

        # Setup test environment
        temp_dir = setup_test_environment()

        try:
            # Initialize WDBX
            wdbx = get_wdbx_core()
            logger.info(f"Initialized WDBX Core with vector dimension {wdbx.vector_dimension}")

            # Create test data
            vectors = create_test_vectors(wdbx)
            blocks = create_test_blocks(wdbx, vectors)

            # Demonstrate vector operations
            demonstrate_vector_operations(vectors)

            # Demonstrate search
            demonstrate_search(wdbx, vectors[0])

            # Demonstrate persistence
            demonstrate_persistence(wdbx, vectors, blocks)

            # Demonstrate memory optimization
            demonstrate_memory_optimization(wdbx)

            # Demonstrate plugin integration (Discord bot)
            demonstrate_plugin_integration()

            # Print final statistics
            stats = wdbx.get_stats()
            logger.info(f"Final statistics: {stats}")

            # Shutdown WDBX
            wdbx.shutdown()
            logger.info("WDBX shutdown complete")

        except Exception as e:
            logger.error(f"Error in example: {e}", exc_info=True)

        finally:
            # Clean up temporary directory in a real application
            # For this example, we'll leave it for inspection
            logger.info(f"Example data saved in: {temp_dir}")
            logger.info("WDBX basic usage example completed")


if __name__ == "__main__":
    run_example()
