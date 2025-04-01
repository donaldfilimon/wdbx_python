# wdbx/cli.py
"""
Command-line interface for the WDBX system.

This module provides a user-friendly command-line interface for interacting
with the WDBX system, allowing users to store embedding vectors, create blocks,
search for similar vectors, and more.
"""
import argparse
import time
import sys
import json
import numpy as np
from typing import Dict, Any, Optional, List
import logging

from wdbx import WDBX
from wdbx.constants import logger, VECTOR_DIMENSION, SHARD_COUNT
from wdbx.data_structures import EmbeddingVector, Block
from wdbx.persona import PersonaManager


def run_example(vector_dim: int = 512, num_shards: int = 4) -> None:
    """
    Run an example demonstration of WDBX functionality.

    Args:
        vector_dim (int): Dimension of embedding vectors
        num_shards (int): Number of shards for storage
    """
    print("═" * 80)
    print("WDBX Example Demonstration")
    print("═" * 80)

    wdbx_instance = WDBX(vector_dimension=vector_dim, num_shards=num_shards)
    print("\n➤ Creating sample embeddings...")
    embeddings = []

    for i in range(10):
        vector = np.random.randn(wdbx_instance.vector_dimension).astype(np.float32)
        vector /= np.linalg.norm(vector)

        embedding = EmbeddingVector(
            vector=vector,
            metadata={"description": f"Sample embedding {i}", "timestamp": time.time()}
        )
        embeddings.append(embedding)
        vector_id = wdbx_instance.store_embedding(embedding)
        print(f"  ✓ Stored embedding with ID: {vector_id}")

    print("\n➤ Creating conversation chain...")

    import uuid
    chain_id = str(uuid.uuid4())
    user_inputs = [
        "How does the WDBX system work?",
        "Tell me more about the multi-persona framework",
        "What's the difference between Abbey and Aviva?",
        "Can you explain neural backtracking?"
    ]
    blocks = []

    persona_manager = PersonaManager(wdbx_instance)

    for i, user_input in enumerate(user_inputs, 1):
        print(f"\n➤ [{i}/{len(user_inputs)}] Processing user input: '{user_input}'")
        context = {"chain_id": chain_id, "block_ids": blocks}
        response, block_id = persona_manager.process_user_input(user_input, context)
        blocks.append(block_id)
        print(f"  User: {user_input}")
        print(f"  AI:   {response[:70]}...")
        print(f"  ✓ Created block with ID: {block_id}")

    print("\n➤ Searching for similar vectors...")
    query_vector = embeddings[0].vector
    results = wdbx_instance.search_similar_vectors(query_vector, top_k=3)

    for i, (vector_id, similarity) in enumerate(results, 1):
        print(f"  {i}. Vector ID: {vector_id}, Similarity: {similarity:.4f}")

    print("\n➤ Creating neural trace...")
    trace_id = wdbx_instance.create_neural_trace(query_vector)
    print(f"  ✓ Created trace with ID: {trace_id}")

    print("\n➤ System Statistics:")
    stats = wdbx_instance.get_system_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n═" * 80)
    print("Example complete!")
    print("═" * 80)


def interactive_mode(wdbx_instance: WDBX) -> None:
    """
    Run the WDBX system in interactive mode.

    Args:
        wdbx_instance (WDBX): WDBX instance to use
    """
    print("═" * 80)
    print("WDBX Interactive Mode")
    print("═" * 80)
    print("Type 'help' for available commands, 'exit' to quit.")

    persona_manager = PersonaManager(wdbx_instance)
    chain_id = None
    blocks = []

    while True:
        try:
            cmd = input("\n> ").strip()

            if not cmd:
                continue

            if cmd.lower() == "exit":
                break

            elif cmd.lower() == "help":
                print("\nAvailable commands:")
                print("  help                       - Show this help message")
                print("  exit                       - Exit interactive mode")
                print("  stats                      - Show system statistics")
                print("  store <description>        - Store a random embedding vector")
                print("  chat <message>             - Send a message to the AI")
                print("  search <vector_id>         - Search for vectors similar to a vector")
                print("  trace <vector_id>          - Create a neural trace for a vector")
                print("  show <block_id>            - Show details of a block")
                print("  context                    - Show current conversation context")
                print("  clear                      - Clear current conversation context")

            elif cmd.lower() == "stats":
                stats = wdbx_instance.get_system_stats()
                print("\nSystem Statistics:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")

            elif cmd.lower().startswith("store "):
                description = cmd[6:].strip()
                if not description:
                    print("Error: Please provide a description for the embedding")
                    continue

                vector = np.random.randn(wdbx_instance.vector_dimension).astype(np.float32)
                vector /= np.linalg.norm(vector)

                embedding = EmbeddingVector(
                    vector=vector,
                    metadata={"description": description, "timestamp": time.time()}
                )

                vector_id = wdbx_instance.store_embedding(embedding)
                print(f"Stored embedding with ID: {vector_id}")

            elif cmd.lower().startswith("chat "):
                message = cmd[5:].strip()
                if not message:
                    print("Error: Please provide a message")
                    continue

                print(f"User: {message}")
                context = {"chain_id": chain_id, "block_ids": blocks}
                response, block_id = persona_manager.process_user_input(message, context)

                if chain_id is None:
                    # Find the chain ID for the first block
                    block = wdbx_instance.block_chain_manager.get_block(block_id)
                    for cid, head in wdbx_instance.block_chain_manager.chain_heads.items():
                        if head == block_id:
                            chain_id = cid
                            break

                blocks.append(block_id)
                print(f"AI: {response}")
                print(f"Created block with ID: {block_id}")

            elif cmd.lower().startswith("search "):
                vector_id = cmd[7:].strip()
                if not vector_id:
                    print("Error: Please provide a vector ID")
                    continue

                embedding = wdbx_instance.vector_store.get(vector_id)
                if not embedding:
                    print(f"Error: Vector with ID {vector_id} not found")
                    continue

                results = wdbx_instance.search_similar_vectors(embedding.vector, top_k=5)
                print(f"Found {len(results)} similar vectors:")

                for i, (vid, similarity) in enumerate(results, 1):
                    print(f"  {i}. Vector ID: {vid}, Similarity: {similarity:.4f}")

            elif cmd.lower().startswith("trace "):
                vector_id = cmd[6:].strip()
                if not vector_id:
                    print("Error: Please provide a vector ID")
                    continue

                embedding = wdbx_instance.vector_store.get(vector_id)
                if not embedding:
                    print(f"Error: Vector with ID {vector_id} not found")
                    continue

                trace_id = wdbx_instance.create_neural_trace(embedding.vector)
                print(f"Created trace with ID: {trace_id}")
                print(f"Activations:")

                activations = wdbx_instance.neural_backtracker.activation_traces[trace_id]
                sorted_activations = sorted(activations.items(), key=lambda x: x[1], reverse=True)

                for i, (block_id, activation) in enumerate(sorted_activations[:5], 1):
                    print(f"  {i}. Block ID: {block_id}, Activation: {activation:.4f}")

            elif cmd.lower().startswith("show "):
                block_id = cmd[5:].strip()
                if not block_id:
                    print("Error: Please provide a block ID")
                    continue

                block = wdbx_instance.block_chain_manager.get_block(block_id)
                if not block:
                    print(f"Error: Block with ID {block_id} not found")
                    continue

                print(f"Block ID: {block.id}")
                print(f"Timestamp: {time.ctime(block.timestamp)}")
                print(f"Previous Hash: {block.previous_hash}")
                print(f"Hash: {block.hash}")
                print(f"Data: {json.dumps(block.data, indent=2)}")
                print(f"Embeddings: {len(block.embeddings)}")
                print(f"Context References: {block.context_references}")

            elif cmd.lower() == "context":
                if not blocks:
                    print("No conversation context available")
                    continue

                context = wdbx_instance.get_conversation_context(blocks)
                print(f"Conversation Blocks: {len(context['blocks'])}")
                print(f"Context Blocks: {len(context['context_blocks'])}")
                print(f"Chains: {context['chains']}")

                for i, block in enumerate(context['blocks'], 1):
                    print(f"\n{i}. Block ID: {block.id}")
                    if 'user_input' in block.data:
                        print(f"   User: {block.data['user_input']}")
                    if 'response' in block.data:
                        print(f"   AI: {block.data['response'][:50]}...")

            elif cmd.lower() == "clear":
                chain_id = None
                blocks = []
                print("Conversation context cleared")

            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands")

        except KeyboardInterrupt:
            print("\nOperation cancelled")

        except Exception as e:
            print(f"Error: {str(e)}")

    print("\nExiting interactive mode...")


def run_server(host: str, port: int, vector_dim: int, num_shards: int) -> None:
    """
    Run the WDBX HTTP server.

    Args:
        host (str): Host to listen on
        port (int): Port to listen on
        vector_dim (int): Dimension of embedding vectors
        num_shards (int): Number of shards for storage
    """
    try:
        from wdbx.server import run_server as start_server
        print(f"Starting WDBX server on {host}:{port}...")
        start_server(host, port, vector_dim, num_shards)
    except ImportError:
        print("Error: aiohttp is required to run the server")
        print("Install it with: pip install aiohttp")
        sys.exit(1)


def main() -> int:
    """
    Main entry point for the WDBX command-line interface.

    Returns:
        int: Exit code
    """
    parser = argparse.ArgumentParser(
        description="WDBX: Wide Distributed Block Exchange",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--vector-dim",
        type=int,
        default=VECTOR_DIMENSION,
        help="Vector dimension"
    )

    parser.add_argument(
        "--shards",
        type=int,
        default=SHARD_COUNT,
        help="Number of shards"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Example command
    example_parser = subparsers.add_parser("example", help="Run example usage")

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Run interactive mode")

    # Server command
    server_parser = subparsers.add_parser("server", help="Run WDBX server")
    server_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host"
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if args.command == "example":
        run_example(args.vector_dim, args.shards)
        return 0

    elif args.command == "interactive":
        wdbx_instance = WDBX(vector_dimension=args.vector_dim, num_shards=args.shards)
        interactive_mode(wdbx_instance)
        return 0

    elif args.command == "server":
        run_server(args.host, args.port, args.vector_dim, args.shards)
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
