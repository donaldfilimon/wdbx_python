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
from typing import Dict, Any, Optional, List, Tuple
import logging
import uuid
import os
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import textwrap

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
    console = Console()
    console.print("═" * 80, style="bold blue")
    console.print("WDBX Example Demonstration", style="bold green")
    console.print("═" * 80, style="bold blue")

    wdbx_instance = WDBX(vector_dimension=vector_dim, num_shards=num_shards)
    console.print("\n➤ Creating sample embeddings...", style="bold yellow")
    embeddings = []

    with Progress() as progress:
        task = progress.add_task("[cyan]Generating embeddings...", total=10)
        for i in range(10):
            vector = np.random.randn(wdbx_instance.vector_dimension).astype(np.float32)
            vector /= np.linalg.norm(vector)

            embedding = EmbeddingVector(
                vector=vector,
                metadata={"description": f"Sample embedding {i}", "timestamp": time.time()}
            )
            embeddings.append(embedding)
            vector_id = wdbx_instance.store_embedding(embedding)
            console.print(f"  ✓ Stored embedding with ID: {vector_id}", style="green")
            progress.update(task, advance=1)

    console.print("\n➤ Creating conversation chain...", style="bold yellow")

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
        console.print(f"\n➤ [{i}/{len(user_inputs)}] Processing user input: '{user_input}'", style="bold cyan")
        context = {"chain_id": chain_id, "block_ids": blocks}
        response, block_id = persona_manager.process_user_input(user_input, context)
        blocks.append(block_id)
        console.print(f"  User: {user_input}", style="bold")
        console.print(f"  AI:   {response[:70]}...", style="italic green")
        console.print(f"  ✓ Created block with ID: {block_id}", style="green")

    console.print("\n➤ Searching for similar vectors...", style="bold yellow")
    query_vector = embeddings[0].vector
    results = wdbx_instance.search_similar_vectors(query_vector, top_k=3)

    table = Table(title="Search Results")
    table.add_column("Rank", style="cyan")
    table.add_column("Vector ID", style="green")
    table.add_column("Similarity", style="magenta")

    for i, (vector_id, similarity) in enumerate(results, 1):
        table.add_row(str(i), vector_id, f"{similarity:.4f}")

    console.print(table)

    console.print("\n➤ Creating neural trace...", style="bold yellow")
    trace_id = wdbx_instance.create_neural_trace(query_vector)
    console.print(f"  ✓ Created trace with ID: {trace_id}", style="green")

    console.print("\n➤ System Statistics:", style="bold yellow")
    stats_table = Table(title="System Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")

    stats = wdbx_instance.get_system_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            stats_table.add_row(key, f"{value:.4f}")
        else:
            stats_table.add_row(key, str(value))

    console.print(stats_table)

    console.print("\n═" * 80, style="bold blue")
    console.print("Example complete!", style="bold green")
    console.print("═" * 80, style="bold blue")


def format_response(response: str, width: int = 70) -> str:
    """
    Format an AI response for display.

    Args:
        response (str): The response text
        width (int): Width to wrap text at

    Returns:
        str: Formatted response
    """
    if not response:
        return ""

    wrapped = textwrap.fill(response, width=width)
    return wrapped


def interactive_mode(wdbx_instance: WDBX) -> None:
    """
    Run the WDBX system in interactive mode.

    Args:
        wdbx_instance (WDBX): WDBX instance to use
    """
    console = Console()
    console.print("═" * 80, style="bold blue")
    console.print("WDBX Interactive Mode", style="bold green")
    console.print("═" * 80, style="bold blue")
    console.print("Type 'help' for available commands, 'exit' to quit.", style="italic")

    persona_manager = PersonaManager(wdbx_instance)
    chain_id = None
    blocks = []
    history = []
    current_persona = "default"

    # Get available personas
    available_personas = ["default", "abbey", "aviva"]  # Default list if not obtainable from persona_manager

    def show_help():
        help_table = Table(title="Available Commands")
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="green")

        commands = [
            ("help", "Show this help message"),
            ("exit", "Exit interactive mode"),
            ("stats", "Show system statistics"),
            ("store <description>", "Store a random embedding vector"),
            ("chat <message>", "Send a message to the AI"),
            ("search <vector_id>", "Search for vectors similar to a vector"),
            ("trace <vector_id>", "Create a neural trace for a vector"),
            ("show <block_id>", "Show details of a block"),
            ("context", "Show current conversation context"),
            ("clear", "Clear current conversation context"),
            ("persona <name>", "Switch to a different persona"),
            ("personas", "List available personas"),
            ("history", "Show conversation history"),
            ("save <filename>", "Save conversation to file"),
            ("load <filename>", "Load conversation from file")
        ]

        for cmd, desc in commands:
            help_table.add_row(cmd, desc)

        console.print(help_table)

    while True:
        try:
            cmd = console.input("\n[bold blue]>[/bold blue] ").strip()
            history.append(cmd)

            if not cmd:
                continue

            if cmd.lower() == "exit":
                break

            elif cmd.lower() == "help":
                show_help()

            elif cmd.lower() == "stats":
                stats = wdbx_instance.get_system_stats()
                stats_table = Table(title="System Statistics")
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="green")

                for key, value in stats.items():
                    if isinstance(value, float):
                        stats_table.add_row(key, f"{value:.4f}")
                    else:
                        stats_table.add_row(key, str(value))

                console.print(stats_table)

            elif cmd.lower().startswith("store "):
                description = cmd[6:].strip()
                if not description:
                    console.print("Error: Please provide a description for the embedding", style="bold red")
                    continue

                vector = np.random.randn(wdbx_instance.vector_dimension).astype(np.float32)
                vector /= np.linalg.norm(vector)

                embedding = EmbeddingVector(
                    vector=vector,
                    metadata={"description": description, "timestamp": time.time()}
                )

                vector_id = wdbx_instance.store_embedding(embedding)
                console.print(f"Stored embedding with ID: {vector_id}", style="green")

            elif cmd.lower().startswith("chat "):
                message = cmd[5:].strip()
                if not message:
                    console.print("Error: Please provide a message", style="bold red")
                    continue

                console.print(f"User: {message}", style="bold")

                with Progress() as progress:
                    task = progress.add_task("[cyan]Processing...", total=100)
                    context = {"chain_id": chain_id, "block_ids": blocks, "persona": current_persona}

                    # Update progress periodically to simulate thinking
                    for i in range(0, 100, 10):
                        time.sleep(0.1)
                        progress.update(task, completed=i)

                    response, block_id = persona_manager.process_user_input(message, context)
                    progress.update(task, completed=100)

                if chain_id is None:
                    # Find the chain ID for the first block
                    block = wdbx_instance.block_chain_manager.get_block(block_id)
                    for cid, head in wdbx_instance.block_chain_manager.chain_heads.items():
                        if head == block_id:
                            chain_id = cid
                            break

                    if chain_id is None:
                        chain_id = str(uuid.uuid4())

                blocks.append(block_id)
                console.print(f"AI ({current_persona}): {format_response(response)}", style="italic green")
                console.print(f"Created block with ID: {block_id}", style="dim")

            elif cmd.lower().startswith("search "):
                vector_id = cmd[7:].strip()
                if not vector_id:
                    console.print("Error: Please provide a vector ID", style="bold red")
                    continue

                embedding = wdbx_instance.vector_store.get(vector_id)
                if not embedding:
                    console.print(f"Error: Vector with ID {vector_id} not found", style="bold red")
                    continue

                results = wdbx_instance.search_similar_vectors(embedding.vector, top_k=5)

                table = Table(title=f"Similar Vectors to {vector_id}")
                table.add_column("Rank", style="cyan")
                table.add_column("Vector ID", style="green")
                table.add_column("Similarity", style="magenta")
                table.add_column("Description", style="yellow")

                for i, (vid, similarity) in enumerate(results, 1):
                    vec = wdbx_instance.vector_store.get(vid)
                    desc = vec.metadata.get("description", "N/A") if vec and vec.metadata else "N/A"
                    table.add_row(str(i), vid, f"{similarity:.4f}", desc)

                console.print(table)

            elif cmd.lower().startswith("trace "):
                vector_id = cmd[6:].strip()
                if not vector_id:
                    console.print("Error: Please provide a vector ID", style="bold red")
                    continue

                embedding = wdbx_instance.vector_store.get(vector_id)
                if not embedding:
                    console.print(f"Error: Vector with ID {vector_id} not found", style="bold red")
                    continue

                with Progress() as progress:
                    task = progress.add_task("[cyan]Creating neural trace...", total=100)
                    for i in range(0, 100, 20):
                        time.sleep(0.1)
                        progress.update(task, completed=i)

                    trace_id = wdbx_instance.create_neural_trace(embedding.vector)
                    progress.update(task, completed=100)

                console.print(f"Created trace with ID: {trace_id}", style="green")

                activations = wdbx_instance.neural_backtracker.activation_traces[trace_id]
                sorted_activations = sorted(activations.items(), key=lambda x: x[1], reverse=True)

                table = Table(title=f"Neural Trace Activations for {trace_id}")
                table.add_column("Rank", style="cyan")
                table.add_column("Block ID", style="green")
                table.add_column("Activation", style="magenta")

                for i, (block_id, activation) in enumerate(sorted_activations[:5], 1):
                    table.add_row(str(i), block_id, f"{activation:.4f}")

                console.print(table)

            elif cmd.lower().startswith("show "):
                block_id = cmd[5:].strip()
                if not block_id:
                    console.print("Error: Please provide a block ID", style="bold red")
                    continue

                block = wdbx_instance.block_chain_manager.get_block(block_id)
                if not block:
                    console.print(f"Error: Block with ID {block_id} not found", style="bold red")
                    continue

                table = Table(title=f"Block Details: {block_id}")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("ID", block.id)
                table.add_row("Timestamp", time.ctime(block.timestamp))
                table.add_row("Previous Hash", block.previous_hash or "None")
                table.add_row("Hash", block.hash)
                table.add_row("Embeddings Count", str(len(block.embeddings)))
                table.add_row("Context References", str(block.context_references))

                console.print(table)

                if block.data:
                    console.print("\nBlock Data:", style="bold yellow")
                    console.print(json.dumps(block.data, indent=2), style="green")

            elif cmd.lower() == "context":
                if not blocks:
                    console.print("No conversation context available", style="yellow")
                    continue

                context = wdbx_instance.get_conversation_context(blocks)
                console.print(f"Conversation Context Summary:", style="bold yellow")
                console.print(f"Chain ID: {chain_id}", style="cyan")
                console.print(f"Active Blocks: {len(blocks)}", style="cyan")
                console.print(f"Total Context Blocks: {len(context['context_blocks'])}", style="cyan")

                table = Table(title="Conversation History")
                table.add_column("Block", style="cyan")
                table.add_column("User", style="yellow")
                table.add_column("AI", style="green")

                for i, block in enumerate(context['blocks'], 1):
                    user_input = block.data.get('user_input', 'N/A')
                    response = block.data.get('response', 'N/A')
                    if len(response) > 50:
                        response = response[:50] + "..."

                    table.add_row(str(i), user_input, response)

                console.print(table)

            elif cmd.lower() == "clear":
                chain_id = None
                blocks = []
                console.print("Conversation context cleared", style="green")

            elif cmd.lower().startswith("persona "):
                persona_name = cmd[8:].strip()
                if not persona_name:
                    console.print("Error: Please provide a persona name", style="bold red")
                    continue

                if persona_name in available_personas:
                    current_persona = persona_name
                    console.print(f"Switched to persona: {current_persona}", style="green")
                else:
                    console.print(f"Error: Persona '{persona_name}' not found", style="bold red")
                    console.print(f"Available personas: {', '.join(available_personas)}", style="yellow")

            elif cmd.lower() == "personas":
                table = Table(title="Available Personas")
                table.add_column("Persona", style="cyan")
                table.add_column("Description", style="green")

                for persona in available_personas:
                    # Get persona description or use default
                    description = "Default conversational AI" if persona == "default" else \
                                 "Abbey - Academic research assistant" if persona == "abbey" else \
                                 "Aviva - Creative storytelling assistant" if persona == "aviva" else \
                                 "No description available"
                    table.add_row(persona, description)

                console.print(table)
                console.print(f"Current persona: {current_persona}", style="bold cyan")

            elif cmd.lower() == "history":
                if not blocks:
                    console.print("No conversation history available", style="yellow")
                    continue

                table = Table(title="Conversation History")
                table.add_column("Turn", style="cyan")
                table.add_column("Speaker", style="green")
                table.add_column("Message", style="yellow")

                block_data = []
                for block_id in blocks:
                    block = wdbx_instance.block_chain_manager.get_block(block_id)
                    if block and block.data:
                        block_data.append(block)

                for i, block in enumerate(block_data, 1):
                    if 'user_input' in block.data:
                        table.add_row(f"{i}.1", "User", block.data['user_input'])

                    if 'response' in block.data:
                        response = block.data['response']
                        if len(response) > 70:
                            response = response[:70] + "..."
                        table.add_row(f"{i}.2", "AI", response)

                console.print(table)

            elif cmd.lower().startswith("save "):
                filename = cmd[5:].strip()
                if not filename:
                    console.print("Error: Please provide a filename", style="bold red")
                    continue

                if not blocks:
                    console.print("Error: No conversation to save", style="bold red")
                    continue

                try:
                    conversation_data = {
                        "chain_id": chain_id,
                        "blocks": blocks,
                        "current_persona": current_persona,
                        "timestamp": time.time()
                    }

                    with open(filename, 'w') as f:
                        json.dump(conversation_data, f, indent=2)

                    console.print(f"Conversation saved to {filename}", style="green")
                except Exception as e:
                    console.print(f"Error saving conversation: {str(e)}", style="bold red")

            elif cmd.lower().startswith("load "):
                filename = cmd[5:].strip()
                if not filename:
                    console.print("Error: Please provide a filename", style="bold red")
                    continue

                if not os.path.exists(filename):
                    console.print(f"Error: File {filename} not found", style="bold red")
                    continue

                try:
                    with open(filename, 'r') as f:
                        conversation_data = json.load(f)

                    chain_id = conversation_data.get("chain_id")
                    blocks = conversation_data.get("blocks", [])
                    current_persona = conversation_data.get("current_persona", "default")

                    console.print(f"Loaded conversation from {filename}", style="green")
                    console.print(f"Chain ID: {chain_id}", style="cyan")
                    console.print(f"Blocks: {len(blocks)}", style="cyan")
                    console.print(f"Current persona: {current_persona}", style="cyan")
                except Exception as e:
                    console.print(f"Error loading conversation: {str(e)}", style="bold red")

            else:
                console.print(f"Unknown command: {cmd}", style="bold red")
                console.print("Type 'help' for available commands", style="yellow")

        except KeyboardInterrupt:
            console.print("\nOperation cancelled", style="yellow")
            continue

        except Exception as e:
            console.print(f"Error: {str(e)}", style="bold red")
            logger.error(f"Exception in interactive mode: {str(e)}", exc_info=True)

    console.print("\nExiting interactive mode...", style="bold blue")


def run_server(host: str, port: int, vector_dim: int, num_shards: int) -> None:
    """
    Run the WDBX HTTP server.

    Args:
        host (str): Host to listen on
        port (int): Port to listen on
        vector_dim (int): Dimension of embedding vectors
        num_shards (int): Number of shards for storage
    """
    console = Console()
    try:
        from wdbx.server import run_server as start_server
        console.print(f"Starting WDBX server on {host}:{port}...", style="bold green")
        console.print(f"Vector dimension: {vector_dim}", style="cyan")
        console.print(f"Number of shards: {num_shards}", style="cyan")
        console.print("Press Ctrl+C to stop the server", style="italic")

        start_server(host, port, vector_dim, num_shards)
    except ImportError:
        console.print("Error: aiohttp is required to run the server", style="bold red")
        console.print("Install it with: pip install aiohttp", style="yellow")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\nServer stopped", style="yellow")


def batch_process(input_file: str, output_file: str, vector_dim: int, num_shards: int) -> None:
    """
    Batch process a file of inputs.

    Args:
        input_file (str): Input file with one query per line
        output_file (str): Output file for results
        vector_dim (int): Vector dimension
        num_shards (int): Number of shards
    """
    console = Console()
    console.print(f"Starting batch processing of {input_file}...", style="bold green")

    if not os.path.exists(input_file):
        console.print(f"Error: Input file {input_file} not found", style="bold red")
        return

    wdbx_instance = WDBX(vector_dimension=vector_dim, num_shards=num_shards)
    persona_manager = PersonaManager(wdbx_instance)
    chain_id = str(uuid.uuid4())
    blocks = []
    results = []

    try:
        with open(input_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        with Progress() as progress:
            task = progress.add_task("[cyan]Processing inputs...", total=len(lines))

            for i, line in enumerate(lines, 1):
                context = {"chain_id": chain_id, "block_ids": blocks}
                response, block_id = persona_manager.process_user_input(line, context)
                blocks.append(block_id)

                results.append({
                    "input": line,
                    "response": response,
                    "block_id": block_id
                })

                progress.update(task, advance=1)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        console.print(f"Processed {len(results)} inputs", style="green")
        console.print(f"Results saved to {output_file}", style="green")

        # Generate summary
        console.print("\nProcessing Summary:", style="bold yellow")
        stats = wdbx_instance.get_system_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                console.print(f"  {key}: {value:.4f}", style="cyan")
            else:
                console.print(f"  {key}: {value}", style="cyan")

    except Exception as e:
        console.print(f"Error during batch processing: {str(e)}", style="bold red")
        logger.error(f"Exception in batch processing: {str(e)}", exc_info=True)


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

    parser.add_argument(
        "--log-file",
        type=str,
        help="Log to file instead of console"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Example command
    subparsers.add_parser("example", help="Run example usage")

    # Interactive command
    subparsers.add_parser("interactive", help="Run interactive mode")

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

    # Batch processing command
    batch_parser = subparsers.add_parser("batch", help="Batch process inputs")
    batch_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file with one query per line"
    )
    batch_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file for results"
    )

    args = parser.parse_args()

    # Set up logging
    log_config = {
        "level": getattr(logging, args.log_level),
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }

    if args.log_file:
        log_config["filename"] = args.log_file
        log_config["filemode"] = "a"

    logging.basicConfig(**log_config)

    try:
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

        elif args.command == "batch":
            batch_process(args.input, args.output, args.vector_dim, args.shards)
            return 0

        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
